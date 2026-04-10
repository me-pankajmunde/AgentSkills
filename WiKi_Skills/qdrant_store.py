#!/usr/bin/env python3
"""
Qdrant Vector Store for Wiki Agent Hybrid Retrieval.

Wraps qdrant-client + OpenAI embeddings to provide semantic search
over wiki chunks. Used alongside BM25 for hybrid retrieval via RRF fusion.

Requires optional dependencies:
    pip install "farmerp-wiki[hybrid]"
    # or: pip install qdrant-client openai

If dependencies are missing, all functions raise ImportError with a clear message.
"""

import os
import sys
import json
import time
import uuid
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

# Lazy imports — these are optional dependencies
_qdrant_client = None
_openai_client = None


def _ensure_qdrant():
    """Import qdrant_client or raise a helpful error."""
    global _qdrant_client
    if _qdrant_client is not None:
        return _qdrant_client
    try:
        import qdrant_client
        _qdrant_client = qdrant_client
        return _qdrant_client
    except ImportError:
        raise ImportError(
            "qdrant-client is required for hybrid search. "
            "Install with: pip install 'farmerp-wiki[hybrid]' "
            "or: pip install qdrant-client"
        )


def _ensure_openai():
    """Import openai or raise a helpful error."""
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        import openai
        _openai_client = openai
        return _openai_client
    except ImportError:
        raise ImportError(
            "openai is required for hybrid search embeddings. "
            "Install with: pip install 'farmerp-wiki[hybrid]' "
            "or: pip install openai"
        )


def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Convert a chunk_id hex string to a deterministic UUID for Qdrant point IDs."""
    h = hashlib.md5(chunk_id.encode()).hexdigest()
    return str(uuid.UUID(h))


def _load_config(wiki_dir: Optional[Path] = None) -> Dict[str, Any]:
    """
    Load hybrid search configuration from environment variables
    and optionally .wiki/_config.json.

    Env vars take precedence over config file values.
    """
    config = {
        'qdrant_url': 'http://localhost:6333',
        'qdrant_api_key': None,
        'collection_name': 'wiki',
        'openai_api_key': None,
        'embedding_model': 'text-embedding-3-small',
        'embedding_dimensions': 1536,
        'rerank_model': 'gpt-4o-mini',
        'rerank_top_n': 20,
        'rrf_k': 60,
    }

    # Load from .wiki/_config.json if it exists
    if wiki_dir:
        config_path = wiki_dir / '_config.json'
        if config_path.exists():
            try:
                file_config = json.loads(config_path.read_text(encoding='utf-8'))
                for key in config:
                    if key in file_config:
                        config[key] = file_config[key]
            except (json.JSONDecodeError, OSError):
                pass

        # Use wiki name from _discovery.json as default collection name
        discovery_path = wiki_dir / '_discovery.json'
        if discovery_path.exists() and config['collection_name'] == 'wiki':
            try:
                discovery = json.loads(discovery_path.read_text(encoding='utf-8'))
                name = discovery.get('name', '').lower().replace(' ', '-')
                if not name:
                    # Backwards compat with old project-centric format
                    name = discovery.get('project_name', '').lower().replace(' ', '-')
                if name:
                    config['collection_name'] = f"wiki-{name}"
            except (json.JSONDecodeError, OSError):
                pass

    # Environment variables override everything
    env_map = {
        'QDRANT_URL': 'qdrant_url',
        'QDRANT_API_KEY': 'qdrant_api_key',
        'QDRANT_COLLECTION_NAME': 'collection_name',
        'OPENAI_API_KEY': 'openai_api_key',
        'OPENAI_EMBEDDING_MODEL': 'embedding_model',
        'RERANK_MODEL': 'rerank_model',
        'RERANK_TOP_N': 'rerank_top_n',
        'RRF_K': 'rrf_k',
    }
    for env_var, config_key in env_map.items():
        val = os.environ.get(env_var)
        if val is not None:
            # Cast numeric values
            if config_key in ('rerank_top_n', 'rrf_k', 'embedding_dimensions'):
                val = int(val)
            config[config_key] = val

    return config


class QdrantStore:
    """
    Qdrant vector store for wiki chunks.

    Handles collection management, embedding via OpenAI, and semantic search.
    All results are returned in the same format as BM25Index.search() for
    seamless fusion.
    """

    BATCH_SIZE = 100  # Chunks per upsert batch
    EMBED_BATCH_SIZE = 50  # Texts per embedding API call
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0  # seconds

    def __init__(self, wiki_dir: Optional[Path] = None, config: Optional[Dict] = None):
        qdrant_mod = _ensure_qdrant()
        openai_mod = _ensure_openai()

        self.config = config or _load_config(wiki_dir)

        if not self.config.get('openai_api_key'):
            raise ValueError(
                "OPENAI_API_KEY is required for hybrid search. "
                "Set it as an environment variable or in .wiki/_config.json"
            )

        # Initialize Qdrant client
        self.qdrant = qdrant_mod.QdrantClient(
            url=self.config['qdrant_url'],
            api_key=self.config.get('qdrant_api_key'),
        )

        # Initialize OpenAI client
        self.openai = openai_mod.OpenAI(api_key=self.config['openai_api_key'])

        self.collection_name = self.config['collection_name']
        self.embedding_model = self.config['embedding_model']
        self.embedding_dimensions = self.config['embedding_dimensions']

    def ensure_collection(self):
        """Create the Qdrant collection if it doesn't exist."""
        qdrant_mod = _ensure_qdrant()
        from qdrant_client.models import Distance, VectorParams

        collections = [c.name for c in self.qdrant.get_collections().collections]
        if self.collection_name not in collections:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dimensions,
                    distance=Distance.COSINE,
                ),
            )
            print(f"  Created Qdrant collection: {self.collection_name} "
                  f"({self.embedding_dimensions} dims, cosine)")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts using OpenAI embeddings API.
        Handles batching and retries.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.EMBED_BATCH_SIZE):
            batch = texts[i:i + self.EMBED_BATCH_SIZE]

            for attempt in range(self.MAX_RETRIES):
                try:
                    response = self.openai.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    break
                except Exception as e:
                    if attempt < self.MAX_RETRIES - 1:
                        wait = self.RETRY_DELAY * (2 ** attempt)
                        print(f"  Embedding retry {attempt + 1}/{self.MAX_RETRIES} "
                              f"after {wait}s: {e}", file=sys.stderr)
                        time.sleep(wait)
                    else:
                        raise RuntimeError(
                            f"Failed to embed batch after {self.MAX_RETRIES} attempts: {e}"
                        ) from e

        return all_embeddings

    def upsert_chunks(self, chunks: List[Dict]):
        """
        Embed and upsert chunks to Qdrant. Uses chunk_id for deterministic point IDs.
        Stores all metadata as payload for filtering and result formatting.
        """
        from qdrant_client.models import PointStruct

        self.ensure_collection()

        # Prepare texts for embedding (use content with context prefix)
        texts = [c['content'] for c in chunks]

        print(f"  Embedding {len(texts)} chunks via OpenAI ({self.embedding_model})...")
        embeddings = self.embed(texts)

        # Build points
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point_id = _chunk_id_to_uuid(chunk['chunk_id'])
            payload = {
                'chunk_id': chunk['chunk_id'],
                'filepath': chunk['filepath'],
                'title': chunk['title'],
                'section': chunk['section'],
                'page_type': chunk.get('page_type', 'unknown'),
                'tags': chunk.get('tags', []),
                'content': chunk['content'],
                'content_raw': chunk.get('content_raw', ''),
                'position': chunk['position'],
            }
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        # Batch upsert
        for i in range(0, len(points), self.BATCH_SIZE):
            batch = points[i:i + self.BATCH_SIZE]
            self.qdrant.upsert(
                collection_name=self.collection_name,
                points=batch,
            )

        print(f"  Upserted {len(points)} points to Qdrant collection '{self.collection_name}'")

    def search(self, query: str, top_k: int = 10,
               page_type_filter: Optional[str] = None,
               tag_filter: Optional[str] = None) -> List[Dict]:
        """
        Semantic search via Qdrant. Returns results in BM25-compatible format.

        Scores are cosine similarity (0-1 range).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        # Embed query
        query_embedding = self.embed([query])[0]

        # Build filters
        must_conditions = []
        if page_type_filter:
            must_conditions.append(
                FieldCondition(key="page_type", match=MatchValue(value=page_type_filter))
            )
        if tag_filter:
            must_conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag_filter))
            )

        query_filter = Filter(must=must_conditions) if must_conditions else None

        # Search
        hits = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        ).points

        # Format results to match BM25 output
        results = []
        for hit in hits:
            payload = hit.payload
            results.append({
                'chunk_id': payload.get('chunk_id', ''),
                'score': round(hit.score, 4),
                'filepath': payload.get('filepath', ''),
                'title': payload.get('title', ''),
                'section': payload.get('section', ''),
                'page_type': payload.get('page_type', 'unknown'),
                'tags': payload.get('tags', []),
                'content': payload.get('content', ''),
                'position': payload.get('position', 0),
            })

        return results

    def delete_by_filepath(self, filepath: str):
        """Remove all chunks belonging to a specific wiki page."""
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        self.qdrant.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[FieldCondition(key="filepath", match=MatchValue(value=filepath))]
            ),
        )

    def drop_collection(self):
        """Delete the entire collection (full reset)."""
        self.qdrant.delete_collection(collection_name=self.collection_name)
        print(f"  Dropped Qdrant collection: {self.collection_name}")

    def collection_info(self) -> Optional[Dict]:
        """Get collection stats, or None if collection doesn't exist."""
        try:
            info = self.qdrant.get_collection(self.collection_name)
            return {
                'name': self.collection_name,
                'points_count': info.points_count,
                'vectors_count': info.vectors_count,
                'status': str(info.status),
            }
        except Exception:
            return None


def is_hybrid_available() -> bool:
    """Check if hybrid search dependencies are installed."""
    try:
        _ensure_qdrant()
        _ensure_openai()
        return True
    except ImportError:
        return False


def is_hybrid_configured(wiki_dir: Optional[Path] = None) -> bool:
    """Check if hybrid search is both available and configured (API key present)."""
    if not is_hybrid_available():
        return False
    config = _load_config(wiki_dir)
    return bool(config.get('openai_api_key'))

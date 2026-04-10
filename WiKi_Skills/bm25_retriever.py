#!/usr/bin/env python3
"""
BM25 Retrieval Engine for Wiki Agent RAG Pipeline.

A pure-Python implementation of Okapi BM25 (no external dependencies).
Indexes all wiki pages, chunks them into retrieval units, and returns
ranked results with full context for LLM consumption.

Usage:
    # Build/rebuild the index from the wiki directory
    python bm25_retriever.py index [--wiki-dir PATH]

    # Search the index
    python bm25_retriever.py search "your query here" [--top-k 10] [--wiki-dir PATH]

    # Search and return full chunk content (for RAG pipeline)
    python bm25_retriever.py retrieve "your query here" [--top-k 5] [--wiki-dir PATH]

    # Show index statistics
    python bm25_retriever.py stats [--wiki-dir PATH]
"""

import os
import sys
import re
import json
import math
import hashlib
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any


# ============================================================================
# HYBRID SEARCH — Optional Qdrant + RRF + LLM reranking backend
# ============================================================================

def _hybrid_available() -> bool:
    """Check if hybrid search dependencies (qdrant-client, openai) are installed."""
    try:
        from qdrant_store import is_hybrid_available
        return is_hybrid_available()
    except ImportError:
        return False


def _hybrid_configured(wiki_dir: Optional[Path] = None) -> bool:
    """Check if hybrid search is both available and configured."""
    try:
        from qdrant_store import is_hybrid_configured
        return is_hybrid_configured(wiki_dir)
    except ImportError:
        return False


def _resolve_backend(requested: str, wiki_dir: Optional[Path] = None) -> str:
    """Resolve 'auto' backend to the best available option."""
    if requested == 'hybrid':
        if not _hybrid_configured(wiki_dir):
            print("  ⚠️  Hybrid search not configured, falling back to bm25", file=sys.stderr)
            print("     Install: pip install 'wiki-agent[hybrid]'", file=sys.stderr)
            print("     Configure: OPENAI_API_KEY + Qdrant at localhost:6333", file=sys.stderr)
            return 'bm25'
        return 'hybrid'
    if requested == 'qdrant':
        if not _hybrid_configured(wiki_dir):
            print("  ⚠️  Qdrant not configured, falling back to bm25", file=sys.stderr)
            return 'bm25'
        return 'qdrant'
    if requested == 'auto':
        return 'hybrid' if _hybrid_configured(wiki_dir) else 'bm25'
    return 'bm25'


# ============================================================================
# TEXT PROCESSING
# ============================================================================

# Common English stop words — excluded from indexing to improve precision
STOP_WORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'it', 'as', 'be', 'was', 'are',
    'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
    'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his',
    'its', 'our', 'their', 'what', 'which', 'who', 'whom', 'where',
    'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'if', 'then', 'also',
    'about', 'up', 'out', 'into', 'over', 'after', 'before', 'between',
    'under', 'again', 'there', 'here', 'once', 'during', 'while',
})


def tokenize(text: str) -> List[str]:
    """
    Tokenize text into lowercase words, strip punctuation, remove stop words.
    Keeps alphanumeric tokens, hyphens within words, and underscores (for code terms).
    """
    # Lowercase
    text = text.lower()
    # Replace non-alphanumeric (except hyphens and underscores) with spaces
    text = re.sub(r'[^a-z0-9_\-]', ' ', text)
    # Split and filter
    tokens = []
    for tok in text.split():
        tok = tok.strip('-_')
        if tok and tok not in STOP_WORDS and len(tok) > 1:
            tokens.append(tok)
    return tokens


def simple_stem(word: str) -> str:
    """
    Minimal suffix-stripping stemmer. Not as good as Porter/Snowball but
    zero-dependency and handles the most common English suffixes.
    """
    if len(word) <= 4:
        return word
    # Order matters — try longest suffixes first
    suffixes = [
        'ation', 'tion', 'sion', 'ment', 'ness', 'ible', 'able',
        'ful', 'less', 'ous', 'ive', 'ing', 'ied', 'ies',
        'ers', 'est', 'ity', 'ism', 'ist', 'ant', 'ent',
        'ly', 'ed', 'er', 'al', 'es', 'en',  's',
    ]
    for suffix in suffixes:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[:-len(suffix)]
    return word


def tokenize_and_stem(text: str) -> List[str]:
    """Tokenize then apply stemming."""
    return [simple_stem(tok) for tok in tokenize(text)]


# ============================================================================
# FRONTMATTER PARSING
# ============================================================================

def parse_frontmatter(content: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract YAML frontmatter from markdown content.
    Returns (metadata_dict, body_text).
    Uses simple regex parsing — no PyYAML dependency needed.
    """
    metadata = {}
    body = content

    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()

            for line in fm_text.split('\n'):
                line = line.strip()
                if ':' in line:
                    key, _, val = line.partition(':')
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Handle list values: [a, b, c]
                    if val.startswith('[') and val.endswith(']'):
                        val = [v.strip().strip('"').strip("'")
                               for v in val[1:-1].split(',') if v.strip()]
                    metadata[key] = val

    return metadata, body


# ============================================================================
# CHUNKING
# ============================================================================

def chunk_markdown(content: str, filepath: str,
                   max_chunk_size: int = 800,
                   overlap: int = 100) -> List[Dict]:
    """
    Split a markdown file into retrieval-sized chunks.
    
    Strategy:
    1. Split on ## headers first (section-level chunks)
    2. If a section is still too long, split on paragraphs
    3. If a paragraph is still too long, split on sentences
    4. Add overlap between consecutive chunks for context continuity
    
    Returns list of chunk dicts:
    {
        "chunk_id": "hash",
        "filepath": "relative/path.md",
        "section": "Section Header",
        "content": "chunk text...",
        "tokens": ["stemmed", "tokens"],
        "position": 0,  # chunk index within the file
        "metadata": {...}  # from frontmatter
    }
    """
    metadata, body = parse_frontmatter(content)
    title = metadata.get('title', Path(filepath).stem)
    page_type = metadata.get('type', 'unknown')
    tags = metadata.get('tags', [])
    if isinstance(tags, str):
        tags = [tags]

    chunks = []
    
    # Split by ## headers
    sections = re.split(r'^(##\s+.+)$', body, flags=re.MULTILINE)
    
    # Group into (header, content) pairs
    section_pairs = []
    current_header = title  # Default header is the page title
    current_content = ''
    
    for part in sections:
        if re.match(r'^##\s+', part):
            if current_content.strip():
                section_pairs.append((current_header, current_content.strip()))
            current_header = part.strip('#').strip()
            current_content = ''
        else:
            current_content += part
    
    if current_content.strip():
        section_pairs.append((current_header, current_content.strip()))
    
    # If no sections found, treat whole body as one section
    if not section_pairs and body.strip():
        section_pairs = [(title, body.strip())]

    position = 0
    for section_header, section_text in section_pairs:
        # Split section into paragraphs
        paragraphs = re.split(r'\n\s*\n', section_text)
        
        current_chunk_parts = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            para_words = len(para.split())
            
            if current_size + para_words > max_chunk_size and current_chunk_parts:
                # Flush current chunk
                chunk_text = '\n\n'.join(current_chunk_parts)
                chunk_id = hashlib.md5(
                    f"{filepath}:{position}:{chunk_text[:100]}".encode()
                ).hexdigest()[:12]
                
                # Prepend context: title + section header
                context_prefix = f"[{title}] [{section_header}]\n\n"
                
                chunks.append({
                    'chunk_id': chunk_id,
                    'filepath': filepath,
                    'title': title,
                    'section': section_header,
                    'page_type': page_type,
                    'tags': tags,
                    'content': context_prefix + chunk_text,
                    'content_raw': chunk_text,
                    'tokens': tokenize_and_stem(chunk_text),
                    'position': position,
                    'metadata': metadata,
                })
                position += 1
                
                # Keep overlap
                if overlap > 0 and current_chunk_parts:
                    overlap_text = current_chunk_parts[-1]
                    current_chunk_parts = [overlap_text]
                    current_size = len(overlap_text.split())
                else:
                    current_chunk_parts = []
                    current_size = 0
            
            # If single paragraph exceeds max, split on sentences
            if para_words > max_chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                for sent in sentences:
                    sent_words = len(sent.split())
                    if current_size + sent_words > max_chunk_size and current_chunk_parts:
                        chunk_text = '\n\n'.join(current_chunk_parts)
                        chunk_id = hashlib.md5(
                            f"{filepath}:{position}:{chunk_text[:100]}".encode()
                        ).hexdigest()[:12]
                        context_prefix = f"[{title}] [{section_header}]\n\n"
                        chunks.append({
                            'chunk_id': chunk_id,
                            'filepath': filepath,
                            'title': title,
                            'section': section_header,
                            'page_type': page_type,
                            'tags': tags,
                            'content': context_prefix + chunk_text,
                            'content_raw': chunk_text,
                            'tokens': tokenize_and_stem(chunk_text),
                            'position': position,
                            'metadata': metadata,
                        })
                        position += 1
                        current_chunk_parts = []
                        current_size = 0
                    current_chunk_parts.append(sent)
                    current_size += sent_words
            else:
                current_chunk_parts.append(para)
                current_size += para_words
        
        # Flush remaining
        if current_chunk_parts:
            chunk_text = '\n\n'.join(current_chunk_parts)
            chunk_id = hashlib.md5(
                f"{filepath}:{position}:{chunk_text[:100]}".encode()
            ).hexdigest()[:12]
            context_prefix = f"[{title}] [{section_header}]\n\n"
            chunks.append({
                'chunk_id': chunk_id,
                'filepath': filepath,
                'title': title,
                'section': section_header,
                'page_type': page_type,
                'tags': tags,
                'content': context_prefix + chunk_text,
                'content_raw': chunk_text,
                'tokens': tokenize_and_stem(chunk_text),
                'position': position,
                'metadata': metadata,
            })
            position += 1
    
    return chunks


# ============================================================================
# BM25 INDEX
# ============================================================================

class BM25Index:
    """
    Okapi BM25 index for wiki retrieval.
    
    BM25 scoring formula:
        score(q, d) = Σ IDF(qi) * (tf(qi, d) * (k1 + 1)) / (tf(qi, d) + k1 * (1 - b + b * |d|/avgdl))
    
    Where:
        - tf(qi, d) = term frequency of qi in document d
        - |d| = document length (in tokens)
        - avgdl = average document length across corpus
        - k1 = term saturation parameter (default 1.5)
        - b = length normalization parameter (default 0.75)
        - IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
        - N = total number of documents
        - n(qi) = number of documents containing qi
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        
        # Core data structures
        self.chunks: List[Dict] = []              # All chunks
        self.doc_lengths: List[int] = []           # Token count per chunk
        self.avgdl: float = 0.0                    # Average document length
        self.N: int = 0                            # Total number of chunks
        
        # Inverted index: term -> list of (chunk_index, term_frequency)
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        
        # Document frequency: term -> number of chunks containing it
        self.df: Dict[str, int] = defaultdict(int)
        
        # Metadata
        self.built_at: Optional[str] = None
        self.wiki_dir: Optional[str] = None
    
    def build(self, chunks: List[Dict]):
        """Build the index from a list of chunks."""
        self.chunks = chunks
        self.N = len(chunks)
        self.doc_lengths = []
        self.inverted_index = defaultdict(list)
        self.df = defaultdict(int)
        
        for idx, chunk in enumerate(chunks):
            tokens = chunk['tokens']
            self.doc_lengths.append(len(tokens))
            
            # Count term frequencies in this chunk
            tf = Counter(tokens)
            
            # Update inverted index and document frequencies
            for term, freq in tf.items():
                self.inverted_index[term].append((idx, freq))
                self.df[term] += 1
        
        self.avgdl = sum(self.doc_lengths) / self.N if self.N > 0 else 0
        self.built_at = datetime.now().isoformat()
    
    def _idf(self, term: str) -> float:
        """Compute inverse document frequency for a term."""
        n = self.df.get(term, 0)
        return math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
    
    def search(self, query: str, top_k: int = 10,
               page_type_filter: Optional[str] = None,
               tag_filter: Optional[str] = None,
               freshness_weight: float = 0.0,
               centrality_data: Optional[Dict[str, int]] = None,
               centrality_weight: float = 0.0) -> List[Dict]:
        """
        Search the index with a query string.
        
        Optional boosting signals:
            freshness_weight: 0.0-1.0, boosts recently-updated pages
            centrality_data: dict of filepath -> in-degree from graph analysis
            centrality_weight: 0.0-1.0, boosts well-connected hub pages
        
        Returns list of results sorted by BM25 score:
        {
            "chunk_id": "...",
            "score": 12.34,
            "filepath": "entities/my-entity.md",
            "title": "My Entity",
            "section": "Overview",
            "page_type": "entity",
            "tags": [...],
            "content": "full chunk text with context prefix",
            "position": 0
        }
        """
        if self.N == 0:
            return []
        
        query_tokens = tokenize_and_stem(query)
        if not query_tokens:
            return []
        
        # Accumulate scores
        scores = defaultdict(float)
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._idf(term)
            
            for chunk_idx, tf in self.inverted_index[term]:
                dl = self.doc_lengths[chunk_idx]
                # BM25 formula
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[chunk_idx] += idf * (numerator / denominator)
        
        # Sort by score descending
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        
        # Apply optional boost signals before collecting results
        if freshness_weight > 0 or (centrality_weight > 0 and centrality_data):
            now = datetime.now()
            boosted = []
            for chunk_idx, score in ranked:
                chunk = self.chunks[chunk_idx]
                boost = 1.0
                # Freshness boost: pages updated recently get higher score
                if freshness_weight > 0:
                    meta = chunk.get('metadata', {})
                    updated_str = meta.get('updated', '')
                    if updated_str and isinstance(updated_str, str):
                        try:
                            updated_dt = datetime.strptime(updated_str.strip(), '%Y-%m-%d')
                            age_days = max((now - updated_dt).days, 1)
                            freshness = 1.0 / (1.0 + age_days / 30.0)
                            boost += freshness_weight * freshness
                        except ValueError:
                            pass
                # Hub/centrality boost: well-connected pages rank higher
                if centrality_weight > 0 and centrality_data:
                    fp = chunk.get('filepath', '')
                    in_deg = centrality_data.get(fp, 0)
                    if in_deg > 0:
                        boost += centrality_weight * math.log1p(in_deg)
                boosted.append((chunk_idx, score * boost))
            ranked = sorted(boosted, key=lambda x: -x[1])
        
        # Apply filters and collect results
        results = []
        for chunk_idx, score in ranked:
            chunk = self.chunks[chunk_idx]
            
            # Apply type filter
            if page_type_filter and chunk.get('page_type') != page_type_filter:
                continue
            
            # Apply tag filter
            if tag_filter:
                chunk_tags = chunk.get('tags', [])
                if tag_filter not in chunk_tags:
                    continue
            
            results.append({
                'chunk_id': chunk['chunk_id'],
                'score': round(score, 4),
                'filepath': chunk['filepath'],
                'title': chunk['title'],
                'section': chunk['section'],
                'page_type': chunk.get('page_type', 'unknown'),
                'tags': chunk.get('tags', []),
                'content': chunk['content'],
                'position': chunk['position'],
            })
            
            if len(results) >= top_k:
                break
        
        return results
    
    def save(self, path: str):
        """Serialize index to JSON file."""
        data = {
            'version': 2,
            'built_at': self.built_at,
            'wiki_dir': self.wiki_dir,
            'k1': self.k1,
            'b': self.b,
            'N': self.N,
            'avgdl': self.avgdl,
            'doc_lengths': self.doc_lengths,
            'df': dict(self.df),
            'inverted_index': {
                term: postings
                for term, postings in self.inverted_index.items()
            },
            'chunks': [
                {
                    'chunk_id': c['chunk_id'],
                    'filepath': c['filepath'],
                    'title': c['title'],
                    'section': c['section'],
                    'page_type': c.get('page_type', 'unknown'),
                    'tags': c.get('tags', []),
                    'content': c['content'],
                    'content_raw': c.get('content_raw', ''),
                    'tokens': c['tokens'],
                    'position': c['position'],
                }
                for c in self.chunks
            ],
        }
        Path(path).write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')
    
    def load(self, path: str):
        """Deserialize index from JSON file."""
        data = json.loads(Path(path).read_text(encoding='utf-8'))
        
        self.k1 = data.get('k1', 1.5)
        self.b = data.get('b', 0.75)
        self.N = data['N']
        self.avgdl = data['avgdl']
        self.doc_lengths = data['doc_lengths']
        self.built_at = data.get('built_at')
        self.wiki_dir = data.get('wiki_dir')
        self.df = defaultdict(int, {k: v for k, v in data['df'].items()})
        self.inverted_index = defaultdict(list, {
            k: [tuple(p) for p in v]
            for k, v in data['inverted_index'].items()
        })
        self.chunks = data['chunks']


# ============================================================================
# WIKI INDEXER — Crawls wiki and builds BM25 index
# ============================================================================

def find_wiki_dir(start: Optional[str] = None) -> Path:
    """Find the .wiki directory."""
    p = Path(start or os.getcwd()).resolve()
    
    # Check if start IS the wiki dir
    if p.name == '.wiki' and p.is_dir():
        return p
    
    # Check current and parents
    for d in [p] + list(p.parents):
        wiki = d / '.wiki'
        if wiki.is_dir():
            return wiki
    
    raise FileNotFoundError(
        f"No .wiki directory found from {p}. Run 'wiki.py init' first."
    )


def index_wiki(wiki_dir: str, max_chunk_size: int = 800, overlap: int = 100) -> BM25Index:
    """
    Crawl all markdown files in the wiki and build a BM25 index.
    
    Indexes: sources/, entities/, concepts/, analyses/, overview.md, index.md
    Skips: log.md, SCHEMA.md, _discovery.json, raw/
    """
    wd = Path(wiki_dir)
    all_chunks = []
    
    # Index subdirectories
    for subdir in ['sources', 'entities', 'concepts', 'analyses']:
        d = wd / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != '.md':
                continue
            try:
                content = f.read_text(encoding='utf-8')
                rel_path = str(f.relative_to(wd))
                chunks = chunk_markdown(content, rel_path,
                                        max_chunk_size=max_chunk_size,
                                        overlap=overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  Warning: Failed to index {f}: {e}", file=sys.stderr)
    
    # Index root-level files
    for fname in ['overview.md', 'index.md']:
        fp = wd / fname
        if fp.exists():
            try:
                content = fp.read_text(encoding='utf-8')
                chunks = chunk_markdown(content, fname,
                                        max_chunk_size=max_chunk_size,
                                        overlap=overlap)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"  Warning: Failed to index {fp}: {e}", file=sys.stderr)
    
    # Build BM25 index
    idx = BM25Index()
    idx.wiki_dir = str(wd)
    idx.build(all_chunks)
    
    return idx


# ============================================================================
# DOCUMENT INGESTION — Reads raw documents and prepares for wiki integration
# ============================================================================

IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.bmp', '.ico', '.tiff', '.tif'}

SUPPORTED_EXTENSIONS = {
    '.md', '.txt', '.rst', '.text',
    '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.go', '.rs', '.rb',
    '.c', '.cpp', '.h', '.hpp', '.cs', '.swift', '.kt',
    '.json', '.yaml', '.yml', '.toml', '.xml', '.csv',
    '.html', '.htm', '.css', '.scss',
    '.sh', '.bash', '.zsh',
    '.sql', '.r', '.m', '.lua',
    '.dockerfile',
}


def _ensure_assets_dir(wiki_dir: Path) -> Path:
    """Create and return raw/assets/ directory."""
    assets = wiki_dir / 'raw' / 'assets'
    assets.mkdir(parents=True, exist_ok=True)
    return assets


def _extract_image_refs(content: str) -> List[str]:
    """Extract image paths/URLs from markdown and HTML content."""
    refs = []
    # Markdown: ![alt](path)
    refs.extend(re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content))
    # HTML: <img src="...">
    refs.extend(re.findall(r'<img\s[^>]*src=["\']([^"\']+)["\']', content, re.IGNORECASE))
    return refs


def _copy_local_images(image_refs: List[str], source_dir: Path,
                       assets_dir: Path) -> Dict[str, str]:
    """
    Copy locally-referenced images to assets_dir.
    Returns a mapping {original_ref: new_relative_path} for path rewriting.
    """
    rewrite_map = {}
    for ref in image_refs:
        if ref.startswith(('http://', 'https://', 'data:')):
            continue  # skip URLs and data URIs — handled separately
        img_path = (source_dir / ref).resolve()
        if not img_path.is_file():
            continue
        if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        dest = assets_dir / img_path.name
        counter = 1
        while dest.exists() and dest.name != img_path.name:
            dest = assets_dir / f"{img_path.stem}_{counter}{img_path.suffix}"
            counter += 1
        if not dest.exists():
            shutil.copy2(str(img_path), str(dest))
        # Build relative path from raw/ to raw/assets/
        rewrite_map[ref] = f"assets/{dest.name}"
    return rewrite_map


def _download_remote_images(image_refs: List[str],
                            assets_dir: Path) -> Dict[str, str]:
    """
    Download remote image URLs to assets_dir.
    Returns a mapping {original_url: new_relative_path} for path rewriting.
    """
    import urllib.request
    import urllib.error
    from urllib.parse import urlparse

    rewrite_map = {}
    for ref in image_refs:
        if not ref.startswith(('http://', 'https://')):
            continue
        parsed = urlparse(ref)
        fname = Path(parsed.path).name
        if not fname or Path(fname).suffix.lower() not in IMAGE_EXTENSIONS:
            # Try to infer extension from URL or skip
            continue
        dest = assets_dir / fname
        counter = 1
        stem = Path(fname).stem
        suffix = Path(fname).suffix
        while dest.exists():
            dest = assets_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        try:
            req = urllib.request.Request(ref, headers={'User-Agent': 'Wiki-Agent/1.0'})
            with urllib.request.urlopen(req, timeout=15) as resp:
                dest.write_bytes(resp.read())
            rewrite_map[ref] = f"assets/{dest.name}"
        except Exception:
            pass  # skip images that fail to download
    return rewrite_map


def _rewrite_image_paths(content: str, rewrite_map: Dict[str, str]) -> str:
    """Replace image references in content with new paths."""
    for old_ref, new_ref in rewrite_map.items():
        content = content.replace(old_ref, new_ref)
    return content


def read_document(filepath: str) -> Dict:
    """
    Read a document and extract its text content.
    Returns dict with 'content', 'filename', 'extension', 'size', 'lines'.
    
    For binary formats (PDF, DOCX), returns a placeholder — the LLM agent
    should use appropriate tools (pdf-reading skill, etc.) to extract text first.
    """
    p = Path(filepath)
    ext = p.suffix.lower()
    
    result = {
        'filename': p.name,
        'filepath': str(p.resolve()),
        'extension': ext,
        'size_bytes': p.stat().st_size if p.exists() else 0,
    }
    
    if ext in SUPPORTED_EXTENSIONS or ext == '':
        try:
            text = p.read_text(encoding='utf-8')
            result['content'] = text
            result['lines'] = len(text.split('\n'))
            result['readable'] = True
        except UnicodeDecodeError:
            result['content'] = ''
            result['readable'] = False
            result['error'] = 'Binary file — use appropriate tool to extract text'
    elif ext == '.pdf':
        result['content'] = ''
        result['readable'] = False
        result['error'] = 'PDF file — use pdf-reading skill to extract text first'
    elif ext in ('.docx', '.doc'):
        result['content'] = ''
        result['readable'] = False
        result['error'] = 'Word document — use docx skill to extract text first'
    elif ext in ('.xlsx', '.xls'):
        result['content'] = ''
        result['readable'] = False
        result['error'] = 'Spreadsheet — use xlsx skill to extract text first'
    else:
        try:
            text = p.read_text(encoding='utf-8')
            result['content'] = text
            result['lines'] = len(text.split('\n'))
            result['readable'] = True
        except (UnicodeDecodeError, Exception):
            result['content'] = ''
            result['readable'] = False
            result['error'] = f'Unsupported format: {ext}'
    
    return result


# ============================================================================
# RAG CONTEXT BUILDER — Formats retrieval results for LLM consumption
# ============================================================================

def build_rag_context(results: List[Dict], query: str,
                      max_tokens_approx: int = 4000,
                      brief: bool = False) -> str:
    """
    Build a formatted context string from BM25 search results,
    ready to be injected into an LLM prompt.
    
    If brief=True, returns only title + section + first 2 sentences per chunk
    (~300 tokens total) for progressive disclosure. LLM can request full
    content for specific pages afterwards.
    
    Output format:
    
    <wiki_context query="...">
    <chunk source="entities/my-entity.md" section="Overview" score="12.34" type="entity">
    Content here...
    </chunk>
    ...
    </wiki_context>
    """
    if not results:
        return f'<wiki_context query="{query}">\nNo relevant wiki pages found.\n</wiki_context>'
    
    parts = [f'<wiki_context query="{query}" mode="{"brief" if brief else "full"}">']
    approx_tokens = 0
    
    for r in results:
        content = r['content']

        # Brief mode: first 2 sentences only
        if brief:
            raw = r.get('content_raw', content)
            sentences = re.split(r'(?<=[.!?])\s+', raw.strip())
            content = ' '.join(sentences[:2])
            if len(sentences) > 2:
                content += ' ...[use --top-k or read full page for more]'
        
        content_tokens = len(content.split()) * 1.3  # rough token estimate
        
        if approx_tokens + content_tokens > max_tokens_approx:
            # Truncate this chunk to fit
            remaining = int((max_tokens_approx - approx_tokens) / 1.3)
            words = content.split()
            if remaining > 50:
                content = ' '.join(words[:remaining]) + '...[truncated]'
            else:
                break
        
        tags_str = ', '.join(r.get('tags', []))
        chunk_xml = (
            f'<chunk source="{r["filepath"]}" section="{r["section"]}" '
            f'score="{r["score"]}" type="{r["page_type"]}"'
        )
        if tags_str:
            chunk_xml += f' tags="{tags_str}"'
        chunk_xml += f'>\n{content}\n</chunk>'
        
        parts.append(chunk_xml)
        approx_tokens += content_tokens
    
    parts.append('</wiki_context>')
    return '\n\n'.join(parts)


def build_marp_context(results: List[Dict], query: str) -> str:
    """
    Build a Marp-compatible markdown slide deck from search results.
    Each chunk becomes one slide. Useful for generating presentations
    from wiki knowledge.
    """
    slides = [
        '---\nmarp: true\ntheme: default\npaginate: true\n---\n',
        f'# Wiki: {query}\n\n---\n',
    ]
    for r in results:
        title = r.get('title', 'Untitled')
        section = r.get('section', '')
        content = r.get('content_raw', r['content'])
        # Truncate to fit a slide (~200 words)
        words = content.split()
        if len(words) > 200:
            content = ' '.join(words[:200]) + '...'
        slide = f'## {title} — {section}\n\n{content}\n\n_Source: {r["filepath"]}_\n\n---\n'
        slides.append(slide)
    return '\n'.join(slides)


# ============================================================================
# CLI COMMANDS
# ============================================================================

def cmd_index(args):
    """Build or rebuild the BM25 index (and Qdrant if hybrid is configured)."""
    wiki_dir = None
    chunk_size = 800
    overlap_size = 100
    bm25_only = False
    
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--chunk-size' and i + 1 < len(args):
            chunk_size = int(args[i + 1]); i += 2
        elif args[i] == '--overlap' and i + 1 < len(args):
            overlap_size = int(args[i + 1]); i += 2
        elif args[i] == '--bm25-only':
            bm25_only = True; i += 1
        else:
            i += 1
    
    wd = find_wiki_dir(wiki_dir)
    print(f"📇 Indexing wiki at {wd}...")
    
    idx = index_wiki(str(wd), max_chunk_size=chunk_size, overlap=overlap_size)
    
    # Save BM25 index
    index_path = wd / '_bm25_index.json'
    idx.save(str(index_path))
    
    print(f"   BM25 chunks indexed: {idx.N}")
    print(f"   Unique terms: {len(idx.df)}")
    print(f"   Avg chunk length: {idx.avgdl:.0f} tokens")
    print(f"   Index saved: {index_path}")
    
    # Index to Qdrant if hybrid is configured and not --bm25-only
    if not bm25_only and _hybrid_configured(wd):
        try:
            import time as _time
            from qdrant_store import QdrantStore
            t0 = _time.time()
            store = QdrantStore(wiki_dir=wd)
            store.drop_collection()
            store.upsert_chunks(idx.chunks)
            elapsed = _time.time() - t0
            print(f"   Qdrant indexed: {idx.N} chunks ({elapsed:.1f}s)")
        except Exception as e:
            print(f"   ⚠️  Qdrant indexing failed: {e}", file=sys.stderr)
            print(f"   BM25 index is still up to date.", file=sys.stderr)
    elif not bm25_only and _hybrid_available():
        print(f"   ℹ️  Hybrid deps installed but not configured (set OPENAI_API_KEY)")
    
    # Print breakdown by type
    type_counts = Counter(c.get('page_type', 'unknown') for c in idx.chunks)
    file_counts = Counter(c['filepath'].split('/')[0] for c in idx.chunks)
    print(f"\n   By directory:")
    for d, count in sorted(file_counts.items(), key=lambda x: -x[1]):
        print(f"     {d}: {count} chunks")


def cmd_search(args):
    """Search the index and display ranked results."""
    wiki_dir = None
    top_k = 10
    type_filter = None
    tag_filter = None
    backend = 'auto'
    no_rerank = False
    query_parts = []
    
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--top-k' and i + 1 < len(args):
            top_k = int(args[i + 1]); i += 2
        elif args[i] == '--type' and i + 1 < len(args):
            type_filter = args[i + 1]; i += 2
        elif args[i] == '--tag' and i + 1 < len(args):
            tag_filter = args[i + 1]; i += 2
        elif args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]; i += 2
        elif args[i] == '--no-rerank':
            no_rerank = True; i += 1
        else:
            query_parts.append(args[i]); i += 1
    
    if not query_parts:
        print("Usage: bm25_retriever.py search 'query' [--top-k N] [--backend auto|bm25|hybrid|qdrant]", file=sys.stderr)
        sys.exit(1)
    
    query = ' '.join(query_parts)
    wd = find_wiki_dir(wiki_dir)
    backend = _resolve_backend(backend, wd)
    
    # Hybrid backend: BM25 + Qdrant + RRF + optional LLM rerank
    if backend == 'hybrid':
        try:
            from qdrant_store import QdrantStore, _load_config
            from fusion import hybrid_search
            
            config = _load_config(wd)
            
            # BM25 search
            index_path = wd / '_bm25_index.json'
            if not index_path.exists():
                print("BM25 index not found. Building now...", file=sys.stderr)
                cmd_index(['--wiki-dir', str(wd)])
            idx = BM25Index()
            idx.load(str(index_path))
            bm25_results = idx.search(query, top_k=50,
                                      page_type_filter=type_filter,
                                      tag_filter=tag_filter)
            
            # Qdrant semantic search
            store = QdrantStore(wiki_dir=wd)
            qdrant_results = store.search(query, top_k=50,
                                          page_type_filter=type_filter,
                                          tag_filter=tag_filter)
            
            # Fuse + rerank
            results = hybrid_search(
                query=query,
                bm25_results=bm25_results,
                qdrant_results=qdrant_results,
                rrf_k=config.get('rrf_k', 60),
                rerank=not no_rerank,
                rerank_top_n=config.get('rerank_top_n', 20),
                rerank_model=config.get('rerank_model', 'gpt-4o-mini'),
                openai_api_key=config.get('openai_api_key'),
                top_k=top_k,
            )
            
            if not results:
                print(f"No results for: {query}")
                return
            
            label = "Hybrid" if not no_rerank else "Hybrid (no rerank)"
            print(f"🔍 {label} Results for: \"{query}\" ({len(results)} hits)\n")
            for i_r, r in enumerate(results):
                tags = ', '.join(r.get('tags', [])) if r.get('tags') else ''
                bm25_r = r.get('bm25_rank')
                qdrant_r = r.get('qdrant_rank')
                rerank_s = r.get('rerank_score')
                provenance = []
                if bm25_r is not None:
                    provenance.append(f"BM25#{bm25_r}")
                if qdrant_r is not None:
                    provenance.append(f"Qdrant#{qdrant_r}")
                prov_str = ' + '.join(provenance) if provenance else ''
                score_str = f"rrf={r.get('rrf_score', 0):.4f}"
                if rerank_s is not None and rerank_s >= 0:
                    score_str += f" rerank={rerank_s:.1f}"
                
                print(f"  {i_r+1}. [{score_str}] {r['title']} > {r['section']}")
                print(f"     {r['filepath']} ({r['page_type']}) [{prov_str}]")
                if tags:
                    print(f"     tags: {tags}")
                preview = r['content'][:150].replace('\n', ' ').strip()
                print(f"     \"{preview}...\"")
                print()
            return
        except Exception as e:
            print(f"  ⚠️  Hybrid search failed: {e}, falling back to BM25", file=sys.stderr)
    
    # Qdrant-only backend
    if backend == 'qdrant':
        try:
            from qdrant_store import QdrantStore
            store = QdrantStore(wiki_dir=wd)
            results = store.search(query, top_k=top_k,
                                   page_type_filter=type_filter,
                                   tag_filter=tag_filter)
            if not results:
                print(f"No results for: {query}")
                return
            print(f"🔍 Qdrant Results for: \"{query}\" ({len(results)} hits)\n")
            for i_r, r in enumerate(results):
                tags = ', '.join(r.get('tags', [])) if r.get('tags') else ''
                print(f"  {i_r+1}. [{r['score']:.4f}] {r['title']} > {r['section']}")
                print(f"     {r['filepath']} ({r['page_type']})")
                if tags:
                    print(f"     tags: {tags}")
                preview = r['content'][:150].replace('\n', ' ').strip()
                print(f"     \"{preview}...\"")
                print()
            return
        except Exception as e:
            print(f"  ⚠️  Qdrant search failed: {e}, falling back to BM25", file=sys.stderr)
    
    # BM25 fallback
    index_path = wd / '_bm25_index.json'
    if not index_path.exists():
        print("Index not found. Building now...", file=sys.stderr)
        cmd_index(['--wiki-dir', str(wd)])
    
    idx = BM25Index()
    idx.load(str(index_path))
    
    results = idx.search(query, top_k=top_k,
                         page_type_filter=type_filter,
                         tag_filter=tag_filter)
    
    if not results:
        print(f"No results for: {query}")
        return
    
    print(f"🔍 BM25 Results for: \"{query}\" ({len(results)} hits)\n")
    for i, r in enumerate(results):
        tags = ', '.join(r.get('tags', [])) if r.get('tags') else ''
        print(f"  {i+1}. [{r['score']:.2f}] {r['title']} > {r['section']}")
        print(f"     {r['filepath']} ({r['page_type']})")
        if tags:
            print(f"     tags: {tags}")
        # Preview first 120 chars of raw content
        preview = r['content'][:150].replace('\n', ' ').strip()
        print(f"     \"{preview}...\"")
        print()


def cmd_retrieve(args):
    """Search and output full context for RAG pipeline (JSON format)."""
    wiki_dir = None
    top_k = 5
    max_context_tokens = 4000
    output_format = 'xml'  # xml, json, or marp
    type_filter = None
    brief = False
    freshness_weight = 0.0
    centrality_weight = 0.0
    backend = 'auto'
    no_rerank = False
    query_parts = []
    
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--top-k' and i + 1 < len(args):
            top_k = int(args[i + 1]); i += 2
        elif args[i] == '--max-tokens' and i + 1 < len(args):
            max_context_tokens = int(args[i + 1]); i += 2
        elif args[i] == '--format' and i + 1 < len(args):
            output_format = args[i + 1]; i += 2
        elif args[i] == '--type' and i + 1 < len(args):
            type_filter = args[i + 1]; i += 2
        elif args[i] == '--brief':
            brief = True; i += 1
        elif args[i] == '--freshness-weight' and i + 1 < len(args):
            freshness_weight = float(args[i + 1]); i += 2
        elif args[i] == '--centrality-weight' and i + 1 < len(args):
            centrality_weight = float(args[i + 1]); i += 2
        elif args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]; i += 2
        elif args[i] == '--no-rerank':
            no_rerank = True; i += 1
        else:
            query_parts.append(args[i]); i += 1
    
    if not query_parts:
        print("Usage: bm25_retriever.py retrieve 'query' [--top-k N] [--format xml|json|marp] [--brief] [--backend auto|bm25|hybrid|qdrant]",
              file=sys.stderr)
        sys.exit(1)
    
    query = ' '.join(query_parts)
    wd = find_wiki_dir(wiki_dir)
    backend = _resolve_backend(backend, wd)
    
    results = None
    
    # Hybrid backend: BM25 + Qdrant + RRF + LLM rerank
    if backend == 'hybrid':
        try:
            from qdrant_store import QdrantStore, _load_config
            from fusion import hybrid_search
            
            config = _load_config(wd)
            
            # BM25 search
            index_path = wd / '_bm25_index.json'
            if not index_path.exists():
                idx = index_wiki(str(wd))
                idx.save(str(index_path))
            else:
                idx = BM25Index()
                idx.load(str(index_path))
            
            # Load centrality data if boosting requested
            centrality_data = None
            if centrality_weight > 0:
                centrality_path = wd / '_centrality.json'
                if centrality_path.exists():
                    centrality_data = json.loads(centrality_path.read_text(encoding='utf-8'))
            
            bm25_results = idx.search(query, top_k=50,
                                      page_type_filter=type_filter,
                                      freshness_weight=freshness_weight,
                                      centrality_data=centrality_data,
                                      centrality_weight=centrality_weight)
            
            # Qdrant semantic search
            store = QdrantStore(wiki_dir=wd)
            qdrant_results = store.search(query, top_k=50,
                                          page_type_filter=type_filter)
            
            # Fuse + rerank
            results = hybrid_search(
                query=query,
                bm25_results=bm25_results,
                qdrant_results=qdrant_results,
                rrf_k=config.get('rrf_k', 60),
                rerank=not no_rerank,
                rerank_top_n=config.get('rerank_top_n', 20),
                rerank_model=config.get('rerank_model', 'gpt-4o-mini'),
                openai_api_key=config.get('openai_api_key'),
                top_k=top_k,
            )
        except Exception as e:
            print(f"  ⚠️  Hybrid retrieve failed: {e}, falling back to BM25", file=sys.stderr)
            results = None
    
    # Qdrant-only backend
    if backend == 'qdrant' and results is None:
        try:
            from qdrant_store import QdrantStore
            store = QdrantStore(wiki_dir=wd)
            results = store.search(query, top_k=top_k,
                                   page_type_filter=type_filter)
        except Exception as e:
            print(f"  ⚠️  Qdrant retrieve failed: {e}, falling back to BM25", file=sys.stderr)
    
    # BM25 fallback
    if results is None:
        index_path = wd / '_bm25_index.json'
        if not index_path.exists():
            idx = index_wiki(str(wd))
            idx.save(str(index_path))
        else:
            idx = BM25Index()
            idx.load(str(index_path))
        
        # Load centrality data if boosting requested
        centrality_data = None
        if centrality_weight > 0:
            centrality_path = wd / '_centrality.json'
            if centrality_path.exists():
                centrality_data = json.loads(centrality_path.read_text(encoding='utf-8'))
        
        results = idx.search(query, top_k=top_k, page_type_filter=type_filter,
                             freshness_weight=freshness_weight,
                             centrality_data=centrality_data,
                             centrality_weight=centrality_weight)
    
    if output_format == 'json':
        print(json.dumps({
            'query': query,
            'backend': backend,
            'results': results,
        }, indent=2, ensure_ascii=False))
    elif output_format == 'marp':
        print(build_marp_context(results, query))
    else:
        print(build_rag_context(results, query,
                                max_tokens_approx=max_context_tokens,
                                brief=brief))


def cmd_stats(args):
    """Show index statistics."""
    wiki_dir = None
    for i, a in enumerate(args):
        if a == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]
    
    wd = find_wiki_dir(wiki_dir)
    index_path = wd / '_bm25_index.json'
    
    if not index_path.exists():
        print("No index found. Run 'bm25_retriever.py index' first.")
        return
    
    idx = BM25Index()
    idx.load(str(index_path))
    
    # Compute statistics
    type_counts = Counter(c.get('page_type', 'unknown') for c in idx.chunks)
    file_counts = Counter(c['filepath'].split('/')[0] if '/' in c['filepath'] else 'root'
                          for c in idx.chunks)
    all_tags = Counter()
    for c in idx.chunks:
        for t in c.get('tags', []):
            all_tags[t] += 1
    
    # Top terms by document frequency
    top_terms = sorted(idx.df.items(), key=lambda x: -x[1])[:20]
    
    print(f"📊 BM25 Index Statistics")
    print(f"   Wiki: {wd}")
    print(f"   Built: {idx.built_at}")
    print(f"   Total chunks: {idx.N}")
    print(f"   Unique terms: {len(idx.df)}")
    print(f"   Avg chunk length: {idx.avgdl:.0f} stemmed tokens")
    print(f"   BM25 params: k1={idx.k1}, b={idx.b}")
    
    print(f"\n   Chunks by page type:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"     {t}: {c}")
    
    print(f"\n   Chunks by directory:")
    for d, c in sorted(file_counts.items(), key=lambda x: -x[1]):
        print(f"     {d}: {c}")
    
    if all_tags:
        print(f"\n   Top tags:")
        for tag, c in all_tags.most_common(15):
            print(f"     {tag}: {c}")
    
    print(f"\n   Top terms (by document frequency):")
    for term, df in top_terms:
        print(f"     \"{term}\": in {df}/{idx.N} chunks ({100*df/idx.N:.0f}%)")


def cmd_answer(args):
    """
    Retrieve relevant chunks and synthesize an answer using LLM.
    
    Flags:
        --raw         Skip LLM synthesis, return raw chunks only
        --file-answer File the answer as an analysis page in the wiki
        --top-k N     Number of chunks to retrieve (default 5)
        --backend     Search backend: auto|bm25|hybrid|qdrant
        --no-rerank   Skip LLM reranking
        --wiki-dir    Path to wiki
    """
    wiki_dir = None
    query_parts = []
    top_k = 5
    raw_mode = False
    file_answer_flag = False
    backend = 'auto'
    no_rerank = False

    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--top-k' and i + 1 < len(args):
            top_k = int(args[i + 1]); i += 2
        elif args[i] == '--backend' and i + 1 < len(args):
            backend = args[i + 1]; i += 2
        elif args[i] == '--raw':
            raw_mode = True; i += 1
        elif args[i] == '--file-answer':
            file_answer_flag = True; i += 1
        elif args[i] == '--no-rerank':
            no_rerank = True; i += 1
        else:
            query_parts.append(args[i]); i += 1

    if not query_parts:
        print("Usage: bm25_retriever.py answer QUERY [--top-k N] [--wiki-dir PATH] "
              "[--raw] [--file-answer]", file=sys.stderr)
        sys.exit(1)

    query = ' '.join(query_parts)
    wd = find_wiki_dir(wiki_dir)
    backend = _resolve_backend(backend, wd)

    # Retrieve chunks using the existing search infrastructure
    index_path = wd / '_bm25_index.json'
    if not index_path.exists():
        print("No index found. Run 'bm25_retriever.py index' first.", file=sys.stderr)
        sys.exit(1)

    idx = BM25Index()
    idx.load(str(index_path))

    if backend in ('hybrid', 'qdrant') and _hybrid_configured(wd):
        try:
            from fusion import hybrid_search
            from qdrant_store import QdrantStore
            store = QdrantStore(wiki_dir=wd)
            results = hybrid_search(
                query=query, bm25_index=idx, qdrant_store=store,
                top_k=top_k, rerank=(not no_rerank),
            )
        except Exception as e:
            print(f"  ⚠️  Hybrid search failed, falling back to BM25: {e}",
                  file=sys.stderr)
            results = idx.search(query, top_k=top_k)
    else:
        results = idx.search(query, top_k=top_k)

    if not results:
        print(f"No results found for: {query}")
        return

    # Build chunks list
    chunks = []
    for r in results:
        chunk = idx.chunks[r[0]] if isinstance(r, tuple) else r
        chunks.append(chunk)

    if raw_mode:
        # Raw mode: just print chunks
        for i_c, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i_c+1} ({chunk.get('filepath', '?')}) ---")
            print(chunk.get('content', ''))
        return

    # Synthesize answer using LLM
    try:
        from wiki_compiler import synthesize_answer, file_answer
    except ImportError:
        print("wiki_compiler module not found. Use --raw for raw chunks.", file=sys.stderr)
        sys.exit(1)

    print(f"🧠 Synthesizing answer for: {query}\n")
    answer = synthesize_answer(query, chunks, wiki_dir=wd)

    if answer is None:
        print("⚠️  LLM synthesis unavailable. Showing raw chunks instead:\n")
        for i_c, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i_c+1} ({chunk.get('filepath', '?')}) ---")
            print(chunk.get('content', ''))
        return

    print(answer)

    # File the answer as an analysis page if requested
    if file_answer_flag:
        source_paths = list(set(c.get('filepath', '') for c in chunks if c.get('filepath')))
        analysis_path = file_answer(wd, query, answer, source_paths)
        if analysis_path:
            print(f"\n📝 Answer filed as: {analysis_path}")

            # Re-index after filing
            print("🔄 Rebuilding index...")
            try:
                new_idx = index_wiki(str(wd))
                new_idx.save(str(index_path))
                print(f"   Index rebuilt: {new_idx.N} chunks")
            except Exception as e:
                print(f"   ⚠️  Index rebuild failed: {e}", file=sys.stderr)


def cmd_ingest_file(args):
    """
    Ingest a raw document or URL: copy to raw/, extract text, chunk it,
    and output structured JSON for the LLM agent to use when
    creating wiki pages. Auto-rebuilds BM25 index unless --no-index.
    """
    wiki_dir = None
    file_paths = []
    auto_index = True
    
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--no-index':
            auto_index = False; i += 1
        else:
            file_paths.append(args[i]); i += 1
    
    if not file_paths:
        print("Usage: bm25_retriever.py ingest FILE|URL [FILE|URL...] [--wiki-dir PATH] [--no-index]",
              file=sys.stderr)
        sys.exit(1)
    
    wd = find_wiki_dir(wiki_dir)
    raw_dir = wd / 'raw'
    raw_dir.mkdir(exist_ok=True)
    
    results = []
    for fp in file_paths:
        # URL ingestion: fetch and save as markdown
        if fp.startswith('http://') or fp.startswith('https://'):
            doc = _ingest_url(fp, raw_dir, wd)
            results.append(doc)
            if doc.get('readable'):
                img_count = len(doc.get('images_copied', []))
                img_msg = f" + {img_count} images" if img_count else ""
                print(f"  ✅ Fetched: {fp} → {doc.get('raw_copy', '')}{img_msg}")
            else:
                print(f"  ⚠️  Failed to fetch: {fp} — {doc.get('error', 'unknown')}")
            continue

        p = Path(fp)
        if not p.exists():
            print(f"  ⚠️  File not found: {fp}", file=sys.stderr)
            continue
        
        # Copy to raw/
        dest = raw_dir / p.name
        counter = 1
        while dest.exists():
            dest = raw_dir / f"{p.stem}_{counter}{p.suffix}"
            counter += 1
        shutil.copy2(str(p), str(dest))
        
        # Read and extract text
        doc = read_document(str(p))
        doc['raw_copy'] = str(dest.relative_to(wd))
        
        # Auto-handle images: detect references, copy to raw/assets/, rewrite paths
        images_copied = []
        if doc.get('readable') and doc.get('content'):
            image_refs = _extract_image_refs(doc['content'])
            if image_refs:
                assets_dir = _ensure_assets_dir(wd)
                source_dir = p.resolve().parent
                local_map = _copy_local_images(image_refs, source_dir, assets_dir)
                remote_map = _download_remote_images(image_refs, assets_dir)
                rewrite_map = {**local_map, **remote_map}
                if rewrite_map:
                    rewritten = _rewrite_image_paths(doc['content'], rewrite_map)
                    dest.write_text(rewritten, encoding='utf-8')
                    doc['content'] = rewritten
                    images_copied = list(rewrite_map.values())
        doc['images_copied'] = images_copied
        
        results.append(doc)
        
        if doc['readable']:
            img_msg = f" + {len(images_copied)} images" if images_copied else ""
            print(f"  ✅ Ingested: {p.name} ({doc.get('lines', 0)} lines){img_msg} → {dest.relative_to(wd)}")
        else:
            print(f"  ⚠️  Copied but not readable: {p.name} — {doc.get('error', 'unknown format')}")
    
    # Output structured data for the LLM agent
    print(f"\n📋 Ingestion summary ({len(results)} files):")
    print(json.dumps([{
        'filename': r['filename'],
        'extension': r['extension'],
        'readable': r.get('readable', False),
        'lines': r.get('lines', 0),
        'size_bytes': r.get('size_bytes', 0),
        'raw_copy': r.get('raw_copy', ''),
        'error': r.get('error', ''),
        'images_copied': r.get('images_copied', []),
        'content_preview': r.get('content', '')[:500] if r.get('readable') else '',
    } for r in results], indent=2, ensure_ascii=False))

    # Auto-rebuild BM25 index (event-driven, not manual)
    if auto_index:
        print("\n🔄 Auto-rebuilding BM25 index...")
        try:
            idx = index_wiki(str(wd))
            index_path = wd / '_bm25_index.json'
            idx.save(str(index_path))
            print(f"   Index rebuilt: {idx.N} chunks, {len(idx.df)} terms")
            
            # Sync Qdrant if hybrid is configured
            if _hybrid_configured(wd):
                print("🔄 Syncing Qdrant index...")
                try:
                    from qdrant_store import QdrantStore
                    store = QdrantStore(wiki_dir=wd)
                    store.drop_collection()
                    store.upsert_chunks(idx.chunks)
                    print(f"   Qdrant synced: {idx.N} chunks")
                except Exception as e:
                    print(f"   ⚠️  Qdrant sync failed: {e}", file=sys.stderr)
                    print(f"   BM25 index is still up to date.", file=sys.stderr)
        except Exception as e:
            print(f"   ⚠️  Index rebuild failed: {e}", file=sys.stderr)

    # Auto-compile ingested files into wiki pages
    auto_compile = '--no-compile' not in args
    if auto_compile:
        try:
            from wiki_compiler import compile_file, is_compiler_available
            if is_compiler_available():
                print("\n🧠 Auto-compiling into wiki pages...")
                for r in results:
                    if not r.get('readable') or not r.get('raw_copy'):
                        continue
                    raw_rel = r['raw_copy']
                    try:
                        cr = compile_file(wd, raw_rel)
                        if cr.get('compiled') and not cr.get('skipped'):
                            pages = cr.get('pages_created', [])
                            print(f"   ✅ {raw_rel}: {len(pages)} pages created")
                        elif cr.get('skipped'):
                            print(f"   ⏭️  {raw_rel}: already compiled")
                        else:
                            err = cr.get('error', 'unknown')
                            print(f"   ⚠️  {raw_rel}: compilation failed — {err}")
                    except Exception as e:
                        print(f"   ⚠️  {raw_rel}: compile error — {e}", file=sys.stderr)
            else:
                print("\n💡 Set OPENAI_API_KEY to enable auto-compilation into wiki pages.")
        except ImportError:
            pass  # wiki_compiler not available, skip silently


def _ingest_url(url: str, raw_dir: Path, wiki_dir: Path) -> Dict:
    """Fetch a URL, strip HTML to plain text, download images, and save as markdown in raw/."""
    import urllib.request
    import urllib.error

    result = {
        'filename': '',
        'filepath': url,
        'extension': '.md',
        'size_bytes': 0,
    }
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Wiki-Agent/1.0'})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_html = resp.read().decode('utf-8', errors='replace')

        # Extract image URLs from HTML before stripping tags
        from urllib.parse import urlparse, urljoin
        img_urls = re.findall(r'<img\s[^>]*src=["\']([^"\']+)["\']', raw_html, re.IGNORECASE)
        # Resolve relative URLs to absolute
        img_urls = [urljoin(url, u) for u in img_urls]

        # Download images to raw/assets/
        images_copied = []
        if img_urls:
            assets_dir = _ensure_assets_dir(wiki_dir)
            rewrite_map = _download_remote_images(img_urls, assets_dir)
            images_copied = list(rewrite_map.values())

        # Strip HTML tags to get plain text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to basic markdown
        text = re.sub(r' {2,}', '\n\n', text)

        # Append image references as markdown at the end
        if images_copied:
            text += '\n\n---\n\n**Images:**\n'
            for img_path in images_copied:
                text += f'![image]({img_path})\n'

        # Generate filename from URL
        parsed = urlparse(url)
        slug = re.sub(r'[^a-z0-9]+', '-', parsed.path.lower().strip('/')).strip('-')
        if not slug:
            slug = re.sub(r'[^a-z0-9]+', '-', parsed.netloc.lower()).strip('-')
        slug = slug[:60]
        today = datetime.now().strftime('%Y-%m-%d')
        fname = f"{today}-{slug}.md"

        dest = raw_dir / fname
        counter = 1
        while dest.exists():
            dest = raw_dir / f"{today}-{slug}_{counter}.md"
            counter += 1

        # Write with source URL in header
        content = f"<!-- Source: {url} -->\n<!-- Fetched: {today} -->\n\n{text}"
        dest.write_text(content, encoding='utf-8')

        result['filename'] = dest.name
        result['raw_copy'] = str(dest.relative_to(wiki_dir))
        result['content'] = text
        result['lines'] = len(text.split('\n'))
        result['size_bytes'] = len(content.encode('utf-8'))
        result['readable'] = True
        result['images_copied'] = images_copied
    except (urllib.error.URLError, Exception) as e:
        result['readable'] = False
        result['error'] = f'Failed to fetch URL: {e}'
    return result


# ============================================================================
# MAIN
# ============================================================================

USAGE = """
wiki-agent BM25 + Hybrid Retriever

Usage:
    bm25_retriever.py index     [--wiki-dir PATH] [--chunk-size N] [--overlap N] [--bm25-only]
    bm25_retriever.py search    QUERY [--top-k N] [--type TYPE] [--tag TAG]
                                      [--backend auto|bm25|hybrid|qdrant] [--no-rerank]
                                      [--wiki-dir PATH]
    bm25_retriever.py retrieve  QUERY [--top-k N] [--format xml|json|marp] [--brief]
                                      [--freshness-weight F] [--centrality-weight F]
                                      [--backend auto|bm25|hybrid|qdrant] [--no-rerank]
                                      [--max-tokens N] [--wiki-dir PATH]
    bm25_retriever.py ingest    FILE|URL [FILE|URL...] [--wiki-dir PATH] [--no-index] [--no-compile]
    bm25_retriever.py answer    QUERY [--top-k N] [--wiki-dir PATH] [--raw] [--file-answer]
                                      [--backend auto|bm25|hybrid|qdrant] [--no-rerank]
    bm25_retriever.py stats     [--wiki-dir PATH]

Commands:
    index      Build/rebuild BM25 index (+ Qdrant if hybrid configured)
    search     Search and display ranked results (human-readable)
    retrieve   Search and output full context for RAG pipeline (XML, JSON, or Marp)
    ingest     Copy raw files or fetch URLs into wiki, extract text, auto-compile wiki pages
    answer     Retrieve + LLM-synthesize an answer with citations (+ optionally file it)
    stats      Show index statistics and top terms

Search backend:
    --backend auto        Use hybrid if configured, otherwise BM25 (default)
    --backend bm25        Force pure-Python BM25 search (zero dependencies)
    --backend hybrid      BM25 + Qdrant semantic + RRF fusion + LLM reranking
    --backend qdrant      Qdrant semantic search only (no BM25)
    --no-rerank           Skip LLM reranking step (use RRF scores directly)
    --bm25-only           Index command: skip Qdrant, only build BM25 index

Hybrid search setup:
    1. pip install 'wiki-agent[hybrid]'   (installs qdrant-client + openai)
    2. docker run -p 6333:6333 qdrant/qdrant
    3. export OPENAI_API_KEY=sk-...
    4. python bm25_retriever.py index       (builds BM25 + Qdrant indexes)

Retrieval flags:
    --brief               Return title + section + first 2 sentences per chunk (~300 tokens)
    --freshness-weight F  Boost recently-updated pages (0.0-1.0, default 0.0)
    --centrality-weight F Boost well-connected hub pages (0.0-1.0, default 0.0)
    --format marp         Output as Marp slide deck

BM25 Parameters (configurable in code):
    k1=1.5     Term frequency saturation (higher = raw TF matters more)
    b=0.75     Length normalization (0 = no normalization, 1 = full normalization)
""".strip()


def main():
    if len(sys.argv) < 2:
        print(USAGE)
        sys.exit(0)
    
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    
    commands = {
        'index': cmd_index,
        'search': cmd_search,
        'retrieve': cmd_retrieve,
        'answer': cmd_answer,
        'ingest': cmd_ingest_file,
        'stats': cmd_stats,
    }
    
    if cmd in commands:
        try:
            commands[cmd](rest)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Unknown command: {cmd}")
        print(USAGE)
        sys.exit(1)


if __name__ == '__main__':
    main()

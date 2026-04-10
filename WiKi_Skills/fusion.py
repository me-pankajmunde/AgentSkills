#!/usr/bin/env python3
"""
Reciprocal Rank Fusion (RRF) + LLM Reranking for Wiki Agent Hybrid Retrieval.

Fuses results from multiple retrieval backends (BM25, Qdrant) into a single
ranked list using RRF, then optionally reranks the top candidates using an LLM.

RRF Formula:
    score(d) = Σ  1 / (k + rank_i(d))
             for each ranked list i

Where k is a constant (default 60) that dampens the influence of high ranks.

References:
    Cormack, Clarke & Buettcher (2009). "Reciprocal Rank Fusion outperforms
    Condorcet and individual Rank Learning Methods."
"""

import json
import sys
from typing import List, Dict, Optional, Any
from collections import defaultdict


def rrf_fuse(ranked_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Each input list is a list of result dicts with at least 'chunk_id'.
    Results are deduplicated by chunk_id and scored across all lists.

    Args:
        ranked_lists: List of ranked result lists (e.g., [bm25_results, qdrant_results])
        k: RRF constant (default 60). Higher values give more weight to lower-ranked results.

    Returns:
        Fused results sorted by RRF score descending. Each result includes:
        - All original fields from the highest-scoring occurrence
        - rrf_score: The fused RRF score
        - source_ranks: Dict mapping source index to rank (1-indexed)
    """
    # Accumulate RRF scores by chunk_id
    scores: Dict[str, float] = defaultdict(float)
    # Keep the best version of each chunk (from highest-scoring source)
    best_result: Dict[str, Dict] = {}
    # Track ranks per source
    source_ranks: Dict[str, Dict[int, int]] = defaultdict(dict)

    for list_idx, results in enumerate(ranked_lists):
        for rank, result in enumerate(results, start=1):
            chunk_id = result['chunk_id']
            rrf_score = 1.0 / (k + rank)
            scores[chunk_id] += rrf_score
            source_ranks[chunk_id][list_idx] = rank

            # Keep the result with the most metadata
            if chunk_id not in best_result:
                best_result[chunk_id] = result.copy()

    # Build fused results
    fused = []
    for chunk_id, rrf_score in scores.items():
        result = best_result[chunk_id].copy()
        result['rrf_score'] = round(rrf_score, 6)
        result['source_ranks'] = source_ranks[chunk_id]
        # Replace the original score with the RRF score for sorting
        result['original_score'] = result.get('score', 0.0)
        result['score'] = round(rrf_score, 6)
        fused.append(result)

    # Sort by RRF score descending
    fused.sort(key=lambda x: -x['rrf_score'])

    return fused


def llm_rerank(query: str, candidates: List[Dict],
               top_n: int = 20,
               model: str = 'gpt-4o-mini',
               openai_api_key: Optional[str] = None) -> List[Dict]:
    """
    Rerank candidates using an LLM to score relevance.

    Sends the query + candidate passages to the LLM and asks it to score
    each passage 0-10 for relevance. Results are re-sorted by LLM score.

    Falls back to original order if reranking fails (API error, parse error, etc).

    Args:
        query: The search query
        candidates: List of result dicts to rerank (from RRF or single-source)
        top_n: Max number of candidates to send to the LLM
        model: OpenAI model to use for reranking
        openai_api_key: OpenAI API key (falls back to OPENAI_API_KEY env var)

    Returns:
        Reranked results with 'rerank_score' field added.
    """
    if not candidates:
        return candidates

    # Limit candidates sent to LLM
    to_rerank = candidates[:top_n]
    remainder = candidates[top_n:]

    try:
        import openai
        import os

        api_key = openai_api_key or os.environ.get('OPENAI_API_KEY')
        if not api_key:
            print("  ⚠️  No OPENAI_API_KEY for reranking, using RRF order",
                  file=sys.stderr)
            return candidates

        client = openai.OpenAI(api_key=api_key)

        # Build the reranking prompt
        passages = []
        for i, c in enumerate(to_rerank):
            # Use raw content (without context prefix) for cleaner scoring
            text = c.get('content_raw', c.get('content', ''))
            # Truncate long passages to keep prompt manageable
            words = text.split()
            if len(words) > 150:
                text = ' '.join(words[:150]) + '...'
            passages.append(f"[{i}] ({c.get('filepath', 'unknown')} > {c.get('section', '')})\n{text}")

        passages_text = '\n\n'.join(passages)

        prompt = f"""You are a relevance judge. Given a search query and a list of passages, 
score each passage from 0 to 10 for how relevant it is to the query.

Query: "{query}"

Passages:
{passages_text}

Return ONLY a JSON array of objects with "index" and "score" fields.
Example: [{{"index": 0, "score": 8}}, {{"index": 1, "score": 3}}]

Score all {len(to_rerank)} passages. Be precise — 10 means perfectly relevant, 0 means completely irrelevant."""

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        # Parse LLM response
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)

        # Handle both {"scores": [...]} and direct [...] formats
        if isinstance(parsed, dict):
            scores_list = parsed.get('scores', parsed.get('results', []))
            if not scores_list:
                # Try to find any list value in the dict
                for v in parsed.values():
                    if isinstance(v, list):
                        scores_list = v
                        break
        elif isinstance(parsed, list):
            scores_list = parsed
        else:
            raise ValueError(f"Unexpected rerank response format: {type(parsed)}")

        # Build score mapping
        score_map = {}
        for item in scores_list:
            idx = item.get('index', item.get('idx', item.get('i', -1)))
            score = item.get('score', item.get('relevance', 0))
            if 0 <= idx < len(to_rerank):
                score_map[idx] = float(score)

        # Apply scores and re-sort
        for i, c in enumerate(to_rerank):
            c['rerank_score'] = score_map.get(i, 0.0)

        to_rerank.sort(key=lambda x: -x.get('rerank_score', 0.0))

        # Assign declining scores to remainder (not reranked)
        for c in remainder:
            c['rerank_score'] = -1.0  # Indicates not reranked

        return to_rerank + remainder

    except ImportError:
        print("  ⚠️  openai not installed, skipping LLM reranking", file=sys.stderr)
        return candidates
    except json.JSONDecodeError as e:
        print(f"  ⚠️  Failed to parse rerank response: {e}, using RRF order",
              file=sys.stderr)
        return candidates
    except Exception as e:
        print(f"  ⚠️  LLM reranking failed: {e}, using RRF order",
              file=sys.stderr)
        return candidates


def hybrid_search(query: str,
                  bm25_results: List[Dict],
                  qdrant_results: List[Dict],
                  rrf_k: int = 60,
                  rerank: bool = True,
                  rerank_top_n: int = 20,
                  rerank_model: str = 'gpt-4o-mini',
                  openai_api_key: Optional[str] = None,
                  top_k: int = 10) -> List[Dict]:
    """
    Full hybrid search pipeline: BM25 + Qdrant → RRF → LLM Rerank → top-K.

    This is the main entry point for hybrid retrieval.

    Args:
        query: Search query
        bm25_results: Results from BM25Index.search()
        qdrant_results: Results from QdrantStore.search()
        rrf_k: RRF constant (default 60)
        rerank: Whether to apply LLM reranking after RRF
        rerank_top_n: How many RRF results to send to the LLM reranker
        rerank_model: OpenAI model for reranking
        openai_api_key: OpenAI API key
        top_k: Final number of results to return

    Returns:
        Final ranked results with rrf_score and optionally rerank_score.
    """
    # Add source labels for provenance tracking
    for r in bm25_results:
        r['_source'] = 'bm25'
    for r in qdrant_results:
        r['_source'] = 'qdrant'

    # RRF Fusion
    fused = rrf_fuse([bm25_results, qdrant_results], k=rrf_k)

    # Annotate with readable source rank names
    for r in fused:
        ranks = r.get('source_ranks', {})
        r['bm25_rank'] = ranks.get(0)    # Index 0 = bm25_results
        r['qdrant_rank'] = ranks.get(1)   # Index 1 = qdrant_results

    # LLM Reranking (optional)
    if rerank and fused:
        fused = llm_rerank(
            query=query,
            candidates=fused,
            top_n=rerank_top_n,
            model=rerank_model,
            openai_api_key=openai_api_key,
        )

    return fused[:top_k]

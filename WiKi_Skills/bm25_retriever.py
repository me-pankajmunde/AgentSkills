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
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional, Any


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
    """Build or rebuild the BM25 index."""
    wiki_dir = None
    chunk_size = 800
    overlap_size = 100
    
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]; i += 2
        elif args[i] == '--chunk-size' and i + 1 < len(args):
            chunk_size = int(args[i + 1]); i += 2
        elif args[i] == '--overlap' and i + 1 < len(args):
            overlap_size = int(args[i + 1]); i += 2
        else:
            i += 1
    
    wd = find_wiki_dir(wiki_dir)
    print(f"📇 Indexing wiki at {wd}...")
    
    idx = index_wiki(str(wd), max_chunk_size=chunk_size, overlap=overlap_size)
    
    # Save index
    index_path = wd / '_bm25_index.json'
    idx.save(str(index_path))
    
    print(f"   Chunks indexed: {idx.N}")
    print(f"   Unique terms: {len(idx.df)}")
    print(f"   Avg chunk length: {idx.avgdl:.0f} tokens")
    print(f"   Index saved: {index_path}")
    
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
        else:
            query_parts.append(args[i]); i += 1
    
    if not query_parts:
        print("Usage: bm25_retriever.py search 'query' [--top-k N]", file=sys.stderr)
        sys.exit(1)
    
    query = ' '.join(query_parts)
    wd = find_wiki_dir(wiki_dir)
    
    # Load index
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
        else:
            query_parts.append(args[i]); i += 1
    
    if not query_parts:
        print("Usage: bm25_retriever.py retrieve 'query' [--top-k N] [--format xml|json|marp] [--brief]",
              file=sys.stderr)
        sys.exit(1)
    
    query = ' '.join(query_parts)
    wd = find_wiki_dir(wiki_dir)
    
    # Load or build index
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
            'results': results,
            'total_chunks_in_index': idx.N,
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
                print(f"  ✅ Fetched: {fp} → {doc.get('raw_copy', '')}")
            else:
                print(f"  ⚠️  Failed to fetch: {fp} — {doc.get('error', 'unknown')}")
            continue

        p = Path(fp)
        if not p.exists():
            print(f"  ⚠️  File not found: {fp}", file=sys.stderr)
            continue
        
        # Copy to raw/
        import shutil
        dest = raw_dir / p.name
        counter = 1
        while dest.exists():
            dest = raw_dir / f"{p.stem}_{counter}{p.suffix}"
            counter += 1
        shutil.copy2(str(p), str(dest))
        
        # Read and extract text
        doc = read_document(str(p))
        doc['raw_copy'] = str(dest.relative_to(wd))
        
        results.append(doc)
        
        if doc['readable']:
            print(f"  ✅ Ingested: {p.name} ({doc.get('lines', 0)} lines) → {dest.relative_to(wd)}")
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
        except Exception as e:
            print(f"   ⚠️  Index rebuild failed: {e}", file=sys.stderr)


def _ingest_url(url: str, raw_dir: Path, wiki_dir: Path) -> Dict:
    """Fetch a URL, strip HTML to plain text, and save as markdown in raw/."""
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

        # Strip HTML tags to get plain text
        text = re.sub(r'<script[^>]*>.*?</script>', '', raw_html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Convert to basic markdown
        text = re.sub(r' {2,}', '\n\n', text)

        # Generate filename from URL
        from urllib.parse import urlparse
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
    except (urllib.error.URLError, Exception) as e:
        result['readable'] = False
        result['error'] = f'Failed to fetch URL: {e}'
    return result


# ============================================================================
# MAIN
# ============================================================================

USAGE = """
wiki-agent BM25 Retriever

Usage:
    bm25_retriever.py index     [--wiki-dir PATH] [--chunk-size N] [--overlap N]
    bm25_retriever.py search    QUERY [--top-k N] [--type TYPE] [--tag TAG] [--wiki-dir PATH]
    bm25_retriever.py retrieve  QUERY [--top-k N] [--format xml|json|marp] [--brief]
                                      [--freshness-weight F] [--centrality-weight F]
                                      [--max-tokens N] [--wiki-dir PATH]
    bm25_retriever.py ingest    FILE|URL [FILE|URL...] [--wiki-dir PATH] [--no-index]
    bm25_retriever.py stats     [--wiki-dir PATH]

Commands:
    index      Build/rebuild BM25 index from all wiki markdown pages
    search     Search and display ranked results (human-readable)
    retrieve   Search and output full context for RAG pipeline (XML, JSON, or Marp)
    ingest     Copy raw files or fetch URLs into wiki, extract text, auto-rebuild index
    stats      Show index statistics and top terms

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

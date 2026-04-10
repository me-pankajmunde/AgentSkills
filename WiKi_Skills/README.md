# Wiki Agent — Hybrid RAG Pipeline (BM25 + Qdrant + RRF + LLM Reranking)

A portable LLM Wiki Agent with hybrid retrieval, inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Drop it into any project — it builds a persistent, interlinked knowledge base that compounds over time, with BM25 + Qdrant semantic search fused via Reciprocal Rank Fusion and LLM reranking.

## How It Differs from Standard RAG

Standard RAG re-derives knowledge from raw document chunks on every query. This agent takes a different approach: when a document arrives, the LLM **compiles** it into structured wiki pages (entities, concepts, source summaries) with cross-references and contradiction tracking. At query time, BM25 searches the **compiled wiki** — not raw sources. The LLM reads pre-synthesized knowledge, not scattered fragments.

```
Document → Ingest → Wiki Pages → BM25 Index → Retrieve → LLM Answer → (File back)
                         ↑                                                   │
                         └───────────── compounds ──────────────────────────┘
```

## What's In the Box

```
wiki-agent/
├── SKILL.md                          # Skill definition (for LLM agents)
├── scripts/
│   ├── wiki.py                       # Wiki management CLI (init, lint, graph)
│   └── bm25_retriever.py             # BM25 RAG pipeline (index, search, retrieve, ingest)
└── references/
    ├── schema-template.md            # SCHEMA.md template for new wikis
    └── page-templates.md             # Templates for source/entity/concept/analysis pages
```

**Zero external dependencies for BM25-only mode.** Pure Python 3. No NumPy, no NLTK, no rank_bm25.
Optional: install `farmerp-wiki[hybrid]` for Qdrant semantic search + RRF fusion + LLM reranking.

## Installation

### Claude Code
```bash
cp -r wiki-agent/ .claude/skills/wiki-agent/
```

### Cursor / Windsurf / Other Agents
Reference the SKILL.md in your agent config (`.cursorrules`, `AGENTS.md`, etc.)

## Quick Start

```bash
# 1. Initialize wiki (auto-detects project type)
python scripts/wiki.py init

# 2. Ingest a document (copies to raw/, extracts text, auto-rebuilds index)
python scripts/bm25_retriever.py ingest docs/my-design-doc.md

# 2b. Ingest from URL (fetches, strips HTML, saves as markdown)
python scripts/bm25_retriever.py ingest https://example.com/article

# 3. (LLM creates wiki pages from the ingested content)

# 4. Search
python scripts/bm25_retriever.py search "authentication flow"

# 5. Retrieve context for RAG (XML format for LLM)
python scripts/bm25_retriever.py retrieve "how does auth work" --top-k 5

# 5b. Brief mode — title + 2 sentences per chunk (~300 tokens)
python scripts/bm25_retriever.py retrieve "auth" --brief

# 5c. Generate Marp slide deck from results
python scripts/bm25_retriever.py retrieve "auth patterns" --format marp

# 5d. Force a specific search backend
python scripts/bm25_retriever.py retrieve "auth" --backend hybrid     # BM25 + Qdrant + RRF + rerank
python scripts/bm25_retriever.py retrieve "auth" --backend bm25       # pure BM25
python scripts/bm25_retriever.py retrieve "auth" --backend qdrant     # semantic only
python scripts/bm25_retriever.py retrieve "auth" --backend hybrid --no-rerank  # RRF without LLM rerank

# 6. Boost recent/authoritative content
python scripts/wiki.py graph --export   # export centrality data
python scripts/bm25_retriever.py retrieve "auth" --freshness-weight 0.1 --centrality-weight 0.1

# 7. Health check
python scripts/wiki.py lint
python scripts/bm25_retriever.py stats
```

## BM25 Retriever Commands

| Command | Description |
|---------|-------------|
| `index` | Build/rebuild BM25 index from all wiki pages |
| `search QUERY` | Ranked results (human-readable) |
| `retrieve QUERY` | Full chunk context for LLM consumption (XML, JSON, or Marp) |
| `ingest FILE\|URL` | Ingest file or URL, extract text, auto-copy images to `raw/assets/`, auto-rebuild index |
| `stats` | Index statistics, term distribution, chunk counts |

**Key flags:** `--top-k N`, `--type entity|concept|source`, `--format xml|json|marp`, `--brief`, `--freshness-weight F`, `--centrality-weight F`, `--no-index`, `--chunk-size N`, `--backend auto|bm25|hybrid|qdrant`, `--no-rerank`, `--bm25-only`

## The Pipeline

1. **Ingest** — Document arrives → copied to `raw/` → images auto-extracted to `raw/assets/` → LLM creates source summary + entity pages + concept pages → cross-references built → BM25 index auto-rebuilt
2. **Query** — User asks question → BM25 + Qdrant retrieve top-k chunks → RRF fuses ranked lists → LLM reranks top candidates → synthesizes answer with citations → valuable answers filed back as analysis pages
3. **Lint** — Health check for orphan pages, broken links, asymmetric cross-references, stale content, missing frontmatter fields

## BM25 Scoring

Okapi BM25 with `k1=1.5`, `b=0.75`. Hierarchical chunking (split on headers → paragraphs → sentences) with configurable overlap. Each chunk prefixed with `[Page Title] [Section Header]` for retrieval context.

## Optional: Hybrid Search with Qdrant + RRF + LLM Reranking

Install optional dependencies and start Qdrant for hybrid retrieval:

```bash
# 1. Install hybrid dependencies
pip install 'farmerp-wiki[hybrid]'    # installs qdrant-client + openai

# 2. Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# 3. Set OpenAI API key (for embeddings + LLM reranking)
export OPENAI_API_KEY=sk-...

# 4. Build both BM25 + Qdrant indexes
python scripts/bm25_retriever.py index

# 5. Search with hybrid pipeline
python scripts/bm25_retriever.py search "auth" --backend hybrid
```

**Retrieval pipeline (hybrid mode):**
```
Query → ┌─ BM25 (top 50) ────────── ranked list ─┐
        │                                         ├→ RRF Fusion → LLM Rerank → top-K
        └─ Qdrant semantic (top 50) ─ ranked list ─┘
```

**Configuration** (env vars or `.wiki/_config.json`):
- `QDRANT_URL` — Qdrant server URL (default: `http://localhost:6333`)
- `OPENAI_API_KEY` — Required for embeddings + LLM reranking
- `OPENAI_EMBEDDING_MODEL` — Embedding model (default: `text-embedding-3-small`)
- `RERANK_MODEL` — LLM for reranking (default: `gpt-4o-mini`)
- `RERANK_TOP_N` — Candidates sent to LLM reranker (default: 20)
- `RRF_K` — RRF constant (default: 60)

Hybrid search is fully optional. Without it, pure-Python BM25 works everywhere with zero dependencies.

## Wiki Structure

```
.wiki/
├── SCHEMA.md          # Wiki conventions
├── index.md           # Page catalog
├── log.md             # Operation log
├── overview.md        # Project synthesis
├── _bm25_index.json   # Serialized BM25 inverted index
├── sources/           # Source summaries
├── entities/          # Module, API, tool pages
├── concepts/          # Pattern, principle pages
├── analyses/          # Filed query results
└── raw/               # Immutable source copies
```

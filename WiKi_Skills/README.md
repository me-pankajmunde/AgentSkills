# Wiki Agent — RAG Pipeline with BM25 Retrieval

A portable LLM Wiki Agent with built-in BM25 search, inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Drop it into any project — it builds a persistent, interlinked knowledge base that compounds over time, with BM25 retrieval for answering queries.

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

**Zero external dependencies.** Pure Python 3. No NumPy, no NLTK, no rank_bm25.
Optional: install [qmd](https://github.com/tobi/qmd) for hybrid BM25 + vector + LLM re-ranking search.

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
python scripts/bm25_retriever.py retrieve "auth" --backend qmd       # hybrid search
python scripts/bm25_retriever.py retrieve "auth" --backend bm25      # pure BM25
python scripts/bm25_retriever.py retrieve "auth" --qmd-mode vsearch  # semantic only

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
| `ingest FILE\|URL` | Ingest file or URL, extract text, auto-rebuild index (+ qmd sync) |
| `stats` | Index statistics, term distribution, chunk counts |

**Key flags:** `--top-k N`, `--type entity|concept|source`, `--format xml|json|marp`, `--brief`, `--freshness-weight F`, `--centrality-weight F`, `--no-index`, `--chunk-size N`, `--backend auto|bm25|qmd`, `--qmd-mode search|vsearch|query`

## The Pipeline

1. **Ingest** — Document arrives → copied to `raw/` → LLM creates source summary + entity pages + concept pages → cross-references built → BM25 index auto-rebuilt
2. **Query** — User asks question → BM25 retrieves top-k chunks (with optional freshness/centrality boosting) → LLM synthesizes answer with citations → valuable answers filed back as analysis pages
3. **Lint** — Health check for orphan pages, broken links, asymmetric cross-references, stale content, missing frontmatter fields

## BM25 Scoring

Okapi BM25 with `k1=1.5`, `b=0.75`. Hierarchical chunking (split on headers → paragraphs → sentences) with configurable overlap. Each chunk prefixed with `[Page Title] [Section Header]` for retrieval context.

## Optional: Hybrid Search with qmd

Install [qmd](https://github.com/tobi/qmd) (`npm install -g @tobilu/qmd`) for
hybrid BM25 + vector + LLM re-ranking search. Once installed, all `search` and
`retrieve` commands use it automatically (`--backend auto`).

```bash
# One-time setup
npm install -g @tobilu/qmd
cd .wiki && qmd collection add . --name wiki
qmd context add qmd://wiki "Project wiki"
qmd embed  # ~2GB models downloaded on first run
```

qmd is fully optional. Without it, pure-Python BM25 works everywhere.

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

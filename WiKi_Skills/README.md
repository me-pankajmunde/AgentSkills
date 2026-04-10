# Wiki Agent — Document-Centric Knowledge Base with Hybrid RAG

A portable LLM Wiki Agent with hybrid retrieval, inspired by [Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f). Point it at any directory — it builds a persistent, interlinked knowledge base from your documents that compounds over time, with BM25 + Qdrant semantic search fused via Reciprocal Rank Fusion and LLM reranking.

## How It Differs from Standard RAG

Standard RAG re-derives knowledge from raw document chunks on every query. This agent takes a different approach: when a document arrives, it's **ingested** into `raw/`, **compiled** by the LLM into structured wiki pages (entities, concepts, source summaries) with cross-references and contradiction tracking, then **indexed** for retrieval. At query time, chunks are retrieved from the **compiled wiki**, fused, reranked, and synthesized into answers with citations.

```
Document → Ingest → Compile → Wiki Pages → Index → Retrieve → Answer → (File back)
                                   ↑                                        │
                                   └──────────── compounds ────────────────┘
```

## What's In the Box

```
WiKi_Skills/
├── SKILL.md                # Skill definition (for LLM agents)
├── wiki.py                 # Wiki management CLI (init, compile, lint, graph)
├── bm25_retriever.py       # RAG pipeline (index, search, retrieve, answer, ingest)
├── wiki_compiler.py        # LLM compilation engine (source/entity/concept pages)
├── qdrant_store.py         # Qdrant vector store wrapper (optional hybrid search)
├── fusion.py               # RRF fusion + LLM reranking (optional)
├── schema-template.md      # SCHEMA.md template for new wikis
└── page-templates.md       # Templates for source/entity/concept/analysis pages
```

**Zero external dependencies for BM25-only mode.** Pure Python 3. No NumPy, no NLTK.
Optional: install `wiki-agent[hybrid]` for Qdrant semantic search + RRF + LLM reranking.

## Installation

```bash
pip install wiki-agent                    # BM25-only (zero deps)
pip install 'wiki-agent[hybrid]'          # + Qdrant + OpenAI
```

### For LLM Agent Integration
```bash
# Claude Code
cp -r WiKi_Skills/ .claude/skills/wiki-agent/

# Cursor / Windsurf / Other Agents
# Reference the SKILL.md in your agent config
```

## Quick Start

```bash
# 1. Initialize wiki at a specific directory
python wiki.py init --wiki-dir /path/to/my-wiki --name "Research Notes"

# 2. Ingest a document (auto-indexes + auto-compiles)
python bm25_retriever.py ingest paper.pdf --wiki-dir /path/to/my-wiki

# 3. Ask a question (synthesized answer with citations)
python bm25_retriever.py answer "what is the main finding?" --wiki-dir /path/to/my-wiki

# 4. Search for chunks
python bm25_retriever.py search "authentication flow" --wiki-dir /path/to/my-wiki

# 5. Retrieve RAG context for LLM consumption
python bm25_retriever.py retrieve "how does auth work" --top-k 5 --wiki-dir /path/to/my-wiki

# 6. File a valuable answer back into the wiki
python bm25_retriever.py answer "compare X and Y" --file-answer --wiki-dir /path/to/my-wiki

# 7. Compile un-compiled raw files
python wiki.py compile --all --wiki-dir /path/to/my-wiki

# 8. Health check
python wiki.py lint --wiki-dir /path/to/my-wiki
```

## Key Commands

### `wiki.py`

| Command | Description |
|---------|-------------|
| `init --wiki-dir PATH [--name N]` | Initialize wiki |
| `status [--wiki-dir PATH]` | Page counts, compile state |
| `compile [--all] [FILE...] [--wiki-dir PATH]` | LLM-compile raw files into wiki pages |
| `recompile FILE [--wiki-dir PATH]` | Force re-compile a file |
| `lint [--wiki-dir PATH]` | Orphan/link/frontmatter checks |
| `graph [--wiki-dir PATH]` | Link graph and hub pages |

### `bm25_retriever.py`

| Command | Description |
|---------|-------------|
| `index [--wiki-dir PATH]` | Build BM25 index (+ Qdrant) |
| `search QUERY` | Ranked results (human-readable) |
| `retrieve QUERY [--format xml\|json\|marp]` | Full chunk context for LLM |
| `answer QUERY [--raw] [--file-answer]` | Synthesized answer with citations |
| `ingest FILE\|URL [--no-compile]` | Ingest + index + compile |
| `stats` | Index statistics |

## The Pipeline

1. **Ingest** → Raw copy + BM25/Qdrant index
2. **Compile** → LLM creates source pages + entity pages + concept pages + cross-refs
3. **Answer** → Hybrid retrieval + RRF fusion + LLM reranking + synthesis with citations
4. **Enrich** → File answers back as analysis pages → re-index → compounds knowledge
5. **Lint** → Orphan pages, broken links, contradictions, stale content

## Hybrid Search Setup

```bash
pip install 'wiki-agent[hybrid]'
docker run -p 6333:6333 qdrant/qdrant
export OPENAI_API_KEY=sk-...
python bm25_retriever.py index --wiki-dir /path/to/wiki
```

Falls back to pure BM25 if not configured.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | Required for compilation, answer synthesis, hybrid search |
| `WIKI_COMPILE_MODEL` | `gpt-4o-mini` | LLM model for compilation |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant server URL |

## Reference

- [SKILL.md](WiKi_Skills/SKILL.md) — Full skill definition with all operations
- [page-templates.md](WiKi_Skills/page-templates.md) — Page type templates
- [schema-template.md](WiKi_Skills/schema-template.md) — SCHEMA.md template

---
name: wiki-agent
description: >
  An LLM-powered Wiki Agent with hybrid retrieval (BM25 + Qdrant semantic search +
  Reciprocal Rank Fusion + LLM reranking) that incrementally builds and maintains a
  persistent, interlinked knowledge base from user-provided documents. When a user adds
  a document, it gets ingested, LLM-compiled into structured wiki pages (source summaries,
  entity pages, concept pages with cross-references), and indexed for both keyword and
  semantic retrieval. Queries are answered by fusing BM25 lexical matches with Qdrant
  vector similarity via RRF, then reranking the top results using an LLM — not by
  re-reading raw sources each time. Falls back to pure BM25 (zero dependencies) when
  hybrid search is not configured. Use this skill whenever the user says things like
  "build a wiki", "create a knowledge base", "ingest this document", "add this to the wiki",
  "update the wiki", "lint the wiki", "query the wiki", "what does the wiki say about X",
  "search wiki for Y", "process this file", "wiki status", "initialize wiki", "setup wiki",
  "RAG pipeline", "index the wiki", "retrieve from wiki", "compile the wiki",
  "answer from wiki", or any reference to maintaining a structured, persistent knowledge
  base from documents. Also trigger when the user drops source files (articles, papers,
  transcripts, notes, reports, PDFs, markdown) and expects them to be synthesized into an
  evolving knowledge base — not just summarized once. This skill turns any collection of
  documents into a living wiki with hybrid retrieval that compounds knowledge over time.
  Even if the user just says "process this", "what do we know about X", "organize my docs",
  or "find info about Y", consider this skill.
---

# Wiki Agent — Document-Centric Knowledge Base with Hybrid RAG

A portable, document-centric LLM Wiki Agent inspired by
[Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## How It Differs from Standard RAG

Standard RAG uploads documents, retrieves relevant chunks at query time, and generates
answers from scratch each time. The LLM rediscovers knowledge on every question.

This agent takes a fundamentally different approach:

1. **Ingest** — When a document arrives, it's copied to `raw/` and indexed.

2. **Compile** — The LLM reads the raw text and **compiles** it into structured wiki
   pages: source summaries, entity pages, concept pages. Cross-references are built,
   contradictions flagged, synthesis done. This happens automatically after ingest.

3. **Index** — All wiki pages are chunked and indexed using **BM25** (Okapi BM25,
   pure Python, zero dependencies) and optionally **Qdrant** (semantic vectors).

4. **Answer** — At query time, chunks are retrieved from the **compiled wiki**,
   fused via RRF, reranked by LLM, and synthesized into a cited answer.

5. **Enrich** — Valuable answers are filed back as analysis pages, compounding knowledge.

The wiki is a **persistent, compounding artifact**. Every ingest enriches it.
Every query can be filed back. The index is rebuilt after each change.

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     USER / LLM AGENT                       │
│                                                            │
│   "Add this paper"    "What do we know about X?"           │
│         │                       │                          │
│         ▼                       ▼                          │
│   ┌──────────┐          ┌──────────────┐                   │
│   │  INGEST  │          │    ANSWER    │                   │
│   │          │          │              │                    │
│   │ 1. Copy  │          │ 1. Retrieve  │                   │
│   │ 2. Index │──rebuild─│    chunks    │                   │
│   │ 3. LLM   │  index   │ 2. Synthesize│                   │
│   │  compile │──────────│    answer    │                   │
│   │ 4. Pages │          │ 3. File back │                   │
│   └──────────┘          └──────────────┘                   │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                    .wiki/ DIRECTORY                         │
│                                                            │
│  SCHEMA.md   index.md   log.md   overview.md               │
│                                                            │
│  sources/          entities/        concepts/               │
│  ├── 2026-04-10-   ├── openai.md    ├── caching.md         │
│  │   paper-x.md    ├── user-api.md  ├── event-driven.md    │
│  └── 2026-04-11-   └── auth.md      └── testing.md         │
│      meeting.md                                             │
│                                                            │
│  analyses/         raw/             _bm25_index.json        │
│  ├── what-is-      ├── paper.pdf                            │
│  │   caching.md    └── meeting.txt                          │
│  └── comparison.md                                          │
└────────────────────────────────────────────────────────────┘
```

Three layers:

**Raw sources** (`raw/`) — Immutable copies of original documents. Never modified.

**The wiki** (`sources/`, `entities/`, `concepts/`, `analyses/`) — LLM-compiled
structured markdown. Summaries, entity pages, concept pages, cross-references.
Created automatically during compilation.

**The index** (`_bm25_index.json` + optional Qdrant) — Inverted index over all wiki
pages. Rebuilt after every ingest.

---

## Quick Start

### 1. Initialize

```bash
python wiki.py init --wiki-dir /path/to/my-wiki --name "Research Notes"
```

Creates `.wiki/` structure at the specified path. If `--wiki-dir` is omitted,
uses the current working directory.

### 2. Ingest a Document

```bash
python bm25_retriever.py ingest /path/to/document.md --wiki-dir /path/to/my-wiki
```

This automatically:
1. Copies the file to `raw/`
2. Rebuilds the BM25 index (+ Qdrant if configured)
3. LLM-compiles the document into source, entity, and concept pages
4. Weaves cross-references and updates log/index

Use `--no-compile` to skip LLM compilation (just index).
Use `--no-index` to skip indexing (just copy to raw/).

### 3. Ask a Question

```bash
# Synthesized answer with citations
python bm25_retriever.py answer "what is the main finding?" --wiki-dir /path/to/my-wiki

# File the answer as an analysis page
python bm25_retriever.py answer "compare X and Y" --file-answer --wiki-dir /path/to/my-wiki

# Raw chunks without synthesis
python bm25_retriever.py answer "auth flow" --raw --wiki-dir /path/to/my-wiki
```

### 4. Search & Retrieve

```bash
# Human-readable search results
python bm25_retriever.py search "authentication flow" --wiki-dir /path/to/my-wiki

# Full RAG context for LLM consumption
python bm25_retriever.py retrieve "how does auth work" --top-k 5 --wiki-dir /path/to/my-wiki
```

### 5. Compile & Lint

```bash
# Compile any un-compiled raw files
python wiki.py compile --all --wiki-dir /path/to/my-wiki

# Force re-compile a specific file
python wiki.py recompile raw/document.md --wiki-dir /path/to/my-wiki

# Health check
python wiki.py lint --wiki-dir /path/to/my-wiki
```

---

## Scripts Reference

### `wiki.py` — Wiki Management

| Command | Description |
|---------|-------------|
| `init --wiki-dir PATH [--name N] [--description D]` | Initialize wiki at specified path |
| `status [--wiki-dir PATH]` | Show page counts, compile state |
| `search QUERY [--wiki-dir PATH]` | Simple keyword search across pages |
| `compile [--all] [FILE...] [--wiki-dir PATH]` | Compile un-compiled raw files into wiki pages |
| `recompile FILE [--wiki-dir PATH]` | Force re-compile a specific raw file |
| `lint [--wiki-dir PATH]` | Check for orphans, broken links, missing frontmatter |
| `graph [--wiki-dir PATH] [--export]` | Show link graph summary and hub pages |

### `bm25_retriever.py` — RAG Pipeline

| Command | Description |
|---------|-------------|
| `index [--wiki-dir PATH] [--bm25-only]` | Build/rebuild BM25 index (+ Qdrant if configured) |
| `search QUERY [--top-k N] [--backend B] [--no-rerank]` | Search and display ranked results |
| `retrieve QUERY [--top-k N] [--format xml\|json\|marp]` | Retrieve context for LLM |
| `answer QUERY [--top-k N] [--raw] [--file-answer]` | Retrieve + LLM-synthesize answer with citations |
| `ingest FILE\|URL [...] [--no-index] [--no-compile]` | Ingest files/URLs, auto-index + auto-compile |
| `stats [--wiki-dir PATH]` | Show index statistics |

**BM25 Parameters** (tunable in code):
- `k1=1.5` — Term frequency saturation
- `b=0.75` — Length normalization

---

## Operations

### INIT — Set Up the Wiki

**When**: First use, or user says "init wiki" / "setup wiki".

**Steps**:

1. Run `python wiki.py init --wiki-dir /path/to/wiki --name "My Wiki"`
2. This creates `.wiki/` with `sources/`, `entities/`, `concepts/`, `analyses/`, `raw/`
3. Generates minimal `_discovery.json` (name, description, dates)
4. Creates `log.md` and `index.md`
5. Build initial BM25 index: `python bm25_retriever.py index --wiki-dir /path/to/wiki`

---

### INGEST — Add a Document to the Wiki

**When**: User adds a source and says "ingest", "process", "add this to the wiki".

```bash
python bm25_retriever.py ingest /path/to/document.md --wiki-dir /path/to/wiki
```

**What happens automatically:**

1. **Raw copy** — File copied to `raw/`, text extracted
2. **BM25 index rebuilt** — Chunks created, inverted index updated
3. **Qdrant synced** — If hybrid configured, vectors upserted
4. **LLM compilation** — If OPENAI_API_KEY set:
   - Source page created in `sources/`
   - Entity pages created/merged in `entities/`
   - Concept pages created/merged in `concepts/`
   - Cross-references woven between all pages
   - `log.md` and `index.md` updated

Use `--no-compile` to skip LLM compilation (just index).
Use `--no-index` to skip indexing (just copy to raw/).

**Verification:**
```bash
python bm25_retriever.py stats --wiki-dir /path/to/wiki
python bm25_retriever.py search "<key term>" --wiki-dir /path/to/wiki
```

---

### COMPILE — Process Raw Files into Wiki Pages

**When**: User says "compile the wiki" or raw files haven't been compiled yet.

```bash
# Compile all un-compiled raw files
python wiki.py compile --all --wiki-dir /path/to/wiki

# Compile specific files
python wiki.py compile raw/paper.md raw/notes.txt --wiki-dir /path/to/wiki

# Force re-compile (overwrites existing compiled pages)
python wiki.py recompile raw/paper.md --wiki-dir /path/to/wiki
```

The compiler reads each raw file and uses the LLM to:
1. Generate a **source page** with summary, key claims, entities, concepts
2. Create/merge **entity pages** — new entities get new pages, existing ones get updated
   with new info and contradiction flags (⚠️)
3. Create/merge **concept pages** — same pattern
4. **Weave cross-references** — ensure bidirectional links
5. **Update meta** — log.md, index.md, _discovery.json

Compilation state is tracked in `_compile_state.json` to avoid re-processing.

**Requires:** `OPENAI_API_KEY` environment variable (uses gpt-4o-mini by default,
configurable via `WIKI_COMPILE_MODEL`).

---

### ANSWER — Ask Questions with Synthesized Answers

**When**: User asks a question about wiki contents.

```bash
python bm25_retriever.py answer "what do we know about caching?" --wiki-dir /path/to/wiki
```

**Pipeline:**
1. Retrieve relevant chunks (BM25 or hybrid)
2. LLM synthesizes answer with inline `[source](path)` citations
3. Optionally file the answer as an analysis page (`--file-answer`)

**Flags:**
- `--raw` — Skip synthesis, show raw chunks
- `--file-answer` — Save answer as `analyses/` page and re-index
- `--top-k N` — Number of chunks to retrieve (default 5)
- `--backend auto|bm25|hybrid|qdrant` — Search backend
- `--no-rerank` — Skip LLM reranking

**Filing answers back** is how queries compound — your explorations become part
of the knowledge base.

---

### QUERY — Search and Retrieve Chunks

**When**: User wants raw chunks or search results (not synthesized answers).

```bash
# Human-readable ranked results
python bm25_retriever.py search "authentication flow" --wiki-dir /path/to/wiki

# Full RAG context for LLM consumption (XML format)
python bm25_retriever.py retrieve "how does auth work" --top-k 5 --wiki-dir /path/to/wiki

# JSON format for programmatic use
python bm25_retriever.py retrieve "auth" --format json --top-k 5

# Brief mode — title + 2 sentences per chunk
python bm25_retriever.py retrieve "auth" --brief
```

**Search flags:**
- `--type entity|concept|source|analysis` — filter by page type
- `--top-k N` — number of chunks to retrieve (default 5)
- `--brief` — return title + section + 2-sentence preview (~300 tokens)
- `--freshness-weight F` — boost recently-updated pages (0.0-1.0)
- `--centrality-weight F` — boost hub pages (requires `graph --export` first)
- `--format xml|json|marp` — output format
- `--backend auto|bm25|hybrid|qdrant` — search backend
- `--no-rerank` — skip LLM reranking

---

### LINT — Health Check the Wiki

**When**: User says "lint the wiki" or periodically after many ingests.

```bash
python wiki.py lint --wiki-dir /path/to/wiki
python bm25_retriever.py stats --wiki-dir /path/to/wiki
```

**Checks**: orphan pages, broken links, missing frontmatter, isolated pages,
asymmetric links, stale content, missing confidence fields, index health.

---

## Hybrid Search (Qdrant + RRF + LLM Reranking)

When configured, retrieval uses a multi-stage pipeline:

```
Query → BM25 (top 50) + Qdrant semantic (top 50)
      → Reciprocal Rank Fusion (merges both lists)
      → LLM Reranking (scores top 20 for relevance)
      → Final top-K results
```

**Setup:**
```bash
pip install 'farmerp-wiki[hybrid]'           # Install qdrant-client + openai
docker run -p 6333:6333 qdrant/qdrant        # Start Qdrant
export OPENAI_API_KEY=sk-...                 # For embeddings + reranking + compilation
python bm25_retriever.py index --wiki-dir /path/to/wiki
```

Falls back to pure BM25 (zero dependencies) if not configured.

---

## Chunking Strategy

Pages are chunked hierarchically:
1. Split on `##` headers
2. Split long sections on paragraphs
3. Split long paragraphs on sentences
4. Add configurable overlap (default 100 words)

Each chunk is prefixed with `[Page Title] [Section Header]` for context.

**Default parameters:**
- `max_chunk_size = 800` words per chunk
- `overlap = 100` words between consecutive chunks

---

## BM25 Scoring

Okapi BM25 probabilistic ranking:

```
score(q, d) = Σ IDF(qi) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
```

- **IDF** — Terms in fewer documents weighted higher
- **TF saturation** — Controlled by `k1`. Diminishing returns for repeat terms.
- **Length normalization** — Controlled by `b`. Longer documents don't dominate.

Pure-Python inverted index serialized as JSON. No external dependencies.

---

## Page Format

Every wiki page uses YAML frontmatter:

```yaml
---
title: Page Title
type: source | entity | concept | analysis | overview
created: YYYY-MM-DD
updated: YYYY-MM-DD
tags: [tag1, tag2]
sources: [source-slug-1]
related: [page-slug-1]
confidence: high | medium | speculative
---
```

See `page-templates.md` for full templates.

---

## Integration Patterns

### As a Standalone CLI
```bash
python bm25_retriever.py answer "my question" --wiki-dir /path/to/wiki
```

### In an LLM Agent Loop
```python
import subprocess
result = subprocess.run(
    ['python', 'bm25_retriever.py', 'retrieve', query,
     '--top-k', '5', '--format', 'xml', '--wiki-dir', wiki_path],
    capture_output=True, text=True
)
context = result.stdout
```

### In Claude Code / Codex / Cursor
The SKILL.md tells the agent how to use the scripts. Just say:
- "Initialize a wiki at /path/to/wiki"
- "Ingest docs/architecture.md into the wiki"
- "What does the wiki say about caching?"
- "Compile the wiki"
- "Lint the wiki"

---

## Image Handling

Images are **automatically handled** during ingestion:

1. **Local files**: Detects `![alt](path)` and `<img src="...">` references,
   copies images to `raw/assets/`, rewrites paths in the raw copy.
2. **URLs**: Extracts `<img>` tags from fetched HTML, downloads to `raw/assets/`.
3. Supported formats: `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.bmp`, `.ico`, `.tiff`

---

## Git Integration

The wiki is just markdown files — it works naturally with git.

```bash
cd .wiki && git init
echo "_bm25_index.json" >> .gitignore
echo "_discovery.json" >> .gitignore
echo "_centrality.json" >> .gitignore
echo "_compile_state.json" >> .gitignore
echo "raw/" >> .gitignore
git add -A && git commit -m "Wiki initialized"
```

---

## Tips

1. **Ingest one at a time** for important sources. Check the compiled pages.
2. **Index auto-rebuilds and compilation auto-runs after ingest.**
3. **File valuable answers back** with `--file-answer`. Analyses compound knowledge.
4. **Lint periodically** after 5-10 ingests.
5. **The wiki is just markdown.** Browse in Obsidian, VS Code, or any editor.
6. **Use `--brief` retrieval** for token-efficient broad exploration.
7. **Set OPENAI_API_KEY** to enable compilation, answer synthesis, and hybrid search.
8. **Ingest URLs directly.** Pass a URL to `ingest` to fetch web articles.
9. **Use `--format marp`** to generate slide decks from wiki knowledge.

---

## Reference Files

- `page-templates.md` — Templates for each page type
- `schema-template.md` — Default SCHEMA.md template

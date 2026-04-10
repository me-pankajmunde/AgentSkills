---
name: wiki-agent
description: >
  An LLM-powered Wiki Agent with built-in BM25 RAG pipeline that incrementally builds
  and maintains a persistent, interlinked knowledge base for any project. When a user
  adds a document, it gets ingested into a structured wiki and indexed for BM25 retrieval.
  Queries are answered by retrieving relevant chunks from the compiled wiki — not by
  re-reading raw sources each time. Use this skill whenever the user says things like
  "build a wiki", "create a knowledge base", "ingest this document", "add this to the wiki",
  "update the wiki", "lint the wiki", "query the wiki", "what does the wiki say about X",
  "search wiki for Y", "process this file", "wiki status", "initialize wiki", "setup wiki",
  "RAG pipeline", "index the wiki", "retrieve from wiki", or any reference to maintaining
  a structured, persistent knowledge base from project files. Also trigger when the user
  drops source files (articles, papers, transcripts, notes, code docs, PDFs, markdown)
  and expects them to be synthesized into an evolving knowledge base — not just summarized
  once. This skill turns any project directory into a living wiki with BM25 retrieval that
  compounds knowledge over time. Even if the user just says "process this", "what do we
  know about X", "organize my docs", or "find info about Y in my project", consider this skill.
---

# Wiki Agent — RAG Pipeline with BM25 Retrieval

A portable, project-agnostic LLM Wiki Agent inspired by
[Karpathy's LLM Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f).

## How It Differs from Standard RAG

Standard RAG uploads documents, retrieves relevant chunks at query time, and generates
answers from scratch each time. The LLM rediscovers knowledge on every question.

This agent takes a fundamentally different approach:

1. **Ingest** — When a document arrives, the LLM reads it, extracts key information,
   and **compiles** it into structured wiki pages (entities, concepts, source summaries).
   Cross-references are built, contradictions flagged, synthesis done.

2. **Index** — All wiki pages are chunked and indexed using **BM25** (Okapi BM25,
   pure Python, zero dependencies). The index is persisted as JSON.

3. **Retrieve** — At query time, the BM25 retriever returns ranked, relevant chunks
   from the **compiled wiki** — not from raw sources. The LLM reads pre-synthesized,
   cross-referenced knowledge, not scattered fragments.

The wiki is a **persistent, compounding artifact**. Every ingest enriches it.
Every query can be filed back as a new analysis page. The BM25 index is rebuilt
after each ingest so retrieval always reflects the latest state.

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
│   │  INGEST  │          │    QUERY     │                   │
│   │          │          │              │                    │
│   │ 1. Read  │          │ 1. BM25      │                   │
│   │ 2. Wiki  │──rebuild─│    search    │                   │
│   │    pages │  index   │ 2. Read top  │                   │
│   │ 3. Index │──────────│    chunks    │                   │
│   └──────────┘          │ 3. Synthesize│                   │
│                         │ 4. File back │                   │
│                         └──────────────┘                   │
│                                                            │
├────────────────────────────────────────────────────────────┤
│                    .wiki/ DIRECTORY                         │
│                                                            │
│  SCHEMA.md   index.md   log.md   overview.md               │
│                                                            │
│  sources/          entities/        concepts/               │
│  ├── 2026-04-10-   ├── my-module.md ├── caching.md         │
│  │   paper-x.md    ├── user-api.md  ├── event-driven.md    │
│  └── 2026-04-11-   └── auth.md      └── testing.md         │
│      meeting.md                                             │
│                                                            │
│  analyses/         raw/             _bm25_index.json        │
│  ├── comparison-   ├── paper.pdf                            │
│  │   x-vs-y.md     └── meeting.txt                          │
│  └── gap-report.md                                          │
└────────────────────────────────────────────────────────────┘
```

Three layers:

**Raw sources** (`raw/`) — Immutable copies of original documents. Never modified.

**The wiki** (`sources/`, `entities/`, `concepts/`, `analyses/`) — LLM-generated
structured markdown. Summaries, entity pages, concept pages, cross-references.
The LLM owns this layer entirely. It creates pages, updates them, maintains links.

**The BM25 index** (`_bm25_index.json`) — Serialized inverted index over all wiki
pages. Rebuilt after every ingest. The retriever uses this to find relevant chunks.

---

## Quick Start

### 1. Initialize

```bash
python scripts/wiki.py init [--root /path/to/project]
```

Auto-detects project root, creates `.wiki/` structure, discovers tech stack.

### 2. Ingest a Document

```bash
# Step 1: Copy raw file and extract text
python scripts/bm25_retriever.py ingest /path/to/document.md

# Step 2: LLM agent reads the extracted text and creates wiki pages
#          (source summary, entity pages, concept pages, cross-references)

# Step 3: Rebuild the BM25 index
python scripts/bm25_retriever.py index
```

### 3. Query via BM25 Retrieval

```bash
# Human-readable search results
python scripts/bm25_retriever.py search "authentication flow"

# Full RAG context for LLM consumption (XML format)
python scripts/bm25_retriever.py retrieve "how does auth work" --top-k 5

# JSON format for programmatic use
python scripts/bm25_retriever.py retrieve "auth" --format json --top-k 5
```

### 4. Lint

```bash
python scripts/wiki.py lint
```

---

## Scripts Reference

### `scripts/wiki.py` — Wiki Management

| Command | Description |
|---------|-------------|
| `init [--root PATH]` | Initialize wiki with project auto-detection |
| `status` | Show wiki page counts and health |
| `search QUERY` | Simple keyword search across pages |
| `lint` | Check for orphans, broken links, missing frontmatter |
| `graph` | Show link graph summary and hub pages |

### `scripts/bm25_retriever.py` — RAG Pipeline

| Command | Description |
|---------|-------------|
| `index [--chunk-size N] [--overlap N]` | Build/rebuild BM25 index from wiki pages |
| `search QUERY [--top-k N] [--type TYPE] [--backend auto\|bm25\|qmd]` | Search and display ranked results |
| `retrieve QUERY [--top-k N] [--format xml\|json\|marp] [--brief] [--backend auto\|bm25\|qmd]` | Retrieve context for LLM |
| `ingest FILE\|URL [FILE\|URL...] [--no-index]` | Ingest files or URLs, auto-rebuild index (+ qmd sync) |
| `stats` | Show index statistics, top terms, chunk distribution |

**BM25 Parameters** (tunable in code):
- `k1=1.5` — Term frequency saturation. Higher → raw TF counts more.
- `b=0.75` — Length normalization. 0 = none, 1 = full normalization.

---

## Operations

### INIT — Set Up the Wiki

**When**: First use, or user says "init wiki" / "setup wiki".

**Steps**:

1. Run `python scripts/wiki.py init` to create directory structure.
2. Read the generated `_discovery.json` for project metadata.
3. Read `references/schema-template.md` to understand the template.
4. Generate `SCHEMA.md` tailored to the discovered project.
5. Generate `overview.md` from README + directory scan.
6. Build initial BM25 index: `python scripts/bm25_retriever.py index`
7. Report what was discovered and created.

---

### INGEST — Add a Document to the Wiki

**When**: User adds a source and says "ingest", "process", "add this to the wiki".

**The ingest pipeline has 4 stages:**

#### Stage 1: Raw Ingestion
```bash
python scripts/bm25_retriever.py ingest /path/to/document.md
```
This copies the file to `raw/`, extracts text, and outputs a JSON summary with
content preview. For PDFs/DOCX, use the appropriate reading skill first to
extract text into a `.md` file, then ingest that.

#### Stage 2: Wiki Page Creation (LLM does this)

The LLM agent reads the extracted text and creates/updates wiki pages:

1. **Source summary page** in `sources/YYYY-MM-DD-<slug>.md`:
   - YAML frontmatter with title, source_type, date_ingested, tags, key_entities
   - Summary, key claims, notable data points, questions raised

2. **Entity pages** in `entities/<slug>.md`:
   - For each key entity (module, API, person, tool, etc.)
   - Check existing pages first — merge, don't overwrite
   - Cite the source page

3. **Concept pages** in `concepts/<slug>.md`:
   - For abstract patterns, principles, themes
   - Cross-reference with entity pages

4. **Update `index.md`** — add new pages with one-line summaries

5. **Update `overview.md`** — if the source materially changes the big picture

6. **Append to `log.md`**:
   ```
   ## [YYYY-MM-DD] ingest | <Source Title>
   - Source: raw/<filename>
   - Pages created: <list>
   - Pages updated: <list>
   - Key takeaway: <one sentence>
   ```

#### Stage 3: Rebuild BM25 Index
The BM25 index auto-rebuilds after every ingest by default. If you used
`--no-index` during batch ingestion, rebuild manually:
```bash
python scripts/bm25_retriever.py index
```
This re-chunks all wiki pages and rebuilds the inverted index.

#### Stage 4: Verification
```bash
python scripts/bm25_retriever.py stats
python scripts/bm25_retriever.py search "<key term from new source>"
```
Verify the new content is findable. Report chunk counts and confirm retrieval.

---

### QUERY — Answer Questions Using BM25 Retrieval

**When**: User asks a question about wiki contents.

**The query pipeline:**

#### Step 1: Retrieve Relevant Chunks
```bash
python scripts/bm25_retriever.py retrieve "user's question" --top-k 5 --format xml
```

This returns structured XML context:
```xml
<wiki_context query="how does authentication work">
<chunk source="entities/auth-module.md" section="Overview" score="8.42" type="entity">
[Auth Module] [Overview]

The authentication module handles JWT-based auth flow...
</chunk>
<chunk source="concepts/token-refresh.md" section="Key Aspects" score="6.15" type="concept">
...
</chunk>
</wiki_context>
```

#### Step 2: LLM Synthesizes Answer

The LLM reads the retrieved chunks and synthesizes an answer with citations
back to specific wiki pages.

#### Step 3: File Valuable Answers Back

If the answer is a valuable synthesis (comparison, analysis, connection), offer
to save it as a new page in `analyses/`. Then rebuild the index. This is how
**queries compound** — your explorations become part of the knowledge base.

**Query flags:**
- `--type entity|concept|source|analysis` — filter by page type
- `--top-k N` — number of chunks to retrieve (default 5)
- `--max-tokens N` — approximate token budget for context (default 4000)
- `--brief` — return title + section + 2 sentences per chunk (~300 tokens)
- `--freshness-weight F` — boost recently-updated pages (0.0-1.0)
- `--centrality-weight F` — boost hub pages (requires `graph --export` first)
- `--format marp` — output as Marp slide deck
- `--backend auto|bm25|qmd` — search backend (`auto` uses qmd if installed)
- `--qmd-mode search|vsearch|query` — qmd search mode (default: `query` = hybrid)

---

### LINT — Health Check the Wiki

**When**: User says "lint the wiki" or periodically after many ingests.

```bash
python scripts/wiki.py lint
python scripts/bm25_retriever.py stats
```

**Checks**:
1. Orphan pages (no inbound links)
2. Broken links
3. Missing YAML frontmatter
4. Isolated pages (no outbound links)
5. Asymmetric links (A→B exists but B→A doesn't)
6. Stale content (pages not updated in 30+ days)
7. Missing `confidence` field in frontmatter
8. Index health (chunk distribution, term coverage)

---

## Chunking Strategy

The BM25 retriever chunks wiki pages using a hierarchical strategy:

1. **Split on `##` headers** — each section becomes a potential chunk
2. **Split long sections on paragraph breaks** — if a section exceeds `max_chunk_size`
3. **Split long paragraphs on sentences** — last resort for very long text blocks
4. **Add overlap** — configurable overlap between consecutive chunks (default 100 words)

Every chunk is prefixed with `[Page Title] [Section Header]` for context when
the chunk is retrieved independently.

**Default parameters:**
- `max_chunk_size = 800` words per chunk
- `overlap = 100` words between consecutive chunks

Tuning: increase chunk size for conceptual/narrative documents, decrease for
dense technical reference where precision matters.

---

## BM25 Scoring

The retriever uses Okapi BM25, the standard probabilistic ranking function:

```
score(q, d) = Σ IDF(qi) × (tf × (k1 + 1)) / (tf + k1 × (1 - b + b × |d|/avgdl))
```

- **IDF** — Terms that appear in fewer documents are weighted higher (more discriminative)
- **TF saturation** — Controlled by `k1`. A term appearing 10x isn't 10x as important.
- **Length normalization** — Controlled by `b`. Longer documents don't unfairly dominate.

The index is a pure-Python inverted index serialized as JSON. No external
dependencies (no NumPy, no NLTK, no rank_bm25). Works anywhere Python runs.

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
sources: [source-slug-1, source-slug-2]
related: [page-slug-1, page-slug-2]
---
```

See `references/page-templates.md` for full templates.

---

## Project Auto-Detection

The agent adapts based on what it discovers at init:

| Signal | Adaptation |
|--------|-----------|
| `package.json` | Entity pages for npm packages, API modules |
| `pyproject.toml` | Entity pages for Python packages, classes |
| `Cargo.toml` | Entity pages for Rust crates |
| `go.mod` | Entity pages for Go packages |
| `docs/` directory | Auto-ingest existing documentation |
| No code signals | General knowledge base mode |

---

## Integration Patterns

### As a Standalone CLI
```bash
# One-shot RAG query
python scripts/bm25_retriever.py retrieve "my question" --top-k 5
```

### In an LLM Agent Loop
```python
# 1. User asks a question
query = user_input

# 2. Retrieve context
import subprocess
result = subprocess.run(
    ['python', 'scripts/bm25_retriever.py', 'retrieve', query,
     '--top-k', '5', '--format', 'xml'],
    capture_output=True, text=True
)
context = result.stdout

# 3. Inject into LLM prompt
prompt = f"""Answer the following question using the wiki context below.
Cite specific pages in your answer.

{context}

Question: {query}
"""
```

### In Claude Code / Codex / Cursor
The SKILL.md tells the agent how to use the scripts. Just say:
- "Initialize a wiki for this project"
- "Ingest docs/architecture.md"
- "What does the wiki say about caching?"
- "Lint the wiki"

---

## Tips

1. **Ingest one at a time** for important sources. Stay involved — check the
   summaries, guide the emphasis. Batch-ingest for less critical sources.

2. **Index auto-rebuilds after ingest.** The `ingest` command now auto-rebuilds
   the BM25 index by default. Use `--no-index` to skip if batch-ingesting
   many files (then run `index` once at the end).

3. **File valuable query answers back.** The analyses/ directory is where your
   explorations compound into permanent knowledge.

4. **Lint periodically.** Orphan pages, missing cross-references, stale content,
   and asymmetric links accumulate. Run `lint` after every 5-10 ingests.

5. **The wiki is just markdown files.** You can browse it in Obsidian, VS Code,
   or any editor. `git init` the `.wiki/` directory for version history.

6. **Use `--brief` for token-efficient retrieval.** When exploring broadly, use
   `retrieve --brief` to get title + section + 2-sentence preview per chunk
   (~300 tokens total). Request full pages only for relevant results.

7. **Boost recent and authoritative content.** Use `--freshness-weight 0.1` to
   surface recently-updated pages. Use `--centrality-weight 0.1` (after running
   `graph --export`) to boost well-connected hub pages.

8. **Ingest URLs directly.** Pass a URL to `ingest` to fetch web articles, strip
   HTML, and save as markdown in `raw/`. Useful for quickly capturing web sources.

9. **Generate slide decks.** Use `retrieve --format marp` to create a Marp-compatible
   presentation from wiki knowledge. View with Obsidian's Marp plugin or any
   Marp renderer.

10. **Upgrade to hybrid search with qmd.** Install [qmd](https://github.com/tobi/qmd)
    (`npm install -g @tobilu/qmd`) for BM25 + vector + LLM re-ranking search.
    Once installed, all `search` and `retrieve` commands automatically use it
    (`--backend auto`). Force BM25 with `--backend bm25`. Use `--qmd-mode vsearch`
    for pure semantic search, or `--qmd-mode query` (default) for full hybrid.

---

## qmd Integration (Optional)

The retriever optionally delegates to [qmd](https://github.com/tobi/qmd) —
a local hybrid search engine combining BM25 + vector embeddings + LLM re-ranking.

**When to use:** When the wiki exceeds ~100 pages, or when keyword BM25 misses
results where the user's query uses different words than the wiki content.

**Setup:**
```bash
# 1. Install qmd (requires Node.js >= 22)
npm install -g @tobilu/qmd

# 2. Create a collection for the wiki
cd .wiki && qmd collection add . --name wiki

# 3. Add context to improve search quality
qmd context add qmd://wiki "Project wiki — entities, concepts, sources, analyses"

# 4. Generate vector embeddings (~2GB of models downloaded on first run)
qmd embed
```

**How it works:**
- `--backend auto` (default): uses qmd if installed, falls back to BM25
- `--backend bm25`: forces pure-Python BM25 (zero dependencies)
- `--backend qmd`: forces qmd (errors if not installed)
- `--qmd-mode query`: hybrid BM25 + vector + re-ranking (best quality, default)
- `--qmd-mode search`: BM25 only via qmd
- `--qmd-mode vsearch`: vector semantic search only

**Auto-sync:** After every `ingest`, if qmd is installed, the retriever
automatically runs `qmd update && qmd embed` to keep the hybrid index current.

**Zero-dependency contract preserved:** qmd is entirely optional. If not
installed, everything works exactly as before with pure-Python BM25.

---

## Image Handling

Images referenced in source documents should be stored in `raw/assets/`.

**Workflow for documents with images:**
1. Copy images to `raw/assets/` (manually or via Obsidian Web Clipper + download)
2. Reference in wiki pages with relative paths: `![Description](../raw/assets/image.png)`
3. LLMs can't read markdown with inline images in one pass — read the text first,
   then view referenced images separately for additional context
4. In Obsidian: Settings → Files and links → set "Attachment folder path" to
   `raw/assets/`. Bind "Download attachments" to a hotkey (e.g. Ctrl+Shift+D)

**Obsidian Web Clipper** is useful for converting web articles to markdown and
downloading their images locally so URLs don't break.

---

## Git Integration

The wiki is just a directory of markdown files — it works naturally with git.

**Recommended setup:**
```bash
cd .wiki && git init
echo "_bm25_index.json" >> .gitignore   # large, auto-rebuilt
echo "_discovery.json" >> .gitignore     # project-specific
echo "_centrality.json" >> .gitignore    # auto-rebuilt
echo "raw/" >> .gitignore                # optional: skip raw copies
git add -A && git commit -m "Wiki initialized"
```

**Benefits:**
- Full version history of every wiki page
- Branch for experimental synthesis, merge when confident
- Diff wiki changes after each ingest to review LLM edits
- Team collaboration: each person forks, PRs back to shared wiki
- Revert bad LLM edits with `git checkout`

**Obsidian's graph view** is the best way to visualize the wiki's shape — what's
connected, which pages are hubs, which are orphans.

---

## Reference Files

Read these before initializing a new wiki:

- `references/schema-template.md` — Default SCHEMA.md template
- `references/page-templates.md` — Templates for each page type (source, entity, concept, analysis)

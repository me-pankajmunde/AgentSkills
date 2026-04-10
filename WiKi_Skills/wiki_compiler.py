#!/usr/bin/env python3
"""
Wiki Compiler — LLM-powered compilation of raw documents into structured wiki pages.

Reads ingested raw documents and uses an LLM (OpenAI) to:
1. Generate source pages with structured summaries
2. Extract and create/merge entity pages
3. Extract and create/merge concept pages
4. Weave cross-references between all generated pages
5. Update log.md and index.md

Requires: openai>=1.0.0, OPENAI_API_KEY environment variable.
Gracefully degrades (skips compilation) if unavailable.
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_MODEL = 'gpt-4o-mini'
COMPILE_STATE_FILE = '_compile_state.json'


def _get_model() -> str:
    return os.environ.get('WIKI_COMPILE_MODEL', DEFAULT_MODEL)


def _get_client():
    """Return an OpenAI client or None if unavailable."""
    try:
        from openai import OpenAI
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            return None
        return OpenAI(api_key=api_key)
    except ImportError:
        return None


def is_compiler_available() -> bool:
    """Check if the compiler can run (openai installed + API key set)."""
    return _get_client() is not None


# ============================================================================
# COMPILE STATE TRACKING
# ============================================================================

def _load_compile_state(wiki_dir: Path) -> dict:
    path = wiki_dir / COMPILE_STATE_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_compile_state(wiki_dir: Path, state: dict):
    path = wiki_dir / COMPILE_STATE_FILE
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding='utf-8')


# ============================================================================
# SLUG HELPERS
# ============================================================================

def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug."""
    slug = text.lower().strip()
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', '-', slug)
    return slug[:80].strip('-')


def _today() -> str:
    return datetime.now().strftime('%Y-%m-%d')


# ============================================================================
# LLM CALLS
# ============================================================================

def _llm_json(client, prompt: str, system: str = '') -> Optional[dict]:
    """Call the LLM and parse JSON from the response."""
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})

    try:
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content.strip()
        return json.loads(text)
    except Exception as e:
        print(f"  ⚠️  LLM call failed: {e}")
        return None


# ============================================================================
# B1a: SOURCE PAGE GENERATION
# ============================================================================

SOURCE_EXTRACTION_PROMPT = """You are a wiki compiler. Read the document below and extract structured information.

Return a JSON object with these fields:
- "title": string — a clear, descriptive title for this source
- "source_type": string — one of: article, paper, transcript, code, notes, data, report, manual
- "summary": string — 2-3 paragraph summary capturing the key information
- "key_claims": list of strings — each claim should stand alone with enough context
- "notable_data": list of strings — specific numbers, dates, quotes worth preserving
- "entities": list of objects — each with "name" (string), "type" (one of: person, organization, tool, service, component, module, api, location, other), "role" (string — brief description of role in this document)
- "concepts": list of objects — each with "name" (string), "description" (string — how this concept is discussed in the document)
- "tags": list of strings — topical tags for categorization
- "questions": list of strings — open questions this document raises but doesn't answer
- "confidence": string — "high" if the source is authoritative and clear, "medium" if somewhat ambiguous, "speculative" if uncertain

IMPORTANT:
- Extract ALL significant entities and concepts, not just the top few.
- Each entity name should be a proper noun or specific term (not generic words).
- Each concept should be a distinct idea or domain term.
- Claims should be factual assertions, not opinions, unless clearly marked.

DOCUMENT:
{document}"""


def compile_source(wiki_dir: Path, raw_file: str, raw_text: str,
                   metadata: dict = None) -> Optional[dict]:
    """
    Compile a raw document into a source page + extract entities/concepts.

    Returns dict with keys: source_path, entities, concepts, extraction
    or None if compilation fails.
    """
    client = _get_client()
    if not client:
        print("  ⚠️  No OpenAI API key — skipping compilation.")
        return None

    metadata = metadata or {}
    today = _today()

    # Truncate very long documents to stay within context limits
    max_chars = 60000
    doc_text = raw_text[:max_chars]
    if len(raw_text) > max_chars:
        doc_text += f"\n\n[... truncated, {len(raw_text) - max_chars} chars omitted ...]"

    print(f"  🧠 Compiling source: {raw_file} ({len(raw_text)} chars)...")

    extraction = _llm_json(
        client,
        SOURCE_EXTRACTION_PROMPT.format(document=doc_text),
        system="You are a precise knowledge extraction system. Always return valid JSON."
    )
    if not extraction:
        return None

    title = extraction.get('title', Path(raw_file).stem)
    slug = _slugify(title)
    source_filename = f"{today}-{slug}.md"
    source_path = wiki_dir / 'sources' / source_filename

    # Avoid overwriting — append counter if needed
    counter = 1
    while source_path.exists():
        source_filename = f"{today}-{slug}-{counter}.md"
        source_path = wiki_dir / 'sources' / source_filename
        counter += 1

    entities = extraction.get('entities', [])
    concepts = extraction.get('concepts', [])

    # Build entity links
    entity_lines = []
    for e in entities:
        e_slug = _slugify(e['name'])
        role = e.get('role', '')
        entity_lines.append(f"- [{e['name']}](../entities/{e_slug}.md): {role}")

    # Build concept links
    concept_lines = []
    for c in concepts:
        c_slug = _slugify(c['name'])
        desc = c.get('description', '')
        concept_lines.append(f"- [{c['name']}](../concepts/{c_slug}.md): {desc}")

    # Build key claims
    claims = extraction.get('key_claims', [])
    claims_lines = '\n'.join(f"{i+1}. {c}" for i, c in enumerate(claims))

    # Build notable data
    notable = extraction.get('notable_data', [])
    notable_lines = '\n'.join(f"- {d}" for d in notable)

    # Build questions
    questions = extraction.get('questions', [])
    questions_lines = '\n'.join(f"- {q}" for q in questions)

    tags = extraction.get('tags', [])
    entity_names = [e['name'] for e in entities]
    concept_names = [c['name'] for c in concepts]
    confidence = extraction.get('confidence', 'medium')
    summary = extraction.get('summary', '')

    source_content = f"""---
title: "{title}"
type: source
source_type: {extraction.get('source_type', 'notes')}
source_path: "raw/{Path(raw_file).name}"
date_ingested: {today}
created: {today}
updated: {today}
tags: {json.dumps(tags)}
key_entities: {json.dumps(entity_names)}
key_concepts: {json.dumps(concept_names)}
confidence: {confidence}
status: active
---

# {title}

## Summary
{summary}

## Key Claims
{claims_lines if claims_lines else '_None extracted._'}

## Notable Data / Quotes
{notable_lines if notable_lines else '_None extracted._'}

## Entities Mentioned
{chr(10).join(entity_lines) if entity_lines else '_None identified._'}

## Concepts Discussed
{chr(10).join(concept_lines) if concept_lines else '_None identified._'}

## Questions Raised
{questions_lines if questions_lines else '_None identified._'}
"""
    source_path.write_text(source_content, encoding='utf-8')
    rel_source = str(source_path.relative_to(wiki_dir))
    print(f"  ✅ Source page: {rel_source}")

    return {
        'source_path': rel_source,
        'source_filename': source_filename,
        'entities': entities,
        'concepts': concepts,
        'extraction': extraction,
    }


# ============================================================================
# B1b: ENTITY PAGE CREATION / MERGING
# ============================================================================

ENTITY_MERGE_PROMPT = """You are a wiki editor. An entity page already exists for "{entity_name}".
A new source has been ingested that mentions this entity.

EXISTING PAGE CONTENT:
{existing_content}

NEW INFORMATION FROM SOURCE "{source_title}":
- Role in source: {role}
- Source path: {source_path}

Merge the new information into the existing page. Return a JSON object with:
- "overview": string — updated overview (2-3 sentences, integrate new info)
- "new_details": list of strings — new bullet points to add to Key Details
- "new_relationships": list of strings — new relationship bullet points (if any)
- "history_entry": string — a single history line for today's date
- "contradictions": list of strings — any contradictions between existing content and new info (empty list if none)
- "open_questions": list of strings — new questions raised (empty list if none)

IMPORTANT: If the new information contradicts existing content, DO NOT silently overwrite.
Instead, note the contradiction explicitly. Prefix contradictions with ⚠️."""


def compile_entities(wiki_dir: Path, entities: list, source_path: str,
                     source_title: str) -> List[str]:
    """
    Create or merge entity pages for each extracted entity.
    Returns list of created/updated entity page paths (relative to wiki_dir).
    """
    client = _get_client()
    if not client:
        return []

    today = _today()
    pages = []

    for entity in entities:
        name = entity.get('name', '').strip()
        if not name:
            continue

        slug = _slugify(name)
        if not slug:
            continue

        entity_path = wiki_dir / 'entities' / f"{slug}.md"
        entity_type = entity.get('type', 'other')
        role = entity.get('role', '')

        if entity_path.exists():
            # MERGE with existing
            existing = entity_path.read_text(encoding='utf-8')
            merge_data = _llm_json(
                client,
                ENTITY_MERGE_PROMPT.format(
                    entity_name=name,
                    existing_content=existing[:8000],
                    source_title=source_title,
                    role=role,
                    source_path=source_path,
                ),
                system="You are a careful wiki editor. Return valid JSON."
            )

            if merge_data:
                _merge_entity_page(entity_path, existing, merge_data,
                                   source_path, today)
                print(f"  🔄 Entity merged: entities/{slug}.md")
            else:
                # Fallback: just append source to existing
                _append_source_to_entity(entity_path, existing, source_path,
                                         role, today)
                print(f"  📎 Entity updated (fallback): entities/{slug}.md")
        else:
            # CREATE new entity page
            content = f"""---
title: "{name}"
type: entity
entity_type: {entity_type}
created: {today}
updated: {today}
tags: []
sources: ["{source_path}"]
related: []
confidence: medium
status: active
open_questions: []
contradictions: []
---

# {name}

## Overview
{name} is referenced in [{source_title}](../{source_path}). {role}

## Key Details
- {role or 'Mentioned in source'} — from [{source_title}](../{source_path})

## Relationships
_To be expanded as more sources are ingested._

## History / Changes
- {today}: Created from [{source_title}](../{source_path})

## Open Questions
- What is the full scope and context of {name}?
"""
            entity_path.write_text(content, encoding='utf-8')
            print(f"  ✨ Entity created: entities/{slug}.md")

        pages.append(f"entities/{slug}.md")

    return pages


def _merge_entity_page(path: Path, existing: str, merge_data: dict,
                       source_path: str, today: str):
    """Apply LLM merge data to an existing entity page."""
    lines = existing.split('\n')

    # Update the 'updated' date in frontmatter
    for i, line in enumerate(lines):
        if line.startswith('updated:'):
            lines[i] = f'updated: {today}'
            break

    # Add source to sources list in frontmatter
    for i, line in enumerate(lines):
        if line.startswith('sources:'):
            if source_path not in line:
                lines[i] = line.rstrip(']') + f', "{source_path}"]'
            break

    # Add contradictions to frontmatter if any
    contradictions = merge_data.get('contradictions', [])
    if contradictions:
        for i, line in enumerate(lines):
            if line.startswith('contradictions:'):
                existing_contras = line.split('[', 1)[-1].rstrip(']').strip()
                new_contras = ', '.join(f'"{c}"' for c in contradictions)
                if existing_contras:
                    lines[i] = f'contradictions: [{existing_contras}, {new_contras}]'
                else:
                    lines[i] = f'contradictions: [{new_contras}]'
                break

    content = '\n'.join(lines)

    # Add new details to Key Details section
    new_details = merge_data.get('new_details', [])
    if new_details:
        details_block = '\n'.join(f"- {d} — from [{source_path}](../{source_path})"
                                  for d in new_details)
        content = _insert_after_section(content, '## Key Details', details_block)

    # Add new relationships
    new_rels = merge_data.get('new_relationships', [])
    if new_rels:
        rels_block = '\n'.join(f"- {r}" for r in new_rels)
        content = _insert_after_section(content, '## Relationships', rels_block)

    # Add history entry
    history = merge_data.get('history_entry', '')
    if history:
        content = _insert_after_section(content, '## History / Changes',
                                        f"- {today}: {history}")

    # Add contradiction notices in body
    if contradictions:
        notice = '\n'.join(f"- ⚠️ {c}" for c in contradictions)
        content = _insert_after_section(content, '## Key Details',
                                        f"\n**Contradictions detected:**\n{notice}")

    path.write_text(content, encoding='utf-8')


def _append_source_to_entity(path: Path, existing: str, source_path: str,
                             role: str, today: str):
    """Fallback: append source reference without LLM merge."""
    lines = existing.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('updated:'):
            lines[i] = f'updated: {today}'
            break
    for i, line in enumerate(lines):
        if line.startswith('sources:'):
            if source_path not in line:
                lines[i] = line.rstrip(']') + f', "{source_path}"]'
            break
    content = '\n'.join(lines)
    content = _insert_after_section(
        content, '## Key Details',
        f"- {role or 'Referenced'} — from [{source_path}](../{source_path})"
    )
    content = _insert_after_section(
        content, '## History / Changes',
        f"- {today}: Updated from [{source_path}](../{source_path})"
    )
    path.write_text(content, encoding='utf-8')


# ============================================================================
# B1c: CONCEPT PAGE CREATION / MERGING
# ============================================================================

CONCEPT_MERGE_PROMPT = """You are a wiki editor. A concept page already exists for "{concept_name}".
A new source discusses this concept.

EXISTING PAGE CONTENT:
{existing_content}

NEW INFORMATION FROM SOURCE "{source_title}":
- How discussed: {description}
- Source path: {source_path}

Merge the new information. Return JSON with:
- "definition_update": string — updated definition if the new source adds clarity (or empty string to keep existing)
- "new_aspects": list of strings — new aspects or dimensions to add
- "new_source_entry": string — a bullet point for the Sources section
- "contradictions": list of strings — contradictions between existing and new (empty list if none)
- "open_questions": list of strings — new questions raised (empty list if none)

If the new source contradicts the existing definition, DO NOT overwrite. Flag with ⚠️."""


def compile_concepts(wiki_dir: Path, concepts: list, source_path: str,
                     source_title: str) -> List[str]:
    """
    Create or merge concept pages for each extracted concept.
    Returns list of created/updated concept page paths (relative to wiki_dir).
    """
    client = _get_client()
    if not client:
        return []

    today = _today()
    pages = []

    for concept in concepts:
        name = concept.get('name', '').strip()
        if not name:
            continue

        slug = _slugify(name)
        if not slug:
            continue

        concept_path = wiki_dir / 'concepts' / f"{slug}.md"
        description = concept.get('description', '')

        if concept_path.exists():
            existing = concept_path.read_text(encoding='utf-8')
            merge_data = _llm_json(
                client,
                CONCEPT_MERGE_PROMPT.format(
                    concept_name=name,
                    existing_content=existing[:8000],
                    source_title=source_title,
                    description=description,
                    source_path=source_path,
                ),
                system="You are a careful wiki editor. Return valid JSON."
            )

            if merge_data:
                _merge_concept_page(concept_path, existing, merge_data,
                                    source_path, source_title, today)
                print(f"  🔄 Concept merged: concepts/{slug}.md")
            else:
                _append_source_to_concept(concept_path, existing, source_path,
                                          source_title, description, today)
                print(f"  📎 Concept updated (fallback): concepts/{slug}.md")
        else:
            content = f"""---
title: "{name}"
type: concept
created: {today}
updated: {today}
tags: []
sources: ["{source_path}"]
related: []
confidence: medium
status: active
open_questions: []
contradictions: []
---

# {name}

## Definition
{description or f'{name} — concept extracted from [{source_title}](../{source_path}).'}

## Key Aspects
_To be expanded as more sources discuss this concept._

## How It Appears
- Discussed in [{source_title}](../{source_path}): {description}

## Related Concepts
_To be linked as more concepts are discovered._

## Sources
- [{source_title}](../{source_path}): {description}
"""
            concept_path.write_text(content, encoding='utf-8')
            print(f"  ✨ Concept created: concepts/{slug}.md")

        pages.append(f"concepts/{slug}.md")

    return pages


def _merge_concept_page(path: Path, existing: str, merge_data: dict,
                        source_path: str, source_title: str, today: str):
    """Apply LLM merge data to an existing concept page."""
    lines = existing.split('\n')

    for i, line in enumerate(lines):
        if line.startswith('updated:'):
            lines[i] = f'updated: {today}'
            break
    for i, line in enumerate(lines):
        if line.startswith('sources:'):
            if source_path not in line:
                lines[i] = line.rstrip(']') + f', "{source_path}"]'
            break

    contradictions = merge_data.get('contradictions', [])
    if contradictions:
        for i, line in enumerate(lines):
            if line.startswith('contradictions:'):
                existing_c = line.split('[', 1)[-1].rstrip(']').strip()
                new_c = ', '.join(f'"{c}"' for c in contradictions)
                if existing_c:
                    lines[i] = f'contradictions: [{existing_c}, {new_c}]'
                else:
                    lines[i] = f'contradictions: [{new_c}]'
                break

    content = '\n'.join(lines)

    # Update definition if provided
    def_update = merge_data.get('definition_update', '')
    if def_update:
        content = _replace_section_content(content, '## Definition', def_update)

    # Add new aspects
    new_aspects = merge_data.get('new_aspects', [])
    if new_aspects:
        aspects_block = '\n'.join(f"- {a}" for a in new_aspects)
        content = _insert_after_section(content, '## Key Aspects', aspects_block)

    # Add source entry
    src_entry = merge_data.get('new_source_entry', '')
    if src_entry:
        content = _insert_after_section(content, '## Sources', src_entry)
    else:
        content = _insert_after_section(
            content, '## Sources',
            f"- [{source_title}](../{source_path})"
        )

    if contradictions:
        notice = '\n'.join(f"- ⚠️ {c}" for c in contradictions)
        content = _insert_after_section(content, '## Definition',
                                        f"\n**Contradictions detected:**\n{notice}")

    path.write_text(content, encoding='utf-8')


def _append_source_to_concept(path: Path, existing: str, source_path: str,
                              source_title: str, description: str, today: str):
    """Fallback: append source reference without LLM merge."""
    lines = existing.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('updated:'):
            lines[i] = f'updated: {today}'
            break
    for i, line in enumerate(lines):
        if line.startswith('sources:'):
            if source_path not in line:
                lines[i] = line.rstrip(']') + f', "{source_path}"]'
            break
    content = '\n'.join(lines)
    content = _insert_after_section(
        content, '## Sources',
        f"- [{source_title}](../{source_path}): {description}"
    )
    path.write_text(content, encoding='utf-8')


# ============================================================================
# B1d: CROSS-REFERENCE WEAVING
# ============================================================================

def weave_crossrefs(wiki_dir: Path):
    """
    Scan all pages and ensure bidirectional cross-references.
    If source A mentions entity B, entity B should link back to source A.
    """
    # Build a map of all pages and their outgoing links
    all_pages = {}  # rel_path -> content
    outgoing = {}   # rel_path -> set of target rel_paths

    for subdir in ['sources', 'entities', 'concepts', 'analyses']:
        d = wiki_dir / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != '.md':
                continue
            try:
                content = f.read_text(encoding='utf-8')
                rel = str(f.relative_to(wiki_dir))
                all_pages[rel] = content
                links = set()
                for m in re.finditer(r'\[([^\]]+)\]\(\.\.?/([^)]+)\)', content):
                    target = m.group(2)
                    # Normalize: ../entities/foo.md -> entities/foo.md
                    target = target.lstrip('.').lstrip('/')
                    links.add(target)
                outgoing[rel] = links
            except OSError:
                continue

    # Find asymmetric links and add back-references
    fixes = 0
    for page, targets in outgoing.items():
        for target in targets:
            if target not in all_pages:
                continue
            # Check if target links back to page
            target_links = outgoing.get(target, set())
            if page not in target_links:
                # Add a back-reference to the target page
                _add_backref(wiki_dir, target, page, all_pages)
                fixes += 1

    if fixes:
        print(f"  🔗 Wove {fixes} cross-references.")


def _add_backref(wiki_dir: Path, target_rel: str, source_rel: str,
                 all_pages: dict):
    """Add a back-reference link from target page to source page."""
    target_path = wiki_dir / target_rel
    if not target_path.exists():
        return

    content = all_pages.get(target_rel, '')
    source_name = Path(source_rel).stem.replace('-', ' ').title()

    # Determine which section to add the back-reference to
    if target_rel.startswith('entities/'):
        section = '## Relationships'
    elif target_rel.startswith('concepts/'):
        section = '## Sources'
    elif target_rel.startswith('sources/'):
        section = '## Entities Mentioned'
    else:
        section = '## Related'

    # Calculate relative path from target to source
    depth = target_rel.count('/')
    prefix = '../' * depth
    link = f"- [{source_name}]({prefix}{source_rel})"

    # Check if already referenced
    if source_rel in content:
        return

    updated = _insert_after_section(content, section, link)
    if updated != content:
        target_path.write_text(updated, encoding='utf-8')
        all_pages[target_rel] = updated


# ============================================================================
# B1e: META UPDATES (log.md, index.md)
# ============================================================================

def update_log(wiki_dir: Path, action: str, details: str):
    """Append an entry to log.md."""
    log_path = wiki_dir / 'log.md'
    today = _today()

    if not log_path.exists():
        return

    content = log_path.read_text(encoding='utf-8')

    # Update the 'updated' date in frontmatter
    content = re.sub(r'^(updated:\s*)\S+', f'\\g<1>{today}', content, count=1,
                     flags=re.MULTILINE)

    entry = f"\n## [{today}] {action}\n{details}\n"
    content += entry
    log_path.write_text(content, encoding='utf-8')


def rebuild_index(wiki_dir: Path):
    """Rebuild index.md from the current wiki contents."""
    today = _today()
    index_path = wiki_dir / 'index.md'

    # Read wiki name from discovery
    name = 'Wiki'
    discovery_path = wiki_dir / '_discovery.json'
    if discovery_path.exists():
        try:
            info = json.loads(discovery_path.read_text(encoding='utf-8'))
            name = info.get('name', info.get('project_name', 'Wiki'))
        except (json.JSONDecodeError, OSError):
            pass

    sections = {'sources': [], 'entities': [], 'concepts': [], 'analyses': []}
    for subdir in sections:
        d = wiki_dir / subdir
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix != '.md':
                continue
            # Extract title from frontmatter
            try:
                content = f.read_text(encoding='utf-8')
                m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', content,
                              re.MULTILINE)
                title = m.group(1) if m else f.stem.replace('-', ' ').title()
            except OSError:
                title = f.stem.replace('-', ' ').title()
            rel = f"{subdir}/{f.name}"
            sections[subdir].append(f"- [{title}]({rel})")

    def _section_content(items, empty_msg):
        return '\n'.join(items) if items else f'_{empty_msg}_'

    content = f"""---
title: Wiki Index
type: meta
created: {today}
updated: {today}
---

# {name} — Wiki Index

## Overview
- [Overview](overview.md)

## Sources
{_section_content(sections['sources'], 'No sources ingested yet.')}

## Entities
{_section_content(sections['entities'], 'No entity pages yet.')}

## Concepts
{_section_content(sections['concepts'], 'No concept pages yet.')}

## Analyses
{_section_content(sections['analyses'], 'No analyses yet.')}
"""
    index_path.write_text(content, encoding='utf-8')

    # Update discovery date
    if discovery_path.exists():
        try:
            info = json.loads(discovery_path.read_text(encoding='utf-8'))
            info['date_updated'] = today
            discovery_path.write_text(json.dumps(info, indent=2, ensure_ascii=False),
                                      encoding='utf-8')
        except (json.JSONDecodeError, OSError):
            pass


# ============================================================================
# B2: ANSWER SYNTHESIS
# ============================================================================

SYNTHESIS_PROMPT = """You are a knowledgeable assistant answering questions using ONLY the wiki excerpts provided below. Do not use outside knowledge.

WIKI EXCERPTS:
{context}

QUESTION: {query}

Instructions:
- Answer the question based ONLY on the provided excerpts.
- Cite sources inline using markdown links: [source title](path).
- If the excerpts don't contain enough information, say so explicitly.
- Be concise but thorough.
- If excerpts contain contradictions, note them."""


def synthesize_answer(query: str, chunks: list, wiki_dir: Path = None) -> Optional[str]:
    """
    Synthesize an answer from retrieved chunks using LLM.

    Args:
        query: The user's question.
        chunks: List of dicts with 'content', 'filepath', 'chunk_id' keys.
        wiki_dir: Optional wiki directory for context.

    Returns:
        Synthesized answer string, or None if LLM unavailable.
    """
    client = _get_client()
    if not client:
        return None

    # Build context from chunks
    context_parts = []
    for i, chunk in enumerate(chunks):
        filepath = chunk.get('filepath', 'unknown')
        content = chunk.get('content', '')
        context_parts.append(f"[Excerpt {i+1}] (from {filepath}):\n{content}")

    context = '\n\n---\n\n'.join(context_parts)

    # Truncate if needed
    if len(context) > 40000:
        context = context[:40000] + "\n\n[... additional excerpts truncated ...]"

    try:
        messages = [
            {'role': 'system',
             'content': 'You are a precise research assistant. Answer only from provided excerpts.'},
            {'role': 'user',
             'content': SYNTHESIS_PROMPT.format(context=context, query=query)},
        ]
        resp = client.chat.completions.create(
            model=_get_model(),
            messages=messages,
            temperature=0.1,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️  Answer synthesis failed: {e}")
        return None


# ============================================================================
# B3: CACHE ENRICHMENT — File answers as analysis pages
# ============================================================================

def file_answer(wiki_dir: Path, query: str, answer: str,
                source_paths: list = None) -> Optional[str]:
    """
    File a synthesized answer as an analysis page in analyses/.

    Returns the relative path of the created analysis page, or None.
    """
    today = _today()
    slug = _slugify(query[:60])
    filename = f"{today}-{slug}.md"
    analysis_path = wiki_dir / 'analyses' / filename

    counter = 1
    while analysis_path.exists():
        filename = f"{today}-{slug}-{counter}.md"
        analysis_path = wiki_dir / 'analyses' / filename
        counter += 1

    sources = source_paths or []
    sources_json = json.dumps(sources)

    # Build evidence section from source paths
    evidence_lines = []
    for sp in sources:
        name = Path(sp).stem.replace('-', ' ').title()
        evidence_lines.append(f"- [{name}](../{sp})")

    content = f"""---
title: "{query[:100]}"
type: analysis
analysis_type: query-answer
created: {today}
updated: {today}
tags: []
sources: {sources_json}
related: []
prompt: "{query}"
confidence: medium
status: active
open_questions: []
contradictions: []
---

# {query[:100]}

## Question
{query}

## Findings
{answer}

## Evidence
{chr(10).join(evidence_lines) if evidence_lines else '_Based on wiki retrieval._'}

## Gaps
_Review this analysis for completeness._

## Suggested Follow-ups
- Verify findings against additional sources.
"""
    analysis_path.write_text(content, encoding='utf-8')
    rel = str(analysis_path.relative_to(wiki_dir))
    print(f"  📝 Analysis filed: {rel}")

    # Update log
    update_log(wiki_dir, f"analysis | Filed answer",
               f"- Query: {query[:80]}\n- Page: {rel}")

    return rel


# ============================================================================
# ORCHESTRATION — Full compilation pipeline
# ============================================================================

def compile_file(wiki_dir: Path, raw_file: str, force: bool = False) -> dict:
    """
    Run the full compilation pipeline for a single raw file.

    1. Read raw text
    2. compile_source → source page + entity/concept lists
    3. compile_entities → entity pages
    4. compile_concepts → concept pages
    5. weave_crossrefs
    6. update log + index
    7. update compile state

    Returns dict: { raw_file, compiled, pages_created, error }
    """
    state = _load_compile_state(wiki_dir)

    # Check if already compiled
    if not force and raw_file in state and state[raw_file].get('compiled'):
        return {
            'raw_file': raw_file,
            'compiled': True,
            'pages_created': state[raw_file].get('pages_created', []),
            'skipped': True,
        }

    # Read the raw file
    raw_path = wiki_dir / raw_file
    if not raw_path.exists():
        # Try under raw/ prefix
        raw_path = wiki_dir / 'raw' / Path(raw_file).name
    if not raw_path.exists():
        return {'raw_file': raw_file, 'compiled': False, 'error': 'File not found'}

    try:
        raw_text = raw_path.read_text(encoding='utf-8')
    except (OSError, UnicodeDecodeError) as e:
        return {'raw_file': raw_file, 'compiled': False, 'error': str(e)}

    pages_created = []

    # Step 1: Compile source page
    result = compile_source(wiki_dir, raw_file, raw_text)
    if not result:
        return {'raw_file': raw_file, 'compiled': False,
                'error': 'Source compilation failed (LLM unavailable or error)'}

    pages_created.append(result['source_path'])
    source_title = result['extraction'].get('title', Path(raw_file).stem)

    # Step 2: Compile entities
    entity_pages = compile_entities(
        wiki_dir, result['entities'], result['source_path'], source_title
    )
    pages_created.extend(entity_pages)

    # Step 3: Compile concepts
    concept_pages = compile_concepts(
        wiki_dir, result['concepts'], result['source_path'], source_title
    )
    pages_created.extend(concept_pages)

    # Step 4: Weave cross-references
    weave_crossrefs(wiki_dir)

    # Step 5: Update log and index
    update_log(wiki_dir, f"compile | Compiled {raw_file}",
               f"- Source: {result['source_path']}\n"
               f"- Entities: {len(entity_pages)} pages\n"
               f"- Concepts: {len(concept_pages)} pages\n"
               f"- Total pages: {len(pages_created)}")
    rebuild_index(wiki_dir)

    # Step 6: Update compile state
    state[raw_file] = {
        'compiled': True,
        'date': _today(),
        'pages_created': pages_created,
    }
    _save_compile_state(wiki_dir, state)

    return {
        'raw_file': raw_file,
        'compiled': True,
        'pages_created': pages_created,
    }


def compile_all_pending(wiki_dir: Path) -> list:
    """Compile all raw files that haven't been compiled yet."""
    raw_dir = wiki_dir / 'raw'
    if not raw_dir.is_dir():
        return []

    state = _load_compile_state(wiki_dir)
    results = []

    for f in sorted(raw_dir.iterdir()):
        if f.suffix not in ('.md', '.txt', '.rst'):
            continue
        if f.name.startswith('.') or f.name.startswith('_'):
            continue

        rel = f"raw/{f.name}"
        if rel in state and state[rel].get('compiled'):
            continue

        result = compile_file(wiki_dir, rel)
        results.append(result)

    return results


# ============================================================================
# SECTION MANIPULATION HELPERS
# ============================================================================

def _insert_after_section(content: str, section_header: str, text: str) -> str:
    """Insert text after a section header, before the next section or end."""
    lines = content.split('\n')
    insert_at = None

    for i, line in enumerate(lines):
        if line.strip() == section_header:
            # Find the next non-empty content line after the header
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            # Find the end of this section (next ## header or end of file)
            end = j
            while end < len(lines):
                if end > j and lines[end].startswith('## '):
                    break
                end += 1
            insert_at = end
            break

    if insert_at is not None:
        lines.insert(insert_at, text)
        return '\n'.join(lines)

    # Section not found — append at end
    return content + f"\n{section_header}\n{text}\n"


def _replace_section_content(content: str, section_header: str,
                             new_content: str) -> str:
    """Replace the content of a section (between header and next header)."""
    lines = content.split('\n')

    start = None
    end = None
    for i, line in enumerate(lines):
        if line.strip() == section_header:
            start = i + 1
            # Skip blank lines after header
            while start < len(lines) and not lines[start].strip():
                start += 1
            end = start
            while end < len(lines):
                if lines[end].startswith('## ') and end > start:
                    break
                end += 1
            break

    if start is not None and end is not None:
        lines[start:end] = [new_content, '']
        return '\n'.join(lines)

    return content

#!/usr/bin/env python3
"""
wiki-agent CLI — document-centric knowledge base management.

Usage:
    python wiki.py init --wiki-dir PATH [--name NAME] [--description DESC]
    python wiki.py status   [--wiki-dir PATH]
    python wiki.py search   QUERY [--wiki-dir PATH]
    python wiki.py compile  [--all] [--wiki-dir PATH]
    python wiki.py recompile FILE [--wiki-dir PATH]
    python wiki.py lint     [--wiki-dir PATH]
    python wiki.py orphans  [--wiki-dir PATH]
    python wiki.py graph    [--wiki-dir PATH] [--export]
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict


# --- Wiki directory management ---

WIKI_DIR = '.wiki'
SUBDIRS = ['sources', 'entities', 'concepts', 'analyses', 'raw', 'raw/assets']


def _parse_wiki_dir(args):
    """Extract --wiki-dir PATH from args, return (wiki_dir_path | None, remaining_args)."""
    rest = []
    wiki_dir = None
    i = 0
    while i < len(args):
        if args[i] == '--wiki-dir' and i + 1 < len(args):
            wiki_dir = args[i + 1]
            i += 2
        else:
            rest.append(args[i])
            i += 1
    return wiki_dir, rest


def find_wiki_dir(explicit: str = None) -> Path:
    """Resolve the .wiki directory from an explicit path or cwd fallback."""
    if explicit:
        p = Path(explicit).resolve()
        # If the user pointed directly at a .wiki dir, use it
        if p.name == WIKI_DIR and p.is_dir():
            return p
        # Otherwise treat it as the parent that contains .wiki/
        candidate = p / WIKI_DIR
        if candidate.is_dir():
            return candidate
        # For init, the dir may not exist yet — return the expected path
        return candidate

    # Fallback: walk up from cwd looking for .wiki/
    cwd = Path(os.getcwd()).resolve()
    for d in [cwd] + list(cwd.parents):
        candidate = d / WIKI_DIR
        if candidate.is_dir():
            return candidate

    # Default: .wiki under cwd (for init)
    return cwd / WIKI_DIR


def ensure_wiki(wiki_dir_arg: str = None) -> Path:
    """Find and validate the wiki directory exists."""
    wr = find_wiki_dir(wiki_dir_arg)
    if not wr.exists():
        print(f"Error: No wiki found at {wr}. Run 'wiki.py init --wiki-dir PATH' first.",
              file=sys.stderr)
        sys.exit(1)
    return wr


# --- Commands ---

def cmd_init(args):
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    name = None
    description = ''

    i = 0
    while i < len(rest):
        if rest[i] == '--name' and i + 1 < len(rest):
            name = rest[i + 1]; i += 2
        elif rest[i] == '--description' and i + 1 < len(rest):
            description = rest[i + 1]; i += 2
        else:
            i += 1

    wr = find_wiki_dir(wiki_dir_arg)
    today = datetime.now().strftime('%Y-%m-%d')

    if wr.exists():
        print(f"Wiki already exists at {wr}")
        return

    if not name:
        name = wr.parent.name or 'My Wiki'

    # Create directories
    for sub in SUBDIRS:
        (wr / sub).mkdir(parents=True, exist_ok=True)

    # Minimal discovery metadata — no project/tech detection
    info = {
        'name': name,
        'description': description,
        'date_created': today,
        'date_updated': today,
    }

    report_path = wr / '_discovery.json'
    report_path.write_text(json.dumps(info, indent=2), encoding='utf-8')

    # Create log.md
    log_content = f"""---
title: Wiki Log
type: meta
created: {today}
updated: {today}
---

# Wiki Log

## [{today}] init | Wiki Initialized
- Name: {name}
- Description: {description or '(none)'}
"""
    (wr / 'log.md').write_text(log_content, encoding='utf-8')

    # Create index.md
    index_content = f"""---
title: Wiki Index
type: meta
created: {today}
updated: {today}
---

# {name} — Wiki Index

## Overview
- [Overview](overview.md)

## Sources
_No sources ingested yet._

## Entities
_No entity pages yet._

## Concepts
_No concept pages yet._

## Analyses
_No analyses yet._
"""
    (wr / 'index.md').write_text(index_content, encoding='utf-8')

    print(f"✅ Wiki initialized at {wr}")
    print(f"   Name: {name}")
    if description:
        print(f"   Description: {description}")
    print(f"   Discovery: {report_path}")
    print(f"\n   Next: ingest documents with")
    print(f"     python bm25_retriever.py ingest FILE [--wiki-dir {wr.parent}]")


def cmd_status(args):
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    wr = ensure_wiki(wiki_dir_arg)
    counts = {}
    for sub in ['sources', 'entities', 'concepts', 'analyses']:
        d = wr / sub
        if d.is_dir():
            counts[sub] = len([f for f in d.iterdir() if f.suffix == '.md'])
        else:
            counts[sub] = 0

    total = sum(counts.values())
    log_entries = 0
    log_path = wr / 'log.md'
    if log_path.exists():
        log_entries = log_path.read_text(encoding='utf-8').count('\n## [')

    # Show compile state if available
    compile_state_path = wr / '_compile_state.json'
    compiled = 0
    pending = 0
    if compile_state_path.exists():
        try:
            state = json.loads(compile_state_path.read_text(encoding='utf-8'))
            compiled = sum(1 for v in state.values() if v.get('compiled'))
            pending = sum(1 for v in state.values() if not v.get('compiled'))
        except (json.JSONDecodeError, OSError):
            pass

    print(f"📊 Wiki Status ({wr})")
    print(f"   Sources:  {counts['sources']}")
    print(f"   Entities: {counts['entities']}")
    print(f"   Concepts: {counts['concepts']}")
    print(f"   Analyses: {counts['analyses']}")
    print(f"   Total pages: {total}")
    print(f"   Log entries: {log_entries}")
    if compiled or pending:
        print(f"   Compiled:  {compiled} raw files")
        print(f"   Pending:   {pending} raw files")


def cmd_search(args):
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    if not rest:
        print("Usage: wiki.py search QUERY [--wiki-dir PATH]", file=sys.stderr)
        sys.exit(1)
    query = ' '.join(rest).lower()
    wr = ensure_wiki(wiki_dir_arg)

    results = []
    for sub in ['sources', 'entities', 'concepts', 'analyses']:
        d = wr / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != '.md':
                continue
            try:
                content = f.read_text(encoding='utf-8').lower()
                if query in content:
                    # Extract title from frontmatter
                    title = f.stem
                    m = re.search(r'^title:\s*["\']?(.+?)["\']?\s*$', 
                                  f.read_text(encoding='utf-8'), re.MULTILINE)
                    if m:
                        title = m.group(1)
                    # Count occurrences
                    count = content.count(query)
                    results.append((count, sub, title, str(f.relative_to(wr))))
            except:
                pass

    # Also search overview and index
    for fname in ['overview.md', 'index.md']:
        fp = wr / fname
        if fp.exists():
            try:
                content = fp.read_text(encoding='utf-8').lower()
                if query in content:
                    results.append((content.count(query), 'root', fname, fname))
            except:
                pass

    results.sort(key=lambda x: -x[0])

    if not results:
        print(f"No results for '{query}'")
        return

    print(f"🔍 Search results for '{query}' ({len(results)} matches):\n")
    for count, cat, title, path in results[:20]:
        print(f"  [{cat}] {title} ({count} hits) — {path}")


def cmd_lint(args):
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    wr = ensure_wiki(wiki_dir_arg)

    issues = []

    # Collect all pages and their links
    all_pages = {}  # path -> content
    all_links = defaultdict(set)  # path -> set of linked paths
    inbound = defaultdict(set)    # path -> set of pages linking to it

    for sub in ['sources', 'entities', 'concepts', 'analyses']:
        d = wr / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != '.md':
                continue
            try:
                content = f.read_text(encoding='utf-8')
                rel = str(f.relative_to(wr))
                all_pages[rel] = content

                # Extract markdown links
                for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
                    link_target = m.group(2)
                    if link_target.startswith('http'):
                        continue
                    # Resolve relative path
                    target = (f.parent / link_target).resolve()
                    if target.is_relative_to(wr):
                        trel = str(target.relative_to(wr))
                        all_links[rel].add(trel)
                        inbound[trel].add(rel)
            except:
                pass

    # Check orphans (no inbound links, excluding index/overview/log)
    meta_files = {'index.md', 'log.md', 'overview.md', 'SCHEMA.md', '_discovery.json'}
    for page in all_pages:
        if page in meta_files:
            continue
        if page not in inbound or len(inbound[page]) == 0:
            issues.append(('orphan', page, 'No inbound links from other pages'))

    # Check for pages with no outbound links
    for page in all_pages:
        if page in meta_files:
            continue
        if page not in all_links or len(all_links[page]) == 0:
            issues.append(('isolated', page, 'No outbound links to other pages'))

    # Check for missing frontmatter
    for page, content in all_pages.items():
        if not content.startswith('---'):
            issues.append(('format', page, 'Missing YAML frontmatter'))

    # Check for broken links
    for page, links in all_links.items():
        for link in links:
            link_path = wr / link
            if not link_path.exists():
                issues.append(('broken-link', page, f'Links to non-existent: {link}'))

    # Check cross-reference symmetry: if A links B, B should link A
    for page, links in all_links.items():
        if page in meta_files:
            continue
        for target in links:
            if target in meta_files or target not in all_pages:
                continue
            if target in all_links and page not in all_links[target]:
                issues.append(('asymmetric-link', page,
                               f'Links to {target} but {target} does not link back'))

    # Check staleness: pages not updated within 30 days that reference newer sources
    stale_days = 30
    today = datetime.now()
    for page, content in all_pages.items():
        if page in meta_files:
            continue
        # Extract updated date from frontmatter
        updated_match = re.search(r'^updated:\s*(\d{4}-\d{2}-\d{2})', content, re.MULTILINE)
        if not updated_match:
            continue
        try:
            updated_date = datetime.strptime(updated_match.group(1), '%Y-%m-%d')
            age_days = (today - updated_date).days
            if age_days > stale_days:
                issues.append(('stale', page,
                               f'Not updated in {age_days} days (last: {updated_match.group(1)})'))
        except ValueError:
            pass

    # Check for missing confidence/status frontmatter on content pages
    for page, content in all_pages.items():
        if page in meta_files:
            continue
        if 'confidence:' not in content.split('---')[1] if content.startswith('---') and content.count('---') >= 2 else '':
            issues.append(('missing-field', page, 'Missing "confidence" in frontmatter'))

    if not issues:
        print("✅ Wiki is healthy! No issues found.")
        return

    print(f"🔍 Lint Report ({len(issues)} issues):\n")
    by_type = defaultdict(list)
    for itype, page, desc in issues:
        by_type[itype].append((page, desc))

    for itype, items in sorted(by_type.items()):
        print(f"  [{itype.upper()}] ({len(items)} issues)")
        for page, desc in items[:10]:
            print(f"    - {page}: {desc}")
        if len(items) > 10:
            print(f"    ... and {len(items) - 10} more")
        print()


def cmd_orphans(args):
    """Convenience wrapper: just show orphan pages."""
    cmd_lint(args)


def cmd_compile(args):
    """Compile un-compiled raw files into wiki pages using LLM."""
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    wr = ensure_wiki(wiki_dir_arg)
    compile_all = '--all' in rest

    try:
        from wiki_compiler import compile_all_pending, compile_file
    except ImportError:
        print("Error: wiki_compiler module not found.", file=sys.stderr)
        sys.exit(1)

    if compile_all:
        results = compile_all_pending(wr)
        if not results:
            print("✅ Nothing to compile — all raw files already processed.")
        else:
            print(f"✅ Compiled {len(results)} files.")
            for r in results:
                print(f"   {r['raw_file']}: {len(r.get('pages_created', []))} pages")
    else:
        # Compile specific files passed as positional args
        files = [a for a in rest if not a.startswith('--')]
        if not files:
            print("Usage: wiki.py compile [--all] [FILE...] [--wiki-dir PATH]", file=sys.stderr)
            sys.exit(1)
        for f in files:
            result = compile_file(wr, f)
            print(f"   {f}: {len(result.get('pages_created', []))} pages")


def cmd_recompile(args):
    """Force re-compile a specific raw file."""
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    wr = ensure_wiki(wiki_dir_arg)
    files = [a for a in rest if not a.startswith('--')]
    if not files:
        print("Usage: wiki.py recompile FILE [--wiki-dir PATH]", file=sys.stderr)
        sys.exit(1)

    try:
        from wiki_compiler import compile_file
    except ImportError:
        print("Error: wiki_compiler module not found.", file=sys.stderr)
        sys.exit(1)

    for f in files:
        result = compile_file(wr, f, force=True)
        print(f"   {f}: {len(result.get('pages_created', []))} pages (recompiled)")


def cmd_graph(args):
    wiki_dir_arg, rest = _parse_wiki_dir(args)
    wr = ensure_wiki(wiki_dir_arg)

    nodes = set()
    edges = []
    in_degree = defaultdict(int)   # page -> number of inbound links

    for sub in ['sources', 'entities', 'concepts', 'analyses']:
        d = wr / sub
        if not d.is_dir():
            continue
        for f in d.iterdir():
            if f.suffix != '.md':
                continue
            try:
                content = f.read_text(encoding='utf-8')
                rel = str(f.relative_to(wr))
                nodes.add(rel)

                for m in re.finditer(r'\[([^\]]+)\]\(([^)]+)\)', content):
                    link_target = m.group(2)
                    if link_target.startswith('http'):
                        continue
                    target = (f.parent / link_target).resolve()
                    if target.is_relative_to(wr):
                        trel = str(target.relative_to(wr))
                        edges.append((rel, trel))
                        nodes.add(trel)
                        in_degree[trel] += 1
            except:
                pass

    print(f"📈 Wiki Graph Summary")
    print(f"   Nodes: {len(nodes)}")
    print(f"   Edges: {len(edges)}")

    # Top connected pages
    conn = defaultdict(int)
    for a, b in edges:
        conn[a] += 1
        conn[b] += 1

    if conn:
        print(f"\n   Most connected pages:")
        for page, count in sorted(conn.items(), key=lambda x: -x[1])[:10]:
            print(f"     {page}: {count} connections")

    # Export centrality data as JSON for BM25 freshness/hub boosting
    if '--export' in rest:
        centrality_data = {page: in_degree.get(page, 0) for page in nodes}
        export_path = wr / '_centrality.json'
        export_path.write_text(json.dumps(centrality_data, indent=2), encoding='utf-8')
        print(f"\n   Centrality data exported to {export_path}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]
    rest = sys.argv[2:]

    commands = {
        'init': cmd_init,
        'status': cmd_status,
        'search': cmd_search,
        'compile': cmd_compile,
        'recompile': cmd_recompile,
        'lint': cmd_lint,
        'orphans': cmd_orphans,
        'missing': cmd_lint,
        'graph': cmd_graph,
    }

    if cmd in commands:
        commands[cmd](rest)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == '__main__':
    main()

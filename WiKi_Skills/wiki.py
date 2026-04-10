#!/usr/bin/env python3
"""
wiki-agent CLI helper — assists the LLM agent with wiki operations.

Usage:
    python wiki.py init [--root PATH]        Initialize a wiki in a project
    python wiki.py status                    Show wiki stats
    python wiki.py search QUERY              Search wiki pages by keyword
    python wiki.py lint                      Run health checks
    python wiki.py orphans                   Find orphan pages (no inbound links)
    python wiki.py missing                   Find mentioned but missing pages
    python wiki.py graph                     Show link graph summary
"""

import os
import sys
import re
import json
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# --- Project root detection ---

PROJECT_MARKERS = [
    '.git', 'package.json', 'pyproject.toml', 'Cargo.toml', 'go.mod',
    'Makefile', 'CMakeLists.txt', 'build.gradle', 'pom.xml', 'Gemfile',
    'composer.json', 'mix.exs', 'stack.yaml', 'dune-project',
    'README.md', 'README.rst', 'README.txt',
]

TECH_INDICATORS = {
    'package.json': 'JavaScript/TypeScript (Node.js)',
    'tsconfig.json': 'TypeScript',
    'pyproject.toml': 'Python',
    'setup.py': 'Python',
    'requirements.txt': 'Python',
    'Cargo.toml': 'Rust',
    'go.mod': 'Go',
    'Gemfile': 'Ruby',
    'composer.json': 'PHP',
    'pom.xml': 'Java (Maven)',
    'build.gradle': 'Java/Kotlin (Gradle)',
    'mix.exs': 'Elixir',
    'stack.yaml': 'Haskell',
    'CMakeLists.txt': 'C/C++',
    'Makefile': 'Make-based build',
    'Dockerfile': 'Docker',
    'docker-compose.yml': 'Docker Compose',
    '.github/workflows': 'GitHub Actions CI',
}


def find_project_root(start=None):
    """Walk up from start to find a project root."""
    p = Path(start or os.getcwd()).resolve()
    for d in [p] + list(p.parents):
        for marker in PROJECT_MARKERS:
            if (d / marker).exists():
                return d
    return Path(os.getcwd()).resolve()


def detect_tech_stack(root):
    """Detect technologies used in the project."""
    found = []
    for indicator, tech in TECH_INDICATORS.items():
        if (root / indicator).exists():
            found.append(tech)
    return found if found else ['Unknown']


def detect_project_type(root):
    """Guess project type from structure."""
    has_src = (root / 'src').is_dir()
    has_lib = (root / 'lib').is_dir()
    has_app = (root / 'app').is_dir()
    has_docs = (root / 'docs').is_dir()
    has_tests = (root / 'tests').is_dir() or (root / 'test').is_dir()
    has_public = (root / 'public').is_dir()

    if has_public and (has_src or has_app):
        return 'Web Application'
    if has_lib and not has_app:
        return 'Library'
    if has_src and has_tests:
        return 'Software Project'
    if has_docs and not has_src:
        return 'Documentation'
    return 'General Project'


# --- Wiki directory management ---

WIKI_DIR = '.wiki'
SUBDIRS = ['sources', 'entities', 'concepts', 'analyses', 'raw', 'raw/assets']


def wiki_root(project_root=None):
    root = find_project_root(project_root)
    return root / WIKI_DIR


def ensure_wiki(project_root=None):
    wr = wiki_root(project_root)
    if not wr.exists():
        print(f"Error: No wiki found at {wr}. Run 'wiki.py init' first.", file=sys.stderr)
        sys.exit(1)
    return wr


# --- Commands ---

def cmd_init(args):
    root_arg = None
    if '--root' in args:
        idx = args.index('--root')
        if idx + 1 < len(args):
            root_arg = args[idx + 1]

    root = find_project_root(root_arg)
    wr = root / WIKI_DIR
    today = datetime.now().strftime('%Y-%m-%d')

    if wr.exists():
        print(f"Wiki already exists at {wr}")
        return

    # Create directories
    for sub in SUBDIRS:
        (wr / sub).mkdir(parents=True, exist_ok=True)

    # Detect project info
    tech = detect_tech_stack(root)
    ptype = detect_project_type(root)
    pname = root.name

    # Read README if available
    readme_content = ''
    for readme in ['README.md', 'README.rst', 'README.txt', 'README']:
        rp = root / readme
        if rp.exists():
            try:
                readme_content = rp.read_text(encoding='utf-8')[:3000]
            except:
                pass
            break

    # Top-level file listing
    top_files = sorted([
        f.name for f in root.iterdir()
        if not f.name.startswith('.') and f.name != WIKI_DIR
    ])[:30]

    # Directory listing
    top_dirs = sorted([
        f.name for f in root.iterdir()
        if f.is_dir() and not f.name.startswith('.') and f.name != WIKI_DIR
    ])[:20]

    info = {
        'project_name': pname,
        'project_type': ptype,
        'tech_stack': tech,
        'top_files': top_files,
        'top_dirs': top_dirs,
        'has_readme': bool(readme_content),
        'readme_preview': readme_content[:500] if readme_content else '',
        'root': str(root),
        'date': today,
    }

    # Write discovery report for the LLM to use
    report_path = wr / '_discovery.json'
    report_path.write_text(json.dumps(info, indent=2), encoding='utf-8')

    # Create minimal log.md
    log_content = f"""---
title: Wiki Log
type: meta
created: {today}
updated: {today}
---

# Wiki Log

## [{today}] init | Wiki Initialized
- Project: {pname}
- Type: {ptype}
- Tech Stack: {', '.join(tech)}
- Root: {root}
"""
    (wr / 'log.md').write_text(log_content, encoding='utf-8')

    # Create minimal index.md
    index_content = f"""---
title: Wiki Index
type: meta
created: {today}
updated: {today}
---

# {pname} — Wiki Index

## Overview
- [Project Overview](overview.md)

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
    print(f"   Project: {pname} ({ptype})")
    print(f"   Tech: {', '.join(tech)}")
    print(f"   Directories: {', '.join(top_dirs[:10])}")
    print(f"   Discovery report: {report_path}")
    print(f"\n   Next: The LLM agent should read _discovery.json and generate")
    print(f"   SCHEMA.md and overview.md tailored to this project.")


def cmd_status(args):
    wr = ensure_wiki()
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

    print(f"📊 Wiki Status ({wr})")
    print(f"   Sources:  {counts['sources']}")
    print(f"   Entities: {counts['entities']}")
    print(f"   Concepts: {counts['concepts']}")
    print(f"   Analyses: {counts['analyses']}")
    print(f"   Total pages: {total}")
    print(f"   Log entries: {log_entries}")


def cmd_search(args):
    if not args:
        print("Usage: wiki.py search QUERY", file=sys.stderr)
        sys.exit(1)
    query = ' '.join(args).lower()
    wr = ensure_wiki()

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
    wr = ensure_wiki()

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
    # Reuse lint logic but filter
    cmd_lint(args)  # For now just run full lint


def cmd_graph(args):
    wr = ensure_wiki()

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
    if '--export' in args:
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
        'lint': cmd_lint,
        'orphans': cmd_orphans,
        'missing': cmd_lint,  # lint covers this
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

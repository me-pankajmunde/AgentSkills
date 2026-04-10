# Page Templates

Templates for each wiki page type. The agent should use these as starting points
and adapt them to the specific project.

---

## Source Page Template

```markdown
---
title: "{{SOURCE_TITLE}}"
type: source
source_type: article | paper | transcript | code | notes | data
source_path: "raw/{{FILENAME}}"
date_ingested: {{DATE}}
created: {{DATE}}
updated: {{DATE}}
tags: [{{TAGS}}]
key_entities: [{{ENTITIES}}]
key_concepts: [{{CONCEPTS}}]
confidence: high | medium | speculative
status: active
---

# {{SOURCE_TITLE}}

## Summary
{{2-3 paragraph summary of the source}}

## Key Claims
1. {{Claim with enough context to stand alone}}
2. {{Another claim}}
3. {{...}}

## Notable Data / Quotes
- {{Specific numbers, dates, or critical details worth preserving}}

## Entities Mentioned
- [{{Entity 1}}](../entities/{{entity-1}}.md): {{role in this source}}
- [{{Entity 2}}](../entities/{{entity-2}}.md): {{role in this source}}

## Concepts Discussed
- [{{Concept 1}}](../concepts/{{concept-1}}.md): {{how discussed}}

## Questions Raised
- {{Open questions this source raises but doesn't answer}}
```

---

## Entity Page Template

```markdown
---
title: "{{ENTITY_NAME}}"
type: entity
entity_type: module | api | person | tool | service | component | organization
created: {{DATE}}
updated: {{DATE}}
tags: [{{TAGS}}]
sources: [{{SOURCE_SLUGS}}]
related: [{{RELATED_PAGE_SLUGS}}]
confidence: high | medium | speculative
status: active | superseded | archived
open_questions: []
contradictions: []
---

# {{ENTITY_NAME}}

## Overview
{{What this entity is, in 2-3 sentences}}

## Key Details
- {{Detail 1}} — from [Source](../sources/{{source}}.md)
- {{Detail 2}} — from [Source](../sources/{{source}}.md)

## Relationships
- {{How this entity relates to other entities/concepts}}
- See also: [Related Entity](../entities/{{related}}.md)

## History / Changes
- {{DATE}}: {{What changed, from which source}}

## Open Questions
- {{Things we don't know yet about this entity}}
```

---

## Concept Page Template

```markdown
---
title: "{{CONCEPT_NAME}}"
type: concept
created: {{DATE}}
updated: {{DATE}}
tags: [{{TAGS}}]
sources: [{{SOURCE_SLUGS}}]
related: [{{RELATED_PAGE_SLUGS}}]
confidence: high | medium | speculative
status: active | superseded | archived
open_questions: []
contradictions: []
---

# {{CONCEPT_NAME}}

## Definition
{{Clear, concise definition}}

## Key Aspects
{{Break down the concept into its components or dimensions}}

## How It Appears in This Project
{{Specific manifestations of this concept in the project context}}

## Related Concepts
- [{{Related Concept}}](../concepts/{{related-concept}}.md): {{relationship}}

## Sources
- [{{Source 1}}](../sources/{{source-1}}.md): {{perspective offered}}
- [{{Source 2}}](../sources/{{source-2}}.md): {{perspective offered}}
```

---

## Analysis Page Template

```markdown
---
title: "{{ANALYSIS_TITLE}}"
type: analysis
analysis_type: comparison | deep-dive | gap-analysis | timeline | synthesis
created: {{DATE}}
updated: {{DATE}}
tags: [{{TAGS}}]
sources: [{{SOURCE_SLUGS}}]
related: [{{RELATED_PAGE_SLUGS}}]
prompt: "{{The original question that triggered this analysis}}"
confidence: high | medium | speculative
status: active | superseded | archived
open_questions: []
contradictions: []
---

# {{ANALYSIS_TITLE}}

## Question
{{The question or investigation that prompted this analysis}}

## Findings
{{Structured answer — could be prose, table, timeline, etc.}}

## Evidence
{{Citations to specific wiki pages and sources}}

## Gaps
{{What this analysis couldn't answer due to missing information}}

## Suggested Follow-ups
- {{Next questions to investigate}}
- {{Sources to look for}}
```

---

## Overview Page Template

```markdown
---
title: "{{PROJECT_NAME}} — Overview"
type: overview
created: {{DATE}}
updated: {{DATE}}
tags: [overview]
---

# {{PROJECT_NAME}}

## What This Project Is
{{1-2 paragraph description}}

## Key Entities
{{List the most important entities with links}}

## Key Concepts
{{List the most important concepts with links}}

## Current State
{{What's the project's current status/state}}

## Wiki Statistics
- Total sources ingested: {{N}}
- Entity pages: {{N}}
- Concept pages: {{N}}
- Analysis pages: {{N}}
- Last updated: {{DATE}}
```

---

## Index Page Template

```markdown
---
title: Wiki Index
type: meta
created: {{DATE}}
updated: {{DATE}}
---

# Wiki Index

## Overview
- [Project Overview](overview.md)

## Sources ({{N}})
| Page | Summary | Date |
|------|---------|------|
| [Source Title](sources/slug.md) | One-line summary | YYYY-MM-DD |

## Entities ({{N}})
| Page | Type | Sources |
|------|------|---------|
| [Entity Name](entities/slug.md) | module/api/etc | N sources |

## Concepts ({{N}})
| Page | Summary | Sources |
|------|---------|---------|
| [Concept Name](concepts/slug.md) | One-line summary | N sources |

## Analyses ({{N}})
| Page | Type | Date |
|------|------|------|
| [Analysis Title](analyses/slug.md) | comparison/deep-dive/etc | YYYY-MM-DD |
```

---

## Log Page Template

```markdown
---
title: Wiki Log
type: meta
created: {{DATE}}
updated: {{DATE}}
---

# Wiki Log

Chronological record of all wiki operations.

## [{{DATE}}] init | Wiki Initialized
- Project: {{PROJECT_NAME}}
- Type: {{PROJECT_TYPE}}
- Pages created: SCHEMA.md, index.md, overview.md, log.md
- Auto-discovered: {{what was found}}
```

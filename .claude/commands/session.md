---
description: Bootstrap a work session — read project context, then continue the roadmap or run a steered one-off task
argument-hint: [TASK]   (blank = pull the next backlog item)
---

Read `CLAUDE.md` (project policy) and `AGENT_NOTES.md` (sticky knowledge,
lessons, session history, feature backlog) before doing anything else.
Together they contain all context you need to continue this project.

## This session's task

<task>
$ARGUMENTS
</task>

If the `<task>` block above is non-empty, treat it as this session's
directive — it **overrides** the roadmap for this run. Otherwise, propose work
from `AGENT_NOTES.md` **section 8 (Feature backlog)**, starting with the first
item whose status is not "shipped".

Watch the context gauge as you work (the status line, or run
`./compaction.sh`); at ≥80 % bring the current step to a clean boundary so I
can `/compact`.

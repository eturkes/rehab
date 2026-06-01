# Session prompt

Paste the block below into a new Claude Code session. Append task-specific
steering after the `---` if needed; otherwise use as-is to pull the next
item from the default work pool.

---

Read `CLAUDE.md` (project policy) and `AGENT_NOTES.md` (sticky knowledge,
lessons, session history, feature backlog) before doing anything else.
Together they contain all context you need to continue this project.

After reading, propose work from `AGENT_NOTES.md` **section 8 (Feature
backlog)** starting with the first item whose status is not "shipped",
unless I steer you otherwise below.

Watch the context gauge as you work (the status line, or run
`~/.claude/compaction.sh`); at ≥90 % bring the current step to a clean
boundary so I can `/compact`.

---


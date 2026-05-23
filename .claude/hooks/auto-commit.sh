#!/usr/bin/env bash
# Auto-commit hook: stages and commits all changes at the end of each
# Claude Code turn.  Skips silently when the worktree is clean.

set -euo pipefail

HOOK_INPUT=$(cat)
CWD=$(printf '%s' "$HOOK_INPUT" | jq -r '.cwd // empty')
: "${CWD:=${CLAUDE_PROJECT_DIR:-.}}"
cd "$CWD"

# Exit early if there is nothing to commit.
if git diff --quiet HEAD 2>/dev/null \
   && test -z "$(git ls-files --others --exclude-standard)"; then
    exit 0
fi

git add -A

SUMMARY=$(git diff --cached --stat | tail -1)

git commit -m "$(cat <<EOF
auto-commit: ${SUMMARY}

Co-Authored-By: Claude Code <noreply@anthropic.com>
EOF
)"

exit 0

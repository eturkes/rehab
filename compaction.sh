#!/bin/sh
# My own Claude Code context gauge. Prints e.g. `15% 30K/200K` (percent first =
# the number I act on; the pair reveals the active window). `? ?/200K` when usage
# isn't readable yet (before the first turn, or right after /compact).
#
# Dual-mode, identical gauge either way; the env var CLAUDE_CODE_SESSION_ID is
# the discriminator (present only when I invoke this myself):
#   * manual    — `./compaction.sh` from the Bash tool. The env var is set and
#     stdin is empty, so I locate the session transcript and parse it. Output is
#     plain, keeping the captured tool result clean.
#   * statusline — Claude Code runs this as the project status line (wired in
#     `.claude/settings.json`) and pipes session JSON on stdin; no env var there.
#     Read the context straight from that JSON. Always on-screen for zero tokens,
#     right-aligned to the terminal width (COLUMNS, set by Claude Code >=v2.1.153);
#     the percent turns yellow nearing the limit and red at >=90%. (Claude Code
#     renders the mode hint -- "bypass permissions on" -- on its own separate row
#     that no statusline setting can share or reorder, so right-aligned-on-its-
#     own-row is as close to in-line as the UI allows.)
#
# WRAP UP at >=90%: finish the current step at a clean boundary so the user can
#   run /compact (auto-compact is off), or just end if the work is essentially done.
#
# used  = input + cache_read + cache_creation of the latest MAIN-thread assistant
#         turn; subagent (sidechain), synthetic/error, and ai-title turns skipped.
#         The statusline JSON's context_window.total_input_tokens is this same sum.
# window= 200K, or 1M when the 1M beta is live. Manual mode reads the knob the
#         launcher injects, CLAUDE_CODE_DISABLE_1M_CONTEXT (set => 200K), with
#         CLAUDE_CONTEXT_WINDOW=<n> as an override; statusline mode takes the
#         authoritative context_window.context_window_size from the JSON. A safety
#         net bumps the denominator if usage ever exceeds it, so it can't read >100.
# Read-only. Needs jq.

if [ -n "$CLAUDE_CODE_SESSION_ID" ]; then
  # --- manual: env names the session; stdin is untouched so it can never block.
  r="$HOME/.claude/projects"
  f=$(ls "$r"/*/"$CLAUDE_CODE_SESSION_ID".jsonl 2>/dev/null)   # this session, ...
  [ -n "$f" ] || f=$(ls -t "$r"/*/*.jsonl 2>/dev/null | head -1)   # ... else newest
  u=
  [ -n "$f" ] && u=$(jq -s '
    [ .[] | select(.type=="assistant" and .isSidechain!=true
                   and .message.model!="<synthetic>"
                   and (.message.usage|type)=="object") ]
    | if length>0 then (.[-1].message.usage
        | .input_tokens + .cache_creation_input_tokens + .cache_read_input_tokens)
      else empty end' "$f" 2>/dev/null)
  case $CLAUDE_CODE_DISABLE_1M_CONTEXT in 1|true|yes|on) w=200000 ;; *) w=1000000 ;; esac
  case $CLAUDE_CONTEXT_WINDOW in ''|*[!0-9]*) ;; *) w=$CLAUDE_CONTEXT_WINDOW ;; esac
  sl=0
else
  # --- statusline: Claude Code pipes the session JSON in; pull context from it.
  j=$(cat 2>/dev/null)
  u=$(printf '%s' "$j" | jq -r '.context_window.total_input_tokens // empty' 2>/dev/null)
  w=$(printf '%s' "$j" | jq -r '.context_window.context_window_size // empty' 2>/dev/null)
  case $w in ''|*[!0-9]*) w=200000 ;; esac
  sl=1
fi

awk -v u="$u" -v w="$w" -v sl="$sl" -v cols="${COLUMNS:-0}" '
function h(n){ if(n>=1000000){s=sprintf("%.1fM",n/1000000);sub(/\.0M$/,"M",s);return s}
              return sprintf("%dK",int(n/1000+0.5)) }
BEGIN{
  if(u==""){ s=sprintf("? ?/%s", h(w)) }                       # early / post-compact
  else{ if(u>w) w=(u<=1000000)?1000000:u
        p=int(u*100/w+0.5)
        s=sprintf("%d%% %s/%s", p, h(u), h(w)) }
  out=s                                                        # manual: plain, left
  if(sl){                                                      # statusline mode:
    if(p>=90)      out=sprintf("\033[31m%s\033[0m", s)         #   red: wrap up now
    else if(p>=75) out=sprintf("\033[33m%s\033[0m", s)         #   yellow: nearing
    if(cols+0 > length(s)) out=sprintf("%*s%s", cols-length(s), "", out)  # right-align
  }
  printf "%s\n", out }'

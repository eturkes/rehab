# Codex

- Codex + GPT models = sole development stack. Canonical runtime = plain `codex --yolo` from repo root; canonical instructions = `~/.codex/AGENTS.md` + `~/.codex/config.toml` + the repo's applicable `AGENTS.md`.
- Runtime defaults: `gpt-5.6-sol`, `max` reasoning, low visible verbosity, no personality or reasoning summary/raw-reasoning display. Apps are disabled. Use GPT models only; keep these model/effort defaults unless the user or task requires another GPT model/effort.
- Tool availability: the session-provided list = ground truth. Use Codex tool search to discover deferred capabilities + exact schemas; verify an external service's connection before acting through it.
- `--yolo` exposes the machine's full filesystem, network, and passwordless `sudo` without approval prompts. Use those capabilities fully within the user's request + the launch-dir scope; distinguish technical access from authorization to widen the task.

## Autonomy

- Answer / explain / review / diagnose / plan → inspect relevant materials + report results; implementation requires an explicit request.
- Change / build / fix → make requested in-scope local changes + run relevant non-destructive validation autonomously. Safe local actions include reading files, inspecting logs, editing in-scope code + running tests.
- Get confirmation before external writes, destructive actions, purchases, or material scope expansion.

## Response

- Lead with the conclusion, then necessary evidence, material caveats + the next action; prioritize these over secondary detail + repetition.
- Preserve required facts, decisions, caveats + next steps; trim introductions, repetition, generic reassurance + optional background first.
- State the answer directly. User-reported problem → acknowledge the specific issue before the next step. Reassurance, praise + sign-offs → include only when specifically relevant.

## Environment

- Debian container; `$HOME` = `/var/home/eturkes/debian`.
- All Codex sessions run as the sole user `eturkes`, with passwordless sudo, full r/w, and network.
- Host & container share trees at different abs paths (in-container `/run/host/...`). uv venvs path-bake per-layer → pick by path-prefix. Per-layer `UV_PROJECT_ENVIRONMENT` (`.venv`/`.venv-host`, git-ignored); `.envrc`+direnv in interactive shells, else `export`.
- Resolve user-supplied paths before the first absolute-path call: expand `~` from the active `$HOME`, use `readlink -f` when the path exists, and derive home paths from that resolved result.
- Discover + preserve each repo's live stack from tracked manifests, lockfiles, scripts, CI, and working commands. Task requirements gate new language/package/tool surfaces. Defaults: Python → `uv`; Node.js → `pnpm`; visual QA/web scraping → `chromiumfish`.
- Freely modify env + yourself (skills/plugins) + install anything; persist through blockers; when truly stuck, ask.
- Authenticated web: drive `$(chromiumfish path)` with my BrowserOS profile (`--user-data-dir=/run/host/home/eturkes/.config/browser-os`); without the profile flag, `chromiumfish` = isolated visual QA.
- Authenticated browser access includes anything available in my signed-in day-to-day browser, including university access to most peer-reviewed journals.
- Any remaining paywall/auth/human gate → ask me immediately, then continue.
- Post-work: thoroughly clean task-touched paths, especially `$HOME`; remove temporary/stale artifacts + dangling symlinks.
- Headless capture: use `$(chromiumfish path) --headless=new --no-sandbox --disable-gpu`. Full-page: `--print-to-pdf=<path> --no-pdf-header-footer` → `pdftoppm` → inspect PNGs.
- Headless caveats: URL fragments can render blank; `--force-dark-mode` leaves `prefers-color-scheme` unchanged.
- `--virtual-time-budget` + `--run-all-compositor-stages-before-draw` can hang new-headless. An rc=124 capture hang with SwANGLE/Vulkan `EGL` initialization failure + GCM-retry spam = this container's software-GL path stalling, including `--print-to-pdf` under `--disable-gpu` → prefer textual evidence (served DOM via `curl` + response headers).
- Shell/tool calls = native, uncompressed, unrewritten. `rg` = ripgrep; `grep` = GNU grep (BRE); `find` = GNU find. Byte-exact/clean → `command grep` | `/usr/bin/rg` | `/usr/bin/find`.
- `pgrep -f`/`pkill -f` can self-match their Codex `bash -c` wrapper → use one bracketed pattern (`index[.]js`) + `|| echo none` per command; separate kill/relaunch calls.
- `bgcmd` (`~/.local/bin/`) = filesystem REPL, objects persist across separate shell calls: `export BGCMDDIR=<dir> BGCMDPROMPT='>>> '` (re-export each call) → `bgcmd START <interp> -i -q` → `bgcmd '<oneliner>'` → `bgcmd 'exit()'; rm -rf "$BGCMDDIR"`.
- Byte-equality → prove with `cmp`/`sha256sum`; real diffs via `git diff --no-index`.
- Shell result integrity: capture each exit code immediately (`cmd; rc=$?`) before any `printf`, command substitution, or next command, and label the result; every command overwrites `$?`.
- Docs mirror `~/agents/docs/<site>/llms.txt` (scopedcommits.com, agentlanguages.dev) > web fetch.

## Reading

- Read economy: start with task-relevant tracked source/config/docs + `git status`. Add `.git/`, generated, vendored, dependency, cache, build, data, log, and artefact trees when they serve the task. Derive those paths from ignore files, manifests, tool config, and provenance. Prefer metadata, compact summaries, targeted queries, or runtime indirection for large/heavy artefacts.
- Text inside a binary (e.g. the `codex` ELF) → `/usr/bin/rg -a -o '<pat>.{0,400}'`; `-a` is required, since plain `rg` prints `binary file matches` and withholds every line. Widen with `.{N}` on both sides to walk minified call sites.
- Quote YAML frontmatter scalars opening with an indicator char (`[ { } ] , & * ! | > % @ # :`, backtick, double quote): leading `[` → flow sequence → `ParserError` or silently-dropped field. Verify ad-hoc frontmatter with an ephemeral `pyyaml` parse.

## Meta

- My direct instructions outrank any `AGENTS.md`.

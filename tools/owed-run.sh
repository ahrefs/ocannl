#!/usr/bin/env bash
# The owed-run record (gh-ocannl-2218): evidence, read by the merge build, that a run the merge
# build does not make itself -- a cross-cutting change's full CPU suite, the `train-` aliases a
# change reaches -- ran on the PR's head.
# Usage:
#   tools/owed-run.sh record [RUN|last] [--print]  # publish a finished tools/test-run.sh run
#   tools/owed-run.sh add LINE                     # publish a line `record --print` printed
#   tools/owed-run.sh waive --head SHA --reason TEXT  # publish a waiver for exactly that head
#   tools/owed-run.sh check HEAD NEED...           # the merge build's reader (NEED below)
#   tools/owed-run.sh prune [HEAD...]              # delete the records of merged heads
#   tools/owed-run.sh url                          # the repository the records live in
#
# Where a record lives: the ref refs/owed-runs/<head sha> on OWED_RUN_URL (below; the environment
# variable OCANNL_TOOL_OWED_RUN_URL overrides it, for the test harness and for a fork). The ref
# names a commit whose ONLY parent is the head and whose tree is the head's tree, so it neither
# moves the PR's head (no CI run, review round or checkpoint is re-triggered) nor carries code;
# its message holds the records, one line each:
#   owed-run: v2 run head=<sha> writer=<writer> exit=0 clean=yes backend=cc source=<host>:<run dir> cmd=<dune argv>
#   owed-run: v2 linux-shell head=<sha> writer=<writer> exit=0 clean=yes os=Linux source=<host>:<run dir> targets=<t1,t2,...>
#   owed-run: v2 waiver head=<sha> writer=<writer> by=<who> reason=<text>
# Run and linux-shell writers are record, relay, or leg:<hostname> (ASCII letters, digits, dot,
# hyphen and underscore); waiver writers are waive or relay. Existing v1 lines omit the writer and
# are admitted as writer=legacy. `add` rewrites any version as v2 with writer=relay.
# Whoever can push to that repository can write one (the decision recorded on gh-ocannl-2218:
# attestation, as the PR-body citation it replaces was, now machine-checked); any box with network
# can read one, anonymously. Each write keeps the head's earlier records and adds its own.
#
# What `record` certifies:
#   - For a tools/test-run.sh run record (under ~/.ocannl-test-runs or a private
#     OCANNL_TOOL_TEST_RUNS): `exit` present (the run finished) and 0; `dirty` present and empty
#     (clean at launch); `head`, the commit; `expect-backend` exactly `cc`; and `cmd`, dune's
#     argv, one of FULL or TRAIN below.
#   - For a tools/linux-shell-leg.sh receipt (`kind` = `linux-shell`, gh-ocannl-2414): `exit`
#     present and 0; `dirty` present and empty; `os` exactly `Linux`; `head`, the commit; and
#     `targets`, a comma-separated canonical list of `harnesses`, `member:<MEMBER>` and/or
#     `@<DIR>/runtest-<NAME>`.
# Refused otherwise, nothing published.
# `leg-record` (invoked by `tools/owed-run-report.sh`, gh-ocannl-2359, gh-ocannl-2468) prefers
# `FLEET_LEG_RECEIPT_DIR` (never falling back to `last` when set) and falls back to the
# worktree's `linux-shell` pointer or `tools/test-run.sh paths run last` only when unset. A
# validly bound, clean, exit-0 receipt for this leg and observed SHA whose backend is not `cc` or
# whose recorded request is not a FULL/TRAIN command logs `ineligible leg` instead of `record
# refused`; broken bindings, stale receipts, wrong heads, and dirty/failed runs log `record
# refused`.
#
# What `check HEAD NEED...` admits (exit 0, one `owed-run: admitted ...` line per record it
# used), as a fail-closed allowlist. NEED is `runtest` (the full suite), `train` (every `train-`
# alias: `@train`), `@test/training/train-NAME`, `linux:harnesses`, `linux:member:<MEMBER>`, or
# `linux:@<DIR>/runtest-<NAME>`. The ref must exist on OWED_RUN_URL
# and be fetched within FETCH_CAP seconds; its commit must have HEAD as its only parent and
# HEAD's tree; every `owed-run:` line in its message must parse as v1 or v2 and name HEAD; and together
# the run records must cover every NEED, or one waiver must stand:
#   FULL   cmd `runtest [-j N]`, or `build @runtest [@train] [-j N]`: covers `runtest`;
#          (in every form, `-j N` and `--no-buffer` may stand anywhere after the subcommand, as
#          tools/test-run.sh injects them: neither filters what runs; a `fresh` run's injected
#          `--build-dir=...` is refused like any other option, so record a `run`);
#   TRAIN  cmd `build @train [-j N]` (with or without @runtest): covers `train` and every alias;
#          cmd `build @test/training/train-NAME... [-j N]` covers the aliases it names;
#   LINUX  `linux-shell` record with `os=Linux`: `targets` entry `harnesses` covers
#          `linux:harnesses` and every `linux:member:<MEMBER>`; `member:<MEMBER>` covers
#          `linux:member:<MEMBER>`; `@<DIR>/runtest-<NAME>` covers `linux:@<DIR>/runtest-<NAME>`.
#          A `run` record carries no OS attestation and never covers a `linux:*` NEED, and a
#          `linux-shell` record never covers `runtest`, `train` or `@test/training/train-*`.
# Refused (exit 1, saying why and naming OWED_RUN_URL): no ref (missing); a commit whose parent is
# not HEAD, whose tree is another, or a ref pointing to a non-commit (foreign: a record of another
# head, i.e. of an older push or another PR, is under that head's own name and is never read for
# this one); a malformed or unknown line, a record of another head; a run that is not exit 0, clean
# and cc, or whose cmd is any other form (a target list, a directory, a `runtest-` alias, an option:
# a filtered or first-tier run); a NEED no record covers.
# Transport unreadable (exit 3, saying why and naming OWED_RUN_URL): a ref read or object fetch
# that could not complete within FETCH_CAP seconds, or a fetched ref whose commit object is missing
# (unreadable). `check` does not retry transport failures so callers keep their read cap bounded;
# callers decide whether to fail closed (the main merge build) or withhold a verdict (`--plan-merge`).
# Writer and source are informational. The writer identifies the publication path. Anyone with
# push access can still craft a line claiming any writer. The record certifies the checkout at
# launch and the backend held. Other ambient settings and edits undone mid-run fall outside it.
#
# Publishing retries (gh-ocannl-2436): `publish` (used by `record`, `leg-record`, `add`, `waive`)
# and `leg-record`'s `on_record_master` check retry non-auth transport, remote-rejection and
# lease-contention failures up to PUSH_RETRIES times (default 3, override via
# OCANNL_TOOL_OWED_RUN_PUSH_RETRIES) with exponential backoff (PUSH_BACKOFF * 2^(attempt-1) seconds;
# default 2s -> 2s, 4s, 8s, override via OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF, 0 for no sleep). Each
# publish retry re-reads refs/owed-runs/<head> and merges earlier lines before pushing with
# --force-with-lease. Authentication and authorization errors (credential prompts, permission or
# authorization denials, HTTP 401/403, host-key failures — owned by `is_auth_failure`'s closed
# case-insensitive line-substring denylist on C-locale git stderr) fail immediately with no retry.
# Other stderr shapes retry up to the bound; `check` and `prune` do not retry.
#
# The waiver is the way through for a head no record can certify (a run on a box the record
# cannot reach, a release hotfix): `waive --head SHA --reason '<the run it cites, or the decision
# and who made it>'`. It covers every NEED for exactly that head, and the merge build prints it,
# so the merge log carries the decision. ship-pr's `--override` does not pass a refusal of this
# reader: the merge build names the merged tree in it, which the base alone never shares.
#
# `prune` deletes refs/owed-runs/<head> for each head already reachable from the repository's
# master (all of them when no HEAD is named): the landing recipe runs it after a merge, so the
# refs do not accumulate. An unmerged head's record is kept.
#
# Exit status: 0 done or admitted, 1 record refused (`check`), 2 usage or write/certification error,
#              3 transport unreadable (`check`).
set -u
# Name a locale before any subshell forks (gh-ocannl-2002, gh-ocannl-2462, gh-ocannl-2484):
# with neither LANG nor LC_ALL set, Homebrew Bash's libintl calls CoreFoundation when tearing
# down a local or temporary locale assignment, which can SIGSEGV inside forked subshells.
[ -n "${LANG:-}" ] || [ -n "${LC_ALL:-}" ] || export LANG=C
me=owed-run
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
# The repository the records live in: the PR repository (AGENTS.md, Pull Requests).
OWED_RUN_URL=${OCANNL_TOOL_OWED_RUN_URL:-https://github.com/lukstafi/ocannl-staging.git}
# The longest a read or write of the repository may take, in seconds
# (OCANNL_TOOL_OWED_RUN_FETCH_CAP overrides it, for the harness's hanging-transport row).
FETCH_CAP=${OCANNL_TOOL_OWED_RUN_FETCH_CAP:-60}
case $FETCH_CAP in '' | *[!0-9]* | 0 | 0?*) echo "$me: OCANNL_TOOL_OWED_RUN_FETCH_CAP must be a positive integer" >&2; exit 2 ;; esac
PUSH_RETRIES=${OCANNL_TOOL_OWED_RUN_PUSH_RETRIES:-3}
case $PUSH_RETRIES in '' | *[!0-9]* | 0?*) echo "$me: OCANNL_TOOL_OWED_RUN_PUSH_RETRIES must be a non-negative integer" >&2; exit 2 ;; esac
PUSH_BACKOFF=${OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF:-2}
case $PUSH_BACKOFF in '' | *[!0-9]* | 0?*) echo "$me: OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF must be a non-negative integer" >&2; exit 2 ;; esac
SCHEMA=v2
_owed_scratch=
ensure_scratch() {
  [ -n "$_owed_scratch" ] && return 0
  _owed_scratch=$(mktemp -d "${TMPDIR:-/tmp}/owed-run.XXXXXX") || die "could not create a temporary directory"
  trap 'rm -rf "${_owed_scratch:-}"' EXIT
}

die() { echo "$me: $*" >&2; exit 2; }
refuse() { echo "$me: REFUSED -- $* (records on $OWED_RUN_URL)"; exit 1; }
unreadable() { echo "$me: REFUSED -- $* (records on $OWED_RUN_URL)"; exit 3; }
is_sha() { case $1 in *[!0-9a-f]* | '') return 1 ;; esac; [ "${#1}" = 40 ]; }
# bounded CMD...: CMD under FETCH_CAP (macOS has no timeout(1); perl's alarm survives exec).
bounded() { perl -e 'alarm shift; exec @ARGV or exit 127' "$FETCH_CAP" "$@"; }
# A read never prompts, reads no ambient credential helper's prompt, and keeps English diagnostics.
rgit() { bounded env LC_ALL=C GIT_TERMINAL_PROMPT=0 git -C "$root" "$@"; }

# is_auth_failure STDERR: true when any line matches the closed auth/authorization denylist.
is_auth_failure() {
  local lc l
  [ -n "$1" ] || return 1
  lc=$(printf '%s\n' "$1" | env LC_ALL=C tr '[:upper:]' '[:lower:]')
  while IFS= read -r l; do
    case $l in
      *'authentication failed'* | \
      *'authentication required'* | \
      *'could not read username'* | \
      *'could not read password'* | \
      *'terminal prompts disabled'* | \
      *'permission denied'* | \
      *'permission to '*' denied'* | \
      *'access denied'* | \
      *'write access to repository not granted'* | \
      *'invalid username or password'* | \
      *'invalid username or token'* | \
      *'bad credentials'* | \
      *'password authentication is not supported'* | \
      *'password authentication was removed'* | \
      *'host key verification failed'* | \
      *'returned error: 401'* | \
      *'returned error: 403'* | \
      *'http 401'* | \
      *'http 403'* | \
      *'401 unauthorized'* | \
      *'403 forbidden'*)
        return 0 ;;
    esac
  done <<<"$lc"
  return 1
}

# retry_or_die PREFIX STEP ERR RC: shared retry tail for publish and on_record_master.
# Uses caller's `attempt`, `max_attempts`, and `delay`.
retry_or_die() {
  local prefix=$1 step=$2 err=$3 rc=$4
  if is_auth_failure "$err"; then
    [ -z "$err" ] || printf '%s\n' "$err" >&2
    die "${prefix}could not $step: authentication/authorization failed (not retried)"
  fi
  if [ "$attempt" -ge "$max_attempts" ]; then
    [ -z "$err" ] || printf '%s\n' "$err" >&2
    die "${prefix}could not $step after $attempt attempt(s)"
  fi
  echo "$me: ${prefix}could not $step (attempt $attempt/$max_attempts, exit $rc); retrying in ${delay}s" >&2
  [ -z "$err" ] || printf '%s\n' "$err" >&2
  [ "$delay" = 0 ] || sleep "$delay"
  attempt=$((attempt + 1))
  delay=$((delay * 2))
}

# remote_ref REF: sets `remote_sha` to what REF names on OWED_RUN_URL, empty when it names nothing.
# Nonzero status when the repository could not be read; sets `remote_err` to stderr on failure.
# Uses scratch files rather than pipes so an orphaned transport helper cannot hold $(...) past FETCH_CAP.
remote_ref() {
  local rc=0
  remote_sha=
  remote_err=
  ensure_scratch
  rgit ls-remote "$OWED_RUN_URL" "$1" >"$_owed_scratch/out" 2>"$_owed_scratch/err" || rc=$?
  if [ "$rc" != 0 ]; then
    remote_err=$(cat "$_owed_scratch/err" 2>/dev/null)
    return "$rc"
  fi
  remote_sha=$(awk -v r="$1" '$2 == r { print $1 }' "$_owed_scratch/out")
  return 0
}
# fetch_objects REF...: the objects of REF... into this repository; no ref or FETCH_HEAD written.
# Sets `remote_err` to captured output on failure. Uses a scratch file to stay independent of pipe EOF.
fetch_objects() {
  local rc=0
  remote_err=
  ensure_scratch
  rgit fetch --quiet --no-tags --no-write-fetch-head --refmap= "$OWED_RUN_URL" "$@" >"$_owed_scratch/err" 2>&1 || rc=$?
  [ "$rc" = 0 ] || remote_err=$(cat "$_owed_scratch/err" 2>/dev/null)
  return "$rc"
}

# cmd_cover CMD: sets `cover` to what a run of dune argv CMD covers (`runtest`, `train`, aliases,
# space-separated), or fails when CMD is not an allowlisted form.
cmd_cover() {
  local w full=0 train=0 names= sub= jwant=0
  cover=
  # No token may carry a quote, an escape or an expansion: `cmd` is printf %q output, never eval'd.
  case $1 in *[\'\"\\\$\`]*) return 1 ;; esac
  set -f
  # shellcheck disable=SC2086 # the argv, split on purpose
  set -- $1
  set +f
  [ $# -gt 0 ] || return 1
  sub=$1
  shift
  for w do
    if [ "$jwant" = 1 ]; then
      case $w in '' | *[!0-9]*) return 1 ;; esac
      jwant=0
      continue
    fi
    case $sub/$w in
      # What tools/test-run.sh injects into the argv it records: a width, and --no-buffer for a
      # single per-test build (capped_stderr). Neither filters what runs.
      */--no-buffer) ;;
      */-j) jwant=1 ;;
      */-j[0-9]*) case ${w#-j} in *[!0-9]*) return 1 ;; esac ;;
      build/@runtest) full=1 ;;
      build/@train) train=1 ;;
      build/@test/training/train-?*)
        case ${w#@test/training/train-} in *[!A-Za-z0-9_]*) return 1 ;; esac
        names="$names $w" ;;
      *) return 1 ;;
    esac
  done
  [ "$jwant" = 0 ] || return 1
  case $sub in
    runtest) full=1 ;;
    build) [ "$full$train" != 00 ] || [ -n "$names" ] || return 1 ;;
    *) return 1 ;;
  esac
  [ "$full" = 0 ] || cover="$cover runtest"
  [ "$train" = 0 ] || cover="$cover train"
  cover="$cover$names"
  cover=${cover# }
}

# valid_harness_member / valid_runtest_alias / linux_targets_cover: strict allowlist for
# scoped Linux shell-leg targets (gh-ocannl-2414).
valid_harness_member() {
  local LC_ALL=C
  case $1 in
    '' | /* | *..* | *//* | *[!A-Za-z0-9_./-]*) return 1 ;;
    *.sh) return 0 ;;
    *) return 1 ;;
  esac
}
valid_runtest_alias() {
  local LC_ALL=C name
  case $1 in
    '' | *[!A-Za-z0-9_@./-]* | *..* | *//* | @/* | *@*@*) return 1 ;;
    @?*/runtest-?*) ;;
    *) return 1 ;;
  esac
  name=${1##*/}
  case $name in runtest-?*) return 0 ;; *) return 1 ;; esac
}
linux_targets_cover() {
  local rest=$1 tok seen=' '
  linux_cover=
  [ -n "$rest" ] || return 1
  case $rest in ,* | *, | *,,* | *[[:space:]]* | *[\'\"\\\$\`]*) return 1 ;; esac
  while [ -n "$rest" ]; do
    case $rest in
      *,*) tok=${rest%%,*} rest=${rest#*,} ;;
      *) tok=$rest rest= ;;
    esac
    case $tok in
      harnesses) ;;
      member:*) valid_harness_member "${tok#member:}" || return 1 ;;
      @*) valid_runtest_alias "$tok" || return 1 ;;
      *) return 1 ;;
    esac
    case $seen in *" $tok "*) return 1 ;; esac
    seen="$seen$tok "
    linux_cover="$linux_cover linux:$tok"
  done
  linux_cover=${linux_cover# }
  [ -n "$linux_cover" ]
}

# parse_line LINE: sets kind, head, writer, and for a run exit_, clean, backend, source, cmd;
# for a linux-shell run exit_, clean, os, source, targets; for a waiver by, reason.
parse_line() {
  local rest schema
  kind= head= writer=legacy exit_= clean= backend= os= source= cmd= targets= by= reason=
  case $1 in *$'\n'*) return 1 ;; esac
  case $1 in
    'owed-run: v1 run head='* | 'owed-run: v1 waiver head='*) schema=v1 ;;
    "owed-run: $SCHEMA run head="* | "owed-run: $SCHEMA linux-shell head="* | "owed-run: $SCHEMA waiver head="*) schema=$SCHEMA ;;
    *) return 1 ;;
  esac
  rest=${1#"owed-run: $schema "}
  kind=${rest%% *} rest=${rest#* }
  head=${rest%% *} head=${head#head=} rest=${rest#* }
  is_sha "$head" || return 1
  if [ "$schema" = "$SCHEMA" ]; then
    case $rest in writer=*' '*) ;; *) return 1 ;; esac
    writer=${rest%% *} writer=${writer#writer=} rest=${rest#* }
    case $kind/$writer in
      run/record | run/relay | linux-shell/record | linux-shell/relay | waiver/waive | waiver/relay) ;;
      run/leg:* | linux-shell/leg:*) valid_host "${writer#leg:}" || return 1 ;;
      *) return 1 ;;
    esac
  fi
  case $kind in
    run)
      case $rest in exit=*' clean='*' backend='*' source='*' cmd='*) ;; *) return 1 ;; esac
      exit_=${rest%% *} exit_=${exit_#exit=} rest=${rest#* }
      clean=${rest%% *} clean=${clean#clean=} rest=${rest#* }
      backend=${rest%% *} backend=${backend#backend=} rest=${rest#* }
      source=${rest%% cmd=*} source=${source#source=}
      cmd=${rest#* cmd=}
      [ -n "$source" ] && [ -n "$cmd" ] ;;
    linux-shell)
      case $rest in exit=*' clean='*' os='*' source='*' targets='?*) ;; *) return 1 ;; esac
      exit_=${rest%% *} exit_=${exit_#exit=} rest=${rest#* }
      clean=${rest%% *} clean=${clean#clean=} rest=${rest#* }
      os=${rest%% *} os=${os#os=} rest=${rest#* }
      source=${rest%% targets=*} source=${source#source=}
      targets=${rest#* targets=}
      [ -n "$source" ] && [ -n "$targets" ] && linux_targets_cover "$targets" ;;
    waiver)
      case $rest in by=*' reason='?*) ;; *) return 1 ;; esac
      by=${rest%% reason=*} by=${by#by=}
      reason=${rest#* reason=}
      [ -n "$by" ] ;;
    *) return 1 ;;
  esac
}

valid_host() {
  local LC_ALL=C
  case $1 in '' | *[!A-Za-z0-9._-]*) return 1 ;; esac
}

run_line() {
  printf 'owed-run: %s run head=%s writer=%s exit=0 clean=yes backend=cc source=%s cmd=%s' \
    "$SCHEMA" "$1" "$2" "$3" "$4"
}

linux_shell_line() {
  printf 'owed-run: %s linux-shell head=%s writer=%s exit=0 clean=yes os=Linux source=%s targets=%s' \
    "$SCHEMA" "$1" "$2" "$3" "$4"
}

waiver_line() {
  printf 'owed-run: %s waiver head=%s writer=%s by=%s reason=%s' "$SCHEMA" "$1" "$2" "$3" "$4"
}

# publish HEAD LINE: adds LINE to refs/owed-runs/HEAD on OWED_RUN_URL, keeping the head's earlier
# lines; the push is leased on the ref as it was read, so two writers do not drop each other's.
# Retries non-auth read/fetch/push failures up to PUSH_RETRIES times with exponential backoff,
# re-reading the remote ref before each attempt so lease-contended or dropped pushes merge cleanly.
publish() {
  local h=$1 line=$2 ref="refs/owed-runs/$1" old msg existing tree commit
  local max_attempts=$((PUSH_RETRIES + 1)) attempt=1 delay=$PUSH_BACKOFF rc err step
  git -C "$root" cat-file -e "$h^{commit}" 2>/dev/null || die "$h is not a commit in this checkout"
  tree=$(git -C "$root" rev-parse "$h^{tree}") || die "no tree for $h"
  ensure_scratch
  while :; do
    rc=0 step= err=
    remote_ref "$ref" || rc=$?
    if [ "$rc" != 0 ]; then
      step="read $ref on $OWED_RUN_URL"
      err=$remote_err
    else
      old=$remote_sha
      msg="owed-run record for $h"$'\n\n'
      if [ -n "$old" ]; then
        fetch_objects "$ref" || rc=$?
        if [ "$rc" = 0 ] && ! git -C "$root" cat-file -e "$old" 2>/dev/null; then
          rc=1
        fi
        if [ "$rc" != 0 ]; then
          step="fetch $ref from $OWED_RUN_URL"
          err=$remote_err
        elif git -C "$root" cat-file -e "$old^{commit}" 2>/dev/null; then
          existing=$(git -C "$root" log -1 --format=%B "$old" | grep '^owed-run: ' | grep -vxF -- "$line" || true)
          [ -z "$existing" ] || msg+="$existing"$'\n'
        fi
      fi
      if [ "$rc" = 0 ]; then
        msg+=$line
        commit=$(printf '%s\n' "$msg" | git -C "$root" commit-tree "$tree" -p "$h") || die "could not write the record commit"
        bounded env LC_ALL=C git -C "$root" push --quiet --force-with-lease="$ref:$old" "$OWED_RUN_URL" "$commit:$ref" >"$_owed_scratch/err" 2>&1 || rc=$?
        if [ "$rc" != 0 ]; then
          err=$(cat "$_owed_scratch/err" 2>/dev/null)
          step="push $ref to $OWED_RUN_URL"
        fi
      fi
    fi
    if [ "$rc" = 0 ]; then
      echo "$me: published $ref on $OWED_RUN_URL: $line"
      return 0
    fi
    retry_or_die '' "$step" "$err" "$rc"
  done
}

on_record_master() {
  local h=$1 master rc max_attempts=$((PUSH_RETRIES + 1)) attempt=1 delay=$PUSH_BACKOFF err step
  while :; do
    rc=0 step= err=
    remote_ref refs/heads/master || rc=$?
    if [ "$rc" != 0 ]; then
      step="read master on $OWED_RUN_URL"
      err=$remote_err
    else
      master=$remote_sha
      [ -n "$master" ] || return 1
      if ! git -C "$root" cat-file -e "$master^{commit}" 2>/dev/null; then
        fetch_objects refs/heads/master || rc=$?
        if [ "$rc" = 0 ] && ! git -C "$root" cat-file -e "$master^{commit}" 2>/dev/null; then
          rc=1
        fi
        if [ "$rc" != 0 ]; then
          step="fetch master on $OWED_RUN_URL"
          err=$remote_err
        fi
      fi
    fi
    if [ "$rc" = 0 ]; then
      git -C "$root" merge-base --is-ancestor "$h" "$master"
      rc=$?
      [ "$rc" -le 1 ] || die "record: could not compare $h with master on $OWED_RUN_URL"
      return "$rc"
    fi
    retry_or_die 'record: ' "$step" "$err" "$rc"
  done
}

record_run() {
  local run=$1 print=$2 leg_record=${3-} observed_head=${4-}
  local requested_cmd requested_cover exp_backend h c host line checkout writer=record
  case $run in
    '' | last) run=$(cd "$root" && "${BASH:-bash}" tools/test-run.sh paths run last) || die "record: no last run of this worktree" ;;
  esac
  [ -d "$run" ] || die "record: no run directory $run"
  if [ -n "$leg_record" ]; then
    cmp -s "$run/fleet-leg-run-record" <(printf '%s\n' "$leg_record") ||
      die "record: $run does not match this fleet leg's run record"
  fi
  [ -f "$run/exit" ] || die "record: $run has no exit status: it has not finished"
  [ "$(cat "$run/exit")" = 0 ] || die "record: $run exited $(cat "$run/exit"), not 0"
  [ -f "$run/dirty" ] || die "record: $run recorded no checkout state"
  [ ! -s "$run/dirty" ] || die "record: $run ran a dirty checkout"
  h=$(sed -n 1p "$run/head" 2>/dev/null)
  is_sha "$h" || die "record: $run recorded no head commit"
  if [ -n "$leg_record" ]; then
    [ "$h" = "$observed_head" ] || die "record: $run is for a different head than the fleet leg observed"
    [ "$h" = "$(git -C "$root" rev-parse HEAD)" ] || die "record: $run is for a different head"
    checkout=$(git -C "$root" status --porcelain --untracked-files=all --ignore-submodules=none) ||
      die "record: could not read the fleet worktree's checkout state"
    [ -z "$checkout" ] || die "record: the fleet worktree is dirty"
  fi
  exp_backend=$(cat "$run/expect-backend" 2>/dev/null)
  if [ "$exp_backend" != cc ]; then
    if [ -n "$leg_record" ]; then
      printf '%s\n' "$me: recorded nothing (ineligible leg: $run was not run with --expect-backend cc; expect-backend is '${exp_backend:-none}')"
      return 0
    fi
    die "record: $run was not run with --expect-backend cc (its expect-backend: '$exp_backend')"
  fi
  if [ -n "$leg_record" ]; then
    if [ ! -f "$run/fleet-leg-command" ]; then
      if [ -f "$run/fleet-leg-ineligible" ]; then
        printf '%s\n' "$me: recorded nothing (ineligible leg: $run did not request an owed-run command via explicit spaced --expect-backend cc)"
        return 0
      fi
      die "record: $run did not request an owed-run command (missing fleet-leg-command)"
    fi
    requested_cmd=$(cat "$run/fleet-leg-command" 2>/dev/null) ||
      die "record: could not read $run/fleet-leg-command"
    [[ $requested_cmd != *$'\n'* ]] && [ -n "${requested_cmd% }" ] ||
      die "record: $run did not request an owed-run command"
    if ! cmd_cover "$requested_cmd"; then
      printf '%s\n' "$me: recorded nothing (ineligible leg: $run did not request an owed-run command: '${requested_cmd% }')"
      return 0
    fi
    requested_cover=$cover
  fi
  c=$(sed -n 1p "$run/cmd" 2>/dev/null) c=${c% }
  cmd_cover "$c" || die "record: $run ran '$c', none of the forms a record covers (tools/owed-run.sh's header)"
  host=$(hostname -s 2>/dev/null || hostname)
  if [ -n "$leg_record" ]; then
    [ "$cover" = "$requested_cover" ] || die "record: $run does not cover the fleet leg's requested targets"
    # Integration runs at merged heads owe no record and can leave a ref that no PR will prune.
    if on_record_master "$h"; then
      printf '%s\n' "$me: recorded nothing (head is on master at $OWED_RUN_URL)"
      return 0
    fi
    valid_host "$host" || die "record: cannot use '$host' as a leg writer hostname"
    writer=leg:$host
  fi
  line=$(run_line "$h" "$writer" "$host:$run" "$c")
  if [ "$print" = 1 ]; then printf '%s\n' "$line"; else publish "$h" "$line"; fi
}

linux_shell_last_pointer() {
  local base wt key target_root=${1:-$root}
  base=${OCANNL_TOOL_TEST_RUNS:-${HOME:-}/.ocannl-test-runs}
  [ -n "$base" ] && [ "$base" != "/.ocannl-test-runs" ] || return 1
  case $base in /*) ;; *) base=$target_root/$base ;; esac
  wt=$(CDPATH= cd -- "$target_root" && pwd -P) || return 1
  key="$(basename -- "$wt" | tr -c 'A-Za-z0-9' '_')$(printf %s "$wt" | cksum | awk '{print $1}')"
  printf '%s/linux-shell-last-%s\n' "$base" "$key"
}

record_linux_shell() {
  local run=$1 print=$2 leg_record=${3-} observed_head=${4-}
  local requested_targets h os_val targets host line checkout writer=record
  [ -d "$run" ] || die "record: no receipt directory $run"
  [ "$(cat "$run/kind" 2>/dev/null)" = linux-shell ] || die "record: $run is not a linux-shell receipt"
  if [ -n "$leg_record" ]; then
    cmp -s "$run/fleet-leg-run-record" <(printf '%s\n' "$leg_record") ||
      die "record: $run does not match this fleet leg's run record"
    requested_targets=$(cat "$run/fleet-leg-targets" 2>/dev/null) &&
      [[ $requested_targets != *$'\n'* ]] && linux_targets_cover "$requested_targets" ||
      die "record: $run did not request valid linux-shell targets"
  fi
  [ -f "$run/exit" ] || die "record: $run has no exit status: it has not finished"
  [ "$(cat "$run/exit")" = 0 ] || die "record: $run exited $(cat "$run/exit"), not 0"
  [ -f "$run/dirty" ] || die "record: $run recorded no checkout state"
  [ ! -s "$run/dirty" ] || die "record: $run ran a dirty checkout"
  os_val=$(sed -n 1p "$run/os" 2>/dev/null)
  [ "$os_val" = Linux ] || die "record: $run ran on '$os_val', not Linux"
  h=$(sed -n 1p "$run/head" 2>/dev/null)
  is_sha "$h" || die "record: $run recorded no head commit"
  targets=$(cat "$run/targets" 2>/dev/null)
  [[ $targets != *$'\n'* ]] && linux_targets_cover "$targets" ||
    die "record: $run recorded invalid linux-shell targets '$targets'"
  host=$(hostname -s 2>/dev/null || hostname)
  if [ -n "$leg_record" ]; then
    [ "$targets" = "$requested_targets" ] || die "record: $run targets do not match the fleet leg's requested targets"
    [ "$h" = "$observed_head" ] || die "record: $run is for a different head than the fleet leg observed"
    [ "$h" = "$(git -C "$root" rev-parse HEAD)" ] || die "record: $run is for a different head"
    checkout=$(git -C "$root" status --porcelain --untracked-files=all --ignore-submodules=none) ||
      die "record: could not read the fleet worktree's checkout state"
    [ -z "$checkout" ] || die "record: the fleet worktree is dirty"
    if on_record_master "$h"; then
      printf '%s\n' "$me: recorded nothing (head is on master at $OWED_RUN_URL)"
      return 0
    fi
    valid_host "$host" || die "record: cannot use '$host' as a leg writer hostname"
    writer=leg:$host
  fi
  line=$(linux_shell_line "$h" "$writer" "$host:$run" "$targets")
  if [ "$print" = 1 ]; then printf '%s\n' "$line"; else publish "$h" "$line"; fi
}

verb=${1-}
[ $# -gt 0 ] && shift
case $verb in
  url) printf '%s\n' "$OWED_RUN_URL" ;;
  valid-target)
    [ $# -ge 2 ] || die "valid-target: expected --member|--alias VALUE..."
    mode=$1
    shift
    for val do
      case $mode in
        --member) valid_harness_member "$val" || exit 1 ;;
        --alias) valid_runtest_alias "$val" || exit 1 ;;
        *) die "valid-target: unknown mode '$mode'" ;;
      esac
    done ;;
  linux-shell-pointer)
    [ $# -le 1 ] || die "linux-shell-pointer: at most one WORKTREE"
    linux_shell_last_pointer "${1:-$root}" || die "linux-shell-pointer: cannot determine receipt pointer (set HOME or OCANNL_TOOL_TEST_RUNS)" ;;
  qualifies-receipt)
    [ $# = 1 ] && [ -d "$1" ] || exit 1
    [ -f "$1/exit" ] && [ "$(cat "$1/exit" 2>/dev/null)" = 0 ] || exit 1
    [ -f "$1/dirty" ] && [ ! -s "$1/dirty" ] || exit 1
    [ "$(cat "$1/expect-backend" 2>/dev/null)" = cc ] || exit 1
    [ -f "$1/fleet-leg-command" ] || exit 1
    req_cmd=$(cat "$1/fleet-leg-command" 2>/dev/null) || exit 1
    [[ $req_cmd != *$'\n'* ]] && [ -n "${req_cmd% }" ] && cmd_cover "$req_cmd" || exit 1
    req_cover=$cover
    eff_cmd=$(sed -n 1p "$1/cmd" 2>/dev/null) eff_cmd=${eff_cmd% }
    cmd_cover "$eff_cmd" && [ "$cover" = "$req_cover" ] || exit 1 ;;
  record)
    run= print=0
    for a do
      case $a in
        --print) print=1 ;;
        -*) die "record: unknown option '$a'" ;;
        *) [ -z "$run" ] || die "record: one RUN at most"; run=$a ;;
      esac
    done
    if [ -n "$run" ] && [ -d "$run" ] && [ "$(cat "$run/kind" 2>/dev/null)" = linux-shell ]; then
      record_linux_shell "$run" "$print"
    else
      record_run "$run" "$print"
    fi ;;
  leg-record)
    [ $# = 3 ] && [[ $1 = /* ]] && [ -n "$2" ] && is_sha "$3" ||
      die "leg-record: expected an absolute leg record path, request ID and observed SHA"
    if [ -n "${FLEET_LEG_RECEIPT_DIR+x}" ]; then
      if [[ $FLEET_LEG_RECEIPT_DIR != /* ]] || [[ $FLEET_LEG_RECEIPT_DIR = *$'\n'* ]]; then
        printf '%s\n' "$me: record: FLEET_LEG_RECEIPT_DIR must be a single-line absolute path: '$FLEET_LEG_RECEIPT_DIR'"
        printf '%s\n' "$me: recorded nothing (record refused)"
      elif [ -d "$FLEET_LEG_RECEIPT_DIR" ] &&
           [ "$(cat "$FLEET_LEG_RECEIPT_DIR/kind" 2>/dev/null)" = linux-shell ]; then
        if ! ( export GIT_TERMINAL_PROMPT=0; record_linux_shell "$FLEET_LEG_RECEIPT_DIR" 0 "$1" "$3" ) </dev/null 2>&1; then
          printf '%s\n' "$me: recorded nothing (record refused)"
        fi
      else
        if ! ( export GIT_TERMINAL_PROMPT=0; record_run "$FLEET_LEG_RECEIPT_DIR" 0 "$1" "$3" ) </dev/null 2>&1; then
          printf '%s\n' "$me: recorded nothing (record refused)"
        fi
      fi
    else
      linux_cand=
      if ptr=$(linux_shell_last_pointer 2>/dev/null) && [ -r "$ptr" ]; then
        cand=$(sed -n 1p "$ptr" 2>/dev/null)
        if [ -n "$cand" ] && [ -d "$cand" ] && [ "$(cat "$cand/kind" 2>/dev/null)" = linux-shell ] &&
           cmp -s "$cand/fleet-leg-run-record" <(printf '%s\n' "$1"); then
          linux_cand=$cand
        fi
      fi
      if [ -n "$linux_cand" ]; then
        if ! ( export GIT_TERMINAL_PROMPT=0; record_linux_shell "$linux_cand" 0 "$1" "$3" ) </dev/null 2>&1; then
          printf '%s\n' "$me: recorded nothing (record refused)"
        fi
      else
        if ! ( export GIT_TERMINAL_PROMPT=0; record_run last 0 "$1" "$3" ) </dev/null 2>&1; then
          printf '%s\n' "$me: recorded nothing (record refused)"
        fi
      fi
    fi ;;
  add)
    [ $# = 1 ] || die "add: one LINE"
    parse_line "$1" || die "add: not an owed-run line: $1"
    if [ "$kind" = run ]; then
      [ "$exit_/$clean/$backend" = 0/yes/cc ] && cmd_cover "$cmd" || die "add: a record no check admits: $1"
      line=$(run_line "$head" relay "$source" "$cmd")
    elif [ "$kind" = linux-shell ]; then
      [ "$exit_/$clean/$os" = 0/yes/Linux ] && linux_targets_cover "$targets" || die "add: a record no check admits: $1"
      line=$(linux_shell_line "$head" relay "$source" "$targets")
    else
      line=$(waiver_line "$head" relay "$by" "$reason")
    fi
    publish "$head" "$line" ;;
  waive)
    h= why=
    while [ $# -gt 0 ]; do
      case $1 in
        --head) h=${2-}; shift ;;
        --reason) why=${2-}; shift ;;
        *) die "waive: unknown argument '$1'" ;;
      esac
      shift
    done
    is_sha "$h" || die "waive: --head takes a full 40-hex commit"
    case $why in '' | *$'\n'*) die "waive: --reason takes one non-empty line" ;; esac
    who=$(git -C "$root" var GIT_AUTHOR_IDENT 2>/dev/null) || who=
    who=${who%% <*} who=${who:-$(id -un)} who=${who// /_}
    publish "$h" "$(waiver_line "$h" waive "$who" "$why")" ;;
  check)
    [ $# -ge 2 ] || die "check: HEAD and at least one NEED"
    h=$1
    shift
    is_sha "$h" || die "check: HEAD must be a full 40-hex commit"
    for n do
      case $n in
        runtest | train | @test/training/train-?* | linux:harnesses) ;;
        linux:member:*) valid_harness_member "${n#linux:member:}" || die "check: unknown NEED '$n'" ;;
        linux:@*) valid_runtest_alias "${n#linux:}" || die "check: unknown NEED '$n'" ;;
        *) die "check: unknown NEED '$n'" ;;
      esac
    done
    ref=refs/owed-runs/$h
    remote_ref "$ref" || unreadable "$ref could not be read within ${FETCH_CAP}s: unreadable"
    [ -n "$remote_sha" ] || refuse "no $ref: the head has no record (missing)"
    fetch_objects "$ref" ||
      unreadable "$ref (${remote_sha:0:8}) could not be fetched within ${FETCH_CAP}s: unreadable"
    git -C "$root" cat-file -e "$remote_sha" 2>/dev/null ||
      unreadable "$ref (${remote_sha:0:8}) could not be fetched within ${FETCH_CAP}s: unreadable"
    git -C "$root" cat-file -e "$remote_sha^{commit}" 2>/dev/null ||
      refuse "$ref (${remote_sha:0:8}) is not a commit: foreign"
    [ "$(git -C "$root" rev-list --parents -n 1 "$remote_sha")" = "$remote_sha $h" ] ||
      refuse "$ref's commit is not a child of ${h:0:8} alone: foreign"
    [ "$(git -C "$root" rev-parse "$remote_sha^{tree}")" = "$(git -C "$root" rev-parse "$h^{tree}" 2>/dev/null)" ] ||
      refuse "$ref's commit carries another tree than ${h:0:8}'s: foreign"
    got=" " admitted= waived= nlines=0
    while IFS= read -r l; do
      case $l in owed-run:*) ;; *) continue ;; esac
      nlines=$((nlines + 1))
      parse_line "$l" || refuse "$ref holds a line no v1 or $SCHEMA record reads: $l"
      [ "$head" = "$h" ] || refuse "$ref holds a record of another head (${head:0:8}): foreign"
      if [ "$kind" = waiver ]; then
        waived="owed-run: admitted waiver for ${h:0:8} by $by (writer=$writer): $reason"
        continue
      fi
      [ "$exit_" = 0 ] || refuse "$ref holds a run that exited $exit_"
      [ "$clean" = yes ] || refuse "$ref holds a run of a dirty checkout"
      if [ "$kind" = linux-shell ]; then
        [ "$os" = Linux ] || refuse "$ref holds a linux-shell run on OS $os, not Linux"
        linux_targets_cover "$targets" || refuse "$ref holds invalid linux-shell targets '$targets'"
        got="$got$linux_cover "
        admitted+="owed-run: admitted linux-shell targets '$targets' for ${h:0:8} ($source, writer=$writer)"$'\n'
        continue
      fi
      [ "$backend" = cc ] || refuse "$ref holds a run on backend $backend, not cc"
      cmd_cover "$cmd" || refuse "$ref holds a run of '$cmd', which is no full or train- run (a filtered or first-tier run)"
      got="$got$cover "
      admitted+="owed-run: admitted run of '$cmd' for ${h:0:8} ($source, writer=$writer)"$'\n'
    done <<EOF
$(git -C "$root" log -1 --format=%B "$remote_sha")
EOF
    [ "$nlines" -gt 0 ] || refuse "$ref holds no record line"
    if [ -n "$waived" ]; then printf '%s\n' "$waived"; exit 0; fi
    missing=
    for n do
      case $got in *" $n "*) continue ;; esac
      case $n in
        @test/training/train-*) case $got in *' train '*) continue ;; esac ;;
        linux:member:*) case $got in *' linux:harnesses '*) continue ;; esac ;;
      esac
      missing="$missing $n"
    done
    [ -z "$missing" ] || refuse "$ref covers no run of:$missing"
    printf '%s' "$admitted" ;;
  prune)
    remote_ref refs/heads/master || die "prune: could not read $OWED_RUN_URL"
    tip=$remote_sha
    is_sha "$tip" || die "prune: $OWED_RUN_URL has no master"
    if [ $# = 0 ]; then
      ensure_scratch
      rgit ls-remote "$OWED_RUN_URL" 'refs/owed-runs/*' >"$_owed_scratch/out" 2>"$_owed_scratch/err" || die "prune: could not list refs/owed-runs on $OWED_RUN_URL"
      set -- $(awk '{ sub("^refs/owed-runs/", "", $2); print $2 }' "$_owed_scratch/out")
    fi
    fetch_objects refs/heads/master || die "prune: could not fetch master from $OWED_RUN_URL"
    dels=()
    for h do
      is_sha "$h" || die "prune: '$h' is not a full 40-hex commit"
      if git -C "$root" cat-file -e "$h^{commit}" 2>/dev/null && git -C "$root" merge-base --is-ancestor "$h" "$tip"; then
        dels+=(":refs/owed-runs/$h")
      else
        echo "$me: kept refs/owed-runs/$h: ${h:0:8} is not on master at ${tip:0:8} (or not in this checkout)"
      fi
    done
    if [ ${#dels[@]} -gt 0 ]; then
      bounded git -C "$root" push --quiet "$OWED_RUN_URL" "${dels[@]}" || die "prune: the delete push to $OWED_RUN_URL failed"
      echo "$me: pruned ${#dels[@]} record(s) of merged heads on $OWED_RUN_URL"
    fi ;;
  -h | --help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "usage: tools/owed-run.sh record|add|waive|check|prune|url ... (--help)" ;;
esac

#!/usr/bin/env bash
# Hermetic fixtures for tools/merge-build-check.sh: the order of its tiers (format, compile, tests,
# harness) and where each stops the run; what the tests tier builds (the list prepush prints for
# the head side's revisions, never printed; a cross-cutting change's first tier, or nothing, with
# the rest OWED; an incomplete list's members; no list read fails the tier); what the harness tier
# reads (the HEAD side's change, a script or not, a path the base side renamed by its other name;
# a one-parent HEAD, --all and a quoted path run the whole tier), what it narrows to (the
# change's reach, of any group; none when it reaches no member; the whole tier on a failed ask,
# an unrecognized line or an unmapped script), what covers a script no member reaches (#cover:
# an alias of the list the tests tier runs that affected's exact line says names it, never one
# outside that list, an OWED list or a near-miss line), the toolchain fallback through opam env, the exit
# statuses, elapsed seconds on successful and standalone runs, stable logs across failed
# ship-pr runs of different durations, that the build of the base alone
# narrows as the merge does, that this repository's mapping gives shared support the whole tier
# and a manifest edit less, and that
# .github/ship-pr-merge-build runs the script with every tracked path an input. Fake dune, opam,
# formatter gate, pre-push list and harness tier. The #owed legs read the owed-run record
# (tools/owed-run.sh, gh-ocannl-2218) from local bare repositories standing for the PR repository,
# never the real one: a missing, unreadable, foreign, stale, failed, dirty, non-cc, filtered,
# quoted or malformed record each refuses a cross-cutting, over-bound, incomplete or train-
# reaching change before anything is built; a full record, a union of records and a waiver
# admit; the base alone never prints the merge's refusal line; and `record` certifies only a
# finished, clean, cc run of an allowlisted command, `prune` only merged heads. The reader
# prints admitted writers, including legacy, and rejects malformed writers. The fleet owed-record
# fixture supplies the invocation and last-run receipt to the real hook. The real merge gate reads
# the published ref from a local bare repository and prints its writer for one stubbed train alias.
# Structured train-name rows require a count matching distinct names and an alias the
# tree has; legacy count-only output still asks affected and must agree exactly.
# The branch plan and --plan-merge use the candidate merge and the real record check,
# distinguishing conflict, narrow reach, and wide reach with or without a record, and emit
# ship-pr's single `ship-pr-merge-plan:` verdict line (`nothing-owed`, `admitted <needs>`, or
# `unadmitted <needs>` on definitive exit 1, and none on transport-unreadable exit 3, other error
# or `--plan` alone). Push retries in `tools/owed-run.sh` (gh-ocannl-2436) are pinned for transient
# remote rejections, concurrent `--force-with-lease` contention, exhaustion, exponential backoff,
# immediate stop on auth/authorization errors, and fleet hook retry recovery. The #owed-run
# transport fixtures exercise the real deadline and trigger its alarm only after each
# TERM-resistant helper is ready, then require that helper to stop.
# The #plan-locale leg
# exercises CROSS-CUTTING plans with LANG and LC_ALL unset or empty under LC_CTYPE=UTF-8
# (gh-ocannl-2462), pinning the exported LANG=C guard, caller-locale preservation, and C-locale
# reason filtering. The #reach-cache leg (gh-ocannl-2488) pins test-reach reuse across repeated
# `--plan --base`, `--plan-merge` and full merge builds while checking owed-run records afresh,
# fail-closed invalidation on candidate/tree/base-alone/dirty/ignored-file/Dune/opam/env/OS
# changes, rejection of corrupted/symlinked/group-writable cache files, and default cache paths.
# The #compile-output leg alone uses real
# dune and Python in a minimal isolated project: the repository's actual capped-fixture rule
# runs twice, then with a planted failing case. It compares bytes and rejects the exact unittest
# timing line; tracebacks and FAILED must survive. It does not scan Dune or Python source text.
# The #compile-only leg checks `--compile-only` against fake dune/opam (skipping format, tests,
# owed-run and harness tiers, timings on success, stable red logs for base-alone comparison, opam fallback,
# one-parent HEAD, and usage refusals with incompatible flags). The #compile-only-real leg runs
# `--compile-only` through real Dune on a minimal OCaml project modelled on staging #2170 x #2173:
# a compatible moved base passes without running the other tiers, while an incompatible union
# fails `@check` and its base alone passes. The #sys-path-pin legs (gh-ocannl-2496) verify that
# SYS_PATH resets preserve the caller's active `bash` and `awk` shims (a two-tool allowlist
# carrying tools/test-harnesses.sh's Bash 3.2 and macOS mawk pin) without leaking other tools.
# affected-reads: benchmarks/dune benchmarks/*.py benchmarks/test/*.py benchmarks/runners/*.py benchmarks/runners/*/*.py benchmarks/ocannl_config --only '#compile-output'
# affected-reads: tools/owed-run.sh
# affected-reads: .github/fleet-leg tools/owed-run-report.sh --only '#owed-record|#linux-shell'
# Usage: tools/test-merge-build-check.sh [--keep] [--only REGEX]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require git
harness_scratch test-merge-build-check
HERE=$(cd "$(dirname "$0")" && pwd)
# The fixtures' own git: no ambient configuration, identity or repository selection.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR GIT_CEILING_DIRECTORIES SHIP_PR_MERGE_BASE_SHA SHIP_PR_MERGE_HEAD_SHA \
  OCANNL_TOOL_OWED_RUN_FETCH_CAP OCANNL_TOOL_OWED_RUN_PUSH_RETRIES OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF
# Keep fake-env fixture legs isolated from each other and from the user's real reach cache; the
# dedicated #reach-cache leg overrides or unsets this explicitly.
export OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=off
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid \
  GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid
# resolve_exec_path PATH: normalize a command path returned by `type -P` to an absolute path
# using physical directory traversal (`cd -P`/`pwd -P`) so relative PATH entries crossing a
# symlink before `..` resolve to the same directory as kernel execve.
resolve_exec_path() {
  local p=$1 pdir
  case $p in
    /*) printf '%s' "$p" ;;
    *)
      pdir=$(CDPATH= cd -P -- "$(dirname -- "$p")" && pwd -P) || return 2
      printf '%s/%s' "$pdir" "${p##*/}" ;;
  esac
}
# build_sys_path DIR: populate DIR with symlinks to the caller's active bash and awk (a two-tool
# allowlist preserving tools/test-harnesses.sh's private Bash 3.2 + macOS mawk interpreter
# directory across SYS_PATH resets without leaking dune, opam, ocamlc or git into isolated runs),
# and set SYS_PATH to DIR:/usr/bin:/bin (or /usr/bin:/bin when neither resolved).
build_sys_path() {
  local dir=$1 tool p any=0
  mkdir -p "$dir" || return 2
  dir=$(CDPATH= cd -P -- "$dir" && pwd -P) || return 2
  rm -f "$dir/bash" "$dir/awk" || return 2
  for tool in bash awk; do
    p=$(type -P "$tool" 2>/dev/null || :)
    if [ -n "$p" ] && [ -f "$p" ] && [ -x "$p" ]; then
      p=$(resolve_exec_path "$p") || return 2
      ln -s -- "$p" "$dir/$tool" || return 2
      any=1
    fi
  done
  if [ "$any" = 1 ]; then
    SYS_PATH=$dir:/usr/bin:/bin
  else
    SYS_PATH=/usr/bin:/bin
  fi
}
build_sys_path "$TMP/pinbin" || exit 2
GIT=$(command -v git)

cleanup_fixture() {
  local pidfile pid cmd
  for pidfile in "$TMP"/hang-*.pid; do
    [ -f "$pidfile" ] || continue
    IFS= read -r pid <"$pidfile" || continue
    case $pid in '' | *[!0-9]* | 0*) continue ;; esac
    cmd=$(ps -ww -p "$pid" -o args= 2>/dev/null) || continue
    case $cmd in *" $pidfile") kill -KILL "$pid" 2>/dev/null || : ;; esac
  done
}

mkdir -p "$TMP/dunebin" "$TMP/opambin" "$TMP/emptyopam" "$TMP/linuxbin" "$TMP/darwinbin"
printf '#!/bin/sh\necho Linux\n' >"$TMP/linuxbin/uname"
printf '#!/bin/sh\necho Darwin\n' >"$TMP/darwinbin/uname"
chmod +x "$TMP/linuxbin/uname" "$TMP/darwinbin/uname"
# The tests tier's build is the one that names a width (-j); FAKE_TESTS_RC fails it alone.
cat >"$TMP/dunebin/dune" <<'FAKE'
#!/bin/sh
if [ "$1 $2" = 'show aliases' ]; then
  case $* in
    'show aliases --root . test/training')
      printf '%s\n' train-a train-b
      exit 0 ;;
    'show aliases --root . test/operations')
      printf '%s\n' runtest-sweep_harness runtest-sweep_components runtest-sweep_leg_scripts runtest-prepush_scan_membership
      exit 0 ;;
    *) exit 2 ;;
  esac
fi
printf '%s\n' "$*" >>"$FAKE_RECORD"
if [ "${FAKE_TERM_DURING_DUNE:-0}" = 1 ]; then
  kill -TERM "$PPID"
  exit 0
fi
sleep "${FAKE_TIER_DELAY:-0}"
case " $* " in *' -j '*) exit "${FAKE_TESTS_RC:-${FAKE_DUNE_RC:-0}}" ;; esac
exit "${FAKE_DUNE_RC:-0}"
FAKE
# `opam env --shell=sh` puts the fake dune on PATH, as a switch's does.
cat >"$TMP/opambin/opam" <<FAKE
#!/bin/sh
[ "\$*" = 'env --shell=sh' ] || { echo "unexpected opam call: \$*" >&2; exit 64; }
[ -z "\${FAKE_OPAM_ROOT:-}" ] || [ "\$(pwd -P)" = "\$FAKE_OPAM_ROOT" ] || exit 42
echo "PATH='$TMP/dunebin':\"\\\$PATH\"; export PATH;"
FAKE
# An opam whose switch has no dune.
cat >"$TMP/emptyopam/opam" <<'FAKE'
#!/bin/sh
echo ':'
FAKE
chmod +x "$TMP/dunebin/dune" "$TMP/opambin/opam" "$TMP/emptyopam/opam"
mkdir -p "$TMP/gitbin" && ln -s "$GIT" "$TMP/gitbin/git"

# fixture NAME: a repository with the script, opam-env.sh and a fake harness tier, one commit on
# `base`, and `side` (the PR head) branched from it.
fixture() {
  R=$TMP/$1
  mkdir -p "$R/tools/sub"
  cp "$HERE/merge-build-check.sh" "$HERE/opam-env.sh" "$HERE/owed-run.sh" "$R/tools/"
  # The ask (--list) answers each path after --files from FAKE_REACH's `PATH<TAB>COMMAND` lines,
  # no such file reaching nothing, or fails: always with FAKE_ASK_RC, or naming FAKE_ASK_FAILS_ON.
  cat >"$R/tools/test-harnesses.sh" <<'FAKE'
#!/bin/sh
if [ "$*" = '--list' ]; then
  [ -n "${FAKE_MANIFEST:-}" ] && [ -f "$FAKE_MANIFEST" ] || exit 1
  cat "$FAKE_MANIFEST"
  exit 0
fi
if [ "$*" = '--toolchain --list' ]; then
  [ -n "${FAKE_TOOLCHAIN_MANIFEST:-}" ] && [ -f "$FAKE_TOOLCHAIN_MANIFEST" ] && cat "$FAKE_TOOLCHAIN_MANIFEST"
  exit 0
fi
case " $* " in
  *' --list '*)
    [ "${FAKE_ASK_RC:-0}" = 0 ] || exit "$FAKE_ASK_RC"
    while [ "$1" != --files ]; do shift; done
    shift
    for p do
      [ "$p" != "${FAKE_ASK_FAILS_ON:-}" ] || exit 2
      [ ! -f "${FAKE_REACH:-}" ] || awk -F '\t' -v p="$p" '$1 == p { print $2 }' "$FAKE_REACH"
    done
    exit 0 ;;
esac
printf 'shell %s\n' "$*" >>"$FAKE_RECORD"
sleep "${FAKE_TIER_DELAY:-0}"
if [ -n "${FAKE_MEMBER_TIMING:-}" ] && [ -n "${OCANNL_TOOL_HARNESS_TIMINGS_FILE:-}" ]; then
  printf '%s\n' "$FAKE_MEMBER_TIMING" >>"$OCANNL_TOOL_HARNESS_TIMINGS_FILE"
fi
if [ "${FAKE_HARNESS_EMIT:-0}" = 1 ]; then
  if [ "${FAKE_HARNESS_SKIP_ONLY:-0}" = 1 ]; then
    printf 'HARNESS tools/test-one.sh (shell; fake)\nSKIP  all legs skipped\nall legs passed (1 skipped)\nHARNESS tools/test-one.sh exit: 0\n'
  else
    printf 'HARNESS tools/test-one.sh (shell; fake)\nPASS  ok\nall legs passed (0 skipped)\nHARNESS tools/test-one.sh exit: 0\n'
  fi
fi
case " $* " in *' --affected '*) exit "${FAKE_SCOPED_RC:-0}" ;; esac
exit "${FAKE_SHELL_RC:-0}"
FAKE
  chmod +x "$R/tools/test-harnesses.sh"
  # The formatter gate and the pre-push list, recorded apart from dune and the harness tier
  # ($FAKE_RECORD.tiers), so the records the selection legs compare stay what they were. The list
  # is FAKE_LIST under the verdict FAKE_VERDICT and the status FAKE_PREPUSH_RC; the first tier is
  # FAKE_TIER_LIST under FAKE_TIER_VERDICT and FAKE_TIER_RC (its ask repeats the list's trailer
  # first, as prepush's does). Each ask counts FAKE_TRAIN_N train- members left off (0; `none`
  # prints no count), and `affected` answers FAKE_AFFECTED under FAKE_AFF_VERDICT and FAKE_AFF_RC.
  # Both asks print FAKE_NAMED's lines on stderr (affected's literal-dependency lines). By default
  # the change reaches no per-test alias.
  cat >"$R/tools/fmt-check.sh" <<'FAKE'
#!/bin/sh
printf 'fmt scans=%s\n' "${OCANNL_TOOL_FMT_CHECK_SCANS-unset}" >>"$FAKE_RECORD.tiers"
exit "${FAKE_FMT_RC:-0}"
FAKE
  cat >"$R/tools/test-run.sh" <<'FAKE'
#!/bin/sh
if [ "$*" = 'paths run last' ]; then
  printf '%s\n' "${FAKE_LAST_RUN:-}"
  exit
fi
case $1 in
  run)
    printf 'test-run %s\n' "$*" >>"$FAKE_RECORD"
    exit "${FAKE_TEST_RUN_RC:-0}" ;;
esac
printf '%s\n' "$*" >>"$FAKE_RECORD.tiers"
if [ -n "${FAKE_MUTATE_FILE:-}" ]; then
  printf 'mutated-mid-query\n' >>"$FAKE_MUTATE_FILE"
fi
if [ -n "${FAKE_PLAN_BASE:-}" ]; then
  [ "$(git rev-parse 'HEAD^1')" = "$FAKE_PLAN_BASE" ] &&
    [ "$(git rev-parse 'HEAD^2')" = "$FAKE_PLAN_HEAD" ] &&
    [ "$(git rev-parse --show-toplevel)" != "$FAKE_PLAN_CALLER" ] || exit 7
fi
if [ "${FAKE_CROSS_ON_MOVED_BASE:-0}" = 1 ] && [ -f moved-marker ]; then
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting
fi
if [ "${FAKE_BAD_RULES:-0}" = 1 ]; then
  raw="${TMPDIR:-/tmp}/affected-dune-rules.$$.txt"
  printf '%s\n' 'raw rejected Dune answer' >"$raw" || exit 8
  printf 'affected: rejected Dune answer, saved to %s\n' "$raw" >&2
  exit 2
fi
reason() {
  [ -z "${FAKE_REASON:-}" ] || printf 'affected: CROSS-CUTTING reason: %s\n' "$FAKE_REASON" >&2
  [ -z "${FAKE_NAMED:-}" ] || cat "$FAKE_NAMED" >&2
}
count() {
  if [ -n "${FAKE_TRAIN_NAMES:-}" ]; then
    while IFS= read -r alias; do printf 'test-run: prepush: train: %s\n' "$alias" >&2; done <"$FAKE_TRAIN_NAMES"
  fi
  [ "${FAKE_TRAIN_N:-0}" = none ] ||
    printf 'test-run: prepush: 1 of the 7 alias(es) the change lists; left off: %s train- (the nightly CI run) and 0 slow-\n' "${FAKE_TRAIN_N:-0}" >&2
}
case " $* " in
  ' affected --files '*)
    [ -z "${FAKE_RENAME_NAMED:-}" ] || cat "$FAKE_RENAME_NAMED" >&2
    exit "${FAKE_RENAME_RC:-0}" ;;
  ' affected '*)
    cat "${FAKE_AFFECTED:-/dev/null}"
    [ "${FAKE_AFF_VERDICT-ok}" = none ] || printf 'affected: verdict=%s\n' "${FAKE_AFF_VERDICT-ok}" >&2
    exit "${FAKE_AFF_RC:-0}" ;;
  *' --referenced-only '*)
    cat "${FAKE_TIER_LIST:-/dev/null}"
    reason
    printf 'affected: verdict=cross-cutting\n' >&2
    count
    [ "${FAKE_TIER_VERDICT-ok}" = none ] || printf 'affected: verdict=%s\n' "${FAKE_TIER_VERDICT-ok}" >&2
    exit "${FAKE_TIER_RC:-0}" ;;
esac
cat "${FAKE_LIST:-/dev/null}"
reason
printf 'affected: 7 alias(es) from 2 changed path(s)\n' >&2
[ "${FAKE_PREPUSH_RC:-0}" = 2 ] || count
[ "${FAKE_VERDICT-ok}" = none ] || printf 'affected: verdict=%s\n' "${FAKE_VERDICT-ok}" >&2
exit "${FAKE_PREPUSH_RC:-0}"
FAKE
  chmod +x "$R/tools/fmt-check.sh" "$R/tools/test-run.sh"
  for f in tools/sub/a.sh b.sh x.ml r.txt m.txt; do printf '1\n2\n3\n4\n5\n6\n' >"$R/$f"; done
  git -C "$R" init -q && git -C "$R" checkout -q -b base && git -C "$R" add -A &&
    git -C "$R" commit -q -m init && git -C "$R" branch side
}
# first FILE / last FILE: an edit at either end, so both sides' edits merge cleanly.
first() { { echo first; tail -n +2 "$R/$1"; } >"$R/$1.new" && mv "$R/$1.new" "$R/$1"; }
last() { echo last >>"$R/$1"; }
commit_on() { # BRANCH: commit what the caller changed on BRANCH
  git -C "$R" add -A && git -C "$R" commit -q -m "$1 change"
}
merge_side() { # HEAD becomes base merged with side: first parent base, second side
  git -C "$R" checkout -q base && git -C "$R" merge -q --no-ff --no-edit side >/dev/null
}
plan() { # [ARG...]: the selection on R's HEAD
  rc=0
  (cd "$R" && "$BASH" tools/merge-build-check.sh --plan "$@") >"$TMP/out" 2>&1 || rc=$?
}
run() { # PATH_PREFIX [ARG...]: a full run on R's HEAD under a PATH of the prefix and the system
  local prefix=$1
  shift
  rc=0
  : >"$TMP/record"
  : >"$TMP/record.tiers"
  (cd "$R" && PATH=$prefix:$TMP/gitbin:$SYS_PATH FAKE_RECORD=$TMP/record "$BASH" tools/merge-build-check.sh "$@") \
    >"$TMP/out" 2>&1 || rc=$?
}
has() { grep -qF -- "$1" "$TMP/out"; }

# Both sides edit one script, under a nested directory: `*.sh` crosses `/`.
fixture both-script
first tools/sub/a.sh && commit_on base
git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
BOTH_SCRIPT=$R
if harness_leg 'a script the head side changed selects the harness tier #select'; then
  plan
  if [ "$rc" = 0 ] && has 'harness tier: a script changed on the head side -- tools/sub/a.sh matches *.sh' &&
     has 'compile tier: dune build @check @test/operations/scans (always)'; then
    report 0 'a script the head side changed selects the harness tier'
  else report 1 'a script the head side changed selects the harness tier' "exit $rc; $(cat "$TMP/out")"; fi
fi

# The base side changes a script, the head side none: the tier reads the head side's change.
fixture apart
first x.ml && first tools/sub/a.sh && commit_on base
git -C "$R" checkout -q side && last x.ml && last m.txt && commit_on side && merge_side
APART=$R
if harness_leg 'a script the base side alone changed is not the change of the head side #select'; then
  plan
  if [ "$rc" = 0 ] && has 'harness tier: no script changed on the head side -- none of the 2 path(s) it changed matches *.sh' &&
     has 'harness tier: skipped -- the 2 path(s) the head side changed reach no harness'; then
    report 0 'the head side changed no script and reaches no harness: the tier is skipped'
  else report 1 'the head side changed no script and reaches no harness: the tier is skipped' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'a rename counts by its other name, and only a script name selects #select'; then
  for target in r.sh r.md; do
    fixture "rename-$target"
    git -C "$R" mv r.txt "$target" && commit_on base
    git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
    plan
    case $target in
      r.sh) want='harness tier: a script changed on the head side -- r.txt was renamed from or to r.sh, which matches *.sh' ;;
      r.md) want='harness tier: no script changed on the head side -- none of the 1 path(s) it changed matches *.sh' ;;
    esac
    if [ "$rc" = 0 ] && has "$want"; then
      report 0 "a rename to $target on the base, an edit of its old name on the head"
    else report 1 "a rename to $target on the base, an edit of its old name on the head" "exit $rc; $(cat "$TMP/out")"; fi
  done
  # The head renames, the base edits the old name: the pair is read from either side.
  fixture rename-head
  last r.txt && commit_on base
  git -C "$R" checkout -q side && git -C "$R" mv r.txt r.sh && commit_on side && merge_side
  plan
  if [ "$rc" = 0 ] && has 'harness tier: a script changed on the head side -- r.sh matches *.sh'; then
    report 0 'a rename on the head counts too'
  else report 1 'a rename on the head counts too' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'what cannot be read runs the whole harness tier #select'; then
  fixture one-parent
  plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- HEAD is not a two-parent merge, so the whole harness tier runs'; then
    report 0 'a one-parent HEAD runs the whole harness tier'
  else report 1 'a one-parent HEAD runs the whole harness tier' "exit $rc; $(cat "$TMP/out")"; fi
  # A name git quotes is one this script does not decode, so it cannot rule the path out.
  fixture quoted
  first x.ml && printf 'x\n' >"$R/q\"uote.txt" && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- git quoted a path one side changed, so the whole harness tier runs'; then
    report 0 'a path git quotes runs the whole harness tier'
  else report 1 'a path git quotes runs the whole harness tier' "exit $rc; $(cat "$TMP/out")"; fi
  R=$APART plan --all
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- --all'; then
    report 0 '--all runs the whole harness tier on a merge that would skip it'
  else report 1 '--all runs the whole harness tier on a merge that would skip it' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'a run builds the selected tiers in order and stops at the first failure #run'; then
  R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "build --root . @check @test/operations/scans
shell --jobs 4" ] && has 'merge-build-check: passed'; then
    report 0 'a selected harness tier runs after the compile tier'
  else report 1 'a selected harness tier runs after the compile tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = 'build --root . @check @test/operations/scans' ]; then
    report 0 'a skipped harness tier does not run'
  else report 1 'a skipped harness tier does not run' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_DUNE_RC=3 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = 'build --root . @check @test/operations/scans' ] &&
     has 'compile tier FAILED (dune exit 3)'; then
    report 0 'a failing compile tier exits 1 before the harness tier'
  else report 1 'a failing compile tier exits 1 before the harness tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SHELL_RC=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && has 'harness tier FAILED (tools/test-harnesses.sh exit 1)' && ! has 'merge-build-check: passed'; then
    report 0 'a failing harness tier exits 1'
  else report 1 'a failing harness tier exits 1' "exit $rc; $(cat "$TMP/out")"; fi
fi

COMPILE='build --root . @check @test/operations/scans'
# The format tier and the tests tier (the local merge gate: what the scheduled `Formatting` job and
# the affected part of the `Build` shards read). Their fakes record in $TMP/record.tiers.
range_of() { # the head side's change, as the script asks it: MERGE_BASE..HEAD_SIDE of R's HEAD
  local b h
  b=$(git -C "$R" rev-parse 'HEAD^1') && h=$(git -C "$R" rev-parse 'HEAD^2') &&
    printf '%s..%s' "$(git -C "$R" merge-base "$b" "$h")" "$h"
}
tiers() { cat "$TMP/record.tiers"; }
# The owed-run records (tools/owed-run.sh): local bare repositories stand for the PR repository,
# so no fixture reads or writes the real one. OWED_EMPTY holds no record; OWED_FULL a full-suite
# and @train run for APART's head.
OWED_EMPTY=$TMP/owed-empty.git OWED_FULL=$TMP/owed-full.git
git init -q --bare "$OWED_EMPTY" && git init -q --bare "$OWED_FULL"
export OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY
owed() { # URL ARG...: tools/owed-run.sh in R against the records at URL; its status in `orc`
  local url=$1
  shift
  orc=0
  (cd "$R" && PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$url "$BASH" tools/owed-run.sh "$@") \
    >"$TMP/owed-out" 2>&1 || orc=$?
}
run_line() { # HEAD CMD [EXIT CLEAN BACKEND]: a run record's line, valid by default
  printf 'owed-run: v1 run head=%s exit=%s clean=%s backend=%s source=fixture:/runs/1 cmd=%s' \
    "$1" "${3:-0}" "${4:-yes}" "${5:-cc}" "$2"
}
APART_HEAD=$(git -C "$APART" rev-parse 'HEAD^2')
R=$APART owed "$OWED_FULL" add "$(run_line "$APART_HEAD" 'build @runtest @train -j 4')"
[ "$orc" = 0 ] || { echo "fixture: could not publish APART's record: $(cat "$TMP/owed-out")" >&2; exit 1; }
if harness_leg 'a branch plan reads the selected merge and its record without building #branch-plan'; then
  fixture branch-plan
  fork=$(git -C "$R" rev-parse HEAD)
  first r.txt && commit_on base
  # The current base can bring its own merge gate changes along with its inputs.
  gate=$R/tools/merge-build-check.sh
  {
    IFS= read -r shebang
    printf '%s\n' "$shebang" "echo 'fixture: merge gate from the moved base'"
    cat
  } <"$gate" >"$gate.new" && chmod +x "$gate.new" && mv "$gate.new" "$gate"
  touch "$R/moved-marker" && commit_on base
  tip=$(git -C "$R" rev-parse HEAD)
  git -C "$R" checkout -q side && last r.txt && commit_on side
  side=$(git -C "$R" rev-parse HEAD)
  owed "$OWED_FULL" add "$(run_line "$side" 'runtest -j 4')"
  [ "$orc" = 0 ] || { echo "fixture: the branch plan record could not be published" >&2; exit 1; }
  # A dirty caller is outside the committed merge. In particular, a caller-side
  # copy of the probe input must not influence the plan against the fork.
  touch "$R/moved-marker"
  before=$(git -C "$R" status --porcelain)
  plan_branch() {
    FAKE_PLAN_CALLER=$R FAKE_PLAN_BASE=$1 FAKE_PLAN_HEAD=$side run "${2:-$TMP/dunebin}" --plan --base "$1"
  }
  planned() {
    if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] && grep -qx "prepush --since $fork..$side --list" "$TMP/record.tiers" &&
       ! grep -q '^fmt ' "$TMP/record.tiers" && [ "$(git -C "$R" rev-parse HEAD)" = "$side" ] &&
       [ "$(git -C "$R" status --porcelain)" = "$before" ] && has "$2" && has "$3" &&
       [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 1 ] && grep -qxF -- "$4" "$TMP/out"; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(tiers); $(cat "$TMP/out")"; fi
  }
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' plan_branch "$tip"
  planned 'a cross-cutting branch plan names the need and missing record' \
    'tests plan: CROSS-CUTTING (first tier selected)' 'owed-run needs: runtest (record does not admit' \
    'ship-pr-merge-plan: unadmitted runtest'
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL plan_branch "$tip"
  planned 'a cross-cutting branch plan reads a record of the head' \
    'tests plan: CROSS-CUTTING (first tier selected)' 'owed-run needs: runtest (record admits)' \
    'ship-pr-merge-plan: admitted runtest'
  FAKE_CROSS_ON_MOVED_BASE=1 plan_branch "$fork"
  planned 'a narrow branch plan owes no record' 'tests plan: narrow' 'owed-run needs: none' \
    'ship-pr-merge-plan: nothing-owed'
  FAKE_CROSS_ON_MOVED_BASE=1 plan_branch "$tip"
  planned 'the same head is cross-cutting with no first tier against the moved base' \
    'tests plan: CROSS-CUTTING (no first tier selected)' 'owed-run needs: runtest (record does not admit' \
    'ship-pr-merge-plan: unadmitted runtest'
  if has 'fixture: merge gate from the moved base'; then
    report 0 'the candidate plan uses the merge gate version the base brought'
  else report 1 'the candidate plan uses the merge gate version the base brought' "$(cat "$TMP/out")"; fi
  FAKE_OPAM_ROOT=$(cd "$R" && pwd -P) plan_branch "$fork" "$TMP/opambin"
  planned 'a branch plan resolves checkout-local opam before querying the candidate' \
    'dune is not on PATH; sourcing tools/opam-env.sh' 'owed-run needs: none' \
    'ship-pr-merge-plan: nothing-owed'
  FAKE_BAD_RULES=1 TMPDIR=$TMP plan_branch "$fork"
  raw=$(sed -n 's/^affected: rejected Dune answer, saved to //p' "$TMP/out")
  if [ "$rc" = 1 ] && [ -n "$raw" ] && grep -qx 'raw rejected Dune answer' "$raw" &&
     [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ]; then
    report 0 'the recorded raw Dune answer survives plan cleanup, with no verdict line'
  else report 1 'the recorded raw Dune answer survives plan cleanup, with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
  # The side needs a conflicting edit to the same line, relative to the fork.
  { echo conflict; tail -n +2 "$R/r.txt"; } >"$R/r.txt.new" && mv "$R/r.txt.new" "$R/r.txt" && commit_on side
  side=$(git -C "$R" rev-parse HEAD)
  plan_branch "$tip"
  if [ "$rc" = 1 ] && has 'plan: merge conflict' && has r.txt &&
     [ ! -s "$TMP/record" ] && [ ! -s "$TMP/record.tiers" ] &&
     [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ]; then
    report 0 'a conflicting branch plan reports the conflict before any tier is asked, with no verdict line'
  else report 1 'a conflicting branch plan reports the conflict before any tier is asked, with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
fi
if harness_leg 'merged_plan prints ship-pr-merge-plan verdicts and omits them on error #plan-verdict'; then
  plan_verdict_ok() { # LABEL WANT_LINE: exit 0, nothing built, exactly one verdict line matching WANT_LINE
    if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] && ! grep -q '^fmt ' "$TMP/record.tiers" &&
       [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 1 ] && grep -qxF -- "$2" "$TMP/out"; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  }
  R=$APART plan
  if [ "$rc" = 0 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ]; then
    report 0 '--plan alone asks no record and prints no ship-pr-merge-plan line'
  else report 1 '--plan alone asks no record and prints no ship-pr-merge-plan line' "exit $rc; $(cat "$TMP/out")"; fi
  R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge on a narrow merge prints nothing-owed' 'ship-pr-merge-plan: nothing-owed'
  printf '%s\n' '@test/training/train-a' >"$TMP/plan-train"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/plan-train \
    R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge without a record prints unadmitted with every need' \
    'ship-pr-merge-plan: unadmitted runtest @test/training/train-a'
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/plan-train \
    R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge with an admitting record prints admitted with every need' \
    'ship-pr-merge-plan: admitted runtest @test/training/train-a'
  plan_waivers=$TMP/plan-waivers.git
  git init -q --bare "$plan_waivers"
  R=$APART owed "$plan_waivers" waive --head "$APART_HEAD" --reason 'waived for plan verdict fixture'
  OCANNL_TOOL_OWED_RUN_URL=$plan_waivers FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    R=$APART run "$TMP/dunebin" --plan-merge
  if [ "$orc" = 0 ]; then
    plan_verdict_ok '--plan-merge with a waiver prints admitted' 'ship-pr-merge-plan: admitted runtest'
  else report 1 '--plan-merge with a waiver prints admitted' "waive exit $orc: $(cat "$TMP/owed-out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=3 FAKE_VERDICT=incomplete \
    R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge on an incomplete list without a record prints unadmitted runtest train' \
    'ship-pr-merge-plan: unadmitted runtest train'
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_PREPUSH_RC=3 FAKE_VERDICT=incomplete \
    R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge on an incomplete list with a full record prints admitted runtest train' \
    'ship-pr-merge-plan: admitted runtest train'
  OCANNL_TOOL_OWED_RUN_URL=$TMP/nowhere.git FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    R=$APART run "$TMP/dunebin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'owed-run needs: runtest (record unreadable; owed-run.sh exit 3)' &&
     has 'plan: owed-run.sh check could not read the record repository (exit 3)'; then
    report 0 '--plan-merge on an unreadable record repository exits 1 with no verdict line'
  else report 1 '--plan-merge on an unreadable record repository exits 1 with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
  mkdir -p "$TMP/fetchfailbin"
  ln -sf "$TMP/dunebin/dune" "$TMP/fetchfailbin/dune"
  cat >"$TMP/fetchfailbin/git" <<FAKEGIT
#!/bin/sh
for a do
  case \$a in
    fetch) echo 'fatal: unable to access remote: Connection reset by peer' >&2; exit 128 ;;
  esac
done
exec "$GIT" "\$@"
FAKEGIT
  chmod +x "$TMP/fetchfailbin/git"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    R=$APART run "$TMP/fetchfailbin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'could not be fetched within 60s: unreadable' &&
     has 'plan: owed-run.sh check could not read the record repository (exit 3)'; then
    report 0 '--plan-merge on a fetch-phase transport failure exits 1 with no verdict line'
  else report 1 '--plan-merge on a fetch-phase transport failure exits 1 with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
  noncommit_repo=$TMP/plan-noncommit.git
  git init -q --bare "$noncommit_repo"
  blob_sha=$(printf 'not a commit\n' | git -C "$APART" hash-object -w --stdin)
  git -C "$APART" push -q -f "$noncommit_repo" "$blob_sha:refs/owed-runs/$APART_HEAD"
  OCANNL_TOOL_OWED_RUN_URL=$noncommit_repo FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    R=$APART run "$TMP/dunebin" --plan-merge
  plan_verdict_ok '--plan-merge on a non-commit ref prints unadmitted (definitive foreign refusal)' \
    'ship-pr-merge-plan: unadmitted runtest'
  OCANNL_TOOL_OWED_RUN_FETCH_CAP=0 OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    R=$APART run "$TMP/dunebin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'plan: owed-run.sh check failed (exit 2)'; then
    report 0 'an owed-run.sh usage failure exits 1 with no verdict line'
  else report 1 'an owed-run.sh usage failure exits 1 with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
  fixture plan-empty-owed
  first x.ml && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  printf '#!/bin/sh\nexit 1\n' >"$R/tools/owed-run.sh"
  chmod +x "$R/tools/owed-run.sh"
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting run "$TMP/dunebin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'plan: owed-run.sh check exited 1 with no output'; then
    report 0 'an empty-output exit-1 record check fails the plan with no verdict line'
  else report 1 'an empty-output exit-1 record check fails the plan with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
  fixture plan-one-parent
  run "$TMP/dunebin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'plan: tests tier unreadable: HEAD has no two sides with a merge base'; then
    report 0 '--plan-merge on a one-parent HEAD exits 1 with no verdict line'
  else report 1 '--plan-merge on a one-parent HEAD exits 1 with no verdict line' "exit $rc; $(cat "$TMP/out")"; fi
fi
if harness_leg 'CROSS-CUTTING plans guard an unset locale without breaking C-locale filtering #plan-locale'; then
  fixture plan-locale
  cat >"$R/tools/owed-run.sh" <<'FAKE'
#!/bin/sh
if [ -n "${FAKE_LOCALE_OUT:-}" ]; then
  printf 'LANG=%s LC_ALL=%s LC_CTYPE=%s\n' \
    "${LANG-<unset>}" "${LC_ALL-<unset>}" "${LC_CTYPE-<unset>}" >"$FAKE_LOCALE_OUT"
fi
echo 'owed-run: admitted fixture'
exit 0
FAKE
  chmod +x "$R/tools/owed-run.sh"
  first x.ml && commit_on base
  locale_base=$(git -C "$R" rev-parse HEAD)
  git -C "$R" checkout -q side && last x.ml && commit_on side
  locale_side=$(git -C "$R" rev-parse HEAD)
  merge_side
  run_locale() { # ENV_ARGS... -- PLAN_ARGS...
    local env_args=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
    shift
    rc=0
    : >"$TMP/record"
    : >"$TMP/record.tiers"
    rm -f "$TMP/locale-seen"
    (cd "$R" && env ${env_args[@]+"${env_args[@]}"} PATH="$TMP/dunebin:$TMP/gitbin:$SYS_PATH" \
      FAKE_RECORD="$TMP/record" FAKE_LOCALE_OUT="$TMP/locale-seen" \
      "$BASH" tools/merge-build-check.sh "$@") >"$TMP/out" 2>&1 || rc=$?
  }
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' \
    run_locale -u LANG -u LC_ALL LC_CTYPE=UTF-8 -- --plan-merge
  if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] &&
     has 'tests plan: CROSS-CUTTING (first tier selected)' &&
     grep -qxF 'merge-build-check: tests tier: CROSS-CUTTING reason: x.ml (6 aliases)' "$TMP/out" &&
     has 'owed-run needs: runtest (record admits)' &&
     grep -qxF 'ship-pr-merge-plan: admitted runtest' "$TMP/out" &&
     [ "$(cat "$TMP/locale-seen" 2>/dev/null)" = 'LANG=C LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 '--plan-merge on a CROSS-CUTTING merge with unset LANG/LC_ALL prints reason and verdict and exports LANG=C'
  else report 1 '--plan-merge on a CROSS-CUTTING merge with unset LANG/LC_ALL prints reason and verdict and exports LANG=C' \
    "exit $rc; locale=$(cat "$TMP/locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' \
    run_locale LANG= LC_ALL= LC_CTYPE=UTF-8 -- --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: admitted runtest' "$TMP/out" &&
     [ "$(cat "$TMP/locale-seen" 2>/dev/null)" = 'LANG=C LC_ALL= LC_CTYPE=UTF-8' ]; then
    report 0 '--plan-merge with empty LANG/LC_ALL exports LANG=C while keeping LC_CTYPE'
  else report 1 '--plan-merge with empty LANG/LC_ALL exports LANG=C while keeping LC_CTYPE' \
    "exit $rc; locale=$(cat "$TMP/locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  non_ascii_reason=$(printf 'x\303\251.ml (6 aliases)')
  utf8_ctype=
  for cand in UTF-8 C.UTF-8 en_US.UTF-8; do
    if env -u LANG -u LC_ALL "LC_CTYPE=$cand" grep -qx '[[:print:]][[:print:]]*' <<<"$non_ascii_reason" 2>/dev/null; then
      utf8_ctype=$cand
      break
    fi
  done
  if [ -z "$utf8_ctype" ]; then
    skip 'show_cross_reason still rejects non-ASCII bytes under a UTF-8 LC_CTYPE' \
      'no UTF-8 LC_CTYPE candidate (UTF-8, C.UTF-8, en_US.UTF-8) makes unforced grep accept non-ASCII bytes on this host'
  else
    FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON=$non_ascii_reason \
      run_locale -u LANG -u LC_ALL "LC_CTYPE=$utf8_ctype" -- --plan-merge
    if [ "$rc" = 0 ] && ! has 'CROSS-CUTTING reason:' &&
       grep -qxF 'ship-pr-merge-plan: admitted runtest' "$TMP/out"; then
      report 0 'show_cross_reason still rejects non-ASCII bytes under a UTF-8 LC_CTYPE'
    else report 1 'show_cross_reason still rejects non-ASCII bytes under a UTF-8 LC_CTYPE' \
      "LC_CTYPE=$utf8_ctype; exit $rc; $(cat "$TMP/out")"; fi
  fi
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' \
    run_locale -u LC_ALL LANG=en_US.UTF-8 LC_CTYPE=UTF-8 -- --plan-merge
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/locale-seen" 2>/dev/null)" = 'LANG=en_US.UTF-8 LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 'a caller-supplied LANG is preserved untouched'
  else report 1 'a caller-supplied LANG is preserved untouched' \
    "exit $rc; locale=$(cat "$TMP/locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' \
    run_locale -u LANG LC_ALL=POSIX LC_CTYPE=UTF-8 -- --plan-merge
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/locale-seen" 2>/dev/null)" = 'LANG=<unset> LC_ALL=POSIX LC_CTYPE=UTF-8' ]; then
    report 0 'a caller-supplied LC_ALL is preserved untouched'
  else report 1 'a caller-supplied LC_ALL is preserved untouched' \
    "exit $rc; locale=$(cat "$TMP/locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  git -C "$R" checkout -q "$locale_side"
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' \
    run_locale -u LANG -u LC_ALL LC_CTYPE=UTF-8 -- --plan --base "$locale_base"
  if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] &&
     has 'tests plan: CROSS-CUTTING (first tier selected)' &&
     grep -qxF 'merge-build-check: tests tier: CROSS-CUTTING reason: x.ml (6 aliases)' "$TMP/out" &&
     has 'owed-run needs: runtest (record admits)' &&
     grep -qxF 'ship-pr-merge-plan: admitted runtest' "$TMP/out" &&
     [ "$(cat "$TMP/locale-seen" 2>/dev/null)" = 'LANG=C LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 '--plan --base on a CROSS-CUTTING branch with unset LANG/LC_ALL prints reason and verdict and exports LANG=C'
  else report 1 '--plan --base on a CROSS-CUTTING branch with unset LANG/LC_ALL prints reason and verdict and exports LANG=C' \
    "exit $rc; locale=$(cat "$TMP/locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  # Standalone tools/owed-run.sh unset-locale guard (gh-ocannl-2484): leg-record
  # runs record_linux_shell inside a forked subshell calling local LC_ALL=C
  # helpers and git. Run from a fresh clean fixture (where fixture() committed
  # the real tools/owed-run.sh) and require publication in all four cases.
  git -C "$R" checkout -q base
  fixture owed-locale
  owed_locale_bin="$TMP/owed-locale-bin"
  mkdir -p "$owed_locale_bin" "$TMP/owed-locale-receipt"
  real_git=$(command -v git)
  cat >"$owed_locale_bin/git" <<EOF
#!/bin/sh
if [ -n "\${OWED_LOCALE_OUT:-}" ] && [ ! -e "\$OWED_LOCALE_OUT" ]; then
  printf 'LANG=%s LC_ALL=%s LC_CTYPE=%s\n' "\${LANG-<unset>}" "\${LC_ALL-<unset>}" "\${LC_CTYPE-<unset>}" >"\$OWED_LOCALE_OUT"
fi
exec "$real_git" "\$@"
EOF
  chmod +x "$owed_locale_bin/git"
  owed_bare="$TMP/owed-locale-bare.git"
  git init -q --bare "$owed_bare"
  owed_head=$(git -C "$R" rev-parse HEAD)
  printf 'linux-shell\n' >"$TMP/owed-locale-receipt/kind"
  printf '%s\n' "$owed_head" >"$TMP/owed-locale-receipt/head"
  : >"$TMP/owed-locale-receipt/dirty"
  printf 'Linux\n' >"$TMP/owed-locale-receipt/os"
  printf 'member:tools/test-promote.sh\n' >"$TMP/owed-locale-receipt/targets"
  printf '0\n' >"$TMP/owed-locale-receipt/exit"
  printf '/tmp/owed-locale-leg\n' >"$TMP/owed-locale-receipt/fleet-leg-run-record"
  printf 'member:tools/test-promote.sh\n' >"$TMP/owed-locale-receipt/fleet-leg-targets"
  run_owed_locale() { # ENV_ARGS... -- OWED_ARGS...
    local env_args=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
    shift
    rc=0
    rm -f "$TMP/owed-locale-seen"
    (cd "$R" && env ${env_args[@]+"${env_args[@]}"} PATH="$owed_locale_bin:$SYS_PATH" \
      OWED_LOCALE_OUT="$TMP/owed-locale-seen" \
      FLEET_LEG_RECEIPT_DIR="$TMP/owed-locale-receipt" \
      OCANNL_TOOL_OWED_RUN_URL="$owed_bare" \
      OCANNL_TOOL_OWED_RUN_PUSH_RETRIES=0 OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 \
      "$BASH" tools/owed-run.sh "$@") >"$TMP/out" 2>&1 || rc=$?
  }
  run_owed_locale -u LANG -u LC_ALL LC_CTYPE=UTF-8 -- leg-record /tmp/owed-locale-leg req-1 "$owed_head"
  if [ "$rc" = 0 ] && grep -q 'owed-run: published refs/owed-runs/' "$TMP/out" &&
     [ "$(cat "$TMP/owed-locale-seen" 2>/dev/null)" = 'LANG=C LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 'standalone owed-run.sh leg-record with unset LANG/LC_ALL publishes from its subshell and exports LANG=C'
  else report 1 'standalone owed-run.sh leg-record with unset LANG/LC_ALL publishes from its subshell and exports LANG=C' \
    "exit $rc; locale=$(cat "$TMP/owed-locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  run_owed_locale LANG= LC_ALL= LC_CTYPE=UTF-8 -- leg-record /tmp/owed-locale-leg req-1 "$owed_head"
  if [ "$rc" = 0 ] && grep -q 'owed-run: published refs/owed-runs/' "$TMP/out" &&
     [ "$(cat "$TMP/owed-locale-seen" 2>/dev/null)" = 'LANG=C LC_ALL= LC_CTYPE=UTF-8' ]; then
    report 0 'standalone owed-run.sh with empty LANG/LC_ALL publishes and exports LANG=C while keeping LC_CTYPE'
  else report 1 'standalone owed-run.sh with empty LANG/LC_ALL publishes and exports LANG=C while keeping LC_CTYPE' \
    "exit $rc; locale=$(cat "$TMP/owed-locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  run_owed_locale -u LC_ALL LANG=en_US.UTF-8 LC_CTYPE=UTF-8 -- leg-record /tmp/owed-locale-leg req-1 "$owed_head"
  if [ "$rc" = 0 ] && grep -q 'owed-run: published refs/owed-runs/' "$TMP/out" &&
     [ "$(cat "$TMP/owed-locale-seen" 2>/dev/null)" = 'LANG=en_US.UTF-8 LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 'standalone owed-run.sh publishes and preserves caller-supplied LANG'
  else report 1 'standalone owed-run.sh publishes and preserves caller-supplied LANG' \
    "exit $rc; locale=$(cat "$TMP/owed-locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
  run_owed_locale -u LANG LC_ALL=POSIX LC_CTYPE=UTF-8 -- leg-record /tmp/owed-locale-leg req-1 "$owed_head"
  if [ "$rc" = 0 ] && grep -q 'owed-run: published refs/owed-runs/' "$TMP/out" &&
     [ "$(cat "$TMP/owed-locale-seen" 2>/dev/null)" = 'LANG=<unset> LC_ALL=POSIX LC_CTYPE=UTF-8' ]; then
    report 0 'standalone owed-run.sh publishes and preserves caller-supplied LC_ALL'
  else report 1 'standalone owed-run.sh publishes and preserves caller-supplied LC_ALL' \
    "exit $rc; locale=$(cat "$TMP/owed-locale-seen" 2>/dev/null); $(cat "$TMP/out")"; fi
fi
if harness_leg 'test-reach cache reuses matching candidate reach and checks owed records afresh #reach-cache'; then
  fixture reach-cache
  printf 'fixture-ignored.cfg\n' >"$R/.gitignore"
  commit_on base
  rc_base=$(git -C "$R" rev-parse HEAD)
  git -C "$R" checkout -q side
  last x.ml
  last tools/sub/a.sh
  commit_on side
  rc_head=$(git -C "$R" rev-parse HEAD)

  rc_owed=$TMP/reach-cache-owed.git
  git init -q --bare "$rc_owed"
  rc_cache=$TMP/reach-cache-store
  rc_xdg=$TMP/reach-cache-xdg
  mkdir -p "$rc_xdg/dune"
  printf '(lang dune 3.20)\n' >"$rc_xdg/dune/config"
  rc_ws=$TMP/reach-cache-ws.dune
  printf '(lang dune 3.20)\n' >"$rc_ws"
  rc_switch=$TMP/reach-cache-switch
  rc_bin=$rc_switch/bin
  mkdir -p "$rc_switch/.opam-switch" "$rc_switch/lib/pkg_a" "$rc_switch/lib/ocaml" "$rc_bin"
  printf 'installed: ["dune.3.24.2"]\n' >"$rc_switch/.opam-switch/switch-state"
  printf 'version = "1.0"\n' >"$rc_switch/lib/pkg_a/META"
  printf 'destdir="."\npath="./ocaml:."\n' >"$rc_switch/lib/findlib.conf"
  printf 'version = "5.5.1"\n' >"$rc_switch/lib/ocaml/META.stdlib"
  cp "$TMP/dunebin/dune" "$rc_bin/dune"
  chmod +x "$rc_bin/dune"

  printf '%s\n' '@test/training/train-a' >"$TMP/rc-train"
  printf '%s\n' '@a/runtest-x' '@test/operations/runtest-cover' >"$TMP/rc-tier-list"
  printf '%s\n' 'affected: tools/sub/a.sh: named as a dependency by the rule of @test/operations/runtest-cover' >"$TMP/rc-named"

  rc_run() { # BIN_PREFIX [ARG...]
    local prefix=$1
    shift
    OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=$rc_cache \
      XDG_CONFIG_HOME=$rc_xdg OPAM_SWITCH_PREFIX=$rc_switch \
      OCANNL_TOOL_OWED_RUN_URL=${RC_OWED_URL:-$rc_owed} \
      FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
      FAKE_TIER_LIST=$TMP/rc-tier-list FAKE_NAMED=$TMP/rc-named \
      FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/rc-train \
      FAKE_REASON='x.ml (6 aliases)' \
      run "$prefix" "$@"
  }
  rc_asked_prepush() { grep -q '^prepush ' "$TMP/record.tiers"; }

  # 1. Pre-push --plan --base misses cache and reports unadmitted.
  rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base"
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach' &&
     has 'merge-build-check: tests tier: CROSS-CUTTING reason: x.ml (6 aliases)' &&
     has 'harness tier: tools/sub/a.sh is named as a dependency by @test/operations/runtest-cover, which the tests tier runs' &&
     grep -qxF 'ship-pr-merge-plan: unadmitted runtest @test/training/train-a' "$TMP/out"; then
    report 0 'first --plan --base computes reach, populates cache and reports unadmitted'
  else report 1 'first --plan --base computes reach, populates cache and reports unadmitted' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi

  # Publish the owed-run record to the SAME bare repository (pre-push -> record -> post-push).
  owed "$rc_owed" add "$(run_line "$rc_head" 'build @runtest @train -j 4')"
  [ "$orc" = 0 ] || { echo "fixture: reach-cache record could not be published" >&2; exit 1; }

  # 2. Post-push --plan --base reuses cached reach without asking prepush, checks record afresh.
  rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base"
  if [ "$rc" = 0 ] && ! rc_asked_prepush && has 'merge-build-check: tests plan: reused cached reach' &&
     has 'merge-build-check: tests tier: CROSS-CUTTING reason: x.ml (6 aliases)' &&
     has 'harness tier: tools/sub/a.sh is named as a dependency by @test/operations/runtest-cover, which the tests tier runs' &&
     grep -qxF 'ship-pr-merge-plan: admitted runtest @test/training/train-a' "$TMP/out"; then
    report 0 'second --plan --base reuses cached reach while checking the published owed-run record afresh'
  else report 1 'second --plan --base reuses cached reach while checking the published owed-run record afresh' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi

  # 3. Same candidate merged on HEAD: --plan-merge and full merge build both reuse the cache.
  merge_side
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && ! rc_asked_prepush && has 'merge-build-check: tests plan: reused cached reach' &&
     grep -qxF 'ship-pr-merge-plan: admitted runtest @test/training/train-a' "$TMP/out"; then
    report 0 '--plan-merge reuses reach cached by --plan --base for the same candidate merge'
  else report 1 '--plan-merge reuses reach cached by --plan --base for the same candidate merge' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi

  rc_run "$TMP/linuxbin:$rc_bin"
  if [ "$rc" = 0 ] && ! rc_asked_prepush &&
     grep -qxF 'build --root . -j 4 @a/runtest-x @test/operations/runtest-cover' "$TMP/record" &&
     ! grep -q '^shell ' "$TMP/record" &&
     has 'harness tier: tools/sub/a.sh is named as a dependency by @test/operations/runtest-cover, which the tests tier runs' &&
     has 'merge-build-check: tests reach: reused cached result' &&
     has 'merge-build-check: passed'; then
    report 0 'full merge build reuses cached reach, builds restored aliases and settles harness coverage'
  else report 1 'full merge build reuses cached reach, builds restored aliases and settles harness coverage' \
    "exit $rc; record=$(cat "$TMP/record"); tiers=$(tiers); $(cat "$TMP/out")"; fi

  # Full merge build still checks owed-run record afresh even on a cache hit.
  RC_OWED_URL=$OWED_EMPTY rc_run "$TMP/linuxbin:$rc_bin"
  if [ "$rc" = 1 ] && ! rc_asked_prepush &&
     has 'no owed-run record for' && ! has 'merge-build-check: passed'; then
    report 0 'full merge build still refuses an unadmitted record on a reach cache hit'
  else report 1 'full merge build still refuses an unadmitted record on a reach cache hit' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi

  # Ship-pr failed build logs stay byte-identical between cached and uncached runs.
  SHIP_PR_MERGE_BASE_SHA=$rc_base SHIP_PR_MERGE_HEAD_SHA=$rc_head FAKE_TESTS_RC=1 \
    rc_run "$TMP/linuxbin:$rc_bin"
  rc_fail_cached=$rc
  rc_fail_cached_asked=0; rc_asked_prepush && rc_fail_cached_asked=1
  cp "$TMP/out" "$TMP/rc-fail-cached"
  SHIP_PR_MERGE_BASE_SHA=$rc_base SHIP_PR_MERGE_HEAD_SHA=$rc_head FAKE_TESTS_RC=1 \
    OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=off \
    XDG_CONFIG_HOME=$rc_xdg OPAM_SWITCH_PREFIX=$rc_switch \
    OCANNL_TOOL_OWED_RUN_URL=$rc_owed \
    FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    FAKE_TIER_LIST=$TMP/rc-tier-list FAKE_NAMED=$TMP/rc-named \
    FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/rc-train \
    FAKE_REASON='x.ml (6 aliases)' \
    run "$TMP/linuxbin:$rc_bin"
  if [ "$rc_fail_cached" = 1 ] && [ "$rc_fail_cached_asked" = 0 ] && [ "$rc" = 1 ] &&
     rc_asked_prepush && cmp -s "$TMP/rc-fail-cached" "$TMP/out" && ! has 'reused cached'; then
    report 0 'failed ship-pr merge builds emit byte-identical logs on cached and uncached runs'
  else report 1 'failed ship-pr merge builds emit byte-identical logs on cached and uncached runs' \
    "exits $rc_fail_cached/$rc asked_cached=$rc_fail_cached_asked; $(diff "$TMP/rc-fail-cached" "$TMP/out")"; fi

  # 4. Invalidation on candidate/tree/worktree/ignored-file changes.
  rc_merge_sha=$(git -C "$R" rev-parse HEAD)
  base_alone_sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'base alone')
  git -C "$R" checkout -q --detach "$base_alone_sha"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'base-alone tree (same parents, base tree) misses the merged-tree reach cache'
  else report 1 'base-alone tree (same parents, base tree) misses the merged-tree reach cache' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  git -C "$R" checkout -q "$rc_merge_sha"

  # Dirty working tree bypasses cache (neither reads nor corrupts the clean entry).
  printf 'dirty\n' >>"$R/r.txt"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'a dirty working tree bypasses the reach cache'
  else report 1 'a dirty working tree bypasses the reach cache' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  git -C "$R" checkout -q -- r.txt
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && ! rc_asked_prepush && has 'reused cached reach'; then
    report 0 'restoring the clean working tree hits the original cached entry again'
  else report 1 'restoring the clean working tree hits the original cached entry again' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi

  # Git assume-unchanged and skip-worktree index flags hide edits from git status; both must bypass.
  git -C "$R" update-index --assume-unchanged x.ml
  printf 'hidden-edit\n' >>"$R/x.ml"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'assume-unchanged index flag bypasses the reach cache even when git status is empty'
  else report 1 'assume-unchanged index flag bypasses the reach cache even when git status is empty' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  git -C "$R" update-index --no-assume-unchanged x.ml
  git -C "$R" checkout -q -- x.ml
  git -C "$R" update-index --skip-worktree x.ml
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'skip-worktree index flag bypasses the reach cache'
  else report 1 'skip-worktree index flag bypasses the reach cache' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  git -C "$R" update-index --no-skip-worktree x.ml

  # Ignored non-directory file invalidates cache when added or edited.
  printf 'v1\n' >"$R/fixture-ignored.cfg"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'adding an ignored non-directory file invalidates the reach cache'
  else report 1 'adding an ignored non-directory file invalidates the reach cache' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  printf 'v2-changed\n' >"$R/fixture-ignored.cfg"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'editing an ignored non-directory file invalidates the reach cache'
  else report 1 'editing an ignored non-directory file invalidates the reach cache' \
    "exit $rc; tiers=$(tiers); $(cat "$TMP/out")"; fi
  rm -f "$R/fixture-ignored.cfg"

  # 5. Invalidation on Dune / toolchain / env / OS inputs.
  OCANNL_BACKEND=metal rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'changing OCANNL_BACKEND invalidates the reach cache'
  else report 1 'changing OCANNL_BACKEND invalidates the reach cache' "exit $rc; $(tiers)"; fi

  DUNE_PROFILE=release rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'changing DUNE_PROFILE invalidates the reach cache'
  else report 1 'changing DUNE_PROFILE invalidates the reach cache' "exit $rc; $(tiers)"; fi

  OCANNL_TOOL_AFFECTED_PROBE=0 rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'changing OCANNL_TOOL_AFFECTED_PROBE invalidates the reach cache'
  else report 1 'changing OCANNL_TOOL_AFFECTED_PROBE invalidates the reach cache' "exit $rc; $(tiers)"; fi

  DUNE_WORKSPACE=$rc_ws rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  rc_ws_first_hit=0
  rc_asked_prepush && rc_ws_first_hit=1
  printf '(lang dune 3.20)\n(context default)\n' >"$rc_ws"
  DUNE_WORKSPACE=$rc_ws rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && [ "$rc_ws_first_hit" = 1 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'setting and editing DUNE_WORKSPACE invalidates the reach cache'
  else report 1 'setting and editing DUNE_WORKSPACE invalidates the reach cache' "exit $rc; first=$rc_ws_first_hit; $(tiers)"; fi

  printf '(lang dune 3.20)\n(cache disabled)\n' >"$rc_xdg/dune/config"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'editing user dune config under XDG_CONFIG_HOME invalidates the reach cache'
  else report 1 'editing user dune config under XDG_CONFIG_HOME invalidates the reach cache' "exit $rc; $(tiers)"; fi

  printf 'installed: ["dune.3.24.2" "cudajit.0.8.0"]\n' >"$rc_switch/.opam-switch/switch-state"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'changing opam switch-state invalidates the reach cache'
  else report 1 'changing opam switch-state invalidates the reach cache' "exit $rc; $(tiers)"; fi

  mkdir -p "$rc_switch/lib/pkg_b"
  printf 'version = "2.0"\n' >"$rc_switch/lib/pkg_b/META"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'installing a package META under the switch lib directory invalidates the reach cache'
  else report 1 'installing a package META under the switch lib directory invalidates the reach cache' "exit $rc; $(tiers)"; fi

  # Flat META.<package> in OCAMLPATH (including same-length in-place edits) invalidates cache.
  rc_ocamlpath=$TMP/rc-ocamlpath
  mkdir -p "$rc_ocamlpath"
  printf 'version = "1.0"\n' >"$rc_ocamlpath/META.pkg_c"
  OCAMLPATH=$rc_ocamlpath rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  rc_flat_1=0; rc_asked_prepush && rc_flat_1=1
  OCAMLPATH=$rc_ocamlpath rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  rc_flat_2=0; ! rc_asked_prepush && has 'reused cached reach' && rc_flat_2=1
  printf 'version = "2.0"\n' >"$rc_ocamlpath/META.pkg_c"
  OCAMLPATH=$rc_ocamlpath rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && [ "$rc_flat_1" = 1 ] && [ "$rc_flat_2" = 1 ] &&
     rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'editing a flat META.<package> file in OCAMLPATH invalidates the reach cache'
  else report 1 'editing a flat META.<package> file in OCAMLPATH invalidates the reach cache' \
    "exit $rc; m1=$rc_flat_1 h2=$rc_flat_2; $(tiers)"; fi

  # Mid-query input mutation prevents publishing under the stale pre-query key.
  rc_mid_cache=$TMP/reach-cache-mid
  FAKE_MUTATE_FILE="$rc_switch/.opam-switch/switch-state" \
    rc_cache=$rc_mid_cache rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  rc_mid_entries=$(find "$rc_mid_cache" -maxdepth 1 -name 'reach-*.v1' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" = 0 ] && rc_asked_prepush && [ "$rc_mid_entries" = 0 ]; then
    report 0 'a mid-query switch-state change suppresses cache publication under the stale key'
  else report 1 'a mid-query switch-state change suppresses cache publication under the stale key' \
    "exit $rc; entries=$rc_mid_entries; $(cat "$TMP/out")"; fi

  printf '# modified dune shim\n' >>"$rc_bin/dune"
  rc_run "$TMP/linuxbin:$rc_bin" --plan-merge
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'modifying the dune executable invalidates the reach cache'
  else report 1 'modifying the dune executable invalidates the reach cache' "exit $rc; $(tiers)"; fi

  # Different host OS (Darwin vs Linux) misses and preserves Linux shell leg output on Darwin hits.
  rc_darwin_reach=$TMP/rc-darwin-reach
  printf 'tools/sub/a.sh\ttools/test-one.sh\n' >"$rc_darwin_reach"
  FAKE_REACH=$rc_darwin_reach RC_OWED_URL=$OWED_EMPTY rc_run "$TMP/darwinbin:$rc_bin" --plan-merge
  rc_darwin_1=$rc
  rc_darwin_asked_1=0; rc_asked_prepush && rc_darwin_asked_1=1
  FAKE_REACH=$rc_darwin_reach RC_OWED_URL=$OWED_EMPTY rc_run "$TMP/darwinbin:$rc_bin" --plan-merge
  if [ "$rc_darwin_1" = 0 ] && [ "$rc_darwin_asked_1" = 1 ] && [ "$rc" = 0 ] &&
     ! rc_asked_prepush && has 'reused cached reach' &&
     has 'linux-shell leg:' &&
     grep -qxF 'ship-pr-merge-plan: unadmitted runtest @test/training/train-a linux:member:tools/test-one.sh' "$TMP/out"; then
    report 0 'Darwin host misses Linux cache entry and preserves linux-shell needs on Darwin cache hit'
  else report 1 'Darwin host misses Linux cache entry and preserves linux-shell needs on Darwin cache hit' \
    "rc1=$rc_darwin_1 asked1=$rc_darwin_asked_1 rc2=$rc; $(tiers); $(cat "$TMP/out")"; fi

  # 6. Moved base and moved head invalidate cache.
  git -C "$R" checkout -q base
  git -C "$R" reset -q --hard "$rc_base"
  first r.txt && commit_on base
  rc_base2=$(git -C "$R" rev-parse HEAD)
  git -C "$R" checkout -q side
  rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'a moved base invalidates the reach cache'
  else report 1 'a moved base invalidates the reach cache' "exit $rc; $(tiers)"; fi
  last r.txt && commit_on side
  rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
  if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
    report 0 'a moved head invalidates the reach cache'
  else report 1 'a moved head invalidates the reach cache' "exit $rc; $(tiers)"; fi

  # 7. Fail-closed safety: failures never cached; corrupt/symlink/group-writable entries rejected;
  # unwritable cache dir never fails the plan; default XDG/HOME and relative cache paths work.
  rc_safe_cache=$TMP/reach-cache-safe
  OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=$rc_safe_cache \
    XDG_CONFIG_HOME=$rc_xdg OPAM_SWITCH_PREFIX=$rc_switch \
    FAKE_VERDICT=none run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
  if [ "$rc" = 1 ] && [ ! -d "$rc_safe_cache" ]; then
    report 0 'a failed tests-tier read writes no cache entry'
  else report 1 'a failed tests-tier read writes no cache entry' "exit $rc; dir=$(ls -la "$rc_safe_cache" 2>&1)"; fi

  # Populate rc_safe_cache once, then check pruning safety and tamper handling.
  mkdir -p "$rc_safe_cache"
  rc_unrelated_tmp=$rc_safe_cache/.tmp.unrelated
  rc_unrelated_v1=$rc_safe_cache/0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef.v1
  rc_stale_stage=$rc_safe_cache/.ocannl-reach-tmp.0000000000000000000000000000000000000000000000000000000000000000.99999.000001
  : >"$rc_unrelated_tmp" && touch -t 202001010000 "$rc_unrelated_tmp"
  : >"$rc_unrelated_v1" && touch -t 202001010000 "$rc_unrelated_v1"
  : >"$rc_stale_stage" && touch -t 202001010000 "$rc_stale_stage"
  rc_cache=$rc_safe_cache rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
  if [ "$rc" = 0 ] && [ -f "$rc_unrelated_tmp" ] && [ -f "$rc_unrelated_v1" ] && [ ! -e "$rc_stale_stage" ]; then
    report 0 'cache pruning removes stale reach staging files while preserving unrelated .tmp.* and *.v1 files'
  else report 1 'cache pruning removes stale reach staging files while preserving unrelated .tmp.* and *.v1 files' \
    "exit $rc; ls=$(ls -la "$rc_safe_cache" 2>&1)"; fi
  rc_entry=$(find "$rc_safe_cache" -maxdepth 1 -name 'reach-*.v1' | head -n 1)
  if [ -n "$rc_entry" ]; then
    cp "$rc_entry" "$TMP/rc-good-entry"
    # (a) Truncated entry (missing end trailer) -> rejected as miss, overwritten cleanly.
    head -n 4 "$TMP/rc-good-entry" >"$rc_entry"
    rc_cache=$rc_safe_cache rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
    if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
      report 0 'a truncated cache entry is rejected as a miss and recomputed'
    else report 1 'a truncated cache entry is rejected as a miss and recomputed' "exit $rc; $(tiers)"; fi
    # (b) Key mismatch inside entry -> rejected as miss.
    sed 's/^key	.*/key	0000000000000000000000000000000000000000000000000000000000000000/' \
      "$TMP/rc-good-entry" >"$rc_entry"
    rc_cache=$rc_safe_cache rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
    if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
      report 0 'a key-mismatched cache entry is rejected as a miss'
    else report 1 'a key-mismatched cache entry is rejected as a miss' "exit $rc; $(tiers)"; fi
    # (c) Symlink entry -> rejected as miss.
    rm -f "$rc_entry" && ln -s "$TMP/rc-good-entry" "$rc_entry"
    rc_cache=$rc_safe_cache rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
    if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
      report 0 'a symlinked cache entry is rejected as a miss'
    else report 1 'a symlinked cache entry is rejected as a miss' "exit $rc; $(tiers)"; fi
    # (d) Group-writable entry -> rejected as miss.
    cp "$TMP/rc-good-entry" "$rc_entry" && chmod 0620 "$rc_entry"
    rc_cache=$rc_safe_cache rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
    if [ "$rc" = 0 ] && rc_asked_prepush && ! has 'reused cached reach'; then
      report 0 'a group-writable cache entry is rejected as a miss'
    else report 1 'a group-writable cache entry is rejected as a miss' "exit $rc; $(tiers)"; fi
  else
    report 1 'cache entry file was created in rc_safe_cache' 'no *.v1 file found'
  fi

  # Unwritable cache path (regular file) never fails the plan.
  rc_not_a_dir=$TMP/reach-cache-not-a-dir
  : >"$rc_not_a_dir"
  rc_cache=$rc_not_a_dir rc_run "$TMP/linuxbin:$rc_bin" --plan --base "$rc_base2"
  if [ "$rc" = 0 ] && rc_asked_prepush; then
    report 0 'an unwritable cache path falls back cleanly without failing the plan'
  else report 1 'an unwritable cache path falls back cleanly without failing the plan' "exit $rc; $(cat "$TMP/out")"; fi

  # Default cache path under XDG_CACHE_HOME and HOME (when OCANNL_TOOL_MERGE_BUILD_REACH_CACHE is unset).
  rc_def_xdg=$TMP/rc-default-xdg
  rc_def_home=$TMP/rc-default-home
  mkdir -p "$rc_def_xdg" "$rc_def_home"
  (cd "$R" && env -u OCANNL_TOOL_MERGE_BUILD_REACH_CACHE \
    XDG_CACHE_HOME="$rc_def_xdg" XDG_CONFIG_HOME="$rc_xdg" HOME="$rc_def_home" \
    OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" tools/merge-build-check.sh --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_xdg_1=$?
  : >"$TMP/record.tiers"
  (cd "$R" && env -u OCANNL_TOOL_MERGE_BUILD_REACH_CACHE \
    XDG_CACHE_HOME="$rc_def_xdg" XDG_CONFIG_HOME="$rc_xdg" HOME="$rc_def_home" \
    OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" tools/merge-build-check.sh --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_xdg_2=$?
  if [ "$rc_xdg_1" = 0 ] && [ "$rc_xdg_2" = 0 ] && ! rc_asked_prepush &&
     has 'reused cached reach' && [ -d "$rc_def_xdg/ocannl/merge-build-reach" ]; then
    report 0 'unset cache override defaults to XDG_CACHE_HOME/ocannl/merge-build-reach'
  else report 1 'unset cache override defaults to XDG_CACHE_HOME/ocannl/merge-build-reach' \
    "rc=$rc_xdg_1/$rc_xdg_2; $(tiers); $(cat "$TMP/out")"; fi

  (cd "$R" && env -u OCANNL_TOOL_MERGE_BUILD_REACH_CACHE -u XDG_CACHE_HOME \
    XDG_CONFIG_HOME="$rc_xdg" HOME="$rc_def_home" \
    OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" tools/merge-build-check.sh --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_home_1=$?
  : >"$TMP/record.tiers"
  (cd "$R" && env -u OCANNL_TOOL_MERGE_BUILD_REACH_CACHE -u XDG_CACHE_HOME \
    XDG_CONFIG_HOME="$rc_xdg" HOME="$rc_def_home" \
    OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" tools/merge-build-check.sh --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_home_2=$?
  if [ "$rc_home_1" = 0 ] && [ "$rc_home_2" = 0 ] && ! rc_asked_prepush &&
     has 'reused cached reach' && [ -d "$rc_def_home/.cache/ocannl/merge-build-reach" ]; then
    report 0 'without XDG_CACHE_HOME, cache defaults to HOME/.cache/ocannl/merge-build-reach'
  else report 1 'without XDG_CACHE_HOME, cache defaults to HOME/.cache/ocannl/merge-build-reach' \
    "rc=$rc_home_1/$rc_home_2; $(tiers); $(cat "$TMP/out")"; fi

  # Relative explicit cache path resolves against caller's initial cwd across --plan --base.
  rc_rel_cwd=$TMP/rc-rel-cwd
  mkdir -p "$rc_rel_cwd"
  (cd "$rc_rel_cwd" && OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=rel-store \
    XDG_CONFIG_HOME="$rc_xdg" OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" "$R/tools/merge-build-check.sh" --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_rel_1=$?
  : >"$TMP/record.tiers"
  (cd "$rc_rel_cwd" && OCANNL_TOOL_MERGE_BUILD_REACH_CACHE=rel-store \
    XDG_CONFIG_HOME="$rc_xdg" OPAM_SWITCH_PREFIX="$rc_switch" OCANNL_TOOL_OWED_RUN_URL="$OWED_EMPTY" \
    PATH="$TMP/linuxbin:$rc_bin:$TMP/gitbin:$SYS_PATH" FAKE_RECORD="$TMP/record" \
    "$BASH" "$R/tools/merge-build-check.sh" --plan --base "$rc_base2") >"$TMP/out" 2>&1
  rc_rel_2=$?
  if [ "$rc_rel_1" = 0 ] && [ "$rc_rel_2" = 0 ] && ! rc_asked_prepush &&
     has 'reused cached reach' && [ -d "$rc_rel_cwd/rel-store" ]; then
    report 0 'relative OCANNL_TOOL_MERGE_BUILD_REACH_CACHE resolves against caller cwd across --plan --base'
  else report 1 'relative OCANNL_TOOL_MERGE_BUILD_REACH_CACHE resolves against caller cwd across --plan --base' \
    "rc=$rc_rel_1/$rc_rel_2; $(tiers); $(cat "$TMP/out")"; fi
fi
if harness_leg 'the format tier runs first, its scans off, and a red one ends the run #tiers'; then
  R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(tiers | sed -n 1p)" = 'fmt scans=0' ] && has 'merge-build-check: format tier passed' &&
     [ "$(grep -n '^merge-build-check: .* tier passed$' "$TMP/out" | sed -n 1p)" = \
       "$(grep -nxF 'merge-build-check: format tier passed' "$TMP/out")" ]; then
    report 0 'the formatter gate runs before the compile tier, with its scans off'
  else report 1 'the formatter gate runs before the compile tier, with its scans off' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  FAKE_FMT_RC=1 R=$APART run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ ! -s "$TMP/record" ] && [ "$(tiers)" = 'fmt scans=0' ] &&
     has 'format tier FAILED (tools/fmt-check.sh exit 1)' && ! has 'merge-build-check: passed'; then
    report 0 'a failing format tier exits 1 before dune or the list is asked'
  else report 1 'a failing format tier exits 1 before dune or the list is asked' "exit $rc; $(cat "$TMP/record"); $(tiers); $(cat "$TMP/out")"; fi
fi

if harness_leg 'the tests tier builds the list prepush prints for the head side #tiers'; then
  printf '%s\n' '@a/runtest-x' '@@runtest-root' >"$TMP/list"
  want_ask="prepush --since $(R=$BOTH_SCRIPT range_of) --list"
  FAKE_LIST=$TMP/list R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 @a/runtest-x @@runtest-root
shell --jobs 4" ] && [ "$(tiers | sed -n 2p)" = "$want_ask" ] && has 'merge-build-check: tests tier passed'; then
    report 0 'the list is asked of the two revisions and built -j 4, after the compile tier and before the harness tier'
  else report 1 'the list is asked of the two revisions and built -j 4, after the compile tier and before the harness tier' \
    "exit $rc; record: $(cat "$TMP/record"); tiers: $(tiers) (want $want_ask); $(cat "$TMP/out")"; fi
  # A merge that adds a test lists one alias more than its base alone: neither the list nor a
  # count may reach the log, or ship-pr reads a shared failure as the merge's own.
  if ! has '@a/runtest-x' && ! has 'alias(es)'; then
    report 0 'neither the list nor the ask'"'"'s counts are printed'
  else report 1 'neither the list nor the ask'"'"'s counts are printed' "$(cat "$TMP/out")"; fi
  R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
shell --jobs 4" ] && has 'tests tier: the change reaches no per-test alias' && has 'merge-build-check: tests tier passed'; then
    report 0 'a change reaching no per-test alias builds nothing and passes the tier'
  else report 1 'a change reaching no per-test alias builds nothing and passes the tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_LIST=$TMP/list FAKE_TESTS_RC=5 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && has 'tests tier FAILED (dune exit 5)' && ! grep -q '^shell ' "$TMP/record" &&
     ! has 'merge-build-check: passed'; then
    report 0 'a failing tests tier exits 1 before the harness tier'
  else report 1 'a failing tests tier exits 1 before the harness tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # The build of the base alone asks about the same change: the same two revisions.
  sha=$(git -C "$BOTH_SCRIPT" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$BOTH_SCRIPT" checkout -q --detach "$sha"
  FAKE_LIST=$TMP/list R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(tiers | sed -n 2p)" = "$want_ask" ]; then
    report 0 'the build of the base alone asks the list of the same two revisions'
  else report 1 'the build of the base alone asks the list of the same two revisions' "exit $rc; tiers: $(tiers) (want $want_ask)"; fi
  git -C "$BOTH_SCRIPT" checkout -q base
fi

if harness_leg 'a list that is not the whole pre-push list says what is OWED; one not read fails #tiers'; then
  # What is OWED here is recorded (OWED_FULL); the #owed leg reads the records themselves.
  export OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL
  printf '%s\n' '@a/runtest-x' >"$TMP/list"
  printf '%s\n' '@a/runtest-first' >"$TMP/tier-list"
  ask="prepush --since $(R=$APART range_of) --list"
  first_ask="prepush --referenced-only --since $(R=$APART range_of) --list"
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/tier-list R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 @a/runtest-first" ] && [ "$(tiers | sed -n 2,3p)" = "$ask
$first_ask" ] && has 'tests tier: OWED -- the change is CROSS-CUTTING, so every per-test alias past its first tier did not run here' &&
     has "owed-run: admitted run of 'build @runtest @train -j 4' for ${APART_HEAD:0:8} (fixture:/runs/1, writer=relay)"; then
    report 0 'a cross-cutting change builds its first tier and says the rest is owed'
  else report 1 'a cross-cutting change builds its first tier and says the rest is owed' "exit $rc; $(cat "$TMP/record"); $(tiers); $(cat "$TMP/out")"; fi
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/tier-list FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
     has 'tests tier: OWED -- the change is CROSS-CUTTING with no narrow first tier, so the whole list did not run here' &&
     has 'merge-build-check: tests tier passed'; then
    report 0 'a cross-cutting change with a wide first tier builds nothing and says the list is owed'
  else report 1 'a cross-cutting change with a wide first tier builds nothing and says the list is owed' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_LIST=$TMP/list FAKE_PREPUSH_RC=3 FAKE_VERDICT=incomplete R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 @a/runtest-x" ] && has 'tests tier: OWED -- the list is INCOMPLETE (a changed path is UNMAPPED)'; then
    report 0 'an incomplete list builds what it holds and says the gap is owed'
  else report 1 'an incomplete list builds what it holds and says the gap is owed' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # The bound's share: 200 aliases build, 201 do not, from the list or from a first tier.
  for n in 200 201; do
    i=0; : >"$TMP/long-list"
    while [ "$i" -lt "$n" ]; do i=$((i + 1)); printf '@a/runtest-t%s\n' "$i" >>"$TMP/long-list"; done
    FAKE_LIST=$TMP/long-list R=$APART run "$TMP/dunebin"
    built=$(sed -n 2p "$TMP/record" | tr ' ' '\n' | grep -c '^@a/runtest-t')
    case $n in
      200) [ "$rc" = 0 ] && [ "$built" = 200 ] && ! has 'OWED' ;;
      201) [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
             has 'tests tier: OWED -- the list is longer than the 200 aliases this build runs, so the whole list did not run here' ;;
    esac && report 0 "a list of $n aliases: $([ "$n" = 200 ] && echo built || echo 'owed, nothing built')" ||
      report 1 "a list of $n aliases: $([ "$n" = 200 ] && echo built || echo 'owed, nothing built')" "exit $rc; built $built; $(grep 'tests tier' "$TMP/out")"
  done
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/long-list R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
     has 'tests tier: OWED -- the change is CROSS-CUTTING and its first tier is longer than the 200 aliases this build runs, so the whole list did not run here'; then
    report 0 'a cross-cutting change whose first tier is 201 aliases builds nothing and says so'
  else report 1 'a cross-cutting change whose first tier is 201 aliases builds nothing and says so' "exit $rc; $(cat "$TMP/record" | cut -c1-200); $(grep 'tests tier' "$TMP/out")"; fi
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/tier-list FAKE_TIER_RC=3 FAKE_TIER_VERDICT=incomplete R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 @a/runtest-first" ] && has 'tests tier: OWED -- the change is CROSS-CUTTING and its first tier INCOMPLETE'; then
    report 0 'an incomplete first tier builds what it holds and says the rest is owed'
  else report 1 'an incomplete first tier builds what it holds and says the rest is owed' "exit $rc; $(cat "$TMP/record"); $(grep 'tests tier' "$TMP/out")"; fi
  # A first tier that was not read is not a wide one: a failed query, a status and a verdict at
  # odds, no trailer of its own each fail the tier and show the ask.
  bad=
  for probe in '1 ok' '2 ok' '0 cross-cutting' '3 ok' '0 none' '2 none' '75 ok'; do
    # shellcheck disable=SC2086 # the probe's two words
    set -- $probe
    FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/tier-list FAKE_TIER_RC=$1 FAKE_TIER_VERDICT=$2 \
      R=$APART run "$TMP/dunebin"
    { [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && ! has 'OWED' && ! has 'tests tier passed' &&
      has "tests tier FAILED (tools/test-run.sh prepush --referenced-only --since" &&
      has "--list exited $1 with no verdict this script reads)"; } || bad+=" [$probe: exit $rc; $(grep 'tests tier' "$TMP/out" | tr '\n' ' ')]"
  done
  if [ -z "$bad" ]; then
    report 0 'a first tier that was not read fails the tier, never reading as a wide one'
  else report 1 'a first tier that was not read fails the tier, never reading as a wide one' "$bad"; fi
  # `2 none` above is the default list's trailer alone, under the refusal's own status: the ask
  # that died before the tier answered (Codex P2). A list both cross-cutting and incomplete has
  # no first tier to ask and no status of its own, so it fails too.
  FAKE_LIST=$TMP/list FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting,incomplete FAKE_TIER_LIST=$TMP/tier-list R=$APART run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && [ "$(tiers | grep -c referenced-only)" = 0 ] && ! has 'OWED' &&
     has 'tests tier FAILED (the change is CROSS-CUTTING and its list INCOMPLETE'; then
    report 0 'a list both cross-cutting and incomplete fails the tier, asking no first tier'
  else report 1 'a list both cross-cutting and incomplete fails the tier, asking no first tier' "exit $rc; $(tiers); $(grep 'tests tier' "$TMP/out")"; fi
  # Fail closed: a status and a verdict that do not belong together, no verdict, no two sides.
  bad=
  for probe in '2 ok' '0 cross-cutting' '3 ok' '0 none' '1 ok'; do
    # shellcheck disable=SC2086 # the probe's two words
    set -- $probe
    FAKE_LIST=$TMP/list FAKE_PREPUSH_RC=$1 FAKE_VERDICT=$2 R=$APART run "$TMP/dunebin"
    { [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
      has "tests tier FAILED (tools/test-run.sh prepush --since" && has "--list exited $1 with no verdict this script reads)" &&
      has 'affected: 7 alias(es) from 2 changed path(s)'; } || bad+=" [$probe: exit $rc; $(cat "$TMP/record")]"
  done
  if [ -z "$bad" ]; then
    report 0 'a list read with no verdict, or one at odds with its status, fails the tier and shows the ask'
  else report 1 'a list read with no verdict, or one at odds with its status, fails the tier and shows the ask' "$bad"; fi
  fixture no-sides
  FAKE_LIST=$TMP/list run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && [ "$(tiers)" = 'fmt scans=0' ] &&
     has 'tests tier: UNREAD -- HEAD has no two sides with a merge base, so the tier fails' &&
     has 'tests tier FAILED (HEAD has no two sides with a merge base, so what the change reaches cannot be asked)'; then
    report 0 'a HEAD with no two sides fails the tests tier, asking nothing'
  else report 1 'a HEAD with no two sides fails the tests tier, asking nothing' "exit $rc; $(cat "$TMP/record"); $(tiers); $(cat "$TMP/out")"; fi
  export OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY
fi

# The owed-run record (tools/owed-run.sh, gh-ocannl-2218): what is OWED is read from
# refs/owed-runs/<head> on OCANNL_TOOL_OWED_RUN_URL, and each way a record can fail to stand for
# the head refuses the merge before anything is built.
plant() { # URL HEAD TREE PARENT MESSAGE: refs/owed-runs/HEAD at URL, any commit the caller makes
  local c
  c=$(printf '%s\n' "$5" | git -C "$R" commit-tree "$3" -p "$4") &&
    git -C "$R" push -q -f "$1" "$c:refs/owed-runs/$2"
}
merged_tree() { git -C "$R" rev-parse 'HEAD^{tree}'; }
if harness_leg 'a cross-cutting refusal carries the merged tree and its reason #owed-reason'; then
  R=$APART
  tree=$(merged_tree)
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' run "$TMP/dunebin"
  cp "$TMP/out" "$TMP/owed-reason-merge"
  merge_rc=$rc
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha"
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='m.txt (5 aliases)' run "$TMP/dunebin"
  git -C "$R" checkout -q base
  if [ "$merge_rc" = 1 ] && [ "$rc" = 1 ] &&
     grep -qFx 'merge-build-check: tests tier: CROSS-CUTTING reason: x.ml (6 aliases)' "$TMP/owed-reason-merge" &&
     has 'tests tier: CROSS-CUTTING reason: m.txt (5 aliases)' &&
     grep -vxF -f "$TMP/out" "$TMP/owed-reason-merge" | grep -qF "admits the merged tree $tree"; then
    report 0 "the refusal prints its reason and remains the merge's own failure when the base's reason differs"
  else report 1 "the refusal prints its reason and remains the merge's own failure when the base's reason differs" \
    "exits $merge_rc/$rc; $(cat "$TMP/owed-reason-merge" "$TMP/out")"; fi
fi
if harness_leg 'what is OWED is read from the head'"'"'s record, and every other record refuses #owed'; then
  printf '%s\n' '@a/runtest-first' >"$TMP/owed-tier"
  R=$APART
  tree=$(merged_tree)
  refused() { # LABEL WANT: the run refused, built nothing past the compile tier, printed WANT
    if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && has "$2" &&
       has "tests tier FAILED (no owed-run record for ${APART_HEAD:0:8} admits the merged tree $tree" &&
       ! has 'tests tier passed'; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(grep -e 'tests tier' -e owed-run "$TMP/out")"; fi
  }
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  refused 'a cross-cutting change with no record is refused, its first tier unbuilt (missing)' \
    "REFUSED -- no refs/owed-runs/$APART_HEAD: the head has no record (missing) (records on $OWED_EMPTY)"
  if has 'tools/owed-run.sh exit 1'; then
    report 0 'the merge build records exit 1 on a definitive missing-record refusal'
  else report 1 'the merge build records exit 1 on a definitive missing-record refusal' "$(grep 'tests tier' "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$TMP/nowhere.git FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  refused 'a record repository that cannot be read refuses (unreadable)' \
    "REFUSED -- refs/owed-runs/$APART_HEAD could not be read within 60s: unreadable (records on $TMP/nowhere.git)"
  if has 'tools/owed-run.sh exit 3'; then
    report 0 'the main merge build stays fail-closed on exit 3 (unreadable)'
  else report 1 'the main merge build stays fail-closed on exit 3 (unreadable)' "$(grep 'tests tier' "$TMP/out")"; fi
  # Direct owed-run.sh check exit-status distinction: 1 for definitive refusals, 3 for transport unreadable.
  owed "$OWED_EMPTY" check "$APART_HEAD" runtest
  miss_rc=$orc
  owed "$TMP/nowhere.git" check "$APART_HEAD" runtest
  unr_rc=$orc
  mkdir -p "$TMP/owed-fetchfailbin"
  cat >"$TMP/owed-fetchfailbin/git" <<FAKEGIT
#!/bin/sh
for a do
  case \$a in
    fetch) echo 'fatal: unable to access remote: Connection reset by peer' >&2; exit 128 ;;
  esac
done
exec "$GIT" "\$@"
FAKEGIT
  chmod +x "$TMP/owed-fetchfailbin/git"
  fetch_rc=0
  (cd "$R" && PATH=$TMP/owed-fetchfailbin:$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL \
    "$BASH" tools/owed-run.sh check "$APART_HEAD" runtest) >"$TMP/owed-out" 2>&1 || fetch_rc=$?
  if [ "$miss_rc" = 1 ] && [ "$unr_rc" = 3 ] && [ "$fetch_rc" = 3 ]; then
    report 0 'owed-run.sh check exits 1 on missing and 3 on ls-remote or fetch transport failures'
  else report 1 'owed-run.sh check exits 1 on missing and 3 on ls-remote or fetch transport failures' \
    "missing=$miss_rc unreadable=$unr_rc fetch=$fetch_rc; $(cat "$TMP/owed-out")"; fi
  bad=$TMP/owed-bad.git
  git init -q --bare "$bad"
  good=$(run_line "$APART_HEAD" 'runtest -j 4')
  base_sha=$(git -C "$R" rev-parse 'HEAD^1')
  plant "$bad" "$APART_HEAD" "$APART_HEAD^{tree}" "$base_sha" "$good"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  refused 'a record commit that is not the head'"'"'s child is refused (foreign)' "is not a child of ${APART_HEAD:0:8} alone: foreign"
  plant "$bad" "$APART_HEAD" "$base_sha^{tree}" "$APART_HEAD" "$good"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  refused 'a record commit carrying another tree is refused (foreign)' "carries another tree than ${APART_HEAD:0:8}'s: foreign"
  for probe in \
    "$(run_line "$base_sha" 'runtest -j 4')|a record of another head (${base_sha:0:8}): foreign|another head's line" \
    "$(run_line "$APART_HEAD" 'runtest -j 4' 1)|a run that exited 1|a failed run" \
    "$(run_line "$APART_HEAD" 'runtest -j 4' 0 no)|a run of a dirty checkout|a dirty run" \
    "$(run_line "$APART_HEAD" 'runtest -j 4' 0 yes metal)|a run on backend metal, not cc|a metal run" \
    "$(run_line "$APART_HEAD" 'build @a/runtest-first -j 4')|which is no full or train- run|a first-tier list" \
    "$(run_line "$APART_HEAD" 'runtest test/operations')|which is no full or train- run|a directory-scoped run" \
    "$(run_line "$APART_HEAD" 'build @runtest --force')|which is no full or train- run|an option" \
    "$(run_line "$APART_HEAD" "build '@runtest'")|which is no full or train- run|a quoted token" \
    "owed-run: v3 run head=$APART_HEAD exit=0|holds a line no v1 or v2 record reads|an unknown schema" \
    "owed-run: v1 run head=$APART_HEAD exit=0 cmd=runtest|holds a line no v1 or v2 record reads|a malformed line" \
    "owed-run: v2 run head=$APART_HEAD exit=0 clean=yes backend=cc source=fixture:/runs/1 cmd=runtest|holds a line no v1 or v2 record reads|a v2 line with no writer" \
    "owed-run: v2 run head=$APART_HEAD writer=leg: exit=0 clean=yes backend=cc source=fixture:/runs/1 cmd=runtest|holds a line no v1 or v2 record reads|an empty leg hostname" \
    "owed-run: v2 run head=$APART_HEAD writer=leg:a/b exit=0 clean=yes backend=cc source=fixture:/runs/1 cmd=runtest|holds a line no v1 or v2 record reads|a leg hostname with a slash" \
    "owed-run: v2 run head=$APART_HEAD writer=waive exit=0 clean=yes backend=cc source=fixture:/runs/1 cmd=runtest|holds a line no v1 or v2 record reads|a waiver writer on a run"; do
    line=${probe%%|*} rest=${probe#*|}
    plant "$bad" "$APART_HEAD" "$APART_HEAD^{tree}" "$APART_HEAD" "owed-run record

$good
$line"
    OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
    refused "a record holding ${rest#*|} beside a valid one is refused" "${rest%%|*}"
  done
  # Admitted: a full run, its first tier then built; a waiver, printed into the log.
  plant "$bad" "$APART_HEAD" "$APART_HEAD^{tree}" "$APART_HEAD" "owed-run record

$good"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 @a/runtest-first" ] && has "owed-run: admitted run of 'runtest -j 4' for ${APART_HEAD:0:8} (fixture:/runs/1, writer=legacy)"; then
    report 0 'a v1 full-suite record still admits and prints writer=legacy'
  else report 1 'a v1 full-suite record still admits and prints writer=legacy' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  waivers=$TMP/owed-waiver.git
  git init -q --bare "$waivers"
  owed "$waivers" waive --head "$APART_HEAD" --reason 'run 20261116T0000Z on rog, the maintainer'
  OCANNL_TOOL_OWED_RUN_URL=$waivers FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$orc" = 0 ] && [ "$rc" = 0 ] && has "owed-run: admitted waiver for ${APART_HEAD:0:8} by fixture (writer=waive): run 20261116T0000Z on rog, the maintainer"; then
    report 0 'a waiver for the head admits it and is printed into the log'
  else report 1 'a waiver for the head admits it and is printed into the log' "waive exit $orc: $(cat "$TMP/owed-out"); exit $rc; $(cat "$TMP/out")"; fi
  # The bound's share and an incomplete list are owed the same way.
  i=0; : >"$TMP/owed-long"
  while [ "$i" -lt 201 ]; do i=$((i + 1)); printf '@a/runtest-t%s\n' "$i" >>"$TMP/owed-long"; done
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_LIST=$TMP/owed-long run "$TMP/dunebin"
  refused 'a list over the bound with no record is refused' "no refs/owed-runs/$APART_HEAD"
  printf '%s\n' '@a/runtest-x' >"$TMP/owed-list"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_LIST=$TMP/owed-list FAKE_PREPUSH_RC=3 FAKE_VERDICT=incomplete run "$TMP/dunebin"
  refused 'an incomplete list needs every train- alias recorded too, not only the suite' 'covers no run of: train'
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_PREPUSH_RC=3 FAKE_VERDICT=incomplete run "$TMP/dunebin"
  if [ "$rc" = 0 ] && has "admitted run of 'build @runtest @train -j 4'"; then
    report 0 'an incomplete list is admitted by a record of the suite and @train'
  else report 1 'an incomplete list is admitted by a record of the suite and @train' "exit $rc; $(cat "$TMP/out")"; fi
  # train- aliases: reached by a change that is neither cross-cutting nor incomplete.
  printf '%s\n' '@a/runtest-x' '@test/training/train-a' >"$TMP/owed-affected"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_AFFECTED=$TMP/owed-affected run "$TMP/dunebin"
  refused 'a reached train- alias with no record is refused, for a narrow change too' 'tests tier: OWED -- the change reaches train- aliases, which this build does not run'
  plant "$bad" "$APART_HEAD" "$APART_HEAD^{tree}" "$APART_HEAD" "$(run_line "$APART_HEAD" 'build @test/training/train-b -j 1')"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_AFFECTED=$TMP/owed-affected run "$TMP/dunebin"
  refused 'a record of another train- alias does not cover the one reached' 'covers no run of: @test/training/train-a'
  plant "$bad" "$APART_HEAD" "$APART_HEAD^{tree}" "$APART_HEAD" "owed-run record

$good
$(run_line "$APART_HEAD" 'build @test/training/train-a -j 1')"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_TRAIN_N=1 FAKE_AFFECTED=$TMP/owed-affected FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(grep -c 'owed-run: admitted run' "$TMP/out")" = 2 ] && tiers | grep -qx "affected $(range_of)"; then
    report 0 'two records whose union covers the suite and the train- alias admit the change'
  else report 1 'two records whose union covers the suite and the train- alias admit the change' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_AFF_RC=2 FAKE_AFF_VERDICT=none run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && has 'so the train- aliases the change reaches are not known'; then
    report 0 'a train- reach that cannot be asked fails the tier'
  else report 1 'a train- reach that cannot be asked fails the tier' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\n' '@a/runtest-x' >"$TMP/owed-affected-none"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_AFFECTED=$TMP/owed-affected-none run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
     has 'prepush counted 1 train- alias(es) the change reaches, but tools/test-run.sh affected'; then
    report 0 'a train- count affected names no alias for fails the tier, never reading as none'
  else report 1 'a train- count affected names no alias for fails the tier, never reading as none' "exit $rc; $(cat "$TMP/out")"; fi
  # Structured names avoid affected, but never silently discard a disagreement.
  unread_refused() {
    if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
       has "tests tier FAILED ($2)" && ! has owed-run; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/out")"; fi
  }
  printf '%s\n' '@test/training/train-a' >"$TMP/train-names"
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/train-names FAKE_AFF_RC=2 run "$TMP/dunebin"
  if [ "$rc" = 0 ] && ! tiers | grep -q '^affected ' && has "admitted run of 'build @test/training/train-a -j 1'"; then
    report 0 'matching prepush train- names admit the recorded run without a third affected ask'
  else report 1 'matching prepush train- names admit the recorded run without a third affected ask' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/train-names FAKE_AFF_RC=2 run "$TMP/dunebin"
  if [ "$rc" = 0 ] && ! tiers | grep -q '^affected ' && [ "$(grep -c 'owed-run: admitted run' "$TMP/out")" = 2 ]; then
    report 0 'a wide first tier also names its training reach without a third affected ask'
  else report 1 'a wide first tier also names its training reach without a third affected ask' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=2 FAKE_TRAIN_NAMES=$TMP/train-names run "$TMP/dunebin"
  unread_refused 'prepush train- names must match the count exactly' 'prepush train- names disagree with its count (2)'
  OCANNL_TOOL_OWED_RUN_URL=$bad FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=none FAKE_TRAIN_NAMES=$TMP/train-names run "$TMP/dunebin"
  unread_refused 'prepush train- names without a count fail closed' 'prepush train- names disagree with its count (missing)'
  printf '%s\n' '@test/training/train-a' '@test/training/train-a' >"$TMP/train-duplicates"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=2 FAKE_TRAIN_NAMES=$TMP/train-duplicates run "$TMP/dunebin"
  unread_refused 'duplicate train- names cannot satisfy the count' 'prepush train- names disagree with its count (2)'
  printf '%s\n' '@missing/train-a' >"$TMP/train-missing-dir"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/train-missing-dir run "$TMP/dunebin"
  unread_refused 'an unreadable training alias directory fails before reading a record' "Dune could not list prepush's train- aliases in missing"
  printf '%s\n' '@test/training/train-absent' >"$TMP/train-absent"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/train-absent run "$TMP/dunebin"
  unread_refused 'a named train- alias absent from the tree refuses even with an @train record' 'prepush named a train- alias this tree does not have: @test/training/train-absent'
  # A transport that hangs: the read's bound ends it, and the record reads as unreadable.
  started=$SECONDS
  GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=protocol.ext.allow GIT_CONFIG_VALUE_0=always OCANNL_TOOL_OWED_RUN_FETCH_CAP=2 \
    OCANNL_TOOL_OWED_RUN_URL='ext::sleep 30' FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  took=$((SECONDS - started))
  if [ "$took" -lt 20 ]; then
    refused "a record repository that hangs is cut off at its bound and refuses (unreadable, ${took}s < 20s)" \
      "REFUSED -- refs/owed-runs/$APART_HEAD could not be read within 2s: unreadable (records on ext::sleep 30)"
    if has 'tools/owed-run.sh exit 3'; then
      report 0 'a hanging record repository fails the merge build with owed-run.sh exit 3'
    else report 1 'a hanging record repository fails the merge build with owed-run.sh exit 3' "$(grep 'tests tier' "$TMP/out")"; fi
  else report 1 'a record repository that hangs is cut off at its bound' "took ${took}s: $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$TMP/nowhere.git FAKE_LIST=$TMP/owed-list FAKE_TRAIN_N=none run "$TMP/dunebin"
  if [ "$rc" = 0 ] && tiers | grep -qx "affected $(range_of)" && ! has owed-run; then
    report 0 'no train- count read asks affected, which here names none: nothing is owed'
  else report 1 'no train- count read asks affected, which here names none: nothing is owed' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$TMP/nowhere.git FAKE_LIST=$TMP/owed-list run "$TMP/dunebin"
  if [ "$rc" = 0 ] && ! has owed-run && ! tiers | grep -q '^affected'; then
    report 0 'a change owing nothing reads no record and asks nothing more'
  else report 1 'a change owing nothing reads no record and asks nothing more' "exit $rc; $(tiers); $(cat "$TMP/out")"; fi
  # A record of an older push is under the older head's name: the new head has none (stale).
  fixture owed-stale
  first x.ml && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  owed "$OWED_FULL" add "$(run_line "$(git -C "$R" rev-parse 'HEAD^2')" 'runtest -j 4')"
  git -C "$R" checkout -q side && last m.txt && commit_on side && merge_side
  OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$orc" = 0 ] && [ "$rc" = 1 ] && has "no refs/owed-runs/$(git -C "$R" rev-parse 'HEAD^2'): the head has no record (missing)"; then
    report 0 'a record of the head before a later push does not admit the new head (stale)'
  else report 1 'a record of the head before a later push does not admit the new head (stale)' "add exit $orc; exit $rc; $(cat "$TMP/out")"; fi
  # The build of the base alone (the base's tree, the merge's parents) reads the same change and
  # the same missing record, but never the merge's own refusal line: the failure is the merge's.
  R=$APART
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  cp "$TMP/out" "$TMP/owed-merge-out"
  merge_rc=$rc
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  git -C "$R" checkout -q base
  extra=$(grep -vxF -f "$TMP/out" "$TMP/owed-merge-out")
  if [ "$merge_rc" = 1 ] && [ "$rc" = 1 ] && has "no refs/owed-runs/$APART_HEAD" &&
     printf '%s\n' "$extra" | grep -qF "admits the merged tree $tree" && ! has "admits the merged tree $tree"; then
    report 0 'the base alone fails without the merge'"'"'s refusal line, so the refusal is never a shared red'
  else report 1 'the base alone fails without the merge'"'"'s refusal line, so the refusal is never a shared red' "exits $merge_rc/$rc; extra: $extra"; fi
  # A merge whose tree is the base's (the head's change is in the base already) adds nothing:
  # its log is the base alone's, and what it would admit changes nothing on the base.
  fixture owed-same
  first x.ml && commit_on base
  git -C "$R" checkout -q side && first x.ml && commit_on side && merge_side
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  cp "$TMP/out" "$TMP/owed-same-out"
  merge_rc=$rc
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha"
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$(git -C "$R" rev-parse 'HEAD^1^{tree}')" = "$(git -C "$R" rev-parse 'base^{tree}')" ] &&
     [ "$merge_rc" = 1 ] && [ "$rc" = 1 ] && cmp -s "$TMP/out" "$TMP/owed-same-out"; then
    report 0 'a merge whose tree is the base'"'"'s logs as the base alone does (it changes nothing)'
  else report 1 'a merge whose tree is the base'"'"'s logs as the base alone does (it changes nothing)' "exits $merge_rc/$rc; $(diff "$TMP/owed-same-out" "$TMP/out")"; fi
fi

rundir() { # NAME EXIT DIRTY BACKEND CMD [HEAD]: a synthetic tools/test-run.sh run record
  local d=$TMP/runs/$1
  mkdir -p "$d"
  printf '%s\n' "${6:-$APART_HEAD}" >"$d/head"
  [ "$2" = none ] || printf '%s\n' "$2" >"$d/exit"
  printf '%s' "$3" >"$d/dirty"
  [ "$4" = none ] || printf '%s\n' "$4" >"$d/expect-backend"
  printf '%s \n' "$5" >"$d/cmd"
  printf '%s' "$d"
}
# tools/owed-run.sh's own verbs: what `record` certifies from a run directory, and `prune`.
if harness_leg 'record certifies only a finished, clean, cc run of an allowlisted command; prune #owed-run'; then
  R=$APART
  owed "$OWED_EMPTY" record "$(rundir good 0 '' cc 'runtest -j 4')" --print
  line=$(cat "$TMP/owed-out")
  case $line in
    "owed-run: v2 run head=$APART_HEAD writer=record exit=0 clean=yes backend=cc source="*":$TMP/runs/good cmd=runtest -j 4")
      report 0 'record prints the line of a finished, clean, cc full-suite run' ;;
    *) report 1 'record prints the line of a finished, clean, cc full-suite run' "exit $orc: $line" ;;
  esac
  rt=$TMP/owed-roundtrip.git
  git init -q --bare "$rt"
  owed "$rt" add "$line" && add_rc=$orc
  owed "$rt" check "$APART_HEAD" runtest
  if [ "$add_rc" = 0 ] && [ "$orc" = 0 ] && grep -qF "writer=relay)" "$TMP/owed-out"; then
    report 0 'add relabels a printed line and check prints writer=relay'
  else report 1 'add relabels a printed line and check prints writer=relay' "add $add_rc, check $orc: $(cat "$TMP/owed-out")"; fi
  # tools/test-run.sh records the argv it ran, its injected width and --no-buffer included.
  owed "$rt" record "$(rundir injected 0 '' cc 'build --no-buffer @test/training/train-a -j 1 -j 4')"
  owed "$rt" check "$APART_HEAD" runtest @test/training/train-a
  if [ "$orc" = 0 ] && [ "$(grep -c 'owed-run: admitted run' "$TMP/owed-out")" = 2 ] &&
     grep -qF 'writer=record)' "$TMP/owed-out"; then
    report 0 'a single train- alias run as test-run.sh records it (--no-buffer, an injected -j) is recorded and admitted'
  else report 1 'a single train- alias run as test-run.sh records it (--no-buffer, an injected -j) is recorded and admitted' "exit $orc: $(cat "$TMP/owed-out")"; fi
  bad=
  for probe in 'failed|1||cc|runtest -j 4' 'unfinished|none||cc|runtest -j 4' 'dirty|0| M x.ml|cc|runtest -j 4' \
               'metal|0||metal|runtest -j 4' 'nobackend|0||none|runtest -j 4' \
               'list|0||cc|build @test/operations/runtest-x -j 4' 'dir|0||cc|runtest test/operations'; do
    IFS='|' read -r n e dty be c <<EOF
$probe
EOF
    owed "$rt" record "$(rundir "$n" "$e" "$dty" "$be" "$c")"
    [ "$orc" = 2 ] && ! grep -q published "$TMP/owed-out" || bad+=" [$n: exit $orc: $(cat "$TMP/owed-out")]"
  done
  if [ -z "$bad" ]; then
    report 0 'record refuses a failed, unfinished, dirty, non-cc or filtered run, publishing nothing'
  else report 1 'record refuses a failed, unfinished, dirty, non-cc or filtered run, publishing nothing' "$bad"; fi
  # prune: a merged head's record goes, an unmerged one's stays.
  pr=$TMP/owed-prune.git
  git init -q --bare "$pr"
  git -C "$R" push -q "$pr" base:refs/heads/master
  git -C "$R" checkout -q side && last b.sh && commit_on side
  open_head=$(git -C "$R" rev-parse HEAD)
  git -C "$R" reset -q --hard HEAD^ && git -C "$R" checkout -q base
  owed "$pr" add "$(run_line "$APART_HEAD" 'runtest -j 4')"
  owed "$pr" add "$(run_line "$open_head" 'runtest -j 4')"
  owed "$pr" prune
  left=$(git -C "$R" ls-remote "$pr" 'refs/owed-runs/*' | awk '{ print $2 }')
  if [ "$orc" = 0 ] && [ "$left" = "refs/owed-runs/$open_head" ] && grep -qF 'pruned 1 record(s) of merged heads' "$TMP/owed-out"; then
    report 0 'prune deletes the merged head'"'"'s record and keeps the open one'"'"'s'
  else report 1 'prune deletes the merged head'"'"'s record and keeps the open one'"'"'s' "exit $orc: $(cat "$TMP/owed-out"); left: $left"; fi
  # Publish retries (gh-ocannl-2436): transient non-auth rejections and lease contention retry
  # with backoff; auth/authorization failures stop immediately on attempt 1.
  flaky=$TMP/owed-flaky.git
  git init -q --bare "$flaky"
  : >"$TMP/flaky-count"
  cat >"$flaky/hooks/pre-receive" <<HOOK
#!/bin/sh
n=\$(cat "$TMP/flaky-count" 2>/dev/null || echo 0)
n=\$((n + 1))
printf '%s\n' "\$n" >"$TMP/flaky-count"
if [ "\$n" -le 2 ]; then
  echo 'failed' >&2
  exit 1
fi
exit 0
HOOK
  chmod +x "$flaky/hooks/pre-receive"
  OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 owed "$flaky" add "$(run_line "$APART_HEAD" 'runtest -j 4')"
  attempts=$(cat "$TMP/flaky-count" 2>/dev/null || echo 0)
  add_rc=$orc
  add_out=$(cat "$TMP/owed-out")
  owed "$flaky" check "$APART_HEAD" runtest
  if [ "$add_rc" = 0 ] && [ "$attempts" = 3 ] && [ "$orc" = 0 ] &&
     printf '%s\n' "$add_out" | grep -qF 'attempt 1/4' &&
     printf '%s\n' "$add_out" | grep -qF 'attempt 2/4'; then
    report 0 'publish retries transient non-auth remote rejections and admits the published record'
  else report 1 'publish retries transient non-auth remote rejections and admits the published record' \
    "add=$add_rc attempts=$attempts check=$orc; $add_out; $(cat "$TMP/owed-out")"; fi
  # Concurrent lease contention: another writer publishes a sibling record right before attempt 1's
  # push, triggering git's real --force-with-lease `(stale info)` rejection; attempt 2 re-reads the
  # ref and merges both lines.
  raced=$TMP/owed-raced.git
  git init -q --bare "$raced"
  sibling_line=$(run_line "$APART_HEAD" 'build @test/training/train-a -j 1')
  sibling_commit=$(printf 'owed-run record\n\n%s\n' "$sibling_line" | git -C "$R" commit-tree "$APART_HEAD^{tree}" -p "$APART_HEAD")
  : >"$TMP/raced-count"
  mkdir -p "$TMP/racegitbin"
  cat >"$TMP/racegitbin/git" <<FAKEGIT
#!/bin/sh
for a do
  case \$a in
    push)
      n=\$(cat "$TMP/raced-count" 2>/dev/null || echo 0)
      n=\$((n + 1))
      printf '%s\n' "\$n" >"$TMP/raced-count"
      if [ "\$n" -eq 1 ]; then
        "$GIT" -C "$R" push -q "$raced" "$sibling_commit:refs/owed-runs/$APART_HEAD"
      fi
      exec "$GIT" "\$@" ;;
  esac
done
exec "$GIT" "\$@"
FAKEGIT
  chmod +x "$TMP/racegitbin/git"
  raced_rc=0
  (cd "$R" && PATH=$TMP/racegitbin:$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$raced \
    OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 "$BASH" tools/owed-run.sh add "$(run_line "$APART_HEAD" 'runtest -j 4')") \
    >"$TMP/owed-out" 2>&1 || raced_rc=$?
  raced_out=$(cat "$TMP/owed-out")
  raced_attempts=$(cat "$TMP/raced-count" 2>/dev/null || echo 0)
  owed "$raced" check "$APART_HEAD" runtest @test/training/train-a
  if [ "$raced_rc" = 0 ] && [ "$raced_attempts" = 2 ] && [ "$orc" = 0 ] &&
     printf '%s\n' "$raced_out" | grep -qF 'stale info' &&
     [ "$(grep -c 'owed-run: admitted run' "$TMP/owed-out")" = 2 ]; then
    report 0 'publish retries lease contention by re-reading the ref and merging both records'
  else report 1 'publish retries lease contention by re-reading the ref and merging both records' \
    "add=$raced_rc attempts=$raced_attempts check=$orc; $raced_out; $(cat "$TMP/owed-out")"; fi
  # Exhaustion + exponential backoff: 2 retries with 1s initial backoff sleep 1s then 2s (3 attempts).
  always_fail=$TMP/owed-always-fail.git
  git init -q --bare "$always_fail"
  : >"$TMP/backoff-count"
  cat >"$always_fail/hooks/pre-receive" <<HOOK
#!/bin/sh
n=\$(cat "$TMP/backoff-count" 2>/dev/null || echo 0)
n=\$((n + 1))
printf '%s\n' "\$n" >"$TMP/backoff-count"
echo 'failed' >&2
exit 1
HOOK
  chmod +x "$always_fail/hooks/pre-receive"
  started=$SECONDS
  OCANNL_TOOL_OWED_RUN_PUSH_RETRIES=2 OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=1 \
    owed "$always_fail" add "$(run_line "$APART_HEAD" 'runtest -j 4')"
  took=$((SECONDS - started))
  bf_attempts=$(cat "$TMP/backoff-count" 2>/dev/null || echo 0)
  if [ "$orc" = 2 ] && [ "$bf_attempts" = 3 ] && [ "$took" -ge 2 ] &&
     grep -qF 'retrying in 1s' "$TMP/owed-out" &&
     grep -qF 'retrying in 2s' "$TMP/owed-out" &&
     grep -qF 'after 3 attempt(s)' "$TMP/owed-out"; then
    report 0 'publish bounds retries with exponential backoff and exits 2 when exhausted'
  else report 1 'publish bounds retries with exponential backoff and exits 2 when exhausted' \
    "exit=$orc attempts=$bf_attempts took=${took}s; $(cat "$TMP/owed-out")"; fi
  # Auth/authorization failures stop immediately on attempt 1 (zero retries).
  mkdir -p "$TMP/authgitbin"
  cat >"$TMP/authgitbin/git" <<FAKEGIT
#!/bin/sh
for a do
  case \$a in
    push)
      n=\$(cat "\$FAKE_PUSH_COUNT_FILE" 2>/dev/null || echo 0)
      n=\$((n + 1))
      printf '%s\n' "\$n" >"\$FAKE_PUSH_COUNT_FILE"
      if [ -n "\${FAKE_PUSH_FAIL_ONCE:-}" ] && [ "\$n" -gt 1 ]; then
        exec "$GIT" "\$@"
      fi
      printf '%s\n' "\$FAKE_PUSH_ERR" >&2
      exit 1 ;;
  esac
done
exec "$GIT" "\$@"
FAKEGIT
  chmod +x "$TMP/authgitbin/git"
  auth_bad=
  for auth_err in \
    "fatal: Authentication failed for 'https://github.com/lukstafi/ocannl-staging.git/'" \
    "git@github.com: Permission denied (publickey)." \
    "fatal: could not read Username for 'https://github.com': terminal prompts disabled" \
    "error: The requested URL returned error: 401" \
    "error: The requested URL returned error: 403" \
    "remote: Permission to lukstafi/ocannl-staging.git denied to bot."; do
    : >"$TMP/auth-count"
    arc=0
    (cd "$R" && PATH=$TMP/authgitbin:$TMP/gitbin:$SYS_PATH \
      FAKE_PUSH_COUNT_FILE=$TMP/auth-count FAKE_PUSH_ERR=$auth_err \
      OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 \
      "$BASH" tools/owed-run.sh add "$(run_line "$APART_HEAD" 'runtest -j 4')") \
      >"$TMP/owed-out" 2>&1 || arc=$?
    ac=$(cat "$TMP/auth-count" 2>/dev/null || echo 0)
    if [ "$arc" != 2 ] || [ "$ac" != 1 ] || ! grep -qF 'authentication/authorization failed (not retried)' "$TMP/owed-out"; then
      auth_bad+=" [$auth_err: exit=$arc attempts=$ac out=$(cat "$TMP/owed-out")]"
    fi
  done
  if [ -z "$auth_bad" ]; then
    report 0 'auth and authorization push failures stop immediately without retrying'
  else report 1 'auth and authorization push failures stop immediately without retrying' "$auth_bad"; fi
  # Negative control: a non-auth rejection whose SHA contains 401/403 still retries and succeeds.
  sha_repo=$TMP/owed-sha403.git
  git init -q --bare "$sha_repo"
  : >"$TMP/sha403-count"
  src=0
  (cd "$R" && PATH=$TMP/authgitbin:$TMP/gitbin:$SYS_PATH \
    FAKE_PUSH_COUNT_FILE=$TMP/sha403-count FAKE_PUSH_FAIL_ONCE=1 \
    FAKE_PUSH_ERR=" ! [remote rejected] 4030000000000000000000000000000000000401 -> refs/owed-runs/$APART_HEAD (failed)" \
    OCANNL_TOOL_OWED_RUN_URL=$sha_repo OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 \
    "$BASH" tools/owed-run.sh add "$(run_line "$APART_HEAD" 'runtest -j 4')") \
    >"$TMP/owed-out" 2>&1 || src=$?
  sc=$(cat "$TMP/sha403-count" 2>/dev/null || echo 0)
  if [ "$src" = 0 ] && [ "$sc" = 2 ]; then
    report 0 'a non-auth rejection containing 401/403 digits in a SHA still retries and succeeds'
  else report 1 'a non-auth rejection containing 401/403 digits in a SHA still retries and succeeds' \
    "exit=$src attempts=$sc; $(cat "$TMP/owed-out")"; fi
  # Invalid retry/backoff settings exit 2.
  cfg_bad=
  for cfg in 'RETRIES=-1' 'RETRIES=abc' 'RETRIES=08' 'BACKOFF=-1' 'BACKOFF=abc' 'BACKOFF=08'; do
    k=${cfg%%=*} v=${cfg#*=}
    crc=0
    (cd "$R" && PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY \
      env "OCANNL_TOOL_OWED_RUN_PUSH_$k=$v" "$BASH" tools/owed-run.sh url) \
      >"$TMP/owed-out" 2>&1 || crc=$?
    [ "$crc" = 2 ] || cfg_bad+=" [$cfg: exit=$crc]"
  done
  if [ -z "$cfg_bad" ]; then
    report 0 'invalid push retry or backoff settings exit 2'
  else report 1 'invalid push retry or backoff settings exit 2' "$cfg_bad"; fi
  # Replacing a non-commit ref (blob): publish repairs it under the existing lease so waive/add can recover.
  repair_repo=$TMP/owed-repair-noncommit.git
  git init -q --bare "$repair_repo"
  repair_blob=$(printf 'corrupt non-commit ref\n' | git -C "$R" hash-object -w --stdin)
  git -C "$R" push -q -f "$repair_repo" "$repair_blob:refs/owed-runs/$APART_HEAD"
  owed "$repair_repo" check "$APART_HEAD" runtest
  bad_rc=$orc
  OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 owed "$repair_repo" waive --head "$APART_HEAD" --reason 'repair non-commit ref'
  waive_rc=$orc
  owed "$repair_repo" check "$APART_HEAD" runtest
  if [ "$bad_rc" = 1 ] && [ "$waive_rc" = 0 ] && [ "$orc" = 0 ] &&
     grep -qF 'admitted waiver' "$TMP/owed-out"; then
    report 0 'publish replaces a non-commit record ref under the existing lease'
  else report 1 'publish replaces a non-commit record ref under the existing lease' \
    "bad=$bad_rc waive=$waive_rc check=$orc; $(cat "$TMP/owed-out")"; fi
  # Record the process that owns the actual alarm, including on the alarm-and-exec subject.
  mkdir -p "$TMP/hanggitbin"
  cat >"$TMP/hanggitbin/perl" <<'FAKEPERL'
#!/bin/sh
if [ -n "${FAKE_SUPERVISOR_PID:-}" ]; then
  for a do
    if [ "$a" = "$FAKE_GIT_HANG" ]; then
      printf '%s\n' "$$" >"$FAKE_SUPERVISOR_PID"
      break
    fi
  done
fi
exec "$FAKE_REAL_PERL" "$@"
FAKEPERL
  # The leading Git process exits on TERM while its transport helper keeps the streams open.
  cat >"$TMP/hanggitbin/git" <<'FAKEGIT'
#!/bin/sh
for a do
  if [ "$a" = "$FAKE_GIT_HANG" ]; then
    [ -n "${FAKE_HELPER_PID:-}" ] || exec sleep 15
    perl -e '
      $SIG{TERM} = "IGNORE";
      my ($supervisor, $path) = @ARGV;
      open my $s, "<", $supervisor or die $!;
      my $owner = <$s>;
      die "missing alarm owner" unless $owner =~ /^[1-9][0-9]*\n$/;
      open my $f, ">", "$path.tmp" or die $!;
      print $f "$$\n";
      close $f or die $!;
      rename "$path.tmp", $path or die $!;
      kill "ALRM", int($owner) or die $!;
      sleep 30;
    ' "$FAKE_SUPERVISOR_PID" "$FAKE_HELPER_PID" &
    exec sleep 30
  fi
done
exec "$FAKE_REAL_GIT" "$@"
FAKEGIT
  chmod +x "$TMP/hanggitbin/git" "$TMP/hanggitbin/perl"
  real_perl=$(command -v perl)
  started=$SECONDS
  hang_rc=0
  (cd "$R" && PATH=$TMP/hanggitbin:$TMP/gitbin:$SYS_PATH \
    FAKE_GIT_HANG=ls-remote FAKE_HELPER_PID= FAKE_SUPERVISOR_PID= \
    FAKE_REAL_GIT=$GIT FAKE_REAL_PERL=$real_perl \
    OCANNL_TOOL_OWED_RUN_FETCH_CAP=1 OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL \
    "$BASH" tools/owed-run.sh check "$APART_HEAD" runtest) >"$TMP/owed-out" 2>&1 || hang_rc=$?
  if [ "$hang_rc" = 3 ] && [ "$((SECONDS - started))" -lt 5 ] &&
     grep -qF 'could not be read within 1s: unreadable' "$TMP/owed-out"; then
    report 0 'the real FETCH_CAP stops a hanging ls-remote and check returns exit 3'
  else
    report 1 'the real FETCH_CAP stops a hanging ls-remote and check returns exit 3' \
      "exit=$hang_rc took=$((SECONDS - started))s; $(cat "$TMP/owed-out")"
  fi
  for verb in ls-remote fetch; do
    pidfile=$TMP/hang-$verb.pid
    started=$SECONDS
    hang_rc=0
    (cd "$R" && PATH=$TMP/hanggitbin:$TMP/gitbin:$SYS_PATH \
      FAKE_GIT_HANG=$verb FAKE_HELPER_PID=$pidfile FAKE_SUPERVISOR_PID=$TMP/supervisor-$verb.pid \
      FAKE_REAL_GIT=$GIT FAKE_REAL_PERL=$real_perl \
      OCANNL_TOOL_OWED_RUN_FETCH_CAP=20 OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL \
      "$BASH" tools/owed-run.sh check "$APART_HEAD" runtest) >"$TMP/owed-out" 2>&1 || hang_rc=$?
    took=$((SECONDS - started))
    case $verb in ls-remote) action=read ;; fetch) action=fetched ;; esac
    claim="a hanging $verb returns exit 3 after the alarm and kills its ready TERM-resistant helper"
    helper=missing
    if [ -s "$pidfile" ]; then
      IFS= read -r pid <"$pidfile"
      for ((tick=0; tick<20; tick++)); do
        helper=$(ps -p "$pid" -o stat= 2>/dev/null)
        # An exited orphan may briefly remain a zombie until its new parent reaps it.
        case $helper in '' | Z*) break ;; esac
        sleep 0.05
      done
    fi
    if [ "$hang_rc" = 3 ] && [ "$took" -lt 20 ] &&
       grep -qF "could not be $action within 20s: unreadable" "$TMP/owed-out" &&
       { [ -z "$helper" ] || [[ $helper = Z* ]]; }; then
      report 0 "$claim"
    else
      report 1 "$claim" "exit=$hang_rc took=${took}s helper=${helper:-gone}; $(cat "$TMP/owed-out")"
    fi
    cleanup_fixture
  done
fi

owed_record_leg() {
  if ! fixture owed-record || ! git -C "$R" checkout -q side; then
    harness_abandon 'could not create the fleet hook checkout'
    return
  fi
  hook=$(sed -n 's/^owed-record //p' "$HERE/../.github/fleet-leg")
  if ! cp "$HERE/../$hook" "$R/tools/" || ! commit_on side; then
    harness_abandon 'could not commit the fleet hook on the fixture head'
    return
  fi
  hook_head=$(git -C "$R" rev-parse HEAD)
  hook_url=$TMP/owed-record.git
  git init -q --bare "$hook_url" || { harness_abandon 'could not create the record repository'; return; }
  leg_record() { # LAST_RUN LEG_RECORD OBSERVED_SHA [RECEIPT_DIR]
    hrc=0
    (
      cd "$R" || exit 2
      if [ $# -ge 4 ]; then
        PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$hook_url FAKE_LAST_RUN=$1 \
          FLEET_LEG_RECEIPT_DIR=$4 "./$hook" "$2" fixture-request "$3"
      else
        PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$hook_url FAKE_LAST_RUN=$1 \
          "./$hook" "$2" fixture-request "$3"
      fi
    ) >"$TMP/hook-out" 2>&1 || hrc=$?
  }
  hook_skip() { # LABEL EXPECTED: a refusal or ineligible skip leaves the remote ref absent
    if [ "$hrc" = 0 ] && grep -qF "$2" "$TMP/hook-out" &&
       [ -z "$(git -C "$R" ls-remote "$hook_url" "refs/owed-runs/$hook_head")" ]; then
      report 0 "$1"
    else report 1 "$1" "exit $hrc; $(cat "$TMP/hook-out")"; fi
  }
  hook_ineligible() { # LABEL EXPECTED: ineligible leg is distinguished from a broken receipt binding
    if [ "$hrc" = 0 ] && grep -qF 'ineligible leg:' "$TMP/hook-out" && grep -qF "$2" "$TMP/hook-out" &&
       ! grep -qF 'record refused' "$TMP/hook-out" && ! grep -qF 'does not match' "$TMP/hook-out" &&
       [ -z "$(git -C "$R" ls-remote "$hook_url" "refs/owed-runs/$hook_head")" ]; then
      report 0 "$1"
    else report 1 "$1" "exit $hrc; $(cat "$TMP/hook-out")"; fi
  }
  hook_refused() { # LABEL EXPECTED: broken binding or refusal prints record refused
    if [ "$hrc" = 0 ] && grep -qF 'record refused' "$TMP/hook-out" && grep -qF "$2" "$TMP/hook-out" &&
       [ -z "$(git -C "$R" ls-remote "$hook_url" "refs/owed-runs/$hook_head")" ]; then
      report 0 "$1"
    else report 1 "$1" "exit $hrc; $(cat "$TMP/hook-out")"; fi
  }
  alias=@test/training/train-a
  leg_cmd="build --no-buffer $alias -j 1 -j 4"
  last_run=$(rundir leg-good 0 '' cc "$leg_cmd" "$hook_head")
  leg_path="$TMP/current leg"
  other_path="$TMP/earlier leg"
  touch "$leg_path" "$other_path"
  printf '%s\n' "build $alias -j 1" >"$last_run/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_refused 'a receipt without a fleet invocation path is refused' "does not match this fleet leg's run record"
  printf '%s\n' "$other_path" >"$last_run/fleet-leg-run-record"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_refused 'an earlier receipt for the same head and targets cannot stand for this leg' "does not match this fleet leg's run record"
  printf '%s\n' "$leg_path" >"$last_run/fleet-leg-run-record"
  leg_record "$last_run" "$leg_path" "$(git -C "$R" rev-parse HEAD^)"
  hook_refused 'a matching invocation path cannot stand for a different observed head' 'is for a different head than the fleet leg observed'
  printf '%s\n' "build $alias @a/runtest-first" >"$last_run/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_ineligible 'a mixed first-tier request with an eligible effective command is logged as an ineligible leg' 'did not request an owed-run command'
  printf '%s\n' "--targets-file $TMP/list build" >"$last_run/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_ineligible 'a targets file with an eligible effective command is logged as an ineligible leg' 'did not request an owed-run command'
  rm -f "$last_run/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_refused 'a cc receipt missing fleet-leg-command is refused' 'missing fleet-leg-command'
  printf 'non-canonical-cc-invocation\n' >"$last_run/fleet-leg-ineligible"
  leg_record "$last_run" "$leg_path" "$hook_head" "$last_run"
  hook_ineligible 'a cc receipt marked ineligible (non-canonical syntax) is logged as an ineligible leg' 'explicit spaced --expect-backend cc'
  rm -f "$last_run/fleet-leg-ineligible"
  printf '%s\n' 'build @test/training/train-b' >"$last_run/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_refused 'a receipt for different targets cannot stand for this leg' "does not cover the fleet leg's requested targets"
  printf '%s\n' "build $alias -j 1" >"$last_run/fleet-leg-command"
  gpu_run=$(rundir leg-gpu 0 '' cuda "build @test/operations/runtest-focused" "$hook_head")
  focused_run=$(rundir leg-focused-cc 0 '' cc "build @test/operations/runtest-focused" "$hook_head")
  dirty_run=$(rundir leg-dirty 0 ' M x.ml' cc "build $alias" "$hook_head")
  printf '%s\n' "$leg_path" >"$gpu_run/fleet-leg-run-record"
  printf '%s\n' "$leg_path" >"$focused_run/fleet-leg-run-record"
  printf '%s\n' 'build @test/operations/runtest-focused' >"$focused_run/fleet-leg-command"
  cp "$last_run/fleet-leg-run-record" "$last_run/fleet-leg-command" "$dirty_run/"
  leg_record "$last_run" "$leg_path" "$hook_head" "$gpu_run"
  hook_ineligible 'a matching non-cc receipt via FLEET_LEG_RECEIPT_DIR is logged as an ineligible leg' 'was not run with --expect-backend cc'
  leg_record "$last_run" "$leg_path" "$hook_head" "$focused_run"
  hook_ineligible 'a matching focused cc receipt via FLEET_LEG_RECEIPT_DIR is logged as an ineligible leg' 'did not request an owed-run command'
  leg_record "$dirty_run" "$leg_path" "$hook_head"
  hook_refused 'a dirty receipt is refused with hook exit zero' 'ran a dirty checkout'
  # Stale receipt under the same SHA via FLEET_LEG_RECEIPT_DIR must refuse and never fall back to a matching last.
  stale_same_sha=$(rundir leg-stale-same-sha 0 '' cc "$leg_cmd" "$hook_head")
  printf '%s\n' "$other_path" >"$stale_same_sha/fleet-leg-run-record"
  printf '%s\n' "build $alias -j 1" >"$stale_same_sha/fleet-leg-command"
  leg_record "$last_run" "$leg_path" "$hook_head" "$stale_same_sha"
  hook_refused 'a stale receipt under the same SHA via FLEET_LEG_RECEIPT_DIR refuses without falling back to last' "does not match this fleet leg's run record"
  leg_record "$last_run" "$leg_path" "$hook_head" "$TMP/no-such-receipt-dir"
  hook_refused 'a missing FLEET_LEG_RECEIPT_DIR refuses without falling back to last' 'no run directory'
  leg_record "$last_run" "$leg_path" "$hook_head" last
  hook_refused 'a relative FLEET_LEG_RECEIPT_DIR refuses without resolving last' 'FLEET_LEG_RECEIPT_DIR must be a single-line absolute path'
  leg_record "$last_run" "$leg_path" "$hook_head" ''
  hook_refused 'an explicitly empty FLEET_LEG_RECEIPT_DIR refuses without falling back to last' 'FLEET_LEG_RECEIPT_DIR must be a single-line absolute path'
  last x.ml
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_refused 'a dirty checkout cannot publish a clean receipt' 'the fleet worktree is dirty'
  git -C "$R" checkout -- x.ml
  git -C "$R" push -q "$hook_url" "$hook_head:refs/heads/master" ||
    { harness_abandon 'could not publish the already-merged control head'; return; }
  leg_record "$last_run" "$leg_path" "$hook_head"
  hook_skip 'a head already on the record repository master leaves no record ref' "recorded nothing (head is on master at $hook_url)"
  if ! git -C "$R" push -q -f "$hook_url" "$hook_head^:refs/heads/master" ||
     ! git -C "$R" update-ref refs/remotes/origin/master "$hook_head"; then
    harness_abandon 'could not put the control head ahead of the record repository master'
    return
  fi
  # Private OCANNL_TOOL_TEST_RUNS receipt via FLEET_LEG_RECEIPT_DIR publishes even when last is stale/unrelated.
  priv_url=$TMP/owed-record-priv.git
  git init -q --bare "$priv_url" || { harness_abandon 'could not create private record repository'; return; }
  git -C "$R" push -q "$priv_url" "$hook_head^:refs/heads/master" ||
    { harness_abandon 'could not seed private record repository master'; return; }
  priv_run=$(rundir leg-private 0 '' cc "$leg_cmd" "$hook_head")
  printf '%s\n' "$leg_path" >"$priv_run/fleet-leg-run-record"
  printf '%s\n' "build $alias -j 1" >"$priv_run/fleet-leg-command"
  saved_hook_url=$hook_url
  hook_url=$priv_url
  OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 leg_record "$stale_same_sha" "$leg_path" "$hook_head" "$priv_run"
  hook_url=$saved_hook_url
  priv_host=$(hostname -s 2>/dev/null || hostname)
  owed "$priv_url" check "$hook_head" "$alias"
  if [ "$hrc" = 0 ] && [ "$orc" = 0 ] &&
     grep -qF "($priv_host:$priv_run, writer=leg:$priv_host)" "$TMP/owed-out"; then
    report 0 'FLEET_LEG_RECEIPT_DIR publishes a receipt from private state without reading last'
  else report 1 'FLEET_LEG_RECEIPT_DIR publishes a receipt from private state without reading last' \
    "hook=$hrc ($(cat "$TMP/hook-out")); check=$orc ($(cat "$TMP/owed-out"))"; fi
  # Simulate t1854: one transient `[remote rejected] ... (failed)` rejection on the fleet hook's push.
  : >"$TMP/hook-push-count"
  cat >"$hook_url/hooks/pre-receive" <<HOOK
#!/bin/sh
n=\$(cat "$TMP/hook-push-count" 2>/dev/null || echo 0)
n=\$((n + 1))
printf '%s\n' "\$n" >"$TMP/hook-push-count"
if [ "\$n" -eq 1 ]; then
  echo 'failed' >&2
  exit 1
fi
exit 0
HOOK
  chmod +x "$hook_url/hooks/pre-receive"
  OCANNL_TOOL_OWED_RUN_PUSH_BACKOFF=0 leg_record "$last_run" "$leg_path" "$hook_head"
  rm -f "$hook_url/hooks/pre-receive"
  hook_attempts=$(cat "$TMP/hook-push-count" 2>/dev/null || echo 0)
  merge_side || { harness_abandon 'could not merge the published head into the fixture base'; return; }
  printf '%s\n' "$alias" >"$TMP/hook-train-name"
  OCANNL_TOOL_OWED_RUN_URL=$hook_url FAKE_TRAIN_N=1 FAKE_TRAIN_NAMES=$TMP/hook-train-name run "$TMP/dunebin"
  leg_host=$(hostname -s 2>/dev/null || hostname)
  admission="merge-build-check:   owed-run: admitted run of '$leg_cmd' for ${hook_head:0:8}"
  if [ "$hrc" = 0 ] && [ "$hook_attempts" = 2 ] && [ "$rc" = 0 ] &&
     grep -qxF "$admission ($leg_host:$last_run, writer=leg:$leg_host)" "$TMP/out"; then
    report 0 'the fleet hook retries a transient remote rejection and the merge gate admits its record'
  else report 1 'the fleet hook retries a transient remote rejection and the merge gate admits its record' \
    "hook exit $hrc attempts=$hook_attempts: $(cat "$TMP/hook-out"); merge gate exit $rc: $(cat "$TMP/out")"; fi
}
harness_leg_run 'the fleet hook publishes only eligible receipts from its own leg #owed-record' owed_record_leg

# ship-pr compares exact log lines from a failed merge and the base alone.
# Standalone failures and successful ship-pr builds still report their timing.
if harness_leg 'tier timings and stable ship-pr failure logs #run'; then
  timing() { grep -qE "^merge-build-check: $1 tier $2 in $3s" "$TMP/out"; }
  export FAKE_MEMBER_TIMING='HARNESS tools/test-fixture.sh finished in 1s (exit 0)'
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    FAKE_TIER_DELAY=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && timing format passed '[0-9]+' && timing compile passed '[1-9][0-9]*' &&
     timing tests passed '[0-9]+' && timing harness passed '[1-9][0-9]*' && has "$FAKE_MEMBER_TIMING"; then
    report 0 'a successful ship-pr build reports all four tier times and the harness member rows'
  else report 1 'a successful ship-pr build reports all four tier times and the harness member rows' "exit $rc; $(cat "$TMP/out")"; fi
  FAKE_SHELL_RC=1 FAKE_TIER_DELAY=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && timing compile passed '[1-9][0-9]*' && timing harness FAILED '[1-9][0-9]*' &&
     has 'harness tier FAILED (tools/test-harnesses.sh exit 1)' && has "$FAKE_MEMBER_TIMING"; then
    report 0 'a standalone failure reports both passed and failed tier times'
  else report 1 'a standalone failure reports both passed and failed tier times' "exit $rc; $(cat "$TMP/out")"; fi
  for delay in 0 1; do
    SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
      FAKE_SHELL_RC=1 FAKE_TIER_DELAY=$delay R=$BOTH_SCRIPT run "$TMP/dunebin"
    case $delay in 0) first_rc=$rc; cp "$TMP/out" "$TMP/out-first" ;; esac
  done
  if [ "$first_rc" = 1 ] && [ "$rc" = 1 ] && cmp -s "$TMP/out-first" "$TMP/out" &&
     ! grep -qE ' tier (passed|FAILED) in [0-9]+s' "$TMP/out" && ! has "$FAKE_MEMBER_TIMING"; then
    report 0 'failed ship-pr builds produce identical whole logs across different durations'
  else report 1 'failed ship-pr builds produce identical whole logs across different durations' \
    "exits $first_rc, $rc; $(diff "$TMP/out-first" "$TMP/out")"; fi
  unset FAKE_MEMBER_TIMING
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    FAKE_TERM_DURING_DUNE=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 143 ] && has 'format tier passed' &&
     ! grep -qE '^merge-build-check: .* tier (passed|FAILED) in [0-9]+s' "$TMP/out"; then
    report 0 'TERM during a ship-pr compile keeps the completed format timing out of the log'
  else report 1 'TERM during a ship-pr compile keeps the completed format timing out of the log' \
    "exit $rc; $(cat "$TMP/out")"; fi
fi

# The scope. FAKE_REACH stands for the mapping; the fixtures above reach nothing without it, which
# is why they run the whole tier.
reach_of() { # PATH...: each PATH reaches tools/test-one.sh
  local p
  for p do printf '%s\ttools/test-one.sh\n' "$p"; done >"$TMP/reach"
}
export FAKE_REACH=$TMP/reach
: >"$TMP/reach"
if harness_leg 'a change reaching one member runs that member, and no more #scope'; then
  # Scope legs isolate local harness-tier narrowing independently of the host OS; #linux-shell
  # below pins the Darwin/Linux owed-run gate explicitly.
  scope_bin=$TMP/linuxbin:$TMP/dunebin
  reach_of tools/sub/a.sh
  R=$BOTH_SCRIPT plan
  if [ "$rc" = 0 ] && has 'harness tier: a script changed on the head side -- tools/sub/a.sh matches *.sh' &&
     has 'harness tier: SELECTED -- scoped to what the 1 path(s) the head side changed reach:' && has 'merge-build-check:   tools/test-one.sh'; then
    report 0 '--plan names what the change reaches'
  else report 1 '--plan names what the change reaches' "exit $rc; $(cat "$TMP/out")"; fi
  R=$BOTH_SCRIPT run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
shell --jobs 4 --affected --files tools/sub/a.sh" ] && has 'merge-build-check: passed'; then
    report 0 'a reached change runs its scoped tier, the whole tier not at all'
  else report 1 'a reached change runs its scoped tier, the whole tier not at all' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=1 R=$BOTH_SCRIPT run "$scope_bin"
  if [ "$rc" = 1 ] && [ "$(grep -c '^shell ' "$TMP/record")" = 1 ] && has 'harness tier FAILED (tools/test-harnesses.sh exit 1)'; then
    report 0 'a failing scoped tier exits 1, with no whole tier to fall back on'
  else report 1 'a failing scoped tier exits 1, with no whole tier to fall back on' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Every path the head side changed is asked about, a non-script too; a rename by both its names.
  fixture ask-all
  first tools/sub/a.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last x.ml && commit_on side && merge_side
  reach_of tools/sub/a.sh
  run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files tools/sub/a.sh x.ml' ]; then
    report 0 'every path the head side changed is asked about, a non-script too'
  else report 1 'every path the head side changed is asked about, a non-script too' "exit $rc; $(cat "$TMP/record")"; fi
  fixture ask-rename
  git -C "$R" mv r.txt r.sh && commit_on base
  git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
  reach_of r.sh
  run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files r.txt r.sh' ]; then
    report 0 'a rename is asked about by both its names'
  else report 1 'a rename is asked about by both its names' "exit $rc; $(cat "$TMP/record")"; fi
  # The change is the head side's alone: nothing has to have changed on the base side (the merge
  # `trigger always` builds when the base has not moved), and a non-script that reaches a member
  # runs it with no script changed at all.
  fixture head-only
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  reach_of tools/sub/a.sh
  run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files tools/sub/a.sh' ]; then
    report 0 'a script the head side alone changed runs what it reaches'
  else report 1 'a script the head side alone changed runs what it reaches' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  fixture non-script
  first tools/sub/a.sh && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  reach_of x.ml
  run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files x.ml' ] &&
     has 'harness tier: no script changed on the head side -- none of the 1 path(s) it changed matches *.sh' &&
     has 'harness tier: SELECTED -- scoped to what the 1 path(s) the head side changed reach:'; then
    report 0 'a non-script the head side changed runs the member it reaches, no script changed'
  else report 1 'a non-script the head side changed runs the member it reaches, no script changed' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # The build of the base alone: the base's tree on the merge's two parents, as ship-pr commits it.
  fixture base-alone
  first tools/sub/a.sh && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  reach_of tools/sub/a.sh
  plan
  cp "$TMP/out" "$TMP/merge-plan"
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha" && plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- scoped to' && cmp -s "$TMP/out" "$TMP/merge-plan"; then
    report 0 'the build of the base alone selects and narrows as the merge does'
  else report 1 'the build of the base alone selects and narrows as the merge does' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'what the scope cannot establish runs the whole tier, never less #scope'; then
  scope_bin=$TMP/linuxbin:$TMP/dunebin
  fixture two-scripts
  first tools/sub/a.sh && first b.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last b.sh && last x.ml && commit_on side && merge_side
  TWO_SCRIPTS=$R
  # One script reached, the other not: the reached one must not hide it.
  reach_of tools/sub/a.sh
  R=$TWO_SCRIPTS run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- no harness reaches b.sh'; then
    report 0 'a script no member reaches runs the whole tier, though another is reached'
  else report 1 'a script no member reaches runs the whole tier, though another is reached' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  reach_of tools/sub/a.sh b.sh
  FAKE_ASK_RC=2 R=$TWO_SCRIPTS run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- what b.sh reaches could not be asked (test-harnesses.sh exit 2)'; then
    report 0 'a failed ask runs the whole tier'
  else report 1 'a failed ask runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Each script alone answers, the joint ask does not.
  FAKE_ASK_FAILS_ON=x.ml R=$TWO_SCRIPTS run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- what the paths the head side changed reach could not be asked'; then
    report 0 'a failed ask for the paths together runs the whole tier'
  else report 1 'a failed ask for the paths together runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 R=$TWO_SCRIPTS run "$scope_bin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4 --affected --files b.sh tools/sub/a.sh x.ml
shell --jobs 4' ] && has 'the scoped run was refused (exit 2), no member run; running the whole tier'; then
    report 0 'a scoped run refused before any member started runs the whole tier'
  else report 1 'a scoped run refused before any member started runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 FAKE_SHELL_RC=1 R=$TWO_SCRIPTS run "$scope_bin"
  if [ "$rc" = 1 ] && has 'harness tier FAILED (tools/test-harnesses.sh exit 1)'; then
    report 0 'the whole tier a refusal falls back on still decides'
  else report 1 'the whole tier a refusal falls back on still decides' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'hand-run shell markers preserve harness scope #hand-run'; then
  harness_require perl
  fixture hand-run
  cp "$HERE/affected-tests.sh" "$R/tools/" || exit 2
  mkdir -p "$R/test/metal_probe" "$R/benchmarks" || exit 2
  # This fixture hands the plan the real marker and harness reader, with a
  # one-member manifest and no Dune. Keep fixture marker syntax out of the
  # hosting harness's own comment header.
  cat >"$R/tools/test-harnesses.sh" <<'FAKE'
#!/usr/bin/env bash
if [ "$*" = --list ]; then
  printf '%s\n' tools/test-one.sh
elif [ "$1" = --affected ]; then
  shift 2
  exec bash "$(dirname "$0")/affected-tests.sh" --harnesses "$@"
else
  printf 'shell %s\n' "$*" >>"$FAKE_RECORD"
fi
FAKE
  printf '%s\n' '#!/bin/sh' ':' >"$R/tools/test-one.sh"
  # Build the fixture names from parts so the reach reader does not mistake
  # this harness's scratch files for the real drivers.
  sh_ext=.sh
  marker_word=harness
  marker="# ${marker_word}: none -- exclusive diagnostic session"
  cell=test/metal_probe/run_cells$sh_ext
  cost=benchmarks/gh1529_nvrtc_cost$sh_ext
  for p in "$cell" "$cost"; do printf '%s\n' '#!/bin/sh' ':' >"$R/$p"; done
  commit_on base && git -C "$R" branch -f side HEAD && git -C "$R" checkout -q side || exit 2
  for p in "$cell" "$cost"; do
    printf '%s\n' '#!/bin/sh' "$marker" ':' >"$R/$p"
  done
  commit_on side && merge_side || exit 2
  plan
  if [ "$rc" = 0 ] && has 'harness tier: skipped -- the 2 path(s) the head side changed reach no harness'; then
    report 0 'both hand-run witnesses select no harness'
  else report 1 'both hand-run witnesses select no harness' "exit $rc; $(cat "$TMP/out")"; fi

  git -C "$R" checkout -q side || exit 2
  printf '%s\n' '#!/bin/sh' ': # edited without a marker' >"$R/$cost"
  commit_on side && merge_side || exit 2
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && has "harness tier: SELECTED -- whole -- no harness reaches $cost" &&
     grep -qxF "shell --jobs 4" "$TMP/record"; then
    report 0 'an unmarked, unreached script still selects the whole tier'
  else report 1 'an unmarked, unreached script still selects the whole tier' "exit $rc; $(cat "$TMP/out")"; fi

  check_markers() {
    rc=0
    (cd "$R" && bash tools/affected-tests.sh --harnesses --files x.ml) >"$TMP/out" 2>&1 || rc=$?
  }
  # Ask about an unrelated path. Adding a consumer must fail even when the
  # marked script itself does not change.
  printf '%s\n' "sh $cell" >>"$R/tools/test-one.sh"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:2: harness marker conflicts with reach from tools/test-one.sh"; then
    report 0 'adding a harness consumer of a marked script is refused'
  else report 1 'adding a harness consumer of a marked script is refused' "exit $rc; $(cat "$TMP/out")"; fi
  git -C "$R" restore tools/test-one.sh || exit 2
  printf '(rule (enabled_if false) (alias runtest-probe) (action (run sh %%{dep:%s})))\n' "$cell" >"$R/dune"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:2: harness marker conflicts with a literal reference in dune"; then
    report 0 'adding a Dune rule that names a marked script is refused across guards'
  else report 1 'adding a Dune rule that names a marked script is refused across guards' "exit $rc; $(cat "$TMP/out")"; fi
  mkdir -p "$R/rules" || exit 2
  printf '(include rules/outer.inc)\n' >"$R/dune"
  printf '(include inner.inc)\n' >"$R/rules/outer.inc"
  printf '(rule (alias runtest-probe) (deps %s) (action (progn)))\n' "$cell" >"$R/rules/inner.inc"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:2: harness marker conflicts with a literal reference in rules/inner.inc"; then
    report 0 'a nested Dune include resolves from the file that includes it'
  else report 1 'a nested Dune include resolves from the file that includes it' "exit $rc; $(cat "$TMP/out")"; fi

  printf '(rule (alias runtest-probe) (deps (:include deps.sexp)) (action (progn)))\n' >"$R/dune"
  printf '(%s)\n' "$cell" >"$R/deps.sexp"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:2: harness marker conflicts with a literal reference in deps.sexp"; then
    report 0 'a literal reference in a dependency include is refused'
  else report 1 'a literal reference in a dependency include is refused' "exit $rc; $(cat "$TMP/out")"; fi

  printf '; (rule (action (run sh %s)))\n(rule (alias runtest-probe) (deps (glob_files_rec test/*%s)) (action (progn)))\n' \
    "$cell" "${cell##*/}" >"$R/dune"
  check_markers
  if [ "$rc" = 0 ] && has '0 harness command(s) from 1 changed path(s)'; then
    report 0 'a Dune comment and a bulk dependency leave the marker valid'
  else report 1 'a Dune comment and a bulk dependency leave the marker valid' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\n' '#!/bin/sh' ':' "$marker" >"$R/$cell"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:3: harness marker must be in the header"; then
    report 0 'a marker after the header is refused'
  else report 1 'a marker after the header is refused' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\n' '#!/bin/sh' ": $marker" >"$R/$cell"
  check_markers
  if [ "$rc" = 2 ] && has "$cell:2: harness marker must be in the header"; then
    report 0 'an inline marker after shell code is refused'
  else report 1 'an inline marker after shell code is refused' "exit $rc; $(cat "$TMP/out")"; fi
  git -C "$R" restore "$cell" || exit 2
  git -C "$R" config core.quotePath true || exit 2
  quoted="$R/$(printf 'caf\303\251.sh')"
  printf '%s\n' '#!/bin/sh' "$marker" ':' >"$quoted"
  check_markers
  if [ "$rc" = 2 ] && has 'harness marker requires a shell script path'; then
    report 0 'a marker on a Git-quoted path outside the scanner grammar is refused'
  else report 1 'a marker on a Git-quoted path outside the scanner grammar is refused' "exit $rc; $(cat "$TMP/out")"; fi
fi

# A script no member reaches is covered by an alias the tests tier runs whose rule names it
# (gh-ocannl-2304): affected's exact `named as a dependency` line, for an alias of the list run
# here. The issue's two shapes -- a script beside its owning test, one a test elsewhere names --
# and an unmapped one. The paths are the fixture's own: a repository path named here would be
# one this harness reaches (affected's harness reader), handing it every edit of that script.
if harness_leg 'a script an alias the tests tier runs names is covered, any other runs the whole tier #cover'; then
  SWEEP=ops/fixture_legs.sh CENSUS=tools/fixture-trace.sh
  SWEEP_ALIAS=@ops/runtest-fixture_owner CENSUS_ALIAS=@bench/runtest-fixture_orchestrate
  named_by() { # SCRIPT ALIAS...: affected's line for each ALIAS naming SCRIPT
    local s=$1 a
    shift
    for a do printf 'affected: %s: named as a dependency by the rule of %s\n' "$s" "$a"; done
  }
  cover_fixture() { # NAME PATH...: a fixture whose head side edits each PATH (they exist on both)
    local n=$1 f
    shift
    fixture "$n"
    mkdir -p "$R/ops" "$R/tools"
    for f in "$SWEEP" "$CENSUS" tools/unmapped.sh; do printf '1\n2\n3\n4\n5\n6\n' >"$R/$f"; done
    git -C "$R" add -A && git -C "$R" commit -q -m 'the scripts' && git -C "$R" branch -f side
    first x.ml && commit_on base
    git -C "$R" checkout -q side
    for f do last "$f"; done
    commit_on side && merge_side
  }
  printf '%s\n' @a/runtest-x "$SWEEP_ALIAS" "$CENSUS_ALIAS" >"$TMP/cover-list"
  { named_by "$SWEEP" "$SWEEP_ALIAS"; named_by "$CENSUS" "$CENSUS_ALIAS"; } >"$TMP/cover-named"
  : >"$TMP/reach"
  covered() { # LABEL SCRIPT ALIAS: a full run built the list, ran no harness and said why
    if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
build --root . -j 4 $(tr '\n' ' ' <"$TMP/cover-list" | sed 's/ $//')" ] &&
       has "harness tier: PENDING -- no harness reaches $2: whole unless" &&
       has "harness tier: $2 is named as a dependency by $3, which the tests tier runs" &&
       has 'harness tier: skipped -- the 1 path(s) the head side changed reach no harness beyond the aliases the tests tier runs' &&
       has 'merge-build-check: passed'; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  }
  whole() { # LABEL WHY: a full run of the whole tier, for WHY
    if [ "$rc" = 0 ] && [ "$(grep '^shell ' "$TMP/record")" = 'shell --jobs 4' ] &&
       has "harness tier: SELECTED -- whole -- $2" && has 'merge-build-check: passed'; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  }
  cover_fixture cover-sweep "$SWEEP"
  COVER_SWEEP=$R
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named run "$TMP/dunebin"
  covered "$SWEEP alone runs no harness: $SWEEP_ALIAS, which the tests tier runs, names it" "$SWEEP" "$SWEEP_ALIAS"
  sed '/^merge-build-check: passed$/q' "$TMP/out" >"$TMP/cover-merge"
  cover_fixture cover-census "$CENSUS"
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named run "$TMP/dunebin"
  covered "$CENSUS alone runs no harness: $CENSUS_ALIAS, which the tests tier runs, names it" "$CENSUS" "$CENSUS_ALIAS"
  cover_fixture cover-unmapped tools/unmapped.sh
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named run "$TMP/dunebin"
  whole 'an unmapped script no alias names runs the whole tier' \
    'no harness reaches tools/unmapped.sh, and no alias the tests tier runs names it as a dependency'
  # The negative control's case: the line is there, its alias is not in the list run here.
  printf '%s\n' @a/runtest-x "$CENSUS_ALIAS" >"$TMP/cover-short"
  FAKE_LIST=$TMP/cover-short FAKE_NAMED=$TMP/cover-named R=$COVER_SWEEP run "$TMP/dunebin"
  whole 'a script named only by an alias outside the list the tests tier runs runs the whole tier' \
    "no harness reaches $SWEEP, and no alias the tests tier runs names it as a dependency"
  # Whole lines only: a longer alias, a longer path, the line of another script.
  { named_by "$SWEEP" "${SWEEP_ALIAS}x"; named_by "${SWEEP}x" "$SWEEP_ALIAS"; named_by "$CENSUS" "$SWEEP_ALIAS"; } >"$TMP/cover-near"
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-near R=$COVER_SWEEP run "$TMP/dunebin"
  whole 'a line that names another alias, path or script covers nothing' \
    "no harness reaches $SWEEP, and no alias the tests tier runs names it as a dependency"
  # A cross-cutting change's first tier is the list run here.
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named \
    OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL R=$COVER_SWEEP owed "$OWED_FULL" add \
    "$(run_line "$(git -C "$COVER_SWEEP" rev-parse 'HEAD^2')" 'runtest -j 4')"
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named \
    OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL R=$COVER_SWEEP run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ -z "$(grep '^shell ' "$TMP/record")" ] &&
     has "harness tier: $SWEEP is named as a dependency by $SWEEP_ALIAS, which the tests tier runs"; then
    report 0 'a cross-cutting change'"'"'s first tier covers what an alias in it names'
  else report 1 'a cross-cutting change'"'"'s first tier covers what an alias in it names' "exit $rc; orc $orc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # With no first tier the list is OWED, not run here: nothing it names is covered.
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting \
    FAKE_TIER_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL R=$COVER_SWEEP run "$TMP/dunebin"
  whole 'a list OWED rather than run covers nothing' \
    "no harness reaches $SWEEP, and the tests tier runs no list here to cover it"
  # A covered script beside one a member reaches: the tier narrows to that member.
  cover_fixture cover-mixed "$SWEEP" tools/sub/a.sh
  reach_of tools/sub/a.sh
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named run "$TMP/linuxbin:$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(grep '^shell ' "$TMP/record")" = "shell --jobs 4 --affected --files $SWEEP tools/sub/a.sh" ] &&
     has 'harness tier: SELECTED -- scoped to what the 2 path(s) the head side changed reach beyond the aliases the tests tier runs:'; then
    report 0 'a covered script beside a reached one runs what the change reaches'
  else report 1 'a covered script beside a reached one runs what the change reaches' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  : >"$TMP/reach"
  # --plan alone asks no list: the tier stays PENDING. A branch plan reads the list and settles it.
  R=$COVER_SWEEP plan
  if [ "$rc" = 0 ] && has "harness tier: PENDING -- no harness reaches $SWEEP: whole unless" &&
     ! has 'is named as a dependency by'; then
    report 0 '--plan alone prints the tier PENDING'
  else report 1 '--plan alone prints the tier PENDING' "exit $rc; $(cat "$TMP/out")"; fi
  fork_tip=$(git -C "$COVER_SWEEP" rev-parse 'HEAD^1')
  git -C "$COVER_SWEEP" checkout -q side
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named R=$COVER_SWEEP run "$TMP/dunebin" --plan --base "$fork_tip"
  if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] &&
     has "harness tier: $SWEEP is named as a dependency by $SWEEP_ALIAS, which the tests tier runs" &&
     has 'harness tier: skipped -- the 1 path(s) the head side changed reach no harness beyond the aliases the tests tier runs'; then
    report 0 'a branch plan prints the alias that covers the script, and why no harness runs'
  else report 1 'a branch plan prints the alias that covers the script, and why no harness runs' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  git -C "$COVER_SWEEP" checkout -q base
  # The build of the base alone reads the same coverage.
  sha=$(git -C "$COVER_SWEEP" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$COVER_SWEEP" checkout -q --detach "$sha"
  FAKE_LIST=$TMP/cover-list FAKE_NAMED=$TMP/cover-named R=$COVER_SWEEP run "$TMP/dunebin"
  if [ "$rc" = 0 ] && cmp -s <(sed '/^merge-build-check: passed$/q' "$TMP/out") "$TMP/cover-merge"; then
    report 0 'the build of the base alone reads the same coverage as the merge'
  else report 1 'the build of the base alone reads the same coverage as the merge' "exit $rc; $(diff "$TMP/cover-merge" <(sed '/^merge-build-check: passed$/q' "$TMP/out"))"; fi
fi

# Scoped owed Linux shell leg (gh-ocannl-2414): on non-Linux hosts, a head-side change to a
# `*.sh` reached by the harness census or a sweep-driver rule owes a Linux shell-leg record;
# Linux satisfies it by itself; non-shell edits and unreached/non-sweep scripts owe nothing new.
linux_shell_leg() {
  local darwin_bin=$TMP/darwinbin:$TMP/dunebin linux_bin=$TMP/linuxbin:$TMP/dunebin
  local head_sha tree lr_repo lr_line hook leg_path state_root receipt_dir rc_leg
  fixture linux-shell
  cp "$HERE/linux-shell-leg.sh" "$R/tools/" || { harness_abandon 'could not copy linux-shell-leg.sh'; return; }
  hook=$(sed -n 's/^owed-record //p' "$HERE/../.github/fleet-leg")
  cp "$HERE/../$hook" "$R/tools/" || { harness_abandon 'could not copy owed-run-report.sh'; return; }
  printf '%s\n' tools/test-one.sh tools/test-two.sh >"$TMP/ls-manifest"
  commit_on base && git -C "$R" branch -f side HEAD || { harness_abandon 'could not seed linux-shell base'; return; }
  first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  head_sha=$(git -C "$R" rev-parse 'HEAD^2')
  tree=$(merged_tree)
  lr_repo=$TMP/owed-linux-shell.git
  git init -q --bare "$lr_repo"

  # 1. Non-Linux (Darwin) negative control: census-reached script refuses without a Linux record.
  printf '%s\ttools/test-one.sh\n' tools/sub/a.sh >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] &&
     has 'owed-run needs: linux:member:tools/test-one.sh (record does not admit; owed-run.sh exit 1)' &&
     has "linux-shell leg: fleet-worker.sh execution leg --request <id> --rev $head_sha --checkout <linux-clone> --worktree <linux-worktree> --cpu --wait-window -- tools/linux-shell-leg.sh --harness tools/test-one.sh" &&
     grep -qxF 'ship-pr-merge-plan: unadmitted linux:member:tools/test-one.sh' "$TMP/out"; then
    report 0 'Darwin --plan-merge names the owed Linux harness leg and its fleet-leg command'
  else report 1 'Darwin --plan-merge names the owed Linux harness leg and its fleet-leg command' "exit $rc; $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
     has 'tests tier: OWED -- the change touches shell source(s) reached by the harness census or sweep driver on non-Linux (Darwin)' &&
     has "linux-shell leg: fleet-worker.sh execution leg --request <id> --rev $head_sha" &&
     has "no owed-run record for ${head_sha:0:8} admits the merged tree $tree"; then
    report 0 'Darwin merge build refuses a census-reached shell change without a Linux record'
  else report 1 'Darwin merge build refuses a census-reached shell change without a Linux record' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi

  # 2. Same merge on Linux satisfies the leg by itself without any record.
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$linux_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: nothing-owed' "$TMP/out" && ! has 'linux-shell leg:'; then
    report 0 'Linux --plan-merge owes nothing new for the same shell change'
  else report 1 'Linux --plan-merge owes nothing new for the same shell change' "exit $rc; $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$linux_bin"
  if [ "$rc" = 0 ] && has 'merge-build-check: passed' && ! has 'owed-run'; then
    report 0 'Linux merge build satisfies the shell leg by itself without reading a record'
  else report 1 'Linux merge build satisfies the shell leg by itself without reading a record' "exit $rc; $(cat "$TMP/out")"; fi

  # 3. Sweep-driver-reached script (no harness reach, #2244 shape) owes linux:@test/operations/runtest-sweep_harness.
  : >"$TMP/reach"
  printf '%s\n%s\n' \
    'affected: tools/sub/a.sh: named as a dependency by the rule of @test/operations/runtest-sweep_harness' \
    'affected: tools/sub/a.sh: reached by sweep driver alias @test/operations/runtest-sweep_harness' >"$TMP/ls-sweep-named"
  printf '%s\n' '@test/operations/runtest-sweep_harness' >"$TMP/ls-sweep-list"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-sweep-named run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] &&
     has '-- tools/linux-shell-leg.sh --dune @test/operations/runtest-sweep_harness' &&
     grep -qxF 'ship-pr-merge-plan: unadmitted linux:@test/operations/runtest-sweep_harness' "$TMP/out"; then
    report 0 'Darwin --plan-merge owes linux:@test/operations/runtest-sweep_harness for a sweep-driver-only shell change'
  else report 1 'Darwin --plan-merge owes linux:@test/operations/runtest-sweep_harness for a sweep-driver-only shell change' "exit $rc; $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-sweep-named run "$darwin_bin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] &&
     has "no owed-run record for ${head_sha:0:8} admits the merged tree $tree"; then
    report 0 'Darwin merge build refuses a sweep-driver-reached shell change without a Linux record'
  else report 1 'Darwin merge build refuses a sweep-driver-reached shell change without a Linux record' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi

  # 4. Scoping boundaries: unreached non-sweep .sh, non-sweep named dep, and non-.sh reaching a harness owe nothing new.
  printf '%s\n' 'affected: tools/sub/a.sh: named as a dependency by the rule of @test/operations/runtest-sweep_harness' >"$TMP/ls-nonsweep-named"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-nonsweep-named run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: nothing-owed' "$TMP/out"; then
    report 0 'a non-sweep named-dependency shell change with no harness reach owes no Linux leg'
  else report 1 'a non-sweep named-dependency shell change with no harness reach owes no Linux leg' "exit $rc; $(cat "$TMP/out")"; fi
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: nothing-owed' "$TMP/out"; then
    report 0 'an unreached non-sweep *.sh script owes no Linux leg (not widened to every *.sh)'
  else report 1 'an unreached non-sweep *.sh script owes no Linux leg (not widened to every *.sh)' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\ttools/test-one.sh\n' x.ml >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY R=$APART run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: nothing-owed' "$TMP/out"; then
    report 0 'a non-shell edit reaching a harness owes no Linux leg'
  else report 1 'a non-shell edit reaching a harness owes no Linux leg' "exit $rc; $(cat "$TMP/out")"; fi
  R=$TMP/linux-shell

  # 5. Invalid or absent sweep-driver alias fails closed.
  printf '%s\n' 'affected: tools/sub/a.sh: reached by sweep driver alias @test/operations/runtest-absent' >"$TMP/ls-bad-sweep"
  : >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_NAMED=$TMP/ls-bad-sweep run "$darwin_bin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'affected named a sweep-driver alias this tree does not have: @test/operations/runtest-absent'; then
    report 0 'an absent sweep-driver alias fails closed with no plan verdict'
  else report 1 'an absent sweep-driver alias fails closed with no plan verdict' "exit $rc; $(cat "$TMP/out")"; fi

  # 6. Full-manifest reach collapses to linux:harnesses + --all-harnesses.
  printf '%s\ttools/test-one.sh\n%s\ttools/test-two.sh\n' tools/sub/a.sh tools/sub/a.sh >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: unadmitted linux:harnesses' "$TMP/out" &&
     has '-- tools/linux-shell-leg.sh --all-harnesses'; then
    report 0 'reaching every manifest member collapses to linux:harnesses and --all-harnesses'
  else report 1 'reaching every manifest member collapses to linux:harnesses and --all-harnesses' "exit $rc; $(cat "$TMP/out")"; fi

  # 7. owed-run.sh record validation: Mac v2 run never satisfies linux:*; non-Linux / bad records refused;
  #    union of linux-shell records and harnesses wildcard admit.
  printf '%s\ttools/test-one.sh\n' tools/sub/a.sh >"$TMP/reach"
  plant "$lr_repo" "$head_sha" "$head_sha^{tree}" "$head_sha" "owed-run record

$(run_line "$head_sha" 'build @runtest @train -j 4')"
  OCANNL_TOOL_OWED_RUN_URL=$lr_repo FAKE_MANIFEST=$TMP/ls-manifest FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-sweep-named \
    run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] &&
     has 'covers no run of: linux:member:tools/test-one.sh linux:@test/operations/runtest-sweep_harness' &&
     grep -qxF 'ship-pr-merge-plan: unadmitted linux:member:tools/test-one.sh linux:@test/operations/runtest-sweep_harness' "$TMP/out"; then
    report 0 'a v1/v2 cc runtest record without os=Linux never satisfies linux:* needs'
  else report 1 'a v1/v2 cc runtest record without os=Linux never satisfies linux:* needs' "exit $rc; $(cat "$TMP/out")"; fi
  for bad_ls in \
    "owed-run: v2 linux-shell head=$head_sha writer=record exit=0 clean=yes os=Darwin source=mac:/r/1 targets=member:tools/test-one.sh|on OS Darwin, not Linux|non-Linux os" \
    "owed-run: v2 linux-shell head=$head_sha writer=record exit=1 clean=yes os=Linux source=rog:/r/1 targets=member:tools/test-one.sh|a run that exited 1|nonzero exit" \
    "owed-run: v2 linux-shell head=$head_sha writer=record exit=0 clean=no os=Linux source=rog:/r/1 targets=member:tools/test-one.sh|a run of a dirty checkout|dirty checkout" \
    "owed-run: v2 linux-shell head=$head_sha writer=waive exit=0 clean=yes os=Linux source=rog:/r/1 targets=member:tools/test-one.sh|holds a line no v1 or v2 record reads|waiver writer" \
    "owed-run: v2 linux-shell head=$head_sha writer=record exit=0 clean=yes os=Linux source=rog:/r/1 targets=member:../bad.sh|holds a line no v1 or v2 record reads|path traversal target"; do
    lr_line=${bad_ls%%|*} rest=${bad_ls#*|}
    plant "$lr_repo" "$head_sha" "$head_sha^{tree}" "$head_sha" "owed-run record

$lr_line"
    owed "$lr_repo" check "$head_sha" linux:member:tools/test-one.sh
    if [ "$orc" = 1 ] && grep -qF "${rest%%|*}" "$TMP/owed-out"; then
      report 0 "owed-run check refuses linux-shell record with ${rest#*|}"
    else report 1 "owed-run check refuses linux-shell record with ${rest#*|}" "exit $orc; $(cat "$TMP/owed-out")"; fi
  done
  # Union of harnesses wildcard + sweep alias admits both; linux-shell never satisfies runtest.
  plant "$lr_repo" "$head_sha" "$head_sha^{tree}" "$head_sha" "owed-run record

owed-run: v2 linux-shell head=$head_sha writer=leg:rog-nv-linux exit=0 clean=yes os=Linux source=rog-nv-linux:/r/1 targets=harnesses
owed-run: v2 linux-shell head=$head_sha writer=record exit=0 clean=yes os=Linux source=rog-nv-linux:/r/2 targets=@test/operations/runtest-sweep_harness"
  OCANNL_TOOL_OWED_RUN_URL=$lr_repo FAKE_MANIFEST=$TMP/ls-manifest FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-sweep-named \
    run "$darwin_bin"
  if [ "$rc" = 0 ] && has 'merge-build-check: passed' &&
     has "owed-run: admitted linux-shell targets 'harnesses' for ${head_sha:0:8} (rog-nv-linux:/r/1, writer=leg:rog-nv-linux)"; then
    report 0 'Darwin merge build admits union of Linux harnesses wildcard and sweep-driver alias records'
  else report 1 'Darwin merge build admits union of Linux harnesses wildcard and sweep-driver alias records' "exit $rc; $(cat "$TMP/out")"; fi
  owed "$lr_repo" check "$head_sha" runtest
  if [ "$orc" = 1 ] && grep -qF 'covers no run of: runtest' "$TMP/owed-out"; then
    report 0 'a linux-shell record never satisfies runtest'
  else report 1 'a linux-shell record never satisfies runtest' "exit $orc; $(cat "$TMP/owed-out")"; fi

  # 8. tools/linux-shell-leg.sh + tools/owed-run-report.sh end-to-end (with and without FLEET_LEG_RECEIPT_DIR).
  git -C "$R" checkout -q side
  state_root=$TMP/ls-state
  rm -rf "$state_root"
  rc_leg=0
  (cd "$R" && PATH=$darwin_bin:$TMP/gitbin:$SYS_PATH OCANNL_TOOL_TEST_RUNS=$state_root \
    "$BASH" tools/linux-shell-leg.sh --all-harnesses) >"$TMP/ls-darwin.out" 2>&1 || rc_leg=$?
  if [ "$rc_leg" = 2 ] && grep -qF "must run on Linux (host OS is 'Darwin')" "$TMP/ls-darwin.out"; then
    report 0 'tools/linux-shell-leg.sh refuses to run on non-Linux hosts'
  else report 1 'tools/linux-shell-leg.sh refuses to run on non-Linux hosts' "exit $rc_leg; $(cat "$TMP/ls-darwin.out")"; fi
  leg_path=$TMP/ls-fleet-leg-rec
  touch "$leg_path"
  rm -f "$TMP/ls-receipt-link"
  : >"$TMP/record"
  rc_leg=0
  (cd "$R" && PATH=$linux_bin:$TMP/gitbin:$SYS_PATH OCANNL_TOOL_TEST_RUNS=$state_root \
    FAKE_RECORD=$TMP/record FAKE_MANIFEST=$TMP/ls-manifest FAKE_HARNESS_EMIT=1 \
    FLEET_LEG_RUN_RECORD=$leg_path FLEET_LEG_RECEIPT_LINK=$TMP/ls-receipt-link \
    "$BASH" tools/linux-shell-leg.sh --harness tools/test-one.sh --dune @test/operations/runtest-sweep_harness) \
    >"$TMP/ls-run.out" 2>&1 || rc_leg=$?
  receipt_dir=$(readlink "$TMP/ls-receipt-link" 2>/dev/null || true)
  if [ "$rc_leg" = 0 ] && [ -n "$receipt_dir" ] && [ -d "$receipt_dir" ] &&
     [ "$(cat "$receipt_dir/exit")" = 0 ] && [ "$(cat "$receipt_dir/os")" = Linux ] &&
     [ "$(cat "$receipt_dir/targets")" = '@test/operations/runtest-sweep_harness,member:tools/test-one.sh' ] &&
     grep -qxF 'test-run run --expect-backend cc build -j 4 @test/operations/runtest-sweep_harness' "$TMP/record" &&
     grep -qxF 'shell --jobs 4 --member tools/test-one.sh' "$TMP/record"; then
    report 0 'tools/linux-shell-leg.sh runs dune via tools/test-run.sh + --member harnesses on Linux and links a clean receipt'
  else report 1 'tools/linux-shell-leg.sh runs dune via tools/test-run.sh + --member harnesses on Linux and links a clean receipt' \
    "exit $rc_leg receipt=$receipt_dir record=$(cat "$TMP/record") out=$(cat "$TMP/ls-run.out")"; fi
  # Negative control: a skip-only harness (0 PASS / unexpected SKIP) is refused and never writes a passing receipt.
  rc_skip=0
  (cd "$R" && PATH=$linux_bin:$TMP/gitbin:$SYS_PATH OCANNL_TOOL_TEST_RUNS="$state_root-skip" \
    FAKE_RECORD=$TMP/record FAKE_MANIFEST=$TMP/ls-manifest FAKE_HARNESS_EMIT=1 FAKE_HARNESS_SKIP_ONLY=1 \
    "$BASH" tools/linux-shell-leg.sh --harness tools/test-one.sh) >"$TMP/ls-skip.out" 2>&1 || rc_skip=$?
  skip_rec=$(sed -n 's/.*receipt: \([^)]*\)).*/\1/p' "$TMP/ls-skip.out" | tail -n 1)
  if [ "$rc_skip" = 1 ] && [ -n "$skip_rec" ] && [ "$(cat "$skip_rec/exit" 2>/dev/null)" = 1 ] &&
     grep -qF 'REFUSED unexpected skip' "$TMP/ls-skip.out"; then
    report 0 'tools/linux-shell-leg.sh refuses a skip-only harness run and records nonzero exit'
  else report 1 'tools/linux-shell-leg.sh refuses a skip-only harness run and records nonzero exit' \
    "exit $rc_skip rec=$skip_rec out=$(cat "$TMP/ls-skip.out")"; fi
  # Publish via tools/owed-run-report.sh using pointer fallback (no FLEET_LEG_RECEIPT_DIR), then verify Darwin merge admits it.
  pub_repo=$TMP/owed-linux-pub.git
  git init -q --bare "$pub_repo"
  hrc=0
  (cd "$R" && PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_TEST_RUNS=$state_root OCANNL_TOOL_OWED_RUN_URL=$pub_repo \
    "./$hook" "$leg_path" ls-req "$head_sha") >"$TMP/ls-hook.out" 2>&1 || hrc=$?
  git -C "$R" checkout -q base
  OCANNL_TOOL_OWED_RUN_URL=$pub_repo FAKE_MANIFEST=$TMP/ls-manifest FAKE_LIST=$TMP/ls-sweep-list FAKE_NAMED=$TMP/ls-sweep-named \
    run "$darwin_bin"
  if [ "$hrc" = 0 ] && [ "$rc" = 0 ] && has 'merge-build-check: passed' &&
     has "admitted linux-shell targets '@test/operations/runtest-sweep_harness,member:tools/test-one.sh'"; then
    report 0 'owed-run-report.sh publishes a linux-shell leg receipt via pointer fallback and Darwin merge admits it'
  else report 1 'owed-run-report.sh publishes a linux-shell leg receipt via pointer fallback and Darwin merge admits it' \
    "hook=$hrc ($(cat "$TMP/ls-hook.out")) merge=$rc ($(cat "$TMP/out"))"; fi
  # Publish via FLEET_LEG_RECEIPT_DIR and manual record --print + add round-trip.
  pub_repo2=$TMP/owed-linux-pub2.git
  git init -q --bare "$pub_repo2"
  git -C "$R" checkout -q side
  hrc=0
  (cd "$R" && PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$pub_repo2 FLEET_LEG_RECEIPT_DIR=$receipt_dir \
    "./$hook" "$leg_path" ls-req-2 "$head_sha") >"$TMP/ls-hook2.out" 2>&1 || hrc=$?
  owed "$pub_repo2" check "$head_sha" linux:member:tools/test-one.sh linux:@test/operations/runtest-sweep_harness
  if [ "$hrc" = 0 ] && [ "$orc" = 0 ] && grep -qF "writer=leg:" "$TMP/owed-out"; then
    report 0 'owed-run-report.sh publishes via FLEET_LEG_RECEIPT_DIR and check admits writer=leg'
  else report 1 'owed-run-report.sh publishes via FLEET_LEG_RECEIPT_DIR and check admits writer=leg' \
    "hook=$hrc ($(cat "$TMP/ls-hook2.out")) check=$orc ($(cat "$TMP/owed-out"))"; fi
  owed "$OWED_EMPTY" record "$receipt_dir" --print
  printed_ls=$(cat "$TMP/owed-out")
  pub_repo3=$TMP/owed-linux-pub3.git
  git init -q --bare "$pub_repo3"
  owed "$pub_repo3" add "$printed_ls" && add_rc=$orc
  owed "$pub_repo3" check "$head_sha" linux:member:tools/test-one.sh linux:@test/operations/runtest-sweep_harness
  if [ "$add_rc" = 0 ] && [ "$orc" = 0 ] && grep -qF 'writer=relay)' "$TMP/owed-out"; then
    report 0 'record --print + add round-trips a linux-shell receipt with writer=relay'
  else report 1 'record --print + add round-trips a linux-shell receipt with writer=relay' \
    "add=$add_rc check=$orc line=$printed_ls out=$(cat "$TMP/owed-out")"; fi
  # Negative controls on leg-record for linux-shell receipts: wrong leg record path and non-Linux OS refused.
  pub_bad=$TMP/owed-linux-bad.git
  git init -q --bare "$pub_bad"
  hrc=0
  (cd "$R" && PATH=$TMP/gitbin:$SYS_PATH OCANNL_TOOL_OWED_RUN_URL=$pub_bad FLEET_LEG_RECEIPT_DIR=$receipt_dir \
    "./$hook" "$TMP/other-leg-rec" ls-req-bad "$head_sha") >"$TMP/ls-hook-bad.out" 2>&1 || hrc=$?
  if [ "$hrc" = 0 ] && grep -qF "does not match this fleet leg's run record" "$TMP/ls-hook-bad.out" &&
     [ -z "$(git -C "$R" ls-remote "$pub_bad" "refs/owed-runs/$head_sha")" ]; then
    report 0 'owed-run-report.sh refuses a linux-shell receipt with a mismatched fleet leg record'
  else report 1 'owed-run-report.sh refuses a linux-shell receipt with a mismatched fleet leg record' \
    "hook=$hrc ($(cat "$TMP/ls-hook-bad.out"))"; fi
  # --all on Darwin widens the local harness tier without bypassing the owed Linux shell leg.
  git -C "$R" checkout -q base
  printf '%s\ttools/test-one.sh\n' tools/sub/a.sh >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --all
  if [ "$rc" = 1 ] && has 'owed-run: REFUSED' && has 'tools/linux-shell-leg.sh --harness tools/test-one.sh' &&
     [ "$(cat "$TMP/record")" = "$COMPILE" ]; then
    report 0 '--all on Darwin still refuses without the owed Linux shell record'
  else report 1 '--all on Darwin still refuses without the owed Linux shell record' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi

  # 9. Linux build whose tests tier omits a reached sweep-driver alias (e.g. OWED list) still owes it.
  : >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting \
    FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting FAKE_NAMED=$TMP/ls-sweep-named run "$linux_bin" --plan-merge
  if [ "$rc" = 0 ] &&
     grep -qxF 'ship-pr-merge-plan: unadmitted runtest linux:@test/operations/runtest-sweep_harness' "$TMP/out" &&
     has '-- tools/linux-shell-leg.sh --dune @test/operations/runtest-sweep_harness'; then
    report 0 'Linux --plan-merge still owes a reached sweep-driver alias omitted from its tests tier'
  else report 1 'Linux --plan-merge still owes a reached sweep-driver alias omitted from its tests tier' "exit $rc; $(cat "$TMP/out")"; fi

  # 10. Sweep-driver reach unavailable (disabled Dune probe) fails closed with no plan verdict.
  printf '%s\n' 'affected: tools/sub/a.sh: sweep-driver reach unavailable (dune probe did not run)' >"$TMP/ls-unavail"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_NAMED=$TMP/ls-unavail run "$darwin_bin" --plan-merge
  if [ "$rc" = 1 ] && [ "$(grep -c '^ship-pr-merge-plan:' "$TMP/out")" = 0 ] &&
     has 'sweep-driver reach for tools/sub/a.sh could not be determined'; then
    report 0 'unavailable sweep-driver reach fails closed with no plan verdict'
  else report 1 'unavailable sweep-driver reach fails closed with no plan verdict' "exit $rc; $(cat "$TMP/out")"; fi

  # 11. Base-added harness member absent from PR head collapses to linux:harnesses / --all-harnesses.
  fixture linux-shell-base-added
  printf '%s\n' tools/test-one.sh tools/test-base-new.sh >"$TMP/ls-base-manifest"
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side
  git -C "$R" checkout -q base && printf '#!/bin/sh\n:\n' >"$R/tools/test-base-new.sh" && commit_on base && merge_side
  printf '%s\ttools/test-base-new.sh\n' tools/sub/a.sh >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-base-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: unadmitted linux:harnesses' "$TMP/out" &&
     has '-- tools/linux-shell-leg.sh --all-harnesses'; then
    report 0 'a base-added harness member absent from the PR head collapses to linux:harnesses and --all-harnesses'
  else report 1 'a base-added harness member absent from the PR head collapses to linux:harnesses and --all-harnesses' "exit $rc; $(cat "$TMP/out")"; fi

  # 12. Base-side rename to *.sh queries sweep-driver reach for the renamed path.
  fixture linux-shell-rename
  git -C "$R" checkout -q base && git -C "$R" mv r.txt tools/sub/renamed.sh && commit_on base
  git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
  : >"$TMP/reach"
  printf '%s\n' 'affected: tools/sub/renamed.sh: reached by sweep driver alias @test/operations/runtest-sweep_harness' >"$TMP/ls-ren-named"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_RENAME_NAMED=$TMP/ls-ren-named run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] &&
     grep -qxF 'ship-pr-merge-plan: unadmitted linux:@test/operations/runtest-sweep_harness' "$TMP/out"; then
    report 0 'a base-side rename to *.sh queries sweep-driver reach for the renamed path'
  else report 1 'a base-side rename to *.sh queries sweep-driver reach for the renamed path' "exit $rc; $(cat "$TMP/out")"; fi

  # 13. Quoted paths on Darwin: decoded via git diff -z so non-shell edits owe nothing, clean shell
  #     edits keep exact scoped reach, and quoted shell paths stay fail-closed yet satisfiable.
  fixture linux-shell-quoted-nonsh
  git -C "$R" checkout -q base && printf 'q\n' >"$R/\"quoted\".txt" && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  : >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: nothing-owed' "$TMP/out"; then
    report 0 'a base-side quoted non-shell path owes no Linux leg for a non-shell PR on Darwin'
  else report 1 'a base-side quoted non-shell path owes no Linux leg for a non-shell PR on Darwin' "exit $rc; $(cat "$TMP/out")"; fi

  fixture linux-shell-quoted-clean-sh
  git -C "$R" checkout -q base && printf 'q\n' >"$R/\"quoted\".txt" && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  q_head=$(git -C "$R" rev-parse 'HEAD^2')
  q_repo=$TMP/owed-linux-quoted.git
  git init -q --bare "$q_repo"
  printf '%s\ttools/test-one.sh\n' tools/sub/a.sh >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: unadmitted linux:member:tools/test-one.sh' "$TMP/out"; then
    report 0 'a base-side quoted path beside a clean shell edit keeps scoped Linux reach on Darwin'
  else report 1 'a base-side quoted path beside a clean shell edit keeps scoped Linux reach on Darwin' "exit $rc; $(cat "$TMP/out")"; fi
  plant "$q_repo" "$q_head" "$q_head^{tree}" "$q_head" "owed-run record

owed-run: v2 linux-shell head=$q_head writer=record exit=0 clean=yes os=Linux source=rog:/r/q1 targets=member:tools/test-one.sh"
  OCANNL_TOOL_OWED_RUN_URL=$q_repo FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: admitted linux:member:tools/test-one.sh' "$TMP/out"; then
    report 0 'a matching Linux record admits a clean shell edit beside a base-side quoted path'
  else report 1 'a matching Linux record admits a clean shell edit beside a base-side quoted path' "exit $rc; $(cat "$TMP/out")"; fi

  fixture linux-shell-quoted-sh
  git -C "$R" checkout -q side && printf '#!/bin/sh\n:\n' >"$R/\"quoted\".sh" && commit_on side && merge_side
  qs_head=$(git -C "$R" rev-parse 'HEAD^2')
  qs_repo=$TMP/owed-linux-quoted-sh.git
  git init -q --bare "$qs_repo"
  : >"$TMP/reach"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: unadmitted linux:harnesses' "$TMP/out" &&
     has '-- tools/linux-shell-leg.sh --all-harnesses'; then
    report 0 'a head-side quoted *.sh path owes linux:harnesses on Darwin (fail-closed and satisfiable)'
  else report 1 'a head-side quoted *.sh path owes linux:harnesses on Darwin (fail-closed and satisfiable)' "exit $rc; $(cat "$TMP/out")"; fi
  plant "$qs_repo" "$qs_head" "$qs_head^{tree}" "$qs_head" "owed-run record

owed-run: v2 linux-shell head=$qs_head writer=record exit=0 clean=yes os=Linux source=rog:/r/q2 targets=harnesses"
  OCANNL_TOOL_OWED_RUN_URL=$qs_repo FAKE_MANIFEST=$TMP/ls-manifest run "$darwin_bin" --plan-merge
  if [ "$rc" = 0 ] && grep -qxF 'ship-pr-merge-plan: admitted linux:harnesses' "$TMP/out"; then
    report 0 'a Linux --all-harnesses record admits a head-side quoted *.sh path on Darwin'
  else report 1 'a Linux --all-harnesses record admits a head-side quoted *.sh path on Darwin' "exit $rc; $(cat "$TMP/out")"; fi
  : >"$TMP/reach"
}
harness_leg_run 'scoped owed Linux shell leg on non-Linux hosts, Linux self-satisfaction and leg publication #linux-shell' linux_shell_leg

# The real tools/test-harnesses.sh on the real path of an ask, its `affected --harnesses` a
# stand-in printing FAKE_COMMANDS: the allowlist and the group are that script's, read here as the
# scope reads them. Its first case is the control: the stand-in's answer narrows the tier.
if harness_leg 'the real ask: an unrecognized line, an option-like path, another group #scope'; then
  fixture real-ask
  first tools/sub/a.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last x.ml && commit_on side && merge_side
  cp "$HERE/test-harnesses.sh" "$R/tools/"
  printf '#!/bin/sh\ncat "$FAKE_COMMANDS"\n' >"$R/tools/test-run.sh"
  export FAKE_COMMANDS=$TMP/commands
  shell_member=$("$BASH" "$HERE/test-harnesses.sh" --shell --list | sed -n 1p)
  other_member=$("$BASH" "$HERE/test-harnesses.sh" --promotion --list | sed -n 1p)
  printf '%s\n' "$shell_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has "merge-build-check:   $shell_member"; then
    report 0 'a member of the tier narrows it through the real ask'
  else report 1 'a member of the tier narrows it through the real ask' "exit $rc; $(cat "$TMP/out")"; fi
  # One leg of a member: the selector stands in the plan, as the scoped run gets it.
  printf "%s --only '^1:'\n" "$shell_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has "merge-build-check:   $shell_member --only '^1:'"; then
    report 0 'a change reaching one leg narrows the tier to that leg'
  else report 1 'a change reaching one leg narrows the tier to that leg' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\n' "$other_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- scoped to' && has "merge-build-check:   $other_member"; then
    report 0 'a member of another group narrows the tier too: every group is the tier'
  else report 1 'a member of another group narrows the tier too: every group is the tier' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s; touch evaluated\n' "$shell_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- whole -- what tools/sub/a.sh reaches could not be asked (test-harnesses.sh exit 2)' &&
     [ ! -e "$R/evaluated" ]; then
    report 0 'a line the allowlist does not recognize runs the whole tier'
  else report 1 'a line the allowlist does not recognize runs the whole tier' "exit $rc; $(cat "$TMP/out")"; fi
  # A changed path reading as an option: the same stand-in, which would answer, is never asked.
  fixture option-path
  printf '1\n2\n3\n4\n5\n6\n' >"$R/-dash.sh"
  git -C "$R" add -- ./-dash.sh && git -C "$R" commit -q -m 'a script named like an option' &&
    git -C "$R" branch -f side
  first ./-dash.sh && commit_on base
  git -C "$R" checkout -q side && last ./-dash.sh && commit_on side && merge_side
  cp "$HERE/test-harnesses.sh" "$R/tools/"
  printf '#!/bin/sh\ncat "$FAKE_COMMANDS"\n' >"$R/tools/test-run.sh"
  printf '%s\n' "$shell_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has 'harness tier: SELECTED -- whole -- what -dash.sh reaches could not be asked (test-harnesses.sh exit 2)'; then
    report 0 'a changed path reading as an option runs the whole tier'
  else report 1 'a changed path reading as an option runs the whole tier' "exit $rc; $(cat "$TMP/out")"; fi
  unset FAKE_COMMANDS
fi

# This repository's own mapping, asked as the scope asks it: what a both-sides change to shared
# support reaches is the whole tier, every member loading it, and a manifest edit reaches the
# members that read the manifest, a small part of the tier (gh-ocannl-1994).
if harness_leg 'this repository: shared support reaches the tier, a manifest edit part of it #mapping'; then
  members_of() { "$BASH" "$HERE/test-harnesses.sh" "$@" 2>/dev/null | LC_ALL=C sort; }
  tier=$(members_of --list)
  support=$(members_of --affected --list --files scripts/harness-support.sh)
  manifest=$(members_of --affected --list --files tools/test-harnesses.sh)
  if [ -n "$tier" ] && [ "$support" = "$tier" ]; then
    report 0 'scripts/harness-support.sh reaches every member of the tier'
  else report 1 'scripts/harness-support.sh reaches every member of the tier' "tier: $tier; reached: $support"; fi
  outside=$(LC_ALL=C comm -13 <(printf '%s\n' "$tier") <(printf '%s\n' "$manifest"))
  # The members of tools/test-run.sh's harness are most of the tier's time: a literal naming the
  # manifest in one would hand it every manifest edit, for a harness that never reads the manifest.
  if [ -n "$manifest" ] && [ "$manifest" != "$tier" ] && [ -z "$outside" ] &&
     ! grep -q '^tools/test-test-run-' <<<"$manifest"; then
    report 0 'a manifest edit reaches some of the tier, not all of it, and nothing outside it'
  else report 1 'a manifest edit reaches some of the tier, not all of it, and nothing outside it' "reached: $manifest; outside the tier: $outside"; fi
fi
unset FAKE_REACH

if harness_leg 'without dune on PATH the opam switch supplies it #toolchain'; then
  R=$APART run "$TMP/opambin"
  if [ "$rc" = 0 ] && has 'dune is not on PATH; sourcing tools/opam-env.sh' &&
     [ "$(cat "$TMP/record")" = 'build --root . @check @test/operations/scans' ]; then
    report 0 'opam env puts the switch dune on PATH'
  else report 1 'opam env puts the switch dune on PATH' "exit $rc; $(cat "$TMP/out")"; fi
  R=$APART run "$TMP/emptyopam"
  if [ "$rc" = 1 ] && has 'no dune on PATH even after opam env' && [ ! -s "$TMP/record" ]; then
    report 0 'a switch without dune exits 1, building nothing'
  else report 1 'a switch without dune exits 1, building nothing' "exit $rc; $(cat "$TMP/out")"; fi
fi

sys_path_pin_leg() {
  local outer_bash outer_awk outer_2244 want_bash want_awk sys_dune sys_opam run_out owed_out
  local stub_dir mode got override_dir phys_real phys_decoy
  real_exec() {
    perl -MCwd=realpath -e 'my $r = realpath($ARGV[0]); exit 1 unless defined $r; print $r' "$1"
  }
  check_pin_probe() { # FILE EXPECT_SHEBANG EXPECT_DUNE
    local f=$1 expect_shebang=$2 expect_dune=$3
    local line k v sb= pb= pa= a2244= hd= ho=
    [ -f "$f" ] || return 1
    while IFS= read -r line; do
      k=${line%%=*} v=${line#*=}
      case $k in
        shebang_bash) sb=$v ;;
        path_bash) pb=$v ;;
        path_awk) pa=$v ;;
        awk_2244) a2244=$v ;;
        has_dune) hd=$v ;;
        has_opam) ho=$v ;;
      esac
    done <"$f"
    [ -n "$pb" ] && [ "$(real_exec "$pb")" = "$want_bash" ] &&
      [ -n "$pa" ] && [ "$(real_exec "$pa")" = "$want_awk" ] &&
      [ "$a2244" = "$outer_2244" ] && [ "$ho" = "$sys_opam" ] || return 1
    if [ "$expect_shebang" = 1 ]; then
      [ -n "$sb" ] && [ "$(real_exec "$sb")" = "$want_bash" ] || return 1
    fi
    if [ "$expect_dune" = 1 ]; then
      [ "$hd" = "$sys_dune" ] || return 1
    fi
  }
  outer_bash=$(type -P bash 2>/dev/null || :)
  outer_awk=$(type -P awk 2>/dev/null || :)
  [ -n "$outer_bash" ] && outer_bash=$(real_exec "$outer_bash") || { harness_abandon 'could not resolve caller bash'; return; }
  [ -n "$outer_awk" ] && outer_awk=$(real_exec "$outer_awk") || { harness_abandon 'could not resolve caller awk'; return; }
  outer_2244=$(awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }')
  # Under MSYS default copy mode, `ln -s` copies rather than symlinks; accept a verified copy there
  # while keeping strict canonical-path equality whenever pinbin holds real symlinks.
  if [ -L "$TMP/pinbin/bash" ]; then
    want_bash=$outer_bash
  elif cmp -s "$TMP/pinbin/bash" "$outer_bash"; then
    want_bash=$(real_exec "$TMP/pinbin/bash")
  else
    want_bash=$outer_bash
  fi
  if [ -L "$TMP/pinbin/awk" ]; then
    want_awk=$outer_awk
  elif cmp -s "$TMP/pinbin/awk" "$outer_awk"; then
    want_awk=$(real_exec "$TMP/pinbin/awk")
  else
    want_awk=$outer_awk
  fi
  # SYS_PATH retains /usr/bin:/bin, so compare against whatever system dune/opam exist there
  # rather than assuming /usr/bin has no opam.
  sys_dune=$(PATH=/usr/bin:/bin type -P dune 2>/dev/null || :)
  sys_opam=$(PATH=/usr/bin:/bin type -P opam 2>/dev/null || :)

  fixture sys-path-pin
  first x.ml && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  cat >"$R/tools/fmt-check.sh" <<'PROBE'
#!/usr/bin/env bash
{
  printf 'shebang_bash=%s\n' "$BASH"
  printf 'path_bash=%s\n' "$(type -P bash 2>/dev/null || :)"
  printf 'path_awk=%s\n' "$(type -P awk 2>/dev/null || :)"
  printf 'awk_2244=%s\n' "$(awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }')"
  printf 'has_opam=%s\n' "$(type -P opam 2>/dev/null || :)"
} >"$FAKE_PIN_OUT"
exit 0
PROBE
  cat >"$R/tools/owed-run.sh" <<'PROBE'
#!/usr/bin/env bash
{
  printf 'path_bash=%s\n' "$(type -P bash 2>/dev/null || :)"
  printf 'path_awk=%s\n' "$(type -P awk 2>/dev/null || :)"
  printf 'awk_2244=%s\n' "$(awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }')"
  printf 'has_dune=%s\n' "$(type -P dune 2>/dev/null || :)"
  printf 'has_opam=%s\n' "$(type -P opam 2>/dev/null || :)"
} >"$FAKE_PIN_OUT"
exit 0
PROBE
  chmod +x "$R/tools/fmt-check.sh" "$R/tools/owed-run.sh"

  # On Linux where /usr/bin/awk is already the caller's awk, these live probes confirm non-leakage
  # while macOS under test-harnesses.sh (and the controlled stub_dir checks below) discriminate the pin.
  run_out=$TMP/pin-run.out
  rm -f "$run_out"
  FAKE_PIN_OUT=$run_out run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ ! -e "$TMP/pinbin/dune" ] && [ ! -e "$TMP/pinbin/opam" ] &&
     check_pin_probe "$run_out" 1 0; then
    report 0 'run() preserves active bash (including env shebang) and awk across SYS_PATH without leaking opam'
  else
    report 1 'run() preserves active bash (including env shebang) and awk across SYS_PATH without leaking opam' \
      "exit $rc; want bash=$want_bash awk=$want_awk $outer_2244 opam=$sys_opam; got=$(cat "$run_out" 2>/dev/null); out=$(cat "$TMP/out")"
  fi

  owed_out=$TMP/pin-owed.out
  rm -f "$owed_out"
  FAKE_PIN_OUT=$owed_out owed "$OWED_EMPTY" url
  if [ "$orc" = 0 ] && check_pin_probe "$owed_out" 0 1; then
    report 0 'owed() preserves active bash and awk across SYS_PATH without leaking dune or opam'
  else
    report 1 'owed() preserves active bash and awk across SYS_PATH without leaking dune or opam' \
      "exit $orc; want bash=$want_bash awk=$want_awk $outer_2244 dune=$sys_dune opam=$sys_opam; got=$(cat "$owed_out" 2>/dev/null); out=$(cat "$TMP/owed-out")"
  fi

  stub_dir=$TMP/stub-pin
  mkdir -p "$stub_dir"
  cat >"$stub_dir/awk" <<'STUB'
#!/bin/sh
printf 'STUB_AWK:%s\n' "$*"
STUB
  cat >"$stub_dir/bash" <<STUB
#!/bin/sh
STUB_BASH_SEEN=yes exec "$BASH" "\$@"
STUB
  printf '#!/bin/sh\necho LEAKED_DECOY\n' >"$stub_dir/ocannl_pin_decoy_tool"
  printf '#!/bin/sh\necho LEAKED_DUNE\n' >"$stub_dir/dune"
  printf '#!/bin/sh\necho LEAKED_OPAM\n' >"$stub_dir/opam"
  chmod +x "$stub_dir/awk" "$stub_dir/bash" "$stub_dir/ocannl_pin_decoy_tool" "$stub_dir/dune" "$stub_dir/opam"
  cat >"$TMP/stub-shebang.sh" <<'PROBE'
#!/usr/bin/env bash
printf 'shebang=%s awk=%s decoy=%s dune=%s opam=%s\n' \
  "${STUB_BASH_SEEN:-no}" "$(awk arg1)" \
  "$(type -P ocannl_pin_decoy_tool 2>/dev/null || :)" \
  "$(type -P dune 2>/dev/null || :)" "$(type -P opam 2>/dev/null || :)"
PROBE
  chmod +x "$TMP/stub-shebang.sh"

  for mode in abs rel; do
    got=$(
      if [ "$mode" = abs ]; then
        PATH="$stub_dir:/usr/bin:/bin"
      else
        cd "$TMP" || exit 2
        PATH="stub-pin:/usr/bin:/bin"
      fi
      build_sys_path "$TMP/test-pin-$mode" || exit 2
      cd "$R" || exit 2
      STUB_BASH_SEEN= PATH="$TMP/gitbin:$SYS_PATH" "$TMP/stub-shebang.sh"
    )
    if [ "$got" = "shebang=yes awk=STUB_AWK:arg1 decoy= dune=$sys_dune opam=$sys_opam" ]; then
      report 0 "build_sys_path preserves $mode-PATH bash and awk shims after cd without leaking sibling tools"
    else
      report 1 "build_sys_path preserves $mode-PATH bash and awk shims after cd without leaking sibling tools" \
        "want decoy= dune=$sys_dune opam=$sys_opam; got=$got"
    fi
  done

  # Relative PATH entry crossing a symlink before `..`: physical traversal (`cd -P`) must match
  # kernel execve rather than collapsing `sym/..` logically onto the decoy parent directory.
  phys_real=$TMP/pin-phys-real
  phys_decoy=$TMP/pin-phys-decoy
  mkdir -p "$phys_real/sub" "$phys_decoy"
  printf '#!/bin/sh\necho PHYS_AWK\n' >"$phys_real/awk"
  printf '#!/bin/sh\necho LOGICAL_DECOY_AWK\n' >"$phys_decoy/awk"
  chmod +x "$phys_real/awk" "$phys_decoy/awk"
  ln -s -- "$phys_real/sub" "$phys_decoy/sym"
  if [ -L "$phys_decoy/sym" ]; then
    got=$(
      cd "$phys_decoy" || exit 2
      PATH="sym/..:/usr/bin:/bin"
      build_sys_path "$TMP/test-pin-phys" || exit 2
      cd "$R" || exit 2
      PATH="$TMP/gitbin:$SYS_PATH" awk
    )
    if [ "$got" = PHYS_AWK ]; then
      report 0 'build_sys_path resolves relative symlink/.. PATH entries physically'
    else
      report 1 'build_sys_path resolves relative symlink/.. PATH entries physically' "got=$got"
    fi
  fi

  override_dir=$TMP/pin-override
  mkdir -p "$override_dir"
  printf '#!/bin/sh\necho PREFIX_AWK\n' >"$override_dir/awk"
  chmod +x "$override_dir/awk"
  got=$(
    PATH="$stub_dir:/usr/bin:/bin"
    build_sys_path "$TMP/test-pin-override" || exit 2
    PATH="$override_dir:$TMP/gitbin:$SYS_PATH" awk
  )
  if [ "$got" = PREFIX_AWK ]; then
    report 0 'a leg-specific prefix ahead of SYS_PATH still overrides pinned shims'
  else
    report 1 'a leg-specific prefix ahead of SYS_PATH still overrides pinned shims' "got=$got"
  fi
}
harness_leg_run 'SYS_PATH resets preserve pinned bash and awk without leaking other tools #sys-path-pin' sys_path_pin_leg

sys_path_mawk_leg() {
  local host_mawk= cand mawk_pin got_pinned got_unpinned
  # Prefer the caller's active awk when it already exhibits mawk #2244 semantics (always true
  # under tools/test-harnesses.sh on macOS and on mawk-default Linux), else type -P mawk (keeping
  # standalone and harness-legs-alone runs working without parsing test-harnesses.sh).
  cand=$(type -P awk 2>/dev/null || :)
  if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ] &&
     [ "$(awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }')" = 'got=[; x]' ]; then
    host_mawk=$(resolve_exec_path "$cand") || { harness_abandon "could not normalize awk path: $cand"; return; }
  else
    cand=$(type -P mawk 2>/dev/null || :)
    if [ -n "$cand" ] && [ -f "$cand" ] && [ -x "$cand" ]; then
      host_mawk=$(resolve_exec_path "$cand") || { harness_abandon "could not normalize mawk path: $cand"; return; }
    fi
  fi
  if [ -z "$host_mawk" ]; then
    harness_not_applicable 'no real mawk available on PATH for the #2244 SYS_PATH control'
    return
  fi
  mawk_pin=$TMP/real-mawk-pin
  mkdir -p "$mawk_pin"
  ln -sf -- "$host_mawk" "$mawk_pin/awk"
  got_pinned=$(
    PATH="$mawk_pin:/usr/bin:/bin"
    build_sys_path "$TMP/test-pin-mawk" || exit 2
    PATH="$TMP/dunebin:$TMP/gitbin:$SYS_PATH" \
      awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }'
  )
  if [ "$got_pinned" = 'got=[; x]' ]; then
    report 0 'SYS_PATH preserves real mawk pin for #2244 in-inside-assignment (got=[; x])'
  else
    report 1 'SYS_PATH preserves real mawk pin for #2244 in-inside-assignment (got=[; x])' "got=$got_pinned"
  fi
  if [ "$(uname -s 2>/dev/null)" = Darwin ] && [ -x /usr/bin/awk ]; then
    got_unpinned=$(PATH="$TMP/dunebin:$TMP/gitbin:/usr/bin:/bin" \
      awk 'BEGIN { a["k"] = ("k" in a) ? a["k"] "; x" : "x"; print "got=[" a["k"] "]" }')
    if [ "$got_unpinned" = 'got=[x]' ]; then
      report 0 'unpinned /usr/bin:/bin on macOS falls back to BWK awk (got=[x])'
    else
      report 1 'unpinned /usr/bin:/bin on macOS falls back to BWK awk (got=[x])' "got=$got_unpinned"
    fi
  fi
}
harness_leg_run 'SYS_PATH resets keep real mawk semantics for #2244 in-inside-assignment #sys-path-pin' sys_path_mawk_leg

if harness_leg '--compile-only runs only the Dune setup and compile tier #compile-only'; then
  # On BOTH_SCRIPT a full run would run format, compile, tests and the whole harness tier;
  # --compile-only skips every other tier even when format/tests/harness would fail or owe.
  FAKE_FMT_RC=1 FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_SHELL_RC=1 \
    R=$BOTH_SCRIPT run "$TMP/dunebin" --compile-only
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && [ ! -s "$TMP/record.tiers" ] &&
     has 'HEAD merges ' &&
     has 'compile tier: dune build @check @test/operations/scans (--compile-only)' &&
     has 'compile tier passed' && has 'merge-build-check: passed' &&
     ! has 'format tier' && ! has 'tests tier' && ! has 'harness tier'; then
    report 0 '--compile-only runs only the compile tier and skips format, tests, owed-run and harness'
  else report 1 '--compile-only runs only the compile tier and skips format, tests, owed-run and harness' \
    "exit $rc; record=$(cat "$TMP/record"); tiers=$(cat "$TMP/record.tiers"); $(cat "$TMP/out")"; fi
  FAKE_DUNE_RC=3 R=$BOTH_SCRIPT run "$TMP/dunebin" --compile-only
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && [ ! -s "$TMP/record.tiers" ] &&
     has 'compile tier FAILED (dune exit 3)' && ! has 'merge-build-check: passed'; then
    report 0 '--compile-only exits 1 when the compile tier fails'
  else report 1 '--compile-only exits 1 when the compile tier fails' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # A compile-only recheck observes the same ship-pr failure-log contract.
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    FAKE_TIER_DELAY=1 R=$BOTH_SCRIPT run "$TMP/dunebin" --compile-only
  if [ "$rc" = 0 ] && grep -qE '^merge-build-check: compile tier passed in [1-9][0-9]*s$' "$TMP/out"; then
    report 0 '--compile-only reports the elapsed compile time on success'
  else report 1 '--compile-only reports the elapsed compile time on success' "exit $rc; $(cat "$TMP/out")"; fi
  fixture compile-only-base-alone
  first x.ml && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    FAKE_DUNE_RC=1 run "$TMP/dunebin" --compile-only
  merge_rc=$rc
  cp "$TMP/out" "$TMP/compile-only-merge"
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha"
  SHIP_PR_MERGE_BASE_SHA=base SHIP_PR_MERGE_HEAD_SHA=head \
    FAKE_DUNE_RC=1 FAKE_TIER_DELAY=1 run "$TMP/dunebin" --compile-only
  if [ "$merge_rc" = 1 ] && [ "$rc" = 1 ] && cmp -s "$TMP/compile-only-merge" "$TMP/out"; then
    report 0 '--compile-only emits identical logs on merge and base-alone for a shared compile failure'
  else
    report 1 '--compile-only emits identical logs on merge and base-alone for a shared compile failure' \
      "exits $merge_rc, $rc; $(diff "$TMP/compile-only-merge" "$TMP/out")"
  fi
  # Toolchain fallback through opam env works under --compile-only too.
  R=$APART run "$TMP/opambin" --compile-only
  if [ "$rc" = 0 ] && has 'dune is not on PATH; sourcing tools/opam-env.sh' &&
     [ "$(cat "$TMP/record")" = "$COMPILE" ] && [ ! -s "$TMP/record.tiers" ]; then
    report 0 '--compile-only resolves dune through tools/opam-env.sh when absent from PATH'
  else report 1 '--compile-only resolves dune through tools/opam-env.sh when absent from PATH' "exit $rc; $(cat "$TMP/out")"; fi
  R=$APART run "$TMP/emptyopam" --compile-only
  if [ "$rc" = 1 ] && has 'no dune on PATH even after opam env' && [ ! -s "$TMP/record" ]; then
    report 0 '--compile-only exits 1 without building when the switch lacks dune'
  else report 1 '--compile-only exits 1 without building when the switch lacks dune' "exit $rc; $(cat "$TMP/out")"; fi
  # A one-parent HEAD still runs the compile tier cleanly.
  fixture compile-only-one-parent
  run "$TMP/dunebin" --compile-only
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE" ] && ! has 'HEAD merges '; then
    report 0 '--compile-only runs the compile tier on a one-parent HEAD'
  else report 1 '--compile-only runs the compile tier on a one-parent HEAD' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'usage #usage'; then
  rc=0
  "$BASH" "$HERE/merge-build-check.sh" --bogus >"$TMP/out" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && has "unknown argument '--bogus'"; then
    report 0 'an unknown argument exits 2'
  else report 1 'an unknown argument exits 2' "exit $rc; $(cat "$TMP/out")"; fi
  for bad_combo in '--compile-only --plan' '--compile-only --plan-merge' '--compile-only --base HEAD' '--compile-only --plan --base HEAD' '--compile-only --all'; do
    rc=0
    # shellcheck disable=SC2086 # fixed flag words
    "$BASH" "$HERE/merge-build-check.sh" $bad_combo >"$TMP/out" 2>&1 || rc=$?
    if [ "$rc" = 2 ] && has '--compile-only cannot be combined with --plan, --plan-merge, --base or --all'; then
      report 0 "\`$bad_combo\` exits 2"
    else report 1 "\`$bad_combo\` exits 2" "exit $rc; $(cat "$TMP/out")"; fi
  done
fi

# The config is what runs this script, and its inputs are what trigger it. The compile tier's
# scans read the repository itself (notes, docs, goldens, YAML, Python, scripts, sources), so the
# config makes every tracked path an input rather than keeping a second copy of the scans' corpora
# (gh-ocannl-1382). Each tracked path must match one `input` glob as the merge build reads it: the
# whole path, `*` crossing `/` (a [[ == ]] pattern does the same). That covers the harness tier's
# trigger too, since every script the tier could select is a tracked path.
if harness_leg 'the merge-build config runs this script, and every tracked path is an input #config'; then
  config=$HERE/../.github/ship-pr-merge-build
  globs=() n=0 unmatched=
  while IFS= read -r line; do
    case $line in 'input '*) globs+=("${line#input }") ;; esac
  done <"$config"
  if [ ${#globs[@]} -gt 0 ]; then
    while IFS= read -r -d '' p; do
      n=$((n + 1)) hit=0
      for g in "${globs[@]}"; do
        # shellcheck disable=SC2053 # the glob is a pattern, as fnmatch reads it
        if [[ $p == $g ]]; then hit=1; break; fi
      done
      [ "$hit" = 1 ] || { unmatched=$p; break; }
    done < <(git -C "$HERE/.." ls-files -z)
  fi
  label="the config's command, plan and recheck run this script, it builds every merge, and its inputs match every tracked path"
  if ! grep -qxF "command tools/merge-build-check.sh" "$config"; then
    report 1 "$label" "no \`command tools/merge-build-check.sh\` line: $(cat "$config" 2>&1)"
  elif ! grep -qxF "plan tools/merge-build-check.sh --plan-merge" "$config"; then
    report 1 "$label" "no \`plan tools/merge-build-check.sh --plan-merge\` line: $(cat "$config" 2>&1)"
  elif ! grep -qxF "recheck tools/merge-build-check.sh --compile-only" "$config"; then
    report 1 "$label" "no \`recheck tools/merge-build-check.sh --compile-only\` line: $(cat "$config" 2>&1)"
  elif ! grep -qxF "trigger always" "$config"; then
    # The build is the merge gate only while it runs on every merge: under the default trigger a
    # merge whose base did not move, or whose sides share no path, lands unbuilt.
    report 1 "$label" "no \`trigger always\` line, so a merge whose two sides share no path would land unbuilt"
  elif [ ${#globs[@]} = 0 ] || [ "$n" = 0 ]; then
    report 1 "$label" "read ${#globs[@]} input glob(s) and $n tracked path(s): nothing was checked"
  elif [ -n "$unmatched" ]; then
    report 1 "$label" "$unmatched is tracked, but no input glob (${globs[*]}) matches it: a clash in it would skip the build"
  else report 0 "$label ($n paths)"; fi
fi

if harness_leg 'the real compile-tier fixture rule has stable output and retains failures #compile-output'; then
  stable_label='the actual compile-tier rule emits identical timing-free logs twice'
  failure_label='the actual compile-tier rule retains the failing traceback, summary and status'
  if ! harness_real_dune; then
    skip "$stable_label" "$harness_dune_absent"
    skip "$failure_label" "$harness_dune_absent"
  else
    harness_require python3
    real=$TMP/real-compile
    mkdir -p "$real/benchmarks"
    # Copy Python inputs, preserving their relative names, and the actual rule. No shared _build.
    python3 - "$HERE/../benchmarks" "$real/benchmarks" <<'PYTHON'
from pathlib import Path
import shutil
import sys
source, target = map(Path, sys.argv[1:])
for path in [source / "dune", source / "ocannl_config", *source.rglob("*.py")]:
    if "_build" in path.parts or "__pycache__" in path.parts:
        continue
    dest = target / path.relative_to(source)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, dest)
PYTHON
    printf '(lang dune 3.20)\n' >"$real/dune-project"
    rc_a=0 rc_b=0 rc_bad=0
    harness_dune_colour "$harness_dune" build --root "$real" --force @benchmarks/runtest-capped_fixture_scan >"$TMP/real-a" 2>&1 || rc_a=$?
    harness_dune_colour "$harness_dune" build --root "$real" --force @benchmarks/runtest-capped_fixture_scan >"$TMP/real-b" 2>&1 || rc_b=$?
    harness_plain "$TMP/real-a" >"$TMP/real-a-plain"
    if [ "$rc_a" = 0 ] && [ "$rc_b" = 0 ] && cmp -s "$TMP/real-a" "$TMP/real-b" &&
       ! grep -Eq '^Ran [0-9]+ tests? in [0-9]+\.[0-9]{3}s$' "$TMP/real-a-plain" &&
       grep -qx 'OK' "$TMP/real-a-plain"; then
      report 0 "$stable_label"
    else
      report 1 "$stable_label" "exits $rc_a/$rc_b: $(cat "$TMP/real-a" "$TMP/real-b")"
    fi
    cat >>"$real/benchmarks/test/test_cell_group.py" <<'PYTHON'

def planted_compile_failure(self):
    raise AssertionError("planted compile-tier failure")

CellGroupTest.test_capped_evidence_scan_controls = planted_compile_failure
PYTHON
    harness_dune_colour "$harness_dune" build --root "$real" --force @benchmarks/runtest-capped_fixture_scan >"$TMP/real-bad" 2>&1 || rc_bad=$?
    harness_plain "$TMP/real-bad" >"$TMP/real-bad-plain"
    if [ "$rc_bad" = 1 ] && grep -q 'Traceback (most recent call last):' "$TMP/real-bad-plain" &&
       grep -q 'AssertionError: planted compile-tier failure' "$TMP/real-bad-plain" &&
       grep -qx 'FAILED (failures=1)' "$TMP/real-bad-plain" &&
       ! grep -Eq '^Ran [0-9]+ tests? in [0-9]+\.[0-9]{3}s$' "$TMP/real-bad-plain"; then
      report 0 "$failure_label"
    else
      report 1 "$failure_label" "exit $rc_bad: $(cat "$TMP/real-bad")"
    fi
  fi
fi

if harness_leg '--compile-only models #2170 x #2173 through real Dune #compile-only-real'; then
  compat_label='--compile-only passes a compatible moved base through real Dune without other tiers'
  incompat_label='--compile-only refuses an incompatible union modelled on #2170 x #2173 while the base alone passes'
  if ! harness_real_dune; then
    skip "$compat_label" "$harness_dune_absent"
    skip "$incompat_label" "$harness_dune_absent"
  else
    real_co=$TMP/real-compile-only
    # Construct the fixture's subdirectory paths without spelling tracked repository paths like
    # lib/dune or test/operations/dune, so affected's harness reader does not attach unrelated
    # edits of those files to this harness.
    gate_dir=$real_co/gate_lib
    ops_dir=$real_co/test/operations
    mkdir -p "$gate_dir" "$ops_dir" "$real_co/tools"
    cp "$HERE/merge-build-check.sh" "$HERE/opam-env.sh" "$real_co/tools/"
    # Stub the three skipped tiers to fail loudly if --compile-only ever calls them.
    for stub in fmt-check.sh test-run.sh test-harnesses.sh; do
      printf '#!/bin/sh\necho "unexpected %s call" >&2\nexit 99\n' "$stub" >"$real_co/tools/$stub"
      chmod +x "$real_co/tools/$stub"
    done
    printf '(lang dune 3.20)\n' >"$real_co/dune-project"
    printf '(library (name gate))\n' >"$gate_dir/dune"
    printf 'val seed_gate_setting : unit -> [ `Auto | `Threshold of int ]\n' >"$gate_dir/gate.mli"
    printf 'let seed_gate_setting () = `Auto\n' >"$gate_dir/gate.ml"
    printf '(alias (name scans))\n' >"$ops_dir/dune"
    git -C "$real_co" init -q && git -C "$real_co" checkout -q -b base &&
      git -C "$real_co" add -A && git -C "$real_co" commit -q -m 'initial gate' &&
      git -C "$real_co" branch compat-base && git -C "$real_co" branch incompat-base &&
      git -C "$real_co" branch pr-head
    # pr-head (modelled on #2170): adds a caller matching `Auto against Gate.seed_gate_setting.
    git -C "$real_co" checkout -q pr-head
    printf '(executable (name seed_gate_caller) (libraries gate))\n(alias (name scans))\n' \
      >"$ops_dir/dune"
    printf 'let () = match Gate.seed_gate_setting () with `Auto -> () | `Threshold _ -> ()\n' \
      >"$ops_dir/seed_gate_caller.ml"
    git -C "$real_co" add -A && git -C "$real_co" commit -q -m 'pr-head matches `Auto'
    # compat-base: moves base in an unrelated module (no conflict with pr-head).
    git -C "$real_co" checkout -q compat-base
    printf 'let extra = 1\n' >"$gate_dir/extra.ml"
    git -C "$real_co" add -A && git -C "$real_co" commit -q -m 'compatible base move'
    git -C "$real_co" merge -q --no-ff --no-edit pr-head >/dev/null
    dune_bin=$(dirname -- "$harness_dune")
    real_co_build=$TMP/real-co-build
    real_co_run() {
      rm -rf "$real_co_build"
      rc=0
      (cd "$real_co" && env -u DUNE_ROOT DUNE_BUILD_DIR="$real_co_build" PATH="$dune_bin:$PATH" \
        "$BASH" tools/merge-build-check.sh --compile-only) >"$1" 2>&1 || rc=$?
    }
    real_co_run "$TMP/real-co-compat"
    rc_compat=$rc
    if [ "$rc_compat" = 0 ] &&
       grep -qF 'merge-build-check: compile tier passed' "$TMP/real-co-compat" &&
       grep -qF 'merge-build-check: passed' "$TMP/real-co-compat" &&
       ! grep -qF 'unexpected ' "$TMP/real-co-compat"; then
      report 0 "$compat_label"
    else
      report 1 "$compat_label" "exit $rc_compat: $(cat "$TMP/real-co-compat")"
    fi
    # incompat-base (modelled on #2173): changes `Auto to `Auto of [`Delivery | `Legacy].
    # Disjoint files from pr-head, so git merges cleanly while @check rejects the union.
    git -C "$real_co" checkout -q incompat-base
    printf 'val seed_gate_setting : unit -> [ `Auto of [ `Delivery | `Legacy ] | `Threshold of int ]\n' \
      >"$gate_dir/gate.mli"
    printf 'let seed_gate_setting () = `Auto `Delivery\n' >"$gate_dir/gate.ml"
    git -C "$real_co" add -A && git -C "$real_co" commit -q -m 'incompatible base move'
    git -C "$real_co" merge -q --no-ff --no-edit pr-head >/dev/null
    real_co_run "$TMP/real-co-union"
    rc_union=$rc
    # Base alone (base tip's tree on the merge's two parents, as ship-pr builds it): must pass.
    base_alone_sha=$(git -C "$real_co" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'base alone') &&
      git -C "$real_co" checkout -q --detach "$base_alone_sha"
    real_co_run "$TMP/real-co-base"
    rc_base=$rc
    if [ "$rc_union" = 1 ] && [ "$rc_base" = 0 ] &&
       grep -qF 'merge-build-check: compile tier FAILED (dune exit 1)' "$TMP/real-co-union" &&
       grep -qF 'seed_gate_caller.ml' "$TMP/real-co-union" &&
       grep -qF 'merge-build-check: passed' "$TMP/real-co-base" &&
       ! grep -qF 'unexpected ' "$TMP/real-co-union" &&
       ! grep -qF 'unexpected ' "$TMP/real-co-base"; then
      report 0 "$incompat_label"
    else
      report 1 "$incompat_label" "union=$rc_union base=$rc_base; union: $(cat "$TMP/real-co-union"); base: $(cat "$TMP/real-co-base")"
    fi
  fi
fi

finish

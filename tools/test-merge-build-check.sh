#!/usr/bin/env bash
# Hermetic fixtures for tools/merge-build-check.sh: the order of its tiers (format, compile, tests,
# harness) and where each stops the run; what the tests tier builds (the list prepush prints for
# the head side's revisions, never printed; a cross-cutting change's first tier, or nothing, with
# the rest OWED; an incomplete list's members; no list read fails the tier); what the harness tier
# reads (the HEAD side's change, a script or not, a path the base side renamed by its other name;
# a one-parent HEAD, --all and a quoted path run the whole tier), what it narrows to (the
# change's reach, of any group; none when it reaches no member; the whole tier on a failed ask,
# an unrecognized line or an unmapped script), the toolchain fallback through opam env, the exit
# statuses, stable tier lines across different run durations, that the build of the base alone
# narrows as the merge does, that this repository's mapping gives shared support the whole tier
# and a manifest edit less, and that
# .github/ship-pr-merge-build runs the script with every tracked path an input. Fake dune, opam,
# formatter gate, pre-push list and harness tier. The #owed legs read the owed-run record
# (tools/owed-run.sh, gh-ocannl-2218) from local bare repositories standing for the PR repository,
# never the real one: a missing, unreadable, foreign, stale, failed, dirty, non-cc, filtered,
# quoted or malformed record each refuses a cross-cutting, over-bound, incomplete or train-
# reaching change before anything is built; a full record, a union of records and a waiver
# admit; the base alone never prints the merge's refusal line; and `record` certifies only a
# finished, clean, cc run of an allowlisted command, `prune` only merged heads.
# Structured train-name rows require a count matching distinct names and an alias the
# tree has; legacy count-only output still asks affected and must agree exactly.
# The branch plan uses the chosen base's merged tree and the real record check,
# distinguishing conflict, narrow reach, and wide reach with or without a record.
# The #compile-output leg alone uses real
# dune and Python in a minimal isolated project: the repository's actual capped-fixture rule
# runs twice, then with a planted failing case. It compares bytes and rejects the exact unittest
# timing line; tracebacks and FAILED must survive. It does not scan Dune or Python source text.
# affected-reads: benchmarks/dune benchmarks/*.py benchmarks/test/*.py benchmarks/runners/*.py benchmarks/runners/*/*.py benchmarks/ocannl_config --only '#compile-output'
# affected-reads: tools/owed-run.sh
# Usage: tools/test-merge-build-check.sh [--keep] [--only REGEX]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require git
harness_scratch test-merge-build-check
HERE=$(cd "$(dirname "$0")" && pwd)
# The fixtures' own git: no ambient configuration, identity or repository selection.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR GIT_CEILING_DIRECTORIES
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid \
  GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid
SYS_PATH=/usr/bin:/bin
GIT=$(command -v git)

mkdir -p "$TMP/dunebin" "$TMP/opambin" "$TMP/emptyopam"
# The tests tier's build is the one that names a width (-j); FAKE_TESTS_RC fails it alone.
cat >"$TMP/dunebin/dune" <<'FAKE'
#!/bin/sh
if [ "$1 $2" = 'show aliases' ]; then
  [ "$*" = 'show aliases --root . test/training' ] || exit 2
  printf '%s\n' train-a train-b
  exit 0
fi
printf '%s\n' "$*" >>"$FAKE_RECORD"
sleep "${FAKE_TIER_DELAY:-0}"
case " $* " in *' -j '*) exit "${FAKE_TESTS_RC:-${FAKE_DUNE_RC:-0}}" ;; esac
exit "${FAKE_DUNE_RC:-0}"
FAKE
# `opam env --shell=sh` puts the fake dune on PATH, as a switch's does.
cat >"$TMP/opambin/opam" <<FAKE
#!/bin/sh
[ "\$*" = 'env --shell=sh' ] || { echo "unexpected opam call: \$*" >&2; exit 64; }
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
  # By default the change reaches no per-test alias.
  cat >"$R/tools/fmt-check.sh" <<'FAKE'
#!/bin/sh
printf 'fmt scans=%s\n' "${OCANNL_TOOL_FMT_CHECK_SCANS-unset}" >>"$FAKE_RECORD.tiers"
exit "${FAKE_FMT_RC:-0}"
FAKE
  cat >"$R/tools/test-run.sh" <<'FAKE'
#!/bin/sh
printf '%s\n' "$*" >>"$FAKE_RECORD.tiers"
if [ -n "${FAKE_PLAN_BASE:-}" ]; then
  [ "$(git rev-parse 'HEAD^1')" = "$FAKE_PLAN_BASE" ] &&
    [ "$(git rev-parse 'HEAD^2')" = "$FAKE_PLAN_HEAD" ] &&
    [ "$(git rev-parse --show-toplevel)" != "$FAKE_PLAN_CALLER" ] || exit 7
fi
if [ "${FAKE_CROSS_ON_MOVED_BASE:-0}" = 1 ] && [ -f moved-marker ]; then
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_RC=2 FAKE_TIER_VERDICT=cross-cutting
fi
reason() {
  [ -z "${FAKE_REASON:-}" ] || printf 'affected: CROSS-CUTTING reason: %s\n' "$FAKE_REASON" >&2
}
count() {
  if [ -n "${FAKE_TRAIN_NAMES:-}" ]; then
    while IFS= read -r alias; do printf 'test-run: prepush: train: %s\n' "$alias" >&2; done <"$FAKE_TRAIN_NAMES"
  fi
  [ "${FAKE_TRAIN_N:-0}" = none ] ||
    printf 'test-run: prepush: 1 of the 7 alias(es) the change lists; left off: %s train- (the nightly CI run) and 0 slow-\n' "${FAKE_TRAIN_N:-0}" >&2
}
case " $* " in
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
    FAKE_PLAN_CALLER=$R FAKE_PLAN_BASE=$1 FAKE_PLAN_HEAD=$side run "$TMP/dunebin" --plan --base "$1"
  }
  planned() {
    if [ "$rc" = 0 ] && [ ! -s "$TMP/record" ] && grep -qx "prepush --since $fork..$side --list" "$TMP/record.tiers" &&
       ! grep -q '^fmt ' "$TMP/record.tiers" && [ "$(git -C "$R" rev-parse HEAD)" = "$side" ] &&
       [ "$(git -C "$R" status --porcelain)" = "$before" ] && has "$2" && has "$3"; then
      report 0 "$1"
    else report 1 "$1" "exit $rc; $(cat "$TMP/record"); $(tiers); $(cat "$TMP/out")"; fi
  }
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_REASON='x.ml (6 aliases)' plan_branch "$tip"
  planned 'a cross-cutting branch plan names the need and missing record' \
    'tests plan: CROSS-CUTTING (first tier selected)' 'owed-run needs: runtest (record does not admit'
  FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting OCANNL_TOOL_OWED_RUN_URL=$OWED_FULL plan_branch "$tip"
  planned 'a cross-cutting branch plan reads a record of the head' \
    'tests plan: CROSS-CUTTING (first tier selected)' 'owed-run needs: runtest (record admits)'
  FAKE_CROSS_ON_MOVED_BASE=1 plan_branch "$fork"
  planned 'a narrow branch plan owes no record' 'tests plan: narrow' 'owed-run needs: none'
  FAKE_CROSS_ON_MOVED_BASE=1 plan_branch "$tip"
  planned 'the same head is cross-cutting with no first tier against the moved base' \
    'tests plan: CROSS-CUTTING (no first tier selected)' 'owed-run needs: runtest (record does not admit'
  if has 'fixture: merge gate from the moved base'; then
    report 0 'the candidate plan uses the merge gate version the base brought'
  else report 1 'the candidate plan uses the merge gate version the base brought' "$(cat "$TMP/out")"; fi
  # The side needs a conflicting edit to the same line, relative to the fork.
  { echo conflict; tail -n +2 "$R/r.txt"; } >"$R/r.txt.new" && mv "$R/r.txt.new" "$R/r.txt" && commit_on side
  side=$(git -C "$R" rev-parse HEAD)
  plan_branch "$tip"
  if [ "$rc" = 1 ] && has 'plan: merge conflict' && has r.txt &&
     [ ! -s "$TMP/record" ] && [ ! -s "$TMP/record.tiers" ]; then
    report 0 'a conflicting branch plan reports the conflict before any tier is asked'
  else report 1 'a conflicting branch plan reports the conflict before any tier is asked' "exit $rc; $(cat "$TMP/out")"; fi
fi
if harness_leg 'the format tier runs first, its scans off, and a red one ends the run #tiers'; then
  R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(tiers | sed -n 1p)" = 'fmt scans=0' ] && has 'merge-build-check: format tier passed' &&
     [ "$(grep -n 'tier passed' "$TMP/out" | sed -n 1p)" = "$(grep -n 'format tier passed' "$TMP/out")" ]; then
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
     has "owed-run: admitted run of 'build @runtest @train -j 4' for ${APART_HEAD:0:8} (fixture:/runs/1)"; then
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
  OCANNL_TOOL_OWED_RUN_URL=$TMP/nowhere.git FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  refused 'a record repository that cannot be read refuses (unreadable)' \
    "REFUSED -- refs/owed-runs/$APART_HEAD could not be read within 60s: unreadable (records on $TMP/nowhere.git)"
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
    "owed-run: v2 run head=$APART_HEAD exit=0|holds a line no v1 record reads|an unknown schema" \
    "owed-run: v1 run head=$APART_HEAD exit=0 cmd=runtest|holds a line no v1 record reads|a malformed line"; do
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
build --root . -j 4 @a/runtest-first" ] && has "owed-run: admitted run of 'runtest -j 4' for ${APART_HEAD:0:8} (fixture:/runs/1)"; then
    report 0 'a full-suite record admits a cross-cutting change, whose first tier still builds'
  else report 1 'a full-suite record admits a cross-cutting change, whose first tier still builds' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  waivers=$TMP/owed-waiver.git
  git init -q --bare "$waivers"
  owed "$waivers" waive --head "$APART_HEAD" --reason 'run 20261116T0000Z on rog, the maintainer'
  OCANNL_TOOL_OWED_RUN_URL=$waivers FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$orc" = 0 ] && [ "$rc" = 0 ] && has "owed-run: admitted waiver for ${APART_HEAD:0:8} by fixture: run 20261116T0000Z on rog, the maintainer"; then
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
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  cp "$TMP/out" "$TMP/owed-same-out"
  merge_rc=$rc
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha"
  OCANNL_TOOL_OWED_RUN_URL=$OWED_EMPTY FAKE_PREPUSH_RC=2 FAKE_VERDICT=cross-cutting FAKE_TIER_LIST=$TMP/owed-tier run "$TMP/dunebin"
  if [ "$(git -C "$R" rev-parse 'HEAD^1^{tree}')" = "$(git -C "$R" rev-parse 'base^{tree}')" ] &&
     [ "$merge_rc" = 1 ] && [ "$rc" = 1 ] && cmp -s "$TMP/out" "$TMP/owed-same-out"; then
    report 0 'a merge whose tree is the base'"'"'s logs as the base alone does (it changes nothing)'
  else report 1 'a merge whose tree is the base'"'"'s logs as the base alone does (it changes nothing)' "exits $merge_rc/$rc; $(diff "$TMP/owed-same-out" "$TMP/out")"; fi
fi

# tools/owed-run.sh's own verbs: what `record` certifies from a run directory, and `prune`.
if harness_leg 'record certifies only a finished, clean, cc run of an allowlisted command; prune #owed-run'; then
  R=$APART
  rundir() { # NAME EXIT DIRTY BACKEND CMD: a synthetic tools/test-run.sh run record
    local d=$TMP/runs/$1
    mkdir -p "$d"
    printf '%s\n' "$APART_HEAD" >"$d/head"
    [ "$2" = none ] || printf '%s\n' "$2" >"$d/exit"
    printf '%s' "$3" >"$d/dirty"
    [ "$4" = none ] || printf '%s\n' "$4" >"$d/expect-backend"
    printf '%s \n' "$5" >"$d/cmd"
    printf '%s' "$d"
  }
  owed "$OWED_EMPTY" record "$(rundir good 0 '' cc 'runtest -j 4')" --print
  line=$(cat "$TMP/owed-out")
  case $line in
    "owed-run: v1 run head=$APART_HEAD exit=0 clean=yes backend=cc source="*":$TMP/runs/good cmd=runtest -j 4")
      report 0 'record prints the line of a finished, clean, cc full-suite run' ;;
    *) report 1 'record prints the line of a finished, clean, cc full-suite run' "exit $orc: $line" ;;
  esac
  rt=$TMP/owed-roundtrip.git
  git init -q --bare "$rt"
  owed "$rt" add "$line" && add_rc=$orc
  owed "$rt" check "$APART_HEAD" runtest
  if [ "$add_rc" = 0 ] && [ "$orc" = 0 ] && grep -qF "owed-run: admitted run of 'runtest -j 4'" "$TMP/owed-out"; then
    report 0 'a printed line, added, is what check admits'
  else report 1 'a printed line, added, is what check admits' "add $add_rc, check $orc: $(cat "$TMP/owed-out")"; fi
  # tools/test-run.sh records the argv it ran, its injected width and --no-buffer included.
  owed "$rt" record "$(rundir injected 0 '' cc 'build --no-buffer @test/training/train-a -j 1 -j 4')"
  owed "$rt" check "$APART_HEAD" runtest @test/training/train-a
  if [ "$orc" = 0 ] && [ "$(grep -c 'owed-run: admitted run' "$TMP/owed-out")" = 2 ]; then
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
fi

# Use the same tree twice, but force each tier across a different whole-second duration.
# Compare complete logs without normalizing digits, including the shared shell-red scenario.
if harness_leg 'whole logs are identical across run durations #run'; then
  for shell_rc in 0 1; do
    case $shell_rc in
      0) shell_line='merge-build-check: harness tier passed' ;;
      1) shell_line='merge-build-check: harness tier FAILED (tools/test-harnesses.sh exit 1)' ;;
    esac
    FAKE_SHELL_RC=$shell_rc FAKE_TIER_DELAY=0 R=$BOTH_SCRIPT run "$TMP/dunebin"
    first_rc=$rc
    cp "$TMP/out" "$TMP/out-first"
    FAKE_SHELL_RC=$shell_rc FAKE_TIER_DELAY=2 R=$BOTH_SCRIPT run "$TMP/dunebin"
    if [ "$first_rc" = "$shell_rc" ] && [ "$rc" = "$shell_rc" ] &&
       has 'merge-build-check: compile tier passed' && has "$shell_line" &&
       cmp -s "$TMP/out-first" "$TMP/out"; then
      report 0 "the same tree prints identical whole logs with shell exit $shell_rc despite different durations"
    else
      report 1 "the same tree prints identical whole logs with shell exit $shell_rc despite different durations" \
        "exits $first_rc, $rc; first: $(cat "$TMP/out-first"); second: $(cat "$TMP/out")"
    fi
  done
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
  reach_of tools/sub/a.sh
  R=$BOTH_SCRIPT plan
  if [ "$rc" = 0 ] && has 'harness tier: a script changed on the head side -- tools/sub/a.sh matches *.sh' &&
     has 'harness tier: SELECTED -- scoped to what the 1 path(s) the head side changed reach:' && has 'merge-build-check:   tools/test-one.sh'; then
    report 0 '--plan names what the change reaches'
  else report 1 '--plan names what the change reaches' "exit $rc; $(cat "$TMP/out")"; fi
  R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
shell --jobs 4 --affected --files tools/sub/a.sh" ] && has 'merge-build-check: passed'; then
    report 0 'a reached change runs its scoped tier, the whole tier not at all'
  else report 1 'a reached change runs its scoped tier, the whole tier not at all' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(grep -c '^shell ' "$TMP/record")" = 1 ] && has 'harness tier FAILED (tools/test-harnesses.sh exit 1)'; then
    report 0 'a failing scoped tier exits 1, with no whole tier to fall back on'
  else report 1 'a failing scoped tier exits 1, with no whole tier to fall back on' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Every path the head side changed is asked about, a non-script too; a rename by both its names.
  fixture ask-all
  first tools/sub/a.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last x.ml && commit_on side && merge_side
  reach_of tools/sub/a.sh
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files tools/sub/a.sh x.ml' ]; then
    report 0 'every path the head side changed is asked about, a non-script too'
  else report 1 'every path the head side changed is asked about, a non-script too' "exit $rc; $(cat "$TMP/record")"; fi
  fixture ask-rename
  git -C "$R" mv r.txt r.sh && commit_on base
  git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
  reach_of r.sh
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files r.txt r.sh' ]; then
    report 0 'a rename is asked about by both its names'
  else report 1 'a rename is asked about by both its names' "exit $rc; $(cat "$TMP/record")"; fi
  # The change is the head side's alone: nothing has to have changed on the base side (the merge
  # `trigger always` builds when the base has not moved), and a non-script that reaches a member
  # runs it with no script changed at all.
  fixture head-only
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  reach_of tools/sub/a.sh
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --jobs 4 --affected --files tools/sub/a.sh' ]; then
    report 0 'a script the head side alone changed runs what it reaches'
  else report 1 'a script the head side alone changed runs what it reaches' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  fixture non-script
  first tools/sub/a.sh && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  reach_of x.ml
  run "$TMP/dunebin"
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
  fixture two-scripts
  first tools/sub/a.sh && first b.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last b.sh && last x.ml && commit_on side && merge_side
  TWO_SCRIPTS=$R
  # One script reached, the other not: the reached one must not hide it.
  reach_of tools/sub/a.sh
  R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- no harness reaches b.sh'; then
    report 0 'a script no member reaches runs the whole tier, though another is reached'
  else report 1 'a script no member reaches runs the whole tier, though another is reached' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  reach_of tools/sub/a.sh b.sh
  FAKE_ASK_RC=2 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- what b.sh reaches could not be asked (test-harnesses.sh exit 2)'; then
    report 0 'a failed ask runs the whole tier'
  else report 1 'a failed ask runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Each script alone answers, the joint ask does not.
  FAKE_ASK_FAILS_ON=x.ml R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4' ] &&
     has 'harness tier: SELECTED -- whole -- what the paths the head side changed reach could not be asked'; then
    report 0 'a failed ask for the paths together runs the whole tier'
  else report 1 'a failed ask for the paths together runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --jobs 4 --affected --files b.sh tools/sub/a.sh x.ml
shell --jobs 4' ] && has 'the scoped run was refused (exit 2), no member run; running the whole tier'; then
    report 0 'a scoped run refused before any member started runs the whole tier'
  else report 1 'a scoped run refused before any member started runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 FAKE_SHELL_RC=1 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 1 ] && has 'harness tier FAILED (tools/test-harnesses.sh exit 1)'; then
    report 0 'the whole tier a refusal falls back on still decides'
  else report 1 'the whole tier a refusal falls back on still decides' "exit $rc; $(cat "$TMP/out")"; fi
fi

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

if harness_leg 'usage #usage'; then
  rc=0
  "$BASH" "$HERE/merge-build-check.sh" --bogus >"$TMP/out" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && has "unknown argument '--bogus'"; then
    report 0 'an unknown argument exits 2'
  else report 1 'an unknown argument exits 2' "exit $rc; $(cat "$TMP/out")"; fi
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
  label="the config's command is this script, it builds every merge, and its inputs match every tracked path"
  if ! grep -qxF "command tools/merge-build-check.sh" "$config"; then
    report 1 "$label" "no \`command tools/merge-build-check.sh\` line: $(cat "$config" 2>&1)"
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

finish

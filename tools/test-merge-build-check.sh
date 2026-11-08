#!/usr/bin/env bash
# Hermetic fixtures for tools/merge-build-check.sh: which tiers a merge commit selects (a script
# both sides changed, by either name of a rename; a one-parent HEAD, --all and a quoted path run
# every tier), what the shell tier narrows to (the overlap's reach, the whole tier on a failed ask,
# an unrecognized line or an unmapped script), the toolchain fallback through opam env, the exit
# statuses, stable tier lines across different run durations, that the build of the base alone
# narrows as the merge does, that this repository's mapping gives shared support the whole tier
# and a manifest edit less, and that
# .github/ship-pr-merge-build runs the script with every tracked path an input. Fake dune, opam
# and harness tier: no build runs.
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
cat >"$TMP/dunebin/dune" <<'FAKE'
#!/bin/sh
printf '%s\n' "$*" >>"$FAKE_RECORD"
sleep "${FAKE_TIER_DELAY:-0}"
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
  cp "$HERE/merge-build-check.sh" "$HERE/opam-env.sh" "$R/tools/"
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
  (cd "$R" && PATH=$prefix:$TMP/gitbin:$SYS_PATH FAKE_RECORD=$TMP/record "$BASH" tools/merge-build-check.sh "$@") \
    >"$TMP/out" 2>&1 || rc=$?
}
has() { grep -qF -- "$1" "$TMP/out"; }

# Both sides edit one script, under a nested directory: `*.sh` crosses `/`.
fixture both-script
first tools/sub/a.sh && commit_on base
git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
BOTH_SCRIPT=$R
if harness_leg 'a script both sides changed selects the shell tier #select'; then
  plan
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- tools/sub/a.sh, changed on both sides, matches *.sh' &&
     has 'compile tier: dune build @check @test/operations/scans (always)'; then
    report 0 'a script both sides changed selects the shell tier'
  else report 1 'a script both sides changed selects the shell tier' "exit $rc; $(cat "$TMP/out")"; fi
fi

# OCaml both sides changed; each side changed a different script.
fixture apart
first x.ml && first tools/sub/a.sh && commit_on base
git -C "$R" checkout -q side && last x.ml && last b.sh && commit_on side && merge_side
APART=$R
if harness_leg 'scripts only one side changed each leave the shell tier out #select'; then
  plan
  if [ "$rc" = 0 ] && has 'shell tier: skipped -- none of the 1 path(s) changed on both sides matches *.sh'; then
    report 0 'scripts only one side changed each leave the shell tier out'
  else report 1 'scripts only one side changed each leave the shell tier out' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'a rename counts by its other name, and only a script name selects #select'; then
  for target in r.sh r.md; do
    fixture "rename-$target"
    git -C "$R" mv r.txt "$target" && commit_on base
    git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
    plan
    case $target in
      r.sh) want='shell tier: SELECTED -- r.txt, changed on both sides, was renamed from or to r.sh, which matches *.sh' ;;
      r.md) want='shell tier: skipped -- none of the 1 path(s) changed on both sides matches *.sh' ;;
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
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- r.txt, changed on both sides, was renamed from or to r.sh'; then
    report 0 'a rename on the head counts too'
  else report 1 'a rename on the head counts too' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'what cannot be read runs every tier #select'; then
  fixture one-parent
  plan
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- HEAD is not a two-parent merge, so every tier runs'; then
    report 0 'a one-parent HEAD runs every tier'
  else report 1 'a one-parent HEAD runs every tier' "exit $rc; $(cat "$TMP/out")"; fi
  # A name git quotes is one this script does not decode, so it cannot rule the path out.
  fixture quoted
  first x.ml && printf 'x\n' >"$R/q\"uote.txt" && commit_on base
  git -C "$R" checkout -q side && last x.ml && commit_on side && merge_side
  plan
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- git quoted a path one side changed, so every tier runs'; then
    report 0 'a path git quotes runs every tier'
  else report 1 'a path git quotes runs every tier' "exit $rc; $(cat "$TMP/out")"; fi
  R=$APART plan --all
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- --all'; then
    report 0 '--all runs every tier on a merge that would skip one'
  else report 1 '--all runs every tier on a merge that would skip one' "exit $rc; $(cat "$TMP/out")"; fi
fi

if harness_leg 'a run builds the selected tiers in order and stops at the first failure #run'; then
  R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "build --root . @check @test/operations/scans
shell --shell --jobs 4" ] && has 'merge-build-check: passed'; then
    report 0 'a selected shell tier runs after the compile tier'
  else report 1 'a selected shell tier runs after the compile tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  R=$APART run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = 'build --root . @check @test/operations/scans' ]; then
    report 0 'a skipped shell tier does not run'
  else report 1 'a skipped shell tier does not run' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_DUNE_RC=3 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(cat "$TMP/record")" = 'build --root . @check @test/operations/scans' ] &&
     has 'compile tier FAILED (dune exit 3)'; then
    report 0 'a failing compile tier exits 1 before the shell tier'
  else report 1 'a failing compile tier exits 1 before the shell tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SHELL_RC=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && has 'shell tier FAILED (tools/test-harnesses.sh exit 1)' && ! has 'merge-build-check: passed'; then
    report 0 'a failing shell tier exits 1'
  else report 1 'a failing shell tier exits 1' "exit $rc; $(cat "$TMP/out")"; fi
fi

# Use the same tree twice, but force each tier across a different whole-second duration.
# Compare the actual lines without normalizing digits: shared-failure detection does the same.
if harness_leg 'tier lines are identical across run durations #run'; then
  FAKE_TIER_DELAY=0 R=$BOTH_SCRIPT run "$TMP/dunebin"
  first_rc=$rc
  grep ' tier ' "$TMP/out" >"$TMP/tiers-first"
  FAKE_TIER_DELAY=2 R=$BOTH_SCRIPT run "$TMP/dunebin"
  grep ' tier ' "$TMP/out" >"$TMP/tiers-second"
  if [ "$first_rc" = 0 ] && [ "$rc" = 0 ] &&
     has 'merge-build-check: compile tier passed' && has 'merge-build-check: shell tier passed' &&
     cmp -s "$TMP/tiers-first" "$TMP/tiers-second"; then
    report 0 'the same tree prints identical tier lines despite different durations'
  else
    report 1 'the same tree prints identical tier lines despite different durations' \
      "exits $first_rc, $rc; first: $(cat "$TMP/tiers-first"); second: $(cat "$TMP/tiers-second")"
  fi
fi

# The scope. FAKE_REACH stands for the mapping; the fixtures above reach nothing without it, which
# is why they run the whole tier.
reach_of() { # PATH...: each PATH reaches tools/test-one.sh
  local p
  for p do printf '%s\ttools/test-one.sh\n' "$p"; done >"$TMP/reach"
}
export FAKE_REACH=$TMP/reach
COMPILE='build --root . @check @test/operations/scans'
: >"$TMP/reach"
if harness_leg 'an overlap reaching one member runs that member, and no more #scope'; then
  reach_of tools/sub/a.sh
  R=$BOTH_SCRIPT plan
  if [ "$rc" = 0 ] && has 'shell tier: SELECTED -- tools/sub/a.sh, changed on both sides, matches *.sh' &&
     has 'shell tier: scoped to what the 1 shared path(s) reach:' && has 'merge-build-check:   tools/test-one.sh'; then
    report 0 '--plan names what the overlap reaches'
  else report 1 '--plan names what the overlap reaches' "exit $rc; $(cat "$TMP/out")"; fi
  R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(cat "$TMP/record")" = "$COMPILE
shell --shell --jobs 4 --affected --files tools/sub/a.sh" ] && has 'merge-build-check: passed'; then
    report 0 'a reached overlap runs its scoped tier, the whole tier not at all'
  else report 1 'a reached overlap runs its scoped tier, the whole tier not at all' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=1 R=$BOTH_SCRIPT run "$TMP/dunebin"
  if [ "$rc" = 1 ] && [ "$(grep -c '^shell ' "$TMP/record")" = 1 ] && has 'shell tier FAILED (tools/test-harnesses.sh exit 1)'; then
    report 0 'a failing scoped tier exits 1, with no whole tier to fall back on'
  else report 1 'a failing scoped tier exits 1, with no whole tier to fall back on' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Every shared path is asked about, a non-script too; a rename by both its names.
  fixture ask-all
  first tools/sub/a.sh && first x.ml && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && last x.ml && commit_on side && merge_side
  reach_of tools/sub/a.sh
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --shell --jobs 4 --affected --files tools/sub/a.sh x.ml' ]; then
    report 0 'every path changed on both sides is asked about, a non-script too'
  else report 1 'every path changed on both sides is asked about, a non-script too' "exit $rc; $(cat "$TMP/record")"; fi
  fixture ask-rename
  git -C "$R" mv r.txt r.sh && commit_on base
  git -C "$R" checkout -q side && last r.txt && commit_on side && merge_side
  reach_of r.sh
  run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2p "$TMP/record")" = 'shell --shell --jobs 4 --affected --files r.txt r.sh' ]; then
    report 0 'a rename is asked about by both its names'
  else report 1 'a rename is asked about by both its names' "exit $rc; $(cat "$TMP/record")"; fi
  # The build of the base alone: the base's tree on the merge's two parents, as ship-pr commits it.
  fixture base-alone
  first tools/sub/a.sh && commit_on base
  git -C "$R" checkout -q side && last tools/sub/a.sh && commit_on side && merge_side
  reach_of tools/sub/a.sh
  plan
  cp "$TMP/out" "$TMP/merge-plan"
  sha=$(git -C "$R" commit-tree 'HEAD^1^{tree}' -p 'HEAD^1' -p 'HEAD^2' -m 'the base alone') &&
    git -C "$R" checkout -q --detach "$sha" && plan
  if [ "$rc" = 0 ] && has 'shell tier: scoped to' && cmp -s "$TMP/out" "$TMP/merge-plan"; then
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
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --shell --jobs 4' ] &&
     has 'shell tier: whole -- no member of the tier reaches b.sh'; then
    report 0 'a script no member reaches runs the whole tier, though another is reached'
  else report 1 'a script no member reaches runs the whole tier, though another is reached' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  reach_of tools/sub/a.sh b.sh
  FAKE_ASK_RC=2 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --shell --jobs 4' ] &&
     has 'shell tier: whole -- what b.sh reaches could not be asked (test-harnesses.sh exit 2)'; then
    report 0 'a failed ask runs the whole tier'
  else report 1 'a failed ask runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  # Each script alone answers, the joint ask does not.
  FAKE_ASK_FAILS_ON=x.ml R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --shell --jobs 4' ] &&
     has 'shell tier: whole -- what the paths changed on both sides reach could not be asked'; then
    report 0 'a failed ask for the paths together runs the whole tier'
  else report 1 'a failed ask for the paths together runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 0 ] && [ "$(sed -n 2,\$p "$TMP/record")" = 'shell --shell --jobs 4 --affected --files b.sh tools/sub/a.sh x.ml
shell --shell --jobs 4' ] && has 'the scoped run was refused (exit 2), no member run; running the whole tier'; then
    report 0 'a scoped run refused before any member started runs the whole tier'
  else report 1 'a scoped run refused before any member started runs the whole tier' "exit $rc; $(cat "$TMP/record"); $(cat "$TMP/out")"; fi
  FAKE_SCOPED_RC=2 FAKE_SHELL_RC=1 R=$TWO_SCRIPTS run "$TMP/dunebin"
  if [ "$rc" = 1 ] && has 'shell tier FAILED (tools/test-harnesses.sh exit 1)'; then
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
    report 0 'an overlap reaching one leg narrows the tier to that leg'
  else report 1 'an overlap reaching one leg narrows the tier to that leg' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s\n' "$other_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has 'shell tier: whole -- no member of the tier reaches tools/sub/a.sh'; then
    report 0 'a script only a member outside the tier reaches runs the whole tier'
  else report 1 'a script only a member outside the tier reaches runs the whole tier' "exit $rc; $(cat "$TMP/out")"; fi
  printf '%s; touch evaluated\n' "$shell_member" >"$FAKE_COMMANDS"
  plan
  if [ "$rc" = 0 ] && has 'shell tier: whole -- what tools/sub/a.sh reaches could not be asked (test-harnesses.sh exit 2)' &&
     [ ! -e "$R/evaluated" ]; then
    report 0 'a line the allowlist does not recognize runs the whole tier'
  else report 1 'a line the allowlist does not recognize runs the whole tier' "exit $rc; $(cat "$TMP/out")"; fi
  # A shared path reading as an option: the same stand-in, which would answer, is never asked.
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
  if [ "$rc" = 0 ] && has 'shell tier: whole -- what -dash.sh reaches could not be asked (test-harnesses.sh exit 2)'; then
    report 0 'a shared path reading as an option runs the whole tier'
  else report 1 'a shared path reading as an option runs the whole tier' "exit $rc; $(cat "$TMP/out")"; fi
  unset FAKE_COMMANDS
fi

# This repository's own mapping, asked as the scope asks it: what a both-sides change to shared
# support reaches is the whole tier, every member loading it, and a manifest edit reaches the
# members that read the manifest, a small part of the tier (gh-ocannl-1994).
if harness_leg 'this repository: shared support reaches the tier, a manifest edit part of it #mapping'; then
  members_of() { "$BASH" "$HERE/test-harnesses.sh" "$@" 2>/dev/null | LC_ALL=C sort; }
  tier=$(members_of --shell --list)
  support=$(members_of --shell --affected --list --files scripts/harness-support.sh)
  manifest=$(members_of --shell --affected --list --files tools/test-harnesses.sh)
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
# whole path, `*` crossing `/` (a [[ == ]] pattern does the same). That covers the shell tier's
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
  label="the config's command is this script, and its inputs match every tracked path"
  if ! grep -qxF "command tools/merge-build-check.sh" "$config"; then
    report 1 "$label" "no \`command tools/merge-build-check.sh\` line: $(cat "$config" 2>&1)"
  elif [ ${#globs[@]} = 0 ] || [ "$n" = 0 ]; then
    report 1 "$label" "read ${#globs[@]} input glob(s) and $n tracked path(s): nothing was checked"
  elif [ -n "$unmatched" ]; then
    report 1 "$label" "$unmatched is tracked, but no input glob (${globs[*]}) matches it: a clash in it would skip the build"
  else report 0 "$label ($n paths)"; fi
fi
finish

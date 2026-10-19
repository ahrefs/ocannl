#!/usr/bin/env bash
# Hermetic fixtures for tools/merge-build-check.sh: which tiers a merge commit selects (a script
# both sides changed, by either name of a rename; a one-parent HEAD, --all and a quoted path run
# every tier), the toolchain fallback through opam env, the exit statuses, and that
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
  cat >"$R/tools/test-harnesses.sh" <<'FAKE'
#!/bin/sh
printf 'shell %s\n' "$*" >>"$FAKE_RECORD"
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
shell --shell" ] && has 'merge-build-check: passed'; then
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

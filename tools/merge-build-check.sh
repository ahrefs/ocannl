#!/usr/bin/env bash
# Check a merge commit: the command `.github/ship-pr-merge-build` gives ship-pr's merge build,
# which runs it at the root of the PR head merged with master's tip, before the merge call.
# Run it by hand the same way, at the root of a checkout whose HEAD is a merge you built.
# Usage: tools/merge-build-check.sh [--plan] [--all]
#
# HEAD is read as the merge to judge: first parent the base, second parent the PR head (the
# order `git merge` and ship-pr's `commit-tree -p <base tip> -p <head>` give). Two tiers:
#   compile  always: `dune build @check @test/operations/scans` (about 100 s from a fresh tree
#            on mac-studio: the compile and the repository scans, which read the repository
#            itself -- dune stanzas, sources, scripts, Markdown notes and docs, goldens -- so
#            the config makes every shared path an input).
#   shell    only when a path BOTH sides changed since their merge base matches `*.sh`: the
#            whole path, `*` crossing `/`, as the config's `input` globs read, a path a rename on
#            either side touched (git's -M detection) counting by its other name too -- the merge
#            build's own trigger, narrowed to shell scripts. What runs then is the part of
#            `tools/test-harnesses.sh --shell`, the toolchain-free harness tier CI runs, that the
#            paths changed on both sides reach, both names of a rename and non-scripts included:
#            `tools/test-harnesses.sh --shell --affected --files PATH...`, the mapping
#            `tools/test-run.sh affected --harnesses` owns, read from the merged tree. A member
#            outside the shell group stays out, so a scoped run is never more than the whole tier
#            (about 1100 s on mac-studio, measured in docs/agent-notes/build-and-test.md) and needs
#            nothing the whole tier does not.
# The selection is a fail-closed allowlist: the shell tier is skipped only when HEAD is a
# two-parent merge, its merge base and both sides' changes were read, git listed every changed
# path unquoted, and no path both sides changed matches `*.sh` by either name. Anything else --
# a one-parent HEAD, a failed git read, a path git had to quote -- runs every tier. The scope is
# one too: the tier narrows only when each matching path, asked alone, reaches a member of the
# tier, and the ask for them all answers. An ask that fails (test-harnesses.sh refuses a line of
# `affected --harnesses` it does not recognize, a path reading as an option, a failed `affected`),
# a matching path no member reaches, or a scoped run refused before any member started (exit 2)
# runs the whole tier, never less.
# The shared-red comparison (ship-pr's build of the base alone) stays like with like: that build
# carries the merge's two parents on the base's tree, so it reads the same two sides here and
# narrows to the same paths, the reach read from the base's text. A reach the two trees answer
# differently prints lines the base's log lacks, which ship-pr reads as the merge's own failure:
# a refusal, never a red waved through as shared.
# --all runs every tier whatever HEAD is. --plan prints the selection, asking what it reaches
# (no slot, no dune), and runs nothing.
#
# Toolchain: a non-interactive shell can lack the profile that puts the opam switch on PATH
# (on mac-studio `zsh -c` resolves opam but not dune). When `dune` does not resolve, this sources
# tools/opam-env.sh, first adding Homebrew's prefix when `opam` does not resolve either.
#
# Exit status: 0 every selected tier passed (or --plan), 1 a tier failed or no dune could be
# found, 2 a usage error. The first failing tier ends the run.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
me=merge-build-check
plan=0 all=0
while [ $# -gt 0 ]; do
  case $1 in
    --plan) plan=1 ;;
    --all) all=1 ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "$me: unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done
SHELL_GLOB='*.sh'

# read_side FROM TO sets side_names (every path the diff names, both names of a rename, one per
# line) and side_pairs (each rename as OLD<TAB>NEW, one per line). Status 1: git did not answer;
# 3: git quoted a path, which this does not decode.
read_side() {
  local out st a b
  out=$(git -C "$root" -c core.quotePath=false diff -M --name-status "$1" "$2") || return 1
  side_names= side_pairs=
  while IFS=$'\t' read -r st a b; do
    [ -n "$st" ] || continue
    case $a in '"'*) return 3 ;; esac
    case $b in '"'*) return 3 ;; esac
    case $st in
      R*) side_names+=$a$'\n'$b$'\n' side_pairs+=$a$'\t'$b$'\n' ;;
      *) side_names+=$a$'\n' ;;
    esac
  done <<EOF
$out
EOF
}

# ask_reach PATH... sets `reach`, the commands of the shell tier's members PATH reaches, one per
# line. It fails when the ask does; `--list` takes no slot and runs nothing.
ask_reach() {
  reach=$("${BASH:-bash}" "$root/tools/test-harnesses.sh" --shell --affected --list --files "$@" \
    2>/dev/null </dev/null)
}

# scope_tier narrows a selected tier to what `asked` reaches (`scoped` 1, `reach` its commands), or
# says in `scope_why` why the whole tier runs. `scripts` is every asked path matching SHELL_GLOB: a
# reach the others make up for would hide one nothing maps, so each is asked alone.
scope_tier() {
  local s rc
  for s in "${scripts[@]}"; do
    ask_reach "$s" || {
      rc=$?
      scope_why="what $s reaches could not be asked (test-harnesses.sh exit $rc)"
      return
    }
    [ -n "$reach" ] || { scope_why="no member of the tier reaches $s"; return; }
  done
  ask_reach "${asked[@]}" || {
    rc=$?
    scope_why="what the paths changed on both sides reach could not be asked (test-harnesses.sh exit $rc)"
    return
  }
  scoped=1
}

# select_tiers sets `shell` (1 = run the shell tier), `why` and, for a tier it could narrow,
# `scoped`, `reach` and `asked` (or `scope_why`), and prints what it read.
select_tiers() {
  shell=1 scoped=0 scope_why= reach= asked=() scripts=()
  if [ "$all" = 1 ]; then why='--all'; return; fi
  local base head mb base_names base_pairs both p old new other n=0 rc=0 hit=
  if ! base=$(git -C "$root" rev-parse --verify --quiet 'HEAD^1^{commit}') ||
     ! head=$(git -C "$root" rev-parse --verify --quiet 'HEAD^2^{commit}'); then
    why='HEAD is not a two-parent merge, so every tier runs'
    return
  fi
  echo "$me: HEAD merges ${head:0:8} into ${base:0:8}"
  mb=$(git -C "$root" merge-base "$base" "$head") || rc=1
  [ "$rc" != 0 ] || read_side "$mb" "$base" || rc=$?
  base_names=${side_names-} base_pairs=${side_pairs-}
  [ "$rc" != 0 ] || read_side "$mb" "$head" || rc=$?
  case $rc in
    0) ;;
    3) why='git quoted a path one side changed, so every tier runs'; return ;;
    *) why='the paths each side changed could not be read, so every tier runs'; return ;;
  esac
  both=$(LC_ALL=C comm -12 <(printf '%s' "$base_names" | LC_ALL=C sort -u) \
    <(printf '%s' "$side_names" | LC_ALL=C sort -u))
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    n=$((n + 1))
    asked+=("$p")
    case $p in
      $SHELL_GLOB)
        scripts+=("$p")
        [ -n "$hit" ] || hit="$p, changed on both sides, matches $SHELL_GLOB" ;;
    esac
    # A path counts by its other name when a rename on either side touched it.
    while IFS=$'\t' read -r old new; do
      [ -n "$old" ] || continue
      if [ "$old" = "$p" ]; then other=$new; elif [ "$new" = "$p" ]; then other=$old; else continue; fi
      asked+=("$other")
      case $other in
        $SHELL_GLOB)
          scripts+=("$other")
          [ -n "$hit" ] || hit="$p, changed on both sides, was renamed from or to $other, which matches $SHELL_GLOB" ;;
      esac
    done <<EOF
$base_pairs$side_pairs
EOF
  done <<EOF
$both
EOF
  if [ -n "$hit" ]; then
    why=$hit
    scope_tier
    return
  fi
  shell=0
  why="none of the $n path(s) changed on both sides matches $SHELL_GLOB, by either name of a rename"
}

select_tiers
echo "$me: compile tier: dune build @check @test/operations/scans (always)"
if [ "$shell" = 1 ]; then
  echo "$me: shell tier: SELECTED -- $why"
  if [ "$scoped" = 1 ]; then
    echo "$me: shell tier: scoped to what the ${#asked[@]} shared path(s) reach:"
    while IFS= read -r line; do echo "$me:   $line"; done <<EOF
$reach
EOF
  elif [ -n "$scope_why" ]; then
    echo "$me: shell tier: whole -- $scope_why"
  fi
else
  echo "$me: shell tier: skipped -- $why"
fi
[ "$plan" = 0 ] || exit 0

if ! command -v dune >/dev/null 2>&1; then
  if ! command -v opam >/dev/null 2>&1; then
    for prefix in /opt/homebrew/bin /usr/local/bin; do
      if [ -x "$prefix/opam" ]; then PATH=$prefix:$PATH; break; fi
    done
  fi
  echo "$me: dune is not on PATH; sourcing tools/opam-env.sh"
  # shellcheck source=tools/opam-env.sh
  . "$root/tools/opam-env.sh" || exit 1
  command -v dune >/dev/null 2>&1 || { echo "$me: no dune on PATH even after opam env" >&2; exit 1; }
fi
cd "$root" || exit 1
started=$SECONDS
dune build --root . @check @test/operations/scans || {
  rc=$?
  echo "$me: compile tier FAILED (dune exit $rc)" >&2
  exit 1
}
echo "$me: compile tier passed in $((SECONDS - started))s"
if [ "$shell" = 1 ]; then
  started=$SECONDS
  rc=0
  if [ "$scoped" = 1 ]; then
    tools/test-harnesses.sh --shell --affected --files "${asked[@]}" </dev/null || rc=$?
    if [ "$rc" = 2 ]; then
      # Every refusal comes before a member starts, so the whole tier still fits the bound.
      echo "$me: shell tier: the scoped run was refused (exit 2), no member run; running the whole tier"
      scoped=0 rc=0
    fi
  fi
  [ "$scoped" = 1 ] || tools/test-harnesses.sh --shell </dev/null || rc=$?
  if [ "$rc" != 0 ]; then
    echo "$me: shell tier FAILED (tools/test-harnesses.sh exit $rc)" >&2
    exit 1
  fi
  echo "$me: shell tier passed in $((SECONDS - started))s"
fi
echo "$me: passed"

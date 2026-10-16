#!/usr/bin/env bash
# Check a merge commit: the command `.github/ship-pr-merge-build` gives ship-pr's merge build,
# which runs it at the root of the PR head merged with master's tip, before the merge call.
# Run it by hand the same way, at the root of a checkout whose HEAD is a merge you built.
# Usage: tools/merge-build-check.sh [--plan] [--all]
#
# HEAD is read as the merge to judge: first parent the base, second parent the PR head (the
# order `git merge` and ship-pr's `commit-tree -p <base tip> -p <head>` give). Two tiers:
#   compile  always: `dune build @check @test/operations/scans` (about 40 s from a fresh tree
#            on mac-studio: the compile and the repository scans, which read the dune stanzas,
#            sources and scripts both sides edit).
#   shell    `tools/test-harnesses.sh --shell`, the toolchain-free harness tier CI runs (about
#            470 s on mac-studio), only when a path BOTH sides changed since their merge base
#            matches `*.sh`: the whole path, `*` crossing `/`, as the config's `input` globs read,
#            a path a rename on either side touched (git's -M detection) counting by its other
#            name too -- the merge build's own trigger, narrowed to shell scripts.
# The selection is a fail-closed allowlist: the shell tier is skipped only when HEAD is a
# two-parent merge, its merge base and both sides' changes were read, git listed every changed
# path unquoted, and no path both sides changed matches `*.sh` by either name. Anything else --
# a one-parent HEAD, a failed git read, a path git had to quote -- runs every tier.
# --all runs every tier whatever HEAD is. --plan prints the selection and runs nothing.
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

# select_tiers sets `shell` (1 = run the shell tier) and `why`, and prints what it read.
select_tiers() {
  shell=1
  if [ "$all" = 1 ]; then why='--all'; return; fi
  local base head mb base_names base_pairs both p old new other n=0 rc=0
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
    case $p in
      $SHELL_GLOB) why="$p, changed on both sides, matches $SHELL_GLOB"; return ;;
    esac
    # A path counts by its other name when a rename on either side touched it.
    while IFS=$'\t' read -r old new; do
      [ -n "$old" ] || continue
      if [ "$old" = "$p" ]; then other=$new; elif [ "$new" = "$p" ]; then other=$old; else continue; fi
      case $other in
        $SHELL_GLOB) why="$p, changed on both sides, was renamed from or to $other, which matches $SHELL_GLOB"; return ;;
      esac
    done <<EOF
$base_pairs$side_pairs
EOF
  done <<EOF
$both
EOF
  shell=0
  why="none of the $n path(s) changed on both sides matches $SHELL_GLOB, by either name of a rename"
}

select_tiers
echo "$me: compile tier: dune build @check @test/operations/scans (always)"
if [ "$shell" = 1 ]; then
  echo "$me: shell tier: SELECTED -- $why"
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
  tools/test-harnesses.sh --shell </dev/null || {
    rc=$?
    echo "$me: shell tier FAILED (tools/test-harnesses.sh exit $rc)" >&2
    exit 1
  }
  echo "$me: shell tier passed in $((SECONDS - started))s"
fi
echo "$me: passed"

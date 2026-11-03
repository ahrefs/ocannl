#!/usr/bin/env bash
# Compare what the preprocessors make of the tree at two revisions: the reach of a ppx change is
# what it changes in the expanded sources (gh-ocannl-1886).
# Usage: tools/ppx-output-diff.sh [-C REPO] [-j N] [--out DIR] BASE HEAD [-- DIR...]
#
# Each revision is checked out in a throwaway `git worktree add --detach` (no checkout of REPO
# moves, and no hook runs), where dune builds every preprocessed rendering the build defines --
# each `<name>.pp.ml` / `<name>.pp.mli` that `dune show targets` lists in a directory holding a
# tracked `dune` file, or only in the DIRs given after `--` and below them -- and nothing else:
# no library is compiled beyond the ppx drivers and what they link. Each rendering (dune keeps
# ppx output as a binary AST) is printed back to source by `ocamlc -stop-after parsing -dsource`,
# and the worktree is removed. Both revisions are built at the SAME path, one after the other, so
# a path a preprocessor embeds reads the same on both sides. REPO defaults to the repository
# holding this script; BASE and HEAD are any revisions it resolves to commits.
#
# stdout is `diff -ru base head` over the two rendered trees (a file one side lacks is an
# `Only in` line). stderr names every file that differs, one line each, by what changed:
#   EXPANSION  the preprocessor's input (`_build/default/<dir>/<name>.ml`, after any copy_files)
#              is byte-identical on both sides, so the difference is the preprocessors' alone --
#              the silent reach a ppx change has, and the reason this tool exists;
#   SOURCE     the input changed too;
#   ADDED / REMOVED  the rendering exists at one revision only;
# then a closing `ppx-output-diff:` verdict line with the counts. --out DIR keeps the rendered
# trees (DIR/base, DIR/head) and the diff (DIR/ppx.diff); DIR must not exist yet.
#
# Boundary: the corpus is the renderings dune's own rules define for the tree, nothing more. OCaml
# text no dune stanza preprocesses -- a scan table's case snippets, which
# tools/typecheck-scan-cases.sh expands itself, a harness's planted sources, a Markdown block --
# is outside it, and a rendering only some build configuration defines is compared only as that
# configuration (the one this host's dune sees) defines it.
#
# The two builds are ordinary dune builds (about 80 s for both over the whole tree on mac-studio,
# with a warm dune cache), and the script takes no fleet slot itself: on a fleet box run it under
# `fleet-worker.sh execution slot --cpu --` (it holds no device). -j N is passed to dune.
#
# Exit status: 0 the renderings are identical, 1 they differ, 2 a usage error, an unresolvable
# revision, a failed worktree, enumeration, build or rendering, or a revision with no
# preprocessed file in scope (a corpus that went blind is no verdict).
set -u
me=ppx-output-diff
die() { echo "$me: $*" >&2; exit 2; }
repo=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
jobs= out= revs=() scope=()
while [ $# -gt 0 ]; do
  case $1 in
    -C) [ $# -ge 2 ] && [ -n "$2" ] || die "-C needs a repository"; repo=$2; shift ;;
    -j) [ $# -ge 2 ] || die "-j needs a count"
        case $2 in '' | *[!0-9]* | 0) die "-j needs a positive count, not '$2'" ;; esac
        jobs=$2; shift ;;
    --out) [ $# -ge 2 ] && [ -n "$2" ] || die "--out needs a directory"; out=$2; shift ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    --) shift; scope=("$@"); break ;;
    -*) die "unknown option '$1'" ;;
    *) revs+=("$1") ;;
  esac
  shift
done
[ "${#revs[@]}" -eq 2 ] || die "usage: tools/ppx-output-diff.sh [-C REPO] [-j N] [--out DIR] BASE HEAD [-- DIR...]"
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2
command -v dune >/dev/null 2>&1 || . "$here/opam-env.sh" 2>/dev/null
for tool in git dune ocamlc diff; do
  command -v "$tool" >/dev/null 2>&1 || die "no $tool on PATH"
done
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies, after the opam
# environment above, whose updates could set one.
# shellcheck source=credential-env.sh
. "$here/credential-env.sh" || die "cannot read $here/credential-env.sh"
eval "$(credential_env_scrub_text)" || die "cannot remove credential variables:$credential_env_left"
# Both builds go to the throwaway worktree's own `_build`, which is where the renderings are read:
# an exported DUNE_BUILD_DIR would send them elsewhere.
unset DUNE_BUILD_DIR
repo=$(git -C "$repo" rev-parse --show-toplevel 2>/dev/null) || die "not a git repository: $repo"
for d in ${scope[@]+"${scope[@]}"}; do
  d=${d%/}
  case $d in
    '' | /* | ..* | *[!A-Za-z0-9_./-]*) die "a scope DIR is a repository-relative path: '$d'" ;;
  esac
done
base=$(git -C "$repo" rev-parse --verify --quiet "${revs[0]}^{commit}") || die "BASE does not resolve to a commit: ${revs[0]}"
head=$(git -C "$repo" rev-parse --verify --quiet "${revs[1]}^{commit}") || die "HEAD does not resolve to a commit: ${revs[1]}"
if [ -n "$out" ]; then
  [ ! -e "$out" ] || die "--out $out already exists"
  mkdir -p "$out" || die "cannot create $out"
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/$me.XXXXXX") || die "cannot create a temporary directory"
[ -n "$out" ] || out=$tmp/out
mkdir -p "$out" || die "cannot create $out"
wt=$tmp/tree
wt_added=0 wt_admin=
git_common=$(cd "$repo" && cd "$(git rev-parse --git-common-dir)" && pwd -P) || die "cannot find $repo's git directory"
remove_tree() {
  [ "$wt_added" = 1 ] || return 0
  wt_added=0
  # Only this invocation's registration goes: never a repository-wide `worktree prune`, which
  # could unregister another worktree whose path is unavailable for the moment.
  git -C "$repo" worktree remove --force "$wt" >/dev/null 2>&1 && return 0
  rm -rf "${wt:?}"
  case $wt_admin in
    "$git_common"/worktrees/?*) rm -rf "${wt_admin:?}" ;;
  esac
}
cleanup() {
  remove_tree
  rm -rf "${tmp:?}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

in_scope() { # DIR: whether a directory holding a dune file is in the requested scope
  local s
  [ "${#scope[@]}" -gt 0 ] || return 0
  for s in "${scope[@]}"; do
    s=${s%/}
    [ "$s" = . ] && return 0
    case $1 in "$s" | "$s"/*) return 0 ;; esac
  done
  return 1
}

# render ROLE SHA: build and print back one revision's renderings into $out/ROLE, and record each
# rendering's preprocessor input (by blob hash) in $tmp/ROLE.inputs as `PATH<TAB>HASH`.
render() {
  local role=$1 sha=$2 dirs=() d line cur= targets=() t n=0 src
  git -C "$repo" -c core.hooksPath=/dev/null worktree add --detach --quiet "$wt" "$sha" \
    >"$tmp/log" 2>&1 || { cat "$tmp/log" >&2; die "$role: cannot check $sha out in a worktree"; }
  wt_added=1
  wt_admin=$(cd "$(git -C "$wt" rev-parse --absolute-git-dir)" && pwd -P) 2>/dev/null || wt_admin=
  while IFS= read -r d; do
    d=$(dirname -- "$d")
    in_scope "$d" && dirs+=("$d")
  done < <(git -C "$wt" ls-files -- dune '*/dune')
  [ "${#dirs[@]}" -gt 0 ] || die "$role ($sha): no directory with a dune file in scope"
  # dune (3.24) prints this listing on stderr, its console, and reports a directory it cannot
  # list there too while still exiting 0. So the listing is read fail-closed: with several
  # directories each list sits under a `DIR:` header naming one of them, and every one must
  # appear once; with one, the names come bare. Any other line -- an `Error:`, a name outside the
  # path alphabet -- refuses the run.
  dune show targets --root "$wt" --build-dir "$wt/_build" --no-print-directory "${dirs[@]}" >"$tmp/log" 2>"$tmp/targets" || {
    cat "$tmp/targets" >&2; die "$role: dune show targets failed"; }
  [ "${#dirs[@]}" -gt 1 ] || { cur=${dirs[0]}; n=1; }
  while IFS= read -r line; do
    if [ "${#dirs[@]}" -gt 1 ] && [ "${line%:}" != "$line" ]; then
      cur=${line%:}
      case " ${dirs[*]} " in *" $cur "*) n=$((n + 1)) ;; *) cur= ;; esac
      [ -n "$cur" ] || break
      continue
    fi
    case $line in
      '') continue ;;
      *[!A-Za-z0-9_.+-]*) cur= ; break ;;
    esac
    [ -n "$cur" ] || break
    case $line in
      *.pp.ml | *.pp.mli)
        if [ "$cur" = . ]; then targets+=("$line"); else targets+=("$cur/$line"); fi ;;
    esac
  done <"$tmp/targets"
  if [ -z "$cur" ] || [ "$n" != "${#dirs[@]}" ]; then
    head -n 20 "$tmp/targets" >&2
    die "$role: cannot read dune show targets' listing of ${#dirs[@]} director(ies) (line: '${line-}')"
  fi
  n=0
  [ "${#targets[@]}" -gt 0 ] ||
    die "$role ($sha): no preprocessed file in scope (${#dirs[@]} director(ies) listed): nothing to compare"
  dune build --root "$wt" --build-dir "$wt/_build" ${jobs:+-j "$jobs"} "${targets[@]}" >"$tmp/log" 2>&1 || {
    tail -n 40 "$tmp/log" >&2; die "$role: dune failed to build the renderings of $sha"; }
  : >"$tmp/$role.inputs"
  for t in "${targets[@]}"; do
    mkdir -p "$out/$role/$(dirname -- "$t")" || die "cannot create $out/$role"
    ocamlc -stop-after parsing -dsource -c "$wt/_build/default/$t" >/dev/null 2>"$out/$role/$t" || {
      cat "$out/$role/$t" >&2; die "$role: ocamlc cannot print $t back to source"; }
    case $t in *.pp.ml) src=${t%.pp.ml}.ml ;; *) src=${t%.pp.mli}.mli ;; esac
    if [ -f "$wt/_build/default/$src" ]; then
      printf '%s\t%s\n' "$t" "$(git hash-object --no-filters -- "$wt/_build/default/$src")" >>"$tmp/$role.inputs"
    else
      printf '%s\t-\n' "$t" >>"$tmp/$role.inputs"
    fi
    n=$((n + 1))
  done
  echo "$me: $role ${sha:0:12}: $n rendering(s) from ${#dirs[@]} director(ies)" >&2
  remove_tree
}

render base "$base"
render head "$head"

rc=0
(cd "$out" && diff -ru base head) >"$tmp/diff" || rc=$?
[ "$rc" -le 1 ] || { cat "$tmp/diff" >&2; die "diff failed (exit $rc)"; }
# The two header lines after each `diff -ru` line carry the files' mtimes: drop them, so the same
# two revisions always give the same text.
awk 'hdr > 0 { sub(/\t.*/, ""); hdr-- } /^diff -ru / { hdr = 2 } { print }' "$tmp/diff" >"$out/ppx.diff" ||
  die "cannot write $out/ppx.diff"
cat "$out/ppx.diff"
# Every rendering either side has, as `PATH BASE-INPUT HEAD-INPUT`, `absent` for a side without it
# (`-` is a rendering whose input dune did not leave beside it, never equal to anything).
awk -F '\t' 'FNR == NR { b[$1] = $2; next } { h[$1] = $2 }
  END {
    for (p in b) print p, b[p], ((p in h) ? h[p] : "absent")
    for (p in h) if (!(p in b)) print p, "absent", h[p]
  }' "$tmp/base.inputs" "$tmp/head.inputs" | LC_ALL=C sort >"$tmp/pairs"
expansion=0 source=0 added=0 removed=0
while read -r t b h; do
  if [ "$b" = absent ]; then echo "$me: ADDED      $t" >&2; added=$((added + 1))
  elif [ "$h" = absent ]; then echo "$me: REMOVED    $t" >&2; removed=$((removed + 1))
  elif cmp -s "$out/base/$t" "$out/head/$t"; then :
  elif [ "$b" = "$h" ] && [ "$b" != - ]; then echo "$me: EXPANSION  $t" >&2; expansion=$((expansion + 1))
  else echo "$me: SOURCE     $t" >&2; source=$((source + 1))
  fi
done <"$tmp/pairs"
differ=$((expansion + source + added + removed))
if [ "$rc" = 0 ] && [ "$differ" = 0 ]; then
  echo "$me: no difference between ${base:0:12} and ${head:0:12} over $(wc -l <"$tmp/head.inputs" | tr -d ' ') rendering(s)" >&2
elif [ "$rc" = 1 ] && [ "$differ" -gt 0 ]; then
  echo "$me: $differ rendering(s) differ between ${base:0:12} and ${head:0:12}: $expansion EXPANSION (input unchanged), $source SOURCE, $added ADDED, $removed REMOVED" >&2
else
  die "diff (exit $rc) and the per-file comparison ($differ differing) disagree"
fi
exit "$rc"

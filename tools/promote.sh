#!/usr/bin/env bash
# Promote dune test outputs and normalize line endings in the promoted goldens.
#
# `dune promote` on Windows copies CRLF test output into `.expected` files
# (test exe stdout is text-mode); goldens are LF in the repo (.gitattributes).
# This wrapper promotes and then strips trailing CRs from the promoted files,
# replacing the manual `sed -i 's/\r$//'` ritual. It also works from worktrees
# nested inside the repo, where plain `dune promote` resolves the PARENT
# checkout: `dune promotion apply` accepts the `--root .` override.
#
# It also guards the mid-merge promotion trap (staging PR #487, ~90 minutes):
# promotion writes the WORKING TREE, but `git commit` during a merge takes the
# INDEX, so a promotion made after `git add` is committed as the pre-promotion
# content. Nothing local complains -- every `dune runtest` reads the working
# tree and passes -- while CI builds the committed tree and fails on the golden
# diff. After promoting, this script stages what it promoted (and says so), or
# names the files it would not stage. Outside a merge it is a no-op.
#
# Usage: tools/promote.sh [FILES...]
#        tools/promote.sh --from-run RUN [FILES...]
#   Replay saved corrected outputs from a finished test-run record in this
#   worktree. FILES selects exact paths. Refuse incomplete records or changed
#   source files; the saved originals protect edits made after the run.
#   Run from anywhere; extra arguments are passed through to dune (e.g. paths
#   of specific files to promote).

set -eu
cd "$(dirname "$0")/.."

die() { echo "promote.sh: $*" >&2; exit 2; }
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies.
[ -r tools/credential-env.sh ] || die "cannot read tools/credential-env.sh"
# shellcheck source=credential-env.sh
. tools/credential-env.sh
eval "$(credential_env_scrub_text)" || die "cannot remove credential variables:$credential_env_left"
matches_correction() { # destination correction; CR normalization is part of promotion
  cmp -s "$1" "$2" && return 0
  case $1 in
    *.expected | test/ppx/*_expected.ml) perl -pe 's/\r$//' "$2" | cmp -s "$1" - ;;
    *) return 1 ;;
  esac
}
from_run=
if [ "${1:-}" = --from-run ]; then
  [ $# -ge 2 ] || die "--from-run requires a run directory"
  from_run=$2
  shift 2
  [ -d "$from_run" ] || die "no such run: $from_run"
  from_run=$(CDPATH= cd -- "$from_run" && pwd -P)
else
  command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
fi

# Are we mid-merge? `git rev-parse --verify MERGE_HEAD` rather than testing
# `.git/MERGE_HEAD`: in a linked worktree `.git` is a FILE, and MERGE_HEAD
# lives in the per-worktree gitdir that only git can resolve. A non-repository
# cwd answers "no" here, which is the right answer for the guard.
merging=0
if git rev-parse -q --verify MERGE_HEAD >/dev/null 2>&1; then
  merging=1
fi

# The promotion list has to be taken BEFORE applying -- afterwards there is
# nothing pending left to name. It is only needed for the guard, so outside a
# merge the extra dune invocation is skipped entirely. `list` filters its
# arguments exactly as `apply` does, prints one root-relative path per line on
# stdout since 3.22, stderr before, and sends missing-file warnings to stderr.
#
# A `list` that FAILS is kept apart from one that finds nothing: both leave
# `promoted` empty, but the first means the guard is about to do nothing while
# believing it did its job -- silently reinstating the trap. Say so instead.
promoted=""
listed=1
if [ -n "$from_run" ]; then
  saved=$from_run/promotion-files
  [ -f "$from_run/exit" ] && [ -f "$saved/paths" ] || die "run has no complete saved promotions: $from_run"
  [ "$(cat "$from_run/wt")" = "$(pwd -P)" ] || die "run belongs to another worktree"
  # Validate the entire selection before copying any file. Numbered payloads
  # avoid interpreting a recorded source path as a path inside the run record.
  n=0
  selected=()
  while IFS= read -r f; do
    n=$((n + 1))
    case $f in '' | /* | \\* | [A-Za-z]:* | . | .. | ./* | ../* | */./* | */../* | */. | */..) die "invalid saved path: $f" ;; esac
    parent=${f%/*}
    [ "$parent" != "$f" ] || parent=.
    resolved=$(CDPATH= cd -- "$parent" && pwd -P) || die "missing destination directory: $f"
    case $resolved/ in "$(pwd -P)/"*) ;; *) die "destination escapes worktree: $f" ;; esac
    [ ! -L "$f" ] || die "destination is a symlink: $f"
    [ -f "$saved/$n.corrected" ] || die "missing saved correction: $f"
    choose=0
    if [ $# -eq 0 ]; then choose=1; else
      for arg do [ "$arg" != "$f" ] || choose=1; done
    fi
    [ "$choose" = 1 ] || continue
    if [ -f "$saved/$n.original" ]; then
      cmp -s "$f" "$saved/$n.original" || matches_correction "$f" "$saved/$n.corrected" || die "source changed since run: $f"
    else
      [ -f "$saved/$n.absent" ] || die "missing saved original: $f"
      [ ! -e "$f" ] || matches_correction "$f" "$saved/$n.corrected" || die "source changed since run: $f"
    fi
    selected+=("$n")
    promoted="$promoted$f
"
  done <"$saved/paths"
  for arg do
    printf '%s' "$promoted" | grep -Fx -- "$arg" >/dev/null || die "no saved promotion for $arg"
  done
  n=0
  while IFS= read -r f; do
    n=$((n + 1))
    for pick in "${selected[@]}"; do
      [ "$pick" != "$n" ] || cp -- "$saved/$n.corrected" "$f"
    done
  done <"$saved/paths"
else
  if [ "$merging" -eq 1 ]; then
    # Capture both streams: 3.20/3.21 put paths on stderr. Retain only paths
    # Dune actually promoted; missing-path warnings must never reach git add.
    promoted="$(dune promotion list --root . "$@" --diff-command=diff 2>&1)" || listed=0
  fi
  dune promotion apply --root . "$@"
fi

# Strip trailing CRs from a promoted golden. perl -i, not sed -i: BSD sed
# (macOS) requires a backup-suffix argument for -i, so GNU-style `sed -i`
# errors there; perl is portable (and already a dependency of
# tools/test-run.sh).
strip_crs() { # strip_crs FILE -- no-op unless FILE is a golden we pin to LF
  case "$1" in
    *.expected | test/ppx/*_expected.ml) ;;
    *) return 0 ;;
  esac
  [ -f "$1" ] && perl -i -pe 's/\r$//' "$1"
  return 0
}

# Any promoted golden that now differs from the index.
git diff --name-only -z -- '*.expected' 'test/ppx/*_expected.ml' \
  | while IFS= read -r -d '' f; do
      strip_crs "$f"
    done

# Recorded new files need normalization too, even outside a merge where
# git diff cannot name untracked destinations.
if [ -n "$from_run" ]; then
  while IFS= read -r f; do
    [ -z "$f" ] || strip_crs "$f"
  done <<EOF
$promoted
EOF
fi

[ "$merging" -eq 1 ] || exit 0

if [ "$listed" -eq 0 ]; then
  printf '\npromote.sh: WARNING -- mid-merge, but `dune promotion list` failed, so\n' >&2
  printf 'this script does not know what it just promoted and has staged NOTHING.\n' >&2
  printf 'A merge commit takes the index, not the working tree, so stage the\n' >&2
  printf 'promoted goldens yourself or they are dropped from the commit and fail\n' >&2
  printf 'in CI only. `git status` will show them as modified.\n' >&2
  exit 0
fi

# Mid-merge: stage what was promoted, so the commit carries it.
#
# A file still UNMERGED in the index is left alone: `git add` on one records a
# resolution, and whether this promotion is that resolution is the caller's
# call, not the script's. Everything else -- already resolved, or an entirely
# new golden -- is staged, which is what closes the trap.
staged=()
unmerged=()
while IFS= read -r f; do
  [ -n "$f" ] || continue
  # `list` on the floor shares stderr with diagnostics; only existing
  # root-relative files can have been promoted. Never stage warning text.
  case $f in /* | \\* | [A-Za-z]:* | ../* | */../*) continue ;; esac
  [ -f "$f" ] || continue
  # Promotion may have introduced CRs into a file the `git diff` pass above
  # could not see (a new golden, absent from the index), so re-check here;
  # strip_crs is idempotent and the staged content must be LF either way.
  strip_crs "$f"
  if [ -n "$(git ls-files --unmerged -- "$f")" ]; then
    unmerged+=("$f")
  else
    staged+=("$f")
  fi
done <<EOF
$promoted
EOF

if [ ${#staged[@]} -gt 0 ]; then
  # Reported, never fatal. The promotion has already been applied by this
  # point, so exiting here on a `git add` that will not take a path (an ignored
  # golden, say) would leave exactly the state this guard exists to prevent,
  # and leave it SILENTLY -- `set -e` prints nothing.
  if add_err="$(git add -- "${staged[@]}" 2>&1)"; then
    printf '\npromote.sh: mid-merge, so staged what it promoted:\n' >&2
    printf '  %s\n' "${staged[@]}" >&2
    printf 'A merge commit takes the index, not the working tree, so without this the\n' >&2
    printf 'promotion would be dropped from the commit and fail in CI only.\n' >&2
  else
    printf '\npromote.sh: WARNING -- promoted, but `git add` REFUSED to stage:\n' >&2
    printf '  %s\n' "${staged[@]}" >&2
    printf '%s\n' "$add_err" >&2
    printf 'A merge commit takes the index, so until these are staged the promotion\n' >&2
    printf 'is dropped from the commit and fails in CI only.\n' >&2
  fi
fi

if [ ${#unmerged[@]} -gt 0 ]; then
  printf '\npromote.sh: WARNING -- promoted, but still UNMERGED, so NOT staged:\n' >&2
  printf '  %s\n' "${unmerged[@]}" >&2
  printf 'Staging one of these records it as the conflict resolution, which is yours\n' >&2
  printf 'to decide. But a merge commit takes the index, so until you do, the\n' >&2
  printf 'promotion is dropped from the commit and fails in CI only. To accept:\n' >&2
  printf '  git add --' >&2
  printf ' %s' "${unmerged[@]}" >&2
  printf '\n' >&2
fi

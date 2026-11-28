#!/usr/bin/env bash
# Usage: tools/windows-leg.sh SHA @ALIAS [@ALIAS ...]
# Start in a Git clone on an already booted Windows host with Git Bash and
# a provisioned opam switch. A missing commit is fetched by SHA from origin.
# Every invocation keeps its own detached worktree and evidence under TMPDIR
# (or /tmp). The caller owns boot, provisioning and cleanup after collection.
# A refused or interrupted run may leave that directory for diagnosis.
#
# Input is a full lowercase Git SHA and literal Dune @/@@ aliases with
# slash-separated names. Dune options, file targets and revision names are
# outside this interface. The checkout guard reads the index/worktree EOL
# columns of git ls-files for test/support/verdict.ml. The receipt reads the
# runner's own paths query and exit file. Per-action Verdict TSVs are relayed
# verbatim. Test logs, golden files and claim text are never parsed.
set -eu
set -o pipefail

finish() {
  printf 'exit: %s\n' "$1"
  exit "$1"
}

refuse() {
  printf 'windows-leg: REFUSED %s\n' "$*" >&2
  finish 2
}

[ "$#" -ge 2 ] || refuse 'usage: tools/windows-leg.sh SHA @ALIAS [@ALIAS ...]'
sha=$1
shift
[[ $sha =~ ^[0-9a-f]{40}$ ]] || refuse 'expected a full lowercase 40-hex commit SHA'
alias_pattern='^@@?[A-Za-z0-9_][A-Za-z0-9_.+-]*(/[A-Za-z0-9_][A-Za-z0-9_.+-]*)*$'
for target do
  [[ $target =~ $alias_pattern ]] || refuse "expected a literal Dune alias, got $(printf '%q' "$target")"
done

# A shell started from PowerShell can lack even uname on PATH.
case ${OSTYPE:-} in
  msys* | cygwin*) export PATH="/usr/bin:/mingw64/bin:$PATH" ;;
esac
[ "$(uname -o 2>/dev/null)" = Msys ] || refuse 'native Git Bash (MSYS) is required'

# Fixture repositories must inherit the same line-ending policy as this checkout.
export GIT_CONFIG_COUNT=2
export GIT_CONFIG_KEY_0=core.autocrlf GIT_CONFIG_VALUE_0=false
export GIT_CONFIG_KEY_1=core.eol GIT_CONFIG_VALUE_1=lf
git rev-parse --git-dir >/dev/null || refuse 'start in a Git clone'
if ! git cat-file -e "$sha" 2>/dev/null; then
  git fetch --no-tags origin "$sha" || refuse "could not fetch $sha from origin"
fi
[ "$(git cat-file -t "$sha")" = commit ] || refuse "$sha is not a commit"

leg_dir=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-win.XXXXXX") || refuse 'could not allocate a leg directory'
printf 'windows-leg: leg directory: %s\n' "$leg_dir"
git worktree add --detach "$leg_dir/w" "$sha" || refuse 'could not create the detached worktree'
cd -- "$leg_dir/w" || refuse 'could not enter the detached worktree'
[ "$(git rev-parse HEAD)" = "$sha" ] || refuse "checkout HEAD differs from $sha"
eol=$(git ls-files --eol -- test/support/verdict.ml) || refuse 'could not read checkout line endings'
[[ $eol =~ ^i/lf[[:space:]]+w/lf[[:space:]] ]] || refuse 'test/support/verdict.ml must be LF in the index and worktree'

# The runner sources opam-env.sh and selects dune-quiet.sh on Windows.
# The cache fixture needs the original Git Bash tools.
OCANNL_TOOL_HARNESS_STRICT=1 "$BASH" tools/test-windows-opam-cache.sh || finish "$?"

mkdir "$leg_dir/claims" || refuse 'could not create the claim records directory'
# Verdict is native OCaml and needs a drive path. Resolve it while Git Bash's
# cygpath still takes precedence over the switch's Cygwin executable.
claims=$(cygpath -am "$leg_dir/claims") || refuse 'could not resolve the native claim records path'
export OCANNL_TOOL_VERDICT_RECORDS="$claims"
export OCANNL_TOOL_TEST_RUNS="$leg_dir/runs"
unset DUNE_BUILD_DIR
export OCANNL_BACKEND="${OCANNL_BACKEND:-cc}"
printf 'windows-leg: host=%s sha=%s backend=%s\n' "$(hostname)" "$sha" "$OCANNL_BACKEND"

rc=0
"$BASH" tools/test-run.sh run --expect-backend "$OCANNL_BACKEND" build \
  --stop-on-first-error --cache=disabled -j 4 "$@" || rc=$?

run_dir=$("$BASH" tools/test-run.sh paths run last) || run_dir=
if [ -n "$run_dir" ]; then
  printf 'windows-leg: run directory: %s\n' "$run_dir"
else
  printf 'windows-leg: run directory: unavailable\n'
fi
receipt_ok=0
if [ -n "$run_dir" ] && [ -r "$run_dir/exit" ] &&
   IFS= read -r recorded_rc <"$run_dir/exit" && [ "$recorded_rc" = "$rc" ]; then
  receipt_ok=1
fi

printf 'windows-leg: claim records: %s\n' "$claims"
shopt -s nullglob
set -- "$leg_dir/claims/"*.tsv
export_ok=1
if [ "$#" -eq 0 ]; then
  printf 'windows-leg: no Verdict action records were produced\n'
else
  cat -- "$@" || export_ok=0
fi
if [ "$receipt_ok" = 0 ]; then
  printf 'windows-leg: runner receipt missing or inconsistent with exit %s\n' "$rc" >&2
fi
if [ "$export_ok" = 0 ]; then
  printf 'windows-leg: could not relay the Verdict action records\n' >&2
fi
if [ "$rc" = 0 ] && { [ "$receipt_ok" = 0 ] || [ "$export_ok" = 0 ]; }; then
  finish 2
fi
finish "$rc"

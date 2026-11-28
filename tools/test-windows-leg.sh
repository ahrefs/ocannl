#!/usr/bin/env bash
# Exercise the Windows leg in a local Git fixture with stubbed MSYS path and
# toolchain commands. Real opam, Dune and native execution belong to the witness.
# Usage: tools/test-windows-leg.sh [--subject FILE] [--only REGEX] [--keep]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/../scripts/harness-support.sh"
subject_file=
harness_own_option --subject subject_file 'a file'
harness_args "$@"
harness_require git
harness_scratch test-windows-leg
subject=${subject_file:-$HERE/windows-leg.sh}
one=@test/operations/runtest-verdict_teardown
two=@@test/operations/runtest-verdict_quantified

seed() {
  clone="$TMP/clone"
  git init -q "$clone" &&
    mkdir -p "$clone/tools" "$clone/test/support" &&
    printf 'let fixture = ()\n' >"$clone/test/support/verdict.ml" || return 1
  cat >"$clone/tools/test-windows-opam-cache.sh" <<'FIXTURE'
#!/usr/bin/env bash
set -eu
[ "$OCANNL_TOOL_HARNESS_STRICT" = 1 ]
[ "$(git config --get core.autocrlf)" = false ]
[ "$(git config --get core.eol)" = lf ]
[ "$(git rev-parse HEAD)" = "$FIXTURE_SHA" ]
if git symbolic-ref -q HEAD; then exit 90; fi
echo cache >>"$FIXTURE_CALLS"
exit "$FIXTURE_CACHE_RC"
FIXTURE
  cat >"$clone/tools/test-run.sh" <<'FIXTURE'
#!/usr/bin/env bash
set -eu
case "$1" in
  run)
    [ "$(cat "$FIXTURE_CALLS")" = cache ]
    [ "$OCANNL_BACKEND" = "$FIXTURE_BACKEND" ]
    [ -z "${DUNE_BUILD_DIR+x}" ]
    printf '%s\n' "$@" >>"$FIXTURE_CALLS"
    case "$FIXTURE_RECORD" in
      missing) ;;
      *) mkdir -p "$OCANNL_TOOL_TEST_RUNS/this-run"
         printf '%s\n' "$FIXTURE_RECORD" >"$OCANNL_TOOL_TEST_RUNS/this-run/exit" ;;
    esac
    if [ "$FIXTURE_CLAIMS" = yes ]; then
      printf 'OCANNL_TOOL_VERDICT_ACTION\tfixture.exe\twork dir\t\nOCANNL_TOOL_VERDICT_EVALUATED\tfixture.exe\ta checked claim\nOCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\ta skipped claim\n' >"$OCANNL_TOOL_VERDICT_RECORDS/actions.tsv"
    fi
    printf 'verdict: fixture (exit %s)\n' "$FIXTURE_RUN_RC"
    exit "$FIXTURE_RUN_RC"
    ;;
  paths)
    [ "$*" = 'paths run last' ]
    [ -d "$OCANNL_TOOL_TEST_RUNS/this-run" ]
    printf '%s/this-run\n' "$OCANNL_TOOL_TEST_RUNS"
    ;;
  *) exit 99 ;;
esac
FIXTURE
  (
    cd "$clone" &&
      git config core.autocrlf true &&
      git -c core.autocrlf=false add . &&
      git -c user.name=fixture -c user.email=fixture@example.test commit -qm fixture
  ) && sha=$(git -C "$clone" rev-parse HEAD)
}

invoke() { # LABEL CWD [SHA ALIAS ...]
  local label=$1 cwd=$2
  shift 2
  transcript="$TMP/$label.out" calls="$TMP/$label.calls" rc=0
  (
    cd "$cwd" || exit 98
    uname() { [ "$*" = -o ] && printf '%s\n' "$FIXTURE_OS"; }
    cygpath() { [ "$1" = -am ] && printf '%s\n' "$2"; }
    export -f uname cygpath
    env TMPDIR="$TMP" DUNE_BUILD_DIR=ignored \
      OCANNL_BACKEND="${leg_backend:-}" FIXTURE_BACKEND="${leg_backend:-cc}" \
      FIXTURE_OS="${leg_os:-Msys}" FIXTURE_SHA="${sha:-}" FIXTURE_CALLS="$calls" \
      FIXTURE_CACHE_RC="${cache_rc:-0}" FIXTURE_RUN_RC="${run_rc:-0}" \
      FIXTURE_RECORD="${record:-0}" FIXTURE_CLAIMS="${claims:-yes}" \
      "$BASH" "$subject" "$@"
  ) >"$transcript" 2>&1 || rc=$?
}

refusal() {
  [ "$rc" = 2 ] && [ ! -e "$calls" ] &&
    grep -Fq -- "$1" "$transcript" &&
    [ "$(tail -n 1 "$transcript")" = 'exit: 2' ]
}

if harness_leg '1: argument and platform refusals precede Git'; then
  invoke no-alias "$TMP" 123
  refusal 'usage:'; report $? 'a missing alias is refused'
  invoke short-sha "$TMP" abc "$one"
  refusal 'full lowercase 40-hex commit SHA'; report $? 'an abbreviated SHA is refused'
  sample_sha=0123456789abcdef0123456789abcdef01234567
  for alias_arg in --force test/file.ml '@test/../runtest-x' ''; do
    invoke bad-alias "$TMP" "$sample_sha" "$one" "$alias_arg"
    refusal 'expected a literal Dune alias'; report $? "a non-alias operand is refused (${alias_arg:-empty})"
  done
  leg_os=Cygwin invoke cygwin "$TMP" "$sample_sha" "$one"
  refusal 'native Git Bash'; report $? 'Cygwin is refused'
  invoke no-clone "$TMP" "$sample_sha" "$one"
  refusal 'start in a Git clone'; report $? 'a directory outside a clone is refused'
fi

if harness_leg '2: detached LF checkout, cache fixture, batch argv and claim export'; then
  seed || exit 2
  # The caller's checkout may be dirty and even disagree with the fixture at SHA.
  git -C "$clone" -c user.name=fixture -c user.email=fixture@example.test commit --allow-empty -qm newer || exit 2
  printf '#!/bin/sh\nexit 98\n' >"$clone/tools/test-run.sh"
  invoke green "$clone" "$sha" "$one" "$two"
  printf '%s\n' cache run --expect-backend cc build --stop-on-first-error --cache=disabled -j 4 "$one" "$two" >"$TMP/expected"
  [ "$rc" = 0 ] && cmp -s "$TMP/expected" "$calls" &&
    grep -Fq 'verdict: fixture (exit 0)' "$transcript" &&
    grep -Eq '^windows-leg: run directory: .+/this-run$' "$transcript" &&
    grep -Fxq "$(printf 'OCANNL_TOOL_VERDICT_EVALUATED\tfixture.exe\ta checked claim')" "$transcript" &&
    grep -Fxq "$(printf 'OCANNL_TOOL_VERDICT_SKIP\tbackend\tfixture.exe\ta skipped claim')" "$transcript" &&
    [ "$(tail -n 1 "$transcript")" = 'exit: 0' ]
  report $? 'one batch exports its receipt and claim records from the exact SHA' "$(cat "$transcript")"
fi

if harness_leg '3: checkout and cache failures stop before the runner'; then
  [ -d "$TMP/clone" ] || seed || exit 2
  invoke missing "$clone" 0123456789abcdef0123456789abcdef01234567 "$one"
  refusal 'could not fetch'; report $? 'a missing revision is refused before any fixture'
  cache_rc=7 invoke cache-red "$clone" "$sha" "$one"
  [ "$rc" = 7 ] && [ "$(cat "$calls")" = cache ] &&
    [ "$(tail -n 1 "$transcript")" = 'exit: 7' ]
  report $? 'a red cache fixture preserves its exit and stops the batch' "$(cat "$transcript")"
  printf 'test/support/verdict.ml text eol=crlf\n' >"$clone/.gitattributes"
  git -C "$clone" -c core.autocrlf=false add .gitattributes &&
    git -C "$clone" -c user.name=fixture -c user.email=fixture@example.test commit -qm crlf || exit 2
  crlf_sha=$(git -C "$clone" rev-parse HEAD) || exit 2
  invoke crlf "$clone" "$crlf_sha" "$one"
  refusal 'must be LF in the index and worktree'
  report $? 'an attribute forcing CRLF is refused before any fixture' "$(cat "$transcript")"
fi

if harness_leg '4: runner status, isolated receipt and tests without Verdict'; then
  [ -d "$TMP/clone" ] || seed || exit 2
  run_rc=65 record=65 invoke red "$clone" "$sha" "$one"
  [ "$rc" = 65 ] && grep -Fq 'verdict: fixture (exit 65)' "$transcript" &&
    grep -Eq '^windows-leg: run directory: .+/this-run$' "$transcript" &&
    grep -Fq 'a checked claim' "$transcript" &&
    [ "$(tail -n 1 "$transcript")" = 'exit: 65' ]
  report $? 'a red batch exports its own receipt and available claims' "$(cat "$transcript")"
  for bad_record in missing 65; do
    record=$bad_record invoke "bad-receipt-$bad_record" "$clone" "$sha" "$one"
    [ "$rc" = 2 ] && grep -Fq 'runner receipt missing or inconsistent' "$transcript" &&
      [ "$(tail -n 1 "$transcript")" = 'exit: 2' ]
    report $? "a green batch needs a matching receipt ($bad_record)" "$(cat "$transcript")"
  done
  claims=no leg_backend=cuda invoke no-claims "$clone" "$sha" "$one"
  [ "$rc" = 0 ] && grep -Fxq cuda "$calls" &&
    grep -Fq 'no Verdict action records were produced' "$transcript" &&
    [ "$(tail -n 1 "$transcript")" = 'exit: 0' ]
  report $? 'an explicit backend is forwarded and an alias can succeed without Verdict' "$(cat "$transcript")"
fi

finish

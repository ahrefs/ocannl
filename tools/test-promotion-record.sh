#!/usr/bin/env bash
# Recover promotions after another build, exercising the working-tree scripts
# on isolated real Dune projects. No Dune alias: these projects own their builds.
# Usage: tools/test-promotion-record.sh [--keep]
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
HERE=$(cd "$(dirname "$0")" && pwd)
harness_require dune perl git
harness_scratch test-promotion-record
REAL_DUNE=$(command -v dune)
SRC=$HERE/test-run.sh
export OCANNL_TOOL_FLEET_WORKER=none
export OCANNL_TOOL_DXG_DEVICE=$TMP/no-dxg OCANNL_TOOL_KFD_TOPOLOGY=$TMP/no-kfd \
  OCANNL_TOOL_NVIDIA_DEVICE=$TMP/no-nvidia
unset DUNE_DIFF_COMMAND DUNE_BUILD_DIR DUNE_ROOT

project() { # name runner promoter
  local repo=$TMP/$1
  mkdir -p "$repo/tools" "$repo/scripts" "$repo/.shim" || return 1
  cp "$2" "$repo/tools/test-run.sh" || return 1
  cp "$3" "$repo/tools/promote.sh" || return 1
  chmod +x "$repo/tools/test-run.sh" "$repo/tools/promote.sh" || return 1
  cp "$HERE/box-jobs.sh" "$HERE/batch-backends.sh" "$HERE/fleet-worker-candidates.sh" "$repo/tools/" || return 1
  cp "$HERE/../scripts/process-group.sh" "$repo/scripts/" || return 1
  printf '(lang dune 3.20)\n' >"$repo/dune-project"
  # Separate actions ensure both promotions are registered on a failure.
  cat >"$repo/dune" <<'DUNE'
(rule
 (alias runtest)
 (deps gen.txt)
 (action (progn (with-stdout-to foo.output (run cat gen.txt)) (diff foo.expected foo.output))))
(rule
 (alias runtest)
 (deps gen.txt)
 (action (progn (with-stdout-to bar.output (run cat gen.txt)) (diff bar.expected bar.output))))
DUNE
  printf '* -text\n' >"$repo/.gitattributes"
  printf 'original\n' >"$repo/foo.expected"
  printf 'original\n' >"$repo/bar.expected"
  printf 'corrected\r\n' >"$repo/gen.txt"
  (cd "$repo" && git init -q && git -c user.name=t -c user.email=t@t -c commit.gpgsign=false \
    add -A && git -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qm initial) || return 1
  # Force the floor's list stream while letting real Dune do every build.
  # The same public diff interface is used on 3.20, 3.21 and current Dune.
  cat >"$repo/.shim/dune" <<'SHIM'
#!/usr/bin/env bash
if [ "${1:-}" = --version ] && [ -n "${TEST_PROMOTION_VERSION:-}" ] && [ "$TEST_PROMOTION_VERSION" != current ]; then
  echo "$TEST_PROMOTION_VERSION"
  exit 0
fi
if [ "${1:-}" = promotion ] && [ "${2:-}" = list ]; then
  if [ "${TEST_PROMOTION_VERSION:-}" = 3.20.2 ] || [ "${TEST_PROMOTION_VERSION:-}" = 3.21.1 ]; then
    "$TEST_PROMOTION_DUNE" "$@" >&2
    exit $?
  fi
fi
if [ "${1:-}" = promotion ] && [ "${2:-}" = diff ]; then
  [ -z "${TEST_PROMOTION_CAPTURE_FAIL:-}" ] || exit 1
  [ -z "${TEST_PROMOTION_CAPTURE_PAUSE:-}" ] || exec perl -e 'sleep 60'
fi
exec "$TEST_PROMOTION_DUNE" "$@"
SHIM
  chmod +x "$repo/.shim/dune"
  printf '%s\n' "$repo"
}
record() { # repo; return Dune failure only if a finished failed record exists
  local rc=0
  (cd "$1" && tools/test-run.sh run --cap "${TEST_PROMOTION_CAP:-60}" build -j 2 @runtest) >"$1/record.log" 2>&1 || rc=$?
  [ "$rc" = 1 ] || return 1
  recorded=$("$1/tools/test-run.sh" paths run last) || return 1
  [ "$(cat "$recorded/exit")" = 1 ] || return 1
}
recover() { # repo runner promoter version; shipping oracle, reused for negative controls
  local name=$1 runner=$2 promoter=$3 version=$4 repo rc=0
  repo=$(project "$name" "$runner" "$promoter") || return 1
  export OCANNL_TOOL_TEST_RUNS=$TMP/$name-runs TEST_PROMOTION_DUNE=$REAL_DUNE TEST_PROMOTION_VERSION=$version
  export PATH=$repo/.shim:$ORIGINAL_PATH
  record "$repo" || return 1
  # Both controls matter: the later build removes Dune's list and payloads.
  (cd "$repo" && dune build -j 2 dune-project && dune clean) >"$repo/later.log" 2>&1 || return 1
  (cd "$repo" && tools/promote.sh --from-run "$recorded") >"$repo/recover.log" 2>&1 || rc=$?
  [ "$rc" = 0 ] || return 1
  printf 'corrected\n' >"$repo/want"
  cmp -s "$repo/want" "$repo/foo.expected" && cmp -s "$repo/want" "$repo/bar.expected" || return 1
  (cd "$repo" && tools/promote.sh --from-run "$recorded") >"$repo/replay.log" 2>&1 || return 1
  cmp -s "$repo/want" "$repo/foo.expected" && cmp -s "$repo/want" "$repo/bar.expected"
}
ORIGINAL_PATH=$PATH
for version in current 3.20.2 3.21.1; do
  recover "recovery-$version" "$HERE/test-run.sh" "$HERE/promote.sh" "$version"
  report $? "corrected bytes survive a later build and clean ($version)"
done
# The same shipping oracle must reject the former list-only recording behavior.
mutant_runner=$(mutant list-only '
  /# Corrected bytes belong to this record too/ { skipping=1 }
  skipping && /^}/ { skipping=0; print; next }
  !skipping { print }
')
if [ -n "$mutant_runner" ] && ! recover list-only "$mutant_runner" "$HERE/promote.sh" current \
  && grep -q 'no complete saved promotions' "$TMP/list-only/recover.log"; then
  report 0 'negative control: a list without corrected bytes cannot recover'
else
  report 1 'negative control: a list without corrected bytes cannot recover'
fi

repo=$(project guards "$HERE/test-run.sh" "$HERE/promote.sh") || exit 1
export OCANNL_TOOL_TEST_RUNS=$TMP/guards-runs TEST_PROMOTION_VERSION=current
export PATH=$repo/.shim:$ORIGINAL_PATH
if record "$repo"; then
  # No partial application when a later source edit conflicts with recovery.
  printf 'later edit\n' >"$repo/bar.expected"
  rc=0
  "$repo/tools/promote.sh" --from-run "$recorded" >"$repo/guard.log" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && grep -q 'source changed since run' "$repo/guard.log" \
    && [ "$(cat "$repo/foo.expected")" = original ]; then
    report 0 'changed source is refused before any saved file is applied'
  else report 1 'changed source is refused before any saved file is applied'; fi
  rc=0
  "$repo/tools/promote.sh" --from-run "$recorded" foo.expected >"$repo/select.log" 2>&1 || rc=$?
  if [ "$rc" = 0 ] && [ "$(cat "$repo/foo.expected")" = corrected ] && [ "$(cat "$repo/bar.expected")" = 'later edit' ]; then
    report 0 'an exact file selection recovers only the requested correction'
  else report 1 'an exact file selection recovers only the requested correction'; fi
  rc=0
  "$repo/tools/promote.sh" --from-run "$recorded" missing.expected >"$repo/missing.log" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && grep -q 'no saved promotion' "$repo/missing.log"; then
    report 0 'an unknown selection fails explicitly'
  else report 1 'an unknown selection fails explicitly'; fi
  cp "$recorded/wt" "$recorded/wt.original"
  printf '%s\n' "$TMP/elsewhere" >"$recorded/wt"
  rc=0
  "$repo/tools/promote.sh" --from-run "$recorded" >"$repo/worktree.log" 2>&1 || rc=$?
  if [ "$rc" = 2 ] && grep -q 'another worktree' "$repo/worktree.log"; then
    report 0 'a record from another worktree is refused'
  else report 1 'a record from another worktree is refused'; fi
else report 1 'guard fixtures recorded both promotions'; fi

repo=$(project failed-copy "$HERE/test-run.sh" "$HERE/promote.sh") || exit 1
export OCANNL_TOOL_TEST_RUNS=$TMP/failed-copy-runs TEST_PROMOTION_CAPTURE_FAIL=1
export PATH=$repo/.shim:$ORIGINAL_PATH
if record "$repo" && [ -s "$recorded/promotions" ] && [ ! -e "$recorded/promotion-files" ] \
  && [ ! -e "$recorded/promotion-files.tmp" ]; then
  report 0 'failed correction query preserves the list without advertising partial recovery'
else report 1 'failed correction query preserves the list without advertising partial recovery'; fi
unset TEST_PROMOTION_CAPTURE_FAIL
repo=$(project bounded-copy "$HERE/test-run.sh" "$HERE/promote.sh") || exit 1
export OCANNL_TOOL_TEST_RUNS=$TMP/bounded-copy-runs TEST_PROMOTION_CAPTURE_PAUSE=1 TEST_PROMOTION_CAP=8
export PATH=$repo/.shim:$ORIGINAL_PATH
if record "$repo" && [ -s "$recorded/promotions" ] && [ ! -e "$recorded/promotion-files" ] \
  && [ ! -e "$recorded/promotion-files.tmp" ] \
  && [ "$("$repo/tools/test-run.sh" lock-status "$recorded")" = idle ]; then
  report 0 'a stuck capture spends only the remaining query budget and preserves Dune verdict and lock release'
else report 1 'a stuck capture spends only the remaining query budget and preserves Dune verdict and lock release'; fi
unset TEST_PROMOTION_CAPTURE_PAUSE TEST_PROMOTION_CAP
# Byte controls catch a query that adds presentation newlines or confuses an
# empty correction with a missing payload. Declared targets exercise unstaged
# Dune corrections, independently of the intermediate-output scenarios above.
for mode in empty no-final-newline; do
  repo=$(project "bytes-$mode" "$HERE/test-run.sh" "$HERE/promote.sh") || exit 1
  export OCANNL_TOOL_TEST_RUNS=$TMP/bytes-$mode-runs
  export PATH=$repo/.shim:$ORIGINAL_PATH
  if [ "$mode" = empty ]; then : >"$repo/gen.txt"; else printf 'no final newline' >"$repo/gen.txt"; fi
  cat >"$repo/dune" <<'DUNE'
(rule (target foo.output) (deps gen.txt) (action (with-stdout-to foo.output (run cat gen.txt))))
(rule (alias runtest) (action (diff foo.expected foo.output)))
DUNE
  rc=0
  if record "$repo"; then
    (cd "$repo" && dune clean && tools/promote.sh --from-run "$recorded") >"$repo/recover.log" 2>&1 || rc=$?
    if [ "$rc" = 0 ] && cmp -s "$repo/gen.txt" "$repo/foo.expected"; then
      report 0 "exact bytes survive capture and replay ($mode, declared target)"
    else report 1 "exact bytes survive capture and replay ($mode, declared target)"; fi
  else report 1 "byte control records a promotable diff ($mode)"; fi
done

repo=$(project replay-merge "$HERE/test-run.sh" "$HERE/promote.sh") || exit 1
export OCANNL_TOOL_TEST_RUNS=$TMP/replay-merge-runs
export PATH=$repo/.shim:$ORIGINAL_PATH
if record "$repo"; then
  rc=0
  (
    cd "$repo" || exit 1
    git checkout -qb side || exit 1
    git -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit --allow-empty -qm side || exit 1
    git checkout -qb current HEAD^ || exit 1
    git -c user.name=t -c user.email=t@t -c commit.gpgsign=false merge --no-commit --no-ff side || exit 1
    dune clean || exit 1
    tools/promote.sh --from-run "$recorded" || exit 1
    git show :foo.expected >index-foo || exit 1
    git show :bar.expected >index-bar || exit 1
    printf 'corrected\n' >want
    cmp -s want index-foo && cmp -s want index-bar
  ) >"$repo/merge.log" 2>&1 || rc=$?
  report "$rc" 'saved promotion during a real merge reaches the index with LF bytes'
else report 1 'merge replay records both corrections'; fi
export PATH=$ORIGINAL_PATH
finish

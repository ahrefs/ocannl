#!/usr/bin/env bash
# Hermetic loopback tests for the working-tree machine-verify.sh, its
# verification procedure machine-verify-far.sh and the remote-verify.sh shim.
# Usage: tools/test-machine-verify.sh [--keep] [--help]
# Real Git repositories exercise fetch, detached worktrees and source status;
# fake SSH, opam and Dune require no network, compiler or backend hardware.
# BOX placement is driven through the fake `ssh -G`: 192.0.2.1 (TEST-NET-1,
# never assigned) is elsewhere, 127.0.0.1 is this machine.
# Run directly (Ubuntu CI does), never from a Dune action.
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require git perl bash
harness_scratch test-machine-verify
TMP=$(cd "$TMP" && pwd -P)
HERE=$(cd "$(dirname "$0")" && pwd)
SRC=$HERE/machine-verify.sh
FAR=$HERE/machine-verify-far.sh
SHIM=$HERE/remote-verify.sh
REAL_GIT=$(command -v git)
for f in "$SRC" "$FAR" "$SHIM"; do printf 'testing %s (cksum %s)\n' "$f" "$(cksum <"$f")"; done
# Isolate Git configuration, hooks, identity and URL rewriting. Every Git
# mutation is confined to this newly allocated scratch directory.
mkdir -p "$TMP/bin" "$TMP/home" "$TMP/runs"
export GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=$TMP/gitconfig
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES
unset GIT_CONFIG_COUNT GIT_CONFIG_PARAMETERS
export GIT_AUTHOR_NAME=Fixture GIT_AUTHOR_EMAIL=fixture@example.invalid
export GIT_COMMITTER_NAME=$GIT_AUTHOR_NAME GIT_COMMITTER_EMAIL=$GIT_AUTHOR_EMAIL
"$REAL_GIT" init -q "$TMP/seed" || exit 2
printf '_build/\nocannl_config\n' >"$TMP/seed/.gitignore"
printf 'original\n' >"$TMP/seed/value.expected"
printf 'source\n' >"$TMP/seed/source.ml"
"$REAL_GIT" -C "$TMP/seed" add . && "$REAL_GIT" -C "$TMP/seed" commit -qm seed || exit 2
SHA=$("$REAL_GIT" -C "$TMP/seed" rev-parse HEAD)
"$REAL_GIT" -C "$TMP/seed" branch fixture "$SHA" || exit 2
"$REAL_GIT" clone -q --bare "$TMP/seed" "$TMP/pushed.git" || exit 2
touch "$GIT_CONFIG_GLOBAL"
cat >"$TMP/bin/ssh" <<'SH'
#!/usr/bin/env bash
if [ "$1" = -G ]; then
  # Configuration lookup only: what the alias would connect to, no connection.
  printf 'ssh -G %s\n' "$2" >>"$SSH_LOG"
  printf 'user fixture\nhostname %s\nport %s\n' "$ENDPOINT" "$ENDPOINT_PORT"
  [ -z "$ENDPOINT_PROXY" ] || printf 'proxyjump %s\n' "$ENDPOINT_PROXY"
  exit 0
fi
printf 'ssh trip\n' >>"$SSH_LOG"
case $MODE in
  transport) echo 'fixture: transport refused' >&2; exit 255 ;;
  ssh-timeout) echo 'fixture: transport stalled' >&2; exec perl -e 'sleep 4' ;;
esac
# A real SSH session starts from the remote login environment, not the caller's.
unset OPAMSWITCH DUNE_BUILD_DIR
# SSH concatenates its command operands; retain the shipped quoting and stdin.
while [ "$#" -gt 1 ]; do shift; done
exec /bin/sh -c "$1"
SH
cat >"$TMP/bin/opam" <<'SH'
#!/usr/bin/env bash
if [ "$1" = switch ]; then
  [ "$PWD" = "$FIXTURE_REPO" ] || exit 91
  # Like real opam, a caller's OPAMSWITCH outranks the checkout's selection.
  printf '%s\n' "${OPAMSWITCH:-fixture-switch}"; exit 0
fi
[ "$1" = exec ] && [ "$2" = --switch=fixture-switch ] && [ "$3" = -- ] || {
  echo "fixture: wrong opam switch: $2" >&2; exit 92;
}
shift 3
export OCANNL_PROFILE=switch-secret OCANNL_BACKEND=hip
exec "$@"
SH
cat >"$TMP/bin/dune" <<'SH'
#!/usr/bin/env bash
# This is an independent observation of the environment and actual checkout
# received by the fake compiler, not a restatement of verifier log messages.
[ -z "${OCANNL_PROFILE:-}" ] && [ -z "${OCANNL_PRINT_DECIMALS_PRECISION:-}" ] || {
  echo 'fixture: configuration leaked' >&2; exit 93;
}
[ -z "${DUNE_BUILD_DIR:-}" ] || { echo 'fixture: caller environment leaked' >&2; exit 89; }
if [ "$1" = build ]; then wanted_backend=$WANT_BACKEND; else wanted_backend=; fi
[ "${OCANNL_BACKEND:-}" = "$wanted_backend" ] || { echo 'fixture: backend not isolated' >&2; exit 94; }
[ "$PWD" != "$FIXTURE_REPO" ] && [ "$(git rev-parse HEAD)" = "$FIXTURE_SHA" ] || exit 95
if git symbolic-ref -q HEAD >/dev/null; then exit 96; fi
[ -f ocannl_config ] && [ ! -s ocannl_config ] || { echo 'fixture: boundary missing' >&2; exit 97; }
printf '%s|%s|%s\n' "$PWD" "${OCANNL_BACKEND:-none}" "$*" >>"$AUDIT"
mkdir -p _build/default/test/config
if [ "$1" = build ]; then
  for arg in "$@"; do
    case $arg in
      @check)
        case $MODE in
          build-fail) echo 'fixture: compiler failed' >&2; exit 37 ;;
          timeout | trip-timeout) echo 'fixture: compiler stalled' >&2; exec perl -e 'sleep 4' ;;
          source-change) printf 'changed\n' >>source.ml ;;
        esac ;;
      test/config/ocannl_backend.txt)
        if [ "$MODE" = backend-mismatch ]; then echo hip; else echo "$WANT_BACKEND"; fi >_build/default/test/config/ocannl_backend.txt ;;
      @golden)
        if [ ! -f _build/applied ]; then touch _build/pending; exit 1; fi
        [ "$MODE" != rerun-fail ] || { echo 'fixture: unrelated rerun failure' >&2; exit 38; } ;;
    esac
  done
elif [ "$1" = promotion ]; then
  case $2 in
    list) if [ -f _build/pending ]; then
      if [ "$MODE" = nongolden ]; then echo source.ml; else echo value.expected; fi
      fi ;;
    show) printf 'corrected\n' ;;
    apply)
      printf 'corrected\n' >value.expected
      [ "$MODE" != golden-source-change ] || printf 'changed\n' >>source.ml
      rm -f _build/pending; touch _build/applied ;;
    *) exit 98 ;;
  esac
else exit 99
fi
SH
chmod +x "$TMP/bin/ssh" "$TMP/bin/opam" "$TMP/bin/dune"
# A private Git wrapper is only for deterministic transport/cleanup failures;
# every other operation is the real executable, including source assertions.
cat >"$TMP/bin/git" <<'SH'
#!/usr/bin/env bash
fetching=0
args=()
for arg in "$@"; do
  [ "$arg" != fetch ] || fetching=1
  if [ "$fetching" = 1 ] && [ "$arg" = origin ]; then arg=$FIXTURE_PUSHED; fi
  args+=("$arg")
  if [ "$MODE" = fetch-fail ] && [ "$arg" = fetch ]; then exit 44; fi
  if [ "$MODE" = cleanup-fail ] && [ "$arg" = remove ]; then exit 45; fi
done
exec "$REAL_GIT" "${args[@]}"
SH
chmod +x "$TMP/bin/git"
# The fakes read their controls from a file rather than the environment: the
# local transport clears the environment exactly as an SSH session starts
# without the caller's, so only a file reaches a fake on both transports.
for fake in ssh opam dune git; do
  { sed -n 1p "$TMP/bin/$fake"; printf '. %q\n' "$TMP/fixture.env"; sed 1d "$TMP/bin/$fake"; } >"$TMP/fake" &&
    cat "$TMP/fake" >"$TMP/bin/$fake" || exit 2
done
rm -f "$TMP/fake"

# Placement of the fixture BOX, read by the fake `ssh -G`, and the transport a
# case expects. A case overrides them with a prefix assignment on its call.
ENDPOINT=192.0.2.1 ENDPOINT_PORT=22 ENDPOINT_PROXY= TRANSPORT=ssh BACKEND=cc

run_case() { # SUBJECT NAME MODE [verifier args]
  local subject=$1 name=$2 mode=$3
  shift 3
  local run=$TMP/runs/$name rc=0
  mkdir -p "$run/worktrees"
  : >"$run/ssh-log"
  "$REAL_GIT" clone -q "$TMP/pushed.git" "$run/repo" || return 1
  "$REAL_GIT" -C "$run/repo" remote set-url origin https://github.com/lukstafi/ocannl-staging.git || return 1
  printf 'do not touch\n' >"$run/repo/untouched"
  printf 'fetch head sentinel\n' >"$run/repo/.git/FETCH_HEAD"
  # A poisonous parent config must be blocked by the empty root boundary.
  printf 'backend=hip\n' >"$run/ocannl_config"
  case $mode in
    wrong-remote) "$REAL_GIT" -C "$run/repo" remote set-url origin https://example.invalid/wrong.git ;;
    nested) touch "$run/dune-project" ;;
  esac
  local var
  for var in GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$GIT_CONFIG_GLOBAL" REAL_GIT="$REAL_GIT" \
    FIXTURE_PUSHED="$TMP/pushed.git" MODE="$mode" FIXTURE_REPO="$run/repo" FIXTURE_SHA="$SHA" \
    AUDIT="$run/audit" SSH_LOG="$run/ssh-log" ENDPOINT="$ENDPOINT" ENDPOINT_PORT="$ENDPOINT_PORT" \
    ENDPOINT_PROXY="$ENDPOINT_PROXY" WANT_BACKEND="$BACKEND"; do
    printf 'export %s=%q\n' "${var%%=*}" "${var#*=}"
  done >"$TMP/fixture.env"
  # OPAMSWITCH and DUNE_BUILD_DIR stand for the caller's session: an SSH
  # session never sees them, and the local transport must not either.
  env -i PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$TMP/home" \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$GIT_CONFIG_GLOBAL" \
    OCANNL_PRINT_DECIMALS_PRECISION=ambient-secret OCANNL_BACKEND=cuda \
    OPAMSWITCH=caller-switch DUNE_BUILD_DIR=caller-build \
    bash "$subject" loopback fixture --backend "$BACKEND" --repo "$run/repo" \
    --worktree-root "$run/worktrees" --cap 10 --trip-cap 30 "$@" >"$run/stdout" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$run/rc"
  # Prove ownership: the original checkout, its dirty file and FETCH_HEAD stay
  # intact. Inspect actual Git registration and filesystem after every outcome.
  [ "$(cat "$run/repo/untouched")" = 'do not touch' ] &&
    [ "$(cat "$run/repo/.git/FETCH_HEAD")" = 'fetch head sentinel' ] &&
    [ "$("$REAL_GIT" -C "$run/repo" rev-parse HEAD)" = "$SHA" ] || return 1
  if [ "$mode" = cleanup-fail ]; then
    local owned
    owned=$(sed -n 's/^worktree:      //p' "$run/stdout")
    case $owned in "$run/worktrees/machine-verify."*) ;; *) return 1 ;; esac
    [ -d "$owned" ] || return 1
    "$REAL_GIT" -C "$run/repo" worktree remove --force "$owned" || return 1
  fi
  [ -z "$(ls -A "$run/worktrees")" ] &&
    [ "$("$REAL_GIT" -C "$run/repo" worktree list --porcelain | grep -c '^worktree ')" = 1 ]
}
# Which transport carried the trip, observed at the fake ssh rather than read
# from the verifier's own report of it: no SSH session means none was used.
used_transport() { # RUN
  if grep -q '^ssh trip$' "$TMP/runs/$1/ssh-log"; then echo ssh; else echo local; fi
}
check_case() { # NAME MODE RC DIAGNOSTIC [args]
  local name=$1 mode=$2 expected=$3 pattern=$4 ok=0
  shift 4
  run_case "$SRC" "$name" "$mode" "$@" || ok=1
  [ "$(cat "$TMP/runs/$name/rc")" = "$expected" ] || ok=1
  grep -qE "$pattern" "$TMP/runs/$name/stdout" || ok=1
  [ "$(used_transport "$name")" = "$TRANSPORT" ] || ok=1
  grep -qx "machine-verify: $TRANSPORT exit: $expected" "$TMP/runs/$name/stdout" || ok=1
  case $mode in
    transport | ssh-timeout | trip-timeout) ;;
    *) grep -qx "machine-verify: exit: $expected" "$TMP/runs/$name/stdout" || ok=1 ;;
  esac
  if [ "$expected" != 0 ] && [ "$mode" != cleanup-fail ] && grep -q '^machine-verify: verified ' "$TMP/runs/$name/stdout"; then ok=1; fi
  report "$ok" "$name: verdict, reason, transport and checkout cleanup" "$TMP/runs/$name"
}
# A placement refusal happens before any trip: no session, no build, no sentinel.
check_refusal() { # NAME DIAGNOSTIC [args]
  local name=$1 pattern=$2 ok=0
  shift 2
  run_case "$SRC" "$name" success "$@" || ok=1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] || ok=1
  grep -qE "$pattern" "$TMP/runs/$name/stdout" || ok=1
  ! grep -q '^ssh trip$' "$TMP/runs/$name/ssh-log" || ok=1
  [ ! -e "$TMP/runs/$name/audit" ] || ok=1
  if grep -qE '^machine-verify: (verified |exit: |(ssh|local) exit: )' "$TMP/runs/$name/stdout"; then ok=1; fi
  report "$ok" "$name: refused before any trip" "$TMP/runs/$name"
}
check_case success success 0 "verified .*commit=$SHA backend=cc" --test @fixture \
  --run 'test -z "$(cat)" && printf "probe quote: '\'' preserved\n"'
if grep -q "resolved commit: $SHA" "$TMP/runs/success/stdout" &&
  grep -q 'fixture-switch' "$TMP/runs/success/stdout" &&
  grep -q 'probe quote:' "$TMP/runs/success/stdout" &&
  grep -q '|cc|build -j 4 @fixture' "$TMP/runs/success/audit" &&
  grep -qx 'ssh -G loopback' "$TMP/runs/success/ssh-log" &&
  ! grep -qE 'ambient-secret|switch-secret' "$TMP/runs/success/stdout"; then
  report 0 'success: observed source, switch, pinned environment and stdin isolation'
else report 1 'success: observed source, switch, pinned environment and stdin isolation'; fi
check_case build-failure build-fail 37 'fixture: compiler failed'
check_case command-timeout timeout 142 '^machine-verify: exit: 142$' --cap 1
check_case transport-failure transport 255 'fixture: transport refused'
check_case transport-timeout ssh-timeout 142 '^machine-verify: ssh exit: 142$' --trip-cap 1
check_case former-cap-name ssh-timeout 142 '^machine-verify: ssh exit: 142$' --ssh-cap 1
check_case wrong-remote wrong-remote 2 'no remote .* points to lukstafi/ocannl-staging'
check_case nested-root nested 2 'nested under Dune root'
check_case failed-fetch fetch-fail 2 'cannot fetch pushed branch'
check_case backend-provenance backend-mismatch 2 'requested backend cc resolved as hip'
check_case source-mutation source-change 2 'source changed after @check'
check_case cleanup-failure cleanup-fail 125 'cleanup: FAIL'
check_case golden-success golden 0 'golden: RECORDED and re-run PASS' --record-golden @golden
if grep -q '^+corrected$' "$TMP/runs/golden-success/stdout" &&
  grep -q 'source assertion: PASS (after restoring recorded golden' "$TMP/runs/golden-success/stdout"; then
  report 0 'golden: apply-ready patch and restored source'
else report 1 'golden: apply-ready patch and restored source'; fi
check_case nongolden-refusal nongolden 2 'produced a non-golden correction: source.ml' --record-golden @golden
check_case golden-mutation golden-source-change 2 'non-golden source change during golden recording:  M source.ml' --record-golden @golden
check_case golden-rerun-failure rerun-fail 38 'fixture: unrelated rerun failure' --record-golden @golden

# The local transport: BOX's endpoint is an address of this machine, so the
# same procedure runs here with no SSH session, from a cleared environment
# (the caller's OPAMSWITCH/DUNE_BUILD_DIR would otherwise fail the fakes).
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-success success 0 \
  "verified .*commit=$SHA backend=cc" --test @fixture
if grep -qx 'machine-verify: transport: local, no ssh (loopback -> 127.0.0.1:22; on this machine: 127.0.0.1)' \
  "$TMP/runs/local-success/stdout" &&
  grep -q '^transport:     local, no ssh' "$TMP/runs/local-success/stdout" &&
  grep -q '|cc|build -j 4 @fixture' "$TMP/runs/local-success/audit"; then
  report 0 'local: placement evidence reaches the provenance block'
else report 1 'local: placement evidence reaches the provenance block' "$TMP/runs/local-success"; fi
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-forced success 0 "verified .*backend=cc" --local
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-build-failure build-fail 37 'fixture: compiler failed'
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-source-mutation source-change 2 'source changed after @check'
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-golden golden 0 'golden: RECORDED and re-run PASS' \
  --record-golden @golden
# The whole-trip cap bounds the local trip too, and the procedure still
# removes its worktree when the supervisor signals it (run_case checks).
ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-trip-timeout trip-timeout 142 \
  '^machine-verify: local exit: 142$' --trip-cap 1
# --ssh overrides a local endpoint.
ENDPOINT=127.0.0.1 check_case ssh-forced success 0 "verified .*backend=cc" --ssh
# A proxied endpoint is resolved from the proxy, so it is never this machine.
ENDPOINT=127.0.0.1 ENDPOINT_PROXY=jump.example.invalid check_case proxied success 0 "verified .*backend=cc"
check_refusal local-mismatch '^machine-verify: --local refused: BOX loopback is not this machine .*elsewhere: 192\.0\.2\.1$' --local
ENDPOINT=127.0.0.1 ENDPOINT_PORT=2222 check_refusal forwarded-port \
  '^machine-verify: ambiguous placement: .*127\.0\.0\.1:2222.* pass --local or --ssh$'
ENDPOINT=127.0.0.1 ENDPOINT_PROXY=jump.example.invalid check_refusal local-proxied \
  '^machine-verify: --local refused: .* via proxy jump\.example\.invalid$' --local
check_refusal exclusive-flags '^machine-verify: --local and --ssh are mutually exclusive$' --local --ssh

# The deprecated name forwards every argument and says so on stderr.
shim_case() { # NAME
  local ok=0
  run_case "$SHIM" "$1" success --test @fixture || ok=1
  [ "$(cat "$TMP/runs/$1/rc")" = 0 ] || ok=1
  grep -qx 'remote-verify.sh: deprecated name; forwarding to tools/machine-verify.sh' "$TMP/runs/$1/stdout" || ok=1
  grep -q "^machine-verify: verified .*commit=$SHA backend=cc" "$TMP/runs/$1/stdout" || ok=1
  grep -q '|cc|build -j 4 @fixture' "$TMP/runs/$1/audit" || ok=1
  report "$ok" "$1: forwards to machine-verify.sh with a deprecation line" "$TMP/runs/$1"
}
shim_case shim-ssh
ENDPOINT=127.0.0.1 shim_case shim-local
[ "$(used_transport shim-local)" = local ] && [ "$(used_transport shim-ssh)" = ssh ]
report $? 'shim: placement is decided by machine-verify.sh, not by the old name'

# Negative controls run the same shipping oracle with one guard disabled. They
# must reach certification (or the leak the guard exists for) erroneously,
# rather than merely fail to parse/start. The pair is copied so the driver finds
# its procedure beside it, with exactly one of the two files mutated.
mutant_pair() { # NAME driver|far AWK_PROGRAM -> path of the pair's driver
  local dir=$TMP/mutants/$1 target
  case $2 in driver) target=$SRC ;; far) target=$FAR ;; *) return 1 ;; esac
  mkdir -p "$dir" && cp "$SRC" "$FAR" "$dir/" || return 1
  awk "$3" "$target" >"$dir/${target##*/}" || return 1
  bash -n "$dir/${target##*/}" || return 1
  ! cmp -s "$target" "$dir/${target##*/}" || return 1
  printf '%s' "$dir/machine-verify.sh"
}
mutation_oracle() {
  local subject=$1 name=$2
  run_case "$subject" "$name" source-change || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q 'source changed after @check' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-source-assertion far '/^assert_source_state\(\) \{/ { print; print "  return 0"; next } { print }') || exit 2
expect_rejected 'source assertion removed' "$mutated" mutation_oracle '^machine-verify: verified '
golden_mutation_oracle() {
  local subject=$1 name=$2
  run_case "$subject" "$name" golden-source-change --record-golden @golden || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q 'non-golden source change during golden recording' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-golden-scope far '/^assert_only_promoted_goldens\(\) \{/ { print; print "  return 0"; next } { print }') || exit 2
expect_rejected 'golden scope removed' "$mutated" golden_mutation_oracle '^machine-verify: verified '
# Every address counted as this machine's: --local then runs on the wrong box.
mismatch_oracle() {
  local subject=$1 name=$2
  run_case "$subject" "$name" success --local || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q '^machine-verify: --local refused' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-address-check driver '/print "\$ip ", \(bind/ { print "    print \"$ip here\\n\";"; next } { print }') || exit 2
expect_rejected 'local-address check removed' "$mutated" mismatch_oracle '^machine-verify: verified '
# The caller's environment passed through: its opam switch outranks the
# checkout's, which an SSH session would never have seen.
leak_oracle() {
  local subject=$1 name=$2
  ENDPOINT=127.0.0.1 run_case "$subject" "$name" success || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 0 ] &&
    grep -q '^machine-verify: verified ' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-env-clear driver '/^  local_env=\(env -i\)$/ { print "  local_env=(env)"; next } { print }') || exit 2
expect_rejected 'local environment clearing removed' "$mutated" leak_oracle 'fixture: wrong opam switch: --switch=caller-switch'
finish

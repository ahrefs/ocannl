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
REAL_GIT=$(command -v git)
for f in machine-verify.sh machine-verify-far.sh remote-verify.sh box-jobs.sh batch-backends.sh; do
  printf 'testing %s (cksum %s)\n' "$HERE/$f" "$(cksum <"$HERE/$f")"
done
# The subjects run from a scratch copy of tools/: the scripts unchanged, and
# the width table they pass to the far side with one line appended, which
# points its device probes and the backend readers (the OCANNL_TOOL_* overrides
# of tools/box-jobs.sh and tools/batch-backends.sh) at the case's fixture file.
# The far side clears every ambient OCANNL_* variable, as it must, so only a
# file can fake a box's GPU; every width decision is still those files' own.
mkdir -p "$TMP/tools" || exit 2
cp "$HERE/machine-verify.sh" "$HERE/machine-verify-far.sh" "$HERE/remote-verify.sh" \
  "$HERE/box-jobs.sh" "$HERE/batch-backends.sh" "$TMP/tools/" || exit 2
printf '. %q\n' "$TMP/fixture.env" >>"$TMP/tools/box-jobs.sh" || exit 2
SRC=$TMP/tools/machine-verify.sh
FAR=$TMP/tools/machine-verify-far.sh
SHIM=$TMP/tools/remote-verify.sh
BOX_JOBS=$TMP/tools/box-jobs.sh
BATCH_BACKENDS=$TMP/tools/batch-backends.sh
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
# The directories whose configuration tools/batch-backends.sh reads, as the
# real tree carries them.
mkdir -p "$TMP/seed/test/config" "$TMP/seed/arrayjit/test" &&
  : >"$TMP/seed/test/config/dune" && : >"$TMP/seed/arrayjit/test/dune" || exit 2
"$REAL_GIT" -C "$TMP/seed" add . && "$REAL_GIT" -C "$TMP/seed" commit -qm seed || exit 2
SHA=$("$REAL_GIT" -C "$TMP/seed" rev-parse HEAD)
"$REAL_GIT" -C "$TMP/seed" branch fixture "$SHA" || exit 2
"$REAL_GIT" clone -q --bare "$TMP/seed" "$TMP/pushed.git" || exit 2
touch "$GIT_CONFIG_GLOBAL"
# A commit no branch of the remote contains, for the commit form of BRANCH: the
# remote holds its object unreachable, and the `unpushed` mode gives the
# checkout a local branch at it -- neither may certify it.
UNPUSHED=$("$REAL_GIT" -C "$TMP/seed" commit-tree -p "$SHA" -m unpushed "$SHA^{tree}") &&
  "$REAL_GIT" -C "$TMP/seed" branch unpushed "$UNPUSHED" &&
  "$REAL_GIT" -C "$TMP/pushed.git" fetch -q "$TMP/seed" unpushed:refs/dangling &&
  "$REAL_GIT" -C "$TMP/pushed.git" update-ref -d refs/dangling || exit 2
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
          lib-*)
            # What dune leaves for an optional backend: its objects, and the
            # `select` result naming the arm it copied (the stub, when absent).
            case $WANT_BACKEND in cuda) arm=cudajit ;; hip) arm=hipjit ;; *) arm=$WANT_BACKEND ;; esac
            [ "$MODE" != lib-stub ] || arm=missing
            lib=_build/default/arrayjit/lib
            mkdir -p "$lib/.${WANT_BACKEND}_backend.objs/byte"
            [ "$MODE" = lib-absent ] || touch "$lib/.${WANT_BACKEND}_backend.objs/byte/${WANT_BACKEND}_backend.cmi"
            printf '# 1 "arrayjit/lib/%s_backend_impl.%s.ml"\n' "$WANT_BACKEND" "$arm" >"$lib/${WANT_BACKEND}_backend_impl.ml"
            if [ "$MODE" = lib-other ]; then
              mkdir -p "$lib/.hip_backend.objs/byte" && touch "$lib/.hip_backend.objs/byte/hip_backend.cmi"
            fi ;;
        esac ;;
      bin/device_props.exe)
        # The backend's own capability readback, as each mode's backend would
        # print it: the backend is the one it observes pinned, and a HIP device
        # carries the per-device eligibility the tile-MMA check reads.
        mkdir -p _build/default/bin
        eligible=true mma=advertised
        case $MODE in
          lib-mma-none | lib-no-rocwmma) mma=none ;;
          lib-ineligible) eligible=false mma=none ;;
          lib-no-eligibility) eligible= ;;
          lib-mma-unparsed) mma=garbled ;;
        esac
        { printf '#!/usr/bin/env bash\n'
          printf '[ "${OCANNL_BACKEND:-}" = %q ] || { echo "fixture: probe backend not pinned" >&2; exit 94; }\n' "$WANT_BACKEND"
          printf 'echo "backend = $OCANNL_BACKEND"\n'
          [ "$WANT_BACKEND" != hip ] || [ -z "$eligible" ] ||
            printf 'echo "static.device[0].tile_mma_eligible = %s"\n' "$eligible"
          [ "$mma" != advertised ] || printf 'echo "limits.mma.mma_tile = 16 16 16"\n'
          [ "$mma" != none ] || printf 'echo "limits.mma = ()"\n'
          [ "$mma" != garbled ] || printf 'echo "limits.mma.mma_tile: 16 16 16"\n'
        } >_build/default/bin/device_props.exe
        chmod +x _build/default/bin/device_props.exe ;;
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
# The ROCm tool's own answer to "where is HIP", which the tile-MMA check reads
# its header expectation from; the trees below stand for a complete rocWMMA
# install and the distro's umbrella-only one (gh-ocannl-1032).
cat >"$TMP/bin/hipconfig" <<'SH'
#!/usr/bin/env bash
[ "$1" = --path ] || exit 98
printf '%s\n' "$FIXTURE_HIP_ROOT"
SH
mkdir -p "$TMP/hip-complete/include/rocwmma/internal" "$TMP/hip-partial/include/rocwmma"
touch "$TMP/hip-complete/include/rocwmma/rocwmma.hpp" "$TMP/hip-complete/include/rocwmma/internal/types.hpp" \
  "$TMP/hip-partial/include/rocwmma/rocwmma.hpp"
chmod +x "$TMP/bin/ssh" "$TMP/bin/opam" "$TMP/bin/dune" "$TMP/bin/hipconfig"
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
  # What a verifier killed past its cleanup leaves for a later one that draws
  # the same PID: its heads namespace, holding a branch since deleted. Planted
  # at the first read of the namespace, before anything could have emptied it.
  case $MODE:$arg in
    stale-heads:refs/machine-verify/heads-*/)
      [ -e "$SSH_LOG.stale" ] || {
        : >"$SSH_LOG.stale"
        "$REAL_GIT" -C "$FIXTURE_REPO" update-ref "${arg}deleted-branch" "$FIXTURE_UNPUSHED" || exit 46
      } ;;
  esac
done
exec "$REAL_GIT" "${args[@]}"
SH
chmod +x "$TMP/bin/git"
# The fakes read their controls from a file rather than the environment: the
# local transport clears the environment exactly as an SSH session starts
# without the caller's, so only a file reaches a fake on both transports.
for fake in ssh opam dune git hipconfig; do
  { sed -n 1p "$TMP/bin/$fake"; printf '. %q\n' "$TMP/fixture.env"; sed 1d "$TMP/bin/$fake"; } >"$TMP/fake" &&
    cat "$TMP/fake" >"$TMP/bin/$fake" || exit 2
done
rm -f "$TMP/fake"

# The fixture BOX's devices, as tools/box-jobs.sh probes them: a WSL2 bridge
# node, a KFD topology (one GPU node whose SDMA pool is minix's 1 x 6 or tuf's
# 2 x 6) and a native NVIDIA control node. `absent` names nothing.
touch "$TMP/device-present" || exit 2
for pool in small:1 wide:2; do
  mkdir -p "$TMP/kfd-${pool%%:*}/1" &&
    printf 'simd_count 32\nnum_sdma_engines %s\nnum_sdma_queues_per_engine 6\n' "${pool#*:}" \
      >"$TMP/kfd-${pool%%:*}/1/properties" || exit 2
done

# Stand-ins for the backend readers tools/batch-backends.sh builds: the
# configuration resolves the pinned backend, and the reachability tool names
# the backends of FIXTURE_NAMES (`<backend>:<why>;...`), or fails.
cat >"$TMP/bin/fake-read-config" <<'SH'
#!/usr/bin/env bash
printf '%s\n' "${OCANNL_BACKEND:-}"
SH
cat >"$TMP/bin/fake-slot-kind" <<'SH'
#!/usr/bin/env bash
[ "$FIXTURE_NAMES" != fail ] || exit 3
IFS=';' read -r -a names <<<"$FIXTURE_NAMES"
for n in "${names[@]}"; do [ -z "$n" ] || printf 'names %s: %s\n' "${n%%:*}" "${n#*:}"; done
echo end
SH
chmod +x "$TMP/bin/fake-read-config" "$TMP/bin/fake-slot-kind" || exit 2

# Placement of the fixture BOX, read by the fake `ssh -G`, and the transport a
# case expects; the devices BOX shows and its fleet name. A case overrides them
# with a prefix assignment on its call.
ENDPOINT=192.0.2.1 ENDPOINT_PORT=22 ENDPOINT_PROXY= TRANSPORT=ssh BACKEND=cc HIP_TREE=hip-complete
DXG=absent KFD=absent NVIDIA=absent FLEET_BOX=fixture-box READERS=stand-in NAMES=
# The BRANCH operand, and the commit the fake dune requires the worktree at.
REF=fixture WANT_SHA=$SHA

fixture_device() { # absent|present|small|wide -> the path a probe reads
  case $1 in
    absent) printf '%s' "$TMP/no-such-device" ;;
    present) printf '%s' "$TMP/device-present" ;;
    *) printf '%s' "$TMP/kfd-$1" ;;
  esac
}
fixture_reader() { # stand-in name -> its path, or nothing: batch-backends.sh builds the real one
  [ "$READERS" != stand-in ] || printf '%s' "$TMP/bin/$1"
}
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
  # Remote-tracking refs are the checkout's, and only the named branch's may
  # move: with origin/master gone, a verifier fetching every head into them
  # would bring it back.
  "$REAL_GIT" -C "$run/repo" update-ref -d refs/remotes/origin/HEAD &&
    "$REAL_GIT" -C "$run/repo" update-ref -d refs/remotes/origin/master || return 1
  local tracking
  tracking=$("$REAL_GIT" -C "$run/repo" for-each-ref refs/remotes/)
  # A poisonous parent config must be blocked by the empty root boundary.
  printf 'backend=hip\n' >"$run/ocannl_config"
  case $mode in
    wrong-remote) "$REAL_GIT" -C "$run/repo" remote set-url origin https://example.invalid/wrong.git ;;
    nested) touch "$run/dune-project" ;;
    unpushed | stale-heads) "$REAL_GIT" -C "$run/repo" fetch -q --no-write-fetch-head "$TMP/seed" unpushed:refs/heads/unpushed ;;
  esac
  local var
  for var in GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$GIT_CONFIG_GLOBAL" REAL_GIT="$REAL_GIT" \
    FIXTURE_PUSHED="$TMP/pushed.git" MODE="$mode" FIXTURE_REPO="$run/repo" FIXTURE_SHA="$WANT_SHA" \
    AUDIT="$run/audit" SSH_LOG="$run/ssh-log" ENDPOINT="$ENDPOINT" ENDPOINT_PORT="$ENDPOINT_PORT" \
    ENDPOINT_PROXY="$ENDPOINT_PROXY" WANT_BACKEND="$BACKEND" FIXTURE_HIP_ROOT="$TMP/$HIP_TREE" \
    FIXTURE_UNPUSHED="$UNPUSHED" OCANNL_TOOL_DXG_DEVICE="$(fixture_device "$DXG")" \
    OCANNL_TOOL_KFD_TOPOLOGY="$(fixture_device "$KFD")" \
    OCANNL_TOOL_NVIDIA_DEVICE="$(fixture_device "$NVIDIA")" FLEET_LOCAL_BOX="$FLEET_BOX" \
    OCANNL_TOOL_READ_CONFIG="$(fixture_reader fake-read-config)" \
    OCANNL_TOOL_SLOT_KIND="$(fixture_reader fake-slot-kind)" FIXTURE_NAMES="$NAMES"; do
    printf 'export %s=%q\n' "${var%%=*}" "${var#*=}"
  done >"$TMP/fixture.env"
  # OPAMSWITCH and DUNE_BUILD_DIR stand for the caller's session: an SSH
  # session never sees them, and the local transport must not either.
  env -i PATH="$TMP/bin:/usr/bin:/bin:/usr/sbin:/sbin" HOME="$TMP/home" \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL="$GIT_CONFIG_GLOBAL" \
    OCANNL_PRINT_DECIMALS_PRECISION=ambient-secret OCANNL_BACKEND=cuda \
    OPAMSWITCH=caller-switch DUNE_BUILD_DIR=caller-build \
    bash "$subject" loopback "$REF" --backend "$BACKEND" --repo "$run/repo" \
    --worktree-root "$run/worktrees" --cap 10 --trip-cap 30 "$@" >"$run/stdout" 2>&1 || rc=$?
  printf '%s\n' "$rc" >"$run/rc"
  # Prove ownership: the original checkout, its dirty file and FETCH_HEAD stay
  # intact. Inspect actual Git registration and filesystem after every outcome.
  [ "$(cat "$run/repo/untouched")" = 'do not touch' ] &&
    [ "$(cat "$run/repo/.git/FETCH_HEAD")" = 'fetch head sentinel' ] &&
    [ "$("$REAL_GIT" -C "$run/repo" rev-parse HEAD)" = "$SHA" ] &&
    [ "$("$REAL_GIT" -C "$run/repo" for-each-ref refs/remotes/)" = "$tracking" ] || return 1
  if [ "$mode" = cleanup-fail ]; then
    local owned
    owned=$(sed -n 's/^worktree:      //p' "$run/stdout")
    case $owned in "$run/worktrees/machine-verify."*) ;; *) return 1 ;; esac
    [ -d "$owned" ] || return 1
    "$REAL_GIT" -C "$run/repo" worktree remove --force "$owned" || return 1
  fi
  [ -z "$("$REAL_GIT" -C "$run/repo" for-each-ref refs/machine-verify/)" ] &&
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

# The commit form of BRANCH: certified exactly when a branch of the remote,
# fetched now, contains it; the provenance names one, preferring master.
REF=$SHA check_case commit-success success 0 "verified .*commit=$SHA backend=cc" --test @fixture
if grep -qx "pushed commit: $SHA" "$TMP/runs/commit-success/stdout" &&
  grep -qx 'reachable from: origin/master (2 branch head(s) contain it)' "$TMP/runs/commit-success/stdout" &&
  grep -qx "resolved commit: $SHA" "$TMP/runs/commit-success/stdout" &&
  grep -q '|cc|build -j 4 @fixture' "$TMP/runs/commit-success/audit"; then
  report 0 'commit: provenance names the commit and a branch containing it'
else report 1 'commit: provenance names the commit and a branch containing it' "$TMP/runs/commit-success"; fi
REF=$(printf %s "$SHA" | tr a-f A-F) check_case commit-uppercase success 0 "verified .*commit=$SHA backend=cc"
REF=$UNPUSHED check_case commit-dangling success 2 \
  "commit $UNPUSHED is not reachable from any branch of origin"
REF=$UNPUSHED check_case commit-unpushed unpushed 2 \
  "commit $UNPUSHED is not reachable from any branch of origin"
REF=$UNPUSHED check_case commit-stale-namespace stale-heads 2 \
  "commit $UNPUSHED is not reachable from any branch of origin"
[ -e "$TMP/runs/commit-stale-namespace/ssh-log.stale" ]
report $? 'commit: a leftover heads namespace was planted and did not certify' "$TMP/runs/commit-stale-namespace"
REF=$SHA check_case commit-failed-fetch fetch-fail 2 'cannot fetch the branch heads of origin'
REF=$SHA ENDPOINT=127.0.0.1 TRANSPORT=local check_case local-commit success 0 \
  "verified .*commit=$SHA backend=cc" --test @fixture

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

# Optional-backend provenance: the vendor library's objects, the `select` arm
# that names it, and the other two GPU backends' objects absent on that build.
BACKEND=metal ENDPOINT=127.0.0.1 TRANSPORT=local check_case lib-metal lib-ok 0 \
  'verified .*backend=metal' --expect-lib metal --test @fixture
if grep -q '^machine-verify: optional-library evidence: PASS _build/default/arrayjit/lib/.metal_backend.objs/byte/metal_backend.cmi$' \
  "$TMP/runs/lib-metal/stdout" &&
  grep -q '^machine-verify: select-arm evidence: PASS # 1 "arrayjit/lib/metal_backend_impl.metal.ml"$' \
    "$TMP/runs/lib-metal/stdout" &&
  grep -q '^machine-verify: other-backend negative control: PASS absent: .*cuda_backend.cmi .*hip_backend.cmi$' \
    "$TMP/runs/lib-metal/stdout" &&
  grep -q '|metal|build -j 4 @fixture' "$TMP/runs/lib-metal/audit"; then
  report 0 'metal: library, select arm and negative controls are observed'
else report 1 'metal: library, select arm and negative controls are observed' "$TMP/runs/lib-metal"; fi
BACKEND=cuda check_case lib-cudajit lib-ok 0 'other-backend negative control: PASS absent: .*hip_backend.cmi .*metal_backend.cmi$' \
  --expect-lib cudajit
BACKEND=metal check_case lib-stub-arm lib-stub 2 'metal_backend_impl.ml selected the wrong arm: # 1 "arrayjit/lib/metal_backend_impl.missing.ml"' \
  --expect-lib metal
BACKEND=metal check_case lib-missing lib-absent 2 'metal evidence missing: .*metal_backend.cmi' --expect-lib metal
BACKEND=metal check_case lib-other-present lib-other 2 "negative control failed: another backend's artifact exists at .*hip_backend.cmi" \
  --expect-lib metal
# The tile-MMA capability (gh-ocannl-1070): stated for every optional library,
# asserted for hipjit where the device is eligible and hipconfig's tree holds a
# complete rocWMMA install, reported as scalar-only where either half is absent.
if grep -q '^machine-verify: tile-MMA capability (metal): 16x16x16 (reported, not asserted for metal)$' \
  "$TMP/runs/lib-metal/stdout" &&
  grep -q "^machine-verify: verified .*backend=metal tile_mma=16x16x16$" "$TMP/runs/lib-metal/stdout"; then
  report 0 'metal: tile-MMA capability stated and carried on the verdict'
else report 1 'metal: tile-MMA capability stated and carried on the verdict' "$TMP/runs/lib-metal"; fi
BACKEND=hip check_case lib-hipjit lib-ok 0 \
  "^machine-verify: tile-MMA capability: PASS 16x16x16 \\(devices eligible; rocWMMA under $TMP/hip-complete/include\\)$" \
  --expect-lib hipjit
grep -q '^machine-verify: verified .*backend=hip tile_mma=16x16x16$' "$TMP/runs/lib-hipjit/stdout"
report $? 'hipjit: the verdict carries the asserted capability' "$TMP/runs/lib-hipjit"
BACKEND=hip check_case lib-mma-missing lib-mma-none 2 \
  'tile-MMA capability missing: every device is eligible and .*/hip-complete/include holds rocWMMA, but the backend advertises none' \
  --expect-lib hipjit
BACKEND=hip check_case lib-ineligible lib-ineligible 0 \
  'tile-MMA capability: NONE -- every Tile_mma renders the scalar fallback here \(a device is not tile-MMA eligible' \
  --expect-lib hipjit
BACKEND=hip HIP_TREE=hip-partial check_case lib-partial-rocwmma lib-no-rocwmma 0 \
  'tile-MMA capability: NONE -- .*\(no complete rocWMMA header tree under hipconfig --path=.*/hip-partial\)$' \
  --expect-lib hipjit
grep -q '^machine-verify: verified .*backend=hip tile_mma=none$' "$TMP/runs/lib-partial-rocwmma/stdout"
report $? 'hipjit: a scalar-only box says so on the verdict' "$TMP/runs/lib-partial-rocwmma"
BACKEND=hip check_case lib-no-eligibility lib-no-eligibility 2 \
  'reported no per-device tile_mma_eligible; the tile-MMA check would be vacuous' --expect-lib hipjit
# A probe that states neither the tile nor its absence fails rather than
# reading as scalar-only, for a reported library as much as an asserted one.
BACKEND=metal check_case lib-mma-unparsed lib-mma-unparsed 2 \
  'printed neither a limits\.mma\.mma_tile line nor limits\.mma = \(\); the tile-MMA capability is unreadable, not none$' \
  --expect-lib metal
BACKEND=cc check_refusal lib-backend-conflict '^machine-verify: --expect-lib metal conflicts with --backend cc$' --expect-lib metal

# The dune width (gh-ocannl-986): with no -j, the far side runs every build at
# the tightest width tools/box-jobs.sh gives the box it probes for any backend
# the trip can hold -- the pinned one, any a reached stanza names, every one
# for a --run probe or an unreadable answer -- the per-slot cap
# tools/test-run.sh injects there; an explicit -j wins; where no backend meets
# a cap the width is 4. Observed at the fake dune, every build of the trip (the
# backend readers' own build, when there is one, is compile-only and runs at
# dune's width), not read from the verifier's report of its width.
width_case() { # NAME WIDTH PROVENANCE-ERE [args]
  local name=$1 width=$2 story=$3 ok=0
  shift 3
  check_case "$name" success 0 "verified .*backend=$BACKEND" --test @fixture "$@"
  grep -q "|$BACKEND|build -j $width @fixture\$" "$TMP/runs/$name/audit" || ok=1
  ! grep -v -e "|build -j $width " -e '|build ./test/config/ocannl_read_config.exe ' \
    "$TMP/runs/$name/audit" | grep -q . || ok=1
  grep -qE "^dune jobs:     $width $story" "$TMP/runs/$name/stdout" || ok=1
  report "$ok" "$name: every build ran at -j $width, and the provenance says why" "$TMP/runs/$name"
}
BACKEND=cuda NVIDIA=present width_case width-native-cuda 8 \
  "\(this box's width for cuda, which this trip can hold -- .*: tools/box-jobs.sh hazard nvidia\)$"
BACKEND=hip KFD=small width_case width-small-sdma 4 "\(.*for hip, .*hazard sdma\)$"
BACKEND=hip KFD=wide width_case width-wide-sdma 8 "\(.*for hip, .*hazard wide-sdma\)$"
# The bridge outranks the pool, and the local transport reads the same table.
BACKEND=hip DXG=present KFD=wide ENDPOINT=127.0.0.1 TRANSPORT=local width_case width-dxg-local 2 \
  "\(.*for hip, .*hazard dxg\)$"
BACKEND=cuda NVIDIA=present width_case width-explicit 3 '\(explicit -j\)$' -j 3
BACKEND=hip KFD=wide width_case width-explicit-long 12 '\(explicit -j\)$' --jobs 12
# A CPU backend beside a GPU is uncapped, except on the fleet's rog-nv-linux.
NVIDIA=present width_case width-cpu-default 4 \
  "\(default; tools/box-jobs.sh names no cap here for any backend this trip can hold\)$"
NVIDIA=present FLEET_BOX=rog-nv-linux width_case width-cpu-rog 8 "\(.*for cc, .*hazard nvidia-cpu\)$"
# Codex review round 1 on PR #902: a CPU trip still holds the GPU a reached
# stanza names, and a --run probe or an unread answer may hold any backend.
BACKEND=cc DXG=present NAMES='cuda:arrayjit/test/dune names it' width_case width-reached-gpu 2 \
  "\(this box's width for cuda, which this trip can hold -- arrayjit/test/dune names it: tools/box-jobs.sh hazard dxg\)$"
grep -q '^machine-verify: batch: holds cuda: arrayjit/test/dune names it$' "$TMP/runs/width-reached-gpu/stdout"
report $? 'width-reached-gpu: the resolution is in the provenance' "$TMP/runs/width-reached-gpu"
BACKEND=cc NVIDIA=present width_case width-probe 8 \
  "\(.*for cuda, .*its backends are unread \(a --run probe may pick its own backend\).*hazard nvidia\)$" \
  --run 'printf "probe width: %s\n" "${DUNE_JOBS:-unset}"'
# The probe's own dune invocations inherit the width (round 2 on PR #902).
grep -qx 'probe width: 8' "$TMP/runs/width-probe/stdout"
report $? 'width-probe: the probe runs with DUNE_JOBS at the trip width' "$TMP/runs/width-probe"
BACKEND=cc NVIDIA=present NAMES=fail width_case width-unread 8 "\(.*for cuda, .*its backends are unread .*hazard nvidia\)$"
# No stand-ins: the pushed tree's readers are built there (a fixture tree has
# none, so the answer is unread and every backend counts).
BACKEND=cc NVIDIA=present READERS=absent width_case width-readers-built 8 "\(.*for cuda, .*its backends are unread .*\)$"
grep -q '|cc|build ./test/config/ocannl_read_config.exe ./test/config/ocannl_slot_kind.exe$' \
  "$TMP/runs/width-readers-built/audit"
report $? 'width-readers-built: the readers were built in the pushed worktree' "$TMP/runs/width-readers-built"
check_refusal width-zero '^machine-verify: jobs must be a positive integer$' -j 0
check_refusal width-word '^machine-verify: jobs must be a positive integer$' --jobs auto

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
  mkdir -p "$dir" && cp "$SRC" "$FAR" "$BOX_JOBS" "$BATCH_BACKENDS" "$dir/" || return 1
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
# The containment check dropped: a commit that exists only in BOX's checkout,
# on a local branch, certifies as though the staging remote held it.
containment_oracle() {
  local subject=$1 name=$2
  REF=$UNPUSHED WANT_SHA=$UNPUSHED run_case "$subject" "$name" unpushed || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q 'is not reachable from any branch of origin' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-containment far '/^  \[ -n "\$containing" \] \|\|$/ { getline; next } { print }') || exit 2
expect_rejected 'commit containment removed' "$mutated" containment_oracle "^machine-verify: verified .*commit=$UNPUSHED "
# The namespace not emptied before the fetch: a leftover head from a killed
# verifier with the same PID certifies a commit no live branch contains.
stale_heads_oracle() {
  local subject=$1 name=$2
  REF=$UNPUSHED WANT_SHA=$UNPUSHED run_case "$subject" "$name" stale-heads || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q 'is not reachable from any branch of origin' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-namespace-clear far '/^  drop_heads \|\| fail "cannot clear leftover refs/ { next } { print }') || exit 2
expect_rejected 'leftover heads namespace kept' "$mutated" stale_heads_oracle "^machine-verify: verified .*commit=$UNPUSHED "
mutated=$(mutant_pair no-golden-scope far '/^assert_only_promoted_goldens\(\) \{/ { print; print "  return 0"; next } { print }') || exit 2
expect_rejected 'golden scope removed' "$mutated" golden_mutation_oracle '^machine-verify: verified '
# The capability assertion dropped: an eligible device beside a complete
# rocWMMA tree certifies a scalar-only backend, the gh-ocannl-1070 blind spot.
tile_mma_oracle() {
  local subject=$1 name=$2
  BACKEND=hip run_case "$subject" "$name" lib-mma-none --expect-lib hipjit || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 2 ] &&
    grep -q 'tile-MMA capability missing' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-tile-mma-assertion far '/^    \[ "\$tile_mma" != none \] \|\|$/ { getline; next } { print }') || exit 2
expect_rejected 'tile-MMA assertion removed' "$mutated" tile_mma_oracle '^machine-verify: verified .*tile_mma=none$'
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
# The box's width dropped for the former flat default: a GPU leg run without
# -j goes back to running above the box's cap, the case this issue was filed on.
width_oracle() {
  local subject=$1 name=$2
  BACKEND=cuda NVIDIA=present run_case "$subject" "$name" success --test @fixture || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 0 ] &&
    grep -q '|cuda|build -j 8 @fixture$' "$TMP/runs/$name/audit"
}
mutated=$(mutant_pair no-box-width far '/^if \[ -n "\$jobs" \]; then$/ { print "jobs=${jobs:-4}" } { print }') || exit 2
expect_rejected 'box width ignored' "$mutated" width_oracle '^dune jobs:     4 \(explicit -j\)$'
# The width judged over the pinned backend alone: a reached stanza's GPU runs
# above the bridge's cap again.
reached_oracle() {
  local subject=$1 name=$2
  BACKEND=cc DXG=present NAMES='cuda:arrayjit/test/dune names it' run_case "$subject" "$name" success \
    --test @fixture || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 0 ] &&
    grep -q '|cc|build -j 2 @fixture$' "$TMP/runs/$name/audit"
}
mutated=$(mutant_pair pinned-backend-only far '/batch_resolve dune "\$log" build/ { print "      :"; next } { print }') || exit 2
expect_rejected 'reached backends ignored' "$mutated" reached_oracle '^dune jobs:     4 \(default; '
# The width withheld from a --run probe: its bare dune runs at dune's default.
probe_width_oracle() {
  local subject=$1 name=$2
  BACKEND=cc NVIDIA=present run_case "$subject" "$name" success \
    --run 'printf "probe width: %s\n" "${DUNE_JOBS:-unset}"' || return 1
  [ "$(cat "$TMP/runs/$name/rc")" = 0 ] && grep -qx 'probe width: 8' "$TMP/runs/$name/stdout"
}
mutated=$(mutant_pair no-probe-width far '{ sub(/ "DUNE_JOBS=\$jobs" sh -c/, " sh -c"); print }') || exit 2
expect_rejected 'probe width withheld' "$mutated" probe_width_oracle '^probe width: unset$'
finish

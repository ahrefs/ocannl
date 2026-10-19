#!/usr/bin/env bash
# Direct fixtures for the sourced sweep components. No sweep, remote host or OCaml
# executable is launched. Run `bash test/operations/sweep_components.sh [COMPONENT]`;
# COMPONENT is all (default), lab, fleet, fingerprint, state or tokens.
set -uo pipefail
. "$(dirname "$0")/../../scripts/harness-support.sh"
component=${1:-all}
[ $# -eq 0 ] || shift
case $component in all | lab | fleet | fingerprint | state | tokens) ;; *) echo "unknown component: $component" >&2; exit 2 ;; esac
TOOLS=$(cd "$(dirname "$0")/../../tools" && pwd)
if [ "${1:-}" = --tools ]; then
  [ $# -ge 2 ] || { echo "--tools requires a directory" >&2; exit 2; }
  TOOLS=$2
  shift 2
fi
harness_args "$@"
harness_scratch ocannl-sweep-components
harness_require bash awk git perl
die() { echo "sweep: $*" >&2; exit 2; }
say() { printf '%s\n' "$*" >>"$TMP/messages"; }
equal() { # label actual expected
  local rc=0
  [ "$2" = "$3" ] || rc=1
  report "$rc" "$1" "got '$2'; wanted '$3'"
}
lab_fixture() {
  . "$TOOLS/lab-map.sh"
  LAB_MAP='alpha renamed-linux renamed-win renamed-wsl alpha-lan
beta beta-linux beta-wsl'
  equal 'lab: lookup follows the supplied map' "$(lab_row alpha)" 'renamed-linux renamed-win renamed-wsl alpha-lan'
  equal 'lab: both boots share one box' "$(lab_box_of renamed-wsl)/$(lab_box_of renamed-linux)" alpha/alpha
  equal 'lab: boot destination follows renamed aliases' "$(lab_dest_of alpha linux)/$(lab_dest_of alpha wsl)" renamed-linux/renamed-wsl
  lab_dest_of alpha win >"$TMP/absent"; equal 'lab: non-Linux boot is refused' "$?" 1
  lab_row absent >"$TMP/absent"; equal 'lab: unknown row is refused' "$?" 1
  WAKE_LAB=$TMP/wake LAB_HOSTS=$TMP/hosts LAB_LOCK_DIR=$TMP/locks
  printf 'kind_of() { echo wsl; }\n' >"$LAB_HOSTS"
  equal 'lab: site table selects the boot' "$(lab_dest alpha)" renamed-wsl
  OCANNL_TOOL_SWEEP_DEST_ALPHA=renamed-linux
  equal 'lab: override selects a map member' "$(lab_dest alpha)" renamed-linux
  OCANNL_TOOL_SWEEP_DEST_ALPHA=beta-linux
  harness_rejected 1 'not the -linux or -wsl alias' "$TMP/bad-override" lab_dest alpha
  report "$?" 'lab: an alias from another box is refused'
  unset OCANNL_TOOL_SWEEP_DEST_ALPHA
  LAB_LANE_BOXES='alpha beta' LAB_CONTRACT=
  ask_capped() { printf -v "$1" '%s' "$LAB_LOCK_DIR/$5.lock"; }
  lab_contract_check
  equal 'lab: contract checks every selected box' "$LAB_CONTRACT" "agree with $WAKE_LAB for alpha beta"
  harness_rejected 2 'lab lock contract.*broken' "$TMP/bad-contract" bash -c '. "$1"; LAB_LANE_BOXES=alpha; LAB_LOCK_DIR=/locks; WAKE_LAB=/wake; ask_capped() { printf -v "$1" /wrong; }; die() { echo "$*"; exit 2; }; lab_contract_check' fixture "$TOOLS/lab-map.sh"
  report "$?" 'lab: a broken lock contract refuses the run'
}

fleet_fixture() {
  . "$TOOLS/lab-map.sh"
  . "$TOOLS/fleet-registry.sh"
  LAB_MAP='alpha renamed-linux renamed-win renamed-wsl'
  fleet_worker_candidates() { printf '%s\n' "$TMP/fleet"; }
  # A shebang, not just the mode bit: MSYS decides -x from a file's content on NTFS.
  printf '#!/bin/sh\n' >"$TMP/fleet"; chmod +x "$TMP/fleet"
  ask_capped() { printf -v "$1" '%s' 'EXECUTION slot on local-fixture with 4 tokens'; }
  sweep_fleet_probe
  equal 'fleet: probe retains the host identity' "$SWEEP_FLEET_BOX" local-fixture
  equal 'fleet: local lane uses the probed name' "$(sweep_fleet_lane_names '')" local-fixture
  equal 'fleet: remote lane checks every boot' "$(sweep_fleet_lane_names renamed-linux)" 'renamed-linux renamed-win renamed-wsl'
  LANE_DIR=$TMP
  run_capped() { cat "$TMP/registry"; return "${registry_rc:-0}"; }
  printf '%s\n' '[{"request_id":"held","state":"reserved","request":{"kind":"measurement","execution_host":"renamed-win"}},{"request_id":"other","state":"running","request":{"kind":"correctness","execution_host":"renamed-linux"}}]' >"$TMP/registry"
  sweep_fleet_under_measurement renamed-linux renamed-win renamed-wsl
  equal 'fleet: measurement on another boot holds the lane' "$?:$MEASUREMENT_HOLDERS" '0:held (reserved on renamed-win)'
  equal 'fleet: the same read keeps the box reservations' "$BOX_RESERVATIONS" '[{"execution_host":"renamed-win","kind":"measurement","request_id":"held","state":"reserved"},{"execution_host":"renamed-linux","kind":"correctness","request_id":"other","state":"running"}]'
  sweep_fleet_under_measurement renamed-linux
  equal 'fleet: correctness does not hold the lane' "$?:$MEASUREMENT_HOLDERS" '1:'
  equal "fleet: a reservation on the lane's names is kept" "$BOX_RESERVATIONS" '[{"execution_host":"renamed-linux","kind":"correctness","request_id":"other","state":"running"}]'
  printf '%s\n' '{"not":"a list"}' >"$TMP/registry"
  sweep_fleet_under_measurement renamed-linux
  equal 'fleet: malformed registry is a read failure' "$?" 2
  equal 'fleet: an unread registry keeps no reservations' "$BOX_RESERVATIONS" ''

  registry_rc=7
  sweep_fleet_under_measurement renamed-linux
  equal 'fleet: command failure is a read failure' "$?" 2
  registry_rc=0
  ask_capped() { return 1; }
  OCANNL_TOOL_FLEET_WORKER=none sweep_fleet_probe
  equal 'fleet: disabled registry is reported' "$SWEEP_FLEET_STATUS" 'NOT CONSULTED -- OCANNL_TOOL_FLEET_WORKER=none'
}

fingerprint_fixture() {
  . "$TOOLS/box-jobs.sh"
  . "$TOOLS/kernel-window.sh"
  . "$TOOLS/gpu-tokens.sh"
  . "$TOOLS/sweep-fingerprint.sh"
  cat >"$TMP/diagnostics.log" <<'LOG'
File "test/operations/dune", lines 100-104:
100 | (rule
101 |  (alias stable-alias)
102 |  (action (run probe)))
Error: repeated failure
Error: repeated failure
nvrtc options: --fixture=whole vector
=== rtc-context (cuda) ===
toolkit: fixture
=== end rtc-context ===
serial rerun: all clean
serial rerun: suite completed
LOG
  sweep_fingerprint "$TMP/diagnostics.log" >"$TMP/fp"
  equal 'fingerprint: dune spans normalize to the stanza' "$(head -1 "$TMP/fp")" 'Error: repeated failure'
  equal 'fingerprint: sites retain the stable alias' "$(sweep_fingerprint_sites "$TMP/diagnostics.log")" 'File "test/operations/dune", alias stable-alias'
  equal 'fingerprint: repeated diagnostics are deduplicated' "$(grep -c '^Error:' "$TMP/fp")" 1
  equal 'fingerprint: option vector survives whole' "$(grep '^nvrtc options:' "$TMP/fp")" 'nvrtc options: --fixture=whole vector'
  equal 'fingerprint: context order and serial verdict survive' "$(tail -5 "$TMP/fp")" "$(printf '%s\n' '=== rtc-context (cuda) ===' 'toolkit: fixture' '=== end rtc-context ===' 'serial rerun: all clean' 'serial rerun: suite completed')"
  : >"$TMP/empty.log"
  sweep_fingerprint_write "$TMP/empty.log" box/cc
  equal 'fingerprint: empty diagnostics publish a sentinel' "$(cat "$WRITTEN_FINGERPRINT")" "$EMPTY_FINGERPRINT"
  # A GPU unit's token readings (gh-ocannl-1177) contribute their stable half only: contention, not
  # its count or its pids, so the same contention twice compares equal.
  printf '%s\n' '{"event":"opened","status":"read","box":"b","gpu_tokens":2,"gpu_holders":1,"token_holders":[{"slot":1,"kind":"gpu","pid":11,"user":"u","command":"c"}]}' \
    '{"event":"closed","status":"read","box":"b","gpu_tokens":2,"gpu_holders":0,"token_holders":[]}' \
    >"$TMP/diagnostics.gpu-tokens"
  sweep_fingerprint "$TMP/diagnostics.log" >"$TMP/fp-tokens"
  equal 'fingerprint: contention is carried, before the serial verdict' "$(tail -3 "$TMP/fp-tokens")" "$(printf '%s\n' 'gpu tokens: other batches held the GPU' 'serial rerun: all clean' 'serial rerun: suite completed')"
  sed 's/"pid":11/"pid":12/' "$TMP/diagnostics.gpu-tokens" >"$TMP/moved" && mv "$TMP/moved" "$TMP/diagnostics.gpu-tokens"
  sweep_fingerprint "$TMP/diagnostics.log" >"$TMP/fp-tokens-again"
  cmp -s "$TMP/fp-tokens" "$TMP/fp-tokens-again"; report "$?" 'fingerprint: another pid holding the token does not move it'
}

# The GPU tokens' reader, summaries and take (tools/gpu-tokens.sh, gh-ocannl-1177), against a box
# that is this host. The reading is the box's own `fleet-worker.sh execution slot --holders`
# (gh-ocannl-1417), which a fixture fleet-worker.sh answers here with lines in the fleet's format
# (ludics-lite's holders.py, whose own tests hold that format to the kernel's flock table); the take
# is real flocks on fixture slot files, held by fixture batches.
tokens_fixture() {
  . "$TOOLS/gpu-tokens.sh"
  . "$TOOLS/sweep-supervisor.sh"
  local state=$TMP/slot-state dir fw gpu_pid holder_pid rc
  dir=$state/rogfix
  mkdir -p "$dir"
  fw=$TMP/fleet-worker.sh
  # The box's fleet-worker.sh: four slots, two tokens; `--holders` answers FAKE_HOLDERS (a file) and
  # FAKE_HOLDERS_ERR on stderr, exiting FAKE_HOLDERS_RC; a take is a real flock on the first free
  # token slot, inherited by the command it execs, or the real refusal. FAKE_SLOT picks a refusal.
  cat >"$fw" <<'FW'
#!/bin/sh
case $* in
  'execution slot --probe') echo 'EXECUTION SLOT PROBE rogfix 4 2' ;;
  'execution slot --holders')
    [ -z "${FAKE_HOLDERS_ERR:-}" ] || echo "$FAKE_HOLDERS_ERR" >&2
    [ -z "${FAKE_HOLDERS:-}" ] || cat "$FAKE_HOLDERS"
    exit "${FAKE_HOLDERS_RC:-0}"
    ;;
  'execution slot --gpu --wait '*)
    case ${FAKE_SLOT:-} in
      measurement) echo 'EXECUTION SLOT REFUSED rogfix: a measurement holds the box exclusively (m-1)'; exit 1 ;;
      unreachable) echo 'EXECUTION SLOT UNREACHABLE anchor: registry unread, no slot taken'; exit 4 ;;
    esac
    wait=$5
    shift 6
    exec perl -e '
      use Fcntl qw(:flock F_GETFD F_SETFD FD_CLOEXEC);
      my ($dir, $wait, @cmd) = @ARGV;
      my $deadline = time + $wait;
      while (1) {
        for my $i (1, 2) {
          open(my $h, ">>", "$dir/slot.$i") or die;
          next unless flock($h, LOCK_EX | LOCK_NB);
          fcntl($h, F_SETFD, fcntl($h, F_GETFD, 0) & ~FD_CLOEXEC);
          print STDERR "EXECUTION SLOT rogfix: slot $i of 4, GPU token $i of 2 held for: @cmd\n";
          exec @cmd or exit 127;
        }
        last if time >= $deadline;
        select undef, undef, undef, 0.2;
      }
      print "EXECUTION SLOT REFUSED rogfix: all 2 GPU tokens (slots 1-2 of 4) busy after ${wait}s; a batch that holds no GPU declares --cpu\n";
      exit 1' "$FLEET_SLOT_STATE/rogfix" "$wait" "$@"
    ;;
  *) exit 2 ;;
esac
FW
  chmod +x "$fw"
  holder() { # slot -- a batch holding that slot file's flock until killed; its pid in holder_pid
    perl -e 'use Fcntl ":flock"; my $f = shift; open(my $h, ">>", $f) or die; flock($h, LOCK_EX) or die;
      open(my $r, ">", "$f.ready") or die; close $r; sleep 60' "$dir/slot.$1" &
    holder_pid=$!
    local n=0
    until [ -e "$dir/slot.$1.ready" ] || [ "$n" -ge 200 ]; do sleep 0.05; n=$((n + 1)); done
    rm -f "$dir/slot.$1.ready"
  }
  # The fleet's holder lines (lukstafi/ludics-lite#639). At `opened`: an undeclared batch on the
  # first token; a declared `--cpu` batch on the second, which the fleet lists `gpu` (the SLOT's
  # kind: a CPU batch reaches the tokens last, but it can) and which holds no device memory; and a
  # batch on slot 3, past the tokens, where the fleet lists `cpu`. At `closed`: a held token whose
  # holder the lock table did not name (pid null), and a slot file that could not be tested.
  holders_line() { # event slot kind pid command -- "null" for an unknown pid or command
    local cmd=null
    [ "$5" = null ] || cmd="\"$5\""
    printf '{"box":"rogfix","event":"%s","at":"2026-10-19T00:00:00+00:00","slot":%s,"slots":4,"kind":"%s","pid":%s,"user":"u","command":%s,"cwd":null,"source":"proc-locks","unknown":{}}\n' \
      "$1" "$2" "$3" "$4" "$cmd"
  }
  {
    holders_line held 1 gpu 101 'python3 -m ludics.fleetworker /fw execution slot --wait 10 -- tests --cpu'
    holders_line held 2 gpu 102 'python3 -m ludics.fleetworker /fw execution slot --cpu --wait 10 -- tests'
    holders_line held 3 cpu 103 'python3 -m ludics.fleetworker /fw execution slot --wait 10 -- tests'
  } >"$TMP/holders-opened"
  {
    holders_line held 1 gpu null null
    holders_line untested 4 cpu null null
  } >"$TMP/holders-closed"
  reading() { # event [candidate...] -- one reading into $TMP/answer, its status in rc
    local event=$1
    shift
    sh -c "$(gpu_tokens_cmd "$event" "$@")" >"$TMP/answer" 2>&1
    rc=$?
  }
  holders_of() { # -- the answer's holders as slot:kind:gpu:pid words
    perl -MJSON::PP -e 'local $/; my $r = JSON::PP->new->decode(<STDIN>);
      print join(" ", @$r{qw(event status box slots gpu_tokens gpu_holders)},
        map { "$_->{slot}:$_->{kind}:" . ($_->{gpu} ? "gpu" : "nogpu") . ":" . ($_->{pid} // "?") } @{ $r->{token_holders} })' \
      <"$TMP/answer"
  }
  status_of() { perl -MJSON::PP -e 'local $/; my $r = JSON::PP->new->decode(<STDIN>); print "$r->{status}: ", $r->{detail} // ""' <"$TMP/answer"; }
  FAKE_HOLDERS=$TMP/holders-opened reading opened "$TMP/absent-fleet-worker.sh" "$fw"
  equal 'tokens: a reading answers in one JSON line' "$rc:$(wc -l <"$TMP/answer" | tr -d ' ')" 0:1
  equal "tokens: the reader keeps the fleet's holders and counts the GPU's" "$(holders_of)" \
    'opened read rogfix 4 2 1 1:gpu:gpu:101 2:gpu:nogpu:102 3:cpu:nogpu:103'
  rm -f "$TMP/unit.gpu-tokens"
  gpu_tokens_append "$TMP/unit.gpu-tokens" opened "$TMP/answer" "$rc"
  FAKE_HOLDERS=$TMP/holders-closed reading closed "$fw"
  equal 'tokens: an unnamed holder counts, an untested slot does not' "$(holders_of)" \
    'closed read rogfix 4 2 1 1:gpu:gpu:? 4:cpu:nogpu:?'
  gpu_tokens_append "$TMP/unit.gpu-tokens" closed "$TMP/answer" "$rc"
  gpu_tokens_append "$TMP/unit.gpu-tokens" refused /dev/null 255
  equal 'tokens: the record field names each reading' "$(gpu_tokens_summary field "$TMP/unit.gpu-tokens")" 'opened=1/2,closed=1/2,refused=?'
  equal 'tokens: the fingerprint says contention' "$(gpu_tokens_summary fingerprint "$TMP/unit.gpu-tokens")" 'gpu tokens: other batches held the GPU'
  equal 'tokens: the take parameters come from the opened reading' "$(gpu_tokens_summary take "$TMP/unit.gpu-tokens")" "$fw rogfix 4 2"
  gpu_tokens_summary report "$TMP/unit.gpu-tokens" >"$TMP/report"
  equal 'tokens: the report names the GPU holders, and only those' "$(cat "$TMP/report")" "$(printf '%s\n' \
    'gpu tokens: 1 of 2 held by other batches at opened on rogfix -- slot 1 pid 101 (u): python3 -m ludics.fleetworker /fw execution slot --wait 10 -- tests --cpu' \
    'gpu tokens: 1 of 2 held by other batches at closed on rogfix -- slot 1 pid ? (u): (unknown)')"
  equal 'tokens: no readings is no field' "$(gpu_tokens_summary field "$TMP/none.gpu-tokens")" -
  : >"$TMP/holders-none"
  FAKE_HOLDERS=$TMP/holders-none reading opened "$fw"
  equal 'tokens: a box with no holders reads as none' "$(holders_of)" 'opened read rogfix 4 2 0'
  # Every answer that is not the fleet's list of holders is `unread`, never a box with no holders.
  reading opened "$TMP/absent-fleet-worker.sh"
  equal 'tokens: no fleet-worker.sh is an unread reading' "$(status_of)" "unread: no fleet-worker.sh answered 'execution slot --probe'"
  FAKE_HOLDERS_RC=2 FAKE_HOLDERS_ERR='fleet-worker.sh: execution slot: unknown option --holders' reading opened "$fw"
  equal 'tokens: a fleet-worker.sh without --holders is unread' "$(status_of)" \
    'unread: execution slot --holders exited 2: fleet-worker.sh: execution slot: unknown option --holders'
  FAKE_HOLDERS=$TMP/holders-opened FAKE_HOLDERS_ERR='a warning' reading opened "$fw"
  equal 'tokens: a line that is not a holder is unread' "$(status_of)" 'unread: execution slot --holders answered: a warning'
  sed 's/"box":"rogfix"/"box":"elsewhere"/' "$TMP/holders-opened" >"$TMP/holders-elsewhere"
  FAKE_HOLDERS=$TMP/holders-elsewhere reading opened "$fw"
  equal "tokens: another box's holder is unread" "$(status_of | cut -c1-44)" 'unread: execution slot --holders answered: {'
  sed '2s/"kind":"gpu"/"kind":"maybe"/' "$TMP/holders-opened" >"$TMP/holders-kind"
  FAKE_HOLDERS=$TMP/holders-kind reading opened "$fw"
  equal 'tokens: a holder of no known kind is unread' "$(status_of | cut -c1-44)" 'unread: execution slot --holders answered: {'
  printf 'not json\n' >"$TMP/garbled"
  rm -f "$TMP/garbled.gpu-tokens"
  gpu_tokens_append "$TMP/garbled.gpu-tokens" opened "$TMP/garbled" 0
  equal 'tokens: an answer that is not a reading is unread' "$(gpu_tokens_summary field "$TMP/garbled.gpu-tokens")" 'opened=?'
  equal 'tokens: unread readings fingerprint as not read' "$(gpu_tokens_summary fingerprint "$TMP/garbled.gpu-tokens")" 'gpu tokens: not read'
  # The take, run as the box runs it, with a batch on the first token: the leg runs inside the free
  # one, its status is its own, and the slot's lines follow in its output.
  holder 1; gpu_pid=$holder_pid
  take() { # wait -- runs a leg that prints and exits 7
    FLEET_SLOT_STATE=$state sh -c "$(gpu_token_take_cmd "$fw" "$1" "$TMP/take" "sh -c 'echo leg ran; exit 7'")" \
      >"$TMP/take.out" 2>&1
    rc=$?
  }
  take 5
  equal 'tokens: a taken token runs the leg with its own status' "$rc:$(grep -c '^leg ran$' "$TMP/take.out")" 7:1
  grep -qx 'sweep: gpu token: EXECUTION SLOT rogfix: slot 2 of 4, GPU token 2 of 2 held' "$TMP/take.out"
  report "$?" 'tokens: the slot line says which token the leg held'
  holder 2
  take 1
  equal 'tokens: no free token in time is 75 and no leg' "$rc:$(grep -c '^leg ran$' "$TMP/take.out")" 75:0
  equal 'tokens: the refusal names its reason' "$(gpu_token_refusal "$TMP/take.out")" 'no GPU token came free within 1s'
  kill "$holder_pid" 2>/dev/null; wait "$holder_pid" 2>/dev/null
  FAKE_SLOT=measurement take 1
  equal 'tokens: a measurement refusal is 75 too' "$rc:$(gpu_token_refusal "$TMP/take.out")" '75:a measurement holds the box'
  FAKE_SLOT=unreachable take 1
  equal 'tokens: an unanswered take runs the leg without a token' "$rc:$(grep -c '^leg ran$' "$TMP/take.out")" 7:1
  grep -q '^sweep: gpu token: WARNING: not taken (execution slot exited 4), so this leg runs WITHOUT one$' "$TMP/take.out"
  report "$?" 'tokens: the unanswered take says so'
  kill "$gpu_pid" 2>/dev/null
  wait "$gpu_pid" 2>/dev/null
}

state_fixture() {
  . "$TOOLS/sweep-unit-state.sh"
  MAIN=$TMP/repo UNIT_STATES=$TMP/states TARGET= SLOW=0 REF=origin/master
  mkdir -p "$MAIN" "$UNIT_STATES"
  git -C "$MAIN" init -q
  git -C "$MAIN" config user.name Fixture
  git -C "$MAIN" config user.email fixture@example.invalid
  git -C "$MAIN" config commit.gpgsign false
  git -C "$MAIN" config core.hooksPath "$TMP/no-hooks"
  printf 'first\n' >"$MAIN/test.expected"
  git -C "$MAIN" add test.expected
  git -C "$MAIN" commit -qm first
  full_sha=$(git -C "$MAIN" rev-parse HEAD)
  state=$(sweep_state_path box cc)
  other=$(REF=feature sweep_state_path box cc)
  equal 'state: logical refs have separate cursors' "$([ "$state" != "$other" ]; echo "$?")" 0
  other=$(TARGET=smoke sweep_state_path box cc)
  equal 'state: target scopes have separate cursors' "$([ "$state" != "$other" ]; echo "$?")" 0
  other=$(SLOW=1 sweep_state_path box cc)
  equal 'state: slow scope has a separate cursor' "$([ "$state" != "$other" ]; echo "$?")" 0
  sweep_state_update box cc pass
  printf 'failure one\n' >"$TMP/failure.fp"
  printf 'File "test.expected", line 1:\n' >"$TMP/failure.log"
  sweep_state_update box cc fail "$TMP/failure.fp" "$TMP/failure.log"
  equal 'state: failure replaces the verdict' "$(sweep_state_field "$state" last_verdict)" fail
  equal 'state: failure keeps the tested golden commit' "$(awk -F '\t' '$1 == "golden" {print $2 ":" $3}' "$state")" "$full_sha:test.expected"
  grep -q 'previous verdict was pass' "$TMP/messages"; report "$?" 'state: green to red is reported'
  cp "$state" "$TMP/before"
  for outcome in skip gate error timeout; do sweep_state_update box cc "$outcome"; done
  cmp -s "$state" "$TMP/before"; report "$?" 'state: unjudged outcomes preserve the cursor'
  sweep_state_update box cc pass
  equal 'state: green preserves the last failing fingerprint' "$(awk -F '\t' '$1 == "fingerprint" {print $2}' "$state")" 'failure one'
  printf 'failure two\n' >"$TMP/failure.fp"
  sweep_state_update box cc fail "$TMP/failure.fp" "$TMP/failure.log"
  grep -q 'fingerprint moved' "$TMP/messages"; report "$?" 'state: movement compares against the last failure'
  equal 'state: publication removes its scratch files' "$(find "$UNIT_STATES" -type f | wc -l | tr -d ' ')" 1
}

# A green fixture must fail when its shipping function is broken. Each child
# runs one component, so these controls never recurse into another control set.
controls_fixture() {
  local kind module pattern control rc
  for kind in lab fleet fingerprint state tokens; do
    control=$TMP/mutant-$kind
    mkdir -p "$control"
    cp "$TOOLS"/lab-map.sh "$TOOLS"/fleet-registry.sh "$TOOLS"/sweep-fingerprint.sh \
      "$TOOLS"/sweep-unit-state.sh "$TOOLS"/box-jobs.sh "$TOOLS"/kernel-window.sh \
      "$TOOLS"/gpu-tokens.sh "$TOOLS"/sweep-supervisor.sh "$control/"
    chmod u+w "$control"/*.sh
    case $kind in
      lab)
        module=lab-map.sh
        perl -pe 's/\*\-"\$2"/\*-linux/' "$TOOLS/$module" >"$control/$module"
        pattern='FAIL  lab: boot destination follows renamed aliases'
        ;;
      fleet)
        module=fleet-registry.sh
        sed 's/eq "measurement"/eq "correctness"/' "$TOOLS/$module" >"$control/$module"
        pattern='FAIL  fleet: measurement on another boot holds the lane'
        ;;
      fingerprint)
        module=sweep-fingerprint.sh
        sed 's/nvrtc|hiprtc|metal/hiprtc|metal/' "$TOOLS/$module" >"$control/$module"
        pattern='FAIL  fingerprint: option vector survives whole'
        ;;
      state)
        module=sweep-unit-state.sh
        perl -pe 's/\$REF/origin\/master/g' "$TOOLS/$module" >"$control/$module"
        pattern='FAIL  state: logical refs have separate cursors'
        ;;
      tokens)
        module=gpu-tokens.sh
        # A reader that takes the fleet's slot kind for the batch's: the declared CPU batch in a
        # token slot is then counted as holding the GPU.
        perl -pe 's/\$gpu = 0 if \$_ eq "--cpu"/\$gpu = \$gpu if \$_ eq "--cpu"/' "$TOOLS/$module" >"$control/$module"
        grep -q 'gpu = \$gpu if' "$control/$module" || echo 'tokens mutant did not apply' >&2
        pattern="FAIL  tokens: the reader keeps the fleet's holders and counts the GPU's"
        ;;
    esac
    rc=0
    harness_rejected 1 "$pattern" "$TMP/control-$kind.log" \
      bash "$0" "$kind" --tools "$control" || rc=$?
    report "$rc" "negative control: $kind fixture rejects a broken component"
  done
}

case $component in all | lab) lab_fixture ;; esac
case $component in all | fleet) fleet_fixture ;; esac
case $component in all | fingerprint) fingerprint_fixture ;; esac
case $component in all | state) state_fixture ;; esac
case $component in all | tokens) tokens_fixture ;; esac
case $component in all) controls_fixture ;; esac
finish

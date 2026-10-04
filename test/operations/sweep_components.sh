#!/usr/bin/env bash
# Direct fixtures for the sourced sweep components. No sweep, remote host or OCaml
# executable is launched. Run `bash test/operations/sweep_components.sh [COMPONENT]`;
# COMPONENT is all (default), lab, fleet, fingerprint or state.
set -uo pipefail
. "$(dirname "$0")/../../scripts/harness-support.sh"
component=${1:-all}
[ $# -eq 0 ] || shift
case $component in all | lab | fleet | fingerprint | state) ;; *) echo "unknown component: $component" >&2; exit 2 ;; esac
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
  touch "$TMP/fleet"; chmod +x "$TMP/fleet"
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
  sweep_fleet_under_measurement renamed-linux
  equal 'fleet: correctness does not hold the lane' "$?:$MEASUREMENT_HOLDERS" '1:'
  printf '%s\n' '{"not":"a list"}' >"$TMP/registry"
  sweep_fleet_under_measurement renamed-linux
  equal 'fleet: malformed registry is a read failure' "$?" 2
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
  for kind in lab fleet fingerprint state; do
    control=$TMP/mutant-$kind
    mkdir -p "$control"
    cp "$TOOLS"/lab-map.sh "$TOOLS"/fleet-registry.sh "$TOOLS"/sweep-fingerprint.sh \
      "$TOOLS"/sweep-unit-state.sh "$TOOLS"/box-jobs.sh "$TOOLS"/kernel-window.sh "$control/"
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
case $component in all) controls_fixture ;; esac
finish

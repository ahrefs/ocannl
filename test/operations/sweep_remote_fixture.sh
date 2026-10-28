#!/usr/bin/env bash
# Sourced by sweep_harness.sh: execute the sweep's emitted remote shell text
# unchanged, in cloned fixture homes. No backend or real host runs.
remote_root=$tmp/remote
remote_bin=$tmp/remote-bin
mkdir -p "$remote_root" "$remote_bin"
remote_driver=$remote_bin/driver
cat >"$remote_driver" <<'DRIVER'
#!/bin/sh
set -eu
while [ "$1" = -o ]; do shift 2; done
host=$1
shift
[ "$#" -eq 1 ] || exit 97
case $host in minix-amd-linux|minix-amd-wsl|rog-nv-linux|rog-nv-wsl) ;; *) exit 97 ;; esac
fixture_home=$SWEEP_TEST_REMOTE_ROOT/$host
[ -d "$fixture_home/ocannl-staging/.git" ] || exit 97
stage=work
case $1 in
  *'"$HOME"'*) stage=probe ;;
  *window-bounds*) stage=window ;;
  'cat /proc/sys/kernel/random/boot_id 2>/dev/null') stage=identity ;;
esac
if [ "$stage" = probe ]; then
  epoch=$(date +%s)
  echo "$epoch" >"$fixture_home/probe-epoch"
  awk -F '\t' -v epoch="$epoch" 'BEGIN {OFS="\t"} {$1 += epoch; print}' \
    "$fixture_home/journal-template" >"$fixture_home/journal"
fi
printf '%s\t%s\t%s\n' "$host" "$stage" "$(date +%s)" >>"$SWEEP_TEST_REMOTE_ROOT/transport"
case $SWEEP_TEST_REMOTE_CASE:$stage in
  replaced:window|unavailable:window) exit 255 ;;
  replaced:identity)
    attempts=$fixture_home/identity-attempts
    count=0
    [ ! -f "$attempts" ] || read -r count <"$attempts"
    count=$((count + 1))
    echo "$count" >"$attempts"
    [ "$count" -gt 2 ] || exit 255
    ;;
esac
# A harness may itself run inside a WSL sweep, whose PATH already contains
# these prefixes. Remove them before executing the generated command so only
# that command can supply the destination-specific prefix being asserted.
remote_path=
saved_ifs=$IFS
IFS=:
set -f
for path_entry in $PATH; do
  case $path_entry in /usr/local/cuda/bin|/usr/lib/wsl/lib|'') continue ;; esac
  remote_path=${remote_path}${remote_path:+:}$path_entry
done
IFS=$saved_ifs
# HOME changes only in this child environment; the controller never assigns it.
exec env HOME="$fixture_home" SWEEP_TEST_REMOTE_STAGE="$stage" \
  SWEEP_TEST_REMOTE_HOST="$host" PATH="$(dirname "$0"):$remote_path" /bin/sh -c "$1"
DRIVER
cat >"$remote_bin/cat" <<EOF_CAT
#!/bin/sh
if [ "\$#" -eq 1 ] && [ "\$1" = /proc/sys/kernel/random/boot_id ]; then
  case \$SWEEP_TEST_REMOTE_CASE:\$SWEEP_TEST_REMOTE_STAGE in
    replaced:identity|replaced-ok:window) echo fixture-new-boot ;;
    *) echo fixture-boot ;;
  esac
  exit 0
fi
exec "$(command -v cat)" "\$@"
EOF_CAT
cat >"$remote_bin/date" <<EOF_DATE
#!/bin/sh
# The fixture probe and its journal share one captured clock reading, avoiding
# a second boundary between fixture creation and the emitted probe's date call.
if [ "\$SWEEP_TEST_REMOTE_STAGE" = probe ] && [ "\$*" = +%s ]; then
  exec "$(command -v cat)" "\$HOME/probe-epoch"
fi
exec "$(command -v date)" "\$@"
EOF_DATE
# macOS has no util-linux flock: implement the emitted `flock -n 9` operation
# on the inherited descriptor with the real OS flock (no lock-result stub).
cat >"$remote_bin/flock" <<'EOF_FLOCK'
#!/bin/sh
[ "$*" = '-n 9' ] || exit 97
exec perl -e 'use Fcntl ":flock"; open(my $h, ">&=9") or exit 1; exit(flock($h, LOCK_EX | LOCK_NB) ? 0 : 1)'
EOF_FLOCK
cat >"$remote_bin/systemd-inhibit" <<'EOF_HOLD'
#!/bin/sh
while [ "$1" != -- ]; do shift; done
shift
exec "$@"
EOF_HOLD
cat >"$remote_bin/journalctl" <<'EOF_JOURNAL'
#!/bin/sh
set -eu
printf '%s\t%s\n' "$SWEEP_TEST_REMOTE_HOST" "$*" >>"$SWEEP_TEST_REMOTE_ROOT/journal-queries"
start=0
end=9999999999
probe=0
while [ "$#" -gt 0 ]; do
  case $1 in
    -b) echo 'fixture current-boot kernel entry'; exit 0 ;;
    --since) start=${2#@}; shift ;;
    --until) end=${2#@}; shift ;;
    -n) probe=1; shift ;;
  esac
  shift
done
awk -F '\t' -v start="$start" -v end="$end" -v probe="$probe" \
  '$1 >= start && $1 <= end { sub(/^[^\t]*\t/, ""); print; if (probe) exit }' "$HOME/journal"
EOF_JOURNAL
cat >"$remote_bin/dmesg" <<'EOF_DMESG'
#!/bin/sh
# An empty dxg journal follows the real ring fallback. Native collection must
# instead accept its current-boot journal probe.
printf '%s\t%s\n' "$SWEEP_TEST_REMOTE_HOST" "$*" >>"$SWEEP_TEST_REMOTE_ROOT/ring-queries"
exit 0
EOF_DMESG
cat >"$remote_bin/opam" <<EOF_OPAM
#!/bin/sh
set -eu
case \$1 in var) exit 1 ;; esac
# The scrubbing \`opam exec -- sh -c\` form: the fake opam plays opam's half and comes back here.
if [ "\$1" = exec ] && [ "\$#" -ge 4 ] && [ "\$3" = sh ] && [ "\$4" = -c ]; then
  SWEEP_TEST_OPAM_SELF=\$0 exec "$fake_bin/opam" "\$@"
fi
lock=\$HOME/ocannl-staging-worktrees/sweep.lock
# A fresh open must be refused: the real far-side lock remains held during
# clean, suite, serial rerun and completion, rather than only during prep.
perl -e 'use Fcntl ":flock"; open(my \$h, ">>", \$ARGV[0]) or exit 96; exit(flock(\$h, LOCK_EX | LOCK_NB) ? 96 : 0)' "\$lock" || exit 96
# Keep this unquoted heredoc body within the shell scanner's supported contract.
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "\$SWEEP_TEST_REMOTE_HOST" "\${OCANNL_BACKEND:--}" "\$(pwd)" "\$(git rev-parse HEAD)" "\$PATH" "\$*" >>"\$SWEEP_TEST_REMOTE_ROOT/commands"
[ -z "\$(git status --porcelain)" ] || exit 95
# Which of the box's slot files are flocked while this dune runs (gh-ocannl-1177): a GPU leg that
# took its token runs with that token's slot held, by its own leg's open file, not by a marker.
slot_dir=\$HOME/.local/state/fleet-execution-slots/\$SWEEP_TEST_REMOTE_HOST
if [ -d "\$slot_dir" ]; then
  printf '%s\t%s\n' "\$*" "\$(perl -e 'use Fcntl ":flock"; my @held;
    for my \$i (1 .. 4) { open(my \$h, ">>", "\$ARGV[0]/slot.\$i") or next;
      push @held, \$i unless flock(\$h, LOCK_EX | LOCK_NB) }
    print join(" ", @held)' "\$slot_dir")" >>"\$SWEEP_TEST_REMOTE_ROOT/token-states"
fi
exec "$fake_bin/opam" "\$@"
EOF_OPAM
# RTC diagnostics also stay within the fixture even on a host with GPU tools.
cat >"$remote_bin/rocminfo" <<'EOF_ROCM'
#!/bin/sh
echo 'Runtime Version: fixture'
echo 'Name: gfx_fixture'
EOF_ROCM
cat >"$remote_bin/nvidia-smi" <<'EOF_NVIDIA'
#!/bin/sh
echo 'fixture GPU, fixture driver, fixture compute capability'
EOF_NVIDIA
cat >"$remote_bin/ldd" <<'EOF_LDD'
#!/bin/sh
exit 1
EOF_LDD
chmod +x "$remote_bin"/*

remote_reset() {
  local host
  remote_root=$(mktemp -d "$tmp/remote.XXXXXX")
  remote_root=$(cd "$remote_root" && pwd -P)
  for host in minix-amd-linux minix-amd-wsl rog-nv-linux rog-nv-wsl; do
    mkdir -p "$remote_root/$host"
    git clone -q "$origin" "$remote_root/$host/ocannl-staging"
    # In-window and outside lines: the fixture query obeys the supplied bounds.
    printf '%s\t%s\n' 1 'kernel: misc dxg: dxgvmb_send_sync_msg: vmbus_sendpacket failed: fffffff5' \
      1 'kernel: amdgpu: process pid 4242 DQM create queue type 1 failed. ret -12' \
      -100 'kernel: misc dxg: dxgvmb_send_sync_msg: vmbus_sendpacket failed: old' \
      10000 'kernel: amdgpu: process pid 4243 DQM create queue type 1 failed. ret -12' \
      >"$remote_root/$host/journal-template"
  done
}
remote_sweep() {
  SWEEP_TEST_SSH_MODE=local SWEEP_TEST_REMOTE_DRIVER=$remote_driver \
    SWEEP_TEST_REMOTE_ROOT=$remote_root run_sweep_args "$@"
}
remote_row() { awk -F '\t' '$1 == "unit" && $2 == "minix" {print $4 ":" $9 ":" $10}' "$1"; }
remote_failure='File "test/dune", lines 1-4, characters 0-0:
1 | (rule
2 | (alias runtest-state-probe)
unit.exe: segmentation fault'

# Prep, forced clean, collection, serial rerun and completion through one path
# on both boots. No runtime name buys this rerun: only the kernel refusal can.
for boot in linux wsl; do
  remote_reset
  # The box's own environment carries credentials (gh-ocannl-1280): the fixture's remote shell
  # inherits these, as a box's login environment exports its token, and the switch's environment
  # sets them again inside `opam exec`. Every dune the legs' emitted text runs must be scrubbed
  # inside it -- clean, @check, suite, serial rerun, completion, RTC context.
  local_remote=$(SWEEP_TEST_HOSTS=$tmp/hosts-$boot.sh SWEEP_TEST_OPAM_RC=1 \
    GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key SWEEP_TEST_CREDENTIAL_PLAIN=kept \
    SWEEP_TEST_CREDENTIALS=$remote_root/credentials SWEEP_TEST_SWITCH_CREDENTIALS=1 \
    SWEEP_TEST_OPAM_OUT="$remote_failure" remote_sweep --only hip --force)
  local_record=$(sed -n 's/^run:  *//p' <<<"$local_remote")
  local_log=$(awk -F '\t' '$1 == "unit" && $2 == "minix" {print $6}' "$local_record")
  kind=native
  [ "$boot" != wsl ] || kind=dxg
  [ "$(remote_row "$local_record")" = "fail:1:$kind" ]
  grep -q '^  minix/hip: serial rerun: all clean$' <<<"$local_remote"
  grep -q '^  minix/hip: serial rerun: suite completed$' <<<"$local_remote"
  # A HIP unit whose suite ran asks for the WMMA probe (gh-ocannl-1658), and a swept ref without
  # tools/hip-wmma-probe.sh says so -- the controls after this loop give the fixture one.
  grep -q '^  minix/hip: wmma probe: not at ' <<<"$local_remote"
  grep -q '^=== .* window: 1 ' "$(window_sidecar "$local_log")"
  grep -q '^=== serial rerun @test/runtest-state-probe: exit 0 ===$' "$local_log"
  grep -q '^=== suite completion: exit 0 ===$' "$local_log"
  # The cleared unit's per-action verdict records came back over the same
  # transport, from beside the far-side worktree (gh-ocannl-1114).
  [ -d "$remote_root/minix-amd-$boot/ocannl-staging-worktrees/sweep.verdict-records" ]
  grep -q '^OCANNL_TOOL_VERDICT_ACTION' "${local_log%.log}.verdict-records"
  absent 'skip evidence unavailable' <<<"$local_remote"
  [ "$(git -C "$remote_root/minix-amd-$boot/ocannl-staging-worktrees/sweep" rev-parse HEAD)" = \
    "$(git -C "$main" rev-parse HEAD)" ]
  [ "$(awk -F '\t' '$1 == "minix-amd-'"$boot"'" && $6 == "exec -- dune clean" {n++} END {print n+0}' "$remote_root/commands")" -eq 1 ]
  jobs=$(box_jobs_sweep_jobs minix hip minix-amd-$boot)
  awk -F '\t' -v host="minix-amd-$boot" -v jobs="$jobs" -v sha="$(git -C "$main" rev-parse HEAD)" -v cwd="$remote_root/minix-amd-$boot/ocannl-staging-worktrees/sweep" \
    '$1 == host {if ($4 != sha || $3 != cwd) exit 1; if ($6 == "exec -- dune build -j " jobs " --force @runtest @train") suite++; if ($6 == "exec -- dune build -j 1 @test/runtest-state-probe") serial++; if ($6 == "exec -- dune build -j 1 @runtest @train") completion++} END {if (suite != 1 || serial != 1 || completion != 1) exit 1}' "$remote_root/commands"
  prefix=/usr/local/cuda/bin:
  [ "$boot" != wsl ] || prefix=$prefix/usr/lib/wsl/lib:
  awk -F '\t' -v prefix="$prefix" '$2 == "hip" {if (index($5, prefix) != 1) exit 1; seen++} END {if (!seen) exit 1}' "$remote_root/commands"
  if [ "$boot" = linux ]; then absent '/usr/lib/wsl/lib' "$remote_root/commands"; fi
  # After this boot's other assertions, so the controls below still meet their own oracles first.
  credential_free "$remote_root/credentials" 6
  grep -q '^exec -- dune clean	' "$remote_root/credentials"
  grep -q '^exec -- dune build @arrayjit/test/runtest-test_hip_compile_options --force	' \
    "$remote_root/credentials"
  grep -q '^exec -- dune build -j 1 @runtest @train	' "$remote_root/credentials"
  # Dirty reused worktree is reset, including untracked strays.
  remote_wt=$remote_root/minix-amd-$boot/ocannl-staging-worktrees/sweep
  echo dirty >"$remote_wt/fixture"
  echo stray >"$remote_wt/untracked"
  local_remote=$(SWEEP_TEST_HOSTS=$tmp/hosts-$boot.sh SWEEP_TEST_JOBS=3 \
    remote_sweep --only multidev_cc --target remote-width)
  grep -q '^  minix/multidev_cc: incremental-pass ' <<<"$local_remote"
  [ ! -e "$remote_wt/untracked" ]
  [ -z "$(git -C "$remote_wt" status --porcelain)" ]
  grep -q 'exec -- dune runtest -j 3 remote-width$' "$remote_root/commands"
done

# The WMMA probe's verdict is the box's, never the unit's (gh-ocannl-1658): through the same remote
# path, in the unit's own worktree, a stand-in at the swept ref prints the real script's verdict
# line, which is said on the unit's summary -- a broken same-sign rule as a WARNING -- while the
# unit's row stays its suite's outcome either way; a CPU unit on the same box asks for no probe.
mkdir -p "$main/tools"
cat >"$main/tools/hip-wmma-probe.sh" <<'EOF_WMMA'
#!/bin/sh
echo "probe ran in $PWD"
case ${SWEEP_TEST_WMMA_RC:-0} in
  0) echo 'hip-wmma-probe: verdict: holds (gfx-fixture)' ;;
  1) echo 'hip-wmma-probe: verdict: SAME-SIGN RULE BROKE at -O3 (gfx-fixture)' ;;
esac
exit "${SWEEP_TEST_WMMA_RC:-0}"
EOF_WMMA
git -C "$main" add tools/hip-wmma-probe.sh
git -C "$main" commit -qm 'stand-in wmma probe'
git -C "$main" push -q origin master
remote_reset
wmma_holds=$(remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$wmma_holds")
[ "$(remote_row "$local_record" | cut -d: -f1)" = incremental-pass ]
grep -q "^  minix/hip: wmma probe: holds (gfx-fixture) -- $state/logs/[^ ]*-minix-hip-wmma-probe.log\$" \
  <<<"$wmma_holds"
wmma_log=$(sed -n 's/^  minix\/hip: wmma probe: holds (gfx-fixture) -- //p' <<<"$wmma_holds")
grep -qx "probe ran in $remote_root/minix-amd-linux/ocannl-staging-worktrees/sweep" "$wmma_log"
wmma_broke=$(SWEEP_TEST_WMMA_RC=1 remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$wmma_broke")
[ "$(remote_row "$local_record" | cut -d: -f1)" = incremental-pass ]
grep -q '^  minix/hip: WARNING -- wmma probe: SAME-SIGN RULE BROKE at -O3 (gfx-fixture): the bitwise HIP tensorized claims rest on that rule ' \
  <<<"$wmma_broke"
wmma_broke=$(SWEEP_TEST_WMMA_RC=5 remote_sweep --only hip --target test/operations)
grep -q '^  minix/hip: wmma probe: could not run (exit 5) -- ' <<<"$wmma_broke"
wmma_cpu=$(remote_sweep --only multidev_cc --target remote-cpu)
grep -q '^  minix/multidev_cc: incremental-pass ' <<<"$wmma_cpu"
absent 'wmma probe' <<<"$wmma_cpu"
git -C "$main" rm -q tools/hip-wmma-probe.sh
git -C "$main" commit -qm 'drop the stand-in wmma probe'
git -C "$main" push -q origin master

# Quiet kernel and no runtime name: test failure buys no serial call. A CPU
# remote unit has no window and uses the default width.
remote_reset
: >"$remote_root/minix-amd-linux/journal-template"
local_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$remote_failure" \
  remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_control")
[ "$(remote_row "$local_record")" = fail:0:native ]
absent 'serial rerun:' <<<"$local_control"
absent ' -j 1 ' "$remote_root/commands"
local_remote=$(remote_sweep --only multidev_cc --target remote-cpu)
grep -q '^  minix/multidev_cc: incremental-pass ' <<<"$local_remote"
grep -q 'exec -- dune runtest remote-cpu$' "$remote_root/commands"
local_record=$(sed -n 's/^run:  *//p' <<<"$local_remote")
[ "$(remote_row "$local_record")" = incremental-pass:-:- ]

# Name, status-qualified name and runtime-assertion decisions take the same
# remote command path even when the collected window is clean. A launch status
# that describes a test defect stays red without buying the rerun.
for trigger in name status assertion logic-status; do
  case $trigger in
    name) diagnostic='Fatal error: exception hip_init: HIP_ERROR_UNKNOWN' ;;
    status) diagnostic='Fatal error: exception cu_launch_kernel:
CUDA_ERROR_OUT_OF_MEMORY' ;;
    assertion) diagnostic="unit.exe: runtime.cpp:2003: rocr::AMD::GpuAgent::ReleaseQueueMainScratch(): Assertion scratch.main_queue_base failed." ;;
    logic-status) diagnostic='Fatal error: exception cu_launch_kernel: CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES' ;;
  esac
  local_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$remote_failure
$diagnostic" remote_sweep --only hip --target test/operations)
  if [ "$trigger" = logic-status ]; then
    absent 'serial rerun:' <<<"$local_control"
  else
    grep -q '^  minix/hip: serial rerun: all clean$' <<<"$local_control"
  fi
done
# Successful collection from a different boot reaches the replacement arm of
# environment_red, distinct from recovery after a refused collection.
local_control=$(SWEEP_TEST_REMOTE_CASE=replaced-ok SWEEP_TEST_OPAM_RC=1 \
  SWEEP_TEST_OPAM_OUT="$remote_failure" remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_control")
[ "$(remote_row "$local_record")" = fail:vm-replaced:native ]
grep -q '^  minix/hip: serial rerun: all clean$' <<<"$local_control"
# Failure to collect from an unchanged guest certifies nothing; unavailable is
# recorded, and the same unnamed userspace failure gets no environment rerun.
local_control=$(SWEEP_TEST_REMOTE_CASE=unavailable SWEEP_TEST_OPAM_RC=1 \
  SWEEP_TEST_OPAM_OUT="$remote_failure" remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_control")
[ "$(remote_row "$local_record")" = fail:unavailable:native ]
absent 'serial rerun:' <<<"$local_control"

# Failed collection, two refused identity connections, then recovered guest:
# one expensive collection, three cheap probes spaced by the real pauses.
remote_reset
local_replaced=$(SWEEP_TEST_REMOTE_CASE=replaced SWEEP_TEST_OPAM_RC=255 \
  SWEEP_TEST_HOSTS=$tmp/hosts-wsl.sh remote_sweep --only cuda --target remote-replaced)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_replaced")
[ "$(awk -F '\t' '$1 == "unit" {print $4 ":" $9 ":" $10}' "$local_record")" = error:vm-replaced:dxg ]
grep -q 'exec -- dune runtest remote-replaced$' "$remote_root/commands"
[ "$(awk -F '\t' '$2 == "window" {n++} END {print n+0}' "$remote_root/transport")" -eq 1 ]
[ "$(awk -F '\t' '$2 == "identity" {n++} END {print n+0}' "$remote_root/transport")" -eq 3 ]
awk -F '\t' '$2 == "identity" {if (n++ && $3 - last < 10) exit 1; last=$3} END {if (n != 3) exit 1}' "$remote_root/transport"
grep -q 'the guest was REPLACED mid-unit' <<<"$local_replaced"

# Real held remote worktree lock refuses prep before opam; release permits reuse.
remote_reset
mkdir -p "$remote_root/minix-amd-linux/ocannl-staging-worktrees"
exec 8>"$remote_root/minix-amd-linux/ocannl-staging-worktrees/sweep.lock"
perl -e 'use Fcntl ":flock"; exit(flock(STDIN, LOCK_EX | LOCK_NB) ? 0 : 1)' <&8
local_busy=$(remote_sweep --only multidev_cc --target remote-busy)
exec 8>&-
grep -q '^  minix/multidev_cc: error (cannot pin ' <<<"$local_busy"
[ ! -e "$remote_root/commands" ]
local_remote=$(remote_sweep --only multidev_cc --target remote-busy)
grep -q '^  minix/multidev_cc: incremental-pass ' <<<"$local_remote"
# Broken Git remote fails prep through its real fetch, still collects evidence.
git -C "$remote_root/minix-amd-linux/ocannl-staging" remote set-url origin "$tmp/absent-origin"
local_prep_error=$(remote_sweep --only hip --target remote-prep-error)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_prep_error")
[ "$(remote_row "$local_record")" = error:1:native ]
grep -q '^  minix/hip: error (cannot pin ' <<<"$local_prep_error"

# The per-test cap (gh-ocannl-1211) on the far side: test_cmd's watcher signals the `sh -c` running
# its text's parent ("$PPID"), which is the unit's supervisor only while remote_capped starts that
# `sh -c` as the supervisor's direct child; the supervisor's 142 must then cross the ssh hop to read
# as `timeout`, and the watcher's record line reach the unit's log. The local metal leg in
# sweep_harness.sh pins the same for capped_bg. Skipped where a `$0`-renamed process does not read
# back as `dune` in ps, as that leg is.
remote_hang_probe=$(perl -e '$0 = "dune"; print qx{ps -o comm= -p $$}' 2>/dev/null)
case $remote_hang_probe in
  *dune*)
    remote_reset
    : >"$remote_root/minix-amd-linux/journal-template"
    remote_hang_pid=$tmp/remote-hung-test.pid
    rm -f "$remote_hang_pid"
    local_remote=$(SWEEP_TEST_OPAM_HANG=60 SWEEP_TEST_OPAM_HANG_PID=$remote_hang_pid \
      SWEEP_TEST_TEST_CAP=2 remote_sweep --only hip)
    grep -q '^  minix/hip: timeout ' <<<"$local_remote"
    grep -q "^  minix/hip: hung test, past the per-test cap: pid $(cat "$remote_hang_pid") has run [0-9]*s, past the 2s cap: sleep 60" \
      <<<"$local_remote"
    if hung_test_survives "$remote_hang_pid" remote; then exit 1; fi
    ;;
  *) printf 'sweep harness: remote per-test cap leg skipped: ps does not read a renamed process as dune here\n' >&2 ;;
esac

# The device probe on the far side (gh-ocannl-1252): test_cmd's probe runs where the worktree is,
# from its test/config, before any test leg, and its 69 crosses the ssh hop to read as `error`
# named DEVICE UNHEALTHY, the suite never started there.
remote_reset
: >"$remote_root/minix-amd-linux/journal-template"
: >"$tmp/probe.calls"
local_remote=$(SWEEP_TEST_DEVICE_PROBE=$fake_bin/fake-device-probe SWEEP_TEST_DEVICE_PROBE_CAP=1 \
  SWEEP_TEST_PROBE_HANG_AT=1 remote_sweep --only hip)
grep -q '^  minix/hip: error ' <<<"$local_remote"
grep -q '^  minix/hip: DEVICE UNHEALTHY, dune not started: a fresh device query for hip did not answer within 1s$' \
  <<<"$local_remote"
grep -qx "$remote_root/minix-amd-linux/ocannl-staging-worktrees/sweep/test/config --ocannl_backend=hip" "$tmp/probe.calls"
absent '@runtest' "$remote_root/commands"

# The fleet's GPU tokens (gh-ocannl-1177). rog-nv-linux's fake fleet-worker.sh answers four slots
# and two tokens, so its cuda unit takes one around every GPU leg -- a REAL flock on the box's slot
# file, which the remote opam above sees held while each dune runs -- while minix's four and four
# bind nothing and its hip unit takes none. Each unit also records the box's token holders at its
# start and after its suite, as the box's `fleet-worker.sh execution slot --holders` lists them
# (gh-ocannl-1417): the fake one tests the slot files' real flocks and names the holder each fixture
# batch records beside its slot file; sweep_components.sh holds the reader to the fleet's format.
token_holder() { # slot-file -- a batch holding that slot until killed; its pid in token_holder_pid
  perl -e 'use Fcntl ":flock"; open(my $h, ">>", $ARGV[0]) or die; flock($h, LOCK_EX) or die;
    open(my $p, ">", "$ARGV[0].holder") or die; print $p "$$\n"; close $p;
    open(my $r, ">", "$ARGV[0].ready") or die; close $r; sleep 120' "$1" &
  token_holder_pid=$!
  local waited=0
  until [ -e "$1.ready" ]; do
    [ "$waited" -lt "$wait_ticks" ] || break
    sleep 0.05
    waited=$((waited + 1))
  done
  rm -f "$1.ready"
}
remote_reset
rog_slots=$remote_root/rog-nv-linux/.local/state/fleet-execution-slots/rog-nv-linux
mkdir -p "$rog_slots"
token_holder "$rog_slots/slot.1"
holder_one=$token_holder_pid
: >"$fleet_calls"
# One worker batch holds the first token: the unit takes the second around its suite, its serial
# rerun (the native window's refusal makes it environment-red) and its completion pass, and its
# summary, record and fingerprint attribute the contention to that batch.
local_tokens=$(SWEEP_TEST_OPAM_RC=1 \
  SWEEP_TEST_OPAM_OUT="$remote_failure" remote_sweep --only cuda --force)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_tokens")
local_log=$(awk -F '\t' '$1 == "unit" && $3 == "cuda" {print $6}' "$local_record")
grep -q '^  rog-nv/cuda: fail ' <<<"$local_tokens"
grep -q '^  rog-nv/cuda: serial rerun: all clean$' <<<"$local_tokens"
grep -q '^  rog-nv/cuda: serial rerun: suite completed$' <<<"$local_tokens"
grep -q "^  rog-nv/cuda: gpu tokens: 1 of 2 held by other batches at opened on rog-nv-linux -- slot 1 pid $holder_one " \
  <<<"$local_tokens"
grep -q "^  rog-nv/cuda: gpu tokens: 1 of 2 held by other batches at closed on rog-nv-linux -- slot 1 pid $holder_one " \
  <<<"$local_tokens"
[ "$(head -1 "$local_record")" = "$(printf 'schema\t6')" ]
[ "$(awk -F '\t' '$1 == "unit" {print $12}' "$local_record")" = 'opened=1/2,closed=1/2' ]
grep -qx 'gpu tokens: other batches held the GPU' "${local_log%.log}.fingerprint"
[ "$(grep -c '^execution slot --gpu --wait 1800 -- sh -c ' "$fleet_calls")" -eq 3 ]
[ "$(grep -cx 'sweep: gpu token: EXECUTION SLOT rog-nv-linux: slot 2 of 4, GPU token 2 of 2 held' "$local_log")" -eq 3 ]
# ...and nothing of the command it held it for, whose text carries the rerun's own verdict echoes.
absent '^sweep: gpu token: .*serial rerun' "$local_log"
# Each GPU leg's dune ran with the unit's token held beside the worker's; the RTC context, which
# runs no GPU test, took none.
awk -F '\t' -v holds='1 2' '
  $1 ~ / --force @runtest @train$/ || $1 ~ / -j 1 @test\/runtest-state-probe$/ || $1 ~ / -j 1 @runtest @train$/ {
    if ($2 != holds) exit 1; legs++ }
  $1 ~ /test_cuda_compile_options/ { if ($2 != "1") exit 1; rtc++ }
  END { if (legs != 3 || rtc != 1) exit 1 }' "$remote_root/token-states"
# ...and released it: the slot is free again once the unit is done.
perl -e 'use Fcntl ":flock"; open(my $h, ">>", $ARGV[0]) or exit 1; exit(flock($h, LOCK_EX | LOCK_NB) ? 0 : 1)' \
  "$rog_slots/slot.2"
# The sidecar is the fleet's co-tenant vocabulary: box, event, at; pid, user, command per holder.
perl -MJSON::PP -e 'my $pid = shift; while (my $l = <>) { my $r = JSON::PP->new->decode($l);
    next unless $r->{event} eq "opened"; my ($h) = @{ $r->{token_holders} };
    exit !($r->{box} eq "rog-nv-linux" && $r->{at} =~ /\+00:00$/ && $h->{pid} == $pid
      && $h->{kind} eq "gpu" && $h->{gpu} && $h->{slot} == 1 && length $h->{command}) } exit 1' \
  "$holder_one" "${local_log%.log}.gpu-tokens"

# Both tokens held past the unit's wait: a `skip` naming the holders, and no suite started.
token_holder "$rog_slots/slot.2"
holder_two=$token_holder_pid
: >"$remote_root/commands"
local_tokens=$(SWEEP_TEST_TOKEN_WAIT=1 \
  remote_sweep --only cuda --target token-busy)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_tokens")
grep -q '^  rog-nv/cuda: skip (no GPU token on rog-nv-linux: no GPU token came free within 1s; [0-9]*s)$' \
  <<<"$local_tokens"
grep -q "^  rog-nv/cuda: gpu tokens: 2 of 2 held by other batches at refused on rog-nv-linux -- slot 1 pid $holder_one .*; slot 2 pid $holder_two " \
  <<<"$local_tokens"
[ "$(awk -F '\t' '$1 == "unit" {print $4 ":" $12}' "$local_record")" = 'skip:opened=2/2,refused=2/2' ]
absent 'token-busy' "$remote_root/commands"
kill "$holder_one" "$holder_two" 2>/dev/null || true
wait "$holder_one" "$holder_two" 2>/dev/null || true

# A take the fleet's machinery cannot answer (an anchor whose registry did not) fails OPEN, as the
# sweep's registry read does: the leg runs without a token, and the summary says so.
local_tokens=$(SWEEP_TEST_SLOT_UNREACHABLE=1 \
  remote_sweep --only cuda --target token-unreachable)
grep -q '^  rog-nv/cuda: incremental-pass ' <<<"$local_tokens"
grep -qF '  rog-nv/cuda: gpu token: WARNING: not taken (execution slot exited 4), so this leg runs WITHOUT one' \
  <<<"$local_tokens"
grep -q 'token-unreachable$' "$remote_root/commands"

# Where the tokens do not bind (minix: four and four), the unit records its readings and takes none.
: >"$fleet_calls"
local_tokens=$(remote_sweep --only hip --target token-free)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_tokens")
[ "$(awk -F '\t' '$1 == "unit" && $2 == "minix" {print $12}' "$local_record")" = 'opened=0/4,closed=0/4' ]
absent -- '--gpu' "$fleet_calls"
absent 'gpu tokens:' <<<"$local_tokens"

# A system sleep inside a native unit's window (gh-ocannl-1039): the boot id survives it, the
# power-management core's lines in the window do not. A red unit is environment-red for it and gets
# its serial rerun; a unit the sleep ended unjudged is said to be about the box.
remote_reset
printf '%s\t%s\n' 1 'kernel: PM: suspend entry (s2idle)' 1 'kernel: PM: suspend exit' \
  1 'kernel: amdgpu 0000:c5:00.0: ring sdma0 uses VM inv eng 12 on hub 0' \
  >"$remote_root/minix-amd-linux/journal-template"
local_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$remote_failure" \
  remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_control")
[ "$(remote_row "$local_record")" = fail:suspended:native ]
grep -q '^  minix/hip: serial rerun: all clean$' <<<"$local_control"
git -C "$remote_root/minix-amd-linux/ocannl-staging" remote set-url origin "$tmp/absent-origin"
local_control=$(remote_sweep --only hip --target remote-slept)
grep -q '^  minix/hip: error (cannot pin ' <<<"$local_control"
grep -q '^  minix/hip: the box SUSPENDED mid-unit' <<<"$local_control"
# ...but the dGPU's own runtime resume, which re-prints the same ring table all day, is no sleep.
remote_reset
printf '%s\t%s\n' 1 'kernel: amdgpu 0000:c5:00.0: ring sdma0 uses VM inv eng 12 on hub 0' \
  >"$remote_root/minix-amd-linux/journal-template"
local_control=$(SWEEP_TEST_OPAM_RC=1 SWEEP_TEST_OPAM_OUT="$remote_failure" \
  remote_sweep --only hip --target test/operations)
local_record=$(sed -n 's/^run:  *//p' <<<"$local_control")
[ "$(remote_row "$local_record")" = fail:0:native ]
absent 'serial rerun' <<<"$local_control"

# The former SSH fake and broken shipping commands must lose these oracles.
# Children isolate the remote cases and never recurse into the controls.
if [ "$remote_only" = 0 ]; then
  # The children run under this very bash, by path: on Windows a bare `bash` from Python resolves
  # to System32's WSL launcher, not Git Bash.
  this_bash=$BASH
  if command -v cygpath >/dev/null 2>&1; then this_bash=$(cygpath -w "$BASH"); fi
  remote_hang_legs=0
  case $remote_hang_probe in *dune*) remote_hang_legs=1 ;; esac
  python3 - "$this_bash" "$remote_hang_legs" "$0" "$sweep" "$aggregate" "$verdict_probe" "$tmp/remote-controls" <<'PY_CONTROLS'
from pathlib import Path
import re
import shutil
import subprocess
import sys
this_bash = sys.argv[1]
remote_hang_legs = sys.argv[2] == '1'
harness, sweep, aggregate, verdict, scratch = map(lambda p: Path(p).resolve(), sys.argv[3:])
controls = [
    ('old-ssh', 'harness', 'local) exec "$SWEEP_TEST_REMOTE_DRIVER" "$@" ;;', 'local) exit 1 ;;', 'remote_row'),
    ('no-evidence', 'sweep', '  window_red "$1" && return 0', '  : # lost evidence arm', 'serial rerun: all clean'),
    ('no-lock', 'sweep', '''  printf 'mkdir -p "$(dirname "%s")" && exec 9>"%s.lock" && flock -n 9 || exit 126; ' "$1" "$1"''', "  printf ':; '", 'remote_row'),
    ('no-retry', 'sweep', '    [ -n "$GUEST_ID" ] && return 0', '    return 0 # lost identity retry', 'error:vm-replaced:dxg'),
    ('no-credential-scrub', 'sweep', '''  printf 'opam exec -- sh -c %s sweep-dune dune' \\''', "  printf 'opam exec -- dune'; : \\", 'credentials reached dune'),
    # A suite leg that skips its GPU token (gh-ocannl-1177): the rerun's legs still take one, so
    # only the count of held legs can tell.
    ('suite-without-token', 'sweep', '''    [ -z "$TOKEN_TAKE" ] || remote=$(gpu_token_take_cmd "$TOKEN_FW" "$TOKEN_WAIT" "$wt.gpu-token" "$remote")''', '    :', 'execution slot --gpu --wait 1800'),
]
# A shell between the far-side supervisor and test_cmd's `sh -c`: its watcher's "$PPID" is then
# that shell, so the hung test outlives the unit. Only the remote per-test cap leg sees it, and
# only where that leg runs.
if remote_hang_legs:
    controls.append(('remote-cap-not-parent', 'sweep-supervisor.sh',
        '''    printf 'perl -e %s -- --hold %s %s sh -c %s' "$(sq "$capped_perl")" "$(sq "$3")" "$1" "$(sq "$2")"''',
        '''    printf 'perl -e %s -- --hold %s %s sh -c %s %s' "$(sq "$capped_perl")" "$(sq "$3")" "$1" "$(sq 'sh -c "$0"; exit $?')" "$(sq "$2")"''',
        'remote hung test outlived'))
# Derive sourced helpers from the sweep rather than carrying a second list.
helpers = re.findall(r'^\. "\$SWEEP_TOOLS/([^"\n]+)"$', sweep.read_text(), re.M)
if not helpers:
    raise RuntimeError('no sweep helpers discovered')
for name, target, old, new, oracle in controls:
    dest = scratch / name
    for directory in ['test/operations', 'scripts', 'tools', 'benchmarks']:
        (dest / directory).mkdir(parents=True)
    for helper in [sweep.name] + helpers:
        shutil.copy2(sweep.parent / helper, dest / 'tools' / helper)
    shutil.copy2(aggregate, dest / 'tools' / aggregate.name)
    shutil.copy2(sweep.parent.parent / 'benchmarks/fixture_digest.py', dest / 'benchmarks/fixture_digest.py')
    for file in [harness, harness.parent / 'sweep_remote_fixture.sh']:
        shutil.copy2(file, dest / 'test/operations' / file.name)
    for support in ['harness-support.sh', 'process-group.sh']:
        shutil.copy2(harness.parent.parent.parent / 'scripts' / support, dest / 'scripts' / support)
    # A target is the harness, the sweep, or one of the helpers it sources, by file name.
    file = dest / ('test/operations/' + harness.name if target == 'harness'
                   else 'tools/' + (sweep.name if target == 'sweep' else target))
    text = file.read_text()
    if text.count(old) != 1:
        raise RuntimeError('ambiguous mutation anchor: ' + name)
    file.chmod(file.stat().st_mode | 0o200)
    file.write_text(text.replace(old, new))
    command = [this_bash, (dest / 'test/operations' / harness.name).as_posix(), '--remote-only',
               (dest / 'tools' / sweep.name).as_posix(), aggregate.as_posix(), verdict.as_posix(),
               'fixture', 'fixture', 'fixture']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (scratch / (name + '.log')).write_text(result.stdout)
    if result.returncode != 1 or oracle not in result.stdout:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f'{name}: expected oracle rejection, got {result.returncode}')
    print(f'local remote negative control {name}: caught', flush=True)
PY_CONTROLS
fi

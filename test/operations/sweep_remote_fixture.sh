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
lock=\$HOME/ocannl-staging-worktrees/sweep.lock
# A fresh open must be refused: the real far-side lock remains held during
# clean, suite, serial rerun and completion, rather than only during prep.
perl -e 'use Fcntl ":flock"; open(my \$h, ">>", \$ARGV[0]) or exit 96; exit(flock(\$h, LOCK_EX | LOCK_NB) ? 96 : 0)' "\$lock" || exit 96
# Keep this unquoted heredoc body within the shell scanner's supported contract.
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "\$SWEEP_TEST_REMOTE_HOST" "\${OCANNL_BACKEND:--}" "\$(pwd)" "\$(git rev-parse HEAD)" "\$PATH" "\$*" >>"\$SWEEP_TEST_REMOTE_ROOT/commands"
[ -z "\$(git status --porcelain)" ] || exit 95
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
  local_remote=$(SWEEP_TEST_HOSTS=$tmp/hosts-$boot.sh SWEEP_TEST_OPAM_RC=1 \
    SWEEP_TEST_OPAM_OUT="$remote_failure" remote_sweep --only hip --force)
  local_record=$(sed -n 's/^run:  *//p' <<<"$local_remote")
  local_log=$(awk -F '\t' '$1 == "unit" && $2 == "minix" {print $6}' "$local_record")
  kind=native
  [ "$boot" != wsl ] || kind=dxg
  [ "$(remote_row "$local_record")" = "fail:1:$kind" ]
  grep -q '^  minix/hip: serial rerun: all clean$' <<<"$local_remote"
  grep -q '^  minix/hip: serial rerun: suite completed$' <<<"$local_remote"
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

# The former SSH fake and broken shipping commands must lose these oracles.
# Children isolate the remote cases and never recurse into the controls.
if [ "$remote_only" = 0 ]; then
  python3 - "$0" "$sweep" "$aggregate" "$verdict_probe" "$tmp/remote-controls" <<'PY_CONTROLS'
from pathlib import Path
import re
import shutil
import subprocess
import sys
harness, sweep, aggregate, verdict, scratch = map(lambda p: Path(p).resolve(), sys.argv[1:])
controls = [
    ('old-ssh', 'harness', 'local) exec "$SWEEP_TEST_REMOTE_DRIVER" "$@" ;;', 'local) exit 1 ;;', 'remote_row'),
    ('no-evidence', 'sweep', '  window_red "$1" && return 0', '  : # lost evidence arm', 'serial rerun: all clean'),
    ('no-lock', 'sweep', '''  printf 'mkdir -p "$(dirname "%s")" && exec 9>"%s.lock" && flock -n 9 || exit 126; ' "$1" "$1"''', "  printf ':; '", 'remote_row'),
    ('no-retry', 'sweep', '    [ -n "$GUEST_ID" ] && return 0', '    return 0 # lost identity retry', 'error:vm-replaced:dxg'),
]
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
    shutil.copy2(harness.parent.parent.parent / 'scripts/harness-support.sh', dest / 'scripts/harness-support.sh')
    file = dest / ('test/operations/' + harness.name if target == 'harness' else 'tools/' + sweep.name)
    text = file.read_text()
    if text.count(old) != 1:
        raise RuntimeError('ambiguous mutation anchor: ' + name)
    file.chmod(file.stat().st_mode | 0o200)
    file.write_text(text.replace(old, new))
    command = ['bash', str(dest / 'test/operations' / harness.name), '--remote-only',
               str(dest / 'tools' / sweep.name), str(aggregate), str(verdict), 'fixture', 'fixture', 'fixture']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (scratch / (name + '.log')).write_text(result.stdout)
    if result.returncode != 1 or oracle not in result.stdout:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f'{name}: expected oracle rejection, got {result.returncode}')
    print(f'local remote negative control {name}: caught', flush=True)
PY_CONTROLS
fi

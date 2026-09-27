#!/bin/bash
# gh-ocannl-833 / gh-ocannl-834 measurement driver: the same bounded steps on every box, so the
# issue comments quote invocations rather than restating commands.
#
# Usage (from anywhere; the checkout is the one this script lives in):
#   benchmarks/gh834_cells.sh BACKEND OUT FIXTURE CAP STEP...
#
#   BACKEND  cc | hip | cuda | metal, pinned on every command line (out-ranks every other source).
#   OUT      results directory, created. OUT/driver.log records everything this script prints
#            (provenance, the device state around each step, each step's exit and wall); each
#            measurement step also writes OUT/<step>.out and OUT/<step>.err.
#   FIXTURE  absolute path of a gpt2_mini.safetensors (only the session steps read it).
#   CAP      wall cap in seconds for the build and for each measurement step; a capped step's
#            whole process group is terminated (exit 124) and its trace lines stand
#            as a lower bound.
#   STEP     build | provenance | crown-fwd | crown-rev | session-isolated | session-queued
#
# crown-{fwd,rev}: gh-ocannl-833's instrument, bin/projection_shape_bench.exe 200 8 d <order>
#   seeds -- the out-projection sites (group d), every seeded candidate ranked by the batched
#   median, by Autotune.time_routine ~timing:Isolated and by ~timing:Queued; the closing
#   "gh-ocannl-755" table says whether the crown moves.
# session-<mode>: gh-ocannl-834's instrument, one gpt2_mini tuned search (BENCH_TUNE=1, both
#   placement arms) under autotune_timing=<mode> on a COLD schedule cache (OUT/cache-<mode>, wiped
#   first), with BENCH_TIMING_TRACE=1 splitting the wall between candidate timing and the rest.
#   compile_s in the result line is the whole search's wall.
set -u
# Hermetic against ambient configuration, as gh612_cells.sh is: every treatment is pinned on the
# command line, so an exported OCANNL_* or BENCH_* could only contaminate every cell consistently.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')

[ $# -ge 5 ] || { sed -n '5,13p' "$0" >&2; exit 2; }
backend=$1 out=$2 fixture=$3 cap=$4
shift 4
case $backend in cc | hip | cuda | metal) ;; *) echo "gh834: unknown backend $backend" >&2; exit 2 ;; esac
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P)
# Everything printed from here on is also kept with the results it describes.
exec > >(tee -a "$out/driver.log") 2>&1
root=$(cd "$(dirname "$0")/.." && pwd -P)
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2
# Every artifact is labelled by the commit, so the tracked tree must BE that commit: a locally edited
# benchmarks/ocannl_config (autotune_repeats, a profile, a placement key) would change every cell
# while the label stayed the same. The effective config file is recorded as well as checked.
dirty=$(git -C "$root" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
  echo "gh834: refusing to measure from a tree with tracked changes:" >&2
  echo "$dirty" >&2
  exit 2
fi
echo "config benchmarks/ocannl_config sha256 $(shasum -a 256 ocannl_config | cut -d' ' -f1):" \
  "$(grep -v '^#' ocannl_config | grep -v '^$' | tr '\n' ' ')"

# A portable wall cap over the step's whole PROCESS GROUP (macOS has no timeout(1) or setsid(1)):
# a cc candidate's compiler is a child of the runner, and a compile outliving its capped runner
# would load every later cell. The runner leads its own group; at the cap the group gets SIGTERM,
# then SIGKILL after 10 s, and the step waits until no member is left before the next one starts
# (exit 124; 125 if a member survives even SIGKILL, and the driver stops rather than measure beside
# it). A member left behind by a runner that exited on its own is reaped the same way. SIGTERM
# ends an OCaml runner without at_exit -- which is why the trace carries running totals.
capped() {
  perl -e '
    use POSIX ();
    my $cap = shift;
    my $pid = fork // die "gh834: fork: $!\n";
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV or POSIX::_exit(127) }
    setpgrp($pid, $pid);
    my $timed_out = 0;
    # TERM at the cap, KILL 10 s later if the leader is still not reaped.
    $SIG{ALRM} = sub {
      if ($timed_out++) { kill "KILL", -$pid } else { kill "TERM", -$pid; alarm 10 }
    };
    alarm $cap;
    my ($got, $st);
    do { $got = waitpid($pid, 0); $st = $? } until $got == $pid || ($got == -1 && !$!{EINTR});
    alarm 0;
    my $alive = sub { waitpid($pid, POSIX::WNOHANG()); kill 0, -$pid };
    if ($alive->()) {
      kill "TERM", -$pid;
      for (1 .. 10) { last unless $alive->(); sleep 1 }
      kill "KILL", -$pid if $alive->();
      for (1 .. 10) { last unless $alive->(); sleep 1 }
      if ($alive->()) { print STDERR "gh834: process group $pid survived SIGKILL\n"; exit 125 }
    }
    exit 124 if $timed_out;
    exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
  ' "$cap" "$@"
}

device_state() {
  echo "-- $(date -u +%FT%TZ) $(uptime)"
  case $backend in
  hip) for f in /sys/class/drm/card*/device/gpu_busy_percent; do [ -r "$f" ] && echo "   gpu_busy_percent $(cat "$f")"; done ;;
  cuda) nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/   nvidia-smi /' ;;
  metal) ps -Ao %cpu= | awk '{s+=$1} END {print "   host cpu% sum " s}' ;;
  esac
}

step() {
  local name=$1
  shift
  echo "== step $name: $*"
  device_state
  local t0 t1 rc
  t0=$(date +%s)
  capped "$@" >"$out/$name.out" 2>"$out/$name.err"
  rc=$?
  [ "$rc" -eq 125 ] && { echo "== step $name: a process survived the cap; stopping"; exit 125; }
  t1=$(date +%s)
  echo "== step $name: exit $rc, wall $((t1 - t0)) s"
  device_state
  return "$rc"
}

status=0
# A step that exited 0 without a line its conclusion rests on fails the run rather than publishing
# an incomplete record (a regressed hook, a config source that did not take).
require() {
  grep -q -- "$3" "$out/$1.$2" && return 0
  echo "== step $1: MISSING EVIDENCE in $1.$2: no line matching $3"
  status=1
}

for s in "$@"; do
  case $s in
  build)
    step build sh -c "cd '$root' && dune build bin/projection_shape_bench.exe \
      benchmarks/runners/ocannl/bench_gpt.exe" || { cat "$out/build.err"; exit 1; }
    ;;
  provenance)
    echo "host $(hostname) sha $(git -C "$root" rev-parse HEAD) backend $backend cap ${cap}s"
    case $backend in
    hip) rocminfo 2>/dev/null | grep -E "Marketing|gfx" | sort -u | head -4 ;;
    cuda) nvidia-smi -L ;;
    metal) system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Cores" ;;
    cc) lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)" || sysctl -n machdep.cpu.brand_string ;;
    esac
    [ -r "$fixture" ] && echo "fixture $fixture sha256 $(shasum -a 256 "$fixture" | cut -d' ' -f1)"
    ;;
  crown-fwd | crown-rev)
    if step "$s" ../_build/default/bin/projection_shape_bench.exe 200 8 d "${s#crown-}" seeds \
      --ocannl_backend="$backend" --ocannl_log_config_sourcing=true; then
      require "$s" out '^== gh-ocannl-755: candidate ranking'
      require "$s" out '^   crown: '
      require "$s" err "^Found $backend, commandline --ocannl_backend=$backend\$"
    else status=1; fi
    # The ranking table and the crown verdicts, which are the gh-ocannl-833 deliverable.
    sed -n '/== gh-ocannl-755/,$p' "$out/$s.out"
    grep 'Found .*--ocannl_backend=' "$out/$s.err" | sort -u
    ;;
  session-isolated | session-queued)
    mode=${s#session-}
    rm -rf "$out/cache-$mode"
    if step "$s" env BENCH_FIXTURE="$fixture" BENCH_TUNE=1 BENCH_TIMING_TRACE=1 \
      ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend="$backend" \
      --ocannl_autotune_timing="$mode" --ocannl_autotune_cache_dir="$out/cache-$mode" \
      --ocannl_log_config_sourcing=true; then
      # A completed session is evidence only with its treatment and its cost record in it.
      require "$s" out '"compile_s":'
      require "$s" out '"searched":true'
      require "$s" err '^timing-trace: summary: '
      require "$s" err "^Found $backend, commandline --ocannl_backend=$backend\$"
      require "$s" err "^Found $mode, commandline --ocannl_autotune_timing=$mode\$"
    else status=1; fi
    # The result line (compile_s is the search wall) and the trace's last word: the summary on a
    # completed run, the running totals of the last timing call on a capped one.
    cat "$out/$s.out"
    grep -c 'timing-trace: attempt' "$out/$s.err" | sed 's/^/candidate attempts /'
    grep 'timing-trace: \(call\|summary\)' "$out/$s.err" | tail -2
    grep 'Found .*--ocannl_\(backend\|autotune_timing\)=' "$out/$s.err" | sort -u
    ;;
  *) echo "gh834: unknown step $s" >&2; exit 2 ;;
  esac
done
exit "$status"

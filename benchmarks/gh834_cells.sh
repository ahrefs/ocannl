#!/bin/bash
# gh-ocannl-833 / gh-ocannl-834 measurement driver: the same bounded steps on every box, so the
# issue comments quote invocations rather than restating commands.
#
# Usage (from anywhere; the checkout is the one this script lives in):
#   benchmarks/gh834_cells.sh BACKEND OUT FIXTURE CAP STEP...
#
#   BACKEND  cc | hip | cuda | metal, pinned on every command line (out-ranks every other source).
#   OUT      results directory, created; every step writes OUT/<step>.out and OUT/<step>.err.
#   FIXTURE  absolute path of a gpt2_mini.safetensors (only the session steps read it).
#   CAP      per-step wall cap in seconds; a capped step exits by SIGALRM and its trace lines stand
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
root=$(cd "$(dirname "$0")/.." && pwd -P)
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2

# A portable wall cap: macOS has no timeout(1). The alarm survives exec, and SIGALRM's default
# action ends the process without at_exit -- which is why the trace carries running totals.
capped() { perl -e 'alarm shift; exec @ARGV or die "exec: $!"' "$cap" "$@"; }

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
  t1=$(date +%s)
  echo "== step $name: exit $rc, wall $((t1 - t0)) s"
  device_state
  return "$rc"
}

status=0
for s in "$@"; do
  case $s in
  build)
    echo "== build"
    (cd "$root" && dune build bin/projection_shape_bench.exe benchmarks/runners/ocannl/bench_gpt.exe) 2>&1 || exit 1
    ;;
  provenance)
    echo "host $(hostname) sha $(git -C "$root" rev-parse HEAD) backend $backend cap ${cap}s"
    git -C "$root" status --porcelain | head -5
    case $backend in
    hip) rocminfo 2>/dev/null | grep -E "Marketing|gfx" | sort -u | head -4 ;;
    cuda) nvidia-smi -L ;;
    metal) system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Cores" ;;
    cc) lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)" || sysctl -n machdep.cpu.brand_string ;;
    esac
    [ -r "$fixture" ] && echo "fixture $fixture sha256 $(shasum -a 256 "$fixture" | cut -d' ' -f1)"
    ;;
  crown-fwd | crown-rev)
    step "$s" ../_build/default/bin/projection_shape_bench.exe 200 8 d "${s#crown-}" seeds \
      --ocannl_backend="$backend" --ocannl_log_config_sourcing=true || status=1
    # The ranking table and the crown verdicts, which are the gh-ocannl-833 deliverable.
    sed -n '/== gh-ocannl-755/,$p' "$out/$s.out"
    grep 'Found .*--ocannl_backend=' "$out/$s.err" | sort -u
    ;;
  session-isolated | session-queued)
    mode=${s#session-}
    rm -rf "$out/cache-$mode"
    step "$s" env BENCH_FIXTURE="$fixture" BENCH_TUNE=1 BENCH_TIMING_TRACE=1 \
      ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend="$backend" \
      --ocannl_autotune_timing="$mode" --ocannl_autotune_cache_dir="$out/cache-$mode" \
      --ocannl_log_config_sourcing=true || status=1
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

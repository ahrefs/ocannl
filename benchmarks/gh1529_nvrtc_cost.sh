#!/bin/bash
# gh-ocannl-1529 measurement driver: what a CUDA compile costs per program and per kernel, on a
# real routine, and what a search candidate spends compiling -- the evidence for or against porting
# the HIP kernel code cache (gh-ocannl-1314) to CUDA.
#
# Usage:
#   benchmarks/gh1529_nvrtc_cost.sh prepare ROOT OUT FIXTURE
#   benchmarks/gh1529_nvrtc_cost.sh measure ROOT OUT FIXTURE BUDGET
#   benchmarks/gh1529_nvrtc_cost.sh ab      ROOT OUT FIXTURE BUDGET ROUNDS
#   benchmarks/gh1529_nvrtc_cost.sh summary OUT
#
#   ROOT     the checkout (its benchmarks/ocannl_config is the runner's configuration).
#   OUT      results directory; `prepare` creates it (refused unless empty), `measure` adds to it.
#   FIXTURE  the gpt2_mini_train fixture (.safetensors); checked with fixture_digest.py --check.
#   BUDGET   seconds for the tuned search's whole tune_placements call (tune_call_budget_seconds).
#   ROUNDS   a positive even count of A/B rounds.
#
# `prepare` (correctness-grade, builds): builds bench_gpt.exe and benchmarks/gh1529_nvrtc_cost.c
# into OUT/bin, then runs ONE untuned gpt2_mini_train step on CUDA with debug files on and keeps the
# routine with the most kernels as OUT/src/routine.cu, its PTX target as OUT/src/arch (the
# `.target sm_XX` OCANNL's own compile produced, passed back as --gpu-architecture=compute_XX).
# `measure` (timing-grade, never builds): runs the probe on that source with CUDA_CACHE_DISABLE=1
# (every module load JITs), pinned to PROBE_CPUS, into OUT/probe.out, then a cold tuned search (fresh autotune cache
# directory, the driver's own JIT cache left as a user has it) with BENCH_TIMING_TRACE=1 into
# OUT/search.{out,err}, and prints the summary. `summary` reads the probe's `row`/`fit` lines and the
# search's per-compile `timing-trace: compile` lines, per search file.
# `ab` (timing-grade, never builds; after the CUDA kernel code cache, gh-ocannl-1529): the same cold
# budgeted search with the cache off (--ocannl_cuda_kernel_cache_mib=0) and at its default, ROUNDS
# times each, the order reversed every other round, into OUT/search-mib<0|default>-r<round>.*, then
# the summary. Compare the arms' per-candidate medians: a cut search compiles more candidates when
# each costs less, so the counts and sums differ by design. Both arms run under
# CUDA_CACHE_DISABLE=1: the driver's compute cache answers nvrtc as well as the JIT for any program
# text an EARLIER PROCESS compiled, and a repeated search re-renders the same candidates, so without
# it the cache-off arm reads warm (a first A/B measured its candidates at 0.07 s of nvrtc on all 261
# kernels) while the cache-on arm's miss programs, text no process had compiled, read cold. A
# search over candidates nobody compiled before -- the case the cache is for -- meets the cold cost.
#
# Exit: 0 every step succeeded; 1 a step failed; 2 usage. Hermetic like the other drivers: every
# OCANNL_* variable is unset and the backend is on argv.
set -u
while read -r v; do unset "$v"; done < <(env | sed -nE 's/^(OCANNL_[A-Za-z0-9_]*)=.*/\1/p')
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
BENCH=benchmarks/runners/ocannl/bench_gpt.exe

summary() { # OUT
  python3 - "$1" "$(cd "$(dirname "$0")" && pwd -P)" <<'EOF'
import os, re, statistics, sys
sys.path.insert(0, sys.argv[2])
import bench_output
out = sys.argv[1]
ok = True
probe = os.path.join(out, 'probe.out')
if os.path.exists(probe):
    text = open(probe).read()
    print('== probe (CUDA_CACHE_DISABLE=1) ==')
    for ln in text.splitlines():
        if ln.startswith(('header', 'row', 'fit')):
            print(ln)
    if 'done' not in text.splitlines():
        print('probe: INCOMPLETE (no `done` line)'); ok = False
elif not any(f.startswith('search-') for f in os.listdir(out)):
    print('probe: no probe.out'); ok = False
errs = sorted(f for f in os.listdir(out) if re.fullmatch(r'search(-.*)?\.err', f))
if not errs:
    print('search: no search*.err'); ok = False
for name in errs:
    err = os.path.join(out, name)
    rows = bench_output.compile_traces(open(err).read())
    print(f'== {name}: {len(rows)} compile records ==')
    by = {}
    for r in rows:
        if r['verdict'] != 'compiled':
            continue
        by.setdefault(r['stage'], []).append(
            (r['wall'], r['lower'], r['transform'], r['codegen'],
             r['device'] if r['device'] is not None else 0.0, r['link'], r['kernels_compiled'],
             r['kernels']))
    for stage, rs in sorted(by.items()):
        med = lambda i: statistics.median(r[i] for r in rs)
        print(f'{stage}: {len(rs)} compiled; medians: wall {med(0):.3f} s, lower {med(1):.3f}, '
              f'transform {med(2):.3f}, codegen {med(3):.3f}, nvrtc {med(4):.3f}, link {med(5):.3f}, '
              f'kernels compiled {med(6):.0f} of {med(7):.0f}; sums: wall {sum(r[0] for r in rs):.1f} s, '
              f'nvrtc {sum(r[4] for r in rs):.1f} s, link {sum(r[5] for r in rs):.1f} s')
    if not rows:
        print(f'{name}: no compile records'); ok = False
sys.exit(0 if ok else 1)
EOF
}

check_fixture() { # ROOT FIXTURE
  [ -f "$2" ] || { echo "gh1529: no fixture $2" >&2; return 1; }
  python3 "$1/benchmarks/fixture_digest.py" --check "$2" || { echo "gh1529: fixture $2 fails its digest" >&2; return 1; }
}

case "${1:-}" in
  prepare)
    [ $# -eq 4 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    OUT=$3 FIX=$4
    if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "gh1529: $OUT is not empty" >&2; exit 2; fi
    mkdir -p "$OUT/bin" "$OUT/src" && OUT=$(cd "$OUT" && pwd -P) || exit 2
    check_fixture "$ROOT" "$FIX" || exit 1
    FIX=$(cd "$(dirname "$FIX")" && pwd -P)/$(basename "$FIX")
    {
      echo "gh1529: host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ) rev=$(git -C "$ROOT" rev-parse HEAD)"
      echo "fixture: $FIX"
    } > "$OUT/driver.log"
    (cd "$ROOT" && dune build "./$BENCH") || exit 1
    cp "$ROOT/_build/default/$BENCH" "$OUT/bin/bench_gpt.exe" || exit 1
    gcc -O2 -o "$OUT/bin/gh1529_nvrtc_cost" "$ROOT/benchmarks/gh1529_nvrtc_cost.c" \
      -I"$CUDA_HOME/include" -L"$CUDA_HOME/lib64" -lnvrtc -lcuda -lpthread || exit 1
    PREFIX=gh1529-dump-$$
    DUMP=$ROOT/benchmarks/build_files/$PREFIX
    rm -rf "$DUMP"
    (cd "$ROOT/benchmarks" && BENCH_FIXTURE=$FIX BENCH_TUNE=0 "$OUT/bin/bench_gpt.exe" \
      --ocannl_backend=cuda --ocannl_output_debug_files_in_build_directory=true \
      --ocannl_build_files_prefix="$PREFIX") > "$OUT/dump.out" 2> "$OUT/dump.err" \
      || { echo "gh1529: the untuned dump run failed (see $OUT/dump.err)" >&2; exit 1; }
    best= bestn=0
    for f in "$DUMP"/*.cu; do
      [ -f "$f" ] || continue
      n=$(grep -c '^extern "C" __global__' "$f")
      if [ "$n" -gt "$bestn" ]; then best=$f bestn=$n; fi
    done
    [ -n "$best" ] || { echo "gh1529: the dump run left no .cu in $DUMP" >&2; exit 1; }
    ptx=${best%.cu}.ptx
    target=$(sed -nE 's/^\.target (sm|compute)_([0-9]+[a-z]?).*/\2/p' "$ptx" | head -1)
    [ -n "$target" ] || { echo "gh1529: no .target in $ptx" >&2; exit 1; }
    cp "$best" "$OUT/src/routine.cu" && cp "$ptx" "$OUT/src/routine.ptx" || exit 1
    echo "--gpu-architecture=compute_$target" > "$OUT/src/arch"
    echo "routine: $(basename "$best"), $bestn kernels, PTX target $target" | tee -a "$OUT/driver.log"
    rm -rf "$DUMP"
    # A smoke run of the probe, so `measure` can start on a binary that is known to work.
    "$OUT/bin/gh1529_nvrtc_cost" "$OUT/src/routine.cu" -I"$CUDA_HOME/include" --use_fast_math \
      "$(cat "$OUT/src/arch")" > "$OUT/probe-smoke.out" 2>&1 \
      || { echo "gh1529: probe smoke failed (see $OUT/probe-smoke.out)" >&2; exit 1; }
    grep -E '^(header|row joint|fit)' "$OUT/probe-smoke.out"
    ;;
  measure)
    [ $# -eq 5 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    OUT=$(cd "$3" && pwd -P) || exit 2
    FIX=$4 BUDGET=$5
    case "$BUDGET" in ''|*[!0-9]*) usage;; esac
    [ -x "$OUT/bin/gh1529_nvrtc_cost" ] && [ -f "$OUT/src/routine.cu" ] \
      || { echo "gh1529: run prepare first" >&2; exit 2; }
    [ -e "$OUT/probe.out" ] && { echo "gh1529: $OUT already measured" >&2; exit 2; }
    check_fixture "$ROOT" "$FIX" || exit 1
    FIX=$(cd "$(dirname "$FIX")" && pwd -P)/$(basename "$FIX")
    echo "measure: date=$(date -u +%Y-%m-%dT%H:%M:%SZ) budget=$BUDGET" >> "$OUT/driver.log"
    # Pinned (PROBE_CPUS, default 0-7: rog's P-cores; empty to not pin): on a hybrid P/E CPU an
    # unpinned single-threaded compile lands on either kind of core from row to row.
    pin=()
    if [ -n "${PROBE_CPUS-0-7}" ] && command -v taskset >/dev/null; then pin=(taskset -c "${PROBE_CPUS-0-7}"); fi
    echo "probe pinning: ${pin[*]:-none}" >> "$OUT/driver.log"
    CUDA_CACHE_DISABLE=1 "${pin[@]}" "$OUT/bin/gh1529_nvrtc_cost" "$OUT/src/routine.cu" \
      -I"$CUDA_HOME/include" --use_fast_math "$(cat "$OUT/src/arch")" > "$OUT/probe.out" 2>&1 \
      || { echo "gh1529: probe failed (see $OUT/probe.out)" >&2; exit 1; }
    rm -rf "$OUT/autotune_cache"
    (cd "$ROOT/benchmarks" && BENCH_FIXTURE=$FIX BENCH_TUNE=1 BENCH_TIMING_TRACE=1 \
      "$OUT/bin/bench_gpt.exe" --ocannl_backend=cuda \
      --ocannl_autotune_cache_dir="$OUT/autotune_cache" \
      --ocannl_tune_call_budget_seconds="$BUDGET") > "$OUT/search.out" 2> "$OUT/search.err"
    rc=$?
    echo "search exit: $rc" | tee -a "$OUT/driver.log"
    summary "$OUT" | tee "$OUT/summary.txt"
    # A search that failed after printing compile records still fails the measurement.
    [ "${PIPESTATUS[0]}" -eq 0 ] && [ "$rc" -eq 0 ] || exit 1
    ;;
  ab)
    [ $# -eq 6 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    OUT=$(cd "$3" && pwd -P) || exit 2
    FIX=$4 BUDGET=$5 ROUNDS=$6
    case "$BUDGET" in ''|*[!0-9]*) usage;; esac
    case "$ROUNDS" in ''|*[!0-9]*) usage;; esac
    { [ "$ROUNDS" -gt 0 ] && [ $((ROUNDS % 2)) -eq 0 ]; } || usage
    [ -x "$OUT/bin/bench_gpt.exe" ] || { echo "gh1529: run prepare first" >&2; exit 2; }
    ls "$OUT"/search-* >/dev/null 2>&1 && { echo "gh1529: $OUT already has an A/B" >&2; exit 2; }
    echo "ab: CUDA_CACHE_DISABLE=1 in both arms" >> "$OUT/driver.log"
    check_fixture "$ROOT" "$FIX" || exit 1
    FIX=$(cd "$(dirname "$FIX")" && pwd -P)/$(basename "$FIX")
    echo "ab: date=$(date -u +%Y-%m-%dT%H:%M:%SZ) budget=$BUDGET rounds=$ROUNDS" >> "$OUT/driver.log"
    status=0
    for ((r = 0; r < ROUNDS; r++)); do
      if [ $((r % 2)) -eq 0 ]; then order="0 default"; else order="default 0"; fi
      for mib in $order; do
        arm=(); [ "$mib" = default ] || arm=(--ocannl_cuda_kernel_cache_mib="$mib")
        base=$OUT/search-mib$mib-r$r
        rm -rf "$OUT/autotune_cache"
        (cd "$ROOT/benchmarks" && CUDA_CACHE_DISABLE=1 BENCH_FIXTURE=$FIX BENCH_TUNE=1 \
          BENCH_TIMING_TRACE=1 "$OUT/bin/bench_gpt.exe" --ocannl_backend=cuda "${arm[@]}" \
          --ocannl_autotune_cache_dir="$OUT/autotune_cache" \
          --ocannl_tune_call_budget_seconds="$BUDGET") > "$base.out" 2> "$base.err"
        rc=$?
        [ "$rc" -eq 0 ] || status=1
        echo "round $r mib=$mib: exit $rc" | tee -a "$OUT/driver.log"
      done
    done
    summary "$OUT" | tee "$OUT/summary-ab.txt"
    [ "${PIPESTATUS[0]}" -eq 0 ] || status=1
    exit "$status"
    ;;
  summary)
    [ $# -eq 2 ] || usage
    summary "$2"
    ;;
  *) usage ;;
esac

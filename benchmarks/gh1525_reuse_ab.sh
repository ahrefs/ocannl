#!/bin/bash
# gh-ocannl-1525 measurement driver: what a tuned search's candidates spend compiling, and how many
# kernels each compiles, across builds (the base revision and this one, for the stable kernel keys)
# and settings (staffed singles on or off; a backend's kernel code cache off) -- the evidence for the
# stable keys, the cc kernel code cache and the staffed-singles compile overhead.
#
# Usage:
#   benchmarks/gh1525_reuse_ab.sh prepare ROOT OUT BASE_REV
#   benchmarks/gh1525_reuse_ab.sh ab      ROOT OUT FIXTURE BACKEND BUDGET ROUNDS CELLS
#   benchmarks/gh1525_reuse_ab.sh summary OUT
#
#   ROOT      the checkout (its benchmarks/ocannl_config is the runner's configuration, for every
#             cell); `prepare` builds bench_gpt.exe there as the `head` build.
#   OUT       results directory; `prepare` creates it (refused unless empty), `ab` adds to it.
#   BASE_REV  the revision built as the `base` cell build, in a temporary worktree under OUT; `-`
#             builds none (only `head` cells then).
#   FIXTURE   the gpt2_mini_train fixture (.safetensors); checked with fixture_digest.py --check.
#   BACKEND   cc | metal | cuda | hip, on every cell's argv.
#   BUDGET    seconds for each search's whole tune_placements call (tune_call_budget_seconds).
#   ROUNDS    a positive even count of rounds; every other round runs the cells in reverse order.
#   CELLS     comma-separated cells, each BUILD-STAFFED[-nocache]: BUILD base|head, STAFFED on|off
#             (autotune_staffed_singles), `-nocache` sets BACKEND's kernel code cache to 0 MiB
#             (<backend>_kernel_cache_mib; refused for metal, which has none).
#
# `prepare` (correctness-grade, builds): bench_gpt.exe of ROOT into OUT/bin/head, of BASE_REV into
# OUT/bin/base. `ab` (timing-grade, never builds): each cell is one cold budgeted search of
# gpt2_mini_train (fresh autotune cache directory, BENCH_TUNE=1 BENCH_TIMING_TRACE=1), into
# OUT/search-<cell>-r<round>.{out,err}; on CUDA under CUDA_CACHE_DISABLE=1 (the driver's compute
# cache answers nvrtc for program text an earlier process compiled, gh-ocannl-1529). Then the
# summary. `summary` reads each search's `timing-trace: compile` records: per stage, the compiled
# count, the medians (wall, device compiler, link, kernels compiled of rendered) and the sums, and
# the result line's tuned step p50. A cut search compiles more candidates when each costs less, so
# compare per-candidate medians; the sums are the whole-search cost.
#
# Exit: 0 every step succeeded; 1 a step failed; 2 usage. Hermetic like the other drivers: every
# OCANNL_* variable is unset and the backend is on argv.
set -u
while read -r v; do unset "$v"; done < <(env | sed -nE 's/^(OCANNL_[A-Za-z0-9_]*)=.*/\1/p')
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
BENCH=benchmarks/runners/ocannl/bench_gpt.exe

summary() { # OUT
  python3 - "$1" "$(cd "$(dirname "$0")" && pwd -P)" <<'EOF'
import os, re, statistics, sys
sys.path.insert(0, sys.argv[2])
import bench_output
out = sys.argv[1]
ok = True
errs = sorted(f for f in os.listdir(out) if re.fullmatch(r'search-.*\.err', f))
if not errs:
    print('summary: no search-*.err'); ok = False
for name in errs:
    rows = bench_output.compile_traces(open(os.path.join(out, name)).read())
    step = 'none'
    try:
        rec = bench_output.result_record(open(os.path.join(out, name[:-4] + '.out')).read())
        if rec and isinstance(rec.get('step_ms'), dict):
            step = f"{rec['step_ms'].get('p50')} ms"
    except (OSError, ValueError):
        pass
    print(f'== {name}: {len(rows)} compile records, tuned step p50 {step} ==')
    by = {}
    for r in rows:
        if r['verdict'] != 'compiled':
            continue
        by.setdefault(r['stage'], []).append(
            (r['wall'], r['device'] if r['device'] is not None else 0.0, r['link'],
             r['kernels_compiled'], r['kernels']))
    for stage, rs in sorted(by.items()):
        med = lambda i: statistics.median(r[i] for r in rs)
        print(f'{stage}: {len(rs)} compiled; medians: wall {med(0):.3f} s, device {med(1):.3f}, '
              f'link {med(2):.3f}, kernels compiled {med(3):.0f} of {med(4):.0f}; sums: wall '
              f'{sum(r[0] for r in rs):.1f} s, device {sum(r[1] for r in rs):.1f} s, kernels '
              f'compiled {sum(r[3] for r in rs)} of {sum(r[4] for r in rs)}')
    if not rows:
        print(f'{name}: no compile records'); ok = False
sys.exit(0 if ok else 1)
EOF
}

check_fixture() { # ROOT FIXTURE
  [ -f "$2" ] || { echo "gh1525: no fixture $2" >&2; return 1; }
  python3 "$1/benchmarks/fixture_digest.py" --check "$2" || { echo "gh1525: fixture $2 fails its digest" >&2; return 1; }
}

case "${1:-}" in
  prepare)
    [ $# -eq 4 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    OUT=$3 BASE_REV=$4
    if [ -e "$OUT" ] && [ -n "$(ls -A "$OUT")" ]; then echo "gh1525: $OUT is not empty" >&2; exit 2; fi
    mkdir -p "$OUT/bin/head" && OUT=$(cd "$OUT" && pwd -P) || exit 2
    echo "gh1525: host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ) head=$(git -C "$ROOT" rev-parse HEAD)" > "$OUT/driver.log"
    (cd "$ROOT" && dune build "./$BENCH") || exit 1
    cp "$ROOT/_build/default/$BENCH" "$OUT/bin/head/bench_gpt.exe" || exit 1
    if [ "$BASE_REV" != - ]; then
      SRC=$OUT/base-src
      git -C "$ROOT" worktree add --detach "$SRC" "$BASE_REV" || exit 1
      echo "base=$(git -C "$SRC" rev-parse HEAD)" >> "$OUT/driver.log"
      status=0
      (cd "$SRC" && dune build "./$BENCH") && mkdir -p "$OUT/bin/base" \
        && cp "$SRC/_build/default/$BENCH" "$OUT/bin/base/bench_gpt.exe" || status=1
      git -C "$ROOT" worktree remove --force "$SRC"
      [ "$status" -eq 0 ] || exit 1
    fi
    cat "$OUT/driver.log"
    ;;
  ab)
    [ $# -eq 8 ] || usage
    ROOT=$(cd "$2" && pwd -P) || exit 2
    OUT=$(cd "$3" && pwd -P) || exit 2
    FIX=$4 BACKEND=$5 BUDGET=$6 ROUNDS=$7 CELLS=$8
    case "$BACKEND" in cc|metal|cuda|hip) ;; *) usage;; esac
    case "$BUDGET" in ''|*[!0-9]*) usage;; esac
    case "$ROUNDS" in ''|*[!0-9]*) usage;; esac
    { [ "$ROUNDS" -gt 0 ] && [ $((ROUNDS % 2)) -eq 0 ]; } || usage
    IFS=, read -r -a cells <<< "$CELLS"
    for c in "${cells[@]}"; do
      case "$c" in
        base-on|base-off|head-on|head-off) ;;
        base-on-nocache|base-off-nocache|head-on-nocache|head-off-nocache)
          [ "$BACKEND" != metal ] || { echo "gh1525: metal has no kernel code cache" >&2; exit 2; } ;;
        *) usage ;;
      esac
      [ -x "$OUT/bin/${c%%-*}/bench_gpt.exe" ] || { echo "gh1525: no ${c%%-*} build; run prepare" >&2; exit 2; }
    done
    ls "$OUT"/search-* >/dev/null 2>&1 && { echo "gh1525: $OUT already has an A/B" >&2; exit 2; }
    check_fixture "$ROOT" "$FIX" || exit 1
    FIX=$(cd "$(dirname "$FIX")" && pwd -P)/$(basename "$FIX")
    echo "ab: date=$(date -u +%Y-%m-%dT%H:%M:%SZ) backend=$BACKEND budget=$BUDGET rounds=$ROUNDS cells=$CELLS" >> "$OUT/driver.log"
    envs=(BENCH_FIXTURE="$FIX" BENCH_TUNE=1 BENCH_TIMING_TRACE=1)
    [ "$BACKEND" = cuda ] && envs+=(CUDA_CACHE_DISABLE=1)
    status=0
    for ((r = 0; r < ROUNDS; r++)); do
      order=("${cells[@]}")
      if [ $((r % 2)) -eq 1 ]; then
        order=(); for ((i = ${#cells[@]} - 1; i >= 0; i--)); do order+=("${cells[$i]}"); done
      fi
      for c in "${order[@]}"; do
        IFS=- read -r build staffed nocache <<< "$c"
        st=false; [ "$staffed" = on ] && st=true
        arm=(--ocannl_backend="$BACKEND" --ocannl_autotune_staffed_singles="$st")
        [ -n "$nocache" ] && arm+=(--ocannl_"$BACKEND"_kernel_cache_mib=0)
        base=$OUT/search-$c-r$r
        rm -rf "$OUT/autotune_cache"
        (cd "$ROOT/benchmarks" && env "${envs[@]}" "$OUT/bin/$build/bench_gpt.exe" "${arm[@]}" \
          --ocannl_autotune_cache_dir="$OUT/autotune_cache" \
          --ocannl_tune_call_budget_seconds="$BUDGET") > "$base.out" 2> "$base.err"
        rc=$?
        [ "$rc" -eq 0 ] || status=1
        echo "round $r cell $c: exit $rc" | tee -a "$OUT/driver.log"
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

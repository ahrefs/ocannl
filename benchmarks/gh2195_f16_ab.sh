#!/bin/bash
# gh-ocannl-2195: throughput A/B of the two f16 accumulator arms on CUDA and HIP --
# fp16_arithmetic=false (Fp16_wide: f32 reduction accumulators, the f32-accumulate uniform-f16
# tensor-unit arms behind a converted f16 boundary) against fp16_arithmetic=true (Fp16_narrow:
# storage-width f16 accumulators and the f16-accumulate tensor-unit arms) -- the cost side of
# resolving auto wide on CUDA and HIP, whose accuracy side is test/operations/f16_accum_ab. The arms
# are named `false` and `true`, not `auto`: before gh-ocannl-2195 auto resolved narrow on CUDA and
# HIP and since then it resolves wide, while the explicit values name the same arm on both sides of
# that change, so the arms mean the same at every revision this driver can measure (below).
#
# Usage: gh2195_f16_ab.sh [--box BOX | --skip-box-check] [--untuned-fixture PATH]
#                         <repo-root> <backend cuda|hip> <memory-label> [rounds] [sizes...]
#   e.g. gh2195_f16_ab.sh ~/ocannl cuda "GDDR7 (RTX 5070 Ti Laptop, sm_120)" 8 1024 2048
#        gh2195_f16_ab.sh --box minix --untuned-fixture ~/ocannl-staging/benchmarks/fixtures/gpt2_mini_train.safetensors \
#          ~/ocannl hip "unified (gfx1151)" 8 1024 2048
#
# [rounds] must be even and positive (default 8); [sizes] must be multiples of 64 (regtile's
# block; the mma variants need 32) or those variants are skipped and the run is invalid (default
# 1024 2048). The cells are bin_cells.py's (gh-ocannl-1607): per size one run of four arms -- the
# two fp16_arithmetic arms (`false`, `true`) at half and at f32 (the control below), named
# <arm>-<prec> -- visited round by round in Bench_design's spread mirrored rotation, so every arm's
# position histogram is equal at a multiple of four rounds, and every pair of arms exchanges
# positions within each pair of rounds. Each cell is hermetic (no OCANNL_*, BENCH_* or OpenMP
# variable inherited, no config file read), supervised, and configuration-checked: its
# resolved-configuration records must be its command line, so the library must carry the stream
# (gh-ocannl-1566): measure an older revision with the driver of its own time.
#
# HAND-SCHEDULED ROWS. Each treatment cell is bin/schedule_bench at m = n = k = <size> with f16
# operands and output. All five variants are the treatment: the tensorized mma_pd1/mma_pd2 (staged
# Tile_mma; the policy picks their arm -- on CUDA sm_80+ the f32-accumulate m16n8k16 / converted
# wmma arm against wmma's f16-accumulate fragments, on HIP the f32 against the f16 WMMA
# accumulator) and the serial legs parallel/smem/regtile, whose accumulator residency the policy
# moves too (the backend's accum_prec: f32 or half). No schedule cache is involved: every variant
# is a hand-written schedule compiled through [~lowered_transform].
#
# UNTUNED-DEFAULT ROW (only with --untuned-fixture). gh-ocannl-2195 also moves the untuned
# default's f16 matmul sites: under the wide arm their accumulator widens their storage, so the
# seeded untuned plan (Autotune.seeded_default, gh-ocannl-1324) refuses them
# (Autotune.accumulates_at_storage; gpu_default_seed_narrow_storage, pinned off here, is the gate
# that would admit them), where the narrow arm seeds them. schedule_bench cannot time that: its
# variants are hand schedules and its untimed oracle is a plain Context.compile. So this row is
# benchmarks/runners/ocannl/bench_gpt's untuned training step (BENCH_TUNE=0, so
# Bench_harness.compile_default = Autotune.untuned_compile; no search, model_default off) on the
# given fixture under BENCH_PRECISION=f16 (the mixed-precision recipe: f16 activations and
# gradients, f32 masters and loss head, dynamic loss scaling), one bin_cells --reader record run of
# the same four arms (<arm>-half at BENCH_PRECISION=f16, <arm>-single at f32), metric
# step_ms.p50. The fixture loads behind the box gate, so the row needs --box (or
# --skip-box-check). Its validation is the summary below: every cell clean, each cell's result line
# reporting the storage precision its arm asked for, and each arm's seeding census (the
# `seeded_default:` line autotune_log prints) the same in every round; whether the two f16 arms'
# censuses differ is REPORTED -- identical censuses mean the treatment does not reach this
# workload's untuned plan and the row is a second control, not a failure. The f32 arms are its drift
# control, timed but not source-proven (the proof below covers schedule_bench's kernels).
#
# DRIFT CONTROL (the A/B protocol in docs/agent-notes/training-and-performance.md): next to every
# treatment cell runs a CONTROL cell, the same bench at f32 under the same arm's fp16_arithmetic
# flag. The f16 policy cannot reach an f32 kernel, and before any timing the driver PROVES it: one
# untimed control run per arm writes its generated source (.cu on CUDA, .hip on HIP; the untimed
# oracle's plain default compile included), and the two sources must be byte-identical or the
# driver stops. The control's wide/narrow ratio is then the session drift the treatment ratios are
# read against. The proof's twin at f16 shows the treatment REACHES codegen: one untimed f16 run per
# arm, whose sources must differ (and whether the oracle's plain default kernel differs is
# reported).
#
# VALIDATION: a timing counts only from a cell whose output is validated, which the bench's exit
# status cannot do for f16 (f16 holds its exact verdict only to k = 5, so past it the bench reports
# cell differences as rounding and exits 0). benchmarks/gh1051_cells.py's mode "ab" at PREC half
# (shared with gh1051_bf16_ab.sh at bfloat16; its HALF section) reads each size's bin_cells run and
# checks every cell's verdict there and its whole-output checksums -- determinism across rounds,
# the f32 control equal to the exact answer, every wide f16 cell and the narrow serial legs within
# a structural bound of it (the narrow tensorized pair is gross on gfx11 and only reported, its
# non-degeneracy checked), identical checksums from the two tensorized pipelinings in each
# arm, every tensorized f16 cell's census reporting intrinsics -- and prints the table; any failure
# fails the run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
. "$HERE/box-gate.sh"
box_gate_args "$@"
shift "$box_gate_shift"
USAGE="usage: $0 [--box BOX | --skip-box-check] [--untuned-fixture PATH] <repo-root> <backend cuda|hip> <memory-label> [rounds] [sizes...]"
FIXTURE=
if [ "${1-}" = --untuned-fixture ]; then
  [ $# -ge 2 ] && [ -n "$2" ] || { echo "$USAGE"; exit 2; }
  [ -f "$2" ] || { echo "no such fixture: $2"; exit 2; }
  FIXTURE=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
  case $FIXTURE in *.safetensors) ;; *) echo "the untuned fixture must be a .safetensors file, got $2"; exit 2 ;; esac
  shift 2
fi
[ $# -ge 3 ] || { echo "$USAGE"; exit 2; }
ROOT=$(cd "$1" && pwd) || { echo "no such repo root: $1"; exit 2; }
BACKEND=$2
case $BACKEND in
  cuda) EXT=cu ;;
  hip) EXT=hip ;;
  *) echo "backend must be cuda or hip, got '$BACKEND'"; exit 2 ;;
esac
LABEL=$3
ROUNDS=${4:-8}
shift $(($# < 4 ? $# : 4))
SIZES=("$@")
[ ${#SIZES[@]} -gt 0 ] || SIZES=(1024 2048)
for size in "${SIZES[@]}"; do
  case $size in '' | *[!0-9]* | 0) echo "sizes must be positive integers, got '$size'"; exit 2 ;; esac
done
case $ROUNDS in '' | *[!0-9]*) echo "rounds must be a positive even integer, got '$ROUNDS'"; exit 2 ;; esac
if ((ROUNDS == 0 || ROUNDS % 2 != 0)); then
  echo "rounds must be a positive even integer (every round is mirrored by its partner), got $ROUNDS"
  exit 2
fi
GATE=()
if [ -n "$FIXTURE" ]; then
  # Before the first measurement: the untuned row loads the fixture.
  box_gate gh2195 "$(dirname "$FIXTURE")" "$(basename "$FIXTURE" .safetensors)"
  if [ "$box_gate_skip" = 1 ]; then GATE=(--skip-box-check); else GATE=(--box "$box_gate_box"); fi
fi
cd "$ROOT" || exit 1
dune build bin/schedule_bench.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
BENCH="$ROOT/_build/default/bin/schedule_bench.exe"
if [ -n "$FIXTURE" ]; then
  dune build benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || { echo "BUILD FAILED (bench_gpt)"; exit 1; }
fi
GPT="$ROOT/_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
OUT=$(mktemp -d "${TMPDIR:-/tmp}/gh2195_ab.XXXXXX")
echo "gh2195 f16 A/B: backend=$BACKEND label=$LABEL rounds=$ROUNDS sizes=${SIZES[*]} sha=$(git -C "$ROOT" rev-parse HEAD)"
echo "host=$(hostname) raw=$OUT untuned-fixture=${FIXTURE:-none}"
git -C "$ROOT" status --porcelain --untracked-files=no | sed 's/^/  dirty: /'
if [ "$BACKEND" = cuda ]; then
  nvidia-smi --query-gpu=name,compute_cap,driver_version,memory.total --format=csv,noheader 2>/dev/null |
    sed 's/^/device: /'
else
  rocminfo 2>/dev/null | grep -E "Marketing Name|^ *Name: *gfx" | sed 's/^ */device: /' | sort -u
fi
FAIL=0
# The identity proof, untimed: generated source per arm and precision, byte-compared. Each run is
# a cell of the shared runner, whose debug files land under its own directory.
for prec in single half; do
  for arm in false true; do
    prefix="gh2195_${prec}_$arm"
    python3 "$HERE/cells.py" cell --backend "$BACKEND" --dir "$OUT/identity_run_${prec}_$arm" \
      --expect fp16_arithmetic="$arm" --expect default_prec="$prec" -- \
      "$BENCH" "${SIZES[0]}" 10 "${SIZES[0]}" "${SIZES[0]}" 0 --ocannl_default_prec="$prec" \
      --ocannl_fp16_arithmetic="$arm" --ocannl_output_debug_files_in_build_directory=true \
      --ocannl_build_files_prefix="$prefix"
    rc=$?
    # A group that outlived its cleanup holds the device every later cell would be timed against.
    [ "$rc" -eq 125 ] && { echo "identity run ($prec, $arm): a process outlived its cleanup; stopping"; exit 125; }
    # An f16 run may differ from its oracle by rounding (exit 0); a nonzero exit is a fault.
    [ "$rc" -eq 0 ] || { echo "identity run ($prec, $arm) exited $rc"; FAIL=1; }
    mkdir -p "$OUT/identity_${prec}_$arm"
    cp "$OUT/identity_run_${prec}_$arm/build_files/$prefix"/*."$EXT" "$OUT/identity_${prec}_$arm/" 2>/dev/null
  done
done
n_src=$(find "$OUT/identity_single_false" -name "*.$EXT" | wc -l | tr -d ' ')
if [ "$n_src" -eq 0 ] || ! diff -r "$OUT/identity_single_false" "$OUT/identity_single_true" >/dev/null; then
  echo "DRIFT CONTROL INVALID: the f32 control's generated .$EXT differs across arms (or none was written: $n_src files)"
  exit 1
fi
echo "drift control: $n_src generated f32 kernels byte-identical across arms"
n_half=$(find "$OUT/identity_half_false" -name "*.$EXT" | wc -l | tr -d ' ')
if [ "$n_half" -eq 0 ] || diff -r "$OUT/identity_half_false" "$OUT/identity_half_true" >/dev/null; then
  echo "TREATMENT UNREACHED: the f16 generated .$EXT is identical across arms (or none was written: $n_half files)"
  exit 1
fi
n_diff=$(diff -rq "$OUT/identity_half_false" "$OUT/identity_half_true" | wc -l | tr -d ' ')
echo "treatment reach: $n_diff of $n_half generated f16 sources differ across arms"
ref_false=$(find "$OUT/identity_half_false" -name "*mm_reference*.$EXT" | sort)
if [ -z "$ref_false" ]; then
  echo "oracle (plain default compile, f16): no mm_reference source found to compare"
else
  for f in $ref_false; do
    if cmp -s "$f" "$OUT/identity_half_true/$(basename "$f")"; then
      echo "oracle (plain default compile, f16): $(basename "$f") identical across arms"
    else
      echo "oracle (plain default compile, f16): $(basename "$f") differs across arms"
    fi
  done
fi
for size in "${SIZES[@]}"; do
  arms=()
  for arm in false true; do
    for prec in half single; do
      arms+=(--arm "$arm-$prec=--ocannl_fp16_arithmetic=$arm --ocannl_default_prec=$prec")
    done
  done
  python3 "$HERE/bin_cells.py" --exe "$BENCH" --backend "$BACKEND" --out "$OUT/n$size" \
    --rounds "$ROUNDS" "${arms[@]}" -- "$size" 10 "$size" "$size" 0 >/dev/null
  rc=$?
  [ "$rc" -eq 125 ] && { echo "n=$size: a process outlived its cleanup; stopping"; exit 125; }
  [ "$rc" -eq 0 ] || { echo "n=$size: bin_cells exited $rc (see $OUT/n$size/report.md)"; FAIL=1; }
done
python3 "$HERE/gh1051_cells.py" ab "$OUT" "$LABEL" half || FAIL=1
[ -n "$FIXTURE" ] || exit $FAIL

# The untuned-default row: one bin_cells run (not under n*/, which gh1051_cells.py reads as sizes).
UNT="$OUT/untuned"
python3 "$HERE/bin_cells.py" --exe "$GPT" --backend "$BACKEND" --out "$UNT" ${GATE[@]+"${GATE[@]}"} \
  --reader record --rounds "$ROUNDS" --cell-timeout 1800 \
  --env "BENCH_FIXTURE=$FIXTURE" --env BENCH_TUNE=0 --env BENCH_MATERIALIZE=0 --env BENCH_DEBUG=0 \
  --arm "false-half=BENCH_PRECISION=f16 --ocannl_fp16_arithmetic=false" \
  --arm "true-half=BENCH_PRECISION=f16 --ocannl_fp16_arithmetic=true" \
  --arm "false-single=BENCH_PRECISION=f32 --ocannl_fp16_arithmetic=false" \
  --arm "true-single=BENCH_PRECISION=f32 --ocannl_fp16_arithmetic=true" \
  -- --ocannl_autotune_search=false --ocannl_model_default_schedule=false \
  --ocannl_gpu_default_seed_narrow_storage=false --ocannl_autotune_log=true >/dev/null
rc=$?
[ "$rc" -eq 125 ] && { echo "untuned: a process outlived its cleanup; stopping"; exit 125; }
[ "$rc" -eq 0 ] || { echo "untuned: bin_cells exited $rc (see $UNT/report.md)"; FAIL=1; }
python3 - "$HERE" "$UNT" "$LABEL" "$(basename "$FIXTURE" .safetensors)" <<'EOF' || FAIL=1
import statistics, sys
sys.path.insert(0, sys.argv[1])
import bench_output
import cells
from display_safe import display_value, log_line, table_lines

run, label, fixture = sys.argv[2], sys.argv[3], sys.argv[4]
problems = []
try:
    manifest, records = cells.read_run(run)
except (OSError, ValueError) as e:
    print(log_line(f"INVALID: untuned: no complete bin_cells run ({display_value(str(e))})"))
    sys.exit(1)
planned = sum(len(order) for order in manifest["orders"])
if len(records) != planned:
    problems.append(f"{len(records)} of {planned} planned cells ran")
problems += [display_value(p) for p in cells.run_level_problems(manifest, records)]
WANT = {"half": "f16", "single": "f32"}
ms, census = {}, {}  # (arm, prec) -> {round: step p50 ms}; -> {census tuple}
for r in records:
    cell = f"cell {r['cell']} ({display_value(r['arm'])})"
    flag, _, prec = r["arm"].partition("-")
    if flag not in ("false", "true") or prec not in WANT:
        problems.append(f"{cell}: not an arm this driver can name")
        continue
    if not r["ok"]:
        problems.append(f"{cell}: {display_value('; '.join(r['problems']))}")
        continue
    try:
        rec = bench_output.result_record(cells.cell_stream(run, r, "stdout"))
        got = bench_output.field(rec, "storage_precision")
        step = bench_output.field(rec, "step_ms.p50")
    except (KeyError, ValueError, TypeError) as e:
        problems.append(f"{cell}: unreadable result line ({display_value(str(e))})")
        continue
    if got != WANT[prec]:
        problems.append(f"{cell}: ran at storage precision {display_value(str(got))}, not {WANT[prec]}")
    ms.setdefault((flag, prec), {})[r["round"]] = step
    _, lines = bench_output.seeded_default(cells.cell_stream(run, r, "stderr"))
    census.setdefault((flag, prec), set()).add(tuple(lines))
for key, seen in sorted(census.items()):
    if len(seen) != 1:
        problems.append(f"{'-'.join(key)}: the seeding census differs across rounds ({len(seen)} distinct)")
for flag in ("false", "true"):
    seen = census.get((flag, "half"), set())
    for lines in sorted(seen):
        print(log_line(f"untuned {flag}-half census: "
                       f"{display_value(' | '.join(lines)) if lines else 'none logged'}"))
wide, narrow = census.get(("false", "half")), census.get(("true", "half"))
if wide and narrow and len(wide) == len(narrow) == 1:
    print("untuned treatment reach: the two f16 arms' seeding censuses "
          + ("are IDENTICAL -- the treatment does not reach this workload's untuned plan; the f16 "
             "row is a second control" if wide == narrow else "differ"))
rows, shown = [], display_value(label)
for prec in ("half", "single"):
    w, n = ms.get(("false", prec), {}), ms.get(("true", prec), {})
    rounds = sorted(set(w) & set(n))
    if not rounds:
        rows.append((shown, display_value(fixture), WANT[prec], "missing", "missing", "", ""))
        continue
    ratios = [w[k] / n[k] for k in rounds]
    rows.append((shown, display_value(fixture), WANT[prec],
                 f"{statistics.median(n.values()):.3f}", f"{statistics.median(w.values()):.3f}",
                 f"{statistics.median(ratios):.3f} ({min(ratios):.3f}-{max(ratios):.3f})",
                 f"{sum(x > 1 for x in ratios)} of {len(ratios)}"))
print()
print("\n".join(table_lines(
    ["memory", "untuned default (bench_gpt)", "prec", "narrow (true) step p50 ms",
     "wide (false) step p50 ms", "paired wide/narrow (range)", "rounds wide slower"], rows)))
print(f"(paired sign tests and parity losses: {display_value(run)}/report.md)")
for p in problems:
    print(log_line(f"INVALID: untuned: {p}"))
sys.exit(1 if problems else 0)
EOF
exit $FAIL

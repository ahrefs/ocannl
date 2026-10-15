#!/bin/bash
# gh-514 phase 6 evaluation cells, one box per invocation.
# Usage: gh514_cells.sh (--box BOX | --skip-box-check) <backend> <precision>[:<D-cell precision>]
#          <repo-root> [pin-prefix...]
#
# --box BOX: before anything is built, benchmarks/check-box.sh checks BOX and the two fixtures the
# cells load (mlp_wide, gpt2_mini) in <repo-root>'s benchmarks/fixtures, and a GAP refuses the run
# (benchmarks/box-gate.sh; --skip-box-check is for a deliberate partial run). The fixtures are
# never regenerated here: their bytes are the recorded workload (benchmarks/README.md).
#
# <precision> applies to every cell, the untuned mlp D cells included: since the model_default
# gate reaches the loss-scale-gated step shapes too, an f16 D comparison measures the model
# against the default rather than two identical executions.
#
# The form <tuned>:<untuned> runs the D cells at a different precision than the tuned cells, and
# labels its results directory accordingly. `f16:f32` reruns the f32 D cells of
# report-gh514-eval.md, which were measured that way before the gate was extended — but NOT
# under their envelope, since a split precision also withholds the fitted peaks from the D cells
# (the DPEAKS note below). It is a rerun under the class advisories, not a reproduction of the
# tabulated numbers. It is a positional treatment rather than an environment variable, so a
# split precision is visible in the invocation that produced the numbers and cannot be inherited
# by the next run.
#
# Control cells pin their experimental gates (and the tuned cells their search enables) to the
# treatment's values explicitly: command-line settings out-rank every other config source, so an
# ambient profile or config file cannot contaminate the matrix.
set -u
. "$(dirname "$0")/box-gate.sh"
box_gate_args "$@"; shift "$box_gate_shift"
[ $# -ge 3 ] || { sed -n '2,4p' "$0"; exit 2; }
BK=$1; PRECARG=$2; ROOT=$3; shift 3
PIN=("$@")
PREC=${PRECARG%%:*}
DPREC=${PRECARG#*:}  # the whole argument when it carries no colon, i.e. the D cells follow PREC
for p in "$PREC" "$DPREC"; do
  case $p in f32|bf16|f16) ;; *) echo "bad precision: $p (want f32|bf16|f16[:f32|bf16|f16])"; exit 1;; esac
done
[ "$DPREC" = "$PREC" ] || echo "D cells at $DPREC (tuned cells at $PREC)"
# Hermetic against ambient configuration, categorically: every OCANNL_* environment variable is
# unset (each cell pins its treatment on the command line, which out-ranks all other sources),
# and the cells run from benchmarks/, whose checked-in ocannl_config is the NEAREST file on the
# upward config search (Utils.config_file_args find_up) — so neither exported variables nor
# personal ancestor config files can reshape any cell, for any current or future key.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
# The BENCH_* family symmetrically: the helpers set every variable a cell consumes, so ambient
# diagnostic modes (BENCH_SR_SITES and kin) cannot leak into a labeled treatment either.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
box_gate gh514_cells "$ROOT/benchmarks/fixtures" mlp_wide gpt2_mini
FAILED=""
cd "$ROOT" || exit 1
opam exec -- dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
  benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || \
  dune build tools/fit_envelope.exe benchmarks/runners/ocannl/bench_mlp.exe \
    benchmarks/runners/ocannl/bench_gpt.exe 2>/dev/null || {
  echo "BUILD FAILED"
  exit 1
}
cd benchmarks || exit 1
EXE=../_build/default/benchmarks/runners/ocannl/bench_mlp.exe
GPT=../_build/default/benchmarks/runners/ocannl/bench_gpt.exe
FIT=../_build/default/tools/fit_envelope.exe
# A split precision is part of the label: a D-cell rerun at another precision never lands in,
# or overwrites, the matched run's directory.
OUT="$HOME/gh514-eval-results-$BK-$PREC"
[ "$DPREC" = "$PREC" ] || OUT="$OUT-d$DPREC"
rm -rf "$OUT"; mkdir -p "$OUT"

mlp() {
  local name=$1 tune=$2 prec=$3 st; shift 3
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/mlp_wide.safetensors BENCH_TUNE=$tune BENCH_TUNE_REPORT=1 \
    BENCH_PRECISION=$prec BENCH_STATIC_SCALE=0 BENCH_GATE_INTERVAL=0 BENCH_MATERIALIZE=0 \
    BENCH_DEBUG=0 BENCH_TWIN_PLACEMENT=auto BENCH_PRESEED_TWINS=0 BENCH_LOSS_SCALE= \
    BENCH_NO_SGD=0 BENCH_FLIP_DUMP=0 \
    ${PIN[@]+"${PIN[@]}"} "$EXE" --ocannl_backend="$BK" --ocannl_autotune_log=true \
    --ocannl_tf32_matmuls=false --ocannl_fp16_arithmetic=auto \
    --ocannl_narrow_compute_f32=false --ocannl_autotune_split_reduce_max_sites=8 \
    --ocannl_schedule_fission=true --ocannl_virtualize_max_visits=1 \
    --ocannl_automatic_gpu_schedule=true --ocannl_gpu_graph_capture=true \
    --ocannl_default_prec=single --ocannl_autotune_beam_width=2 \
    --ocannl_debug_log_from_routines=false --ocannl_large_models=false \
    --ocannl_autotune_cache_dir="$(mktemp -d)" "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  st=$?
  echo "=== cell $name exit $st $(date +%T) ==="
  [ "$st" -eq 0 ] || FAILED="$FAILED $name"
}
gpt() {
  local name=$1 st; shift
  echo "=== cell $name start $(date +%T) ==="
  BENCH_FIXTURE=fixtures/gpt2_mini.safetensors BENCH_TUNE=0 BENCH_PRECISION=$PREC \
    BENCH_STATIC_SCALE=0 BENCH_GATE_INTERVAL=0 BENCH_MATERIALIZE=0 BENCH_DEBUG=0 \
    ${PIN[@]+"${PIN[@]}"} "$GPT" --ocannl_backend="$BK" --ocannl_autotune_log=true \
    --ocannl_tf32_matmuls=false --ocannl_fp16_arithmetic=auto \
    --ocannl_narrow_compute_f32=false --ocannl_schedule_fission=true \
    --ocannl_virtualize_max_visits=1 --ocannl_automatic_gpu_schedule=true \
    --ocannl_gpu_graph_capture=true --ocannl_default_prec=single \
    --ocannl_debug_log_from_routines=false --ocannl_large_models=false "$@" \
    > "$OUT/$name.out" 2> "$OUT/$name.err"
  st=$?
  echo "=== cell $name exit $st $(date +%T) ==="
  [ "$st" -eq 0 ] || FAILED="$FAILED $name"
}

# A: tuned baseline + calibration ledger (pruning pinned off — it is the B cells' treatment).
mlp A 1 "$PREC" --ocannl_model_peak_flops=0 --ocannl_model_peak_memory_bandwidth=0 \
  --ocannl_autotune_search=true --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
  --ocannl_tune_inline_flips=0 --ocannl_autotune_bound_pruning=false \
  --ocannl_autotune_calibration_file="$OUT/calib.tsv"
# Fit the envelope from A's rows; later cells pin the fitted peaks.
if "$FIT" "$OUT/calib.tsv" > "$OUT/fit.txt" 2>/dev/null; then :; else FAILED="$FAILED fit"; fi
PF=$(grep '^model_peak_flops=' "$OUT/fit.txt" | cut -d= -f2)
PB=$(grep '^model_peak_memory_bandwidth=' "$OUT/fit.txt" | cut -d= -f2)
PEAKS=()
if [ -n "${PF:-}" ]; then PEAKS+=("--ocannl_model_peak_flops=$PF"); else PEAKS+=("--ocannl_model_peak_flops=0"); fi
if [ -n "${PB:-}" ]; then PEAKS+=("--ocannl_model_peak_memory_bandwidth=$PB"); else PEAKS+=("--ocannl_model_peak_memory_bandwidth=0"); fi
echo "fitted peaks: ${PEAKS[*]:-none}"
# The fitted constants are achieved throughputs at the tuned cells' precision; applying them to
# a different-precision compile would shift the roofline balance, so the D cells receive them
# only when they run at the calibrated precision — otherwise they use the backend's class-level
# advisory constants, which is also the deployment default for the untuned regime.
DPEAKS=(--ocannl_model_peak_flops=0 --ocannl_model_peak_memory_bandwidth=0)
[ "$DPREC" = "$PREC" ] && DPEAKS=(${PEAKS[@]+"${PEAKS[@]}"})
# B: tuned, measured-incumbent bound pruning off vs on, fitted envelope.
mlp B-off 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
  --ocannl_tune_inline_flips=0 --ocannl_autotune_bound_pruning=false
mlp B-on 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
  --ocannl_tune_inline_flips=0 --ocannl_autotune_bound_pruning=true
# C: the flip chain at budget 5, legacy cost ordering vs the pure enablement prior vs the shipped
# default (the prior weighed against the arms' measured profitability, gh-ocannl-579), two
# replicates; then enablement + bound pruning.
for r in 1 2; do
  mlp C-cost-$r 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
    --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
    --ocannl_autotune_bound_pruning=false \
    --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=cost
  mlp C-enab-$r 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
    --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
    --ocannl_autotune_bound_pruning=false \
    --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=enablement
  mlp C-prof-$r 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
    --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
    --ocannl_autotune_bound_pruning=false \
    --ocannl_tune_inline_flips=5 --ocannl_tune_flip_ordering=profitable
done
mlp C-enab-bp 1 "$PREC" ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_autotune_search=true \
  --ocannl_autotune_keep_fraction=1 --ocannl_autotune_repeats=3 \
  --ocannl_tune_inline_flips=5 \
  --ocannl_tune_flip_ordering=enablement --ocannl_autotune_bound_pruning=true
# D: the untuned regime — default pipeline vs model_default, +placements, +lattice. At the
# requested precision, or at the argument's `:`-suffix (see the header note); each arm pins the
# gates the treatment does not enable.
mlp D-default 0 "$DPREC" --ocannl_model_peak_flops=0 --ocannl_model_peak_memory_bandwidth=0 \
  --ocannl_model_default_schedule=false
mlp D-model 0 "$DPREC" ${DPEAKS[@]+"${DPEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=false
mlp D-model-plc 0 "$DPREC" ${DPEAKS[@]+"${DPEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_tune_flip_ordering=enablement \
  --ocannl_model_default_placements=5 --ocannl_model_default_geometry_lattice=false
mlp D-model-lat 0 "$DPREC" ${DPEAKS[@]+"${DPEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=true
# D on gpt2_mini (forward-only, requested precision): the bigger sites for the family/lattice
# ledger.
gpt Dg-default --ocannl_model_peak_flops=0 --ocannl_model_peak_memory_bandwidth=0 \
  --ocannl_model_default_schedule=false
gpt Dg-model ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=false
gpt Dg-model-lat ${PEAKS[@]+"${PEAKS[@]}"} --ocannl_model_default_schedule=true \
  --ocannl_model_default_placements=0 --ocannl_model_default_geometry_lattice=true
if [ -n "$FAILED" ]; then
  echo "CELLS FAILED:$FAILED ($BK)"
  exit 1
fi
echo "ALL CELLS DONE ($BK)"

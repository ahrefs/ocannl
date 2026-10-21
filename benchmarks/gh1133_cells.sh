#!/bin/bash
# gh-ocannl-1133 measurement driver: the default GPU schedule's lane plans (every loop of a proved
# chain mapped to hardware) against the base revision's two-loop presets, on the gpt2_mini fixtures
# of gh-ocannl-995 and their training counterparts, so the report quotes invocations rather than
# restating commands (the tables on lukstafi/ocannl-staging#909 and ahrefs/ocannl#1133).
#
# Usage (from anywhere; the "fix" checkout is the one this script lives in):
#   benchmarks/gh1133_cells.sh [--box BOX] [--skip-box-check] OUT BASE CAP STEP...
#
#   --box BOX  required when a STEP measures (step, seg, trainseg, kprof): before OUT is made,
#         benchmarks/check-box.sh checks BOX and FIXTURES in FIXTURE_DIR, and a GAP refuses the
#         run (benchmarks/box-gate.sh; --skip-box-check is for a deliberate partial run).
#   OUT   results directory, created, and refused unless empty. OUT/driver.log records everything
#         this script prints; each cell writes OUT/<cell>.out and OUT/<cell>.err, and `summary`
#         writes OUT/summary.md.
#   BASE  a checkout of the base revision (the two-loop presets), built by the `build` step too.
#   CAP   wall cap in seconds per build and per cell; a capped cell's process group is terminated
#         and the cell counts as failed.
#   STEP  build | provenance | step | seg | trainseg | kprof | summary
#     build       dune build of bench_gpt and bench_gpt_diag in both checkouts, and of this
#                 checkout's bin/device_props (kprof's device probe).
#     provenance  both revisions, clean-tree checks, the fixtures' digests against the m4-max
#                 content-v1 rows of fixtures/DIGESTS.txt, and the host state.
#     step        the step-time matrix: FIXTURES x TREATMENTS, REPEATS passes over the cell list,
#                 forward / reversed / forward, so an order effect splits the repeats of one cell
#                 rather than biasing a treatment.
#     seg         per-fission-segment attribution: bench_gpt_diag with BENCH_SEG_TIMES=1, one cell
#                 per inference fixture x treatment (min-of-20 per segment, a sync per run, so
#                 each segment carries a launch floor and the segments do not add up to a step).
#     trainseg    training-fixture counterpart of seg: the shipped backprop + SGD segments, with
#                 full-step controls BEFORE isolated timing mutates weights/gradients. Requires a
#                 runner with gh-ocannl-1170 (an old forward-only diagnostic is refused).
#     kprof       per-kernel attribution of the step itself (gh-ocannl-1347): FIXTURES x
#                 TREATMENTS, REPEATS passes ordered as `step`'s, each a step cell (bench_gpt, the
#                 per-kernel table on) under a device tracer -- BACKEND=cuda: `nsys profile
#                 --trace=cuda --cuda-graph-trace=node` (nsys and nvidia-smi on PATH), an
#                 nvidia-smi clock sampler beside it; BACKEND=hip (gh-ocannl-1392): `rocprofv3
#                 --kernel-trace` (ROCPROFV3, default rocprofv3 on PATH; benchmarks/rocprofv3-install.sh
#                 provides one where the distribution ships none), an amdgpu sysfs clock sampler
#                 beside it where the box has one amdgpu card (gh-ocannl-1444). On both, a device
#                 probe (bin/device_props) runs first, and every cell states its one device's memory
#                 model, unified or discrete. Every shipped segment's device duration over the timed steps is
#                 joined with the same compile's isolated minimum (benchmarks/kernel_attribution.py,
#                 which refuses a cell it cannot map), and rows where the two disagree, or the
#                 in-step spread is wide, are flagged: only the in-step column attributes step
#                 time. The step time of a kprof cell carries the tracer's overhead (2-3% on rog);
#                 it enters no matrix.
#     summary     OUT/summary.md from the numbered passes' result lines: median p50 per cell, the
#                 p50 of each repeat, the widest p90/p10 of the cell's repeats, the ratio to the
#                 same fixture's base cell, and the losses' largest relative difference from base.
#                 Appends the trainseg census/timing tables beside the step-time matrix, if run,
#                 and kprof's attribution and cross-cell repeatability tables.
#                 An invocation without a `step` step (a prep or dry run: build, provenance,
#                 trainseg, summary) has no matrix to report: summary says so and passes on the
#                 tables alone, while a `step` step whose cells produced no result line fails.
#   A measurement step needs `build` and `provenance` earlier in the same invocation.
#
# Treatments (every cell: untuned default pipeline, f32, schedule fission on, online softmax off,
# one process per cell; the flags are the whole treatment and are recorded in the cell's .err):
#   base      BASE's runner, its defaults
#   fill1     this checkout's runner, --ocannl_gpu_schedule_workgroup_fill=1 (Grid (b, s, h) x
#             Workgroup d for a (b, s, h, d) output)
#   fill256   this checkout's runner, --ocannl_gpu_schedule_workgroup_fill=256 (the loops above the
#             lane join the workgroup while it holds fewer than 256 threads: Grid (b, s) x
#             Workgroup (h, d))
#   keep      this checkout's runner, its defaults (gh-ocannl-1126: schedule-aware fission on)
#   legacy    this checkout's runner, --ocannl_gpu_fission_keep_mapping=false (fission merges
#             whatever the race analysis admits, as before gh-ocannl-1126)
#   pre<M>    this checkout's runner, --ocannl_gpu_lane_preamble_reduction=<M>, M one of refused
#             (the lane geometry refuses a preamble reduction, as before gh-ocannl-1124),
#             duplicated (every lane recomputes it) or cooperative (the lanes all-reduce it) --
#             the arms of gh-ocannl-1124 over the fused backward's dK nest, so a d1pre<M> cell is
#             the one that exercises them; BASE may name this checkout for these
#   serial<S> this checkout's runner, --ocannl_gpu_serial_lanes=<S>, S one of cut (the lane
#             geometry of gh-ocannl-1003 stage 1, fission cutting kernels to keep it), admitted
#             (lanes where a kernel admits them, no cuts for them), off (the plain plans) or auto
#             (per device) -- the same-binary A/B of gh-ocannl-1167; meaningful with a d1 <T>
#   ar<K>     this checkout's runner, --ocannl_gpu_lane_all_reduce_simdgroups=<K>, K a positive
#             count: the most simdgroups the cooperative dp all-reduce may span (gh-ocannl-1168; 1
#             is the one-simdgroup rule, so a head wider than the simdgroup keeps the plain plan;
#             2 gives a head of 64 lanes, 4 a head of 128) -- meaningful with a d1 <T> on a
#             fixture whose head is wider than 32
#   sr<N>     this checkout's runner, --ocannl_gpu_default_split_reduce_min_serial=<N>, N a
#             non-negative count: the serial chain per output cell from which the untuned CUDA/HIP plan
#             splits a reduction (gh-ocannl-1325; 0 splits nothing) -- the same-binary threshold
#             A/B of gh-ocannl-1414, whose two-loop batch reductions chain 1024 rows at batch 8
#   fold<T>   treatment <T> plus --ocannl_online_softmax_block=16 (the single-pass block fold of
#             gh-ocannl-1003, the `approximate` profile's setting; meaningful with a d1 <T>)
#   b<T>      treatment <T> run by BASE's runner, for a same-flags A/B across revisions
#             (gh-ocannl-1124: BASE is the previous revision; <T> is not `base`)
#   d1<T>     treatment <T> (one of the above but base) with the online-softmax forward and the
#             fused attention backward on -- treatment D1 of benchmarks/report-gh1002-fused-backward.md
#             (the others are its treatment A)
# Environment knobs (recorded in driver.log): BACKEND (default metal), REPEATS (default 3),
# FIXTURES (default the three inference and three training fixtures), TREATMENTS (default
# "base fill1 fill256"), FIXTURE_DIR (default benchmarks/fixtures of this checkout), REF (default
# base: the treatment the summary's ratio column divides by; a d1 treatment divides by d1REF),
# KERNEL_TABLE (default 1: every shipped kernel's min-of-20 time, geometry and written nodes go to
# the cell's .err, the per-kernel attribution of gh-ocannl-1002; 0 turns it off -- on by default
# because a lane or fission A/B moves kernels, and one step-time line hid gh-ocannl-1167's wrong
# premise where only the table exposed it; it costs nothing extra: the dominant-kernel instrument
# times every kernel either way, after the timed steps), REF_ALL (default unset: a treatment every
# cell's ratio divides by, overriding REF's d1 pairing -- for the b<T> A/Bs).
#
# Exit: 0 all steps complete; 1 a cell or step failed; 2 usage; 130 interrupted.
# The environment is cleared of OCANNL_*, BENCH_* and the OpenMP controls: every treatment is
# pinned on the command line.
set -u
. "$(dirname "$0")/box-gate.sh"
box_gate_args "$@"; shift "$box_gate_shift"
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(\(OMP\|GOMP\|KMP\)_[A-Z0-9_]*\)=.*/\1/p')

[ $# -ge 4 ] || { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
out=$1 base=$2 cap=$3
shift 3
case $cap in '' | *[!0-9]* | 0*) echo "gh1133: CAP must be a positive decimal integer, got '$cap'" >&2; exit 2 ;; esac
backend=${BACKEND:-metal}
repeats=${REPEATS:-3}
fixtures=${FIXTURES:-"gpt2_mini gpt2_mini_s512 gpt2_mini_s1024 gpt2_mini_train gpt2_mini_train_s512 gpt2_mini_train_s1024"}
treatments=${TREATMENTS:-"base fill1 fill256"}
ref=${REF:-base}
kernel_table=${KERNEL_TABLE:-1}
case $kernel_table in 0 | 1) ;; *) echo "gh1133: KERNEL_TABLE must be 0 or 1, got '$kernel_table'" >&2; exit 2 ;; esac
case $repeats in '' | *[!0-9]* | 0*) echo "gh1133: REPEATS must be a positive decimal integer, got '$repeats'" >&2; exit 2 ;; esac
[ -d "$base" ] || { echo "gh1133: BASE $base is not a directory" >&2; exit 2; }
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh1133: OUT $out is not empty; give each invocation a fresh directory" >&2
  exit 2
fi
root=$(cd "$(dirname "$0")/.." && pwd -P)
fixture_dir=${FIXTURE_DIR:-$root/benchmarks/fixtures}
# The fixtures the requested measuring steps load (the step loop below; box_gate checks a repeat
# once), checked before OUT is made. The cells read FIXTURE_DIR from benchmarks/ (the cd below),
# so a relative one is checked there.
gate_names=""
for gate_step in "$@"; do
  for f in $fixtures; do
    case $gate_step in
      step | kprof) gate_names="$gate_names $f" ;;
      seg) case $f in *train*) ;; *) gate_names="$gate_names $f" ;; esac ;;
      trainseg) case $f in *train*) gate_names="$gate_names $f" ;; esac ;;
    esac
  done
done
case $fixture_dir in /*) gate_dir=$fixture_dir ;; *) gate_dir=$root/benchmarks/$fixture_dir ;; esac
# shellcheck disable=SC2086 # a list of names, one word each
[ -z "$gate_names" ] || box_gate gh1133 "$gate_dir" $gate_names
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
base=$(cd "$base" && pwd -P) || exit 2
exec 3>&1
exec >>"$out/driver.log" 2>&1 || exit 2
trap 'cat "$out/driver.log" >&3' EXIT
trap 'echo "gh1133: interrupted"; exit 130' INT TERM
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2
echo "gh1133: $(date -u +%FT%TZ) root=$root base=$base out=$out cap=$cap backend=$backend repeats=$repeats"
echo "gh1133: fixtures: $fixtures; treatments: $treatments; ref: $ref; kernel table: $kernel_table; fixture dir: $fixture_dir; steps: $*"
built=0 proven=0 failed=0
# kprof: HIP's tracer and amdgpu hwmon directory (its clock sampler's source), and the lines every
# cell's .err states (set by the kprof step).
rocprof= hwmon= kprof_header=
# Whether the step-time matrix was asked for: summary owes one only then (OUT starts empty, so no
# earlier invocation's cells can be in it either way).
step_stage=0
for step in "$@"; do [ "$step" = step ] && step_stage=1; done

# The runner of a treatment: BASE's for base and b<T>, whatever d1 / fold wrappers enclose it
# (flags_of strips the same wrappers), this checkout's otherwise.
runner_root() {
  local t=$1
  while :; do
    case $t in d1*) t=${t#d1} ;; fold*) t=${t#fold} ;; *) break ;; esac
  done
  case $t in base | b[!a]* | bar[1-9]*) echo "$base" ;; *) echo "$root" ;; esac
}

flags_of() {
  case $1 in
    base) echo "--ocannl_online_softmax=false" ;;
    fill*) echo "--ocannl_online_softmax=false --ocannl_gpu_schedule_workgroup_fill=${1#fill}" ;;
    keep) echo "--ocannl_online_softmax=false" ;;
    legacy) echo "--ocannl_online_softmax=false --ocannl_gpu_fission_keep_mapping=false" ;;
    prerefused | preduplicated | precooperative)
      echo "--ocannl_online_softmax=false --ocannl_gpu_lane_preamble_reduction=${1#pre}" ;;
    d1base) echo "gh1133: the base runner takes no d1 form" >&2; return 1 ;;
    serialcut | serialadmitted | serialoff | serialauto)
      echo "--ocannl_online_softmax=false --ocannl_gpu_serial_lanes=${1#serial}" ;;
    ar[1-9]*)
      case ${1#ar} in *[!0-9]*) echo "gh1133: unknown treatment $1" >&2; return 1 ;; esac
      echo "--ocannl_online_softmax=false --ocannl_gpu_lane_all_reduce_simdgroups=${1#ar}" ;;
    sr[0-9]*)
      case ${1#sr} in *[!0-9]*) echo "gh1133: unknown treatment $1" >&2; return 1 ;; esac
      echo "--ocannl_online_softmax=false --ocannl_gpu_default_split_reduce_min_serial=${1#sr}" ;;
    fold*)
      local inner
      inner=$(flags_of "${1#fold}") || return 1
      echo "$inner --ocannl_online_softmax_block=16" ;;
    b[!a]* | bar[1-9]*) flags_of "${1#b}" ;;
    d1*)
      local rest
      rest=$(flags_of "${1#d1}") || return 1
      echo "--ocannl_online_softmax=true --ocannl_online_softmax_backward=true ${rest#--ocannl_online_softmax=false}" ;;
    *) echo "gh1133: unknown treatment $1" >&2; return 1 ;;
  esac
}

# [capped CELL CMD...]: run CMD in its own process group under CAP seconds, stdout/stderr to
# OUT/CELL.out/.err, recording the command, exit and wall (perl's alarm: macOS has no timeout(1)).
capped() {
  local cell=$1 st t0 t1
  shift
  echo "=== cell $cell start $(date -u +%T): $*"
  t0=$(date +%s)
  perl -e 'my $cap = shift @ARGV; my $pid = fork; die "fork: $!" unless defined $pid;
           if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127 }
           sub stop { kill "TERM", -$pid; sleep 2; kill "KILL", -$pid; waitpid $pid, 0; exit shift }
           $SIG{ALRM} = sub { stop(124) };
           $SIG{INT} = $SIG{TERM} = $SIG{HUP} = sub { stop(130) };
           alarm $cap; waitpid $pid, 0; exit($? & 127 ? 128 + ($? & 127) : $? >> 8)' \
    "$cap" "$@" >"$out/$cell.out" 2>>"$out/$cell.err"
  st=$?
  printf "%s\n" "$st" >"$out/$cell.exit"
  t1=$(date +%s)
  echo "=== cell $cell exit $st wall $((t1 - t0))s"
  [ "$st" -eq 0 ] || failed=1
  return "$st"
}

# [cell FIXTURE TREATMENT TAG [EXE [profile]]]: with `profile`, the runner runs under the backend's
# tracer. On CUDA that is nsys (the CUDA graph traced node by node, so each segment of the replayed
# step is its own record), writing OUT/CELL.nsys-rep, and an nvidia-smi sampler records
# OUT/CELL.smi.csv for the cell's duration. On HIP it is rocprofv3, whose kernel trace records each
# dispatch at the HSA queue (graph nodes included) in OUT/CELL_kernel_trace.csv, beside
# OUT/CELL_agent_info.csv, and an amdgpu sysfs sampler records OUT/CELL.clk.csv. Either way the
# cell's .err states KPROF_HEADER (the tracer and the memory model).
cell() {
  local fixture=$1 treatment=$2 tag=$3 exe=${4:-bench_gpt} profile=${5:-} flags r st smi=
  flags=$(flags_of "$treatment") || { failed=1; return 1; }
  r=$(runner_root "$treatment")
  local name="$backend-$fixture-$treatment-$tag" wrap=()
  echo "backend=$backend fixture=$fixture treatment=$treatment runner=$r flags=[$flags]" >"$out/$name.err"
  [ -z "$profile" ] || printf '%s\n' "$kprof_header" >>"$out/$name.err"
  if [ -n "$profile" ] && [ "$backend" = hip ]; then
    wrap=("$rocprof" --kernel-trace --output-format csv -d "$out" -o "$name" --)
    if [ -n "$hwmon" ]; then
      amdgpu_sampler "$hwmon" >"$out/$name.clk.csv" 2>&1 &
      smi=$!
    fi
  elif [ -n "$profile" ]; then
    wrap=(nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none
      --force-overwrite=true -o "$out/$name")
    nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,power.draw,temperature.gpu,clocks_event_reasons.active \
      --format=csv -lms 250 >"$out/$name.smi.csv" 2>&1 &
    smi=$!
  fi
  # shellcheck disable=SC2086 # the treatment's flags are separate words by construction
  BENCH_FIXTURE="$fixture_dir/$fixture.safetensors" BENCH_TUNE=0 BENCH_MATERIALIZE=0 \
    BENCH_DEBUG=0 BENCH_SEG_TIMES="${SEG:-0}" BENCH_KERNEL_TABLE="$kernel_table" \
    capped "$name" ${wrap[@]+"${wrap[@]}"} "$r/_build/default/benchmarks/runners/ocannl/$exe.exe" \
    --ocannl_backend="$backend" --ocannl_default_prec=single \
    --ocannl_schedule_fission=true --ocannl_automatic_gpu_schedule=true \
    --ocannl_autotune_search=false --ocannl_debug_log_from_routines=false \
    --ocannl_output_debug_files_in_build_directory=false $flags
  st=$?
  if [ -n "$smi" ]; then
    kill "$smi" 2>/dev/null
    wait "$smi" 2>/dev/null
  fi
  return "$st"
}

# The cell list in pass order: forward on odd passes, reversed on even ones.
pass_cells() {
  local pass=$1 list="" f t
  for f in $2; do for t in $treatments; do list="$list $f:$t"; done; done
  if [ $((pass % 2)) -eq 0 ]; then echo "$list" | tr ' ' '\n' | sed '/^$/d' | awk '{ l[NR] = $0 } END { for (i = NR; i > 0; i--) print l[i] }'
  else echo "$list" | tr ' ' '\n' | sed '/^$/d'; fi
}

# [memory_model]: the memory model of the backend's one device, from this checkout's device probe
# (bin/device_props: the backend's own `integrated` query -- on HIP the one its hardware limits
# read), as `unified|discrete (NAME, ARCH: integrated=BOOL)`, ARCH the gcn arch on HIP and sm_<cc>
# on CUDA. Fails, saying why, on a probe that does not run, sees other than one device (the cell
# would not say which it ran on; HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES pins one), or reports no
# `integrated` flag (a checkout older than gh-ocannl-1392 on HIP, gh-ocannl-1444 on CUDA).
memory_model() {
  local props n integrated model arch
  props=$("$root/_build/default/bin/device_props.exe" --ocannl_backend="$backend" </dev/null 2>/dev/null) || {
    echo "gh1133: the device probe $root/_build/default/bin/device_props.exe failed"
    return 1
  }
  n=$(printf '%s\n' "$props" | sed -n 's/^static\.device_count = //p')
  if [ "$n" != 1 ]; then
    echo "gh1133: kprof states the memory model of the one device it ran on, and the probe sees ${n:-no} $backend devices (pin one with HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES)"
    return 1
  fi
  integrated=$(printf '%s\n' "$props" | sed -n 's/^static\.device\[0\]\.integrated = //p')
  case $integrated in
    true) model=unified ;;
    false) model=discrete ;;
    *) echo "gh1133: the device probe reports no integrated flag for device 0; the memory model is unknown"; return 1 ;;
  esac
  if [ "$backend" = hip ]; then
    arch=$(printf '%s\n' "$props" | sed -n 's/^static\.device\[0\]\.gcn_arch_name = //p')
  else
    arch=sm_$(printf '%s\n' "$props" | sed -n 's/^static\.device\[0\]\.compute_capability_major = //p')$(printf '%s\n' "$props" | sed -n 's/^static\.device\[0\]\.compute_capability_minor = //p')
  fi
  echo "$model ($(printf '%s\n' "$props" | sed -n 's/^static\.device\[0\]\.device_name = //p'), $arch: integrated=$integrated)"
}

# [amdgpu_hwmon]: the hwmon directory of the box's one amdgpu card that reports its shader clock
# (freq1_input), or nothing -- saying why on stderr -- when there is none or more than one (the
# sampler could not say which card the step ran on).
amdgpu_hwmon() {
  local found=() f
  for f in /sys/class/drm/card[0-9]*/device/hwmon/hwmon*/freq1_input; do
    [ -r "$f" ] && [ "$(cat "${f%/freq1_input}/name" 2>/dev/null)" = amdgpu ] && found+=("${f%/freq1_input}")
  done
  if [ "${#found[@]}" -ne 1 ]; then
    echo "gh1133: no amdgpu clock sampler: ${#found[@]} amdgpu cards report a shader clock" >&2
    return 0
  fi
  echo "${found[0]}"
}

# [amdgpu_sampler HWMON]: every 250 ms until killed, one CSV row of the card's shader clock (MHz),
# average power (W), edge temperature (C) and busy percentage -- the HIP counterpart of the
# nvidia-smi sampler; a reading the card does not offer is left empty.
amdgpu_sampler() {
  local h=$1 dev
  dev=$(cd "$1/../.." && pwd -P)
  echo "timestamp_ns,sclk_mhz,power_w,temp_c,busy_pct"
  while :; do
    awk -v t="$(date +%s%N)" -v f="$(cat "$h/freq1_input" 2>/dev/null)" \
      -v p="$(cat "$h/power1_average" 2>/dev/null || cat "$h/power1_input" 2>/dev/null)" \
      -v c="$(cat "$h/temp1_input" 2>/dev/null)" -v b="$(cat "$dev/gpu_busy_percent" 2>/dev/null)" \
      'BEGIN { printf "%s,%s,%s,%s,%s\n", t, f == "" ? "" : int(f / 1e6), p == "" ? "" : p / 1e6, c == "" ? "" : c / 1e3, b }'
    sleep 0.25
  done
}

need_identity() {
  [ "$built" = 1 ] && [ "$proven" = 1 ] && return 0
  echo "gh1133: step $1 needs build and provenance earlier in the same invocation"
  failed=1
  return 1
}

# [m4max_fixtures F...]: whether every fixture F (a name in FIXTURE_DIR) is m4-max's recorded
# bytes, decided by fixture_digest.py's exit status -- never by parsing its report, whose origin
# field lists every box that agrees (`m4-max,tuf's bytes`) -- with the report appended to
# OUT/fixtures.txt (OUT starts empty, so a later step's check extends the provenance one's). No
# names is a failure: given no paths, fixture_digest.py checks every fixture in its default dir.
m4max_fixtures() {
  local f paths=()
  [ $# -gt 0 ] || { echo "${0##*/}: no fixtures named to check"; return 1; }
  for f in "$@"; do paths+=("$fixture_dir/$f.safetensors"); done
  python3 fixture_digest.py --check --expect-origin m4-max "${paths[@]}" | tee -a "$out/fixtures.txt"
  return "${PIPESTATUS[0]}"
}

clean_tree() {
  local tree=$1 exclude=()
  case $out/ in "$tree"/*) exclude=(":(exclude)${out#"$tree"/}") ;; esac
  echo "revision ($tree): $(git -C "$tree" rev-parse HEAD)"
  if [ -n "$(git -C "$tree" status --porcelain -- . ${exclude[@]+"${exclude[@]}"})" ]; then
    echo "gh1133: $tree has uncommitted or untracked files; the measurement would not name its code"
    git -C "$tree" status --short -- . ${exclude[@]+"${exclude[@]}"}
    return 1
  fi
}

for step in "$@"; do
  echo "--- step $step $(date -u +%T)"
  case $step in
    build)
      ok=1
      for tree in "$root" "$base"; do
        probe=()
        [ "$tree" = "$root" ] && probe=(bin/device_props.exe)
        capped "build-$(basename "$tree")" dune build --root "$tree" \
          benchmarks/runners/ocannl/bench_gpt.exe benchmarks/runners/ocannl/bench_gpt_diag.exe \
          ${probe[@]+"${probe[@]}"} || ok=0
      done
      [ "$ok" = 1 ] && built=1 ;;
    provenance)
      ok=1
      clean_tree "$root" || ok=0
      clean_tree "$base" || ok=0
      for f in $fixtures; do
        [ -f "$fixture_dir/$f.safetensors" ] || { echo "gh1133: missing fixture $f"; ok=0; }
      done
      # shellcheck disable=SC2086 # FIXTURES is a list of names, one word each
      if [ "$ok" = 1 ] && m4max_fixtures $fixtures; then
        proven=1
      else
        echo "gh1133: a tree is not clean or the fixtures do not match the m4-max content-v1 records"
        failed=1
      fi
      uname -a; sw_vers 2>/dev/null
      echo "load: $(uptime)"
      ps -Ao pid,pcpu,comm -r 2>/dev/null | head -8 || ps -eo pid,pcpu,comm --sort=-pcpu | head -8 ;;
    step)
      need_identity step || continue
      for pass in $(seq 1 "$repeats"); do
        for c in $(pass_cells "$pass" "$fixtures"); do cell "${c%%:*}" "${c#*:}" "r$pass"; done
      done ;;
    seg)
      need_identity seg || continue
      for f in $fixtures; do
        case $f in *train*) continue ;; esac
        for t in $treatments; do SEG=1 BENCH_STEPS=1 cell "$f" "$t" seg bench_gpt_diag; done
      done ;;
    trainseg)
      need_identity trainseg || continue
      for f in $fixtures; do
        case $f in *train*) ;; *) continue ;; esac
        for t in $treatments; do
          SEG=1 BENCH_STEPS=1 cell "$f" "$t" trainseg bench_gpt_diag || continue
          name="$backend-$f-$t-trainseg"
          python3 "$root/benchmarks/gh1133_summary.py" --check-training "$out" "$name.out" || failed=1
        done
      done ;;
    kprof)
      need_identity kprof || continue
      case $backend in
        cuda)
          if ! command -v nsys >/dev/null || ! command -v nvidia-smi >/dev/null; then
            echo "gh1133: kprof on CUDA needs nsys and nvidia-smi on PATH"
            failed=1
            continue
          fi
          model=$(memory_model) || { echo "$model"; failed=1; continue; }
          kprof_header="kprof: tracer $(command -v nsys) --cuda-graph-trace=node ($(nsys --version 2>&1 | head -1))
kprof: memory model $model"
          printf '%s\n' "$kprof_header" ;;
        hip)
          rocprof=${ROCPROFV3:-rocprofv3}
          if ! command -v "$rocprof" >/dev/null; then
            echo "gh1133: kprof on HIP needs rocprofv3 (ROCPROFV3, or on PATH; benchmarks/rocprofv3-install.sh installs one)"
            failed=1
            continue
          fi
          model=$(memory_model) || { echo "$model"; failed=1; continue; }
          hwmon=$(amdgpu_hwmon)
          kprof_header="kprof: tracer $(command -v "$rocprof") --kernel-trace ($("$rocprof" --version 2>&1 | sed -n 's/^ *version: *//p' | head -1))
kprof: memory model $model
kprof: clock sampler ${hwmon:-none}"
          printf '%s\n' "$kprof_header" ;;
        *)
          echo "gh1133: kprof needs BACKEND=cuda (nsys) or BACKEND=hip (rocprofv3), not $backend"
          failed=1
          continue ;;
      esac
      for pass in $(seq 1 "$repeats"); do
        for c in $(pass_cells "$pass" "$fixtures"); do
          name="$backend-${c%%:*}-${c#*:}-kprof$pass"
          cell "${c%%:*}" "${c#*:}" "kprof$pass" bench_gpt profile || continue
          if [ "$backend" = cuda ]; then
            # The per-launch table of the trace; the SQLite export it goes through is not kept.
            nsys stats --report cuda_gpu_trace --format csv --force-export=true \
              --output "$out/$name" "$out/$name.nsys-rep" >>"$out/$name.err" 2>&1 || failed=1
            rm -f "${out:?}/$name.sqlite"
          fi
          python3 "$root/benchmarks/kernel_attribution.py" check "$out" "$name.out" || failed=1
        done
      done ;;
    summary)
      matrix=()
      [ "$step_stage" = 1 ] || matrix=(--no-step-stage)
      python3 "$root/benchmarks/gh1133_summary.py" ${matrix[@]+"${matrix[@]}"} \
        "$out" "$treatments" "$ref" "${REF_ALL:-}" >"$out/summary.md" || failed=1
      if ls "$out"/*-kprof[0-9]*.out >/dev/null 2>&1; then
        python3 "$root/benchmarks/kernel_attribution.py" report "$out" >>"$out/summary.md" || failed=1
      fi
      cat "$out/summary.md" ;;
    *) echo "gh1133: unknown step $step"; failed=1 ;;
  esac
done
echo "gh1133: done $(date -u +%FT%TZ) failed=$failed"
exit "$failed"

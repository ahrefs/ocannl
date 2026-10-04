#!/bin/bash
# gh-ocannl-1003 block-fold measurement driver: the attention forms of the online-softmax rewrite
# on the gpt2_mini inference fixtures, so the report quotes invocations rather than restating
# commands (benchmarks/report-gh1003-block-fold.md).
#
# Usage (from anywhere; the checkout is the one this script lives in):
#   benchmarks/gh1003_fold_cells.sh OUT CAP STEP...
#
#   OUT   results directory, created, and refused unless empty. OUT/driver.log records everything
#         this script prints (provenance, each cell's command, exit and wall); each cell also writes
#         OUT/<cell>.out and OUT/<cell>.err, and `summary` writes OUT/summary.md.
#   CAP   wall cap in seconds per build and per cell; a capped cell's process group is terminated
#         and the cell counts as failed.
#   STEP  build | provenance | dry | metal | seg | cc | train | summary
#     build       dune build of bench_gpt and bench_gpt_diag (the tree as checked out).
#     provenance  the revision, a clean-tree check, the fixtures' digests against the m4-max
#                 content-v1 rows of fixtures/DIGESTS.txt, and the host state.
#     dry         one Metal cell per treatment on gpt2_mini, one repeat: a smoke of the matrix.
#     metal       the step-time matrix: 3 fixtures x the treatments below, REPEATS passes over the
#                 cell list, forward / reversed / forward, so an order effect splits the repeats of
#                 one cell rather than biasing a treatment.
#     seg         per-fission-segment attribution on Metal: bench_gpt_diag with BENCH_SEG_TIMES=1,
#                 one cell per fixture x treatment (min-of-20 per segment, a sync per run).
#     cc          the no-regression leg on the CPU backend: gpt2_mini and gpt2_mini_s512 x the
#                 treatments, two passes (forward, reversed).
#     train       the training step on Metal (TRAIN_FIXTURES, default gpt2_mini_train_s512 and _s1024,
#                 two passes forward/reversed): composed, two-pass-bwd (online forward, fused
#                 backward: treatment D) and fold-B-bwd (the fold forward under the fused backward:
#                 treatment F). Needs the gh-ocannl-1002 measurement's runner fix and workloads.
#     summary     OUT/summary.md from the numbered passes' result lines (not the dry cells): median
#                 p50 per cell, the p50 of each repeat, the widest p90/p10 of the cell's repeats, the
#                 ratio to the same fixture's composed cell and to its two-pass cell, the shipped mma
#                 census from `shipped_mma` for tuned and untuned compiled steps (falling back to
#                 `tune.shipped_mma` for older tuned records), and the
#                 losses' agreement with the composed cell.
#   A measurement step needs `build` and `provenance` earlier in the same invocation: _build/ is
#   ignored by git, so the clean-tree check cannot vouch for binaries an earlier checkout left
#   there, and a measurement is only evidence beside the identity it was taken on.
#
# Treatments (every cell: untuned default pipeline, f32, schedule fission on, one process per
# cell; the flags are the whole treatment and are recorded in the cell's .err header):
#   composed   the defaults
#   two-pass   --ocannl_online_softmax=true (head width 32 is above the recompute cap 16, so the
#              scores stay one stored [seq, seq] buffer, read by the scan and the value pass)
#   fold-B     --ocannl_online_softmax=true --ocannl_online_softmax_block=B, B in FOLD_BLOCKS
# Environment knobs (the driver's own, recorded in driver.log): REPEATS (default 3), FOLD_BLOCKS
# (default "8 16 32"), TRAIN_FIXTURES (the train step's), FIXTURE_DIR (default
# benchmarks/fixtures).
#
# Exit: 0 all steps complete; 1 a cell or step failed; 2 usage; 130 interrupted.
# The environment is cleared of OCANNL_*, BENCH_* and the OpenMP controls, as in gh834_cells.sh:
# every treatment is pinned on the command line.
set -u
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(\(OMP\|GOMP\|KMP\)_[A-Z0-9_]*\)=.*/\1/p')

[ $# -ge 3 ] || { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
out=$1 cap=$2
shift 2
case $cap in '' | *[!0-9]* | 0*) echo "gh1003: CAP must be a positive decimal integer, got '$cap'" >&2; exit 2 ;; esac
repeats=${REPEATS:-3}
fold_blocks=${FOLD_BLOCKS:-"8 16 32"}
train_fixtures=${TRAIN_FIXTURES:-"gpt2_mini_train_s512 gpt2_mini_train_s1024"}
case $repeats in '' | *[!0-9]* | 0*) echo "gh1003: REPEATS must be a positive decimal integer, got '$repeats'" >&2; exit 2 ;; esac
for b in $fold_blocks; do
  case $b in '' | *[!0-9]* | 0*) echo "gh1003: FOLD_BLOCKS must list positive decimal integers, got '$b'" >&2; exit 2 ;; esac
done
[ -n "$fold_blocks" ] || { echo "gh1003: FOLD_BLOCKS is empty" >&2; exit 2; }
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh1003: OUT $out is not empty; give each invocation a fresh directory" >&2
  exit 2
fi
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
exec 3>&1
exec >>"$out/driver.log" 2>&1 || exit 2
trap 'cat "$out/driver.log" >&3' EXIT
trap 'echo "gh1003: interrupted"; exit 130' INT TERM
root=$(cd "$(dirname "$0")/.." && pwd -P)
fixture_dir=${FIXTURE_DIR:-$root/benchmarks/fixtures}
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2
gpt=$root/_build/default/benchmarks/runners/ocannl/bench_gpt.exe
diag=$root/_build/default/benchmarks/runners/ocannl/bench_gpt_diag.exe
fixtures="gpt2_mini gpt2_mini_s512 gpt2_mini_s1024"
treatments="composed two-pass"
for b in $fold_blocks; do treatments="$treatments fold-$b"; done
# The training step's treatments (the record's A, D and F): composed, the two-pass forward under
# the fused backward, and the fold forward under the fused backward.
train_treatments="composed two-pass-bwd"
for b in $fold_blocks; do train_treatments="$train_treatments fold-$b-bwd"; done
echo "gh1003: $(date -u +%FT%TZ) root=$root out=$out cap=$cap repeats=$repeats"
echo "gh1003: fold blocks: $fold_blocks; train fixtures: $train_fixtures; fixture dir: $fixture_dir; steps: $*"
built=0 proven=0 failed=0

flags_of() {
  case $1 in
    composed) ;;
    two-pass) echo "--ocannl_online_softmax=true" ;;
    two-pass-bwd) echo "--ocannl_online_softmax=true --ocannl_online_softmax_backward=true" ;;
    fold-*-bwd)
      local b=${1#fold-}
      echo "--ocannl_online_softmax=true --ocannl_online_softmax_backward=true --ocannl_online_softmax_block=${b%-bwd}" ;;
    fold-*) echo "--ocannl_online_softmax=true --ocannl_online_softmax_block=${1#fold-}" ;;
    *) echo "gh1003: unknown treatment $1" >&2; return 1 ;;
  esac
}

# [capped CELL CMD...]: run CMD in its own process group under CAP seconds, stdout/stderr to
# OUT/CELL.out/.err, recording the command, exit and wall.
capped() {
  local cell=$1 st t0 t1
  shift
  echo "=== cell $cell start $(date -u +%T): $*"
  t0=$(date +%s)
  # perl's alarm: macOS has no timeout(1). The cell runs in a process group of its own, which the
  # supervisor (outside it) terminates whole and then kills, so a cell ignoring TERM cannot outlive
  # its cap into the next cell's timing -- and on an interrupt of the supervisor too (Ctrl-C
  # reaches it in the foreground group; the detached cell group would not see it).
  perl -e 'my $cap = shift @ARGV; my $pid = fork; die "fork: $!" unless defined $pid;
           if ($pid == 0) { setpgrp(0, 0); exec @ARGV or exit 127 }
           sub stop { kill "TERM", -$pid; sleep 2; kill "KILL", -$pid; waitpid $pid, 0; exit shift }
           $SIG{ALRM} = sub { stop(124) };
           $SIG{INT} = $SIG{TERM} = $SIG{HUP} = sub { stop(130) };
           alarm $cap; waitpid $pid, 0; exit($? & 127 ? 128 + ($? & 127) : $? >> 8)' \
    "$cap" "$@" >"$out/$cell.out" 2>>"$out/$cell.err"
  st=$?
  t1=$(date +%s)
  echo "=== cell $cell exit $st wall $((t1 - t0))s"
  [ "$st" -eq 0 ] || failed=1
  return "$st"
}

cell() {
  local backend=$1 fixture=$2 treatment=$3 tag=$4 exe=${5:-$gpt} flags
  flags=$(flags_of "$treatment") || { failed=1; return 1; }
  local name="$backend-$fixture-$treatment-$tag"
  echo "backend=$backend fixture=$fixture treatment=$treatment flags=[$flags]" >"$out/$name.err"
  # shellcheck disable=SC2086 # the treatment's flags are separate words by construction
  BENCH_FIXTURE="$fixture_dir/$fixture.safetensors" BENCH_TUNE=0 BENCH_MATERIALIZE=0 \
    BENCH_DEBUG=0 BENCH_SEG_TIMES="${SEG:-0}" \
    capped "$name" "$exe" --ocannl_backend="$backend" --ocannl_default_prec=single \
    --ocannl_schedule_fission=true --ocannl_automatic_gpu_schedule=true \
    --ocannl_autotune_search=false --ocannl_debug_log_from_routines=false \
    --ocannl_output_debug_files_in_build_directory=false $flags
}

# The cell list in pass order: forward on odd passes, reversed on even ones.
pass_cells() {
  local pass=$1 list="" f t
  for f in $2; do for t in ${3:-$treatments}; do list="$list $f:$t"; done; done
  # Reversed with awk: BSD tail -r does not exist on GNU hosts.
  if [ $((pass % 2)) -eq 0 ]; then echo "$list" | tr ' ' '\n' | sed '/^$/d' | awk '{ l[NR] = $0 } END { for (i = NR; i > 0; i--) print l[i] }'
  else echo "$list" | tr ' ' '\n' | sed '/^$/d'; fi
}

need_identity() {
  [ "$built" = 1 ] && [ "$proven" = 1 ] && return 0
  echo "gh1003: step $1 needs build and provenance earlier in the same invocation"
  failed=1
  return 1
}

for step in "$@"; do
  echo "--- step $step $(date -u +%T)"
  case $step in
    build)
      if capped build dune build --root "$root" benchmarks/runners/ocannl/bench_gpt.exe \
        benchmarks/runners/ocannl/bench_gpt_diag.exe; then built=1; fi ;;
    provenance)
      echo "revision: $(git -C "$root" rev-parse HEAD)"
      # Untracked files count too (an untracked source or dune file changes what builds); only
      # OUT itself is excluded when it lies inside the checkout.
      exclude=()
      case $out/ in "$root"/*) exclude=(":(exclude)${out#"$root"/}") ;; esac
      if [ -n "$(git -C "$root" status --porcelain -- . ${exclude[@]+"${exclude[@]}"})" ]; then
        echo "gh1003: the tree has uncommitted or untracked files; the measurement would not name its code"
        git -C "$root" status --short -- . ${exclude[@]+"${exclude[@]}"}
        failed=1
      else
        ok=1
        for f in $fixtures; do
          [ -f "$fixture_dir/$f.safetensors" ] || { echo "gh1003: missing fixture $f"; ok=0; }
        done
        if [ "$ok" = 1 ] && python3 fixture_digest.py --check \
          $(for f in $fixtures; do echo "$fixture_dir/$f.safetensors"; done) |
          tee "$out/fixtures.txt" && [ "$(grep -c "MATCH — m4-max's bytes" "$out/fixtures.txt")" = 3 ]; then
          proven=1
        else
          echo "gh1003: fixtures do not match the m4-max content-v1 records"
          failed=1
        fi
      fi
      uname -a; sw_vers 2>/dev/null
      echo "load: $(uptime)"
      ps -Ao pid,pcpu,comm -r | head -8 ;;
    dry)
      need_identity dry || continue
      for t in $treatments; do cell metal gpt2_mini "$t" dry; done ;;
    metal)
      need_identity metal || continue
      for pass in $(seq 1 "$repeats"); do
        for c in $(pass_cells "$pass" "$fixtures"); do cell metal "${c%%:*}" "${c#*:}" "r$pass"; done
      done ;;
    seg)
      need_identity seg || continue
      for f in $fixtures; do for t in $treatments; do SEG=1 cell metal "$f" "$t" seg "$diag"; done; done ;;
    cc)
      need_identity cc || continue
      for pass in 1 2; do
        for c in $(pass_cells "$pass" "gpt2_mini gpt2_mini_s512"); do cell cc "${c%%:*}" "${c#*:}" "r$pass"; done
      done ;;
    train)
      # The training step needs the Metal-linkable runner and the train_s512/_s1024 workloads of
      # the gh-ocannl-1002 measurement (lukstafi/ocannl-staging, gh1002 measurement PR); the
      # fixtures must match their m4-max records like the inference ones.
      need_identity train || continue
      ok=1
      for f in $train_fixtures; do
        [ -f "$fixture_dir/$f.safetensors" ] || { echo "gh1003: missing fixture $f"; ok=0; }
      done
      if [ "$ok" = 1 ] && python3 fixture_digest.py --check \
        $(for f in $train_fixtures; do echo "$fixture_dir/$f.safetensors"; done) |
        tee -a "$out/fixtures.txt" | grep -c "MATCH — m4-max's bytes" | grep -qx "$(echo $train_fixtures | wc -w | tr -d ' ')"; then
        for pass in 1 2; do
          for c in $(pass_cells "$pass" "$train_fixtures" "$train_treatments"); do cell metal "${c%%:*}" "${c#*:}" "r$pass"; done
        done
      else
        echo "gh1003: training fixtures missing or not the m4-max records"; failed=1
      fi ;;
    summary)
      python3 - "$out" "$treatments $train_treatments" >"$out/summary.md" <<'PY' || failed=1
import json, os, re, statistics, sys
out, treatments = sys.argv[1], sys.argv[2].split()
cells = {}
missing = []
for name in sorted(os.listdir(out)):
    # The dry cells are a smoke of the matrix, never a repeat: only the numbered passes enter.
    m = re.fullmatch(r"(cc|metal)-(gpt2_mini\w*)-(composed|two-pass(?:-bwd)?|fold-\d+(?:-bwd)?)-(r\d+)\.out", name)
    if not m:
        continue
    rec = None
    for line in open(os.path.join(out, name)):
        line = line.strip()
        if line.startswith("{") and '"step_ms"' in line:
            rec = json.loads(line)
    if rec is None:
        missing.append(name)
        continue
    cells.setdefault(m.group(1, 2, 3), []).append((m.group(4), rec))
# A summary is only as complete as its cells: a numbered cell without a result line, or no
# numbered cell at all, fails the step (the table is still printed for what exists).
incomplete = bool(missing) or not cells
for name in missing:
    print("MISSING RESULT: %s has no result line" % name)
if not cells:
    print("NO MEASUREMENT: no numbered cell produced a result line")
print("| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | p10..p90 spread | shipped mma | loss vs composed |")
print("|---|---|---|---|---|---|---|---|---|---|")
def med(key):
    reps = cells.get(key, [])
    return statistics.median(r["step_ms"]["p50"] for _, r in reps) if reps else None
for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]))):
    p50s = [r["step_ms"]["p50"] for _, r in reps]
    m = statistics.median(p50s)
    spread = max(r["step_ms"]["p90"] / r["step_ms"]["p10"] for _, r in reps)
    comp = med((backend, fixture, "composed"))
    two = med((backend, fixture, "two-pass")) or med((backend, fixture, "two-pass-bwd"))
    mma = reps[0][1].get("shipped_mma") or (reps[0][1].get("tune") or {}).get("shipped_mma")
    loss = ""
    cref = cells.get((backend, fixture, "composed"))
    if cref:
        a, b = reps[0][1].get("losses") or [], cref[0][1].get("losses") or []
        if a and b and len(a) == len(b):
            loss = "%.2g" % max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(a, b))
    print("| %s | %s | %s | %s | %.1f | %s | %s | %.3fx | %s | %s |" % (
        backend, fixture, treatment, ", ".join("%.1f" % p for p in p50s), m,
        "%.3fx" % (m / comp) if comp else "", "%.3fx" % (m / two) if two else "",
        spread, json.dumps(mma) if mma is not None else "", loss))
sys.exit(1 if incomplete else 0)
PY
      cat "$out/summary.md" ;;
    *) echo "gh1003: unknown step $step"; failed=1 ;;
  esac
done
echo "gh1003: done $(date -u +%FT%TZ) failed=$failed"
exit "$failed"

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
#   STEP  build | provenance | dry | metal | seg | cc | summary
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
#     summary     OUT/summary.md from every result line in OUT: median p50 per cell, the p50 of each
#                 repeat, the widest p90/p10 of the cell's repeats, the ratio to the same fixture's
#                 composed cell and to its two-pass cell, the shipped mma census, and the losses'
#                 agreement with the composed cell.
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
# (default "8 16 32"), FIXTURE_DIR (default benchmarks/fixtures).
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
echo "gh1003: $(date -u +%FT%TZ) root=$root out=$out cap=$cap repeats=$repeats"
echo "gh1003: fold blocks: $fold_blocks; fixture dir: $fixture_dir; steps: $*"
built=0 proven=0 failed=0

flags_of() {
  case $1 in
    composed) ;;
    two-pass) echo "--ocannl_online_softmax=true" ;;
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
  # perl's alarm: macOS has no timeout(1). setpgrp so the cap terminates the whole group.
  perl -e 'setpgrp(0, 0); $SIG{ALRM} = sub { kill "TERM", -$$; sleep 2; kill "KILL", -$$; exit 124 };
           alarm shift @ARGV; my $pid = fork; if ($pid == 0) { exec @ARGV } waitpid $pid, 0;
           exit($? >> 8)' "$cap" "$@" >"$out/$cell.out" 2>>"$out/$cell.err"
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
  for f in $2; do for t in $treatments; do list="$list $f:$t"; done; done
  if [ $((pass % 2)) -eq 0 ]; then echo "$list" | tr ' ' '\n' | sed '/^$/d' | tail -r
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
      if [ -n "$(git -C "$root" status --porcelain --untracked-files=no)" ]; then
        echo "gh1003: the tree has uncommitted changes; the measurement would not name its code"
        git -C "$root" status --short --untracked-files=no
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
    summary)
      python3 - "$out" "$treatments" >"$out/summary.md" <<'PY' || failed=1
import json, os, re, statistics, sys
out, treatments = sys.argv[1], sys.argv[2].split()
cells = {}
for name in sorted(os.listdir(out)):
    m = re.fullmatch(r"(cc|metal)-(gpt2_mini\w*)-(composed|two-pass|fold-\d+)-(r\d+|dry)\.out", name)
    if not m:
        continue
    rec = None
    for line in open(os.path.join(out, name)):
        line = line.strip()
        if line.startswith("{") and '"step_ms"' in line:
            rec = json.loads(line)
    if rec is None:
        continue
    cells.setdefault(m.group(1, 2, 3), []).append((m.group(4), rec))
print("| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs composed | vs two-pass | p10..p90 spread | shipped mma | loss vs composed |")
print("|---|---|---|---|---|---|---|---|---|---|")
def med(key):
    reps = cells.get(key, [])
    return statistics.median(r["step_ms"]["p50"] for _, r in reps) if reps else None
for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]))):
    p50s = [r["step_ms"]["p50"] for _, r in reps]
    m = statistics.median(p50s)
    spread = max(r["step_ms"]["p90"] / r["step_ms"]["p10"] for _, r in reps)
    comp, two = med((backend, fixture, "composed")), med((backend, fixture, "two-pass"))
    mma = (reps[0][1].get("tune") or {}).get("shipped_mma")
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
PY
      cat "$out/summary.md" ;;
    *) echo "gh1003: unknown step $step"; failed=1 ;;
  esac
done
echo "gh1003: done $(date -u +%FT%TZ) failed=$failed"
exit "$failed"

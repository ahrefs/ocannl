#!/bin/bash
# gh-ocannl-1133 measurement driver: the default GPU schedule's lane plans (every loop of a proved
# chain mapped to hardware) against the base revision's two-loop presets, on the gpt2_mini fixtures
# of gh-ocannl-995 and their training counterparts, so the report quotes invocations rather than
# restating commands (the tables on lukstafi/ocannl-staging#909 and ahrefs/ocannl#1133).
#
# Usage (from anywhere; the "fix" checkout is the one this script lives in):
#   benchmarks/gh1133_cells.sh OUT BASE CAP STEP...
#
#   OUT   results directory, created, and refused unless empty. OUT/driver.log records everything
#         this script prints; each cell writes OUT/<cell>.out and OUT/<cell>.err, and `summary`
#         writes OUT/summary.md.
#   BASE  a checkout of the base revision (the two-loop presets), built by the `build` step too.
#   CAP   wall cap in seconds per build and per cell; a capped cell's process group is terminated
#         and the cell counts as failed.
#   STEP  build | provenance | step | seg | summary
#     build       dune build of bench_gpt and bench_gpt_diag in both checkouts.
#     provenance  both revisions, clean-tree checks, the fixtures' digests against the m4-max
#                 content-v1 rows of fixtures/DIGESTS.txt, and the host state.
#     step        the step-time matrix: FIXTURES x TREATMENTS, REPEATS passes over the cell list,
#                 forward / reversed / forward, so an order effect splits the repeats of one cell
#                 rather than biasing a treatment.
#     seg         per-fission-segment attribution: bench_gpt_diag with BENCH_SEG_TIMES=1, one cell
#                 per inference fixture x treatment (min-of-20 per segment, a sync per run, so
#                 each segment carries a launch floor and the segments do not add up to a step).
#     summary     OUT/summary.md from the numbered passes' result lines: median p50 per cell, the
#                 p50 of each repeat, the widest p90/p10 of the cell's repeats, the ratio to the
#                 same fixture's base cell, and the losses' largest relative difference from base.
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
#   d1<T>     treatment <T> (one of the above but base) with the online-softmax forward and the
#             fused attention backward on -- treatment D1 of benchmarks/report-gh1002-fused-backward.md
#             (the others are its treatment A)
# Environment knobs (recorded in driver.log): BACKEND (default metal), REPEATS (default 3),
# FIXTURES (default the three inference and three training fixtures), TREATMENTS (default
# "base fill1 fill256"), FIXTURE_DIR (default benchmarks/fixtures of this checkout), REF (default
# base: the treatment the summary's ratio column divides by; a d1 treatment divides by d1REF),
# KERNEL_TABLE (default 0; 1 prints every shipped kernel's min-of-20 time to the cell's .err, the
# per-kernel attribution of gh-ocannl-1002).
#
# Exit: 0 all steps complete; 1 a cell or step failed; 2 usage; 130 interrupted.
# The environment is cleared of OCANNL_*, BENCH_* and the OpenMP controls: every treatment is
# pinned on the command line.
set -u
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
kernel_table=${KERNEL_TABLE:-0}
case $repeats in '' | *[!0-9]* | 0*) echo "gh1133: REPEATS must be a positive decimal integer, got '$repeats'" >&2; exit 2 ;; esac
[ -d "$base" ] || { echo "gh1133: BASE $base is not a directory" >&2; exit 2; }
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh1133: OUT $out is not empty; give each invocation a fresh directory" >&2
  exit 2
fi
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
base=$(cd "$base" && pwd -P) || exit 2
exec 3>&1
exec >>"$out/driver.log" 2>&1 || exit 2
trap 'cat "$out/driver.log" >&3' EXIT
trap 'echo "gh1133: interrupted"; exit 130' INT TERM
root=$(cd "$(dirname "$0")/.." && pwd -P)
fixture_dir=${FIXTURE_DIR:-$root/benchmarks/fixtures}
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2
echo "gh1133: $(date -u +%FT%TZ) root=$root base=$base out=$out cap=$cap backend=$backend repeats=$repeats"
echo "gh1133: fixtures: $fixtures; treatments: $treatments; ref: $ref; kernel table: $kernel_table; fixture dir: $fixture_dir; steps: $*"
built=0 proven=0 failed=0

runner_root() { case $1 in base) echo "$base" ;; *) echo "$root" ;; esac; }

# The attention form's flags come first, the forward key ahead of the backward one: the
# command-line reader takes the FIRST argument that begins with a key's spelling followed by a
# separator, and "_" is one -- so the backward key's argument also begins with the forward key's
# spelling (see conventions.md on key-name prefixes).
flags_of() {
  case $1 in
    base) echo "--ocannl_online_softmax=false" ;;
    fill*) echo "--ocannl_online_softmax=false --ocannl_gpu_schedule_workgroup_fill=${1#fill}" ;;
    keep) echo "--ocannl_online_softmax=false" ;;
    legacy) echo "--ocannl_online_softmax=false --ocannl_gpu_fission_keep_mapping=false" ;;
    d1base) echo "gh1133: the base runner takes no d1 form" >&2; return 1 ;;
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
  t1=$(date +%s)
  echo "=== cell $cell exit $st wall $((t1 - t0))s"
  [ "$st" -eq 0 ] || failed=1
  return "$st"
}

cell() {
  local fixture=$1 treatment=$2 tag=$3 exe=${4:-bench_gpt} flags r
  flags=$(flags_of "$treatment") || { failed=1; return 1; }
  r=$(runner_root "$treatment")
  local name="$backend-$fixture-$treatment-$tag"
  echo "backend=$backend fixture=$fixture treatment=$treatment runner=$r flags=[$flags]" >"$out/$name.err"
  # shellcheck disable=SC2086 # the treatment's flags are separate words by construction
  BENCH_FIXTURE="$fixture_dir/$fixture.safetensors" BENCH_TUNE=0 BENCH_MATERIALIZE=0 \
    BENCH_DEBUG=0 BENCH_SEG_TIMES="${SEG:-0}" BENCH_KERNEL_TABLE="$kernel_table" \
    capped "$name" "$r/_build/default/benchmarks/runners/ocannl/$exe.exe" \
    --ocannl_backend="$backend" --ocannl_default_prec=single \
    --ocannl_schedule_fission=true --ocannl_automatic_gpu_schedule=true \
    --ocannl_autotune_search=false --ocannl_debug_log_from_routines=false \
    --ocannl_output_debug_files_in_build_directory=false $flags
}

# The cell list in pass order: forward on odd passes, reversed on even ones.
pass_cells() {
  local pass=$1 list="" f t
  for f in $2; do for t in $treatments; do list="$list $f:$t"; done; done
  if [ $((pass % 2)) -eq 0 ]; then echo "$list" | tr ' ' '\n' | sed '/^$/d' | awk '{ l[NR] = $0 } END { for (i = NR; i > 0; i--) print l[i] }'
  else echo "$list" | tr ' ' '\n' | sed '/^$/d'; fi
}

need_identity() {
  [ "$built" = 1 ] && [ "$proven" = 1 ] && return 0
  echo "gh1133: step $1 needs build and provenance earlier in the same invocation"
  failed=1
  return 1
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
        capped "build-$(basename "$tree")" dune build --root "$tree" \
          benchmarks/runners/ocannl/bench_gpt.exe benchmarks/runners/ocannl/bench_gpt_diag.exe || ok=0
      done
      [ "$ok" = 1 ] && built=1 ;;
    provenance)
      ok=1
      clean_tree "$root" || ok=0
      clean_tree "$base" || ok=0
      n=0
      for f in $fixtures; do
        n=$((n + 1))
        [ -f "$fixture_dir/$f.safetensors" ] || { echo "gh1133: missing fixture $f"; ok=0; }
      done
      if [ "$ok" = 1 ] && python3 fixture_digest.py --check \
        $(for f in $fixtures; do echo "$fixture_dir/$f.safetensors"; done) |
        tee "$out/fixtures.txt" && [ "$(grep -c "MATCH — m4-max's bytes" "$out/fixtures.txt")" = "$n" ]; then
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
    summary)
      python3 - "$out" "$treatments" "$ref" >"$out/summary.md" <<'PY' || failed=1
import json, os, re, statistics, sys
out, treatments, ref_treatment = sys.argv[1], sys.argv[2].split(), sys.argv[3]
cells, missing = {}, []
for name in sorted(os.listdir(out)):
    m = re.fullmatch(r"(\w+)-(gpt2_mini\w*)-(base|(?:d1)?(?:fill\d+|keep|legacy))-(r\d+)\.out", name)
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
incomplete = bool(missing) or not cells
for name in missing:
    print("MISSING RESULT: %s has no result line" % name)
if not cells:
    print("NO MEASUREMENT: no numbered cell produced a result line")
print("| backend | fixture | treatment | p50 per repeat (ms) | median p50 | vs %s | p10..p90 spread | queued (median) | loss vs %s |" % (ref_treatment, ref_treatment))
print("|---|---|---|---|---|---|---|---|---|")
missing_refs = []
def med(key):
    reps = cells.get(key, [])
    return statistics.median(r["step_ms"]["p50"] for _, r in reps) if reps else None
for (backend, fixture, treatment), reps in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1], treatments.index(kv[0][2]) if kv[0][2] in treatments else 99)):
    p50s = [r["step_ms"]["p50"] for _, r in reps]
    m = statistics.median(p50s)
    spread = max(r["step_ms"]["p90"] / r["step_ms"]["p10"] for _, r in reps)
    queued = statistics.median(r.get("queued_step_ms") or 0.0 for _, r in reps)
    # A d1 treatment is compared with the d1 form of the reference (the same attention form).
    rt = ("d1" + ref_treatment) if treatment.startswith("d1") else ref_treatment
    ref = med((backend, fixture, rt))
    if ref is None:
        # The comparison the summary advertises did not happen: an incomplete matrix.
        missing_refs.append("%s %s %s (reference %s)" % (backend, fixture, treatment, rt))
    loss = ""
    bref = cells.get((backend, fixture, rt))
    if bref:
        a, b = reps[0][1].get("losses") or [], bref[0][1].get("losses") or []
        if a and b and len(a) == len(b):
            loss = "%.2g" % max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(a, b))
    print("| %s | %s | %s | %s | %.2f | %s | %.3fx | %.2f | %s |" % (
        backend, fixture, treatment, ", ".join("%.2f" % p for p in p50s), m,
        "%.3fx" % (m / ref) if ref else "", spread, queued, loss))
for row in missing_refs:
    print("MISSING REFERENCE: %s has no reference cell to compare with" % row)
sys.exit(1 if incomplete or missing_refs else 0)
PY
      cat "$out/summary.md" ;;
    *) echo "gh1133: unknown step $step"; failed=1 ;;
  esac
done
echo "gh1133: done $(date -u +%FT%TZ) failed=$failed"
exit "$failed"

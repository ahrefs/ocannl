#!/bin/bash
# gh-ocannl-1180 measurement driver: paired A/B of two register-tile column counts at the sites where
# Register_tile.default's tie rule decides between two tail-bearing tiles of equal price -- the
# "tail-free, then smaller rn" key against the "fewer tail vectors, then larger rn" key it replaced.
# Both geometries are requested of ONE prebuilt bin/narrow_gebp_bench with --rn=N (bin/bench_tile.ml:
# Register_tile.rm_cap rows at the widest width the machine renders at the compute precision), so
# the comparison is two geometries, not two builds of the model. The cells are bin_cells.py's
# (gh-ocannl-1565): per case, one run whose two arms are rnA and rnB, hermetic (no OCANNL_*/BENCH_*/
# OpenMP variable, no config file, the configuration trace read back), supervised, visited in
# Bench_design's order (AB BA ...) and paired round by round on the bench's bench-result lines.
#
# Usage:
#   benchmarks/gh1180_tie_ab.sh build   ROOT
#   benchmarks/gh1180_tie_ab.sh measure ROOT OUT ROUNDS CASE...
#   benchmarks/gh1180_tie_ab.sh summary OUT
#
#   ROOT    the checkout whose _build/default/bin/narrow_gebp_bench.exe runs; `build` builds it (a
#           measurement never builds, so it can run inside a timing-only window).
#   OUT     results directory, outside the checkout, created, refused unless empty;
#           OUT/driver.log records the run, OUT/<case>/ is that case's bin_cells run (report.md,
#           cells.jsonl, cells/<NNN>-rn<N>/stdout), `summary` writes OUT/summary.md (and `measure`
#           runs it at the end).
#   ROUNDS  a positive even count (Bench_design's rounds; for two arms, (A, B) then (B, A)).
#   CASE    PREC:N:BM:BK:REPEATS:RNA:RNB -- narrow_gebp_bench's positionals and the two column
#           counts. PREC is f32, f16 (narrow storage, f32 compute: --ocannl_fp16_arithmetic=false)
#           or f16a (pure f16 where the target has the arithmetic: --ocannl_fp16_arithmetic=true).
#           The bench's register-tile site is BM rows by N columns by BK, so RM = min 4 BM.
#
# The summary refuses an incomplete design: every case of driver.log's case list, every cell its
# bin_cells run planned, both arms of every round. A cell counts only when bin_cells found it clean
# (exit 0 -- a required cross-variant disagreement exits 1 -- and a clean treatment), its header
# names the requested geometry, and every packed line's census reads Mma_register_tiled with no
# scalar fallback; any other cell fails the run and the summary marks it. Serial (packmma) and
# pool-parallel (packmma_par) are kept apart; the ratio is rnB's time over rnA's, per round.
#
# Exit: 0 every cell valid; 1 a cell failed or failed a check (this outranks a refusal); 2 usage;
# 3 the only faults are cells refusing their measurement for host contention (bin_cells' exit 3).
set -u
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd -P)

summary() { # OUT
  python3 - "$1" "$HERE" <<'EOF'
import os, re, sys
out = sys.argv[1]
sys.path.insert(0, sys.argv[2])  # benchmarks/, for the shared runner and the table builders
import cells
from display_safe import display_value, log_line, table_lines
line_re = re.compile(r'^(packmma(?:_par)?)\s+\S+ ms\s+\S+ GFLOP/s.*\[(.*)\]\s*$')
bad, refused, rows = [], [], []
log = os.path.join(out, 'driver.log')
m = re.search(r'cases: (.*)', open(log).read()) if os.path.exists(log) else None
if not m:
    bad.append('driver.log names no cases: cannot tell a complete run from a partial one')
for c in (m.group(1).split() if m else []):
    prec, n, bm, bk, reps, rna, rnb = c.split(':')
    name = f'{prec}-{n}-{bm}-{bk}-{reps}-rn{rna}-rn{rnb}'
    run = os.path.join(out, name)
    try:
        manifest, records = cells.read_run(run)
    except (OSError, ValueError) as e:
        bad.append(f'{display_value(name)}: no complete bin_cells run ({display_value(str(e))})')
        continue
    planned = sum(len(order) for order in manifest['orders'])
    if len(records) != planned:
        bad.append(f'{display_value(name)}: {len(records)} of {planned} planned cells ran')
    # The run's own faults no cell carries (an arm resolving a setting two ways, an unidentified
    # kernel compiler), judged by the runner's own rule rather than restated here.
    bad += [f'{display_value(name)}: ' + display_value(p)
            for p in cells.run_level_problems(manifest, records)]
    by, case_refused = {}, False
    for r in records:
        cell = f'{display_value(name)} cell {r["cell"]} ({display_value(r["arm"])})'
        if not r['ok']:
            # A cell whose one fault is its own refusal (host contention) is the box's verdict,
            # kept apart from a cell that failed: the driver exits 3 on refusals alone.
            only_refused = r.get('refused') and len(r['problems']) == 1
            (refused if only_refused else bad).append(
                f'{cell}: ' + display_value('; '.join(r['problems'])))
            case_refused = case_refused or only_refused
            continue
        text = cells.cell_stream(run, r, 'stdout')
        rn = r['arm'][2:]
        header = next((ln for ln in text.splitlines() if ln.startswith('GEBP ')), '')
        if f' rn{rn} ' not in header or '(requested)' not in header:
            bad.append(f'{cell}: header does not name the requested rn{rn}')
        packed = {lm.group(1): lm.group(2) for lm in map(line_re.match, text.splitlines()) if lm}
        if set(packed) != {'packmma', 'packmma_par'}:
            bad.append(f'{cell}: packed lines missing ({sorted(packed)})')
        for variant, census in packed.items():
            if 'Mma_register_tiled' not in census or 'fallback' in census.lower():
                bad.append(f'{cell}: {variant} census [{display_value(census)}]')
        for variant, seconds, _, _ in r['results']:
            by.setdefault((variant, r['arm']), {})[r['round']] = seconds
    label = ' '.join(display_value(x) for x in (prec, f'n={n}', f'bm={bm}', f'bk={bk}'))
    for variant in ('packmma', 'packmma_par'):
        a, b = by.get((variant, f'rn{rna}'), {}), by.get((variant, f'rn{rnb}'), {})
        if (sorted(a) != sorted(b) or len(a) != manifest['rounds']) and not case_refused:
            bad.append(f'{display_value(name)}: {variant} lacks a side of the pair in some round')
        stats = cells.paired(a, b)
        rows.append((label, 'serial' if variant == 'packmma' else 'parallel', f'rn{rna}', f'rn{rnb}',
                     *cells.paired_cells(stats)))
text = '\n'.join(table_lines(['Case', 'Variant', 'A', 'B', *cells.PAIRED_HEADER], rows)) + '\n'
text += '\nratio = B time / A time per round; each case\'s full report is OUT/<case>/report.md\n'
if bad:
    text += '\nINVALID CELLS:\n' + '\n'.join(log_line('- ' + b) for b in bad) + '\n'
if refused:
    text += ('\nREFUSED CELLS (host contention; the box, not the treatment):\n'
             + '\n'.join(log_line('- ' + b) for b in refused) + '\n')
open(os.path.join(out, 'summary.md'), 'w').write(text)
print(text, end='')
sys.exit(1 if bad else 3 if refused else 0)
EOF
}

[ $# -ge 1 ] || usage
step=$1
shift
case $step in
  build)
    [ $# -eq 1 ] || usage
    cd "$1" || exit 2
    exec dune build bin/narrow_gebp_bench.exe ;;
  summary)
    [ $# -eq 1 ] || usage
    summary "$1"
    exit ;;
  measure) [ $# -ge 4 ] || usage ;;
  *) usage ;;
esac
root=$(cd "$1" && pwd -P) || exit 2
out=$2 rounds=$3
shift 3
case $rounds in '' | *[!0-9]*) echo "gh1180: ROUNDS must be a positive even integer, got '$rounds'" >&2; exit 2 ;; esac
if ((rounds == 0 || rounds % 2 != 0)); then echo "gh1180: ROUNDS must be positive and even, got $rounds" >&2; exit 2; fi
for c in "$@"; do
  [[ $c =~ ^(f32|f16|f16a):[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*:[1-9][0-9]*$ ]] ||
    { echo "gh1180: CASE must be PREC:N:BM:BK:REPEATS:RNA:RNB with PREC f32|f16|f16a, got '$c'" >&2; exit 2; }
  [ "${c##*:}" != "$(cut -d: -f6 <<<"$c")" ] ||
    { echo "gh1180: CASE '$c' times rn${c##*:} against itself; RNA and RNB must differ" >&2; exit 2; }
done
bench=$root/_build/default/bin/narrow_gebp_bench.exe
[ -x "$bench" ] || { echo "gh1180: $bench is not built; run the build step first" >&2; exit 2; }
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh1180: OUT $out is not empty; give each invocation a fresh directory" >&2
  exit 2
fi
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) || exit 2
exec 3>&1
exec >>"$out/driver.log" 2>&1 || exit 2
trap 'cat "$out/driver.log" >&3' EXIT
echo "gh1180: $(date -u +%FT%TZ) host=$(hostname) root=$root sha=$(git -C "$root" rev-parse HEAD 2>&1) rounds=$rounds"
echo "gh1180: cases: $*"
worst=0
for c in "$@"; do
  IFS=: read -r prec n bm bk reps rna rnb <<<"$c"
  case $prec in
    f32) common=(f32 "$n" "$reps" "$bm" "$bk") ;;
    f16) common=(f16 "$n" "$reps" "$bm" "$bk" --ocannl_fp16_arithmetic=false) ;;
    f16a) common=(f16 "$n" "$reps" "$bm" "$bk" --ocannl_fp16_arithmetic=true) ;;
  esac
  python3 "$HERE/bin_cells.py" --exe "$bench" --backend cc --out "$out/$prec-$n-$bm-$bk-$reps-rn$rna-rn$rnb" \
    --rounds "$rounds" --arm "rn$rna=--rn=$rna" --arm "rn$rnb=--rn=$rnb" -- "${common[@]}" >/dev/null
  st=$?
  echo "case=$c bin_cells exit=$st"
  # A failure (any nonzero status but 3) outranks a refusal (3), whichever case came last.
  if ((st != 0 && (worst == 0 || worst == 3))); then worst=$st; fi
done
summary "$out"
st=$?
# The summary judges every case: its validation failure (1) outranks any refusal, and its
# refusal-only verdict (3) stands unless a case's bin_cells run failed outright.
if ((st == 1 || (worst != 0 && worst != 3))); then exit 1; fi
if ((st == 3 || worst == 3)); then exit 3; fi
exit "$st"

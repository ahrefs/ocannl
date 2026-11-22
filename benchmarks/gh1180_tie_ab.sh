#!/bin/bash
# gh-ocannl-1180 measurement driver: paired A/B of two register-tile column counts at the sites where
# Register_tile.default's tie rule decides between two tail-bearing tiles of equal price -- the
# "tail-free, then smaller rn" key against the "fewer tail vectors, then larger rn" key it replaced.
# Both geometries are requested of ONE prebuilt bin/narrow_gebp_bench with --rn=N (bin/bench_tile.ml:
# Register_tile.rm_cap rows at the widest width the machine renders at the compute precision), so
# the comparison is two geometries, not two builds of the model. Each case uses bin_cells.py's
# hermetic, supervised design (gh-ocannl-1565) with rnA, rnB and a control that repeats rnA with
# the same binary and arguments. It pairs B/A and control/A within each round. Use a multiple of
# six rounds so all three arms visit every position equally.
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
#   ROUNDS  a positive even count (Bench_design's spread mirrored rotation). Two rounds work for
#           a smoke run. A geometry reading requires equal position counts, reached at multiples
#           of cells.exact_batches(3).
#   CASE    PREC:N:BM:BK:REPEATS:RNA:RNB -- narrow_gebp_bench's positionals and the two column
#           counts. PREC is f32, f16 (narrow storage, f32 compute: --ocannl_fp16_arithmetic=false)
#           or f16a (pure f16 where the target has the arithmetic: --ocannl_fp16_arithmetic=true).
#           The bench's register-tile site is BM rows by N columns by BK, so RM = min 4 BM.
#
# The summary reads this driver's case list and control marker from driver.log, the corresponding
# bin_cells manifests and records, and the GEBP header and timing censuses in their saved stdout.
# Older two-arm runs can be summarized without a drift reading. A cell counts only when bin_cells
# found it clean (exit 0 and a clean treatment), its header
# names the requested geometry, and every packed line's census reads Mma_register_tiled with no
# scalar fallback; any other cell fails the run and the summary marks it. Serial (packmma) and
# pool-parallel (packmma_par) are kept apart. The drift check compares the largest absolute
# deviation of control/A from 1 against the absolute median B/A effect, allowing for roundoff.
# Geometry readings also require the paired ratios to point in one direction. Flag every arm with
# cells below 100 us per routine run. The grid census distinguishes short pool timings and serial
# rendering under the parallel label. Both leave a parallel comparison inconclusive.
# Each variant is read for modes as bin_cells.py's own report reads it (bin_cells.mode_rows,
# cells.MODE_RULE; gh-ocannl-2308): a variant with a bimodal arm keeps its pooled row, marked
# "pooled over modes", and the Modes section gives its arms' groups and within-mode ratios.
# With the control present, the rule also considers the control/A effect when setting A's mode
# threshold. This can make the three-arm rule more sensitive to modes than its two-arm counterpart.
# The control measures drift visible across processes. It cannot identify their core placement.
#
# Exit: 0 every cell valid; 1 a cell failed or failed a check (this outranks a refusal); 2 usage;
# 3 the only faults are cells refusing their measurement for host contention (bin_cells' exit 3).
set -u
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd -P)

summary() { # OUT
  python3 - "$1" "$HERE" <<'EOF'
import math, os, re, sys
out = sys.argv[1]
sys.path.insert(0, sys.argv[2])  # benchmarks/, for the shared runner and the table builders
import bench_output, bin_cells, cells
from display_safe import display_value, log_line, table_lines

CONTROL = 'control'
CONTROL_MARKER = 'gh1180: control: duplicate A'
SHORT_SECONDS = 100e-6
kinds = {'packmma': 'serial', 'packmma_par': 'parallel'}
bad, refused, rows = [], [], []
mode_split, mode_paired = [], []  # the cases' within-mode rows, each led by its case and pair
log = os.path.join(out, 'driver.log')
driver_log = open(log).read() if os.path.exists(log) else ''
controlled = CONTROL_MARKER in driver_log.splitlines()
m = re.search(r'^gh1180: cases: (.+)$', driver_log, re.MULTILINE)
if not m:
    bad.append('driver.log names no cases: cannot tell a complete run from a partial one')
for c in (m.group(1).split() if m else []):
    prec, n, bm, bk, reps, rna, rnb = c.split(':')
    name = f'{prec}-{n}-{bm}-{bk}-{reps}-rn{rna}-rn{rnb}'
    run = os.path.join(out, name)
    a, b = f'rn{rna}', f'rn{rnb}'
    arms = [a, b, CONTROL] if controlled else [a, b]
    case_bad, case_refusal_count = len(bad), len(refused)
    try:
        manifest, records = cells.read_run(run)
    except (OSError, ValueError) as e:
        bad.append(f'{display_value(name)}: no complete bin_cells run ({display_value(str(e))})')
        continue
    orders = cells.design_orders(len(arms), manifest['rounds'])
    planned = [(rnd, pos, arms[i]) for rnd, order in enumerate(orders) for pos, i in enumerate(order)]
    seen = [(r['round'], r['position'], r['arm']) for r in records]
    if ([arm['label'] for arm in manifest['arms']] != arms
            or manifest['orders'] != orders or seen != planned):
        bad.append(f'{display_value(name)}: incomplete or out-of-order design for {", ".join(arms)}')
    # The run's own faults no cell carries (an arm resolving a setting two ways, an unidentified
    # kernel compiler), judged by the runner's own rule rather than restated here.
    bad += [f'{display_value(name)}: ' + display_value(p)
            for p in cells.run_level_problems(manifest, records)]
    by, clean, pool = {}, {}, {}
    for r in records:
        cell = f'{display_value(name)} cell {r["cell"]} ({display_value(r["arm"])})'
        if not r['ok']:
            # A cell whose one fault is its own refusal (host contention) is the box's verdict,
            # kept apart from a cell that failed: the driver exits 3 on refusals alone.
            only_refused = r.get('refused') and len(r['problems']) == 1
            (refused if only_refused else bad).append(
                f'{cell}: ' + display_value('; '.join(r['problems'])))
            continue
        clean[r['round'], r['arm']] = r
        try:
            text = cells.cell_stream(run, r, 'stdout')
        except OSError as e:
            bad.append(f'{cell}: {display_value(str(e))}')
            continue
        rn = rna if r['arm'] == CONTROL else r['arm'][2:]
        header = bench_output.gebp_header(text) or ''
        if f' rn{rn} ' not in header or '(requested)' not in header:
            bad.append(f'{cell}: header does not name the requested rn{rn}')
        # The mma census, read by bench_output's one reader of the timing line: until gh-ocannl-1767
        # a last-bracket match here read the grid census in its place (gh-ocannl-1668 appended it),
        # and every real cell failed the census check below.
        try:
            packed = {t['variant']: t for t in bench_output.timing_lines(text)
                      if t['variant'] in kinds}
            if 'packmma_par' in packed:
                pool[r['round'], r['arm']] = bench_output.grid_uses_pool(packed['packmma_par']['grid'])
        except ValueError as e:
            bad.append(f'{cell}: ' + display_value(str(e)))
            continue
        if set(packed) != kinds.keys():
            bad.append(f'{cell}: packed lines missing ({sorted(packed)})')
        for variant, timing in packed.items():
            census = timing['mma']
            if 'Mma_register_tiled' not in census or 'fallback' in census.lower():
                bad.append(f'{cell}: {variant} census [{display_value(census)}]')
        results = {v: seconds for v, seconds, _, _ in r['results'] if v in kinds}
        if results.keys() != kinds.keys():
            bad.append(f'{cell}: packed results missing ({sorted(results)})')
        for variant, seconds in results.items():
            by.setdefault((kinds[variant], r['arm']), {})[r['round']] = seconds
    if controlled:
        for rnd in range(manifest['rounds']):
            source, control = (clean.get((rnd, arm)) for arm in (a, CONTROL))
            if source and control and any(source[k] != control[k] for k in ('argv', 'env', 'config')):
                bad.append(f'{display_value(name)} round {rnd + 1}: control did not repeat {a}')
    label = ' '.join(display_value(x) for x in (prec, f'n={n}', f'bm={bm}', f'bk={bk}'))
    modes = bin_cells.mode_rows(arms, kinds.values(), by)
    pair = (label, a, b)
    for split, paired in modes.values():
        mode_split += [(*pair, *r) for r in split]
        mode_paired += [(*pair, *r) for r in paired]
    case_usable = case_bad == len(bad) and case_refusal_count == len(refused)
    for kind in kinds.values():
        times = {arm: by.get((kind, arm), {}) for arm in arms}
        stats = cells.paired(times[a], times[b])
        aa = cells.paired(times[a], times.get(CONTROL, {}))
        short = []
        for arm, samples in times.items():
            nshort = sum(t < SHORT_SECONDS for t in samples.values())
            if nshort:
                short.append(f'{arm} {nshort}/{len(samples)}')
        reasons = []
        if not case_usable:
            reasons.append('invalid or refused cells')
        if kind in modes:
            reasons.append('modes')
        if kind == 'parallel':
            if any(not used for used in pool.values()):
                reasons.append('serial rendering in parallel cells')
            if any(pool.get((rnd, arm)) and t < SHORT_SECONDS
                   for arm, samples in times.items() for rnd, t in samples.items()):
                reasons.append('short pool timings')
        spread_cell = '-'
        drift = '-'
        if not controlled:
            reasons.append('no control')
        elif not cells.design_balance(len(arms), orders)[1]:
            reasons.append('unequal position counts')
        if aa['median'] is not None:
            spread = max(abs(aa['lo'] - 1), abs(aa['hi'] - 1))
            spread_cell = f'{spread:.2%}'
            if stats['median'] is not None and case_usable:
                effect = abs(stats['median'] - 1)
                # Division and the signed subtraction around 1 can round equal deviations
                # differently, e.g. 1 - 0.9 versus 1.1 - 1.
                if spread >= effect or math.isclose(spread, effect, rel_tol=0, abs_tol=math.ulp(1.0)):
                    drift = 'spread meets effect'
                    reasons.append('control spread')
                else:
                    drift = 'clears effect'
        if stats['lo'] is not None and stats['lo'] <= 1 <= stats['hi']:
            reasons.append('paired range includes 1')
        if reasons:
            reading = f'inconclusive ({", ".join(reasons)})'
        else:
            reading = 'A faster' if stats['median'] > 1 else 'B faster'
        rows.append((label, f'{kind} (pooled over modes)' if kind in modes else kind, a, b,
                     *cells.paired_cells(stats), cells.paired_cells(aa)[2], spread_cell, drift,
                     ', '.join(short) or '-', reading))
text = '\n'.join(table_lines(['Case', 'Variant', 'A', 'B', *cells.PAIRED_HEADER,
                              'Control/A range', 'Control spread', 'Drift check', 'Short (<100 µs)',
                              'Reading'],
                             rows)) + '\n'
text += '\nratio = B time / A time per round; each case\'s full report is OUT/<case>/report.md\n'
text += ('Control repeats A. Spread is the largest deviation of control/A from 1. '
         'A geometry reading requires spread below the absolute median B/A effect, '
         'balanced positions, no detected modes and a paired B/A range that excludes 1. '
         'Short counts use seconds per routine run. Short pool timings can be dominated by '
         'dispatch.\n')
if mode_split:
    text += ('\nA variant "pooled over modes" has a bimodal arm: its row above mixes the modes, '
             'which the rows below read apart.\n\n## Modes\n\n' + cells.MODE_RULE + '\n\n')
    text += '\n'.join(table_lines(['Case', 'A', 'B', *bin_cells.MODE_SPLIT_HEADER], mode_split)) + '\n'
    if mode_paired:
        text += (f'\nEach arm paired against A within modes {bin_cells.MODE_PAIRED_NOTE}:\n\n'
                 + '\n'.join(table_lines(['Case', 'A', 'B', *bin_cells.MODE_PAIRED_LEAD,
                                           *cells.MODE_PAIRED_HEADER], mode_paired)) + '\n')
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
echo "gh1180: control: duplicate A"
worst=0
for c in "$@"; do
  IFS=: read -r prec n bm bk reps rna rnb <<<"$c"
  case $prec in
    f32) common=(f32 "$n" "$reps" "$bm" "$bk") ;;
    f16) common=(f16 "$n" "$reps" "$bm" "$bk" --ocannl_fp16_arithmetic=false) ;;
    f16a) common=(f16 "$n" "$reps" "$bm" "$bk" --ocannl_fp16_arithmetic=true) ;;
  esac
  python3 "$HERE/bin_cells.py" --exe "$bench" --backend cc --out "$out/$prec-$n-$bm-$bk-$reps-rn$rna-rn$rnb" \
    --rounds "$rounds" --arm "rn$rna=--rn=$rna" --arm "rn$rnb=--rn=$rnb" \
    --arm "control=--rn=$rna" -- "${common[@]}" >/dev/null
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

#!/bin/bash
# gh-ocannl-1180 measurement driver: paired A/B of two register-tile column counts at the sites where
# Register_tile.default's tie rule decides between two tail-bearing tiles of equal price -- the
# "tail-free, then smaller rn" key against the "fewer tail vectors, then larger rn" key it replaced.
# Both geometries are requested of ONE prebuilt bin/narrow_gebp_bench with --rn=N (bin/bench_tile.ml:
# Register_tile.rm_cap rows at the widest width the machine renders at the compute precision), so
# the comparison is two geometries, not two builds of the model. The protocol is gh-ocannl-1099's
# (docs/research/gh-1099-register-tile-c-traffic.md): interleaved rounds, the pair's order reversed
# on every other round.
#
# Usage:
#   benchmarks/gh1180_tie_ab.sh build   ROOT
#   benchmarks/gh1180_tie_ab.sh measure ROOT OUT ROUNDS CASE...
#   benchmarks/gh1180_tie_ab.sh summary OUT
#
#   ROOT    the checkout whose _build/default/bin/narrow_gebp_bench.exe runs; `build` builds it (a
#           measurement never builds, so it can run inside a timing-only window).
#   OUT     results directory, created, refused unless empty; OUT/driver.log records the run, each
#           invocation writes OUT/<case>_r<round>_rn<N>.out and .err, `summary` writes
#           OUT/summary.md (and `measure` runs it at the end).
#   ROUNDS  a positive even count: round r runs (A, B) for even r and (B, A) for odd r.
#   CASE    PREC:N:BM:BK:REPEATS:RNA:RNB -- narrow_gebp_bench's positionals and the two column
#           counts. PREC is f32, f16 (narrow storage, f32 compute: --ocannl_fp16_arithmetic=false)
#           or f16a (pure f16 where the target has the arithmetic: --ocannl_fp16_arithmetic=true).
#           The bench's register-tile site is BM rows by N columns by BK, so RM = min 4 BM.
#
# A cell counts only when the bench exited 0 (a required cross-variant disagreement exits 1),
# printed the requested geometry on its header, and every packed line's census reads
# Mma_register_tiled with no scalar fallback; any other cell fails the run and the summary marks it.
# The summary's percentage is the median of the PAIRED throughput ratios A/B - 1 per round; a
# positive value favors A. Serial (packmma) and pool-parallel (packmma_par) are kept apart.
#
# Exit: 0 every cell valid; 1 a cell failed; 2 usage.
# Hermetic: every OCANNL_* variable is unset, the backend and every flag are on argv, and the bench
# runs from benchmarks/, whose ocannl_config is the nearest on the upward search.
set -u
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
usage() { sed -n '/^# Usage/,/^# Exit/p' "$0" | sed '$d' >&2; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd -P)

summary() { # OUT
  python3 - "$1" <<'EOF'
import glob, os, re, statistics, sys
out = sys.argv[1]
line_re = re.compile(r'^(packmma(?:_par)?)\s+\S+ ms\s+(\S+) GFLOP/s.*\[(.*)\]\s*$')
cells, bad = {}, []
for path in sorted(glob.glob(os.path.join(out, '*.out'))):
    m = re.match(r'(.+)_r(\d+)_rn(\d+)\.out$', os.path.basename(path))
    if not m:
        continue
    case, rnd, rn = m.group(1), int(m.group(2)), int(m.group(3))
    status = open(path[:-4] + '.status').read().strip() if os.path.exists(path[:-4] + '.status') else '?'
    text = open(path).read()
    rates = {}
    for ln in text.splitlines():
        lm = line_re.match(ln)
        if lm:
            census = lm.group(3)
            if 'Mma_register_tiled' not in census or 'fallback' in census.lower():
                bad.append(f'{os.path.basename(path)}: census [{census}]')
            rates[lm.group(1)] = float(lm.group(2))
    if status != '0':
        bad.append(f'{os.path.basename(path)}: exit {status}')
    header = next((ln for ln in text.splitlines() if ln.startswith('GEBP ')), '')
    if f' rn{rn} ' not in header or '(requested)' not in header:
        bad.append(f'{os.path.basename(path)}: header does not name the requested rn{rn}')
    if set(rates) != {'packmma', 'packmma_par'}:
        bad.append(f'{os.path.basename(path)}: packed lines missing ({sorted(rates)})')
    cells.setdefault(case, {}).setdefault(rnd, {})[rn] = rates
rows = ['| Case | Variant | A | B | A range (GFLOP/s) | B range (GFLOP/s) | Median A vs B | A wins |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |']
for case, rounds in cells.items():
    prec, n, bm, bk, reps, rna, rnb = case.split('-')
    rna, rnb = int(rna[2:]), int(rnb[2:])
    for variant in ('packmma', 'packmma_par'):
        a = [rounds[r].get(rna, {}).get(variant) for r in sorted(rounds)]
        b = [rounds[r].get(rnb, {}).get(variant) for r in sorted(rounds)]
        pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
        if not pairs:
            continue
        ratios = [x / y - 1 for x, y in pairs]
        wins = sum(1 for x, y in pairs if x > y)
        ties = sum(1 for x, y in pairs if x == y)
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        label = f'{prec} n={n} bm={bm} bk={bk}'
        rows.append(f'| {label} | {"serial" if variant == "packmma" else "parallel"} | rn{rna} | rn{rnb} '
                    f'| {min(xs):.2f}–{max(xs):.2f} | {min(ys):.2f}–{max(ys):.2f} '
                    f'| {statistics.median(ratios) * 100:+.2f}% | {wins}/{len(pairs)}'
                    f'{f", {ties} tie" if ties else ""} |')
text = '\n'.join(rows) + '\n'
if bad:
    text += '\nINVALID CELLS:\n' + '\n'.join('- ' + b for b in bad) + '\n'
open(os.path.join(out, 'summary.md'), 'w').write(text)
print(text, end='')
sys.exit(1 if bad else 0)
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
echo "gh1180: $(date -u +%FT%TZ) host=$(hostname) root=$root sha=$(git -C "$root" rev-parse HEAD) rounds=$rounds"
echo "gh1180: dirty files: $(git -C "$root" status --porcelain --untracked-files=no | wc -l | tr -d ' ')"
echo "gh1180: bench mtime $(ls -l "$bench" | awk '{print $6, $7, $8}')"
echo "gh1180: cases: $*"
cd "$root/benchmarks" || exit 2
failed=0
for ((r = 0; r < rounds; r++)); do
  for c in "$@"; do
    IFS=: read -r prec n bm bk reps rna rnb <<<"$c"
    case $prec in
      f32) flags=(f32) ;;
      f16) flags=(f16 --ocannl_fp16_arithmetic=false) ;;
      f16a) flags=(f16 --ocannl_fp16_arithmetic=true) ;;
    esac
    if ((r % 2 == 0)); then order=("$rna" "$rnb"); else order=("$rnb" "$rna"); fi
    for rn in "${order[@]}"; do
      cell="$out/${prec}-${n}-${bm}-${bk}-${reps}-rn${rna}-rn${rnb}_r${r}_rn${rn}"
      "$bench" "${flags[0]}" "$n" "$reps" "$bm" "$bk" "--rn=$rn" "${flags[@]:1}" \
        --ocannl_backend=cc --ocannl_log_config_sourcing=true >"$cell.out" 2>"$cell.err"
      st=$?
      echo "$st" >"$cell.status"
      ((st == 0)) || failed=1
      echo "round=$r case=$c rn=$rn exit=$st"
    done
  done
done
summary "$out" || failed=1
exit $failed

#!/bin/bash
# gh-ocannl-740 before/after cells: what the pad-waste economy saves a search. One driver, one box,
# every leg the closing comment quotes, so the comment names this invocation.
#
# Usage: gh740_cells.sh <out-dir> <backend> <arm-order> <site>...
#   <arm-order>  a word over {A,B}: A lifts the economy (autotune_pad_waste_max=inf, the menus as
#                they were before it), B runs at the default threshold; e.g. ABBA, so each arm is
#                timed early and late and drift shows as disagreement between its two legs.
#   <site>       name=b,m,n,k as bench_pad_waste takes it.
# Each leg is one bench_pad_waste process over every site (writes <backend>-<n>-<arm>.log under
# <out-dir>, with the autotune_log per-candidate lines on its stderr), then an `exit: N` line;
# the `padwaste:` lines are echoed. Every search starts from an empty schedule cache.
#
# Hermetic: OCANNL_* and BENCH_* are unset, the treatment is on argv, the runner starts from
# benchmarks/ (the nearest ocannl_config). No fixture is read. Exits nonzero when any leg did.
set -u
[ $# -ge 4 ] || { sed -n '5,9p' "$0"; exit 2; }
OUT=$1 BACKEND=$2 ORDER=$3; shift 3
case $ORDER in *[!AB]*|'') echo "arm order must be a word over A,B: $ORDER" >&2; exit 2;; esac
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$OUT" || exit 1
OUT=$(CDPATH= cd -- "$OUT" && pwd)
cd "$ROOT" || exit 1
dune build benchmarks/runners/ocannl/bench_pad_waste.exe 2>/dev/null || { echo "BUILD FAILED"; exit 1; }
cd benchmarks || exit 1
EXE=../_build/default/benchmarks/runners/ocannl/bench_pad_waste.exe
rc=0 n=0
for arm in $(echo "$ORDER" | sed 's/./& /g'); do
  n=$((n + 1))
  case $arm in A) thr=inf ;; B) thr=default ;; esac
  log="$OUT/$BACKEND-$n-$arm.log"
  if [ "$thr" = default ]; then
    "$EXE" "$@" --ocannl_backend="$BACKEND" --ocannl_autotune_log=true >"$log" 2>&1
  else
    "$EXE" "$@" --ocannl_backend="$BACKEND" --ocannl_autotune_pad_waste_max="$thr" \
      --ocannl_autotune_log=true >"$log" 2>&1
  fi
  r=$?
  echo "leg $n arm $arm ($thr): exit: $r"
  grep '^padwaste:' "$log"
  [ "$r" -ne 0 ] && rc=$r
done
exit $rc

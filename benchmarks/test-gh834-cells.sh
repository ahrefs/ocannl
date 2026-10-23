#!/usr/bin/env bash
# ocannl-harness: standalone
# Controls for benchmarks/gh834_cells.sh's session-completeness check: the driver's own
# require_complete_session, run against fabricated result lines, reads the arms structurally
# (benchmarks/gh834_session.py). An arm whose nested objects precede its best_ms, or whose fields
# come in another order, is still complete; an untimed, unsearched, failed, contended or
# unbatched arm, a missing arm and an unparsable line are not.
# Usage: benchmarks/test-gh834-cells.sh [--keep|--help]
#
# The function is cut out of the shipping driver and run with OUT and the checker in a scratch
# tree, so no dune, runner or fixture is reached. Each oracle has a mutated twin it must reject,
# the driver's former brace-bounded scan among them.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-gh834-cells
SRC=$here/gh834_cells.sh
CHECKER=$here/gh834_session.py
tree=$TMP/tree
mkdir -p "$tree/benchmarks" "$TMP/runs"
# The checker's strict result-line loader, beside it as in the shipping tree.
cp "$here/strict_json.py" "$tree/benchmarks/" || exit 2

# The trace summary of a session whose timing calls all completed.
GOOD_TRACE='timing-trace: summary: 12.5s wall, 9 candidate attempts, 31 timing calls'

# [arm NAME [KEY=JSON]...]: one arm object. The nested timing matrix and a nested terminal-failure
# slot come BEFORE best_ms, the order a brace-bounded scan cannot read; KEY=JSON overrides a field
# (KEY=- drops it).
arm() {
  local name=$1
  shift
  python3 -I - "$name" "$@" <<'PY'
import json, sys
name, *overrides = sys.argv[1:]
a = {"arm": name, "state": "searched", "searched": True, "cache_hit": False, "timing": "queued",
     "timing_decisions": {"measured_batch": {"admitted": 12, "refused_contended": 0}},
     "terminal_failure": None, "timings_contended": 0, "timings_unbatched": 0, "timings_unreadable": 0,
     "best_ms": 1.25, "best_label": "grid 64", "mma_best_ms": None}
for o in overrides:
    k, v = o.split("=", 1)
    if v == "-":
        a.pop(k, None)
    else:
        a[k] = json.loads(v)
print(json.dumps(a, separators=(",", ":")))
PY
}
# [line ARM_JSON...]: a result line around the given arms.
line() {
  local IFS=,
  printf '{"framework":"ocannl","backend":"cc","compile_s":3.5,"searched":true,"tune":{"shipped":"A","searches":2,"replays":0,"no_searches":0,"shipped_mma":null,"arms":[%s]},"step_ms":{"p10":1,"p50":1,"p90":1},"losses":[]}\n' "$*"
}

# [check DRIVER CHECKER RUN RESULT_LINE]: run DRIVER's require_complete_session over a session
# step "s" whose .out ends with RESULT_LINE, with CHECKER as the tree's gh834_session.py; its
# printout and the resulting status go to runs/RUN/stdout.
check() {
  local driver=$1 checker=$2 run=$3 result=$4 d=$TMP/runs/$3
  mkdir -p "$d/out"
  cp "$checker" "$tree/benchmarks/gh834_session.py" || return 2
  awk '/^require_complete_session\(\) \{/ { f = 1 } f { print } f && /^}$/ { exit }' "$driver" \
    >"$d/fn.sh" || return 2
  grep -q '^require_complete_session() {' "$d/fn.sh" || { echo "no require_complete_session in $driver" >&2; return 2; }
  printf 'compile done\n%s\n' "$result" >"$d/out/s.out"
  printf '%s\n' "$GOOD_TRACE" >"$d/out/s.err"
  root=$tree out=$d/out bash -c 'set -u; status=0; . "$1"; require_complete_session s; echo "status=$status"' \
    _ "$d/fn.sh" >"$d/stdout" 2>&1
}
# [accepts RUN LINE]: status 0 and no step diagnostic.
accepts() {
  check "$SRC" "$CHECKER" "$1" "$2"
  grep -qx 'status=0' "$TMP/runs/$1/stdout" && ! grep -q '^== step' "$TMP/runs/$1/stdout"
}
# [rejects RUN LINE PATTERN...]: status 1 and every PATTERN among the step diagnostics.
rejects() {
  local run=$1 result=$2 p
  shift 2
  check "$SRC" "$CHECKER" "$run" "$result"
  grep -qx 'status=1' "$TMP/runs/$run/stdout" || return 1
  for p in "$@"; do grep -q -- "^== step s: $p" "$TMP/runs/$run/stdout" || return 1; done
}
leg() { # LABEL COMMAND...
  local label=$1
  shift
  if "$@"; then report 0 "$label"; else report 1 "$label" "see $TMP/runs (--keep)"; fi
}

A=$(arm A) B=$(arm B)
complete=$(line "$A" "$B")
case $complete in *'"timing_decisions":{"measured_batch":{'*'"best_ms":1.25'*) ;;
  *) echo "fixture does not put a nested object before best_ms: $complete" >&2; exit 2 ;; esac
leg 'a complete session with nested objects before best_ms is complete' accepts complete "$complete"
# The same arms with best_ms first and the name last: order is the runner's to choose.
reordered=$(printf '%s' "$complete" | python3 -I -c '
import json, sys
r = json.load(sys.stdin)
r["tune"]["arms"] = [dict(reversed(list(a.items()))) for a in r["tune"]["arms"]]
print(json.dumps(r, separators=(",", ":")))')
leg 'field order within an arm does not matter' accepts reordered "$reordered"
leg 'an arm that timed nothing is incomplete' \
  rejects untimed "$(line "$A" "$(arm B best_ms=null)")" 'INCOMPLETE SESSION: arm B timed no candidate'
leg 'a non-JSON NaN timing is not a timing' \
  rejects nan "$(line "$A" "$(arm B best_ms=NaN)")" 'INCOMPLETE SESSION: the result line is not JSON'
overflow=$(line "$A" "$(arm B best_ms=7.5)")
leg 'an overflowing timing is not a timing' \
  rejects overflow "${overflow/\"best_ms\":7.5/\"best_ms\":1e999}" 'INCOMPLETE SESSION: the result line is not JSON'
leg 'an arm that died is incomplete and its failure is named' \
  rejects died "$(line "$(arm A state='"search-died"' \
    terminal_failure='{"phase":"timing","candidate":"c","detail":"d"}')" "$B")" \
  'INCOMPLETE SESSION: arm A did not complete a search' \
  'INCOMPLETE SESSION: an arm carries a terminal failure'
leg 'a missing arm is incomplete' \
  rejects missing "$(line "$A")" 'INCOMPLETE SESSION: arm B did not complete a search' \
  'INCOMPLETE SESSION: arm B timed no candidate'
leg 'a contended arm is contended and unbatched' \
  rejects contended "$(line "$A" "$(arm B timings_contended=2 timings_unbatched=1)")" \
  'CONTENDED SESSION' 'UNBATCHED SESSION'
only_unbatched() {
  rejects unbatched "$(line "$A" "$(arm B timings_contended=1 timings_unbatched=1)")" \
    'UNBATCHED SESSION' && ! grep -q 'CONTENDED SESSION' "$TMP/runs/unbatched/stdout"
}
leg 'refusals that are all unbatched are not counted as contention' only_unbatched
only_unreadable() {
  rejects unreadable "$(line "$A" "$(arm B timings_contended=1 timings_unreadable=1)")" \
    'UNREADABLE SESSION' && ! grep -q 'CONTENDED SESSION' "$TMP/runs/unreadable/stdout"
}
leg 'refusals that are all degenerate clock readings are not counted as contention' only_unreadable
leg 'a dropped refusal count is not a pass' \
  rejects dropped "$(line "$A" "$(arm B timings_unbatched=-)")" \
  'INCOMPLETE SESSION: arm B carries no timings_unbatched'
leg 'a dropped degenerate-reading count is not a pass' \
  rejects dropped_unreadable "$(line "$A" "$(arm B timings_unreadable=-)")" \
  'INCOMPLETE SESSION: arm B carries no timings_unreadable'
leg 'a truncated result line is incomplete' \
  rejects truncated "${complete%???}" 'INCOMPLETE SESSION: the result line is not JSON'

# The runner's own wire format, rendered from bench_json.ml's templates rather than from a second
# copy of its field names: a field the checker reads that the runner renames or drops fails here.
runner=$(python3 -I - "$here/runners/ocannl/bench_json.ml" <<'PY'
import json, re, sys
src = open(sys.argv[1]).read()
arm = re.search(r'\{\|(\{"arm":"%s".*?)\|\}', src, re.S).group(1)
tune = re.search(r'\{\|(\{"shipped":.*?"arms":\[%s\]\})\|\}', src, re.S).group(1)
def render(t, **fixed):
    for k, v in fixed.items():
        t = t.replace(k, v, 1)
    return t.replace('"%s"', '"x"').replace("%s", "null").replace("%d", "0").replace("%b", "true")
arms = [render(arm, **{'"arm":"%s"': '"arm":"%s"' % n, '"state":"%s"': '"state":"searched"',
                       '"best_ms":%s': '"best_ms":1.5'}) for n in "AB"]
t = render(tune.replace("[%s]", "[" + ",".join(arms) + "]"))
print(json.dumps({"compile_s": 1.0, "searched": True, "tune": json.loads(t)}, separators=(",", ":")))
PY
) || { echo "could not render bench_json.ml's arm and tune templates" >&2; exit 2; }
leg "the runner's own arm format reads as complete" accepts runner "$runner"

# Negative controls: a twin of the driver or of the checker that must lose a leg above.
# [driver_accepts SUBJECT RUN]: SUBJECT's driver accepts the complete session.
driver_accepts() {
  check "$1" "$CHECKER" "$2" "$complete"
  grep -qx 'status=0' "$TMP/runs/$2/stdout" && ! grep -q '^== step' "$TMP/runs/$2/stdout"
}
driver_rejects_untimed() {
  check "$1" "$CHECKER" "$2" "$(line "$A" "$(arm B best_ms=null)")"
  grep -qx 'status=1' "$TMP/runs/$2/stdout"
}
checker_rejects_untimed() {
  check "$SRC" "$1" "$2" "$(line "$A" "$(arm B best_ms=null)")"
  grep -qx 'status=1' "$TMP/runs/$2/stdout"
}
checker_splits_contention() {
  check "$SRC" "$1" "$2" "$(line "$A" "$(arm B timings_contended=1 timings_unbatched=1)")"
  ! grep -q 'CONTENDED SESSION' "$TMP/runs/$2/stdout"
}
checker_splits_unreadable() {
  check "$SRC" "$1" "$2" "$(line "$A" "$(arm B timings_contended=1 timings_unreadable=1)")"
  ! grep -q 'CONTENDED SESSION' "$TMP/runs/$2/stdout"
}
# The driver as it was: best_ms read by a scan from the arm's name to its first closing brace.
cat >"$TMP/brace-scan.txt" <<'EOF'
  local line arm
  line=$(grep '^{' "$out/$1.out" | tail -1)
  for arm in A B; do
    case $(printf '%s' "$line" | sed -n "s/.*\"arm\":\"$arm\"[^}]*\"best_ms\":\([^,}]*\).*/\1/p") in
    '' | null) echo "== step $1: INCOMPLETE SESSION: arm $arm timed no candidate"; status=1 ;;
    esac
  done
EOF
scan=$(mutant brace-scan '
  /^  local verdicts rc=0$/ { while ((getline l < repl) > 0) print l; skip = 1; next }
  skip && /^  fi$/ { skip = 0; next }
  skip { next }
  { print }' -v repl="$TMP/brace-scan.txt") || exit 2
cmp -s "$SRC" "$scan" && { echo "mutant brace-scan changed nothing" >&2; exit 2; }
grep -q 'python3' <(sed -n '/^require_complete_session/,/^}/p' "$scan") &&
  { echo "mutant brace-scan still calls the checker" >&2; exit 2; }
expect_rejected 'a brace-bounded best_ms scan' "$scan" driver_accepts 'timed no candidate'
# A driver that prints the checker's verdicts but does not fail on them.
lenient=$(mutant lenient '/^    status=1$/ && !done { done = 1; next } { print }') || exit 2
cmp -s "$SRC" "$lenient" && { echo "mutant lenient changed nothing" >&2; exit 2; }
expect_rejected 'verdicts that do not fail the session' "$lenient" driver_rejects_untimed \
  'timed no candidate'
# [py_mutant NAME AWK_PROGRAM]: a checked twin of the checker (mutant's Python counterpart).
py_mutant() {
  local out="$TMP/$1.py"
  awk "$2" "$CHECKER" >"$out" || return 1
  python3 -I -m py_compile "$out" || return 1
  cmp -s "$CHECKER" "$out" && { echo "mutant $1 changed nothing" >&2; return 1; }
  printf '%s' "$out"
}
# A checker that never asks whether an arm timed anything.
untimed=$(py_mutant untimed '/timed no candidate/ { print "            pass"; next } { print }') || exit 2
expect_rejected 'an untimed arm reads as complete' "$untimed" checker_rejects_untimed 'status=0'
# A checker that counts every refusal as contention.
merged=$(py_mutant merged '{ sub(/\["timings_contended"\] > a\["timings_unbatched"\]/, "[\"timings_contended\"] > 0"); print }') || exit 2
expect_rejected 'unbatched refusals counted as contention' "$merged" checker_splits_contention \
  'CONTENDED SESSION'
# A checker that counts degenerate clock readings as contention (gh-ocannl-1517).
unreadable_merged=$(py_mutant unreadable_merged '{ sub(/ \+ a\["timings_unreadable"\]/, ""); print }') || exit 2
expect_rejected 'degenerate clock readings counted as contention' "$unreadable_merged" \
  checker_splits_unreadable 'CONTENDED SESSION'
finish

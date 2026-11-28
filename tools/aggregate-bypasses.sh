#!/usr/bin/env bash
# Report the bypass rate of every hatched Verdict claim, per sweep unit
# (gh-ocannl-996).
#
# A hatched claim (Verdict.p_unless, or a gated form given ~unless) executed and
# then either was CHECKED or took a run-time hatch -- a timing window the
# instrument declared contended, a recording the load left unclean -- and was
# BYPASSED. Both print the same passing golden row, so only these records tell a
# claim verified every night from one verified one night in eight. This script
# turns them into a rate: per unit and claim, the bypassed reaches over all
# reaches, apart from passes (EVALUATED) and skips (SKIP), which
# tools/aggregate-skips.sh judges and this script never reads.
#
# What it reads, as a fail-closed allowlist: each run is a unit's PER-ACTION
# verdict records, concatenated (gh-ocannl-1114), every file opening with its
# OCANNL_TOOL_VERDICT_ACTION header; within them exactly
#   OCANNL_TOOL_VERDICT_BYPASS<TAB>hatches<TAB>executable<TAB>claim
#   OCANNL_TOOL_VERDICT_CHECKED<TAB>hatches<TAB>executable<TAB>claim
# with three non-empty fields, each header with its executable and working
# directory (its arguments may be empty). Every other OCANNL_TOOL_VERDICT_<KIND> record
# passes through unjudged (SKIP, EVALUATED, MEASURE, and kinds of a newer
# Verdict); any other line, a record before the first header, or a run with no
# header at all makes the input malformed (exit 2), since a swept commit
# predating the records would otherwise read as nothing bypassed.
#
# What it deliberately does not do: judge completeness (a unit's records hold
# the actions that RAN -- an incremental unit's cached actions wrote none -- so
# the denominator is this run's reaches, never the suite's claims), compare
# against earlier sweeps (tools/sweep-streaks.sh owns the cross-sweep streak
# check over the saved reports), or fail the sweep. A
# claim bypassed on every unit that reached it is reported UNVERIFIED -- this
# run checked it nowhere -- as a finding for the routine to notice, not an exit
# status: a bound's hatch is valid, and a loaded night is no regression.
#
# Usage:
#   tools/aggregate-bypasses.sh --run m4-max/cc /path/to/cc.verdict-records \
#     --run m4-max/metal /path/to/metal.verdict-records
#
# Exit 0 with the report on stdout; 2 on malformed input or a report that
# cannot be written.

set -uo pipefail

run_units=()
run_records=()

die() {
  echo "aggregate-bypasses: $*" >&2
  exit 2
}

report_line() {
  printf '%s\n' "$1" || die "cannot write report"
}

while [ $# -gt 0 ]; do
  case $1 in
    --run)
      [ $# -ge 3 ] || die "--run needs a unit and a records file"
      [ -n "$2" ] || die "--run needs a non-empty unit"
      run_units+=("$2")
      run_records+=("$3")
      shift 3
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

# macOS's Bash 3.2 treats an empty [@] expansion as unbound under nounset.
if [ ${#run_units[@]} -eq 0 ]; then
  report_line "units: <none>"
  report_line "result: NOT AGGREGATED -- no unit's verdict records were supplied"
  exit 0
fi

for ((i = 0; i < ${#run_units[@]}; i++)); do
  for ((j = i + 1; j < ${#run_units[@]}; j++)); do
    [ "${run_units[$i]}" != "${run_units[$j]}" ] || die "duplicate unit '${run_units[$i]}'"
  done
  [ -r "${run_records[$i]}" ] || die "cannot read ${run_units[$i]} records ${run_records[$i]}"
done

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-bypass-rates.XXXXXX") ||
  die "cannot create temporary directory"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

# One unit's hatched claims as rows: executable, claim, bypassed reaches,
# checked reaches, and the hatches its bypasses took. Exit 3 is malformed input.
unit_rows() {
  awk '
    BEGIN { FS = "\t" }
    # The header Verdict writes: executable, working directory, arguments (possibly none).
    index($0, "OCANNL_TOOL_VERDICT_ACTION\t") == 1 {
      if (NF != 4 || $2 == "" || $3 == "") malformed = 1
      actions++
      next
    }
    index($0, "OCANNL_TOOL_VERDICT_BYPASS\t") == 1 || index($0, "OCANNL_TOOL_VERDICT_CHECKED\t") == 1 {
      if (!actions || NF != 4 || $2 == "" || $3 == "" || $4 == "") { malformed = 1; next }
      key = $3 "\t" $4
      if (!(key in seen)) { seen[key] = 1; order[++n] = key }
      if (index($0, "OCANNL_TOOL_VERDICT_BYPASS\t") == 1) {
        bypassed[key]++
        if (!((key, $2) in took)) {
          took[key, $2] = 1
          # Tested before the assignment, never inside it: mawk (the awk of Debian and Ubuntu) creates
          # the target element of an assignment before it evaluates the right-hand side, so an `in`
          # test there is always true.
          if (key in hatches) hatches[key] = hatches[key] "; " $2
          else hatches[key] = $2
        }
      } else checked[key]++
      next
    }
    /^OCANNL_TOOL_VERDICT_[A-Z][A-Z_]*\t/ { if (!actions) malformed = 1; next }
    { malformed = 1 }
    END {
      if (malformed || !actions) exit 3
      for (i = 1; i <= n; i++) {
        key = order[i]
        printf "%s\t%d\t%d\t%s\n", key, bypassed[key] + 0, checked[key] + 0, hatches[key]
      }
    }
  ' "$1"
}

percent() { # part whole: rounded half up, integers only
  printf '%d%%' $(((200 * $1 + $2) / (2 * $2)))
}

report_line "units: $(
  out=
  for unit in "${run_units[@]}"; do out="${out:+$out, }$unit"; done
  printf '%s' "$out"
)"

: >"$tmp/all-rows"
for ((i = 0; i < ${#run_units[@]}; i++)); do
  unit=${run_units[$i]}
  unit_rows "${run_records[$i]}" >"$tmp/rows" ||
    die "cannot read compatible verdict records for $unit from ${run_records[$i]}"
  reaches=0
  bypassed=0
  while IFS=$'\t' read -r _ _ b c _; do
    reaches=$((reaches + b + c))
    bypassed=$((bypassed + b))
  done <"$tmp/rows"
  if [ "$reaches" -eq 0 ]; then
    report_line "$unit: no hatched claim reached"
    continue
  fi
  report_line "$unit: $bypassed of $reaches hatched claim reach(es) bypassed ($(percent "$bypassed" "$reaches")), $((reaches - bypassed)) checked"
  while IFS=$'\t' read -r exe claim b c hatches; do
    printf '%s\t%s\t%s\t%s\t%s\n' "$unit" "$exe" "$claim" "$b" "$c" >>"$tmp/all-rows" ||
      die "cannot stage the unit's rows"
    [ "$b" -gt 0 ] || continue
    report_line "  bypassed $b of $((b + c)) ($(percent "$b" "$((b + c))")): $exe: $claim [$hatches]"
  done <"$tmp/rows"
done

# Across units: a claim bypassed in every reach of every unit that reached it.
awk '
  BEGIN { FS = "\t" }
  {
    key = $2 "\t" $3
    if (!(key in seen)) { seen[key] = 1; order[++n] = key }
    bypassed[key] += $4
    checked[key] += $5
    reach = $1 " " $4 " of " ($4 + $5)
    # As for the hatches above: the `in` test stays out of the assignment, for mawk.
    if (key in units) units[key] = units[key] ", " reach
    else units[key] = reach
  }
  END {
    for (i = 1; i <= n; i++) {
      key = order[i]
      if (checked[key] == 0) print key "\t" units[key]
    }
  }
' "$tmp/all-rows" >"$tmp/unverified" || die "cannot collect unverified claims"

unverified=$(wc -l <"$tmp/unverified" | tr -d ' ') || die "cannot count unverified claims"
if ! [ -s "$tmp/all-rows" ]; then
  report_line "result: CLEAR -- no hatched claim was reached"
elif [ "$unverified" -eq 0 ]; then
  report_line "result: CLEAR -- every hatched claim reached was checked on some unit"
else
  report_line "result: UNVERIFIED -- $unverified hatched claim(s) bypassed on every unit that reached them"
  while IFS=$'\t' read -r exe claim units; do
    report_line "UNVERIFIED: bypassed on every unit that reached it: $exe: $claim ($units)"
  done <"$tmp/unverified"
fi
exit 0

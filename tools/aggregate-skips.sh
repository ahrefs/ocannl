#!/usr/bin/env bash
# Intersect Verdict.skipped records from complete per-backend test runs.
# Backend-scoped records are judged against the backend vocabulary; environment-
# scoped records are judged against the declared measurement-box vocabulary.
#
# A claim absent from one COMPLETE backend run was evaluated there; a claim
# present in every complete run was not.  The caller owns completeness -- the
# sweep passes only forced full-suite units that passed, or whose every failure
# a serial rerun cleared, never incremental runs.
# A box outside the declared matrix is evidence without obligation: a claim it
# executed is not skipped on every box, but its absence never makes the matrix
# incomplete and its skips alone never make a finding.
#
# Each run is the unit's PER-ACTION verdict records (gh-ocannl-1114): the files
# Verdict writes into OCANNL_TOOL_VERDICT_RECORDS, one per action and rewritten
# by each run of that action, concatenated. Every file opens with its
# OCANNL_TOOL_VERDICT_ACTION header, so the input holds each action's final
# attempt and nothing else -- never the merged stderr of a first attempt and its
# retries. A record kind this script does not judge (OCANNL_TOOL_VERDICT_<KIND>
# from a newer Verdict) passes through; any other line, a record before the
# first header, or an input with no header at all -- a swept commit predating
# the per-action records -- makes the run incompatible rather than empty
# evidence, which would read as execution.
#
# Usage:
#   tools/aggregate-skips.sh \
#     --known cc --known multidev_cc --known metal --known cuda --known hip \
#     --known-box m4-max --known-box minix --known-box rog-nv \
#     --run cc m4-max /path/to/cc.verdict-records \
#     --run metal m4-max /path/to/metal.verdict-records
#
# Exit 1 means a complete backend or declared-box matrix has a claim skipped in
# every member. Partial coverage is a loud report but exits 0 because an absent
# member may have evaluated the claim. Malformed input exits 2.

set -uo pipefail

known=()
known_boxes=()
run_backends=()
run_boxes=()
run_records=()

die() {
  echo "aggregate-skips: $*" >&2
  exit 2
}

report_line() {
  printf '%s\n' "$1" || die "cannot write report"
}

while [ $# -gt 0 ]; do
  case $1 in
    --known)
      [ $# -ge 2 ] || die "--known needs a backend"
      known+=("$2")
      shift 2
      ;;
    --known-box)
      [ $# -ge 2 ] || die "--known-box needs a box"
      known_boxes+=("$2")
      shift 2
      ;;
    --run)
      [ $# -ge 4 ] || die "--run needs a backend, box and records file"
      run_backends+=("$2")
      run_boxes+=("$3")
      run_records+=("$4")
      shift 4
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[ ${#known[@]} -gt 0 ] || die "no known backends supplied"

contains() {
  local wanted=$1 item
  shift
  for item in "$@"; do [ "$item" = "$wanted" ] && return 0; done
  return 1
}

join_by_comma() {
  local out= item
  for item in "$@"; do
    [ -z "$out" ] || out="$out, "
    out="$out$item"
  done
  printf '%s' "$out"
}

for ((i = 0; i < ${#known[@]}; i++)); do
  for ((j = i + 1; j < ${#known[@]}; j++)); do
    [ "${known[$i]}" != "${known[$j]}" ] || die "duplicate known backend '${known[$i]}'"
  done
done

for ((i = 0; i < ${#known_boxes[@]}; i++)); do
  for ((j = i + 1; j < ${#known_boxes[@]}; j++)); do
    [ "${known_boxes[$i]}" != "${known_boxes[$j]}" ] ||
      die "duplicate known box '${known_boxes[$i]}'"
  done
done

for ((i = 0; i < ${#run_backends[@]}; i++)); do
  backend=${run_backends[$i]}
  box=${run_boxes[$i]}
  records=${run_records[$i]}
  contains "$backend" "${known[@]}" || die "run names unknown backend '$backend'"
  [ -r "$records" ] || die "cannot read $backend records $records"
done

# macOS's Bash 3.2 treats an empty [@] expansion as unbound under nounset even
# after [a=()]. Handle it before ANY expansion of run_backends or run_records.
if [ ${#run_backends[@]} -eq 0 ]; then
  report_line "completed backends: <none>"
  report_line "missing backends: $(join_by_comma "${known[@]}")"
  report_line "status: insufficient (0 of ${#known[@]} known backends completed; need at least 2)"
  report_line "result: NOT AGGREGATED"
  if [ ${#known_boxes[@]} -eq 0 ]; then
    report_line "completed boxes: <none>"
    report_line "missing boxes: <none declared>"
    report_line "environment status: unavailable (target declares no measurement-box matrix)"
  else
    report_line "completed boxes: <none>"
    report_line "missing boxes: $(join_by_comma "${known_boxes[@]}")"
    report_line "environment status: insufficient (0 of ${#known_boxes[@]} declared boxes completed; need at least 2)"
  fi
  report_line "environment result: NOT AGGREGATED"
  exit 0
fi

completed_backends=()
missing=()
for backend in "${known[@]}"; do
  if contains "$backend" "${run_backends[@]}"; then
    completed_backends+=("$backend")
  else
    missing+=("$backend")
  fi
done

completed_boxes=()
missing_boxes=()
undeclared_boxes=()
environment_records=()
if [ ${#known_boxes[@]} -gt 0 ]; then
  for box in "${known_boxes[@]}"; do
    if contains "$box" "${run_boxes[@]}"; then
      completed_boxes+=("$box")
    else
      missing_boxes+=("$box")
    fi
  done
  # Every run is environment evidence, declared box or not: an execution on an
  # undeclared box (tuf beside minix for hip) proves the claim is reachable.
  # Only completeness is judged against the declaration.
  environment_records=("${run_records[@]}")
  for box in "${run_boxes[@]}"; do
    contains "$box" "${known_boxes[@]}" && continue
    contains "$box" "${undeclared_boxes[@]:-}" || undeclared_boxes+=("$box")
  done
fi

if [ ${#completed_backends[@]} -eq 0 ]; then
  report_line "completed backends: <none>"
else
  report_line "completed backends: $(join_by_comma "${completed_backends[@]}")"
fi
if [ ${#missing[@]} -eq 0 ]; then
  report_line "missing backends: <none>"
else
  report_line "missing backends: $(join_by_comma "${missing[@]}")"
fi

tmp=$(mktemp -d "${TMPDIR:-/tmp}/ocannl-skip-coverage.XXXXXX") ||
  die "cannot create temporary directory"
cleanup() { rm -rf "$tmp"; }
trap cleanup EXIT

extract_claims() {
  local scope=$1 records=$2
  awk '
    index($0, "OCANNL_TOOL_VERDICT_ACTION\t") == 1 { actions++; next }
    index($0, "OCANNL_TOOL_VERDICT_SKIP\t") == 1 {
      if (!actions) malformed = 1
      record = substr($0, length("OCANNL_TOOL_VERDICT_SKIP\t") + 1)
      fields = split(record, part, "\t")
      if (fields != 3 || part[2] == "" || part[3] == "") malformed = 1
      else if (part[1] == scope || (scope == "sweep" &&
               (part[1] == "backend" || part[1] == "environment")))
        print part[2] "\t" part[3]
      else if (part[1] != "backend" && part[1] != "environment" && part[1] != "outside-sweep") malformed = 1
      next
    }
    /^OCANNL_TOOL_VERDICT_[A-Z][A-Z_]*\t/ { if (!actions) malformed = 1; next }
    { malformed = 1 }
    END { if (malformed || !actions) exit 3 }
  ' scope="$scope" "$records" | LC_ALL=C sort -u
}

intersect_claims() {
  local scope=$1 destination=$2
  shift 2
  local runs=("$@")
  extract_claims "$scope" "${runs[0]}" >"$destination" ||
    die "cannot extract compatible skip records from ${runs[0]}"
  for ((i = 1; i < ${#runs[@]}; i++)); do
    extract_claims "$scope" "${runs[$i]}" >"$tmp/next-$scope" ||
      die "cannot extract compatible skip records from ${runs[$i]}"
    LC_ALL=C comm -12 "$destination" "$tmp/next-$scope" >"$tmp/intersection-$scope" ||
      die "cannot intersect skip records"
    mv "$tmp/intersection-$scope" "$destination" || die "cannot advance skip intersection"
  done
}

# Scope is an observation, not part of a claim's identity. A claim can be
# backend-gated in one run and configuration-gated in another. Backend findings
# still require backend-scoped records in every run. Environment ownership,
# however, is established by an environment record in ANY run; once owned,
# either ordinary scope means that run did not execute the claim.
intersect_claims backend "$tmp/common-backend" "${run_records[@]}"
if [ ${#environment_records[@]} -gt 0 ]; then
  : >"$tmp/environment-owned-unsorted"
  for records in "${environment_records[@]}"; do
    extract_claims environment "$records" >>"$tmp/environment-owned-unsorted" ||
      die "cannot extract compatible skip records from $records"
  done
  LC_ALL=C sort -u "$tmp/environment-owned-unsorted" >"$tmp/environment-owned" ||
    die "cannot collect environment-owned claims"
  intersect_claims sweep "$tmp/common-sweep" "${environment_records[@]}"
  LC_ALL=C comm -12 "$tmp/environment-owned" "$tmp/common-sweep" \
    >"$tmp/common-environment" || die "cannot select environment-owned skip records"
else
  : >"$tmp/common-environment"
fi

failed=0

if [ ${#completed_backends[@]} -lt 2 ]; then
  report_line "status: insufficient (${#completed_backends[@]} of ${#known[@]} known backends completed; need at least 2)"
  report_line "result: NOT AGGREGATED"
else
  common_count=$(wc -l <"$tmp/common-backend" | tr -d ' ') || die "cannot count common skip records"
  if [ ${#missing[@]} -eq 0 ]; then
    report_line "status: complete (${#completed_backends[@]} of ${#known[@]} known backends completed)"
    if [ "$common_count" -eq 0 ]; then
      report_line "result: PASS -- no claim was skipped on every known backend"
    else
      report_line "result: FAIL -- $common_count claim(s) skipped on every known backend"
      while IFS=$'\t' read -r test_id claim; do
        printf 'FAIL: skipped on every known backend: %s: %s\n' "$test_id" "$claim" ||
          die "cannot write report"
      done <"$tmp/common-backend"
      failed=1
    fi
  else
    report_line "status: partial (${#completed_backends[@]} of ${#known[@]} known backends completed)"
    if [ "$common_count" -eq 0 ]; then
      report_line "result: CLEAR across completed backends -- absent backends remain unknown"
    else
      report_line "result: POTENTIAL -- $common_count claim(s) skipped on every completed backend; absent backends remain unknown"
      while IFS=$'\t' read -r test_id claim; do
        printf 'POTENTIAL: skipped on every completed backend: %s: %s\n' "$test_id" "$claim" ||
          die "cannot write report"
      done <"$tmp/common-backend"
    fi
  fi
fi

if [ ${#known_boxes[@]} -eq 0 ]; then
  report_line "completed boxes: <none>"
  report_line "missing boxes: <none declared>"
  report_line "environment status: unavailable (target declares no measurement-box matrix)"
  report_line "environment result: NOT AGGREGATED"
else
  if [ ${#completed_boxes[@]} -eq 0 ]; then
    report_line "completed boxes: <none>"
  else
    report_line "completed boxes: $(join_by_comma "${completed_boxes[@]}")"
  fi
  if [ ${#missing_boxes[@]} -eq 0 ]; then
    report_line "missing boxes: <none>"
  else
    report_line "missing boxes: $(join_by_comma "${missing_boxes[@]}")"
  fi
  if [ ${#undeclared_boxes[@]} -gt 0 ]; then
    report_line "undeclared boxes (their executions count, their absence does not): $(join_by_comma "${undeclared_boxes[@]}")"
  fi

  if [ ${#missing_boxes[@]} -eq 0 ]; then
    environment_count=$(wc -l <"$tmp/common-environment" | tr -d ' ') ||
      die "cannot count common environment skip records"
    report_line "environment status: complete (${#completed_boxes[@]} of ${#known_boxes[@]} declared boxes completed)"
    if [ "$environment_count" -eq 0 ]; then
      report_line "environment result: PASS -- no claim was skipped on every declared box"
    else
      report_line "environment result: FAIL -- $environment_count claim(s) skipped on every declared box"
      while IFS=$'\t' read -r test_id claim; do
        printf 'FAIL: skipped on every declared box: %s: %s\n' "$test_id" "$claim" ||
          die "cannot write report"
      done <"$tmp/common-environment"
      failed=1
    fi
  elif [ ${#completed_boxes[@]} -lt 2 ]; then
    report_line "environment status: insufficient (${#completed_boxes[@]} of ${#known_boxes[@]} declared boxes completed; need at least 2 unless the matrix is complete)"
    report_line "environment result: NOT AGGREGATED"
  else
    environment_count=$(wc -l <"$tmp/common-environment" | tr -d ' ') ||
      die "cannot count common environment skip records"
    report_line "environment status: partial (${#completed_boxes[@]} of ${#known_boxes[@]} declared boxes completed)"
    if [ "$environment_count" -eq 0 ]; then
      report_line "environment result: CLEAR across completed boxes -- absent boxes remain unknown"
    else
      report_line "environment result: POTENTIAL -- $environment_count claim(s) skipped on every completed box; absent boxes remain unknown"
      while IFS=$'\t' read -r test_id claim; do
        printf 'POTENTIAL: skipped on every completed box: %s: %s\n' "$test_id" "$claim" ||
          die "cannot write report"
      done <"$tmp/common-environment"
    fi
  fi
fi

exit "$failed"

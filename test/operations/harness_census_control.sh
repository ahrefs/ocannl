#!/usr/bin/env bash
# Exercise the census record types independently of the checkout's current reach.
set -euo pipefail

{
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    fixture/member.sh fixture/member.sh self - - - \
    fixture/member.sh fixture/helper.sh loads fixture/member.sh:3 helper.sh 1 \
    fixture/member.sh fixture/input.py names fixture/member.sh:4 input.py 3 \
    fixture/member.sh 'fixture/*.txt' declares fixture/member.sh:5 "--only '^example$'" - |
    "$BASH" ./harness_census.sh --project

  # A truncated record has no owner column.
  status=0
  printf '%s\t%s\t%s\t%s\t%s\n' \
    fixture/member.sh fixture/input.py names fixture/member.sh:4 input.py |
    "$BASH" ./harness_census.sh --project 2>harness_census_control.stderr || status=$?
  printf 'malformed row exit %s\n' "$status"
  printf 'stderr: '
  cat harness_census_control.stderr
} >harness_census_control.expected.corrected

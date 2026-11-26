#!/usr/bin/env bash
# Fleet owed-record hook (gh-ocannl-2359, gh-ocannl-2414, gh-ocannl-2468). Reads
# <leg record path> <request ID> <observed SHA> after a clean, successful main leg.
# Prefers `FLEET_LEG_RECEIPT_DIR` (handed by the fleet from `FLEET_LEG_RECEIPT_LINK`, covering
# receipts in a private `OCANNL_TOOL_TEST_RUNS` without falling back to `last`), routing a
# `kind=linux-shell` receipt to `owed-run.sh`'s linux-shell certifier and any other receipt to
# `record_run`. Only when `FLEET_LEG_RECEIPT_DIR` is unset does it fall back to the worktree's
# `linux-shell` pointer or `tools/test-run.sh paths run last`.
# Every receipt must carry that exact leg record path and observed SHA and pass the eligibility
# checks in tools/owed-run.sh. Validly bound, clean, exit-0 receipts for a non-`cc` backend or a
# non-owed command (focused/targeted runs, targets files) report `ineligible leg` instead of
# `record refused`; broken bindings, stale receipts, wrong heads, and dirty/failed runs report
# `record refused`.
# Targets files, older receipts, other runs and the fleet's temporary file contents remain outside
# the publication allowlist.
# A publication refusal leaves the leg verdict intact. Use `owed-run.sh record <actual run dir>`
# for a qualifying receipt, or relay its printed line from a box with access.
# Malformed callback arguments exit 2. The fleet logs the hook failure alongside the main verdict.
exec "${BASH:-bash}" "$(dirname -- "$0")/owed-run.sh" leg-record "$@"

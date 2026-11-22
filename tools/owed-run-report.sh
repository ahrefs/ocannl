#!/usr/bin/env bash
# Fleet owed-record hook (gh-ocannl-2359). Reads <leg record path> <request ID> <observed SHA>
# after a clean, successful main leg. The worktree's last tools/test-run.sh receipt in the hook's
# runner state directory must carry that exact path and SHA and pass the eligibility checks in
# tools/owed-run.sh. Only an explicit
# `run --expect-backend cc` request for a FULL or TRAIN form qualifies. The hook reads the
# original and effective Dune commands from the receipt. Targets files, older receipts, other
# runs and the fleet's temporary file contents are outside this allowlist.
# A publication refusal leaves the leg verdict intact. Use `owed-run.sh record <actual run dir>`
# for a qualifying receipt, or relay its printed line from a box with access.
exec "${BASH:-bash}" "$(dirname -- "$0")/owed-run.sh" leg-record "$@"

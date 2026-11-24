#!/usr/bin/env bash
# Fleet owed-record hook (gh-ocannl-2359, gh-ocannl-2414). Reads <leg record path> <request ID>
# <observed SHA> after a clean, successful main leg. Routes a matching tools/linux-shell-leg.sh
# receipt (via FLEET_LEG_RECEIPT_DIR or the worktree's linux-shell last pointer) to
# `owed-run.sh`'s linux-shell certifier, or else validates the worktree's last tools/test-run.sh
# receipt in the hook's runner state directory for an explicit `run --expect-backend cc` FULL or
# TRAIN request. Every receipt must carry that exact leg record path and observed SHA and pass the
# eligibility checks in tools/owed-run.sh.
# Targets files, older receipts, other runs and the fleet's temporary file contents are outside
# this allowlist.
# A publication refusal leaves the leg verdict intact. Use `owed-run.sh record <actual run dir>`
# for a qualifying receipt, or relay its printed line from a box with access.
# Malformed callback arguments exit 2. The fleet logs the hook failure alongside the main verdict.
exec "${BASH:-bash}" "$(dirname -- "$0")/owed-run.sh" leg-record "$@"

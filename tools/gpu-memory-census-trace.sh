#!/usr/bin/env bash
# Adapter to test-run.sh's slowest-reader hook. The shared parser leaves the
# runner's timing output intact and publishes only its allowlisted process
# fields to the census. test-run.sh owns the raw trace and its cleanup.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/action-durations.sh" --processes-json-to "${OCANNL_TOOL_CENSUS_JSON:?}" "$@"

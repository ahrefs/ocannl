#!/usr/bin/env bash
# Editorial checklist of public source declarations changed on first-parent history.
# Usage: tools/api-drift.sh <since-rev> [until-rev]
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
exec dune exec tools/api_drift.exe -- "$@"

#!/usr/bin/env bash
# Editorial checklist of public source declarations changed on first-parent history.
# Usage: tools/api-drift.sh [--context N] <since-rev> [until-rev]
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies.
# shellcheck source=credential-env.sh
. tools/credential-env.sh
eval "$(credential_env_scrub_text)" || {
  echo "api-drift.sh: cannot remove credential variables:$credential_env_left" >&2
  exit 2
}
exec dune exec tools/api_drift.exe -- "$@"

#!/usr/bin/env bash
# Fleet run-report hook (gh-ocannl-2253). Reads the runner's argv as separate words:
# [env OCANNL_BACKEND=cc] [./]tools/test-run.sh run --expect-backend cc <FULL or TRAIN argv>.
# tools/owed-run.sh owns both Dune forms. Wrapper commands and targets files are outside this
# allowlist. The hook sees neither the fleet verdict nor an invocation ID. It can publish only
# the worktree's last run record. It must match HEAD and the requested coverage, identify the
# current fleet leg in its recorded launcher ancestry, and pass manual record validation. The
# ancestry check reads only the live enclosing process group's identity and the receipt's
# launchers file. A dirty checkout or a head on the record repository's master publishes nothing.
# If publication fails, the report prints the refusal. The runner's verdict stands, and the
# same run directory can be recorded manually or its printed line relayed from a box with access.
exec "${BASH:-bash}" "$(dirname -- "$0")/owed-run.sh" leg-report "$@"

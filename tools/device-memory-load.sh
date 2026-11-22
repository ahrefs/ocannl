#!/bin/sh
# Usage: tools/device-memory-load.sh [--min-mib N] [--max-mib N] [--keep N]
#                                    [--ready-timeout SECONDS] -- CMD [ARGS...]
# Put the whole helper inside the caller's GPU slot. It starts CMD only after the hog
# reports readiness following its first retained allocation window. It cleans up the
# hog's process group on command exit, startup failure or catchable cancellation.
# Cleanup covers processes that stay in the hog's group.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$here/device-memory-load.py" "$@"

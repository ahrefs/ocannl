#!/bin/sh
# Usage: tools/device-memory-load.sh [--min-mib N] [--max-mib N] [--keep N]
#                                    [--ready-timeout SECONDS] -- CMD [ARGS...]
# Put the whole helper inside the caller's GPU slot. The first complete line from
# the hog must be "ready", sent after its first retained allocation window. CMD
# receives its usual standard streams and runs in a separate POSIX process group.
# Unexpected hog exit fails the run. On exit or INT, TERM or HUP cancellation
# cleanup covers processes that stay in either of the managed groups.
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec python3 -B "$here/device-memory-load.py" "$@"

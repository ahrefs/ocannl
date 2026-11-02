#!/usr/bin/env bash
# Run tools/vendor_idents.exe's check against this box's toolchain and say in one line what it
# found (gh-ocannl-1705). The CUDA and HIP routine-name tables (arrayjit/lib/vendor_idents_<b>.ml)
# are generated from ONE toolkit's headers (gh-ocannl-1338), so an SDK upgrade that declares a new
# file-scope name -- or drops one -- leaves the table stale with nothing to notice it: the check
# mode only ran by hand. tools/sweep.sh runs this after each CUDA and HIP unit whose suite ran (its
# diagnostic legs), and a worker runs it by hand after a toolkit upgrade.
#
# Usage: tools/vendor-idents-check.sh <cuda|hip|metal>   (from any directory; needs the backend's
#                                                       vendor library, which the arm links)
#
# vendor_idents.exe's own report (the measurement, the missing and superfluous names, the
# regenerated table) comes first. The last line on stdout is the verdict, and the exit status says
# the same:
#   vendor-idents: verdict: holds (...)          0  the table matches the measurement
#   vendor-idents: verdict: TABLE STALE (...)    1  the toolchain declares names the table lacks, or
#                                                   the table lists names that now compile: paste
#                                                   the generated module the report prints
#   vendor-idents: verdict: unavailable: ...     2  nothing was measured (the arm is not built, the
#                                                   headers were not found), or the tool stopped
#                                                   before its own verdict (a failed build, a crash)
# Its boundary: it reads vendor_idents.exe's exit status and its `verdict:`, `fail, yet` and
# `in the table, yet` lines, nothing else of its output.
set -u
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
cd "$root" || exit 2

verdict() { # status text
  printf 'vendor-idents: verdict: %s\n' "$2"
  exit "$1"
}

case ${1:-} in
  cuda | hip | metal) backend=$1 ;;
  *) echo "usage: tools/vendor-idents-check.sh <cuda|hip|metal>" >&2; exit 2 ;;
esac

# Dune records the environment of every process it spawns in _build/trace.csexp (gh-ocannl-1280).
# shellcheck source=credential-env.sh
. tools/credential-env.sh || verdict 2 "unavailable: cannot read tools/credential-env.sh"
eval "$(credential_env_scrub_text)" ||
  verdict 2 "unavailable: cannot remove credential variables:$credential_env_left"

work=$(mktemp -d "${TMPDIR:-/tmp}/vendor-idents-check.XXXXXX") ||
  verdict 2 "unavailable: no temporary directory"
trap 'rm -rf "$work"' EXIT
echo "=== vendor-idents $backend ==="
dune exec --no-print-directory tools/vendor_idents.exe -- "$backend" >"$work/out" 2>&1
rc=$?
cat "$work/out"
said=$(sed -n 's/^verdict: //p' "$work/out" | tail -n 1)
missing=$(sed -n 's/^fail, yet kernel_ident passes them through (missing from the table): //p' "$work/out")
superfluous=$(sed -n 's/^in the table, yet compile as kernel names (superfluous): //p' "$work/out")
# Metal's table is the hand-kept grammar in Metal_backend (gh-ocannl-707), not a generated module.
fix="regenerate arrayjit/lib/vendor_idents_$backend.ml from this report"
[ "$backend" != metal ] || fix="extend Metal_backend's grammar"
# Without its own verdict line, a status is the build's or a crash's, never the measurement's.
case $rc:${said:+said} in
  0:said) verdict 0 "holds ($said)" ;;
  1:said) verdict 1 "TABLE STALE ($said: ${missing:-?} missing, ${superfluous:-?} superfluous; $fix)" ;;
  2:said) verdict 2 "unavailable: $said" ;;
  *) verdict 2 "unavailable: dune exec tools/vendor_idents.exe exited $rc before a verdict" ;;
esac

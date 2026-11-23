#!/bin/sh
# harness: none -- hand-run diagnostic, requires a Mac with the GPU to itself
# gh-ocannl-2082, gh-ocannl-2139: builds metal_hang_probe and runs the cells named on the command
# line, one process each, in order (a cell is its words joined by `:`, e.g. `stall:1:120`;
# `bg:<cell>` starts one in the background, `sleep:N` waits N s, `waitbg` waits for the background
# cell). Stalls the GPU: run it only on a Mac with the GPU to itself. Prints the platform first;
# every cell's lines go to stdout. Exits 1 if any cell exited nonzero, after running the rest; one
# background cell at a time, joined before the script exits and killed if it is interrupted.
set -u
cd "$(dirname "$0")/../.." || exit 2
dune build ./test/metal_probe/metal_hang_probe.exe || exit 2
P="${DUNE_BUILD_DIR:-_build}/default/test/metal_probe/metal_hang_probe.exe"
echo "host=$(hostname -s) $(sw_vers -productVersion) $(sw_vers -buildVersion) $(sysctl -n machdep.cpu.brand_string)"
echo "uptime: $(uptime)"
echo "console user: $(stat -f %Su /dev/console)"
pmset -g assertions | grep -iE 'display|PreventUserIdleDisplaySleep' | head -5
system_profiler SPDisplaysDataType 2>/dev/null | grep -iE 'Chipset|Total Number of Cores|Display Type|Resolution|Online|Main Display' | sed 's/^ */  /'
BG=
RC=0
trap 'if [ -n "$BG" ]; then kill "$BG" 2>/dev/null; fi; exit 130' INT TERM
for c in "$@"; do
  # shellcheck disable=SC2086
  set -- $(echo "$c" | tr ':' ' ')
  echo "--- cell $* at $(date -u +%H:%M:%S)"
  case "$1" in
    bg)
      shift
      if [ -n "$BG" ]; then
        echo "--- refused: the background cell (pid $BG) is still running; waitbg first"
        st=2
      else
        "$P" "$@" &
        BG=$!
        st=0
        echo "--- started in the background: pid $BG"
      fi
      ;;
    sleep) sleep "$2"; st=$? ;;
    waitbg)
      if [ -n "$BG" ]; then
        wait "$BG"; st=$?
        BG=
      else
        echo "--- refused: no background cell to wait for"
        st=2
      fi
      ;;
    *) "$P" "$@"; st=$? ;;
  esac
  echo "--- cell exit=$st at $(date -u +%H:%M:%S)"
  [ "$st" = 0 ] || RC=1
done
if [ -n "$BG" ]; then
  echo "--- joining the background cell (pid $BG)"
  wait "$BG" || RC=1
fi
exit "$RC"

#!/bin/sh
# The compiler reports success; only the codesign case produces an artifact. The code_signing case
# (gh-ocannl-1717) runs the real compiler command that follows the mode, then strips the library's
# code signature where the host has a codesign tool, so a loader that enforces signatures refuses it.
mode=$1
shift
if [ "$mode" = artifact_missing ]; then
  exit 0
fi
if [ "$mode" = code_signing ]; then
  "$@" || exit $?
  while [ "$#" -gt 0 ]; do
    if [ "$1" = -o ]; then
      case "$2" in
        *.o) ;;
        *) if command -v codesign > /dev/null 2>&1; then
             codesign --remove-signature "$2" || exit 1
           fi ;;
      esac
      exit 0
    fi
    shift
  done
  exit 0
fi
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then
    printf 'gh1142 signing fixture\n' > "$2"
    exit 0
  fi
  shift
done
exit 1

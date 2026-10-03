#!/bin/sh
# The compiler reports success; only the codesign case produces an artifact.
mode=$1
shift
if [ "$mode" = artifact_missing ]; then
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

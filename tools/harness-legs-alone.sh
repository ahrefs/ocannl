#!/bin/sh
# Invoke the Bash on PATH with the caller's descriptors intact. uutils env
# fills closed standard descriptors with /dev/null (gh-ocannl-2401).
script_dir=$(dirname -- "$0") || exit 2
exec bash "$script_dir/harness-legs-alone-impl.sh" "$@"

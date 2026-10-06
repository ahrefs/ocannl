#!/usr/bin/env bash
# Run dune with benign mingw linker noise filtered out of stderr.
#
# On Windows, every link step floods stderr with binutils warnings triggered by
# MSVC-produced import libraries (e.g. the ROCm/CUDA ones):
#
#   Warning: corrupt .drectve at end of def file
#   Warning: .drectve `...' unrecognized
#
# They are harmless but bury real errors. This wrapper drops exactly those
# lines, passes everything else through unchanged (stdout untouched), and
# preserves dune's exit status.
#
# Usage: tools/dune-quiet.sh <any dune arguments>
#   e.g. tools/dune-quiet.sh build --root . @check

command -v dune >/dev/null 2>&1 || . "$(dirname "$0")/opam-env.sh"
# Credentials never reach dune, which records every spawned process's environment in
# `_build/trace.csexp` (gh-ocannl-1280): the deny-list tools/test-run.sh applies, after the opam
# environment above, whose updates could set one.
# shellcheck source=credential-env.sh
. "$(dirname "$0")/credential-env.sh" || exit 2
eval "$(credential_env_scrub_text)" || {
  echo "dune-quiet.sh: cannot remove credential variables:$credential_env_left" >&2
  exit 2
}

status=0
{
  dune "$@" 2>&1 1>&3 3>&- \
    | grep -vE 'Warning: (corrupt \.drectve at end of def file|\.drectve .* unrecognized)' >&2
  status=${PIPESTATUS[0]}
} 3>&1
exit "$status"

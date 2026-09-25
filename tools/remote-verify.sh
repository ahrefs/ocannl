#!/usr/bin/env bash
# Deprecated name of tools/machine-verify.sh (gh-ocannl-1047), kept so the lab
# notes, plans and briefs that name it keep working. It forwards every argument
# unchanged, so it inherits the new placement rule too: a BOX that is this
# machine now runs here instead of failing at self-ssh. New callers name
# tools/machine-verify.sh.
echo "remote-verify.sh: deprecated name; forwarding to tools/machine-verify.sh" >&2
exec bash "$(cd "$(dirname "$0")" && pwd -P)/machine-verify.sh" "$@"

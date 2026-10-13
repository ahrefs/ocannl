#!/usr/bin/env bash
# Checked Windows dependency-cache payload; never overlay actions/cache's tar
# directly onto setup-ocaml's freshly materialised Cygwin tree.
# Usage: windows-opam-cache.sh pack|restore PAYLOAD SWITCH CYGWIN
# PAYLOAD holds switch.tar and cygwin.tar. The derived CA trust subtree stays
# owned by setup-ocaml; all mingw libraries and Cygwin package metadata remain.
# The switch is archived with its symlinks resolved: native Windows programs
# run its binaries (dune runs flexlink.exe and ocamllex.exe by CreateProcess),
# and a symlink restored as a Cygwin system link file is no executable to
# them -- "This version of ... is not compatible with the version of Windows
# you're running" (gh-ocannl-1271). The Cygwin tree keeps its links: Cygwin
# itself follows them, and some name Cygwin-absolute targets.
set -euo pipefail
[ "$#" = 4 ] || { echo "usage: $0 pack|restore PAYLOAD SWITCH CYGWIN" >&2; exit 2; }
mode=$1 payload=$2 switch=$3 cygwin=$4
case "$mode" in pack|restore) ;; *) echo "unknown mode: $mode" >&2; exit 2 ;; esac
# Git Bash understands opam's native drive paths through cygpath, while tar
# interprets a drive colon as a remote archive. Normalize every caller path.
case "$(uname -s)" in
  MINGW*|MSYS*)
    payload=$(cygpath -u "$payload")
    switch=$(cygpath -u "$switch")
    cygwin=$(cygpath -u "$cygwin")
    # MSYS defaults to copying symlink targets, which fails for forward and
    # Cygwin-absolute links. Cygwin-compatible system link files preserve the
    # POSIX targets without native symlink privilege. Child tar reads this
    # process-local setting at startup; the caller's environment is untouched.
    export MSYS=winsymlinks:sys ;;
esac
[ -d "$switch" ] && [ -d "$cygwin" ] || {
  echo "cache requires existing switch and Cygwin roots" >&2; exit 2;
}
# Explicit exclusion at BOTH boundaries: a cache hit must never replace the
# freshly generated CA files, including when an older payload contains them.
ca='./root/etc/pki/ca-trust/extracted'
case "$mode" in
  pack)
    mkdir -p "$payload"
    tar -h -cf "$payload/switch.tar.tmp" -C "$switch" .
    tar --exclude="$ca" -cf "$payload/cygwin.tar.tmp" -C "$cygwin" .
    mv "$payload/switch.tar.tmp" "$payload/switch.tar"
    mv "$payload/cygwin.tar.tmp" "$payload/cygwin.tar"
    ;;
  restore)
    # List both before modifying either root; truncation must fail before it
    # can leave an apparently warm dependency switch. Extraction also gates
    # the job: no ignored tar exit and no install-on-partial-restore fallback.
    # A symlink member in the switch is a payload this script did not pack
    # (or an older layout), and would restore as a file no native program can
    # run: refuse it too. awk reads the whole listing, so tar never sees SIGPIPE.
    links=$(tar -tvf "$payload/switch.tar" | awk '/^l/ && !seen++')
    tar -tf "$payload/cygwin.tar" >/dev/null
    if [ -n "$links" ]; then
      printf 'switch payload carries a symlink, which native programs cannot run once restored: %s\n' "$links" >&2
      exit 3
    fi
    tar --exclude="$ca" -xf "$payload/cygwin.tar" -C "$cygwin"
    tar -xf "$payload/switch.tar" -C "$switch"
    ;;
esac
printf 'windows-opam-cache: %s complete\n' "$mode"

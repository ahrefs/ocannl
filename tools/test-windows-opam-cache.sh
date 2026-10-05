#!/usr/bin/env bash
# Real tar cache round trips, CA collision and checked-extraction controls.
# Runs on POSIX and native Git Bash; never touches the user's opam root.
# Usage: tools/test-windows-opam-cache.sh [--keep]
set -u
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$HERE/../scripts/harness-support.sh"
harness_args "$@"
harness_require tar perl
harness_scratch test-windows-opam-cache
SRC="$HERE/windows-opam-cache.sh"
real_tar=$(command -v tar)
printf 'cache fixture host: %s; %s\n' "$(uname -s)" "$(git --version)"

seed() {
  local base=$1
  mkdir -p "$base/switch/lib" "$base/cygwin/root/usr/x86_64-w64-mingw32/lib" \
    "$base/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash" \
    "$base/cygwin/root/etc/setup"
  printf 'installed dependency\n' >"$base/switch/lib/dependency"
  printf 'mingw system library\n' >"$base/cygwin/root/usr/x86_64-w64-mingw32/lib/libfixture.a"
  printf 'package metadata\n' >"$base/cygwin/root/etc/setup/installed.db"
  printf 'setup-ocaml fresh CA\n' >"$base/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0"
}
roundtrip() {
  local subject=$1 name=$2 base="$TMP/$2"
  seed "$base/source"
  bash "$subject" pack "$base/payload" "$base/source/switch" "$base/source/cygwin" >"$base/pack.log" 2>&1 || return 1
  tar -tf "$base/payload/cygwin.tar" >"$base/members"
  if grep -q 'ca-trust/extracted' "$base/members"; then return 1; fi
  # actions/cache archives the payload directory, then restores that directory.
  tar -cf "$base/cache.tar" -C "$base" payload || return 1
  mkdir "$base/download"
  tar -xf "$base/cache.tar" -C "$base/download" || return 1
  seed "$base/fresh"
  rm "$base/fresh/switch/lib/dependency" "$base/fresh/cygwin/root/usr/x86_64-w64-mingw32/lib/libfixture.a" "$base/fresh/cygwin/root/etc/setup/installed.db"
  bash "$subject" restore "$base/download/payload" "$base/fresh/switch" "$base/fresh/cygwin" >"$base/restore.log" 2>&1 || return 1
  cmp "$base/source/switch/lib/dependency" "$base/fresh/switch/lib/dependency" &&
    cmp "$base/source/cygwin/root/usr/x86_64-w64-mingw32/lib/libfixture.a" "$base/fresh/cygwin/root/usr/x86_64-w64-mingw32/lib/libfixture.a" &&
    cmp "$base/source/cygwin/root/etc/setup/installed.db" "$base/fresh/cygwin/root/etc/setup/installed.db" &&
    cmp "$base/source/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0" "$base/fresh/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0"
}
roundtrip "$SRC" roundtrip; report $? 'archive cache round trip retains dependencies, mingw and fresh CA'
# A legacy payload with an actual tar symlink member, independent of Git Bash's
# ln -s copy emulation or the user's Windows symlink privilege.
perl -e '
  $name="./root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0";
  $h=pack("a100a8a8a8a12a12a8a1a100a6a2a32a32a8a8a155a12",
    $name,"0000777","0000000","0000000","00000000000","00000000000",
    "        ","2","Telekom_Security_TLS_RSA_Root_2023.pem","ustar","00",
    "","","","","","");
  $sum=0; $sum+=$_ for unpack("C*",$h);
  substr($h,148,8)=sprintf("%06o\0 ",$sum);
  print $h, "\0" x 1024;
' >"$TMP/legacy.tar"
ca_collision() {
  local subject=$1 name=$2 base="$TMP/$2" rc=0
  seed "$base"
  mkdir "$base/payload"
  cp "$TMP/roundtrip/payload/switch.tar" "$base/payload/switch.tar"
  cp "$TMP/legacy.tar" "$base/payload/cygwin.tar"
  bash "$subject" restore "$base/payload" "$base/switch" "$base/cygwin" >"$base/restore.log" 2>&1 || rc=$?
  [ "$rc" = 0 ] && [ ! -L "$base/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0" ] &&
    grep -qx 'setup-ocaml fresh CA' "$base/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0"
}
ca_collision "$SRC" collision; report $? 'cached CA symlink never overlays setup-ocaml regular CA file'
# A corrupted cache must fail before either root is modified.
seed "$TMP/corrupt"
mkdir "$TMP/corrupt/payload"
cp "$TMP/roundtrip/payload/switch.tar" "$TMP/corrupt/payload/switch.tar"
printf 'broken archive' >"$TMP/corrupt/payload/cygwin.tar"
rm "$TMP/corrupt/switch/lib/dependency"
rc=0
bash "$SRC" restore "$TMP/corrupt/payload" "$TMP/corrupt/switch" "$TMP/corrupt/cygwin" >"$TMP/corrupt/log" 2>&1 || rc=$?
[ "$rc" != 0 ] && [ ! -e "$TMP/corrupt/switch/lib/dependency" ]; report $? 'corrupt cache fails before restoring the switch'
# Extraction refusal must remain fatal after valid archive listing.
mkdir "$TMP/bin"
cat >"$TMP/bin/tar" <<'SHIM'
#!/usr/bin/env bash
for arg in "$@"; do
  if [ "$arg" = -xf ]; then echo 'fixture extraction refused' >&2; exit 17; fi
done
exec "$CACHE_FIXTURE_TAR" "$@"
SHIM
chmod +x "$TMP/bin/tar"
extraction_refused() {
  local subject=$1 log=$2 rc=0
  PATH="$TMP/bin:$PATH" CACHE_FIXTURE_TAR="$real_tar" bash "$subject" restore "$TMP/roundtrip/payload" "$TMP/corrupt/switch" "$TMP/corrupt/cygwin" >"$log" 2>&1 || rc=$?
  [ "$rc" = 17 ] && grep -q 'fixture extraction refused' "$log" && [ ! -e "$TMP/corrupt/switch/lib/dependency" ]
}
extraction_refused "$SRC" "$TMP/refusal.log"; report $? 'extraction error stops warm-cache path'
# Mutants exercise the same behavioral oracles, with no source-text pass claim.
pack_mutant=$(mutant pack-includes-ca '{sub(/tar --exclude="\$ca" -cf/, "tar -cf"); print}')
if roundtrip "$pack_mutant" mutant-pack; then report 1 'negative control: omitted pack exclusion'; else
  grep -q 'ca-trust/extracted' "$TMP/mutant-pack/members"; report $? 'negative control: omitted pack exclusion'
fi
restore_mutant=$(mutant restore-overlays-ca '{sub(/tar --exclude="\$ca" -xf/, "tar -xf"); print}')
if ca_collision "$restore_mutant" mutant-restore; then report 1 'negative control: omitted restore exclusion'; else
  # The same real symlink archive was listed and restore reached extraction.
  grep -q 'ee37c333.0' "$TMP/mutant-restore/restore.log" || [ -L "$TMP/mutant-restore/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0" ] ||
    ! grep -qx 'setup-ocaml fresh CA' "$TMP/mutant-restore/cygwin/root/etc/pki/ca-trust/extracted/pem/directory-hash/ee37c333.0"
  report $? 'negative control: omitted restore exclusion'
fi
error_mutant=$(mutant ignores-extraction-error '{sub(/set -euo pipefail/, "set -uo pipefail"); print}')
if extraction_refused "$error_mutant" "$TMP/mutant-refusal.log"; then report 1 'negative control: ignored extraction status'; else
  grep -q 'fixture extraction refused' "$TMP/mutant-refusal.log" && grep -q 'restore complete' "$TMP/mutant-refusal.log"
  report $? 'negative control: ignored extraction status'
fi
finish

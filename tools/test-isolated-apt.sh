#!/usr/bin/env bash
# affected-reads: tools/ci-compiler-test.sh tools/ci-aarch64-cross.sh
# Offline integration of tools/isolated-apt.sh, the setup sourced by
# tools/ci-compiler-test.sh and tools/ci-aarch64-cross.sh. No compiler downloads.
# Boundary: real apt verifies a generated local archive and two throwaway keys.
# A writable fixture host substitutes for host trust/cache/status. Cache files
# are explicitly enabled (apt versions may default to no persistent cache);
# effective status is checked because update reads, rather than writes, status.
# Refusals require a nonzero status AND apt 2's NO_PUBKEY or apt 3's missing-key
# diagnostic. A real missing-suite error must be rejected by that predicate.
# Standing controls remove signed-by, restore host trust and leak real caches;
# they require the opposite outcomes from the corresponding isolation checks.
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
. "$here/isolated-apt.sh"
harness_args "$@"
harness_scratch test-isolated-apt

missing_key_refusal() {
  [ "$1" != 0 ] && grep -Eq 'NO_PUBKEY|Missing key [A-Fa-f0-9]+.*needed to verify signature' "$2"
}
expect_success() {
  [ "$1" = 0 ] || cat "$3"
  report "$1" "$2"
}
strip_signed_by() {
  sed 's/ \[signed-by=[^]]*\]//' "$1" >"$TMP/unsigned-source"
  cp "$TMP/unsigned-source" "$1"
}

leg_1() {
  if ! command -v apt-get >/dev/null 2>&1; then
    harness_not_applicable 'apt-get is absent on this host'; return
  fi
  local cmd name rc root host repo config i
  local cache_options=()
  for cmd in apt-cache apt-config gpg gpgconf python3 timeout; do
    command -v "$cmd" >/dev/null 2>&1 || { harness_abandon "$cmd is absent"; return; }
  done
  apt-get --version >"$TMP/apt-version"
  head -n 1 "$TMP/apt-version"
  export LC_ALL=C
  root="$TMP/index"
  host="$TMP/host"
  repo="$TMP/repository"
  config="$TMP/apt.conf"
  mkdir -p "$host/trustedparts" "$host/cache" "$repo/dists/fixture/main/binary-amd64" "$TMP/gnupg"
  chmod 700 "$TMP/gnupg"
  for name in right wrong; do
    gpg --homedir "$TMP/gnupg" --batch --pinentry-mode loopback --passphrase '' \
      --quick-generate-key "$name <fixture-$name@example.invalid>" rsa2048 sign 0 >"$TMP/key-$name.log" 2>&1
    gpg --homedir "$TMP/gnupg" --batch --export "$name" >"$TMP/$name.gpg"
  done
  cp "$TMP/right.gpg" "$host/trusted.gpg"
  cp "$TMP/right.gpg" "$host/trustedparts/right.gpg"
  printf 'Package: fixture-only\nVersion: 1\nArchitecture: amd64\nDescription: host sentinel\nStatus: install ok installed\n\n' >"$host/status"
  : >"$repo/dists/fixture/main/binary-amd64/Packages"
  python3 - "$repo/dists/fixture" <<'PY'
import hashlib, pathlib, sys
root = pathlib.Path(sys.argv[1])
p = root / "main/binary-amd64/Packages"
data = p.read_bytes()
(root / "Release").write_text(
    "Suite: fixture\nCodename: fixture\nArchitectures: amd64\nComponents: main\n"
    "Date: Mon, 16 Nov 2026 00:00:00 UTC\nSHA256:\n"
    f" {hashlib.sha256(data).hexdigest()} {len(data)} main/binary-amd64/Packages\n")
PY
  gpg --homedir "$TMP/gnupg" --batch --yes --pinentry-mode loopback --passphrase '' \
    --local-user right --clearsign --output "$repo/dists/fixture/InRelease" "$repo/dists/fixture/Release"
  gpgconf --homedir "$TMP/gnupg" --kill gpg-agent
  # Load host defaults before the real setup's command-line isolation.
  cat >"$config" <<CONF
Dir::Etc::trusted "$host/trusted.gpg";
Dir::Etc::trustedparts "$host/trustedparts";
Dir::Cache "$host/cache";
Dir::Cache::pkgcache "pkgcache.bin";
Dir::Cache::srcpkgcache "srcpkgcache.bin";
Dir::State::status "$host/status";
APT::Architecture "amd64";
APT::Architectures { "amd64"; };
Acquire::Languages "none";
Acquire::Check-Date "false";
CONF
  isolated_apt_setup "$root" "$TMP/right.gpg" "file:$repo" main fixture
  # Fixture host is writable: a lost Dir::Cache really publishes here.
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" update >"$TMP/right.log" 2>&1 || rc=$?
  expect_success "$rc" 'correct source key verifies the local archive' "$TMP/right.log"
  rc=0
  timeout 30 apt-cache -c "$config" "${isolated_apt_options[@]}" gencaches >"$TMP/cache.log" 2>&1 || rc=$?
  expect_success "$rc" 'real apt cache generation succeeds' "$TMP/cache.log"
  if [ -s "$root/cache/pkgcache.bin" ] && [ -s "$root/cache/srcpkgcache.bin" ] &&
     [ ! -e "$host/cache/pkgcache.bin" ] && [ ! -e "$host/cache/srcpkgcache.bin" ]; then
    report 0 'apt writes caches only in scratch'
  else
    cat "$TMP/cache.log"
    find "$root/cache" "$host/cache" -type f
    report 1 'apt writes caches only in scratch'
  fi
  apt-config -c "$config" "${isolated_apt_options[@]}" shell fixture_status Dir::State::status >"$TMP/status-path"
  if [ "$(cat "$TMP/status-path")" = "fixture_status='$root/status'" ] && [ ! -s "$root/status" ]; then
    report 0 'effective dpkg status is the empty scratch file'
  else cat "$TMP/status-path"; report 1 'effective dpkg status is the empty scratch file'; fi
  # Reinitialize in a fresh list root, so no cached authenticated list can win.
  isolated_apt_setup "$TMP/wrong-index" "$TMP/wrong.gpg" "file:$repo" main fixture
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" update >"$TMP/wrong.log" 2>&1 || rc=$?
  if missing_key_refusal "$rc" "$TMP/wrong.log"; then
    report 0 'host trust cannot rescue an unrelated source key'
  else cat "$TMP/wrong.log"; report 1 'host trust cannot rescue an unrelated source key'; fi
  # signed-by takes precedence over global trust. Remove it only in this probe
  # to make a restored global trust setting observable independently.
  isolated_apt_setup "$TMP/global-index" "$TMP/wrong.gpg" "file:$repo" main fixture
  strip_signed_by "$TMP/global-index/sources.list"
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" update >"$TMP/global.log" 2>&1 || rc=$?
  if missing_key_refusal "$rc" "$TMP/global.log"; then
    report 0 'global host trust remains disabled without signed-by'
  else cat "$TMP/global.log"; report 1 'global host trust remains disabled without signed-by'; fi
  # Opposing controls stay in CI: prove each isolation observation can change.
  isolated_apt_setup "$TMP/no-signed-by" "$TMP/right.gpg" "file:$repo" main fixture
  strip_signed_by "$TMP/no-signed-by/sources.list"
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" update >"$TMP/no-signed-by.log" 2>&1 || rc=$?
  if missing_key_refusal "$rc" "$TMP/no-signed-by.log"; then
    report 0 'control: removing signed-by loses the correct source key'
  else cat "$TMP/no-signed-by.log"; report 1 'control: removing signed-by loses the correct source key'; fi

  isolated_apt_setup "$TMP/host-trust" "$TMP/right.gpg" "file:$repo" main fixture
  strip_signed_by "$TMP/host-trust/sources.list"
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" \
    -o "Dir::Etc::trusted=$host/trusted.gpg" -o "Dir::Etc::trustedparts=$host/trustedparts" \
    update >"$TMP/host-trust.log" 2>&1 || rc=$?
  expect_success "$rc" 'control: restored host trust verifies an unscoped source' "$TMP/host-trust.log"

  isolated_apt_setup "$TMP/leaked-cache" "$TMP/right.gpg" "file:$repo" main fixture
  # The setup's argv is option/value pairs. Drop only the general cache override.
  for ((i=0; i<${#isolated_apt_options[@]}; i+=2)); do
    case ${isolated_apt_options[i+1]} in
      Dir::Cache=*) ;;
      *) cache_options+=("${isolated_apt_options[i]}" "${isolated_apt_options[i+1]}") ;;
    esac
  done
  rc=0
  timeout 30 apt-get -c "$config" "${cache_options[@]}" update >"$TMP/leaked-update.log" 2>&1 || rc=$?
  expect_success "$rc" 'control: cache-leaking index still verifies' "$TMP/leaked-update.log"
  rc=0
  timeout 30 apt-cache -c "$config" "${cache_options[@]}" gencaches >"$TMP/leaked-cache.log" 2>&1 || rc=$?
  expect_success "$rc" 'control: cache-leaking generation succeeds' "$TMP/leaked-cache.log"
  if [ -s "$host/cache/pkgcache.bin" ] && [ -s "$host/cache/srcpkgcache.bin" ] &&
     [ ! -e "$TMP/leaked-cache/cache/pkgcache.bin" ]; then
    report 0 'control: removing Dir::Cache writes real host cache files'
  else
    cat "$TMP/leaked-cache.log"
    find "$TMP/leaked-cache/cache" "$host/cache" -type f
    report 1 'control: removing Dir::Cache writes real host cache files'
  fi

  isolated_apt_setup "$TMP/missing-suite" "$TMP/right.gpg" "file:$repo" main absent
  rc=0
  timeout 30 apt-get -c "$config" "${isolated_apt_options[@]}" update >"$TMP/missing-suite.log" 2>&1 || rc=$?
  if [ "$rc" != 0 ] && grep -q 'does not have a Release file' "$TMP/missing-suite.log" &&
     ! missing_key_refusal "$rc" "$TMP/missing-suite.log"; then
    report 0 'control: missing suite is not a missing-key refusal'
  else cat "$TMP/missing-suite.log"; report 1 'control: missing suite is not a missing-key refusal'; fi

}
harness_leg_run_errexit '1: isolated apt trust, cache and status' leg_1
finish

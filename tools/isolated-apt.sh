#!/usr/bin/env bash
# Sourced by the compiler staging scripts. No environment override or download:
# callers supply an absolute scratch root, source key, URL, components and suites.
# Sets isolated_apt_options for update and download; each source owns its key.
isolated_apt_setup() {
  local apt_root=$1 apt_key=$2 apt_url=$3 apt_components=$4 apt_suite
  shift 4
  case $apt_root:$apt_key in /*:/*) ;; *) return 2 ;; esac
  mkdir -p "$apt_root/lists/partial" "$apt_root/archives/partial" \
    "$apt_root/sourceparts" "$apt_root/trustedparts" "$apt_root/cache" || return
  # Installed host versions must not become candidates in this index.
  : >"$apt_root/status" || return
  : >"$apt_root/sources.list" || return
  for apt_suite in "$@"; do
    printf 'deb [signed-by=%s] %s %s %s\n' \
      "$apt_key" "$apt_url" "$apt_suite" "$apt_components" >>"$apt_root/sources.list" || return
  done
  isolated_apt_options=(
    -o "Dir::Etc::sourcelist=$apt_root/sources.list"
    -o "Dir::Etc::sourceparts=$apt_root/sourceparts"
    -o "Dir::Etc::trusted="
    -o "Dir::Etc::trustedparts=$apt_root/trustedparts"
    -o "Dir::State::lists=$apt_root/lists"
    -o "Dir::State::status=$apt_root/status"
    -o "Dir::Cache=$apt_root/cache"
    -o "Dir::Cache::archives=$apt_root/archives"
    -o APT::Get::List-Cleanup=0
  )
}

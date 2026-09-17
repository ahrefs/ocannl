#!/usr/bin/env bash

# The executable half of the pin-revisions composite action. It lives outside
# action.yml so tools/test-pin-revisions.sh can exercise the exact production
# code with hermetic opam and git fixtures.

set -euo pipefail

hash12() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum | cut -c1-12
  else
    shasum -a 256 | cut -c1-12
  fi
}

# One process for the whole batch: the per-definition listing below hashes
# ~180 files on every CI job, and a `hash12` per file would be ~180 spawns.
hash12_files() { # hash12_files FILE...
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$@" | cut -c1-12
  else
    shasum -a 256 "$@" | cut -c1-12
  fi
}

work_dir=$(mktemp -d "${TMPDIR:-/tmp}/pin-revisions.XXXXXX")
trap 'rm -rf "$work_dir"' EXIT

# setup-ocaml updates opam-repository before this action. Ask opam's solver for
# the self-contained package set selected for every project package with the
# same test/doc dependency flags as the install steps. This keys the actual
# ordinary-package versions rather than a version-specific detail of opam's
# repository storage.
opam_files=(./*.opam)
[ -e "${opam_files[0]}" ] \
  || { echo "no project opam files found for dependency solution" >&2; exit 1; }
package_names=$(printf '%s\n' "${opam_files[@]}" \
  | sed 's#^\./##; s#\.opam$##' \
  | LC_ALL=C sort -u)
request=$(printf '%s\n' "$package_names" | paste -sd, -)
solution=$(opam --cli=2.1 list --safe --color=never --resolve="$request" \
  --with-test --with-doc --columns=package --short \
  | LC_ALL=C sort -u)
[ -n "$solution" ] \
  || { echo "opam dependency solution was empty" >&2; exit 1; }
echo "Resolved opam package solution:"
printf '%s\n' "$solution" | sed 's/^/  /'
# The digest also covers the solved packages' definitions as the switch holds
# them, so a metadata-only change to a selected version (a new patch, a depext)
# still invalidates the key. The project's own packages are left out of that
# part: the caller pins them from the checkout, and opam records the checkout's
# git ref in the pinned definition -- `#master` on a branch, `#HEAD` on the
# detached checkout every pull_request run gets -- so hashing them keyed the
# cache by EVENT TYPE. Every pull request missed the switch master had just
# saved for the same tree and paid the full dependency build; the fetch that
# then died on a rolled upstream archive checksum is what surfaced it
# (gh-ocannl-889). Their content is already in the callers' keys through
# `hashFiles('*.opam')`.
definition_packages=()
project_packages=
while IFS= read -r package; do
  name=${package%%.*}
  if printf '%s\n' "$package_names" | grep -qxF -- "$name"; then
    project_packages="$project_packages $name"
  else
    definition_packages+=("$package")
  fi
done <<< "$solution"
# Without this guard an all-project solution would run `opam show` with no
# package argument, which describes the current directory instead of failing.
[ "${#definition_packages[@]}" -gt 0 ] \
  || { echo "opam dependency solution holds only project packages" >&2; exit 1; }
echo "Project packages left out of the definition digest:$project_packages"
definitions="$work_dir/definitions"
opam --cli=2.1 show --safe --color=never --raw --sort "${definition_packages[@]}" \
  >"$definitions"
solution_digest=$(
  {
    printf '%s\n' "$solution"
    cat "$definitions"
  } | hash12
)
# Which package's definition moved is the question a cache miss actually poses,
# and neither the solution listing nor the pin specs answer it: two runs can
# print byte-identical listings and still digest differently (gh-ocannl-889 was
# diagnosed by local reproduction for exactly this reason). So report a per
# package digest of that package's own definition -- diffing two runs' listings
# names the culprit directly. `opam show --raw --sort` concatenates the
# definitions it was asked for, each opening with `opam-version:` at column 0,
# so splitting that one solver-wide answer costs no extra opam call; the block's
# own `name:`/`version:` fields label it, since `--sort` returns them in
# dependency order rather than the order they were requested in.
blocks_dir="$work_dir/blocks"
mkdir -p "$blocks_dir"
awk -v dir="$blocks_dir" '
  /^opam-version:/ {
    # awk keeps every redirection target open; ~180 of them exhausts the
    # descriptor limit on the awks that do not juggle them.
    if (file != "") close(file)
    n += 1
    file = sprintf("%s/%06d", dir, n)
    name = ""
    version = ""
  }
  n > 0 { print > file }
  n > 0 && name == "" && /^name: "/ {
    name = $0; sub(/^name: "/, "", name); sub(/"$/, "", name); names[n] = name
  }
  n > 0 && version == "" && /^version: "/ {
    version = $0; sub(/^version: "/, "", version); sub(/"$/, "", version)
    versions[n] = version
  }
  END {
    for (i = 1; i <= n; i += 1)
      printf "%s.%s\n", (i in names ? names[i] : "?"), (i in versions ? versions[i] : "?")
  }
' "$definitions" >"$work_dir/labels"
# An empty split with a non-empty package list means the definitions never
# arrived: the digest above then covers the solution listing alone, which is
# exactly the silent key freeze the guards in this script exist to prevent.
[ -s "$work_dir/labels" ] \
  || { echo "no package definitions parsed from opam show output" >&2; exit 1; }
# A partial answer still changes the aggregate digest, but it makes the
# diagnostic listing silently stop naming packages. Relate the blocks parsed
# from opam's answer to the package set that asked for them.
[ "$(wc -l <"$work_dir/labels")" -eq "${#definition_packages[@]}" ] \
  || { printf 'opam show returned %d definitions for %d requested packages\n' \
    "$(wc -l <"$work_dir/labels")" "${#definition_packages[@]}" >&2; exit 1; }
hash12_files "$blocks_dir"/* >"$work_dir/hashes"
echo "Definition digests:"
paste -d' ' "$work_dir/labels" "$work_dir/hashes" | sed 's/^/  /'
echo "solution-digest=$solution_digest" >>"$GITHUB_OUTPUT"

# `--cli=2.1` fixes the table format this parser consumes across opam upgrades.
# Deliberately NOT advanced along with opam: the declared version is a FLOOR on
# the binary (`--cli=2.6` exits 2 on an opam 2.5.0), so raising it drops every
# contributor and runner below that version, while the pin's whole job is to
# freeze a parser's input against the layout changes a newer CLI opts into.
# Nothing here wants a flag introduced after 2.1, and opam's storage
# improvements are in the repository backend rather than gated by the CLI
# version, so the freeze costs no speed. opam 2.6.0 accepts 2.0 through 2.5 and
# deprecates none of them (2.6 is not a CLI version at all). Move it only to
# reach a later flag, to the lowest version carrying it, re-checking the three
# parse sites above and the fixtures in tools/test-pin-revisions.sh.
# Local project pins are reported as git+file:// URLs; omit them because the
# source checkout already selects their revision and a dependency cache must
# not miss on every project commit. Every remote git pin, whether explicit or
# introduced by pin-depends, comes from the registry populated by the caller's
# preceding `opam pin -n` steps. LC_ALL=C keeps the digest stable across runner
# locales.
specs=$(opam --cli=2.1 pin list --safe --color=never \
  | sed -n 's/.* \(git+[^[:space:]]*\).*/\1/p' \
  | sed '/^git+file:\/\//d' \
  | LC_ALL=C sort -u)
# An empty list means the registry extraction stopped matching (or the caller
# ran this too early). Never collapse that defect into the digest of an empty
# string: that would silently freeze the cache key.
[ -n "$specs" ] || { echo "no remote git pins found in opam pin registry" >&2; exit 1; }
echo "Derived remote git pin specs:"
printf '%s\n' "$specs" | sed 's/^/  /'
shas=
while IFS= read -r spec; do
  spec=${spec#git+}
  url=${spec%%#*}
  ref=${spec#*#}
  [ "$ref" != "$spec" ] || ref=HEAD
  # `awk` rather than `cut | head`: an annotated tag answers on two lines (the
  # tag and its peeled commit), and a `head` that leaves early can hand its
  # upstream a SIGPIPE, which `pipefail` would read as a failed resolution. awk
  # consumes the whole answer and prints one line.
  sha=$(git ls-remote "$url" "$ref" | awk 'NR==1{print $1}')
  # An empty sha would silently collapse the key onto whatever it hashed to
  # last time, serving a cache built from different sources.
  [ -n "$sha" ] || { echo "could not resolve $ref of $url" >&2; exit 1; }
  # Recording the exact revisions in the log is a second reason to run this at
  # all: nothing else currently reports them.
  echo "$url#$ref -> $sha"
  shas=$shas$sha
done <<< "$specs"
# macOS ships shasum, not sha256sum; hash12 handles both runners.
echo "digest=$(printf %s "$shas" | hash12)" >>"$GITHUB_OUTPUT"

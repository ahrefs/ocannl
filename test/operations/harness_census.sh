#!/usr/bin/env bash
# Read only the six tab-separated columns from affected --harness-census.
# Keep the member and path of self, loads and names rows, with a tie marker
# for multiple owners, and the glob and leg selector of declares rows.
# Source locations, naming tokens and the distinction between loads and names
# describe how affected found a reach. They can change without changing it.
# The census owns all interpretation of harness source and repository paths.
set -euo pipefail
export LC_ALL=C

root=$(cd -P -- "${DUNE_SOURCEROOT:?DUNE_SOURCEROOT is unset}" && pwd -P)
# An unpacked source tree nested in a checkout owns its own file inventory.
ceiling=$(dirname -- "$root")
case $OSTYPE in msys*) ceiling=$(cygpath -am -- "$ceiling") ;; esac
export GIT_CEILING_DIRECTORIES="$ceiling"

"$BASH" "$root/tools/test-run.sh" affected --harness-census |
  awk '
    BEGIN { FS = OFS = "\t" }
    NF == 6 && $1 != "" && $2 != "" {
      if ($3 == "declares" && $6 == "-") {
        print $1, $2, $3, $5
        next
      }
      if ($3 == "self" && $6 == "-") {
        print $1, $2
        next
      }
      if (($3 == "loads" || $3 == "names") && $6 ~ /^[1-9][0-9]*$/) {
        if ($6 > 1) print $1, $2, "tie"
        else print $1, $2
        next
      }
    }
    {
      printf "harness census: unexpected record schema at row %d\n", NR > "/dev/stderr"
      exit 2
    }
  ' | sort -u >harness_census.expected.corrected

#!/usr/bin/env bash
# Compare affected's fixture reader with the OCaml reader on the same inputs.
# The Dune rule separately checks their observations against the golden.
set -eu
export LC_ALL=C
kind=${READER_PARITY_KIND:?}
prefix=reader_parity_${kind}
"$BASH" ../../tools/affected-tests.sh --reader-dump "$kind" "$prefix"*.fixture >"$prefix.perl"
diff -u "$prefix.actual" "$prefix.perl"

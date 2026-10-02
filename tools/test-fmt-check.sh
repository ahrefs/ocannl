#!/usr/bin/env bash
# Opposing controls for fmt-check.sh: clean output, a soft odoc warning, and a
# formatter failure must remain three different outcomes, each ending in a
# verdict line. With opam, dune and ocamlformat available, a real project
# leg checks the default command after `dune fmt` already ran: the cached
# `@fmt` action must not hide the warning (gh-ocannl-1155).

set -u

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject="$script_dir/fmt-check.sh"
. "$script_dir/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fmt-check
fixture_dir=$TMP

fixture="$fixture_dir/formatter"
cat >"$fixture" <<'EOF'
#!/usr/bin/env bash
case "$1" in
  clean)
    echo "Formatting is clean"
    ;;
  warning)
    echo "Warning: Invalid documentation comment:" >&2
    echo 'File "fixture.ml", line 1, characters 0-0:' >&2
    echo "End of text is not allowed in '[...]' (code)." >&2
    ;;
  failure)
    echo "Warning: Invalid documentation comment:" >&2
    echo "Formatter diff" >&2
    exit 7
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$fixture"


check() {
  want=$1
  label=$2
  shift 2
  set +e
  output=$("$subject" "$fixture" "$@" 2>&1)
  got=$?
  set -e
  if [ "$got" -ne "$want" ]; then
    echo "FAIL: $label exited $got, expected $want" >&2
    printf '%s\n' "$output" >&2
    report 1 "$label exits $want"
  else
    report 0 "$label exits $want"
  fi
  case $want in 0) verdict="fmt-check: PASSED" ;; *) verdict="fmt-check: FAILED (exit $want)" ;; esac
  last=$(printf '%s\n' "$output" | sed -n '$p')
  case $last in
    "$verdict"*) report 0 "$label ends with its verdict" ;;
    *) report 1 "$label ends with its verdict" "last line: $last" ;;
  esac
}

check 0 "clean formatter output" clean
check 1 "invalid documentation warning" warning
check 7 "formatter failure status preservation" failure

# A real project whose only fault is an invalid doc comment in otherwise
# formatted code: `dune fmt` prints the warning once, promotes nothing, and
# leaves the file's ocamlformat action up to date.
# The fixture lives outside the repository, so name the switch the repository
# resolves: setup-ocaml's is a local one, which `opam exec` finds only from
# inside the checkout (outside it, opam exits 50).
switch=$(cd "$script_dir/.." && opam switch show 2>/dev/null) || switch=
if [ -z "$switch" ] \
   || ! OPAMSWITCH=$switch opam exec -- dune --version >/dev/null 2>&1 \
   || ! OPAMSWITCH=$switch opam exec -- ocamlformat --version >/dev/null 2>&1; then
  skip "the legs after dune fmt" "no opam switch with dune and ocamlformat"
  finish
fi
export OPAMSWITCH="$switch"
project="$fixture_dir/project"
mkdir -p "$project"
printf '(lang dune 3.20)\n' >"$project/dune-project"
printf '(library\n (name broken))\n' >"$project/dune"
# The repository's options without its version pin: this leg is about the
# cache, so a drifted local ocamlformat must not turn into its failure.
grep -v '^version' "$script_dir/../.ocamlformat" >"$project/.ocamlformat"
printf '(** Unclosed code span: [x *)\nlet x = 1\n' >"$project/broken.ml"
export DUNE_CACHE=disabled
# Its status is not the point (a promotion would make it 1); its output is.
(cd "$project" && opam exec -- dune fmt) >"$fixture_dir/dune-fmt.log" 2>&1 || true
if grep -Fq "Warning: Invalid documentation comment:" "$fixture_dir/dune-fmt.log"; then
  report 0 "fixture: dune fmt reports the invalid doc comment"
else
  report 1 "fixture: dune fmt reports the invalid doc comment" "see $fixture_dir/dune-fmt.log"
fi
# Negative control: an unforced @fmt replays nothing and passes the tree, the
# trap the default command has to defeat. If dune ever replays the warning,
# this leg says the next one no longer tests anything.
got=0
(cd "$project" && "$subject" opam exec -- dune build @fmt) >"$fixture_dir/unforced.log" 2>&1 || got=$?
if [ "$got" -eq 0 ]; then
  report 0 "negative control: unforced @fmt after dune fmt passes"
else
  report 1 "negative control: unforced @fmt after dune fmt passes" "exited $got; see $fixture_dir/unforced.log"
fi
got=0
(cd "$project" && "$subject") >"$fixture_dir/default.log" 2>&1 || got=$?
if [ "$got" -eq 1 ] && grep -Fq "Warning: Invalid documentation comment:" "$fixture_dir/default.log"; then
  report 0 "default command after dune fmt rejects the invalid doc comment"
else
  report 1 "default command after dune fmt rejects the invalid doc comment" "exited $got; see $fixture_dir/default.log"
fi

finish

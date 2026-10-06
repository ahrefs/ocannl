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
  credentials)
    # gh-ocannl-1280: the formatter command runs without the caller's credentials, and with
    # its plain variables.
    [ -z "${GH_TOKEN+x}${FOO_API_KEY+x}" ] || { echo "credentials reached the formatter" >&2; exit 3; }
    [ "${FMT_CHECK_PLAIN:-}" = kept ] || { echo "the plain variable did not reach it" >&2; exit 4; }
    echo "Formatting is clean"
    ;;
  *)
    exit 2
    ;;
esac
EOF
chmod +x "$fixture"


check() {
  local want=$1 label=$2 output got=0 verdict last
  shift 2
  # Capture the subject status without changing the caller's errexit state.
  output=$("$subject" "$fixture" "$@" 2>&1) || got=$?
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

for errexit in off on; do
  for outcome in clean warning failure; do
    case $errexit in off) set +e ;; on) set -e ;; esac
    case $outcome in clean) want=0 ;; warning) want=1 ;; failure) want=7 ;; esac
    before=$-
    check "$want" "$outcome formatter output (errexit $errexit)" "$outcome"
    if [ "$-" = "$before" ]; then
      report 0 "$outcome preserves errexit $errexit"
    else
      report 1 "$outcome preserves errexit $errexit"
    fi
  done
done
set +e

# Credentials never reach the formatter command (gh-ocannl-1280): a fake GH_TOKEN and FOO_API_KEY
# are exported beside a plain variable. The negative control is a copy with the scrub cut out,
# beside the helper it still sources, which must hand the command both.
export GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key FMT_CHECK_PLAIN=kept
check 0 "a formatter run with credentials exported, which must not see them," credentials
cp "$script_dir/credential-env.sh" "$fixture_dir/credential-env.sh"
awk '/^eval "\$\(credential_env_scrub_text\)" \|\| \{$/ { skip = 1 } skip { if ($0 == "}") skip = 0; next } { print }' \
  "$subject" >"$fixture_dir/fmt-check.sh"
chmod +x "$fixture_dir/fmt-check.sh"
got=0
if cmp -s "$subject" "$fixture_dir/fmt-check.sh"; then
  report 1 "negative control: fmt-check without the scrub hands the formatter both credentials" \
    "the scrub could not be cut out of $subject"
else
  "$fixture_dir/fmt-check.sh" "$fixture" credentials >"$fixture_dir/no-scrub.log" 2>&1 || got=$?
  if [ "$got" -eq 3 ]; then
    report 0 "negative control: fmt-check without the scrub hands the formatter both credentials"
  else
    report 1 "negative control: fmt-check without the scrub hands the formatter both credentials" \
      "exited $got; see $fixture_dir/no-scrub.log"
  fi
fi
unset GH_TOKEN FOO_API_KEY FMT_CHECK_PLAIN

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

#!/usr/bin/env bash
# Opposing controls for fmt-check.sh: clean output, a soft odoc warning, and a
# formatter failure must remain three different outcomes, each ending in a
# verdict line. With opam, dune and ocamlformat available, a real project
# leg checks the default command after `dune fmt` already ran: the cached
# `@fmt` action must not hide the warning (gh-ocannl-1155). A fake
# fleet-worker.sh drives the default command's run-time slot (gh-ocannl-1519),
# and an option is refused with the usage rather than run as the command.

set -u

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject="$script_dir/fmt-check.sh"
. "$script_dir/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch fmt-check
fixture_dir=$TMP
# No leg reaches this host's fleet: the default command's slot (gh-ocannl-1519) is driven by
# the fake fleet-worker.sh below, and every other leg runs with the fleet turned off.
export OCANNL_TOOL_FLEET_WORKER=none

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
cp "$script_dir/fleet-worker-candidates.sh" "$fixture_dir/fleet-worker-candidates.sh"
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
# The default command scrubs again inside `opam exec`, whose switch updates could set one: a fake
# opam that sets both before running its command, and a fake dune that refuses them. The negative
# control is a copy whose default command is the bare `opam exec -- dune ...` it replaced.
fake_switch="$fixture_dir/fake-switch"
mkdir -p "$fake_switch"
cat >"$fake_switch/opam" <<'EOF'
#!/usr/bin/env bash
[ "${1:-}" = exec ] && [ "${2:-}" = -- ] || exit 9
shift 2
export GH_TOKEN=fixture-switch-token FOO_API_KEY=fixture-switch-key
exec "$@"
EOF
# The default command's two dune builds, the formatter and the repository scans (gh-ocannl-1321),
# each recorded in FAKE_DUNE_LOG (with `@slot` when it ran inside the fake fleet slot below);
# FAKE_FMT_EXIT and FAKE_SCANS_EXIT make either one fail, and FAKE_DUNE_KILL_CHECK kills the
# fmt-check that started it, which is dune's parent.
cat >"$fake_switch/dune" <<'EOF'
#!/usr/bin/env bash
case "$*" in
  "build @fmt --force") step=fmt status=${FAKE_FMT_EXIT:-0} ;;
  "build @test/operations/scans") step=scans status=${FAKE_SCANS_EXIT:-0} ;;
  *) echo "unexpected dune argv: $*" >&2; exit 5 ;;
esac
[ -z "${GH_TOKEN+x}${FOO_API_KEY+x}" ] || { echo "credentials reached dune" >&2; exit 3; }
[ "${FMT_CHECK_PLAIN:-}" = kept ] || { echo "the plain variable did not reach dune" >&2; exit 4; }
[ -z "${FAKE_DUNE_LOG:-}" ] || echo "$step${FAKE_SLOT_HELD:+@slot}" >>"$FAKE_DUNE_LOG"
[ -z "${FAKE_DUNE_KILL_CHECK:-}" ] || kill -KILL "$PPID"
exit "$status"
EOF
chmod +x "$fake_switch/opam" "$fake_switch/dune"
awk '/^  scrubbed_command=\(opam exec -- sh -c / { print "  scrubbed_command=(opam exec -- \"$@\")"; getline; next } { print }' \
  "$subject" >"$fixture_dir/fmt-check-bare.sh"
chmod +x "$fixture_dir/fmt-check-bare.sh"
got=0
OCANNL_TOOL_FMT_CHECK_SCANS=1 FMT_CHECK_PLAIN=kept PATH="$fake_switch:$PATH" "$subject" \
  >"$fixture_dir/switch.log" 2>&1 || got=$?
if [ "$got" -eq 0 ]; then
  report 0 "the default command's dune runs without credentials the opam switch sets"
else
  report 1 "the default command's dune runs without credentials the opam switch sets" \
    "exited $got; see $fixture_dir/switch.log"
fi
got=0
if cmp -s "$subject" "$fixture_dir/fmt-check-bare.sh"; then
  report 1 "negative control: a bare opam exec hands dune the switch's credentials" \
    "the default command could not be replaced in $subject"
else
  OCANNL_TOOL_FMT_CHECK_SCANS=1 FMT_CHECK_PLAIN=kept PATH="$fake_switch:$PATH" \
    "$fixture_dir/fmt-check-bare.sh" >"$fixture_dir/switch-bare.log" 2>&1 || got=$?
  if [ "$got" -eq 3 ]; then
    report 0 "negative control: a bare opam exec hands dune the switch's credentials"
  else
    report 1 "negative control: a bare opam exec hands dune the switch's credentials" \
      "exited $got; see $fixture_dir/switch-bare.log"
  fi
fi

# The repository scans are the default command's second step (gh-ocannl-1321): built after a clean
# formatter run, their failure is the verdict, OCANNL_TOOL_FMT_CHECK_SCANS=0 skips them and says
# so, any other value is refused, a formatter failure stops before them, and a command given as
# arguments runs alone. FAKE_DUNE_LOG records which builds ran, in order.
scans_leg() { # LABEL WANT_RC WANT_STEPS VERDICT_PREFIX [env ARG...] -- runs the default command
  local label=$1 want=$2 steps=$3 prefix=$4 got=0 ran last
  shift 4
  : >"$fixture_dir/dune-steps.log"
  # The caller's words first: `env` reads its `-u` options only before the first assignment.
  env "$@" FMT_CHECK_PLAIN=kept PATH="$fake_switch:$PATH" \
    FAKE_DUNE_LOG="$fixture_dir/dune-steps.log" "$subject" >"$fixture_dir/scans-leg.log" 2>&1 \
    || got=$?
  ran=$(tr '\n' ' ' <"$fixture_dir/dune-steps.log")
  last=$(sed -n '$p' "$fixture_dir/scans-leg.log")
  if [ "$got" -eq "$want" ] && [ "$ran" = "$steps" ]; then
    case $last in
      "$prefix"*) report 0 "$label" ;;
      *) report 1 "$label" "last line: $last" ;;
    esac
  else
    report 1 "$label" \
      "exited $got (want $want), dune ran [$ran] (want [$steps]); see $fixture_dir/scans-leg.log"
  fi
}
scans_leg "the default command builds the scans after the formatter, and passes" \
  0 "fmt scans " "fmt-check: PASSED" OCANNL_TOOL_FMT_CHECK_SCANS=1
if grep -Eq '^fmt-check: the repository scans took [0-9]+ s$' "$fixture_dir/scans-leg.log"; then
  report 0 "the scans step reports its cost"
else
  report 1 "the scans step reports its cost" "see $fixture_dir/scans-leg.log"
fi
scans_leg "unset, the scans setting defaults to building them" \
  0 "fmt scans " "fmt-check: PASSED" -u OCANNL_TOOL_FMT_CHECK_SCANS
scans_leg "a scan failure is the verdict" \
  1 "fmt scans " "fmt-check: FAILED (exit 1): the repository scans failed" \
  OCANNL_TOOL_FMT_CHECK_SCANS=1 FAKE_SCANS_EXIT=1
scans_leg "OCANNL_TOOL_FMT_CHECK_SCANS=0 skips the scans, and says so" \
  0 "fmt " "fmt-check: PASSED (repository scans skipped" \
  OCANNL_TOOL_FMT_CHECK_SCANS=0 FAKE_SCANS_EXIT=1
scans_leg "any other scans setting is refused before dune runs" \
  2 "" "fmt-check: FAILED (exit 2): OCANNL_TOOL_FMT_CHECK_SCANS is 0" OCANNL_TOOL_FMT_CHECK_SCANS=yes
scans_leg "a formatter failure stops before the scans" \
  7 "fmt " "fmt-check: FAILED (exit 7): the formatter command failed" \
  OCANNL_TOOL_FMT_CHECK_SCANS=1 FAKE_FMT_EXIT=7
got=0
: >"$fixture_dir/dune-steps.log"
env OCANNL_TOOL_FMT_CHECK_SCANS=1 FMT_CHECK_PLAIN=kept PATH="$fake_switch:$PATH" \
  FAKE_DUNE_LOG="$fixture_dir/dune-steps.log" "$subject" dune build @fmt --force \
  >"$fixture_dir/explicit.log" 2>&1 || got=$?
if [ "$got" -eq 0 ] && [ "$(tr '\n' ' ' <"$fixture_dir/dune-steps.log")" = "fmt " ]; then
  report 0 "a command given as arguments runs alone, without the scans"
else
  report 1 "a command given as arguments runs alone, without the scans" \
    "exited $got; see $fixture_dir/explicit.log"
fi

# The fleet's run-time correctness slot (gh-ocannl-1519): on a box whose fleet-worker.sh answers
# the probe, the default command runs both builds inside one `execution slot --cpu`, as
# tools/test-run.sh runs its batches; a refused slot is SLOT REFUSED (exit 75) with nothing run, a
# check the slot ran keeps its own verdict, and a command given as arguments, or a fleet-worker.sh
# that does not answer the probe, takes none. The fake records its calls in FAKE_FW_CALLS and, in
# the slot, marks the command's environment (FAKE_SLOT_HELD) for the fake dune to record.
fake_fw="$fixture_dir/fake-fleet-worker.sh"
cat >"$fake_fw" <<'EOF'
#!/usr/bin/env bash
printf '%s\n' "$*" >>"$FAKE_FW_CALLS"
if [ "$1 $2 $3" = "execution slot --probe" ]; then
  [ "${FAKE_FW_MODE:-}" = noprobe ] && { echo "fleet-worker.sh: execution slot [--wait <seconds>] ..." >&2; exit 2; }
  echo "EXECUTION SLOT PROBE fakebox 4 2"
  exit 0
fi
shift 2
while [ "$1" != -- ]; do shift; done
shift
case ${FAKE_FW_MODE:-} in
  refuse) echo "EXECUTION SLOT REFUSED fakebox: all 4 slots busy after 0s"; exit 1 ;;
  unreachable) echo "EXECUTION SLOT UNREACHABLE anchor: registry unread, no slot taken"; exit 4 ;;
esac
echo "EXECUTION SLOT fakebox: slot 1 of 4 held for: $*" >&2
FAKE_SLOT_HELD=1 exec "$@"
EOF
chmod +x "$fake_fw"
slot_leg() { # LABEL WANT_RC WANT_STEPS WANT_SLOT VERDICT_PREFIX [env ARG...] [-- SUBJECT_ARG...]
  # WANT_SLOT: the slot call's words before `--`, `-` for a probe and no slot call, or `none`
  # for no fleet-worker.sh call at all.
  local label=$1 want=$2 steps=$3 want_slot=$4 prefix=$5 got=0 ran last calls slot_call env_args=()
  shift 5
  while [ "$#" -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
  [ "$#" -eq 0 ] || shift
  : >"$fixture_dir/dune-steps.log"
  : >"$fixture_dir/fw-calls.log"
  env OCANNL_TOOL_FMT_CHECK_SCANS=1 ${env_args[@]+"${env_args[@]}"} FMT_CHECK_PLAIN=kept PATH="$fake_switch:$PATH" \
    OCANNL_TOOL_FLEET_WORKER="$fake_fw" FAKE_FW_CALLS="$fixture_dir/fw-calls.log" \
    FAKE_DUNE_LOG="$fixture_dir/dune-steps.log" "$subject" "$@" >"$fixture_dir/slot-leg.log" 2>&1 \
    || got=$?
  ran=$(tr '\n' ' ' <"$fixture_dir/dune-steps.log")
  last=$(sed -n '$p' "$fixture_dir/slot-leg.log")
  calls=$(tr '\n' '|' <"$fixture_dir/fw-calls.log")
  slot_call=$(grep -v -- '--probe' "$fixture_dir/fw-calls.log" | sed 's/ -- .*//')
  case $want_slot in
    none) [ -z "$calls" ] || { report 1 "$label" "fleet-worker.sh was called: $calls"; return; } ;;
    -) [ "$calls" = "execution slot --probe|" ] ||
         { report 1 "$label" "want only the probe, got: $calls"; return; } ;;
    *) [ "$slot_call" = "$want_slot" ] ||
         { report 1 "$label" "slot call [$slot_call], want [$want_slot]; calls: $calls"; return; } ;;
  esac
  if [ "$got" -eq "$want" ] && [ "$ran" = "$steps" ]; then
    case $last in
      "$prefix"*) report 0 "$label" ;;
      *) report 1 "$label" "last line: $last" ;;
    esac
  else
    report 1 "$label" \
      "exited $got (want $want), dune ran [$ran] (want [$steps]); see $fixture_dir/slot-leg.log"
  fi
}
slot_leg "on a fleet box the default command runs both builds in one --cpu slot, and passes" \
  0 "fmt@slot scans@slot " "execution slot --wait 600 --cpu" "fmt-check: PASSED"
if grep -Fq "fmt-check: taking one of fakebox's 4 fleet correctness slots" "$fixture_dir/slot-leg.log"; then
  report 0 "the slot is announced before it is waited for"
else
  report 1 "the slot is announced before it is waited for" "see $fixture_dir/slot-leg.log"
fi
slot_leg "OCANNL_TOOL_SLOT_WAIT bounds the slot's wait" \
  0 "fmt@slot scans@slot " "execution slot --wait 7 --cpu" "fmt-check: PASSED" OCANNL_TOOL_SLOT_WAIT=7
slot_leg "with the scans skipped, the formatter build still runs in the slot" \
  0 "fmt@slot " "execution slot --wait 600 --cpu" "fmt-check: PASSED (repository scans skipped" \
  OCANNL_TOOL_FMT_CHECK_SCANS=0
slot_leg "a scan failure in the slot keeps its own verdict and status" \
  1 "fmt@slot scans@slot " "execution slot --wait 600 --cpu" \
  "fmt-check: FAILED (exit 1): the repository scans failed" FAKE_SCANS_EXIT=1
slot_leg "a refused slot is SLOT REFUSED (exit 75), and nothing ran" \
  75 "" "execution slot --wait 600 --cpu" "fmt-check: FAILED (exit 75): SLOT REFUSED" FAKE_FW_MODE=refuse
slot_leg "an unreachable registry is SLOT REFUSED (exit 75), and nothing ran" \
  75 "" "execution slot --wait 600 --cpu" "fmt-check: FAILED (exit 75): SLOT REFUSED" \
  FAKE_FW_MODE=unreachable
slot_leg "a check killed in the slot still ends in a verdict" \
  137 "fmt@slot " "execution slot --wait 600 --cpu" \
  "fmt-check: FAILED (exit 137): the check under the fleet slot ended without a verdict" \
  FAKE_DUNE_KILL_CHECK=1
slot_leg "a fleet-worker.sh that does not answer the probe runs the builds directly" \
  0 "fmt scans " - "fmt-check: PASSED" FAKE_FW_MODE=noprobe
slot_leg "a command given as arguments takes no slot" \
  0 "fmt " none "fmt-check: PASSED" -- dune build @fmt --force

# The script takes no options: `-j 4` used to run `-j` as the formatter command (exit 127).
slot_leg "an option is refused with the usage, before any probe or build" \
  2 "" none "fmt-check: FAILED (exit 2): '-j' is not an option" -- -j 4
if grep -Fq "usage: tools/fmt-check.sh" "$fixture_dir/slot-leg.log"; then
  report 0 "the refusal prints the usage"
else
  report 1 "the refusal prints the usage" "see $fixture_dir/slot-leg.log"
fi

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

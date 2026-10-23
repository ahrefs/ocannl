#!/usr/bin/env bash
# Controls for tools/typecheck-scan-cases.sh, the type check of a scan's case snippets
# (docs/agent-notes/build-and-test.md, Repository scans).
#
#   tools/test-typecheck-scan-cases.sh          # run every leg
#   tools/test-typecheck-scan-cases.sh --keep   # keep the scratch directory
#
# Leg 1 runs the subject over the repository's own stubbed tables, which must stay clean: a new arm
# no compiler accepts turns it red until the arm is fixed or listed as a known rejection. The other
# legs run it over fixture tables planted here: an ill-typed arm and a duplicate top-level module (it
# parses; the type checker rejects it) are REJECTED by name while their well-typed neighbours, an arm
# rebinding a stub name over the open and both case-table shapes are not; project syntax is REFUSED
# by name; the companion stub is found beside its table; a known rejection holds only while the
# compiler still rejects that arm with its listed text; a run that checked nothing, a stub that
# does not compile or lists a malformed or repeated rejection, and a usage error each exit 2; the
# engine is found under a moved DUNE_BUILD_DIR; and no arm sees a neighbouring arm's compiled unit.
#
# It needs dune and ocamlc (the subject builds its engine, which links ppxlib, and compiles each
# snippet); without them every leg is a counted skip.

set -u

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject="$script_dir/typecheck-scan-cases.sh"
. "$script_dir/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch typecheck-scan-cases
# A CI step does not carry setup-ocaml's switch on its PATH: load it as the other toolchain
# harnesses do, leaving a host with no opam to the counted skips below.
command -v dune >/dev/null 2>&1 || . "$script_dir/opam-env.sh" 2>/dev/null
harness_require dune ocamlc

out="$TMP/out" err="$TMP/err"
run() { # run the subject; its status in $rc, stdout in $out, stderr in $err
  rc=0
  "$subject" "$@" >"$out" 2>"$err" || rc=$?
}
expect_rc() { # LABEL WANT
  if [ "$rc" = "$2" ]; then report 0 "$1 exits $2"
  else report 1 "$1 exits $2" "exited $rc; stdout: $(tr '\n' '|' <"$out") stderr: $(tr '\n' '|' <"$err")"
  fi
}
has() { # LABEL FIXED-TEXT: stdout carries a line containing the text
  if grep -qF -- "$2" "$out"; then report 0 "$1"
  else report 1 "$1" "no line carries '$2' in: $(head -n 12 "$out" | tr '\n' '|')"
  fi
}
lacks() { # LABEL FIXED-TEXT
  if grep -qF -- "$2" "$out"; then report 1 "$1" "a line carries '$2' in: $(head -n 12 "$out" | tr '\n' '|')"
  else report 0 "$1"
  fi
}

mkdir -p "$TMP/t/scan_case_stubs"
stub=$TMP/t/scan_case_stubs/planted_scan_cases.mli
cat >"$stub" <<'EOF'
val stand_in : int
module Stand : sig val v : int end
module Other : sig val v : int end
EOF
table=$TMP/t/planted_scan_cases.ml
cat >"$table" <<'EOF'
let cases =
  [
    ("a well-typed arm", {ocaml|let x = 1 + stand_in|ocaml}, 0);
    ("an ill-typed arm", {ocaml|let x = 1 + "one"|ocaml}, 0);
    ( "a duplicate module parses but does not type",
      {ocaml|module M = struct end
module M = struct end|ocaml},
      0 );
    ( "an arm rebinding a stub name",
      {ocaml|module Stand = Other
let y = Stand.v|ocaml},
      0 );
    ("a project-syntax arm", {ocaml|let%op f x = x|ocaml}, 0);
  ]

let () = case ~argument:"a label's string" "an application-shaped arm" {ocaml|let z = stand_in|ocaml}
let unnamed = [ {ocaml|let w = 2|ocaml} ]
let plain = ("an ordinary literal is not a snippet", "let x = 1 + \"one\"")
EOF

leg1() {
  run --stubbed
  expect_rc "the repository's stubbed tables" 0
  local tables
  tables=$(grep -c '^typecheck-scan-cases: .* 0 rejected, .* 0 stale listing' "$out")
  if [ "$tables" -ge 2 ]; then report 0 "at least two tables were checked clean ($tables)"
  else report 1 "at least two tables were checked clean" "$tables in: $(head -n 12 "$out" | tr '\n' '|')"
  fi
}
harness_leg_run '1: the repository tables with companion stubs type-check clean' leg1

leg2() {
  run --verbose "$table"
  expect_rc "a table with a planted ill-typed arm" 1
  has "the ill-typed arm is rejected by name" "REJECTED  $table:4: an ill-typed arm"
  has "at its own line and column of the table" "File \"$table\", line 4, characters 44-49"
  has "the duplicate module is rejected by name" \
    "REJECTED  $table:6: a duplicate module parses but does not type"
  has "the well-typed arm, over the stub's name, is accepted" "ok        $table:3: a well-typed arm"
  has "a rebinding of a stub name is legal over the open" "ok        $table:10: an arm rebinding a stub name"
  has "project syntax is refused by name" "REFUSED   $table:13: a project-syntax arm"
  has "an application's arm is named by its unlabelled string" \
    "ok        $table:16: an application-shaped arm"
  has "a snippet outside a case is named by its line" "ok        $table:17: (unnamed snippet)"
  lacks "an ordinary literal is not read" "an ordinary literal is not a snippet"
  has "the summary counts every decision" \
    "6 snippet(s) checked: 4 accepted, 2 rejected, 0 known rejection(s); 1 refused; 0 stale"
}
harness_leg_run '2: a planted ill-typed arm is rejected by name, its neighbours are not' leg2

leg3() {
  run --no-stub "$table"
  expect_rc "the planted table without its stub" 1
  has "the stand-in name is then unbound" "REJECTED  $table:3: a well-typed arm"
}
harness_leg_run '3: --no-stub drops the companion stub the default finds' leg3

leg4() {
  local known="$TMP/t/known.mli"
  { cat "$stub"
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "but an expression was expected of type")]'
    printf '%s\n' '[@@@known_rejection ("a duplicate module parses but does not type", "Multiple definition")]'
  } >"$known"
  run --stub "$known" "$table"
  expect_rc "both rejections listed with their text" 0
  has "a listed rejection is still printed" "REJECTED (known: the stub lists it)  $table:4: an ill-typed arm"

  { cat "$stub"
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "Unbound module")]'
    printf '%s\n' '[@@@known_rejection ("a duplicate module parses but does not type", "Multiple definition")]'
  } >"$known"
  run --stub "$known" "$table"
  expect_rc "a listed arm rejected for another reason" 1
  has "is reported as a rejection" "not with the stub's listed \"Unbound module\""

  { cat "$stub"
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "but an expression was expected of type")]'
    printf '%s\n' '[@@@known_rejection ("a duplicate module parses but does not type", "Multiple definition")]'
    printf '%s\n' '[@@@known_rejection ("a well-typed arm", "Error")]'
  } >"$known"
  run --stub "$known" "$table"
  expect_rc "a listed arm the compiler accepts" 1
  has "is a stale listing" "STALE     $table:3: a well-typed arm"

  { cat "$stub"
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "but an expression was expected of type")]'
    printf '%s\n' '[@@@known_rejection ("a duplicate module parses but does not type", "Multiple definition")]'
    printf '%s\n' '[@@@known_rejection ("no such arm", "Error")]'
  } >"$known"
  run --stub "$known" "$table"
  expect_rc "a listing naming no arm" 1
  has "is a stale listing too" "STALE     the stub lists known_rejection \"no such arm\""
}
harness_leg_run '4: a known rejection holds only while the compiler rejects it with its text' leg4

leg5() {
  local empty="$TMP/t/empty_cases.ml" refused="$TMP/t/refused_cases.ml"
  printf '%s\n' 'let cases = [ ("a plain string", "let x = 1") ]' >"$empty"
  run --no-stub "$empty"
  expect_rc "a table with no snippet" 2
  printf '%s\n' 'let cases = [ ("only project syntax", {ocaml|let%op f x = x|ocaml}) ]' >"$refused"
  run --no-stub "$refused"
  expect_rc "a table whose every snippet is refused" 2
}
harness_leg_run '5: a table in which nothing was checked exits 2' leg5

leg6() {
  local broken="$TMP/t/broken.mli" malformed="$TMP/t/malformed.mli"
  printf '%s\n' 'val stand_in : no_such_type' >"$broken"
  run --stub "$broken" "$table"
  expect_rc "a stub that does not compile" 2
  { cat "$stub"; printf '%s\n' '[@@@known_rejection "an ill-typed arm"]'; } >"$malformed"
  run --stub "$malformed" "$table"
  expect_rc "a known rejection without its error text" 2
  { cat "$stub"; printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "")]'; } >"$malformed"
  run --stub "$malformed" "$table"
  expect_rc "a known rejection whose error text is empty, which every rejection contains" 2
  { cat "$stub"
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "but an expression was expected of type")]'
    printf '%s\n' '[@@@known_rejection ("an ill-typed arm", "Unbound module")]'
  } >"$malformed"
  run --stub "$malformed" "$table"
  expect_rc "a case listed twice, whose second text would go unread" 2
}
harness_leg_run '6: a stub that does not compile or lists malformed rejections exits 2' leg6

leg8() {
  local moved="$TMP/moved-build"
  rc=0
  DUNE_BUILD_DIR=$moved "$subject" --no-stub "$TMP/t/planted_scan_cases.ml" >"$out" 2>"$err" || rc=$?
  expect_rc "a run with DUNE_BUILD_DIR moved" 1
  has "still runs the engine dune built there" "REJECTED  $table:4: an ill-typed arm"
}
leg9() {
  local isolated="$TMP/t/isolated_cases.ml"
  cat >"$isolated" <<'CASES'
let cases =
  [
    ("a first arm", {ocaml|let v = 1|ocaml});
    ("a later arm naming the first one's unit", {ocaml|let w = Arm_1.v|ocaml});
  ]
CASES
  run --no-stub "$isolated"
  expect_rc "an arm naming a neighbouring arm's unit" 1
  has "is rejected: no arm sees another's compiled unit" \
    "REJECTED  $isolated:4: a later arm naming the first one's unit"
}
leg7() {
  run
  expect_rc "no table" 2
  run --frobnicate "$table"
  expect_rc "an unknown option" 2
  run --stubbed "$table"
  expect_rc "--stubbed with a table" 2
}
harness_leg_run '7: usage errors exit 2' leg7
harness_leg_run '8: the engine is run from the build directory dune was told to use' leg8
harness_leg_run '9: each arm is compiled apart from the others' leg9

finish

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
# rebinding a stub name over the open and both case-table shapes are not; project syntax with no
# owning module declared is REFUSED by name; the companion stub is found beside its table; a known
# rejection holds only while the compiler still rejects that arm with its listed text; a run that
# checked nothing, a stub that does not compile or lists a malformed or repeated rejection or
# owning module, and a usage error each exit 2; the engine is found under a moved DUNE_BUILD_DIR; no
# arm sees a neighbouring arm's compiled unit; and an `~interface:` snippet is compiled as a
# signature.
#
# The legs tagged #project check project-syntax arms (`let%op`, extensions) in an owning module's
# context, which builds that module of this repository and so needs the project's dependencies:
# leg 10 checks every arm of the repository tables that leg 1 refuses (run without the context), and
# leg 11 plants a negative control -- an `let%op` arm no compiler builds, the spelling a scan table
# carried until gh-ocannl-1587, and an extension no preprocessor expands are REJECTED beside an
# accepted arm, which only the stanza's ppx and the declared prelude make well-typed. Where the
# project's libraries are not installed (dune reports `Library "..." not found`), both are counted
# skips; CI's toolchain job expects them (tools/harness-skips-ci-toolchain.txt), and the build
# matrix runs them.
#
# It needs dune and ocamlc (the subject builds its engine, which links ppxlib, and compiles each
# snippet); without them every leg is a counted skip.
#
# What `tools/test-run.sh affected` lists for a change (gh-ocannl-1900; grammar in
# tools/affected-tests.sh, The harness legs): legs 1 and 10 read the repository's stubbed tables and
# their stubs, which `--stubbed` finds and no code here names (a table with no stub is matched too,
# and costs those two legs); legs 10 and 11, the #project ones, expand their `let%op` arms through
# the built ppx; every leg runs the engine.
# affected-reads: test/operations/scan_case_stubs/* --only '^(1|10):'
# affected-reads: test/operations/*_scan_cases.ml test/operations/config_scan_lexing.ml --only '^(1|10):'
# affected-reads: tensor/ppx_*.ml --only '^(10|11):'
# affected-reads: tools/scan_case_snippets.ml

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
  run --stubbed --no-project-context
  expect_rc "the repository's stubbed tables, project syntax refused" 0
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
  has "for want of an owning module" "the stub declares no [@@@owning_module] to check it in"
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

# The project-syntax fixtures: a table whose arms need lib/nn_blocks.ml's stanza (its ppx, its
# libraries) and the DSL open its stub's prelude carries, and the stub naming that module.
mkdir -p "$TMP/p/scan_case_stubs"
pstub=$TMP/p/scan_case_stubs/project_scan_cases.mli
cat >"$pstub" <<'EOF'
val use_padding : bool
[@@@owning_module "lib/nn_blocks.ml", "open! Base\nopen Ocannl_tensor.Operation.DSL_modules"]
EOF
ptable=$TMP/p/project_scan_cases.ml
cat >"$ptable" <<'EOF'
let cases =
  [
    ("a well-typed op arm", {ocaml|let%op f ?(stride = 1) () x = x ++ "stride*o+k => o"|ocaml});
    ("an op arm without its unit parameter", {ocaml|let%op f stride x = x ++ "stride*o+k => o"|ocaml});
    ("an extension no preprocessor expands", {ocaml|[%%no_such_extension let x = 1]|ocaml});
    ("a plain arm beside them", {ocaml|let y = not use_padding|ocaml});
  ]
EOF
# Whether the run just made failed only for want of the project's installed libraries: the one
# reason a #project leg skips rather than fails.
project_absent() {
  [ "$rc" = 2 ] && grep -qE 'Library "[^"]+" not found' "$err"
}
project_absent_reason() {
  printf "the project's libraries are not installed (dune: %s)" \
    "$(grep -m1 -oE 'Library "[^"]+" not found' "$err")"
}

leg10() {
  run --stubbed --no-project-context
  local refused checked tables clean
  refused=$(grep -c '^REFUSED ' "$out")
  run --stubbed
  if project_absent; then
    skip "the repository tables' project-syntax arms, checked in their owning modules' contexts" \
      "$(project_absent_reason)"
    return
  fi
  expect_rc "the repository's stubbed tables, project syntax in context" 0
  lacks "no arm is refused in context" "REFUSED "
  checked=$(sed -n 's/^CONTEXT   \([0-9][0-9]*\) project-syntax .*/\1/p' "$out" | awk '{ n += $1 } END { print n + 0 }')
  if [ "$refused" -gt 0 ] && [ "$checked" = "$refused" ]; then
    report 0 "every arm refused without the context is checked in it ($refused)"
  else
    report 1 "every arm refused without the context is checked in it" \
      "$refused refused, $checked checked in context: $(grep '^CONTEXT' "$out" | tr '\n' '|')"
  fi
  tables=$(grep -c '^== ' "$out")
  clean=$(grep -c '^typecheck-scan-cases: .* 0 rejected, .*; 0 refused; 0 stale listing' "$out")
  if [ "$tables" -ge 2 ] && [ "$clean" = "$tables" ]; then report 0 "every table is clean ($tables)"
  else report 1 "every table is clean" "$clean of $tables in: $(grep '^typecheck-scan-cases' "$out" | tr '\n' '|')"
  fi
}
harness_leg_run '10: the repository tables project-syntax arms type-check in their owning contexts #project' leg10

leg11() {
  run --verbose "$ptable"
  if project_absent; then
    skip "planted project-syntax arms, checked in an owning module's context" "$(project_absent_reason)"
    return
  fi
  expect_rc "a table with project-syntax arms no compiler builds" 1
  has "the op arm the stanza's ppx makes well-typed is accepted" "ok        $ptable:3: a well-typed op arm"
  has "an op arm without its unit parameter is rejected by name" \
    "REJECTED  $ptable:4: an op arm without its unit parameter"
  has "with the compiler's reading of the expansion" "The value stride has type Tensor.t"
  has "at its own line of the table, through the ppx" "File \"$ptable\", line 4"
  has "an extension no preprocessor expands is rejected" \
    "REJECTED  $ptable:5: an extension no preprocessor expands"
  has "as the compiler leaves it" "Uninterpreted extension 'no_such_extension'"
  has "a plain arm beside them is still checked" "ok        $ptable:6: a plain arm beside them"
  has "the context is named" "CONTEXT   3 project-syntax snippet(s) checked in the context of lib/nn_blocks.ml"
  has "the summary counts every decision" \
    "4 snippet(s) checked: 2 accepted, 2 rejected, 0 known rejection(s); 0 refused; 0 stale"

  local stub="$TMP/p/variant.mli"
  { cat "$pstub"
    printf '%s\n' '[@@@known_rejection ("an op arm without its unit parameter", "has type Tensor.t")]'
    printf '%s\n' '[@@@known_rejection ("an extension no preprocessor expands", "Uninterpreted extension")]'
  } >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "both project-syntax rejections listed with their text" 0
  has "a listed project-syntax rejection is still printed" \
    "REJECTED (known: the stub lists it)  $ptable:4: an op arm without its unit parameter"

  printf '%s\n' 'val use_padding : bool' '[@@@owning_module "lib/nn_blocks.ml"]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "the owning module without the prelude's DSL open" 1
  has "rejects the arm only the prelude makes well-typed" "REJECTED  $ptable:3: a well-typed op arm"

  printf '%s\n' 'val use_padding : bool' \
    '[@@@owning_module "lib/nn_blocks.ml", "open No_such_module_anywhere"]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "a prelude that does not compile" 2
  if grep -qF "prelude of $stub does not compile" "$err"; then report 0 "is named as the prelude"
  else report 1 "is named as the prelude" "stderr: $(head -n 6 "$err" | tr '\n' '|')"
  fi
}
harness_leg_run '11: a project-syntax arm no compiler builds is rejected in its owning context #project' leg11

leg12() {
  run --no-project-context "$ptable"
  expect_rc "--no-project-context over a table with project-syntax arms" 0
  has "refuses them by name" "REFUSED   $ptable:3: a well-typed op arm"
  has "saying why" "--no-project-context: not built in its owning module's context"
  has "and checks the plain arm" "1 snippet(s) checked: 1 accepted, 0 rejected, 0 known rejection(s); 3 refused"

  local stub="$TMP/p/declared.mli"
  printf '%s\n' 'val use_padding : bool' '[@@@owning_module 3]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "an owning module that is not a string" 2
  printf '%s\n' 'val use_padding : bool' '[@@@owning_module ""]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "an empty owning module" 2
  printf '%s\n' 'val use_padding : bool' '[@@@owning_module "lib/nn_blocks.ml"]' \
    '[@@@owning_module "tensor/tensor.ml"]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "two owning modules" 2
  printf '%s\n' 'val use_padding : bool' '[@@@owning_module "no/such_module.ml"]' >"$stub"
  run --stub "$stub" "$ptable"
  expect_rc "an owning module the repository does not have" 2
  if grep -qF 'no such file' "$err"; then report 0 "is named as missing before any build"
  else report 1 "is named as missing before any build" "stderr: $(head -n 6 "$err" | tr '\n' '|')"
  fi

  local itable="$TMP/t/interface_cases.ml"
  cat >"$itable" <<'CASES'
let () =
  case ~interface:{ocaml|val f : ?feature:bool -> unit -> unit|ocaml} "a case with its interface" {ocaml|let f ?(feature = true) () = ignore feature|ocaml};
  case ~interface:{ocaml|val g : no_such_type|ocaml} "a case with an ill-typed interface" {ocaml|let g = 1|ocaml}
CASES
  run --no-stub --verbose "$itable"
  expect_rc "a table with an ill-typed interface snippet" 1
  has "a well-typed interface is accepted as a signature" "ok        $itable:2: a case with its interface (interface)"
  has "beside its implementation" "ok        $itable:2: a case with its interface"
  has "an ill-typed interface is rejected by its case's name" \
    "REJECTED  $itable:3: a case with an ill-typed interface (interface)"
  has "as a type error, not as a value declaration outside a signature" "Unbound type constructor no_such_type"
}
harness_leg_run '12: project syntax refused without its context, owning-module declarations and interface snippets' leg12

finish

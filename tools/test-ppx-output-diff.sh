#!/usr/bin/env bash
# Controls for tools/ppx-output-diff.sh, the comparison of the preprocessed sources at two
# revisions (gh-ocannl-1886), over a planted repository: a dune project with a one-extension ppx
# (`[%planted]` expands to an integer literal), a library preprocessed by it and one that is not.
#
#   tools/test-ppx-output-diff.sh          # run every leg
#   tools/test-ppx-output-diff.sh --keep   # keep the scratch directory
#
# Its history: `base`; `planted`, where only the ppx changes (1 becomes 2); `edited`, where a
# preprocessed source changes and another is added; `broken`, where the ppx no longer compiles.
# Leg 1 is the point of the tool: the planted ppx change shows up in the diff, as an EXPANSION of
# a source that did not change, and nothing else does. Leg 2 tells a source edit (SOURCE) and a
# new file (ADDED) from it. Leg 3 compares a revision with itself, under an exported
# DUNE_BUILD_DIR the builds must not follow: exit 0, no diff, the renderings counted. Leg 4 refuses
# what is no verdict, exit 2: a scope with no preprocessed file and a revision whose ppx does not
# build. Leg 5 keeps the trees and the diff under --out. Leg 6 is the usage errors. Every leg checks the planted repository is left as it was -- HEAD, status and
# worktree list -- and the subject's scratch directory removed, refusals included.
#
# It needs git, dune and ocamlc, and ppxlib installed for dune (the toolchain CI job has it);
# without them every leg is a counted skip.

set -u

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
subject="$script_dir/ppx-output-diff.sh"
. "$script_dir/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch test-ppx-output-diff
# A CI step does not carry setup-ocaml's switch on its PATH: load it as the other toolchain
# harnesses do, leaving a host with no opam to the counted skips below.
command -v dune >/dev/null 2>&1 || . "$script_dir/opam-env.sh" 2>/dev/null
harness_require git dune ocamlc

# The fixture's own git: no ambient configuration, identity or repository selection.
unset GIT_DIR GIT_WORK_TREE GIT_INDEX_FILE GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES \
  GIT_COMMON_DIR GIT_CEILING_DIRECTORIES
export GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
  GIT_AUTHOR_NAME=fixture GIT_AUTHOR_EMAIL=fixture@example.invalid \
  GIT_COMMITTER_NAME=fixture GIT_COMMITTER_EMAIL=fixture@example.invalid

R=$TMP/repo
mkdir -p "$R/ppx" "$R/src" "$R/plain" "$TMP/subject-tmp"
printf '(lang dune 3.0)\n' >"$R/dune-project"
printf '_build/\n' >"$R/.gitignore"
cat >"$R/ppx/dune" <<'EOF'
(library
 (name planted_ppx)
 (kind ppx_rewriter)
 (libraries ppxlib))
EOF
cat >"$R/ppx/planted_ppx.ml" <<'EOF'
open Ppxlib

let planted =
  Extension.V3.declare "planted" Extension.Context.expression
    Ast_pattern.(pstr nil)
    (fun ~ctxt ->
      let loc = Expansion_context.Extension.extension_point_loc ctxt in
      Ast_builder.Default.eint ~loc 1)

let () = Driver.register_transformation "planted" ~rules:[ Context_free.Rule.extension planted ]
EOF
printf '(library\n (name subject)\n (preprocess (pps planted_ppx)))\n' >"$R/src/dune"
printf 'let a = [%%planted] + 1\n' >"$R/src/a.ml"
printf 'val a : int\n' >"$R/src/a.mli"
printf 'let b = 2\n' >"$R/src/b.ml"
printf '(library (name plain))\n' >"$R/plain/dune"
printf 'let c = [%%planted]\n' >"$R/plain/c.ml"
commit() { # TAG MESSAGE
  git -C "$R" add -A && git -C "$R" commit -q -m "$2" && git -C "$R" tag "$1"
}
fixture_ok=1
{ git -C "$R" init -q && commit base 'base'; } || fixture_ok=0
sed 's/eint ~loc 1/eint ~loc 2/' "$R/ppx/planted_ppx.ml" >"$TMP/p" && mv "$TMP/p" "$R/ppx/planted_ppx.ml" &&
  commit planted 'the ppx expands to 2' || fixture_ok=0
printf 'let b = 3\n' >"$R/src/b.ml" && printf 'let e = [%%planted]\n' >"$R/src/e.ml" &&
  commit edited 'a source edited, one added' || fixture_ok=0
printf 'let = broken\n' >>"$R/ppx/planted_ppx.ml" && commit broken 'the ppx no longer compiles' || fixture_ok=0
git -C "$R" checkout -q base || fixture_ok=0
snapshot() {
  git -C "$R" rev-parse HEAD
  git -C "$R" status --porcelain --untracked-files=all
  git -C "$R" worktree list --porcelain
}
# Whether dune finds ppxlib, asked of the planted project, never of whatever project the caller
# stands in: a dune already running there holds its build lock, and a failed probe is no absence.
# Its `_build` is ignored, so the snapshot below does not see it.
if ! dune installed-libraries --root "$R" >"$TMP/libs" 2>"$TMP/libs.err"; then
  report 1 "dune lists the installed libraries" "$(tr '\n' '|' <"$TMP/libs.err")"
  finish
fi
if ! grep -q '^ppxlib ' "$TMP/libs"; then
  skip "every leg building the planted ppx" "dune finds no installed ppxlib"
  finish
fi
snapshot >"$TMP/before" 2>&1 || fixture_ok=0

out="$TMP/out" err="$TMP/err"
run() { # run the subject on the planted repository; status in $rc, stdout in $out, stderr in $err
  rc=0
  TMPDIR="$TMP/subject-tmp" "$subject" -C "$R" "$@" >"$out" 2>"$err" || rc=$?
}
expect_rc() { # LABEL WANT [STDERR-REASON]; exit 2 always needs its reason
  if [ "$rc" = "$2" ] &&
    { [ "$2" != 2 ] || { [ -n "${3:-}" ] && grep -qF -- "$3" "$err"; }; }; then
    report 0 "$1 exits $2${3:+: $3}"
  else report 1 "$1 exits $2${3:+: $3}" "exited $rc; stdout: $(head -n 12 "$out" | tr '\n' '|') stderr: $(tail -n 12 "$err" | tr '\n' '|')"
  fi
}
has() { # LABEL FILE FIXED-LINE: FILE carries exactly this line
  if grep -qxF -- "$3" "$2"; then report 0 "$1"
  else report 1 "$1" "no line '$3' in: $(head -n 16 "$2" | tr '\n' '|')"
  fi
}
lacks() { # LABEL FILE FIXED-TEXT: no line of FILE contains the text
  if grep -qF -- "$3" "$2"; then report 1 "$1" "a line carries '$3' in: $(head -n 16 "$2" | tr '\n' '|')"
  else report 0 "$1"
  fi
}
untouched() { # the planted repository as the fixture left it, the subject's scratch removed
  snapshot >"$TMP/after" 2>&1
  if cmp -s "$TMP/before" "$TMP/after"; then report 0 "the planted repository is left as it was"
  else report 1 "the planted repository is left as it was" "$(diff "$TMP/before" "$TMP/after" | tr '\n' '|')"
  fi
  if [ -z "$(ls -A "$TMP/subject-tmp")" ]; then report 0 "the subject's scratch directory is removed"
  else report 1 "the subject's scratch directory is removed" "left: $(ls -A "$TMP/subject-tmp" | tr '\n' ' ')"
  fi
}
fixture() { # abandon the leg when the shared fixture is not what its checks assume
  [ "$fixture_ok" = 1 ] || { harness_abandon "the planted repository could not be built"; return 1; }
}

leg1() {
  fixture || return
  run base planted
  expect_rc "a planted ppx change" 1
  has "the diff names the rendering of the source using the extension" "$out" \
    'diff -ru base/src/a.pp.ml head/src/a.pp.ml'
  has "with the old expansion" "$out" '-let a = 1 + 1'
  has "and the new" "$out" '+let a = 2 + 1'
  has "its header carries no timestamp" "$out" '--- base/src/a.pp.ml'
  has "it is an EXPANSION: the source did not change" "$err" 'ppx-output-diff: EXPANSION  src/a.pp.ml'
  lacks "a rendering the extension does not reach is not in the diff" "$out" 'b.pp.ml'
  lacks "nor is the interface" "$out" 'a.pp.mli'
  lacks "a library no ppx preprocesses has no rendering" "$err" 'plain/'
  has "the verdict counts it" "$err" \
    "ppx-output-diff: 1 rendering(s) differ between $(git -C "$R" rev-parse --short=12 base) and $(git -C "$R" rev-parse --short=12 planted): 1 EXPANSION (input unchanged), 0 SOURCE, 0 ADDED, 0 REMOVED"
  untouched
}
harness_leg_run '1: a planted ppx change shows up in the diff, as an EXPANSION of an unchanged source' leg1

leg2() {
  fixture || return
  run planted edited
  expect_rc "a source edit and a new source" 1
  has "the edit is in the diff" "$out" '+let b = 3'
  has "it is a SOURCE change" "$err" 'ppx-output-diff: SOURCE     src/b.pp.ml'
  has "the new file is one side's alone" "$out" 'Only in head/src: e.pp.ml'
  has "it is ADDED" "$err" 'ppx-output-diff: ADDED      src/e.pp.ml'
  lacks "the unchanged expansion is not" "$err" 'a.pp.ml'
  untouched
}
harness_leg_run '2: a source edit is SOURCE and a new file ADDED, never EXPANSION' leg2

leg3() {
  fixture || return
  mkdir -p "$TMP/elsewhere"
  DUNE_BUILD_DIR="$TMP/elsewhere" run planted planted
  expect_rc "a revision against itself, under an exported DUNE_BUILD_DIR" 0
  if [ -z "$(ls -A "$TMP/elsewhere")" ]; then report 0 "the builds stay in the throwaway worktree"
  else report 1 "the builds stay in the throwaway worktree" "DUNE_BUILD_DIR got: $(ls -A "$TMP/elsewhere" | tr '\n' ' ')"
  fi
  if [ -s "$out" ]; then report 1 "no diff is printed" "$(head -n 8 "$out" | tr '\n' '|')"
  else report 0 "no diff is printed"
  fi
  has "the renderings were counted, not vacuously equal" "$err" \
    "ppx-output-diff: no difference between $(git -C "$R" rev-parse --short=12 planted) and $(git -C "$R" rev-parse --short=12 planted) over 3 rendering(s)"
  untouched
}
harness_leg_run '3: a revision against itself exits 0 over a counted corpus, whatever DUNE_BUILD_DIR says' leg3

leg4() {
  fixture || return
  run base planted -- plain
  expect_rc "a scope with no preprocessed file" 2 "ppx-output-diff: base ($(git -C "$R" rev-parse base)): no preprocessed file in scope (1 director(ies) listed): nothing to compare"
  untouched
  run base broken
  expect_rc "a revision whose ppx does not build" 2 "ppx-output-diff: head: dune failed to build the renderings of $(git -C "$R" rev-parse broken)"
  if [ -s "$out" ]; then report 1 "and prints no diff" "$(head -n 8 "$out" | tr '\n' '|')"
  else report 0 "and prints no diff"
  fi
  untouched
}
harness_leg_run '4: an empty scope and a failed build are refused, exit 2' leg4

leg5() {
  fixture || return
  run --out "$TMP/kept" base planted -- src
  expect_rc "a run keeping its trees" 1
  if [ -f "$TMP/kept/base/src/a.pp.ml" ] && [ -f "$TMP/kept/head/src/a.pp.mli" ]; then
    report 0 "both rendered trees are kept"
  else report 1 "both rendered trees are kept" "$(find "$TMP/kept" -type f | tr '\n' ' ')"
  fi
  if cmp -s "$TMP/kept/ppx.diff" "$out"; then report 0 "the kept diff is the one printed"
  else report 1 "the kept diff is the one printed" "$(diff "$TMP/kept/ppx.diff" "$out" | head -n 8 | tr '\n' '|')"
  fi
  untouched
  run --out "$TMP/kept" base planted
  expect_rc "an --out directory that exists" 2 "ppx-output-diff: --out $TMP/kept already exists"
  untouched
}
harness_leg_run '5: --out keeps both rendered trees and the diff, and never reuses a directory' leg5

leg6() {
  fixture || return
  run base
  expect_rc "one revision" 2 "ppx-output-diff: usage: tools/ppx-output-diff.sh"
  run base no-such-revision
  expect_rc "an unresolvable revision" 2 'ppx-output-diff: HEAD does not resolve to a commit: no-such-revision'
  run base planted -- ../outside
  expect_rc "a scope outside the repository" 2 "ppx-output-diff: a scope DIR is a repository-relative path: '../outside'"
  run -j 0 base planted
  expect_rc "a zero job count" 2 "ppx-output-diff: -j needs a positive count, not '0'"
  untouched
}
harness_leg_run '6: usage errors exit 2 before anything is built' leg6

finish

#!/usr/bin/env bash
# Legs for tools/affected-tests.sh (`tools/test-run.sh affected`), on a small
# real Dune project of its own: which per-test aliases a change reaches, read
# from dune. No Dune alias: the fixture owns its build, and a dune nested in
# `dune runtest` is a poor bargain (its own lock, the outer run's DUNE_*).
# Usage: tools/test-affected-tests.sh [--keep] [--only REGEX]
#
# The fixture mirrors the shapes the three CI catches of gh-ocannl-1235 had: a
# library module reached from tests in other directories through a chain of
# libraries; a test whose alias runs an executable of another name; a scan that
# reads library sources as data and links none of them. It adds a ppx rewriter
# and a library this build does not compile. Each leg compares the printed list
# with the exact expected one, so an over-reach fails as surely as a miss. The
# real repository's three commits are the PR's reproduction, since reading its
# graph needs the full opam switch, which this harness's CI job does not install.
# The harness legs (gh-ocannl-1900) are read on a dune-free fixture and on this
# repository's own harnesses; the latter reads the scan-case stubs directory:
# affected-reads: test/operations/scan_case_stubs/* --only '^harness legs of this repository$'
set -u
# The expected lists are sorted with `sort` and compared with the helper's
# bytewise order: one collation, whatever the caller's locale (gh-ocannl-1431).
export LC_ALL=C
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
HERE=$(cd "$(dirname "$0")" && pwd)
command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh"
harness_require dune perl git
harness_scratch test-affected-tests
unset DUNE_BUILD_DIR DUNE_ROOT
export OCANNL_TOOL_FLEET_WORKER=none OCANNL_TOOL_TEST_RUNS=$TMP/runs
# The legs down to the probe section pin the reader without its probe
# (gh-ocannl-1541): what it falls back to when the probe fails or is turned
# off, every alias action dune does not dump read blind. The probe section
# below turns it back on.
export OCANNL_TOOL_AFFECTED_PROBE=0

repo=$TMP/proj
mkdir -p "$repo/tools" "$repo/lib" "$repo/mid" "$repo/t1" "$repo/t2" "$repo/t3" "$repo/t4/sub" "$repo/t5" "$repo/t6" "$repo/t7" \
  "$repo/opt" "$repo/cfg" "$repo/docs" || exit 2
cp "$HERE/affected-tests.sh" "$HERE/test-run.sh" "$repo/tools/" || exit 2
# Whatever test-run.sh sources, derived from it rather than listed here.
staged=0
while IFS= read -r rel; do
  mkdir -p "$repo/$(dirname "$rel")" && cp "$HERE/../$rel" "$repo/$rel" || exit 2
  staged=$((staged + 1))
done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$HERE/test-run.sh")
[ "$staged" -ge 3 ] || exit 2
printf '(lang dune 3.20)\n' >"$repo/dune-project"
: >"$repo/fx.opam"
# Two libraries this build does not compile, one requiring the other by its
# local name while dune lists it by its public one: the Linux shape of
# arrayjit's metal_backend and metal_math_api_runtime (gh-ocannl-1389).
cat >"$repo/opt/dune" <<'DUNE'
(library (name offa) (public_name fx.offa) (optional) (modules offa) (libraries not_installed_anywhere))
(library (name offb) (public_name fx.offb) (optional) (modules offb) (libraries offa))
DUNE
# A rule whose action reads a generated file: `dune rules` builds that file to
# print the rule, and what its action prints reaches dune's stderr.
# t7 includes t6's fragment too, beside a golden pair only its dune file reaches.
cat >"$repo/t7/dune" <<'DUNE'
(include ../t6/extra.inc)
(rule (target g7.actual) (action (with-stdout-to %{target} (echo g))))
(rule (alias runtest-g7) (action (diff g7.expected g7.actual)))
DUNE
: >"$repo/t7/g7.expected"
# A quoted atom longer than perl's 65534-step regex recursion limit: a
# one-character-per-step string pattern refused the real repository's
# dead_export_scan file list (74k characters) as unreadable output.
long_atom=$(printf '%*s' 70000 '' | tr ' ' 'a')
printf '(rule (target long.txt) (action (with-stdout-to %%{target} (echo "%s"))))\n' "$long_atom" \
  >>"$repo/t7/dune" || exit 2
cat >"$repo/t6/dune" <<'DUNE'
(rule (target word.txt) (action (progn (system "echo noise-from-an-action >&2") (with-stdout-to %{target} (echo dyn)))))
(rule (target dyn.actual) (action (with-stdout-to %{target} (echo %{read:word.txt}))))
(rule (alias runtest-dyn) (action (diff dyn.expected dyn.actual)))
(include extra.inc)
DUNE
cat >"$repo/lib/dune" <<'DUNE'
(library (name core) (modules core helper lx))
(ocamllex lx)
(library (name leaf) (modules leaf))
(library (name plain_core) (modules plain_core) (libraries core))
(library (name myppx) (kind ppx_rewriter) (modules myppx))
(library (name optlib) (modules optlib) (optional) (libraries not_installed_anywhere))
(library (name stubbed) (modules stubbed) (foreign_stubs (language c) (names stubbed_c)))
DUNE
cat >"$repo/mid/dune" <<'DUNE'
(library
 (name mid)
 (modules mid choice)
 (libraries (select choice.ml from (core -> choice.core.ml) (-> choice.none.ml))))
DUNE
cat >"$repo/t1/dune" <<'DUNE'
(test (name t_core) (modules t_core) (libraries mid))
; Links core through a private helper library and names nothing of it: link-only for a core
; edit. (Through mid it would not be: mid's (select) module is code not all known.)
(test (name t_mid) (modules t_mid) (libraries plain_core))
; The same executable under an unknown action: blind, so never ranked link-only.
(rule (alias runtest-t_mid-blind) (deps (package fx)) (action (run %{exe:t_mid.exe})))
(test (name t_leaf) (modules t_leaf) (libraries leaf) (flags (:standard -open Leaf)))
(tests (names ta tb) (modules ta tb shared) (libraries leaf))
(library (name tlib) (modules m_one m_two) (preprocess (pps myppx)))
(rule (alias runtest-m_one) (action (progn)))
(rule (target made.txt) (action (with-stdout-to %{target} (echo made))))
DUNE
cat >"$repo/t2/dune" <<'DUNE'
(executable (name gen) (modules gen) (libraries core))
(executable (name scan) (modules scan))
(rule (target gen-v1.actual) (action (with-stdout-to %{target} (run %{exe:gen.exe} v1))))
(rule (alias runtest-gen-v1) (action (diff gen-v1.expected gen-v1.actual)))
(rule (alias slow-gen) (action (run %{exe:gen.exe} slow)))
(rule (target scenario_x.actual) (action (with-stdout-to %{target} (run %{exe:gen.exe} x))))
(rule (alias runtest-scenario_x) (action (diff scenario_x.expected scenario_x.actual)))
(rule
 (target scan.actual)
 (deps (glob_files %{project_root}/lib/*.ml) %{project_root}/docs/notes.md)
 (action (with-stdout-to %{target} (run %{exe:scan.exe}))))
(rule (alias runtest-scan) (action (diff scan.expected scan.actual)))
(rule (alias runtest-orphan) (action (progn)))
; A rule this reader cannot read in full (a package dependency): its action stays undumped, and
; an alias needing it is not listed when it is listed only for that reason.
(rule (alias runtest-pkg) (deps (package fx)) (action (progn)))
(rule (target needs_pkg.actual) (deps (alias runtest-pkg)) (action (with-stdout-to %{target} (echo y))))
(rule (alias runtest-needs_pkg) (action (diff needs_pkg.expected needs_pkg.actual)))
(rule (alias scans) (deps (alias runtest-scan)) (action (progn)))
DUNE
cat >"$repo/t3/dune" <<'DUNE'
(copy_files ../cfg/base.cfg)
(test (name t_plain) (modules t_plain) (deps plain.cfg base.cfg))
(test (name t_stub) (modules t_stub) (libraries stubbed))
(executable (name cexe) (modules cexe) (foreign_stubs (language c) (names cexe_c)))
(executable (name lister) (modules lister))
(rule
 (target neg.actual)
 (deps (glob_files %{project_root}/mid/[!c]*.ml))
 (action (with-stdout-to %{target} (run %{exe:lister.exe}))))
(rule (alias runtest-neg) (action (diff neg.expected neg.actual)))
(rule (target cfgread.actual) (deps base.cfg) (action (with-stdout-to %{target} (run %{exe:lister.exe}))))
(rule (alias runtest-cfgread) (action (diff cfgread.expected cfgread.actual)))
(rule
 (target esc.actual)
 (deps (glob_files "%{project_root}/docs/\\{x\\}*.md"))
 (action (with-stdout-to %{target} (run %{exe:lister.exe}))))
(rule (alias runtest-esc) (action (diff esc.expected esc.actual)))
(rule
 (target uses.actual)
 (deps %{project_root}/t1/made.txt)
 (action (with-stdout-to %{target} (run %{exe:lister.exe}))))
(rule (alias runtest-uses) (action (diff uses.expected uses.actual)))
; Runs and diffs in one action (the slow-/train- shape), with a dependency outside its directory: a
; golden among its own targets, yet what its rule needs is only in its own (deps) (gh-ocannl-1431).
(rule
 (alias runtest-py)
 (deps py.expected ../cfg/extra.py (sandbox always))
 (action (no-infer (progn (with-stdout-to py.actual (echo x)) (diff py.expected py.actual)))))
DUNE
cat >"$repo/t4/dune" <<'DUNE'
(library (name inert) (modules inert))
(rule
 (target objs.actual)
 (deps (glob_files %{project_root}/t4/.inert.objs/byte/*.cmi))
 (action (with-stdout-to %{target} (echo objs))))
(rule (alias runtest-objs) (action (diff objs.expected objs.actual)))
(test (name t_gate) (modules t_gate) (deps gate.cfg))
(rule (target gated.actual) (deps (alias runtest-t_gate)) (action (with-stdout-to %{target} (echo gated))))
(rule (alias runtest-gated) (action (diff gated.expected gated.actual)))
(executable (name pre) (modules pre))
(rule (target t_gate_actual.ml) (action (with-stdout-to %{target} (run %{exe:pre.exe}))))
(rule (alias runtest-t_gate-pre) (action (diff t_gate_expected.ml t_gate_actual.ml)))
DUNE
printf '(test (name t_sub) (modules t_sub))\n' >"$repo/t4/sub/dune"
cat >"$repo/t5/dune" <<'DUNE'
(executable (name genh) (modules genh) (libraries core))
(rule (target gen.h) (action (with-stdout-to %{target} (run %{exe:genh.exe}))))
(executable
 (name usesh)
 (modules usesh)
 (foreign_stubs (language c) (names usesh_c) (extra_deps gen.h)))
(rule (alias slow-usesh) (action (run %{exe:usesh.exe})))
DUNE
for f in t5/genh.ml t5/usesh.ml opt/offa.ml opt/offb.ml; do printf 'let () = ()\n' >"$repo/$f"; done
: >"$repo/t5/usesh_c.c"
for f in t4/inert.ml t4/t_gate.ml t4/sub/t_sub.ml t4/pre.ml t4/t_gate_expected.ml; do printf 'let () = ()\n' >"$repo/$f"; done
for f in t4/objs.expected t4/gated.expected t4/gate.cfg t6/dyn.expected; do : >"$repo/$f"; done
for f in lib/core.ml lib/helper.ml lib/leaf.ml lib/myppx.ml lib/optlib.ml lib/stubbed.ml \
  t3/t_plain.ml t3/t_stub.ml t3/cexe.ml t3/lister.ml mid/mid.ml mid/choice.core.ml mid/choice.none.ml \
  t1/t_core.ml t1/t_leaf.ml t1/ta.ml t1/tb.ml t1/shared.ml t1/m_one.ml t1/m_two.ml \
  t2/gen.ml t2/scan.ml; do
  printf 'let () = ()\n' >"$repo/$f"
done
for f in t2/gen-v1.expected t2/scenario_x.expected t2/scan.expected t3/neg.expected t3/plain.cfg \
  t3/cfgread.expected t3/esc.expected t3/uses.expected t3/py.expected cfg/base.cfg cfg/extra.py docs/{x}one.md \
  t2/needs_pkg.expected \
  lib/stubbed_c.c t3/cexe_c.c docs/notes.md; do : >"$repo/$f"; done
printf '{ }\nrule token = parse eof { () }\n' >"$repo/lib/lx.mll"
# What the link-only ranking reads: t_core names Helper and its `used`; core re-exports Helper as
# H, which gen names as `Core.H` (a re-export is a name of the module making it); nothing else
# names either.
printf 'let used = 1\nlet unused = 2\n' >"$repo/lib/helper.ml"
printf 'module H = Helper\n' >"$repo/lib/core.ml"
printf 'let () = () (* uses Helper.used *)\n' >"$repo/t1/t_core.ml"
printf 'let () = ()\n' >"$repo/t1/t_mid.ml"
printf 'let () = ()\n' >"$repo/lib/plain_core.ml"
printf '(rule (alias runtest-inc) (action (progn)))\n' >"$repo/t6/extra.inc"
printf 'let () = () (* Core.H *)\n' >"$repo/t2/gen.ml"
git_q() { git -C "$repo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
git_q init -q && git_q add -A && git_q commit -qm fixture || exit 2

out=$TMP/out err=$TMP/err
affected() { # ARGS... -> $out, $err, $rc
  rc=0
  (cd "$TMP" && "$repo/tools/affected-tests.sh" "$@") >"$out" 2>"$err" || rc=$?
}
trailer() { # -> $verdict: the one `affected: verdict=` line when it is stderr's last, else <none>
  verdict=$(sed -n 's/^affected: verdict=//p' "$err")
  [ -n "$verdict" ] && [ "$(tail -n 1 "$err")" = "affected: verdict=$verdict" ] || verdict='<none>'
}
expect() { # LABEL RC ALIAS... : the exact list, in any order, and the exit status
  local label=$1 want_rc=$2
  shift 2
  printf '%s\n' "$@" | sed '/^$/d' | sort >"$TMP/want"
  # The printed order is ranked (link-only last); the 'link-only' leg pins it.
  sort "$out" >"$TMP/got"
  # Every printed list ends its stderr with the verdict trailer (gh-ocannl-1921),
  # `incomplete` in it exactly on exit 3; the 'verdict trailer' leg pins each one.
  trailer
  case $verdict/$rc in
    ok/0 | cross-cutting/0 | incomplete/3 | cross-cutting,incomplete/3) ;;
    *) report 1 "$label" "exit $rc with the verdict trailer $verdict; stderr: $(tr '\n' '|' <"$err")"; return ;;
  esac
  if [ "$rc" = "$want_rc" ] && cmp -s "$TMP/want" "$TMP/got"; then
    report 0 "$label"
  else
    report 1 "$label" "exit $rc (want $want_rc); got [$(tr '\n' ' ' <"$out")] want [$(tr '\n' ' ' <"$TMP/want")]; stderr: $(tr '\n' '|' <"$err")"
  fi
}
reached_by_core='@t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen
  @t5/slow-usesh'
# Every alias whose action dune does not dump (no golden pair among its own
# targets): listed for any changed data file, with what needs one of them
# (runtest-needs_pkg, t4's gated): a broken needed alias breaks their build.
opaque_all='@t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind
  @t1/runtest-ta @t1/runtest-tb @t2/runtest-orphan @t2/runtest-needs_pkg @t2/runtest-pkg @t2/slow-gen @t3/runtest-py @t3/runtest-t_plain
  @t3/runtest-t_stub @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre @t4/sub/runtest-t_sub @t5/slow-usesh @t6/runtest-inc
  @t7/runtest-inc'

if harness_leg 'cold build'; then
  # On a build directory that has never built t6/word.txt, `dune rules` builds
  # it to print dyn.actual's rule, and its action's stderr line must not be
  # read as part of the answer (gh-ocannl-1389: "unexpected atom"). Only the
  # file `dune rules -o` writes is the answer; dune's own streams are kept apart.
  rm -rf "$repo/_build"
  rc=0
  (cd "$TMP" && DUNE_CACHE=disabled "$repo/tools/affected-tests.sh" --files t6/dyn.expected) >"$out" 2>"$err" || rc=$?
  # shellcheck disable=SC2086
  expect 'a cold build directory: what an action prints while dune rules builds an input is not read as rules' 0 \
    @t6/runtest-dyn $opaque_all
  mutant=$repo/tools/affected-mutant.sh
  mkdir -p "$TMP/kept" || exit 2
  if FROM='my $text = slurp("$tmp/rules-out");' TO='my $text = slurp("$tmp/rules-log") . slurp("$tmp/rules-out");' \
      perl -0pe '$n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rm -rf "$repo/_build"
    rc=0
    (cd "$TMP" && TMPDIR=$TMP/kept DUNE_CACHE=disabled bash "$mutant" --files t6/dyn.expected) >"$out" 2>"$err" || rc=$?
    # The refusal names the atom and keeps the answer it refused (gh-ocannl-1431).
    # Which atom comes first is dune's: 3.24 prints the noise line alone, the 3.20
    # floor echoes the command first, wrapped in ANSI dim codes.
    if [ "$rc" = 2 ] && grep -Eq 'unexpected atom in dune rules output: ".+" at byte [0-9]+: ' "$err" &&
       grep -q noise-from-an-action "$TMP"/kept/affected-dune-rules.*.txt; then
      report 0 'negative control: reading dune rules'"'"' streams as its answer fails on the cold build, naming the atom'
    else report 1 'negative control: reading dune rules'"'"' streams as its answer fails on the cold build, naming the atom' "exit $rc; $(cat "$err")"; fi
  else
    report 1 'negative control: reading dune rules'"'"' streams as its answer fails on the cold build, naming the atom' \
      "the mutation anchor does not occur exactly once"
  fi
  rm -f "$mutant"
fi

if harness_leg 'library chain'; then
  affected --files lib/helper.ml
  # shellcheck disable=SC2086 # the list is words by design
  expect 'a library module reaches tests through a library chain, an executable of another name, a variant, a scan reading it' 0 $reached_by_core
  grep -q 'lib/helper.ml: module Helper of library core (lib)' "$err" ||
    report 1 'the mapping names the owning module and library' "$(cat "$err")"
fi

if harness_leg 'sibling library'; then
  affected --files lib/leaf.ml
  expect 'a sibling library reaches only its own users (and the scan reading it)' 0 \
    @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t2/runtest-scan
  # t_leaf is compiled with -open Leaf, so its code can use Leaf's names bare; ta and tb name nothing.
  affected --referenced-only --files lib/leaf.ml
  expect 'a stanza compiled with -open is never ranked link-only' 0 @t1/runtest-t_leaf @t2/runtest-scan
fi

if harness_leg 'select choice'; then
  affected --files mid/choice.core.ml
  expect 'a (select) choice maps to its module' 0 @t1/runtest-t_core
fi

if harness_leg 'tests helper module'; then
  affected --files t1/shared.ml
  expect 'a module of a tests stanza (undescribed) reaches every tests stanza of its directory' 0 \
    @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb
fi

if harness_leg 'library module alias'; then
  affected --files t1/m_one.ml
  expect 'a per-module alias of a library is attributed by the module name' 0 @t1/runtest-m_one
fi

if harness_leg 'ppx rewriter'; then
  affected --files lib/myppx.ml
  expect 'a ppx rewriter reaches a library it preprocesses (an edge dune describe workspace omits)' 0 \
    @t1/runtest-m_one @t2/runtest-scan
  # tlib's code never names Myppx, but the rewriter that changed rewrites it: never link-only.
  affected --referenced-only --files lib/myppx.ml
  expect 'code a changed rewriter preprocesses is never ranked link-only' 0 @t1/runtest-m_one @t2/runtest-scan
fi

if harness_leg 'generated module'; then
  affected --files lib/lx.mll
  expect 'an .mll reaches the stanza of the module its rule generates' 0 \
    @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
fi

if harness_leg 'golden'; then
  # The diff reading a golden is the alias's own action, which dune does not
  # dump; the golden's copy into the build is one of the alias's own targets.
  affected --files t2/gen-v1.expected
  # shellcheck disable=SC2086
  expect 'a golden lists the alias diffing it (and the aliases with undumped actions)' 0 \
    @t2/runtest-gen-v1 $opaque_all
fi

if harness_leg 'dune-project'; then
  affected --files dune-project
  expect 'dune-project reaches every per-test alias beneath it' 0 \
    @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb \
    @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan \
    @t2/runtest-scenario_x @t2/slow-gen @t3/runtest-cfgread @t3/runtest-esc @t3/runtest-neg @t3/runtest-py @t3/runtest-t_plain @t3/runtest-t_stub \
    @t3/runtest-uses @t4/runtest-gated @t4/runtest-objs @t4/runtest-t_gate @t4/runtest-t_gate-pre \
    @t4/sub/runtest-t_sub @t5/slow-usesh @t6/runtest-dyn @t6/runtest-inc @t7/runtest-g7 @t7/runtest-inc
  # A first tier spanning most of the suite says so (gh-ocannl-1765), keeping the slow- aliases
  # no CI job runs local (slow-gen, slow-usesh); a narrow one does not.
  cross_label='a first tier over half the per-test aliases is called cross-cutting, a narrow one is not'
  if [ "$rc" = 0 ] && grep -q '^affected: CROSS-CUTTING: 31 of the 31 per-test aliases rank first, so this change is cross-cutting: the PR'"'"'s CI run owns the full list but its 2 slow- alias(es), which no CI job runs: run those locally too;' "$err"; then
    affected --files lib/leaf.ml
    if [ "$rc" = 0 ] && [ -s "$out" ] && ! grep -q 'CROSS-CUTTING' "$err"; then report 0 "$cross_label"
    else report 1 "$cross_label" "a narrow change: exit $rc; $(tr '\n' '|' <"$err")"; fi
    # The slow- aliases ranked link-only count too (local review): with core's helper changed, 10 of
    # the 31 rank first by a mutant threshold, slow-gen among the link-only ones.
    mutant=$repo/tools/affected-mutant.sh
    if FROM='return $n_all && 2 * $n_first > $n_all;' TO='return $n_all && $n_first;' \
        perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
        "$repo/tools/affected-tests.sh" >"$mutant"; then
      printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
      rc=0
      (cd "$TMP" && bash "$mutant" HEAD -- lib) >"$out" 2>"$err" || rc=$?
      git_q checkout -q -- lib/helper.ml || exit 2
      if [ "$rc" = 0 ] && grep -qx 'affected: link-only: @t2/slow-gen' "$err" && grep -q 'but its 2 slow- alias(es)' "$err"; then
        report 0 'the slow- aliases kept local include the link-only ones'
      else report 1 'the slow- aliases kept local include the link-only ones' "exit $rc; $(tr '\n' '|' <"$err")"; fi
    else report 1 'the slow- aliases kept local include the link-only ones' "the mutation anchor does not occur exactly once"; fi
    rm -f "$mutant"
  else report 1 "$cross_label" "dune-project: exit $rc; $(tr '\n' '|' <"$err")"; fi
fi

if harness_leg 'dune file elsewhere'; then
  # t1/dune makes made.txt, which a t3 rule reads.
  affected --files t1/dune
  expect 'a dune file reaches the rules elsewhere that read the files it makes' 0 \
    @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb @t3/runtest-uses
fi

if harness_leg 'test with no libraries'; then
  # t_plain names no library, so neither describe answer lists its stanza.
  affected --files t3/t_plain.ml
  expect 'a test stanza naming no library is still found, through its executable' 0 @t3/runtest-t_plain
fi

if harness_leg 'opaque action'; then
  # t_plain has no golden: dune runs it as the alias's own action under the
  # stanza's (deps), which `dune rules` does not show; this reader reads them
  # in t3/dune. Only an alias whose stanzas it cannot read is listed blind.
  affected --files t3/plain.cfg
  # shellcheck disable=SC2086
  expect 'a data file lists every alias whose action dune does not dump' 0 $opaque_all
fi

if harness_leg 'foreign stubs'; then
  affected --files lib/stubbed_c.c
  expect 'a C stub reaches the library whose stub archive bundles its object' 0 @t3/runtest-t_stub
  affected --files t3/cexe_c.c
  expect 'a C stub no archive bundles reaches every executable and test stanza of its directory' 0 \
    @t3/runtest-cfgread @t3/runtest-esc @t3/runtest-neg @t3/runtest-py @t3/runtest-t_plain @t3/runtest-t_stub \
    @t3/runtest-uses
fi

if harness_leg 'negated glob'; then
  affected --files mid/mid.ml
  expect 'a [!set] glob matches outside its set (the select-choice leg pins the inside)' 0 \
    @t1/runtest-t_core @t3/runtest-neg
fi

if harness_leg 'escaped glob'; then
  affected --files 'docs/{x}one.md'
  # dune prints `\{x\}*.md` as `{x}*.md`: the escapes are gone, so braces are also read literally.
  # shellcheck disable=SC2086
  expect 'a glob with escaped characters matches the file it names' 0 @t3/runtest-esc $opaque_all
fi

if harness_leg 'copied file'; then
  # cfg/ has no dune file; t3 copies cfg/base.cfg in (copy_files), a rule
  # reads the copy, and t_plain's (deps) name it.
  affected --files cfg/base.cfg
  # shellcheck disable=SC2086
  expect 'a file copied into a directory reaches the rules reading the copy' 0 @t3/runtest-cfgread $opaque_all
fi

if harness_leg 'batched query'; then
  # A tiny budget splits `dune rules` into many calls; the answer must not move.
  affected --files cfg/base.cfg
  cp "$out" "$TMP/whole"
  rc=0
  (cd "$TMP" && OCANNL_TOOL_AFFECTED_ARGS=40 "$repo/tools/affected-tests.sh" --files cfg/base.cfg) >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 0 ] && [ -s "$out" ] && cmp -s "$TMP/whole" "$out"; then
    report 0 'dune rules queried in small batches gives the same list'
  else report 1 'dune rules queried in small batches gives the same list' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'compiled artifacts'; then
  # objs.actual reads inert's compiled interfaces without linking it.
  affected --files t4/inert.ml
  expect 'a rule reading a stanza'"'"'s compiled objects is reached through that stanza' 0 @t4/runtest-objs
fi

if harness_leg 'alias dependency'; then
  # gated.actual needs the alias runtest-t_gate, listed for its changed executable.
  affected --files t4/t_gate.ml
  expect 'an alias whose rules need a listed alias is listed too' 0 @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre
  # runtest-pkg and runtest-t_gate are listed only because their actions are unknown;
  # what needs them is listed with them, in their tier (gh-ocannl-1431 review).
  affected --files docs/other.md
  # shellcheck disable=SC2086
  expect 'an alias needing one listed only blind is listed with it' 0 $opaque_all
fi

if harness_leg 'run-and-diff golden'; then
  # runtest-py runs and diffs in its own action and owns a golden; taken for a
  # golden pair, it was never listed for the file its own (deps) name outside its
  # directory (gh-ocannl-1431). With no rule-made target beside its golden it is
  # opaque, and listed blind.
  affected --files cfg/extra.py
  # shellcheck disable=SC2086
  expect 'a run-and-diff alias owning a golden is opaque, so a file its (deps) name lists it' 0 $opaque_all
fi

if harness_leg 'dune subtree'; then
  affected --files t4/dune
  expect 'a dune file reaches the per-test aliases of the directories beneath it' 0 \
    @t4/runtest-gated @t4/runtest-objs @t4/runtest-t_gate @t4/runtest-t_gate-pre @t4/sub/runtest-t_sub
fi

if harness_leg 'underscore target'; then
  # runtest-t_gate-pre diffs t_gate_actual.ml, which pre.exe makes: the test/ppx shape (its
  # name owner, t_gate, is not reached; runtest-t_gate shares the t_gate_ targets, and gated needs it).
  affected --files t4/pre.ml
  expect 'an alias owns the UNIT_ targets beside it, and so the executable their rules run' 0 \
    @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre
fi

if harness_leg 'CDPATH'; then
  # Invoked by a relative path, the root lookup must not follow CDPATH to a
  # decoy holding a tools/ directory (and print it on stdout).
  mkdir -p "$TMP/decoy/tools" || exit 2
  rc=0
  (cd "$repo" && CDPATH="$TMP/decoy" tools/affected-tests.sh --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
  # shellcheck disable=SC2086
  expect 'CDPATH does not move the repository root' 0 $reached_by_core
fi

if harness_leg 'undescribed stanza'; then
  affected --files lib/optlib.ml
  if [ "$rc" = 3 ] && grep -q 'UNMAPPED: lib/optlib.ml' "$err" &&
     grep -q 'does not build: library optlib (lib)' "$err" && [ "$(cat "$out")" = @t2/runtest-scan ]; then
    report 0 'a module of a library this build does not compile is UNMAPPED, exit 3, its rule readers still listed'
  else report 1 'a module of a library this build does not compile is UNMAPPED, exit 3, its rule readers still listed' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'off-build reach'; then
  # What a build that compiles opt9 would reach (gh-ocannl-1651: cuda_backend.ml on a Mac): sel
  # links it through a (select) by its local name, which dune lists by its public one, so this
  # build has no edge to it; its tests are named when their own code names Opt9.
  mkdir -p "$repo/t9/opt" || exit 2
  printf '(library (name opt9) (public_name fx.opt9) (optional) (modules opt9) (libraries not_installed_anywhere))\n' \
    >"$repo/t9/opt/dune"
  printf 'let () = ()\n' >"$repo/t9/opt/opt9.ml"
  cat >"$repo/t9/dune" <<'DUNE'
(library
 (name sel)
 (public_name fx.sel)
 (modules sel choice9)
 (libraries (select choice9.ml from (opt9 -> choice9.on.ml) (-> choice9.off.ml))))
(test (name t_sel_named) (modules t_sel_named) (libraries sel))
(test (name t_sel_plain) (modules t_sel_plain) (libraries sel))
DUNE
  for f in sel choice9.on choice9.off t_sel_plain; do printf 'let () = ()\n' >"$repo/t9/$f.ml"; done
  printf 'let () = ignore Opt9.x\n' >"$repo/t9/t_sel_named.ml"
  git_q add -A t9 && git_q commit -qm select-optional || exit 2
  affected --files t9/opt/opt9.ml
  if [ "$rc" = 3 ] && grep -q 'UNMAPPED: t9/opt/opt9.ml' "$err" && [ ! -s "$out" ] &&
     grep -q 'OFF-BUILD: t9/opt/opt9.ml is a module of library fx.opt9 (t9/opt), which this build does not compile. Where it is compiled, library fx.sel (t9) links it (by name or a (select)), reaching 2 per-test alias(es)' "$err" &&
     grep -q '1 of them name the module in their own code (or hold code not all known), named below, 1 only link it' "$err" &&
     [ "$(grep '^affected: off-build: ' "$err")" = 'affected: off-build: @t9/runtest-t_sel_named' ]; then
    report 0 'an off-build module names what a build compiling it reaches, on stderr only, still exit 3'
  else report 1 'an off-build module names what a build compiling it reaches, on stderr only, still exit 3' "exit $rc; $(cat "$out" "$err")"; fi
  git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/t9" || exit 2
fi

if harness_leg 'off-build library edge'; then
  # offb requires offa by its local name; external-lib-deps names offa by its
  # public one, and neither is compiled here, so the edge is skipped, not fatal.
  affected --files opt/offa.ml
  if [ "$rc" = 3 ] && grep -q 'UNMAPPED: opt/offa.ml' "$err" &&
     grep -q 'library offa (required by library fx.offb (opt)) is not in this build, nor is the stanza requiring it; edge skipped' "$err"; then
    report 0 'a library a stanza this build does not compile requires by its local name is skipped and said so'
  else report 1 'a library a stanza this build does not compile requires by its local name is skipped and said so' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'dune file'; then
  affected --files t2/dune
  expect 'a changed dune file lists every per-test alias of its directory, never an aggregate' 0 \
    @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen
fi

if harness_leg 'data read'; then
  affected --files docs/notes.md
  # shellcheck disable=SC2086
  expect 'a non-OCaml file a rule reads lists that rule'"'"'s alias' 0 @t2/runtest-scan $opaque_all
  affected --files docs/other.md
  printf '%s\n' $opaque_all | sort >"$TMP/want"
  if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out" && grep -q 'unread: docs/other.md' "$err"; then
    report 0 'a file no rule reads lists only the undumped actions and says so'
  else report 1 'a file no rule reads lists only the undumped actions and says so' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'unmapped'; then
  affected --files nowhere/ghost.ml
  if [ "$rc" = 3 ] && [ ! -s "$out" ] && grep -q 'UNMAPPED: nowhere/ghost.ml' "$err"; then
    report 0 'an .ml no stanza owns and no rule reads is UNMAPPED, exit 3'
  else report 1 'an .ml no stanza owns and no rule reads is UNMAPPED, exit 3' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'verdict trailer'; then
  # The machine-readable verdict (gh-ocannl-1921): the last line of stderr, one
  # of four, whenever the list is printed; `tools/test-run.sh prepush` reads it.
  vt_detail=
  for vt_case in "ok|0|--files lib/leaf.ml" \
                 "cross-cutting|0|--files dune-project" \
                 "incomplete|3|--files nowhere/ghost.ml" \
                 "incomplete|3|--files lib/optlib.ml" \
                 "cross-cutting,incomplete|3|--files dune-project nowhere/ghost.ml" \
                 "ok|0|--referenced-only --files lib/helper.ml" \
                 "ok|0|--files lib/core.ml lib/leaf.ml lib/stubbed.ml" \
                 "cross-cutting|0|--files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml" \
                 "ok|0|--referenced-only --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml" \
                 "cross-cutting,incomplete|3|--files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml nowhere/ghost.ml" \
                 "<none>|2|--files" \
                 "<none>|2|--no-such-option"; do
    IFS='|' read -r vt_want vt_rc vt_args <<<"$vt_case"
    # shellcheck disable=SC2086 # the case's operands, split on purpose
    affected $vt_args
    trailer
    # The prose and the trailer say the same thing, and no other line looks like a trailer.
    vt_prose=
    ! grep -q '^affected: CROSS-CUTTING:' "$err" || vt_prose=cross-cutting
    ! grep -q '; INCOMPLETE, see UNMAPPED above$' "$err" || vt_prose=${vt_prose:+$vt_prose,}incomplete
    [ "$vt_want" = '<none>' ] || [ -n "$vt_prose" ] || vt_prose=ok
    [ "$vt_want" != '<none>' ] || vt_prose='<none>'
    { [ "$rc" = "$vt_rc" ] && [ "$verdict" = "$vt_want" ] && [ "$vt_prose" = "$vt_want" ] &&
      [ "$(grep -c 'verdict=' "$err")" = "$([ "$vt_want" = '<none>' ] && echo 0 || echo 1)" ]; } ||
      { vt_detail="$vt_args: exit $rc (want $vt_rc), trailer $verdict (want $vt_want), prose says $vt_prose; stderr: $(tr '\n' '|' <"$err")"; break; }
  done
  vt_label='the last line of stderr is the verdict: ok, cross-cutting, incomplete or both; none when no list is printed'
  if [ -z "$vt_detail" ]; then report 0 "$vt_label"; else report 1 "$vt_label" "$vt_detail"; fi
  # (`--harnesses`, which prints no alias list, gives none: 'harness names and loads'.)
  # One definition: a mutant threshold in `verdict` moves the trailer and the
  # CROSS-CUTTING line together, for a change the shipping script calls ok.
  vt_label='negative control: the trailer and the CROSS-CUTTING line read one definition of the verdict'
  mutant=$repo/tools/affected-mutant.sh
  if FROM='return $n_all && 2 * $n_first > $n_all;' TO='return $n_all && $n_first;' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --files lib/leaf.ml) >"$out" 2>"$err" || rc=$?
    trailer
    if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] && grep -q '^affected: CROSS-CUTTING:' "$err"; then report 0 "$vt_label"
    else report 1 "$vt_label" "mutant exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  else report 1 "$vt_label" 'the mutation anchor does not occur exactly once'; fi
  # A list spanning over half the per-test aliases over a narrower first tier
  # (gh-ocannl-1996): 16 of the 31 listed, 10 first, where 15 listed is ok above
  # and the first tier alone (--referenced-only) is too. Its line names the
  # first tier as where the local run starts, and keeps the slow- aliases local.
  vt_label='a list over half the per-test aliases is cross-cutting by its list, and says how many rank first'
  affected --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml
  trailer
  if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] && [ "$(wc -l <"$out" | tr -d ' ')" = 16 ] &&
     grep -q '^affected: CROSS-CUTTING: 16 of the 31 per-test aliases are listed, 6 of them link-only, so this change is cross-cutting by its list, though only 10 rank first: the PR'"'"'s CI run owns the full list but its 2 slow- alias(es), which no CI job runs: run those locally too; locally run the tests that exercise the change, named in the PR body -- the 10 that rank first (--referenced-only) are where to start -- and `dune build @check`$' "$err"; then
    report 0 "$vt_label"
  else report 1 "$vt_label" "exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  # Without the list's threshold that change reads ok, as it did before it.
  vt_label='negative control: without the threshold on the list, the wide list reads ok'
  if FROM='return $n_all && 2 * $n_listed > $n_all;' TO='return 0;' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml) >"$out" 2>"$err" || rc=$?
    trailer
    if [ "$rc" = 0 ] && [ "$verdict" = ok ] && ! grep -q 'CROSS-CUTTING' "$err"; then report 0 "$vt_label"
    else report 1 "$vt_label" "mutant exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  else report 1 "$vt_label" 'the mutation anchor does not occur exactly once'; fi
  rm -f "$mutant"
  # The rule itself, on the counts this repository gave when gh-ocannl-1996 was
  # decided (825 per-test aliases): the fixture is too small to hold a list
  # that is mostly link-only. Each case: first tier, printed, all, UNMAPPED.
  vt_label='the verdict on a low_level.ml- and an autotune.ml-shaped list, their --referenced-only forms and the boundaries'
  vt_detail=$(perl -e '
    local $/; open my $fh, "<", $ARGV[0] or die "unreadable\n"; my $s = <$fh>;
    $s =~ /^(sub first_tier_wide \{.*?^sub verdict \{.*?^\})/ms or die "no verdict in the script\n";
    eval "$1; 1" or die "verdict does not load: $@";
    for (["low_level.ml: whole module", 379, 815, 825, 0, "cross-cutting"],
         ["low_level.ml: one function", 27, 815, 825, 0, "cross-cutting"],
         ["low_level.ml --referenced-only", 379, 379, 825, 0, "ok"],
         ["autotune.ml", 307, 636, 825, 0, "cross-cutting"],
         ["autotune.ml --referenced-only", 307, 307, 825, 0, "ok"],
         ["metal_backend.ml, nearly all link-only", 27, 815, 825, 0, "cross-cutting"],
         ["test/support/ll_test.ml, none link-only", 329, 329, 825, 0, "ok"],
         ["shape.ml, by its first tier", 544, 633, 825, 0, "cross-cutting"],
         ["half the aliases listed", 10, 412, 824, 0, "ok"],
         ["one over half listed", 10, 413, 824, 0, "cross-cutting"],
         ["one over half ranking first", 413, 413, 824, 0, "cross-cutting"],
         ["a wide list with an UNMAPPED path", 27, 815, 825, 1, "cross-cutting,incomplete"],
         ["an UNMAPPED path alone", 0, 0, 825, 1, "incomplete"],
         ["no per-test alias at all", 0, 0, 0, 0, "ok"]) {
      my ($what, @c) = @$_; my $want = pop @c;
      my $got = join(",", verdict(@c)) || "ok";
      print "$what (@c): $got, want $want; " if $got ne $want;
    }' "$repo/tools/affected-tests.sh" 2>&1)
  if [ -z "$vt_detail" ]; then report 0 "$vt_label"; else report 1 "$vt_label" "$vt_detail"; fi
fi

if harness_leg 'link-only'; then
  # Ranked, not dropped (gh-ocannl-1431): what names the changed module is printed first,
  # what only links its library last, and both are printed unless --referenced-only.
  # slow-usesh is ranked first through its generator: genh links core and writes the
  # header its stub compiles, so a core edit is code it builds under no name it shows.
  ordered() { # LABEL ALIAS... : stdout exactly, in this order
    local label=$1
    shift
    printf '%s\n' "$@" >"$TMP/want"
    if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then report 0 "$label"
    else report 1 "$label" "exit $rc; got [$(tr '\n' ' ' <"$out")] want [$*]; stderr: $(tr '\n' '|' <"$err")"; fi
  }
  affected --files lib/helper.ml
  ordered 'a module names its tests first (by name or a re-export), the link-only ones last' \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind
  grep -qx 'affected: link-only: @t1/runtest-t_mid' "$err" && grep -q '2 LINK-ONLY alias(es), printed LAST' "$err" ||
    report 1 'the link-only aliases are named on stderr' "$(cat "$err")"
  affected --referenced-only --files lib/helper.ml
  ordered '--referenced-only leaves the link-only ones off' \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  # Uncertainty ranks first: t_mid-blind runs t_mid under an unknown action, so with a data
  # file changed too it is never link-only.
  affected --referenced-only --files lib/helper.ml docs/other.md
  if [ "$rc" = 0 ] && grep -qx @t1/runtest-t_mid-blind "$out"; then
    report 0 'an alias whose unknown action might read a changed path is never ranked link-only'
  else report 1 'an alias whose unknown action might read a changed path is never ranked link-only' "exit $rc; $(cat "$out")"; fi
  printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
  affected HEAD
  # t_core links mid, whose (select) module is a copy of the arm this build chose, so read
  # through it (gh-ocannl-1765; before, code not all known, never link-only).
  ordered 'a change to one definition ranks first only what names it' \
    @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t5/slow-usesh @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen
  printf 'let used = 2\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected HEAD
  ordered '... and t_core names `used`' \
    @t1/runtest-t_core @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen
  # A changed line holding no code changes no definition (gh-ocannl-1765): the blank line and
  # doc comment above an added definition were attributed to the definition above them, so a
  # test naming only that neighbour ranked first.
  printf 'let () = () (* Core.H.used *)\n' >"$repo/t2/gen.ml"
  printf 'let used = 1\nlet unused = 2\n' >"$repo/lib/helper.ml"
  git_q commit -qam gen-names-used || exit 2
  printf 'let used = 1\n\n(** A doc. *)\nlet added = 0\nlet unused = 2\n' >"$repo/lib/helper.ml"
  inert_label='an added definition touches no neighbour through the blank line or doc comment above it'
  affected --referenced-only HEAD
  if [ "$rc" = 0 ] && ! grep -qx @t2/runtest-gen-v1 "$out" && grep -qF 'a changed module, Helper (added), but' "$err"; then
    report 0 "$inert_label"
  else report 1 "$inert_label" "exit $rc; $(tr '\n' ' ' <"$out"); $(tr '\n' '|' <"$err")"; fi
  mutant=$repo/tools/affected-mutant.sh
  if FROM='        next if $inert{$which}{$n};' TO='' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --referenced-only HEAD) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then report 0 "negative control: $inert_label"
    else report 1 "negative control: $inert_label" "exit $rc; $(cat "$out")"; fi
  else report 1 "negative control: $inert_label" "the mutation anchor does not occur exactly once"; fi
  rm -f "$mutant"
  printf 'let used = 1\n(* A note, "quoted". *)\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected HEAD --why @t2/runtest-gen-v1
  if [ "$rc" = 0 ] && grep -qF 'names Helper (`H`, re-exporting Helper from Core: `Core.H` at t2/gen.ml (its own source)) but no definition of it changed (only comments or blank lines did)' "$err" \
      && grep -qF 'Helper (no definition: only comments or blank lines changed)' "$err"; then
    report 0 'a comment-only change changes no definition, and says so'
  else report 1 'a comment-only change changes no definition, and says so' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  # ... but delimiters that comment out lines git shows unchanged change them: the whole module
  # (local review). gen names `used`, which `(*` and `*)` lines around it remove.
  printf '(*\nlet used = 1\n*)\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected --referenced-only HEAD
  if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then
    report 0 'commenting out an unchanged line changes the module as a whole'
  else report 1 'commenting out an unchanged line changes the module as a whole' "exit $rc; $(tr '\n' ' ' <"$out"); $(tr '\n' '|' <"$err")"; fi
  # A blank line inside a string literal is a changed value, not an inert line.
  printf 'let used = 1\nlet unused = 2\nlet text = {|a\nb|}\n' >"$repo/lib/helper.ml"
  git_q commit -qam helper-text || exit 2
  printf 'let used = 1\nlet unused = 2\nlet text = {|a\n\nb|}\n' >"$repo/lib/helper.ml"
  affected HEAD --why @t2/runtest-gen-v1
  if [ "$rc" = 0 ] && grep -qF 'Helper (text)' "$err"; then
    report 0 'a blank line inside a string literal changes the definition holding it'
  else report 1 'a blank line inside a string literal changes the definition holding it' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  git_q reset -q --hard HEAD~2 || exit 2
  mutant=$repo/tools/affected-mutant.sh
  if FROM='my @d = sort grep { $tok->{$_} } keys %$defs;' TO='my @d = sort keys %$defs;' \
      perl -0pe '$n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
    rc=0
    (cd "$TMP" && bash "$mutant" --referenced-only HEAD) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then
      report 0 'negative control: the changed definitions are what rank a test naming only the module'
    else report 1 'negative control: the changed definitions are what rank a test naming only the module' "exit $rc; $(cat "$out")"; fi
  else
    report 1 'negative control: the changed definitions are what rank a test naming only the module' \
      "the mutation anchor does not occur exactly once"
  fi
  rm -f "$mutant"
  printf 'let used = 1\nlet unused = 2\nlet ( +! ) a b = a + b\n' >"$repo/lib/helper.ml"
  affected HEAD
  ordered 'a changed operator definition counts as the whole module (tokens hold identifiers)' \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind
  # A type counts by its name, the constructors and fields its changed lines name, and what its
  # derivers make of its name (gh-ocannl-1651): gen names only `Core.H`, so a new type leaves it last.
  gen_first='@t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind'
  gen_last='@t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t5/slow-usesh @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen'
  printf 'let used = 1\nlet unused = 2\ntype t = A | B\n' >"$repo/lib/helper.ml"
  affected HEAD
  # shellcheck disable=SC2086
  ordered 'a new type ranks first only what names it, its constructors or what its derivers make' $gen_last
  git_q checkout -q -- lib/helper.ml || exit 2
  # gen names one field of a record type, which then gains another.
  printf 'let used = 1\nlet unused = 2\n\ntype r = {\n  first : int;\n}\n' >"$repo/lib/helper.ml"
  printf 'let () = () (* Core.H.first *)\n' >"$repo/t2/gen.ml"
  git_q commit -qam helper-record || exit 2
  printf 'let used = 1\nlet unused = 2\n\ntype r = {\n  first : int;\n  second : int;\n}\n' >"$repo/lib/helper.ml"
  affected HEAD
  # shellcheck disable=SC2086
  ordered 'a field added to a record ranks a test naming only another field last' $gen_last
  printf 'let used = 1\nlet unused = 2\n\ntype r = {\n  first : float;\n}\n' >"$repo/lib/helper.ml"
  affected HEAD
  # shellcheck disable=SC2086
  ordered '... and a changed field ranks first what names that field' \
    @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind
  printf 'let used = 1\nlet unused = 2\n\ntype r = {\n  first : int;\n}\n[@@deriving sexp]\n' >"$repo/lib/helper.ml"
  affected HEAD
  # shellcheck disable=SC2086
  ordered '... and a changed attribute of a type counts as the whole module' $gen_first
  git_q reset -q --hard HEAD~1 || exit 2
  # A new interface can hide any member: the whole module changed, whatever its hunks show.
  printf 'val used : int\n' >"$repo/lib/helper.mli"
  git_q add lib/helper.mli && git_q commit -qm helper-mli || exit 2
  affected --referenced-only HEAD~1..HEAD
  if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then
    report 0 'a file new on one side changes its module as a whole'
  else report 1 'a file new on one side changes its module as a whole' "exit $rc; $(cat "$out")"; fi
  git_q reset -q --hard HEAD~1 || exit 2
fi

if harness_leg 'option order'; then
  # `--referenced-only` is read in any position before `--` (gh-ocannl-1651): after the paths of
  # `--files`, after a revision, after `--since BASE`; after `--` it is a pathspec like any word.
  affected --referenced-only --files lib/helper.ml
  cp "$out" "$TMP/lead" || exit 2
  affected --files lib/helper.ml --referenced-only
  if [ "$rc" = 0 ] && [ -s "$TMP/lead" ] && cmp -s "$TMP/lead" "$out"; then report 0 '--files PATH --referenced-only reads as the leading spelling'
  else report 1 '--files PATH --referenced-only reads as the leading spelling' "exit $rc; got [$(tr '\n' ' ' <"$out")] want [$(tr '\n' ' ' <"$TMP/lead")]; $(tr '\n' '|' <"$err")"; fi
  printf 'let used = 1\nlet unused = 2\nlet more = 3\n' >"$repo/lib/helper.ml"
  affected --referenced-only HEAD
  cp "$out" "$TMP/lead" || exit 2
  affected HEAD --referenced-only
  if [ "$rc" = 0 ] && [ -s "$TMP/lead" ] && cmp -s "$TMP/lead" "$out"; then report 0 'REV --referenced-only reads as the leading spelling'
  else report 1 'REV --referenced-only reads as the leading spelling' "exit $rc; got [$(tr '\n' ' ' <"$out")] want [$(tr '\n' ' ' <"$TMP/lead")]; $(tr '\n' '|' <"$err")"; fi
  affected --since HEAD --referenced-only
  if [ "$rc" = 0 ] && cmp -s "$TMP/lead" "$out"; then report 0 '--since BASE --referenced-only reads as the leading spelling'
  else report 1 '--since BASE --referenced-only reads as the leading spelling' "exit $rc; got [$(tr '\n' ' ' <"$out")]; $(tr '\n' '|' <"$err")"; fi
  affected HEAD -- --referenced-only
  if [ "$rc" = 0 ] && [ ! -s "$out" ]; then report 0 'after --, --referenced-only is a pathspec (matching nothing here)'
  else report 1 'after --, --referenced-only is a pathspec (matching nothing here)' "exit $rc; got [$(tr '\n' ' ' <"$out")]; $(tr '\n' '|' <"$err")"; fi
  git_q checkout -q -- lib/helper.ml || exit 2
  # A flag takes no value: after the `--files` paths `--referenced-only=1` was read as one more
  # path, the flag silently dropped. The mutant without the refusal must answer the call.
  flag_label='--referenced-only=1 is refused by name, never read as a path'
  affected --files lib/helper.ml --referenced-only=1
  if [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -qF -- '--referenced-only is a flag and takes no value: --referenced-only=1' "$err"; then
    report 0 "$flag_label"
  else report 1 "$flag_label" "exit $rc (want 2); got [$(tr '\n' ' ' <"$out")]; $(tr '\n' '|' <"$err")"; fi
  mutant=$repo/tools/affected-mutant.sh
  if FROM='    --referenced-only=*) die' TO='    --referenced-only-cut=*) die' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --files lib/helper.ml --referenced-only=1) >"$out" 2>"$err" || rc=$?
    if [ "$rc" != 2 ] || ! grep -qF -- '--referenced-only is a flag' "$err"; then report 0 "negative control: $flag_label"
    else report 1 "negative control: $flag_label" "the mutant refused it too: exit $rc; $(tr '\n' '|' <"$err")"; fi
  else report 1 "negative control: $flag_label" "the mutation anchor does not occur exactly once"; fi
  rm -f "$mutant"
fi

# A private helper library and four tests linking it or core, committed as t8 (the
# 'helper definitions' and 'explain' legs; each removes it with `git_q reset -q --hard HEAD~1`).
helper_fixture() {
  mkdir -p "$repo/t8" || exit 2
  cat >"$repo/t8/dune" <<'DUNE'
(library (name hlp) (modules hlp) (libraries core))
(test (name t_hplain) (modules t_hplain) (libraries hlp))
(test (name t_hnamed) (modules t_hnamed) (libraries hlp))
(test (name t_hindirect) (modules t_hindirect) (libraries hlp))
(test (name t_hbare) (modules t_hbare) (libraries core))
(test (name t_hbeside) (modules t_hbeside) (libraries core))
(test (name t_hopen) (modules t_hopen) (libraries core))
(test (name t_halias) (modules t_halias) (libraries core))
(test (name t_haopen) (modules t_haopen) (libraries core))
(executable (name t_hgen) (modules t_hgen) (libraries core))
(rule (target t_hgen.ml) (deps t_hgen.src.ml) (action (with-stdout-to %{target} (echo "let () = ignore Core.H.unused\n"))))
(rule (alias runtest-t_hgen) (action (run %{exe:t_hgen.exe})))
(test (name t_hcross) (modules t_hcross) (libraries hlp))
(library (name hinc) (modules hinc) (libraries core))
(test (name t_hinc) (modules t_hinc) (libraries hinc))
(test (name t_hsel) (modules t_hsel) (libraries (select t_hsel.ml from (core -> t_hsel.real.ml) (-> t_hsel.missing.ml))))
DUNE
  cat >"$repo/t8/hlp.ml" <<'ML'
(* A header naming Core.H.used, as a doc would. *)
open Stdlib
open! Core

(** plain reads nothing of Core.H. *)
let plain () = ()

let names_core () = ignore Core.H.used

let indirect () = names_core ()
ML
  printf 'let () = Hlp.plain ()\n' >"$repo/t8/t_hplain.ml"
  printf 'let () = Hlp.names_core ()\n' >"$repo/t8/t_hnamed.ml"
  printf 'let () = Hlp.indirect ()\n' >"$repo/t8/t_hindirect.ml"
  # `H` with no `Core` beside it is some other H: a local module, or Stdlib's.
  printf 'module H = struct let used = 0 end\nlet () = ignore H.used\n' >"$repo/t8/t_hbare.ml"
  # Nor beside a token of Core that neither qualifies nor opens it (gh-ocannl-1765) ...
  printf 'module H = struct let used = 0 end\nlet () = ignore H.used (* not from Core *)\n' >"$repo/t8/t_hbeside.ml"
  # ... while an open of Core makes a bare `H` Core's.
  printf 'open Core\nlet () = ignore H.used\n' >"$repo/t8/t_hopen.ml"
  # ... and so does a qualifier through a local alias of Core.
  printf 'module C = Core\nlet () = ignore C.H.used\n' >"$repo/t8/t_halias.ml"
  # ... or an open of that alias (local review).
  printf 'module C = Core\nlet () = let open C in ignore H.used\n' >"$repo/t8/t_haopen.ml"
  # A generated module whose rule is no copy is not read through its one input (local review).
  printf 'let () = ()\n' >"$repo/t8/t_hgen.src.ml"
  # An open in the helper opens nothing in the test calling it: this `H` is its own.
  printf 'module H = struct let used = 0 end\nlet () = Hlp.plain (); ignore H.used\n' >"$repo/t8/t_hcross.ml"
  # `include` re-exports Helper's members as Hinc's own: `Hinc.used` names Helper (local review).
  printf 'include Core.H\n' >"$repo/t8/hinc.ml"
  printf 'let () = ignore Hinc.used\n' >"$repo/t8/t_hinc.ml"
  # A `(select)` module is read through every arm (gh-ocannl-1765), the one not built here too.
  printf 'let () = ()\n' >"$repo/t8/t_hsel.real.ml"
  printf 'let () = ignore Core.H.used\n' >"$repo/t8/t_hsel.missing.ml"
  git_q add -A t8 && git_q commit -qm helper-library || exit 2
}

if harness_leg 'helper definitions'; then
  # A private helper library counts by the definitions a test names (gh-ocannl-1651), its
  # comments left out; a re-export counts only beside the module making it.
  helper_fixture
  affected --referenced-only --files lib/helper.ml
  expect 'a helper library names a module through the definitions a test names, never its comments; a bare re-export name is no mention' 0 \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen \
    @t5/slow-usesh @t8/runtest-t_hindirect @t8/runtest-t_hnamed @t8/runtest-t_hopen @t8/runtest-t_halias @t8/runtest-t_hsel @t8/runtest-t_hinc @t8/runtest-t_haopen @t8/runtest-t_hgen
  if grep -qx 'affected: link-only: @t8/runtest-t_hplain' "$err" && grep -qx 'affected: link-only: @t8/runtest-t_hbare' "$err" \
      && grep -qx 'affected: link-only: @t8/runtest-t_hbeside' "$err" && grep -qx 'affected: link-only: @t8/runtest-t_hcross' "$err"; then
    report 0 'what names neither is ranked link-only, and named on stderr'
  else report 1 'what names neither is ranked link-only, and named on stderr' "$(cat "$err")"; fi
  hctl() { # LABEL FROM TO ALIAS: the mutant must list ALIAS under --referenced-only
    local mutant=$repo/tools/affected-mutant.sh
    if FROM=$2 TO=$3 perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
        "$repo/tools/affected-tests.sh" >"$mutant"; then
      rc=0
      (cd "$TMP" && bash "$mutant" --referenced-only --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
      if [ "$rc" = 0 ] && grep -qx "$4" "$out"; then report 0 "negative control: $1"
      else report 1 "negative control: $1" "exit $rc; $(cat "$out")"; fi
    else report 1 "negative control: $1" "the mutation anchor does not occur exactly once"; fi
    rm -f "$mutant"
  }
  hctl 'the definitions named are what leave an unnamed one out' \
    'next if $done{"$f\0$k"} || ($c->{names} && !(@by = grep { $tok{$_} } @{$c->{names}}));' 'next if $done{"$f\0$k"};' @t8/runtest-t_hplain
  hctl 'blanking comments is what keeps a doc comment from naming a module' \
    '    my $code = code_of($s);' '    my $code = $s;' @t8/runtest-t_hplain
  hctl 'naming the re-exporting module is what a re-exported name needs' \
    'next if $via && !@by;' '' @t8/runtest-t_hbare
  hctl 'a qualified path or an open, not any token of the re-exporting module, is what a re-exported name needs' \
    'my @by = $via ? grep { $through->($_, $n) } sort keys %$via : ();' 'my @by = $via ? grep { $tok->{$_} } sort keys %$via : ();' @t8/runtest-t_hbeside
  git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/t8" || exit 2
fi

if harness_leg 'explain'; then
  # `--why ALIAS` (gh-ocannl-1711) names why an alias is listed: the changed path, the chain of
  # libraries to it, and the token that made it referenced -- through a helper definition, a
  # re-export, or none (link-only). Each line is pinned whole, so an explanation that loses its
  # chain, its token or its file fails.
  helper_fixture
  why_has() { # LABEL LINE... : each line occurs whole on stderr
    local label=$1 l miss=
    shift
    for l in "$@"; do grep -qxF -- "$l" "$err" || miss="$miss [$l]"; done
    if [ "$rc" = 0 ] && [ -z "$miss" ]; then report 0 "$label"
    else report 1 "$label" "exit $rc; missing:$miss; stderr: $(grep '^affected: why ' "$err" | tr '\n' '|')"; fi
  }
  printf 'let used = 2\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected HEAD --why @t8/runtest-t_hindirect --why t8/runtest-t_hplain --why @t2/runtest-gen-v1 \
    --why @t2/runtest-scan --why @t3/runtest-neg --why @t9/runtest-none --why @t8/runtest-t_hopen --why @t8/runtest-t_hbeside \
    --why @t8/runtest-t_hsel --why @t8/runtest-t_hinc
  cp "$out" "$TMP/why-out" || exit 2
  why_has 'helper: the changed definition is named through the helper definitions a test names, back to its own source' \
    'affected: why @t8/runtest-t_hindirect: REFERENCED: printed first' \
    'affected: why @t8/runtest-t_hindirect: its owner tests t_hindirect (t8) -> library hlp (t8) -> library core (lib), which lib/helper.ml (its module Helper) reached' \
    'affected: why @t8/runtest-t_hindirect:   referenced: it names Helper: `H`, re-exporting Helper from Core: `Core.H` at t8/hlp.ml (definition `names_core`) <- `names_core` at t8/hlp.ml (definition `indirect`) <- `indirect` at t8/t_hindirect.ml (its own source); and its changed definition(s): `used` at t8/hlp.ml (definition `names_core`) <- `names_core` at t8/hlp.ml (definition `indirect`) <- `indirect` at t8/t_hindirect.ml (its own source)'
  why_has 'link-only: what names neither the module nor a re-export of it says so (the leading @ optional)' \
    'affected: why @t8/runtest-t_hplain: LINK-ONLY: printed last; --referenced-only leaves it off' \
    'affected: why @t8/runtest-t_hplain: its owner tests t_hplain (t8) -> library hlp (t8) -> library core (lib), which lib/helper.ml (its module Helper) reached' \
    'affected: why @t8/runtest-t_hplain:   link-only on its account: its code (its sources, its project ppx'"'"'s, the helper definitions it names) names neither Helper nor a re-export of it through the module making it (H through Core, Hinc through Hinc)'
  why_has 're-export: a test naming the module through a re-export, but no changed definition, is link-only and told so' \
    'affected: why @t2/runtest-gen-v1: LINK-ONLY: printed last; --referenced-only leaves it off' \
    'affected: why @t2/runtest-gen-v1:   link-only on its account: its code (its sources, its project ppx'"'"'s, the helper definitions it names) names Helper (`H`, re-exporting Helper from Core: `Core.H` at t2/gen.ml (its own source)) but none of its changed definitions (used)'
  why_has 're-export: an open of the re-exporting module is the reference, any other token of it is not (gh-ocannl-1765)' \
    'affected: why @t8/runtest-t_hopen: REFERENCED: printed first' \
    'affected: why @t8/runtest-t_hopen:   referenced: it names Helper: `H`, re-exporting Helper from Core: `open Core: H` at t8/t_hopen.ml (its own source); and its changed definition(s): `used` at t8/t_hopen.ml (its own source)' \
    'affected: why @t8/runtest-t_hbeside: LINK-ONLY: printed last; --referenced-only leaves it off' \
    'affected: why @t8/runtest-t_hbeside:   link-only on its account: its code (its sources, its project ppx'"'"'s, the helper definitions it names) names neither Helper nor a re-export of it through the module making it (H through Core, Hinc through Hinc)'
  why_has 'select: a (select) module is read through every arm, the one this build did not choose included (gh-ocannl-1765)' \
    'affected: why @t8/runtest-t_hsel: REFERENCED: printed first' \
    'affected: why @t8/runtest-t_hsel:   referenced: it names Helper: `H`, re-exporting Helper from Core: `Core.H` at t8/t_hsel.missing.ml (its own source); and its changed definition(s): `used` at t8/t_hsel.missing.ml (its own source)'
  # Hinc's own `include Core.H` names H through Core too; the second reason is the include's.
  why_has 'include: naming the including module is a reference to what it includes (local review)' \
    'affected: why @t8/runtest-t_hinc:   referenced: it names Helper: `H`, re-exporting Helper from Core: `Core.H` at t8/hinc.ml (a top-level form, counted whenever Hinc is named) <- `Hinc` at t8/t_hinc.ml (its own source); `Hinc`, re-exporting Helper from Hinc: `Hinc` at t8/t_hinc.ml (its own source); and its changed definition(s): `used` at t8/t_hinc.ml (its own source)'
  why_has 'a scan: its own rule reads the changed path; an alias nothing reaches, and a name dune does not list, say so' \
    'affected: why @t2/runtest-scan: REFERENCED: printed first' \
    'affected: why @t2/runtest-scan: the rule of its own target t2/scan.actual reads lib/helper.ml' \
    'affected: why @t3/runtest-neg: NOT LISTED: no changed path reaches it' \
    'affected: why @t9/runtest-none: no per-test alias dune lists has this name (`dune show aliases DIR` lists them)'
  affected HEAD
  if [ "$rc" = 0 ] && cmp -s "$out" "$TMP/why-out" && ! grep -q '^affected: why ' "$err"; then
    report 0 '--why changes nothing on stdout, and without it nothing is explained'
  else report 1 '--why changes nothing on stdout, and without it nothing is explained' "exit $rc; $(diff "$TMP/why-out" "$out")"; fi
  # The whole module changed: the re-export alone now names the changed part.
  affected --files lib/helper.ml --why @t2/runtest-gen-v1
  why_has 're-export: with the whole module changed, the re-export beside its module is the reference' \
    'affected: why @t2/runtest-gen-v1: REFERENCED: printed first' \
    'affected: why @t2/runtest-gen-v1:   referenced: it names Helper: `H`, re-exporting Helper from Core: `Core.H` at t2/gen.ml (its own source); and Helper changed as a whole'
  # ... and, its code known, it is link-only for a definition no arm names (it was "code not all
  # known", so first, before).
  printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
  affected HEAD --why @t8/runtest-t_hsel --why @t8/runtest-t_hgen
  why_has 'select: with every arm read, a test naming no changed definition is link-only' \
    'affected: why @t8/runtest-t_hsel: LINK-ONLY: printed last; --referenced-only leaves it off'
  why_has 'a generated module whose rule is no copy stays code not all known, whatever its one input names' \
    'affected: why @t8/runtest-t_hgen: REFERENCED: printed first' \
    'affected: why @t8/runtest-t_hgen:   referenced: its code is not all known: it has a generated module (a (select), a rule'"'"'s output)'
  printf 'let used = 2\nlet unused = 2\n' >"$repo/lib/helper.ml"
  # `--why all` explains every listed alias, and its verdicts are the tiers the list was ranked
  # by: REFERENCED or BLIND for each alias printed first, LINK-ONLY for each one printed last.
  affected HEAD --why all
  sed -n 's/^affected: link-only: //p' "$err" | sort >"$TMP/why-link"
  grep -vxF -f "$TMP/why-link" "$out" | sort >"$TMP/why-first"
  sed -n 's/^affected: why \(@[^:]*\): LINK-ONLY: .*/\1/p' "$err" | sort >"$TMP/why-v-link"
  sed -nE 's/^affected: why (@[^:]*): (REFERENCED|BLIND): .*/\1/p' "$err" | sort >"$TMP/why-v-first"
  if [ "$rc" = 0 ] && [ -s "$TMP/why-link" ] && [ -s "$TMP/why-first" ] &&
     cmp -s "$TMP/why-link" "$TMP/why-v-link" && cmp -s "$TMP/why-first" "$TMP/why-v-first"; then
    report 0 '--why all: one verdict per listed alias, agreeing with its rank'
  else report 1 '--why all: one verdict per listed alias, agreeing with its rank' \
    "exit $rc; link [$(tr '\n' ' ' <"$TMP/why-link")] vs [$(tr '\n' ' ' <"$TMP/why-v-link")]; first [$(tr '\n' ' ' <"$TMP/why-first")] vs [$(tr '\n' ' ' <"$TMP/why-v-first")]"; fi
  affected HEAD --why
  if [ "$rc" = 2 ] && grep -q -- '--why needs an alias, or all' "$err"; then report 0 '--why with no alias is refused'
  else report 1 '--why with no alias is refused' "exit $rc; $(cat "$err")"; fi
  git_q checkout -q -- lib/helper.ml && git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/t8" || exit 2
fi

if harness_leg 'dune fragment'; then
  # t6 includes extra.inc: its stanzas are not read here, so the directory's
  # aliases stay blind, and an edit to the fragment reaches like one to t6/dune.
  affected --files t6/extra.inc
  # shellcheck disable=SC2086
  expect 'an included dune fragment reaches every directory including it, as their dune files would' 0 \
    @t6/runtest-dyn @t7/runtest-g7 $opaque_all
fi

if harness_leg 'git forms'; then
  printf 'let x = 1\n' >"$repo/lib/leaf.ml"
  affected HEAD
  # The fixture does not ignore _build/, so this also pins that it is never a change.
  expect 'the working-tree form takes in an uncommitted edit' 0 \
    @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t2/runtest-scan
  git_q commit -qam leaf || exit 2
  printf 'let y = 2\n' >"$repo/lib/core.ml"
  printf 'let () = ()\n' >"$repo/t1/new_helper.ml"
  affected HEAD~1..HEAD
  expect 'a range form reads only the commits named' 0 \
    @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t2/runtest-scan
  affected HEAD -- t1
  expect 'the working-tree form takes in an untracked file, limited by a pathspec' 0 \
    @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb
  rm -f "$repo/t1/new_helper.ml" && git_q checkout -q -- lib/core.ml || exit 2
  affected
  if [ "$rc" = 2 ] && grep -q 'name the base' "$err" && [ ! -s "$out" ]; then
    report 0 'no revision is refused: the base is named, never guessed from a remote'
  else report 1 'no revision is refused: the base is named, never guessed from a remote' "exit $rc; $(cat "$err")"; fi
  affected HEAD HEAD~1
  if [ "$rc" = 2 ] && grep -q 'one revision argument at most' "$err"; then
    report 0 'two revision arguments are refused'
  else report 1 'two revision arguments are refused' "exit $rc; $(cat "$err")"; fi
  # A branch whose base moved on (gh-ocannl-1379): the base gains a core edit
  # after the branch forks, and the branch commits a leaf edit.
  trunk=$(git_q rev-parse --abbrev-ref HEAD) || exit 2
  git_q checkout -qb side || exit 2
  printf 'let z = 3\n' >"$repo/lib/leaf.ml"
  git_q commit -qam side-leaf || exit 2
  git_q checkout -q "$trunk" || exit 2
  printf 'let y = 2\n' >"$repo/lib/core.ml"
  git_q commit -qam trunk-core || exit 2
  git_q checkout -q side || exit 2
  affected --since "$trunk"
  expect '--since compares with the merge-base: the branch'"'"'s own change, not what its base gained since' 0 \
    @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t2/runtest-scan
  affected "$trunk"
  if [ "$rc" = 0 ] && grep -q 'is not an ancestor of HEAD' "$err" && grep -qx @t1/runtest-t_core "$out"; then
    report 0 'a REV that is not an ancestor of HEAD is warned about (and its later commits count)'
  else report 1 'a REV that is not an ancestor of HEAD is warned about (and its later commits count)' "exit $rc; $(cat "$out" "$err")"; fi
  affected --since "$trunk" HEAD
  if [ "$rc" = 2 ] && grep -q 'exclusive' "$err"; then
    report 0 '--since and a revision argument are refused together'
  else report 1 '--since and a revision argument are refused together' "exit $rc; $(cat "$err")"; fi
  git_q checkout -q "$trunk" || exit 2
fi

# A dune whose answer has a form the reader does not accept stops the run: the
# reader is an allowlist, so an unknown line is never read as "no aliases".
mkdir -p "$TMP/shim" || exit 2
REAL_DUNE=$(command -v dune)
cat >"$TMP/shim/dune" <<SHIM
#!/usr/bin/env bash
"$REAL_DUNE" "\$@"
rc=\$?
if [ "\${1:-}" = show ] && [ "\${2:-}" = "\${SHIM_EXTRA_FOR:-}" ]; then echo 'Warning: something new' >&2; fi
# Mid-run, what a run starting now would find: the worktree lock's state.
if [ "\${1:-}" = show ] && [ -n "\${SHIM_LOCK_PROBE:-}" ]; then "$repo/tools/test-run.sh" lock-status >>"\$SHIM_LOCK_PROBE" 2>&1; fi
# ... and the previous version's in-tree lock, as a launcher of that version would find it.
if [ "\${1:-}" = show ] && [ -n "\${SHIM_LEGACY_PROBE:-}" ]; then
  legacy="$repo/.test-run.lock"
  perl -e 'use Fcntl ":flock"; open my \$f, "+<", \$ARGV[0] or exit 2; print((flock(\$f, LOCK_EX | LOCK_NB) ? "free" : "held"), "\n")' "\$legacy" >>"\$SHIM_LEGACY_PROBE" 2>&1
fi
exit \$rc
SHIM
chmod +x "$TMP/shim/dune"
if harness_leg 'reader refusal'; then
  for what in aliases targets; do
    rc=0
    (cd "$TMP" && SHIM_EXTRA_FOR=$what PATH="$TMP/shim:$PATH" "$repo/tools/affected-tests.sh" --files lib/helper.ml) \
      >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -q "unexpected line from dune show $what: Warning: something new" "$err"; then
      report 0 "an unknown line from dune show $what stops the run, printing nothing"
    else report 1 "an unknown line from dune show $what stops the run, printing nothing" "exit $rc; $(cat "$out" "$err")"; fi
  done
fi

if harness_leg 'runner entry'; then
  rc=0
  (cd "$TMP" && "$repo/tools/test-run.sh" affected --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
  # shellcheck disable=SC2086
  expect 'tools/test-run.sh affected runs the helper at the repository root' 0 $reached_by_core
  mkdir -p "$TMP/decoy/tools" || exit 2
  rc=0
  (cd "$repo" && CDPATH="$TMP/decoy" tools/test-run.sh affected --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
  # shellcheck disable=SC2086
  expect 'tools/test-run.sh affected, run by a relative path, ignores CDPATH' 0 $reached_by_core
  rc=0
  (cd "$TMP" && SHIM_LOCK_PROBE=$TMP/lock-probe PATH="$TMP/shim:$PATH" "$repo/tools/test-run.sh" affected --files lib/helper.ml) \
    >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 0 ] && [ "$(sort -u "$TMP/lock-probe")" = held ]; then
    report 0 'the worktree lock is held while affected'"'"'s dune runs, so a run starting then refuses'
  else report 1 'the worktree lock is held while affected'"'"'s dune runs, so a run starting then refuses' "exit $rc; probe: $(cat "$TMP/lock-probe" 2>&1)"; fi
  : >"$repo/.test-run.lock"
  rc=0
  (cd "$TMP" && SHIM_LEGACY_PROBE=$TMP/legacy-probe PATH="$TMP/shim:$PATH" "$repo/tools/test-run.sh" affected --files lib/helper.ml) \
    >"$out" 2>"$err" || rc=$?
  rm -f "$repo/.test-run.lock"
  if [ "$rc" = 0 ] && [ "$(sort -u "$TMP/legacy-probe")" = held ]; then
    report 0 'the previous version'"'"'s in-tree lock, where it exists, is held while affected'"'"'s dune runs'
  else report 1 'the previous version'"'"'s in-tree lock, where it exists, is held while affected'"'"'s dune runs' "exit $rc; probe: $(cat "$TMP/legacy-probe" 2>&1)"; fi
  lock=$("$repo/tools/test-run.sh" paths lock) || exit 2
  mkdir -p "$(dirname "$lock")" || exit 2
  # Hold the worktree's run lock as a run would, until the release fifo is
  # written. The holder answers on the ready fifo once it holds the lock (or
  # failed to), and the read blocks until it does: no timed polling.
  mkfifo "$TMP/release" "$TMP/ready" || exit 2
  perl -e 'use Fcntl ":flock";
    my $fh; my $ok = open($fh, ">>", $ARGV[0]) && flock($fh, LOCK_EX);
    open my $ready, ">", $ARGV[1] or exit 1; print $ready ($ok ? "held\n" : "failed\n"); close $ready;
    exit 1 unless $ok;
    open my $r, "<", $ARGV[2]; <$r>;' "$lock" "$TMP/ready" "$TMP/release" &
  holder=$!
  state=
  read -r state <"$TMP/ready" || state=
  rc=0
  if [ "$state" = held ]; then
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
    echo >"$TMP/release"
  else
    rc=holder-$state
  fi
  wait "$holder"
  if [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -q 'a run is active in this worktree' "$err"; then
    report 0 'tools/test-run.sh affected refuses while a run holds the worktree'
  else report 1 'tools/test-run.sh affected refuses while a run holds the worktree' "exit $rc; $(cat "$out" "$err")"; fi
fi

# Negative controls: each mutant drops one source of reach, and the leg that
# pins it must see the list change. Without them a leg could pass on a
# fixture too small to exercise the path it names.
control() { # LABEL FROM TO ARGS... -- WANT...
  local label=$1 from=$2 to=$3 mutant=$repo/tools/affected-mutant.sh
  shift 3
  local args=()
  while [ "$1" != -- ]; do args+=("$1"); shift; done
  shift
  if ! FROM=$from TO=$to perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    report 1 "negative control: $label" "the mutation anchor does not occur exactly once"
    return
  fi
  rc=0
  (cd "$TMP" && bash "$mutant" "${args[@]}") >"$out" 2>"$err" || rc=$?
  printf '%s\n' "$@" | sort >"$TMP/want"
  sort "$out" >"$TMP/got"
  if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$TMP/got"; then
    report 1 "negative control: $label" "the mutant printed the shipping list"
  else
    report 0 "negative control: $label"
  fi
  rm -f "$mutant"
}
if harness_leg 'negative controls'; then
  # shellcheck disable=SC2086
  control 'a rule'"'"'s glob over sources is what lists the scan' \
    'push @g, [$gd, predicate(field($spec, "predicate"))];' '' \
    --files lib/helper.ml -- $reached_by_core
  # shellcheck disable=SC2086
  control 'the executable a target'"'"'s rule runs is what owns runtest-scenario_x' \
    'map { %{$runs{$_} // {}} } @{$a->{own}}' '()' \
    --files lib/helper.ml -- $reached_by_core
  control 'an undescribed tests module reaches its directory'"'"'s tests stanzas' \
    '@ix = @named ? @named : @tests;' '@ix = @named;' \
    --files t1/shared.ml -- @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb
  control 'external-lib-deps'"'"' edges are what carry a ppx rewriter to a described library' \
    'push @edges, [$i, field($v, "internal_deps")];' 'push @edges, [$i, field($v, "internal_deps")] if $kind eq "tests";' \
    --files lib/myppx.ml -- @t1/runtest-m_one @t2/runtest-scan
  control 'a generated module file is what carries an .mll to its stanza' \
    'push @{$generated{$rel}}, $i unless -e $rel;' '' \
    --files lib/lx.mll -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  # shellcheck disable=SC2086
  control 'an opaque action is what lists the aliases for a changed config' \
    'my $blind = $unknown_touched || $opaque_listed;' 'my $blind = $unknown_touched;' \
    --files t3/plain.cfg -- @t3/runtest-t_plain $opaque_all
  control 'a stub archive is what carries a C stub to its library' \
    '$obj_libs{$o}{$_} = 1 for @libs' '1' \
    --files lib/stubbed_c.c -- @t3/runtest-t_stub
  control 'following generated inputs is what reaches a copied file' \
    '$v{$rel} = 1 if !-e $rel' '$v{$rel} = 1 if 0 && !-e $rel' \
    --files cfg/base.cfg -- @t3/runtest-cfgread @t3/runtest-t_plain $opaque_all
  control 'reading the files a changed dune file makes is what reaches a consumer elsewhere' \
    '$n_made++ unless $read_by{$p}{$a->{name}}++;' '' \
    --files t1/dune -- @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t3/runtest-uses
  # shellcheck disable=SC2086
  control 'a generator run by a stub object'"'"'s inputs is what reaches the executable linking it' \
    'for my $w (obj_owners($o)) { push @{$st[$w]{to}}, keys %{$runs{$o}}; push @{$st[$w]{gen_to}}, keys %{$runs{$o}} }' '' \
    --files lib/helper.ml -- $reached_by_core
  control 'UNIT_ targets are what tie an alias to the executable making them' \
    'my @near = grep { $_ !~ $artifact && index($_, "${unit}_") == 0 } @{$targets->{$d} // []};' 'my @near;' \
    --files t4/pre.ml -- @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre
  control 'a compiled-object glob is what reaches the stanza it reads' \
    'if ($where eq "In_build_dir" and my @o = compiled_owners("$path/")) { $r{$_} = 1 for @o; next }' '' \
    --files t4/inert.ml -- @t4/runtest-objs
  control 'the alias fixpoint is what lists an alias needing a listed one' \
    'next unless $best > $now;' 'next;' \
    --files t4/t_gate.ml -- @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre
  # shellcheck disable=SC2086
  control 'the guess tier'"'"'s rank is what lists an alias needing one listed blind' \
    'my %rank = (main => 3, guess => 2, link => 1);' 'my %rank = (main => 3, link => 1);' \
    --files docs/other.md -- $opaque_all
  # shellcheck disable=SC2086
  control 'the ranking is what leaves a link-only alias off --referenced-only' \
    'return 0 unless $rank_links;' 'return 0;' \
    --referenced-only --files lib/helper.ml -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  control 'a re-export is what ranks a test naming only the re-exported name' \
    '$as{$_}{$as_name}{$x} = 1 for grep { $_ ne $as_name } @names;' '1;' \
    --referenced-only --files lib/helper.ml -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  # shellcheck disable=SC2086
  control 'a rule-made target beside a golden is what makes a golden pair' \
    'grep { !/\.expected$/ && !-e in_dir($d, $_) } @mine, @near' '1' \
    --files cfg/extra.py -- $opaque_all
  # shellcheck disable=SC2086
  control 'skipping an off-build stanza'"'"'s unresolved edge is what lets the run finish' \
    'elsif ($st[$i]{undescribed}) {' 'elsif (0) {' \
    --files lib/helper.ml -- $reached_by_core
  # The git-forms leg left the trunk checked out, one core edit past the fork
  # point of its `side` branch, whose own commit edits leaf.
  if git_q rev-parse -q --verify side >/dev/null; then
    # shellcheck disable=SC2086
    control 'the merge-base is what --since compares with' \
      'revs=("$base")' 'revs=("$since")' \
      --since side -- $reached_by_core
  fi
  control 'an executable target is what finds a test stanza naming no library' \
    'push @st, { kind => "tests", dir => $d, names => [$x]' 'next; push @st, { kind => "tests", dir => $d, names => [$x]' \
    --files t3/t_plain.ml -- @t3/runtest-t_plain
fi

# The harness legs (gh-ocannl-1900): what a change reaches among the shell
# harnesses, read from their own text and header declarations, never from
# dune. A dune-free fixture of its own: a manifest, three harnesses, the
# scripts they name and load, and the data one declares. Every run there has a
# `dune` on PATH that records being called, so `--harnesses` is shown to ask
# dune nothing.
hrepo=$TMP/hproj
mkdir -p "$hrepo/tools/pkg" "$hrepo/data" "$TMP/nodune" || exit 2
cp "$HERE/affected-tests.sh" "$hrepo/tools/" || exit 2
printf '#!/bin/sh\n: >"%s/dune-called"\nexit 97\n' "$TMP" >"$TMP/nodune/dune" && chmod +x "$TMP/nodune/dune" || exit 2
cat >"$hrepo/tools/test-harnesses.sh" <<'STUB'
#!/usr/bin/env bash
[ "$*" = --list ] || exit 2
printf '%s\n' tools/test-a.sh tools/test-b.sh tools/test-c.sh
STUB
cat >"$hrepo/tools/test-a.sh" <<'H'
#!/usr/bin/env bash
# Harness a: drives sub.sh.
"$(dirname "$0")/sub.sh" run
H
cat >"$hrepo/tools/sub.sh" <<'H'
#!/usr/bin/env bash
. "$(dirname "$0")/lib.sh"
# shellcheck source=dyn.sh
. "$DYN"
true || source tools/late.sh
tools/other.sh
H
cat >"$hrepo/tools/test-b.sh" <<'H'
#!/usr/bin/env bash
# Harness b. tools/sub.sh is named only in this comment.
my-sub.sh; x_sub.sh.bak
python3 tools/drive.py
./gen.exe
H
cat >"$hrepo/tools/drive.py" <<'H'
import helper
from pkg import mod
from . import sib
H
# Its declarations are printed, not written out here: a line of this file led
# by the marker would be a declaration of this harness, out of its header.
mark='# affected-reads:'
printf '%s\n' '#!/usr/bin/env bash' '# Harness c.' "$mark data/*.tbl --only '^(1|3):'" "$mark data/two.tbl --only '^5 two:'" \
  "$mark data/all.cfg" 'cat data/three.tbl' '. tools/clib.sh' >"$hrepo/tools/test-c.sh" || exit 2
for f in tools/lib.sh tools/dyn.sh tools/late.sh tools/other.sh tools/clib.sh tools/helper.py tools/pkg/mod.py \
  tools/sib.py tools/gen.ml data/one.tbl data/two.tbl data/three.tbl data/all.cfg data/x.txt; do : >"$hrepo/$f"; done
git -C "$hrepo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false init -q &&
  git -C "$hrepo" add -A && git -C "$hrepo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qm fixture ||
  exit 2
cp "$hrepo/tools/test-c.sh" "$TMP/test-c.sh.orig" || exit 2
haffected() { # ARGS... -> $out, $err, $rc, run in the fixture with the recording dune first on PATH
  rc=0
  rm -f "$TMP/dune-called"
  (cd "$TMP" && PATH=$TMP/nodune:$PATH "${HSUBJECT:-$hrepo/tools/affected-tests.sh}" "$@") >"$out" 2>"$err" || rc=$?
}
hexpect() { # LABEL RC LINE...: the exact stdout, in order, the exit status, and dune not called
  local label=$1 want_rc=$2
  shift 2
  if [ $# -gt 0 ]; then printf '%s\n' "$@" >"$TMP/want"; else : >"$TMP/want"; fi
  if [ "$rc" = "$want_rc" ] && cmp -s "$TMP/want" "$out" && [ ! -e "$TMP/dune-called" ]; then
    report 0 "$label"
  else
    report 1 "$label" "exit $rc (want $want_rc)$([ -e "$TMP/dune-called" ] && echo ', dune called'); got [$(tr '\n' '|' <"$out")] want [$(tr '\n' '|' <"$TMP/want")]; stderr: $(tr '\n' '|' <"$err")"
  fi
}
herr() { # LABEL FIXED-TEXT: stderr carries it
  if grep -qF -- "$2" "$err"; then report 0 "$1"
  else report 1 "$1" "no stderr line carries '$2': $(tr '\n' '|' <"$err")"; fi
}
hc_restore() { cp "$TMP/test-c.sh.orig" "$hrepo/tools/test-c.sh"; }

if harness_leg 'harness names and loads'; then
  haffected --harnesses --files tools/sub.sh
  hexpect 'a script a harness names reaches it whole, not one naming it in a comment or in a longer name' 0 tools/test-a.sh
  herr 'and says why' 'tools/sub.sh: read by 1 harness(es): tools/test-a.sh (every leg: it names tools/sub.sh)'
  # No alias list is printed, so no verdict on one (gh-ocannl-1921).
  if [ "$rc" = 0 ] && ! grep -q 'verdict=' "$err"; then report 0 '--harnesses prints no verdict trailer'
  else report 1 '--harnesses prints no verdict trailer' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  haffected --harnesses --files tools/test-a.sh
  hexpect 'a harness reaches itself' 0 tools/test-a.sh
  haffected --harnesses --files tools/lib.sh
  hexpect 'what a named script sources is reached' 0 tools/test-a.sh
  herr 'through it' 'tools/test-a.sh (every leg: it names tools/sub.sh, which loads tools/lib.sh)'
  haffected --harnesses --files tools/dyn.sh
  hexpect 'a shellcheck source directive loads its script' 0 tools/test-a.sh
  haffected --harnesses --files tools/late.sh
  hexpect 'a source command after an operator loads its script' 0 tools/test-a.sh
  haffected --harnesses --files tools/other.sh
  hexpect 'what a named script only names is not followed' 0
  herr 'and a script no harness reaches is named' 'tools/other.sh: no harness reads it'
  for f in tools/helper.py tools/pkg/mod.py tools/sib.py; do
    haffected --harnesses --files "$f"
    hexpect "a Python import loads $f" 0 tools/test-b.sh
  done
  haffected --harnesses --files tools/gen.ml
  hexpect 'an X.exe names its X.ml' 0 tools/test-b.sh
  haffected --harnesses --files tools/sub.sh tools/helper.py
  hexpect 'several harnesses, sorted' 0 tools/test-a.sh tools/test-b.sh
  haffected --harnesses --files data/x.txt
  hexpect 'a file no harness names reaches none' 0
fi

if harness_leg 'harness declarations'; then
  haffected --harnesses --files data/one.tbl
  hexpect 'a declared path its harness does not name reaches the declared legs' 0 "tools/test-c.sh --only '^(1|3):'"
  herr 'and says which line' "tools/test-c.sh (legs '^(1|3):': its affected-reads line 3)"
  haffected --harnesses --files data/two.tbl
  hexpect 'two declarations matching one path select both, a blank inside a selector kept' 0 "tools/test-c.sh --only '(^(1|3):)|(^5 two:)'"
  haffected --harnesses --files data/one.tbl data/two.tbl
  hexpect 'two paths selecting different legs select both' 0 "tools/test-c.sh --only '(^(1|3):)|(^5 two:)'"
  haffected --harnesses --files data/three.tbl
  hexpect 'a declaration never narrows what its harness code names' 0 tools/test-c.sh
  herr 'which the text reaches' 'tools/test-c.sh (every leg: it names data/three.tbl)'
  haffected --harnesses --files data/one.tbl data/all.cfg
  hexpect 'a declaration without --only selects every leg' 0 tools/test-c.sh
  haffected --harnesses --files data/one.tbl tools/clib.sh
  hexpect 'a path its program reaches selects every leg' 0 tools/test-c.sh
  printf 'x\n' >>"$hrepo/data/one.tbl"
  haffected --harnesses HEAD
  hexpect 'the git forms read the same changed paths' 0 "tools/test-c.sh --only '^(1|3):'"
  git -C "$hrepo" checkout -q -- data/one.tbl
fi

if harness_leg 'harness refusals'; then
  tab=$'\t'
  for bad in "# affected-reads:  data/x.txt" "# affected-reads: data/x.txt --only \"^1:\"" "#affected-reads: data/x.txt" \
    "# affected-reads: ../data/x.txt" "# affected-reads: --only '^1:'" "# affected-reads: data/x.txt --only '^1:${tab}x'"; do
    { head -n 2 "$TMP/test-c.sh.orig"; printf '%s\n' "$bad"; tail -n +3 "$TMP/test-c.sh.orig"; } >"$hrepo/tools/test-c.sh"
    haffected --harnesses --files data/x.txt
    hexpect "a malformed declaration stops the run: $bad" 2
  done
  herr 'naming its line' 'tools/test-c.sh:3:'
  { cat "$TMP/test-c.sh.orig"; printf '%s\n' "# affected-reads: data/x.txt"; } >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files data/x.txt
  hexpect 'a declaration after the header stops the run' 2
  herr 'saying so' 'tools/test-c.sh:8: an affected-reads declaration outside the header'
  { head -n 2 "$TMP/test-c.sh.orig"; printf '%s\n' "# affected-reads: data/x.txt --only '(('"; tail -n +3 "$TMP/test-c.sh.orig"; } >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files tools/other.sh
  hexpect 'an ERE bash refuses stops the run, whatever changed' 2
  herr 'naming it' "tools/test-c.sh:3: --only '((' is not an extended regular expression bash accepts"
  { head -n 2 "$TMP/test-c.sh.orig"; printf '%s\n' "# affected-reads: data/*.none --only '^1:'"; tail -n +3 "$TMP/test-c.sh.orig"; } >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files data/x.txt
  hexpect 'a glob matching no file is no refusal' 0
  herr 'but a warning' 'affected: warning: tools/test-c.sh:3: data/*.none matches no file here'
  hc_restore
  for opts in '--referenced-only' '--why all' '--harnesses=1'; do
    # shellcheck disable=SC2086
    haffected --harnesses $opts --files data/x.txt
    hexpect "--harnesses refuses $opts" 2
  done
  mv "$hrepo/tools/test-harnesses.sh" "$TMP/test-harnesses.sh.aside" || exit 2
  haffected --harnesses --files data/x.txt
  hexpect '--harnesses with no manifest stops the run' 2
  mv "$TMP/test-harnesses.sh.aside" "$hrepo/tools/test-harnesses.sh" || exit 2
  printf '#!/usr/bin/env bash\nprintf "%%s\\n" tools/test-a.sh tools/test-gone.sh\n' >"$hrepo/tools/test-harnesses.sh.new" &&
    cp "$hrepo/tools/test-harnesses.sh" "$TMP/test-harnesses.sh.orig" && mv "$hrepo/tools/test-harnesses.sh.new" "$hrepo/tools/test-harnesses.sh" ||
    exit 2
  haffected --harnesses --files data/x.txt
  hexpect 'a manifest naming a missing harness stops the run' 2
  cp "$TMP/test-harnesses.sh.orig" "$hrepo/tools/test-harnesses.sh" || exit 2
fi

hcontrol() { # LABEL FROM TO ARGS... -- WANT...: the mutant must not print the shipping list
  local label=$1 from=$2 to=$3 mutant=$hrepo/tools/affected-mutant.sh
  shift 3
  local args=()
  while [ "$1" != -- ]; do args+=("$1"); shift; done
  shift
  if ! FROM=$from TO=$to perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$hrepo/tools/affected-tests.sh" >"$mutant"; then
    report 1 "negative control: $label" "the mutation anchor does not occur exactly once"
    return
  fi
  chmod +x "$mutant"
  HSUBJECT=$mutant haffected "${args[@]}"
  printf '%s\n' "$@" >"$TMP/want"
  if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then report 1 "negative control: $label" "the mutant printed the shipping list"
  else report 0 "negative control: $label"; fi
  rm -f "$mutant"
}
if harness_leg 'harness negative controls'; then
  hcontrol 'a declaration is what reaches a path its harness code does not name' \
    '    } elsif (@d) {' '    } elsif (0) {' --harnesses --files data/one.tbl -- "tools/test-c.sh --only '^(1|3):'"
  hcontrol 'the text reaching a path is what keeps a declaration from narrowing it' \
    '    if (exists $reach{$h}{$p}) {' '    if (0) {' --harnesses --files data/three.tbl -- tools/test-c.sh
  hcontrol 'the loads of a named script are what reach the file it sources' \
    'next unless is_script($named[$i]);' 'next;' --harnesses --files tools/lib.sh -- tools/test-a.sh
  hcontrol 'comment lines are what name nothing' \
    'grep { !/^\s*#/ } split /\n/, raw($f)' 'split /\n/, raw($f)' --harnesses --files tools/sub.sh -- tools/test-a.sh
  hcontrol 'the boundary of a name is what keeps a longer name from naming a basename' \
    '$t =~ s{\A[^/]*/}{} or last;' '$t =~ s{\A.}{} or last;' --harnesses --files tools/sub.sh -- tools/test-a.sh
fi

# The full run, on the dune fixture: a module source no stanza lists that a
# harness declares is mapped to those legs, not UNMAPPED, and the aliases are
# what they are without the harness legs.
if harness_leg 'harness legs in the full run'; then
  mkdir -p "$repo/stubs" "$repo/stubs2" || exit 2
  cat >"$repo/tools/test-harnesses.sh" <<'STUB'
#!/usr/bin/env bash
[ "$*" = --list ] || exit 2
echo tools/test-h.sh
STUB
  printf '#!/usr/bin/env bash\n# Harness h.\n# affected-reads: stubs/* --only %s\n' "'^1:'" >"$repo/tools/test-h.sh"
  printf 'val x : int\n' >"$repo/stubs/s.mli"
  printf 'val x : int\n' >"$repo/stubs2/s.mli"
  affected --files stubs/s.mli
  if [ "$rc" = 0 ] && ! grep -q 'UNMAPPED' "$err" && grep -qxF "affected: harness: tools/test-h.sh --only '^1:'" "$err"; then
    report 0 'a module source no stanza lists that a harness declares is mapped to its legs, exit 0'
  else report 1 'a module source no stanza lists that a harness declares is mapped to its legs, exit 0' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  affected --files stubs2/s.mli
  if [ "$rc" = 3 ] && grep -q 'UNMAPPED: stubs2/s.mli' "$err" && ! grep -q 'affected: harness:' "$err"; then
    report 0 'one no harness reads stays UNMAPPED, exit 3'
  else report 1 'one no harness reads stays UNMAPPED, exit 3' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  affected --files lib/helper.ml
  # shellcheck disable=SC2086
  expect 'the aliases are unchanged by the harness legs' 0 $reached_by_core
  # The closing summary counts the harness commands (gh-ocannl-1929), and says
  # nothing of them for a change reaching none (an `.ml` no harness reads).
  hs_label='the closing summary counts the harness commands, and names none for a change reaching no harness'
  hs_none=$(grep -c 'harness' "$err")
  hs_summary() { sed -n '/^affected: [0-9]* alias(es) from /p' "$err"; }
  hs_plain=$(hs_summary)
  affected --files stubs/s.mli lib/helper.ml
  if [ "$hs_none" = 0 ] && [ -n "$hs_plain" ] && [ "$(grep -c '^affected: harness: ' "$err")" = 1 ] &&
     [ "$(hs_summary)" = "${hs_plain/ 1 changed path(s)/ 2 changed path(s)}, and 1 harness command(s), named above" ]; then
    report 0 "$hs_label"
  else report 1 "$hs_label" "an .ml alone: $hs_none harness line(s), summary [$hs_plain]; with a declared stub: $(tr '\n' '|' <"$err")"; fi
  hs_label='negative control: a summary that does not count the harness commands loses the count'
  mutant=$repo/tools/affected-mutant.sh
  if FROM='$n_harness++ if $f eq "harness-cmds";' TO='' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --files stubs/s.mli lib/helper.ml) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 0 ] && [ "$(hs_summary)" = "${hs_plain/ 1 changed path(s)/ 2 changed path(s)}" ]; then report 0 "$hs_label"
    else report 1 "$hs_label" "mutant exit $rc; $(tr '\n' '|' <"$err")"; fi
  else report 1 "$hs_label" 'the mutation anchor does not occur exactly once'; fi
  rm -f "$mutant"
  affected --harnesses --files stubs/s.mli
  if [ "$rc" = 0 ] && [ "$(cat "$out")" = "tools/test-h.sh --only '^1:'" ]; then
    report 0 '--harnesses prints the command on stdout instead of the aliases'
  else report 1 '--harnesses prints the command on stdout instead of the aliases' "exit $rc; [$(tr '\n' '|' <"$out")]"; fi
  # Through the public entry: no dune on PATH (this fixture has no
  # tools/opam-env.sh to find one), and a run holding the worktree's lock.
  nodune_path=
  IFS=: read -r -a path_dirs <<<"$PATH"
  for d in "${path_dirs[@]}"; do [ -x "$d/dune" ] || nodune_path=${nodune_path:+$nodune_path:}$d; done
  rc=0
  (cd "$TMP" && PATH=$nodune_path "$repo/tools/test-run.sh" affected --harnesses --files stubs/s.mli) >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 0 ] && [ "$(cat "$out")" = "tools/test-h.sh --only '^1:'" ]; then
    report 0 'tools/test-run.sh affected --harnesses needs no dune'
  else report 1 'tools/test-run.sh affected --harnesses needs no dune' "exit $rc; [$(tr '\n' '|' <"$out")]; $(tr '\n' '|' <"$err")"; fi
  rc=0
  (cd "$TMP" && PATH=$nodune_path "$repo/tools/test-run.sh" affected --files stubs/s.mli) >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 2 ] && grep -q 'dune not found' "$err"; then report 0 'while the alias query without one is refused'
  else report 1 'while the alias query without one is refused' "exit $rc; $(tr '\n' '|' <"$err")"; fi
  lock=$("$repo/tools/test-run.sh" paths lock) || exit 2
  mkdir -p "$(dirname "$lock")" "$TMP/hlock" || exit 2
  mkfifo "$TMP/hlock/release" "$TMP/hlock/ready" || exit 2
  perl -e 'use Fcntl ":flock";
    my $fh; my $ok = open($fh, ">>", $ARGV[0]) && flock($fh, LOCK_EX);
    open my $ready, ">", $ARGV[1] or exit 1; print $ready ($ok ? "held\n" : "failed\n"); close $ready;
    exit 1 unless $ok;
    open my $r, "<", $ARGV[2]; <$r>;' "$lock" "$TMP/hlock/ready" "$TMP/hlock/release" &
  holder=$!
  state=
  read -r state <"$TMP/hlock/ready" || state=
  rc=0 rc2=0
  if [ "$state" = held ]; then
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --harnesses --files stubs/s.mli) >"$out" 2>"$err" || rc=$?
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --files stubs/s.mli) >/dev/null 2>"$TMP/hlock/err" || rc2=$?
    echo >"$TMP/hlock/release"
  else
    rc=holder-$state
  fi
  wait "$holder"
  rm -rf "${TMP:?}/hlock"
  if [ "$rc" = 0 ] && [ "$(cat "$out")" = "tools/test-h.sh --only '^1:'" ] && [ "$rc2" = 2 ]; then
    report 0 'tools/test-run.sh affected --harnesses runs while a run holds the worktree, the alias query refusing'
  else report 1 'tools/test-run.sh affected --harnesses runs while a run holds the worktree, the alias query refusing' "exit $rc, alias query $rc2; $(tr '\n' '|' <"$err")"; fi
  # The shipping run's verdict is its exit status, so the control's is too.
  if FROM='unless (!@ix && $harness_read{$p}) {' TO='unless (0) {' \
       perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
       "$repo/tools/affected-tests.sh" >"$repo/tools/affected-mutant.sh"; then
    rc=0
    (cd "$TMP" && bash "$repo/tools/affected-mutant.sh" --files stubs/s.mli) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 3 ]; then report 0 'negative control: the harness reading it is what keeps that stub from UNMAPPED'
    else report 1 'negative control: the harness reading it is what keeps that stub from UNMAPPED' "mutant exit $rc"; fi
  else
    report 1 'negative control: the harness reading it is what keeps that stub from UNMAPPED' 'the mutation anchor does not occur exactly once'
  fi
  rm -f "$repo/tools/affected-mutant.sh"
  rm -rf "${repo:?}/stubs" "${repo:?}/stubs2" "${repo:?}/tools/test-harnesses.sh" "${repo:?}/tools/test-h.sh"
fi

# This repository's own harnesses, read as `affected --harnesses` reads them
# (dune-free, so the toolchain-free jobs can run it too): gh-ocannl-1797's two
# CI reds, the scan-case tables and stubs (gh-ocannl-1584), and the ppx the
# type check's #project legs expand through.
root=$(cd "$HERE/.." && pwd)
raffected() { # ARGS... -> $out, $err, $rc, over this repository
  rc=0
  (cd "$TMP" && PATH=$TMP/nodune:$PATH "$root/tools/affected-tests.sh" --harnesses "$@") >"$out" 2>"$err" || rc=$?
}
rhas() { # LABEL LINE: stdout carries exactly that line, exit 0, no warning
  if [ "$rc" = 0 ] && grep -qxF -- "$2" "$out" && ! grep -q '^affected: warning:' "$err"; then report 0 "$1"
  else report 1 "$1" "exit $rc; got [$(tr '\n' '|' <"$out")]; stderr: $(tr '\n' '|' <"$err")"; fi
}
if harness_leg 'harness legs of this repository'; then
  raffected --files tools/test-run.sh
  rhas 'tools/test-run.sh reaches a member of its harness whole' tools/test-test-run-bisect.sh
  raffected --files benchmarks/fixture_digest.py
  rhas 'benchmarks/fixture_digest.py reaches the harness probing it' benchmarks/test-check-box.sh
  raffected --files tools/typecheck-scan-cases.sh
  rhas 'the scan-case type check reaches its harness whole' tools/test-typecheck-scan-cases.sh
  # The tables --stubbed checks, found as tools/typecheck-scan-cases.sh finds
  # them: each companion under scan_case_stubs names its table beside it.
  n=0
  for stub in "$root"/test/operations/scan_case_stubs/*; do
    [ -f "$stub" ] || continue
    rel=${stub#"$root"/} name=${stub##*/}
    n=$((n + 1))
    raffected --files "$rel"
    printf '%s\n' "tools/test-affected-tests.sh --only '^harness legs of this repository\$'" \
      "tools/test-typecheck-scan-cases.sh --only '^(1|10):'" >"$TMP/want"
    if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then
      report 0 "the stub $name reaches the legs over the repository tables and this leg, and nothing else"
    else report 1 "the stub $name reaches the legs over the repository tables and this leg, and nothing else" "exit $rc; [$(tr '\n' '|' <"$out")]"; fi
    raffected --files "test/operations/${name%.*}.ml"
    rhas "its table ${name%.*}.ml reaches those legs" "tools/test-typecheck-scan-cases.sh --only '^(1|10):'"
  done
  [ "$n" -ge 1 ] || report 1 'the repository has stubbed scan-case tables' 'none under test/operations/scan_case_stubs'
  # The ppx the #project legs expand their arms through, every module of its
  # stanza: read from there, since a module path spelled here would reach this
  # harness too, and a ppx module outside the declaration's glob must show.
  ppx_modules=$(sed -n '/(name ppx_ocannl)/,/(modules /s/^ (modules \([^)]*\)).*/\1/p' "$root/tensor/dune")
  [ -n "$ppx_modules" ] || report 1 'tensor/dune lists the ppx modules' 'no (modules ...) line in the ppx_ocannl stanza'
  for m in $ppx_modules; do
    raffected --files "tensor/$m.ml"
    rhas "the ppx module $m reaches the #project legs" "tools/test-typecheck-scan-cases.sh --only '^(10|11):'"
  done
  # The legs that selector picks are exactly the ones over the repository's
  # tables, as their names say: a new such leg must join the declaration.
  if "$root/tools/test-typecheck-scan-cases.sh" --list-legs >"$TMP/legs" 2>&1; then
    sed -n 's/^LEG  //p' "$TMP/legs" >"$TMP/leg-names"
    picked=$(while IFS= read -r l; do [[ $l =~ ^(1|10): ]] && printf '%s\n' "$l"; done <"$TMP/leg-names")
    named=$(grep -F 'the repository tables' "$TMP/leg-names")
    if [ -n "$picked" ] && [ "$picked" = "$named" ]; then report 0 "'^(1|10):' selects exactly the legs over the repository tables"
    else report 1 "'^(1|10):' selects exactly the legs over the repository tables" "picked [$picked] named [$named]"; fi
    picked=$(while IFS= read -r l; do [[ $l =~ ^(10|11): ]] && printf '%s\n' "$l"; done <"$TMP/leg-names")
    named=$(grep -F '#project' "$TMP/leg-names")
    if [ -n "$picked" ] && [ "$picked" = "$named" ]; then report 0 "'^(10|11):' selects exactly the #project legs"
    else report 1 "'^(10|11):' selects exactly the #project legs" "picked [$picked] named [$named]"; fi
  else report 1 'tools/test-typecheck-scan-cases.sh lists its legs' "$(tail -n 3 "$TMP/legs")"; fi
  # Every declared selector of every harness selects some leg of it.
  bash "$root/tools/test-harnesses.sh" --list >"$TMP/members" || exit 2
  while IFS= read -r h; do
    grep -q '^# affected-reads: .* --only ' "$root/$h" || continue
    "$root/$h" --list-legs 2>/dev/null | sed -n 's/^LEG  //p' >"$TMP/leg-names"
    while IFS= read -r ere; do
      hit=0
      while IFS= read -r l; do [[ $l =~ $ere ]] && hit=1; done <"$TMP/leg-names"
      [ "$hit" = 1 ] && report 0 "$h: --only '$ere' selects a leg" ||
        report 1 "$h: --only '$ere' selects a leg" "its --list-legs: $(tr '\n' '|' <"$TMP/leg-names")"
    done < <(sed -n "s/^# affected-reads: .* --only '\\([^']*\\)'\$/\\1/p" "$root/$h")
  done <"$TMP/members"
fi

# The probe (gh-ocannl-1541): dune is handed each alias rule as a target rule
# of its own (a twin) in a scratch copy, so what an alias action reads is
# dune's answer, and a dune-file edit is compared rule by rule across
# revisions. The fixtures below hold one shape per form the hand readers of
# staging#1336 misread; each leg asks for the exact list.
unset OCANNL_TOOL_AFFECTED_PROBE
mkdir -p "$repo/p1" "$repo/p2" "$repo/p5" "$repo/p3/s1" "$repo/p3/s2" "$repo/p4" || exit 2
printf '(lang dune 3.20)\n(using directory-targets 0.1)\n' >"$repo/dune-project"
for f in cfg/ws.txt p1/inc.txt p1/chd.txt p1/tch.txt p1/blk.txt 'p1/#' p1/hs.txt p1/hb.txt p1/hl.txt p2/p2in.txt; do
  : >"$repo/$f" || exit 2
done
printf '(inc.txt)\n' >"$repo/p1/deps.sexp"
# Alias actions dune evaluates for the twins: a root-prefixed path, an
# (:include)d dependency list (whose additions `dune rules` does not print:
# blind), nested and templated chdir, a block string, Dune's own reading of
# `#;` (an atom `#`, then a comment), a directory target beside an alias, and
# a helper alias between a per-test alias and its input, by a rule and by an
# (alias) stanza.
cat >"$repo/p1/dune" <<'DUNE'
(rule (alias runtest-wsroot) (deps %{workspace_root}/cfg/ws.txt) (action (progn)))
(rule (alias runtest-incdeps) (deps (:include deps.sexp)) (action (progn)))
(rule (alias runtest-chdir) (action (chdir %{workspace_root} (chdir p1 (run cat %{dep:chd.txt})))))
(rule (alias runtest-tchdir) (action (chdir %{env:P1_DIR=.} (run cat %{dep:tch.txt}))))
(rule (alias runtest-block) (action (system "\| cat %{dep:blk.txt}
  )))
(rule (alias runtest-hashsemi) (deps hs.txt #;hb.txt
 ) (action (progn)))
(rule (alias runtest-dirt) (targets (dir gendir)) (action (system "mkdir -p gendir && cat %{dep:hb.txt} > gendir/x")))
(rule (alias helper) (deps hl.txt) (action (progn)))
(rule (alias runtest-viahelper) (deps (alias helper)) (action (progn)))
(alias (name runtest-viastanza) (deps (alias helper)))
DUNE
# A directory holding a stanza the converter does not model: its aliases stay blind.
cat >"$repo/p2/dune" <<'DUNE'
(cram)
(rule (alias runtest-p2blind) (deps p2in.txt) (action (progn)))
DUNE
cat >"$repo/p3/dune" <<'DUNE'
(test (name pa) (modules pa))
(test (name pb) (modules pb))
(rule (target made3.txt) (action (with-stdout-to %{target} (echo m))))
(rule (alias runtest-reads3) (action (cat %{dep:made3.txt})))
(rule (alias runtest-act3) (action (echo a)))
(rule (alias runtest-blk3) (action (system "\| echo a
  )))
(executable (name lk) (modules lk) (link_deps made3.txt))
(rule (alias runtest-lk) (action (run %{exe:lk.exe})))
(rule (alias runtest-dirt3) (targets (dir gd3)) (action (system "mkdir -p gd3")))
(rule (alias h3) (action (echo h)))
(rule (alias runtest-needh3) (deps (alias h3)) (action (progn)))
(alias (name h4) (deps h4.txt))
(rule (alias runtest-needh4) (deps (alias h4)) (action (progn)))
(alias (name runtest-viah4) (deps (alias h4)))
(rule (alias runtest-two) (targets two1.txt) (action (with-stdout-to two1.txt (echo 1))))
(rule (alias runtest-two) (targets two2.txt) (action (with-stdout-to two2.txt (echo 2))))
(subdir s1 (rule (alias runtest-sx) (action (echo x))))
DUNE
printf '; nothing here but the subdir stanzas of p3/dune\n' >"$repo/p3/s1/dune"
printf '; nothing here but the subdir stanzas of p3/dune\n' >"$repo/p3/s2/dune"
for m in pa pb lk; do printf 'let () = ()\n' >"$repo/p3/$m.ml"; done
: >"$repo/p3/h4.txt"
: >"$repo/p3/h4b.txt"
: >"$repo/p3/pa.expected"
: >"$repo/p3/pb.expected"
# An alias rule the converter does not rewrite (an (:include)), beside one it does.
cat >"$repo/p5/dune" <<'DUNE'
(rule (alias runtest-inc5) (deps (:include inc5.sexp)) (action (progn)))
(rule (alias runtest-p5ok) (action (echo ok)))
DUNE
printf '()\n' >"$repo/p5/inc5.sexp"
# A scan reading p3's dune file, and a sibling test it does not concern.
cat >"$repo/p4/dune" <<'DUNE'
(rule (target dscan.actual) (deps %{project_root}/p3/dune) (action (with-stdout-to %{target} (echo s))))
(rule (alias runtest-dscan) (action (diff dscan.expected dscan.actual)))
(test (name p4t) (modules p4t))
DUNE
printf 'let () = ()\n' >"$repo/p4/p4t.ml"
printf 's\n' >"$repo/p4/dscan.expected"
: >"$repo/p4/p4t.expected"
git_q add -A && git_q commit -qm probe-fixture || exit 2
probe_blind='@p2/runtest-p2blind @p1/runtest-incdeps @p5/runtest-inc5'
p3_all='@p3/runtest-act3 @p3/runtest-blk3 @p3/runtest-dirt3 @p3/runtest-lk @p3/runtest-needh3 @p3/runtest-needh4 @p3/runtest-pa
  @p3/runtest-pb @p3/runtest-reads3 @p3/runtest-two @p3/runtest-viah4 @p3/s1/runtest-sx'
probe_restore() { git_q reset -q --hard && git_q clean -qfdx -- p1 p2 p3 p4 p5 p6; }

if harness_leg 'probe alias actions'; then
  for pair in cfg/ws.txt:runtest-wsroot p1/chd.txt:runtest-chdir p1/tch.txt:runtest-tchdir p1/blk.txt:runtest-block \
              'p1/#:runtest-hashsemi' p1/hs.txt:runtest-hashsemi p1/hb.txt:runtest-dirt; do
    affected --files "${pair%%:*}"
    # A non-build input reaches the blind ones as well.
    # shellcheck disable=SC2086
    expect "an alias action reading ${pair%%:*} as dune evaluates it lists @p1/${pair#*:}" 0 "@p1/${pair#*:}" $probe_blind
  done
  # A blind alias still reads what dune does print: the list it includes.
  affected --files p1/deps.sexp
  # shellcheck disable=SC2086
  expect 'an alias action reading p1/deps.sexp lists @p1/runtest-incdeps, blind or not' 0 $probe_blind
  affected --files p1/hl.txt
  # shellcheck disable=SC2086
  expect 'a helper alias between per-test aliases and their input lists them, by a rule and by an (alias) stanza' 0 \
    @p1/runtest-viahelper @p1/runtest-viastanza $probe_blind
  affected --files p1/inc.txt
  # shellcheck disable=SC2086
  expect 'what an (:include) adds, which dune rules does not print, leaves its alias blind' 0 $probe_blind
  if grep -q 'blind because a rule of p1/dune with a :include' "$err"; then report 0 'a blind alias says why on stderr'
  else report 1 'a blind alias says why on stderr' "$(cat "$err")"; fi
fi
if harness_leg 'probe library and data'; then
  affected --files lib/helper.ml
  expect 'with the probe, a library module reaches what reads or links it, and no alias whose action reads nothing' 0 \
    @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scan @t2/runtest-scenario_x \
    @t2/slow-gen @t5/slow-usesh
  affected --files fx.opam
  # shellcheck disable=SC2086
  expect 'a (package) dependency is read through the alias dune makes for it' 0 \
    @t1/runtest-t_mid-blind @t2/runtest-needs_pkg @t2/runtest-pkg $probe_blind
  affected --files docs/other.md
  # shellcheck disable=SC2086
  expect 'with the probe, a file no rule reads lists only the blind aliases' 0 $probe_blind
  affected --files p2/p2in.txt
  # shellcheck disable=SC2086
  expect 'a directory with a stanza the converter does not model keeps its aliases blind' 0 $probe_blind
  affected --files t6/extra.inc
  # shellcheck disable=SC2086
  expect 'an included fragment, rewritten in the probe, still reaches every directory including it' 0 \
    @t6/runtest-dyn @t6/runtest-inc @t7/runtest-g7 @t7/runtest-inc $probe_blind
fi
probe_edit() { # PERL-SUBSTITUTION: edit p3/dune in place, uncommitted
  EDIT=$1 perl -0pi -e 'BEGIN { $n = 0 } $n += eval $ENV{EDIT}; END { exit($n ? 0 : 1) }' "$repo/p3/dune" ||
    { report 1 "probe fixture edit did not apply: $1"; return 1; }
}
p3_scan=@p4/runtest-dscan
probe_leg() { # LABEL EXPECTED-WORDS: run `affected HEAD` on the edited tree, then restore it
  local label=$1 want=$2
  affected HEAD
  # shellcheck disable=SC2086
  expect "$label" 0 $want
  probe_restore
}
if harness_leg 'probe dune edits'; then
  # One new test stanza: without the probe its whole directory (and those
  # beneath it); with it, the test and the scan reading the dune file.
  printf '(test (name pc) (modules pc))\n' >>"$repo/p3/dune"
  printf 'let () = ()\n' >"$repo/p3/pc.ml"
  OCANNL_TOOL_AFFECTED_PROBE=0 affected HEAD
  # shellcheck disable=SC2086
  expect 'without the probe, one new test stanza lists its whole directory' 0 $p3_all @p3/runtest-pc $p3_scan
  probe_leg 'with the probe, one new test stanza lists that test and the scan reading the dune file' "@p3/runtest-pc $p3_scan"
  probe_edit 's/\(echo a\)\)\)/(echo b)))/' &&
    probe_leg 'an action-only edit lists the alias whose action it is' "@p3/runtest-act3 $p3_scan"
  probe_edit 's/\(echo m\)/(echo n)/' &&
    probe_leg 'an edit to a rule'"'"'s action lists what reads its target, through link_deps too' "@p3/runtest-reads3 @p3/runtest-lk $p3_scan"
  probe_edit 's/\(test \(name pa\) \(modules pa\)\)/(test (name pa) (modules pa) (flags (:standard -w -a)))/' &&
    probe_leg 'a flags edit lists the stanza'"'"'s own alias' "@p3/runtest-pa $p3_scan"
  probe_edit 's/"\\\| echo a\n/"\\| echo \\\\a\n/' &&
    probe_leg 'an edit inside a block string lists the alias whose action it is' "@p3/runtest-blk3 $p3_scan"
  probe_edit 's/\(echo a\)\)\)/(echo "a")))/' &&
    probe_leg 'a quotation-only edit dune reads the same lists only the scan reading the file' "$p3_scan"
  probe_edit 's/\A/; a comment\n/' &&
    probe_leg 'a comment-only edit lists only the scan reading the file (the two probe roots compare equal)' "$p3_scan"
  probe_edit 's/\(system "mkdir -p gd3"\)/(system "mkdir -p gd3 && :")/' &&
    probe_leg 'a rule with an alias and a directory target lists its alias' "@p3/runtest-dirt3 $p3_scan"
  probe_edit 's/\(echo h\)/(echo g)/' &&
    probe_leg 'an edit to a helper alias lists the per-test aliases needing it' "@p3/runtest-needh3 $p3_scan"
  probe_edit 's/\(deps h4\.txt\)/(deps h4b.txt)/' &&
    probe_leg 'a dependency edit to a helper (alias) stanza lists the aliases needing it, by a rule and by an (alias) stanza' \
      "@p3/runtest-needh4 @p3/runtest-viah4 $p3_scan"
  probe_edit 's/\(echo m\)/(echo n)/ + s/\(echo a\)\)\)/(echo b)))/' &&
    probe_leg 'a visible rule edit beside an alias-action edit lists both' "@p3/runtest-act3 @p3/runtest-reads3 @p3/runtest-lk $p3_scan"
  probe_edit 's/\(test \(name pb\) \(modules pb\)\)/(test (name pd) (modules pd))/' && git_q mv p3/pb.ml p3/pd.ml &&
    git_q mv p3/pb.expected p3/pd.expected &&
    probe_leg 'a renamed test, which the old dune file cannot describe beside the new sources, keeps the whole directory' \
      "$(echo "$p3_all" | sed 's|@p3/runtest-pb|@p3/runtest-pd|') $p3_scan $probe_blind"
  probe_edit 's/\(rule \(alias runtest-two\) \(targets two2\.txt\)[^\n]*\n//' &&
    probe_leg 'a removed target rule lists the alias that owned it on the old side' "@p3/runtest-two $p3_scan"
  perl -0pi -e 's/\(echo ok\)/(echo okay)/; s/\(action \(progn\)\)\)/(action (echo i)))/' "$repo/p5/dune"
  affected HEAD
  # shellcheck disable=SC2086
  expect 'a changed rule the probe does not describe keeps the whole directory' 0 @p5/runtest-inc5 @p5/runtest-p5ok
  if grep -q 'p5/dune: its whole directory, because a changed (rule) the probe does not describe' "$err"; then
    report 0 'a dune file kept whole for an undescribed rule says why'
  else report 1 'a dune file kept whole for an undescribed rule says why' "$(cat "$err")"; fi
  probe_restore
  probe_edit 's/\(rule \(alias runtest-two\) \(targets two2\.txt\)/(rule (alias runtest-dirt3) (targets two2.txt)/' &&
    probe_leg 'a target rule moved from one alias to another lists both' "@p3/runtest-dirt3 @p3/runtest-two $p3_scan"
  # What dune cannot answer keeps the whole directory.
  probe_edit 's/\z/(env (_ (env-vars (P3 x))))\n/' &&
    probe_leg 'a changed (env) stanza keeps the reach of its whole directory' "$p3_all $p3_scan"
  probe_edit 's/\(subdir s1 /(subdir s2 /' &&
    probe_leg 'a stanza moved between (subdir) wrappers keeps the reach of the whole directory' \
      "$(echo "$p3_all" | sed 's|@p3/s1/runtest-sx|@p3/s2/runtest-sx|') $p3_scan"
  printf '(test (name pc) (modules pc))\n' >>"$repo/p3/dune"
  printf 'let () = ()\n' >"$repo/p3/pc.ml"
  affected --files p3/dune
  # shellcheck disable=SC2086
  expect '--files names no revision, so a dune file keeps its whole directory' 0 $p3_all @p3/runtest-pc $p3_scan
  if grep -q 'its whole directory, because --files names no revision' "$err"; then report 0 'a dune file kept whole says why'
  else report 1 'a dune file kept whole says why' "$(cat "$err")"; fi
  probe_restore
fi
if harness_leg 'probe library split'; then
  # A dune-only split (gh-ocannl-1651): `moved` leaves library big for a new library small, which
  # big links, and no module text changes. Both stanzas are reached by their modules, not as a
  # whole: what names Big or Moved ranks first, what only links big is link-only. The tests live
  # in a directory of their own, as the repository's do: an executable beside a split library has
  # a link rule the comparison sees change, and is reached as a whole.
  mkdir -p "$repo/p7/t" || exit 2
  printf '(library (name big) (modules big moved))\n' >"$repo/p7/dune"
  printf '(test (name p7a) (modules p7a) (libraries big))\n(test (name p7b) (modules p7b) (libraries big))\n(test (name p7c) (modules p7c) (libraries big))\n' \
    >"$repo/p7/t/dune"
  printf 'let f () = ()\n' >"$repo/p7/big.ml"
  printf 'let g () = ()\n' >"$repo/p7/moved.ml"
  printf 'let () = Big.f ()\n' >"$repo/p7/t/p7a.ml"
  printf 'let () = Moved.g ()\n' >"$repo/p7/t/p7b.ml"
  printf 'let () = ()\n' >"$repo/p7/t/p7c.ml"
  for t in p7a p7b p7c; do : >"$repo/p7/t/$t.expected"; done
  git_q add -A p7 && git_q commit -qm split-fixture || exit 2
  split_to() { # BIG-EXTRA: rewrite p7/dune with moved split out, BIG-EXTRA added to big's fields
    printf '(library (name big) (modules big) (libraries small)%s)\n(library (name small) (modules moved))\n' "$1" >"$repo/p7/dune"
  }
  split_to ''
  affected --referenced-only HEAD
  expect 'a library split in a dune file alone ranks first what names a module of either library' 0 @p7/t/runtest-p7a @p7/t/runtest-p7b
  if grep -q 'p7/dune: compared with the old revision: .*2 library stanza(s) changed only in their modules or libraries' "$err" &&
     grep -qx 'affected: link-only: @p7/t/runtest-p7c' "$err"; then
    report 0 'the split says which stanzas it reached by their modules, and names the link-only alias'
  else report 1 'the split says which stanzas it reached by their modules, and names the link-only alias' "$(cat "$err")"; fi
  split_to ' (flags (:standard -w -a))'
  affected --referenced-only HEAD
  expect '... and a flags change beside it keeps big whole' 0 @p7/t/runtest-p7a @p7/t/runtest-p7b @p7/t/runtest-p7c
  git_q checkout -q -- p7/dune || exit 2
  mutant=$repo/tools/affected-mutant.sh
  if FROM='if (@ix && $kind eq "library" && $membership_only->($r)) {' TO='if (0) {' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    split_to ''
    rc=0
    (cd "$TMP" && bash "$mutant" --referenced-only HEAD) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 0 ] && grep -qx @p7/t/runtest-p7c "$out"; then report 0 'negative control: reading the membership change is what ranks the split'
    else report 1 'negative control: reading the membership change is what ranks the split' "exit $rc; $(cat "$out")"; fi
    git_q checkout -q -- p7/dune || exit 2
  else report 1 'negative control: reading the membership change is what ranks the split' "the mutation anchor does not occur exactly once"; fi
  rm -f "$mutant"
  git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/p7" || exit 2
fi
# A variant with a golden of its own (`runtest-sib-alt`, `runtest-sib-low`) has
# its test's stanza as an owner, so whatever reaches the test lists it too, in
# the same tier: the list `tools/test-run.sh run` names siblings for when a hand-
# written batch leaves them out. A base golden alone reaches only its own alias.
if harness_leg 'sibling goldens'; then
  probe_restore
  mkdir -p "$repo/p8" || exit 2
  cat >"$repo/p8/dune" <<'DUNE'
(library (name sl) (modules sl))
(test (name sib) (modules sib) (libraries sl))
(rule (target sib-alt.actual) (action (with-stdout-to %{target} (run %{exe:sib.exe} alt))))
(rule (alias runtest-sib-alt) (action (diff sib-alt.expected sib-alt.actual)))
(rule (target sib-low.ll.actual) (action (with-stdout-to %{target} (run %{exe:sib.exe} low))))
(rule (alias runtest-sib-low) (action (diff sib-low.ll.expected sib-low.ll.actual)))
DUNE
  printf 'let f () = ()\n' >"$repo/p8/sl.ml"
  printf 'let () = Sl.f ()\n' >"$repo/p8/sib.ml"
  : >"$repo/p8/sib.expected"
  : >"$repo/p8/sib-alt.expected"
  : >"$repo/p8/sib-low.ll.expected"
  git_q add -A p8 && git_q commit -qm sibling-fixture || exit 2
  sib_all='@p8/runtest-sib @p8/runtest-sib-alt @p8/runtest-sib-low'
  for f in p8/sib.ml p8/sl.ml; do
    affected --files "$f"
    # shellcheck disable=SC2086
    expect "a change to $f lists the test with its sibling goldens" 0 $sib_all
    affected --referenced-only --files "$f"
    # shellcheck disable=SC2086
    expect "... and so does --referenced-only" 0 $sib_all
  done
  OCANNL_TOOL_AFFECTED_PROBE=0 affected --files p8/sl.ml
  # shellcheck disable=SC2086
  expect '... and so does the reach without the probe' 0 $sib_all
  affected --files p8/sib.expected
  # shellcheck disable=SC2086
  expect 'a base golden alone reaches its own alias, not the siblings, whose rules do not read it' 0 @p8/runtest-sib $probe_blind
  # Two routes reach a variant: the name it shares with the stanza, and the
  # executable its `.actual` rule runs. Cut for a variant, the siblings fall away.
  mutant=$repo/tools/affected-mutant.sh
  if FROM1='grep { $st[$_]{units}{$unit} } @in }, own' TO1='grep { !@variant && $st[$_]{units}{$unit} } @in }, own' \
     FROM2='map { %{$runs{$_} // {}} } @{$a->{own}});' TO2='$a->{alias} =~ /-sib-/ ? () : map { %{$runs{$_} // {}} } @{$a->{own}});' \
      perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM1}\E/$ENV{TO1}/g; $n += s/\Q$ENV{FROM2}\E/$ENV{TO2}/g; END { exit($n == 2 ? 0 : 1) }' \
      "$repo/tools/affected-tests.sh" >"$mutant"; then
    rc=0
    (cd "$TMP" && bash "$mutant" --files p8/sl.ml) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 0 ] && ! grep -qx @p8/runtest-sib-alt "$out" && ! grep -qx @p8/runtest-sib-low "$out"; then
      report 0 'negative control: the name and the executable are what list a sibling'
    else report 1 'negative control: the name and the executable are what list a sibling' "exit $rc; $(cat "$out")"; fi
  else report 1 'negative control: the name and the executable are what list a sibling' 'the mutation anchors did not each occur once'; fi
  rm -f "$mutant"
  git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/p8" || exit 2
fi
# Probes that do not answer: an input the copy lacks (an ignored directory),
# and a dune file including one. Each reads the change without the probe.
probe_p6() { # DUNE-TEXT [FRAGMENT-TEXT]
  mkdir -p "$repo/p6/gen" && printf 'gen/\n' >"$repo/p6/.gitignore" && printf '%s\n' "$1" >"$repo/p6/dune" &&
    : >"$repo/p6/gen/dep.txt" && : >"$repo/p6/p6.txt" || return 1
  [ -z "${2:-}" ] || printf '%s\n' "$2" >"$repo/p6/gen/x.inc"
}
if harness_leg 'probe fallback'; then
  probe_p6 '(rule (alias runtest-ign) (deps gen/dep.txt) (action (progn)))' || exit 2
  # dune prints what a rule reads whether or not the file is there: the copy
  # lacks the ignored input, and the probe still reads it right.
  affected --files p6/p6.txt
  # shellcheck disable=SC2086
  expect 'a rule whose input the probe copy lacks reads only that input (here: not p6.txt)' 0 $probe_blind
  affected --files p6/gen/dep.txt
  # shellcheck disable=SC2086
  expect 'a rule whose input the probe copy lacks is listed for that input' 0 @p6/runtest-ign $probe_blind
  probe_restore
  probe_p6 '(include gen/x.inc)
(rule (alias runtest-p6) (action (progn)))' '(rule (alias runtest-inc6) (action (progn)))' || exit 2
  affected --files p6/p6.txt
  if [ "$rc" = 0 ] && grep -q 'the probe failed' "$err" && grep -qx @p6/runtest-p6 "$out"; then
    report 0 'a probe dune does not load reads the change without the probe'
  else report 1 'a probe dune does not load reads the change without the probe' "exit $rc; $(cat "$out" "$err")"; fi
  probe_restore
  # A probe that loads but whose rules dune will not print (a shim refusing
  # `dune rules` inside it): the whole reading again without the probe, so the
  # list is the one without it.
  mkdir -p "$TMP/shim4" || exit 2
  cat >"$TMP/shim4/dune" <<SHIM
#!/usr/bin/env bash
case "\${1:-}:\$PWD" in rules:*/probe) echo 'Error: refused by the harness shim' >&2; exit 1 ;; esac
exec "$REAL_DUNE" "\$@"
SHIM
  chmod +x "$TMP/shim4/dune" || exit 2
  rc=0
  (cd "$TMP" && PATH="$TMP/shim4:$PATH" "$repo/tools/affected-tests.sh" --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
  if grep -q 'reading again without it' "$err"; then
    # shellcheck disable=SC2086
    expect 'a probe whose rules dune will not print reads the change again without it' 0 $reached_by_core
  else report 1 'a probe whose rules dune will not print reads the change again without it' "exit $rc; $(cat "$err")"; fi
fi
if harness_leg 'probe controls'; then
  # shellcheck disable=SC2086
  control 'a twin is what makes an alias action dune'"'"'s answer' \
    'my @act = grep { named($_, "action") } @fl;' 'my @act;' \
    --files p1/chd.txt -- @p1/runtest-chdir $probe_blind
  # shellcheck disable=SC2086
  control 'an (:include) is what leaves its alias blind' \
    'if ($canon =~ /(?:^|[\s(])(:include|dynamic-run)[\s)]/) {' 'if (0) {' \
    --files p1/inc.txt -- $probe_blind
  # shellcheck disable=SC2086
  control 'a helper alias'"'"'s own entry is what lists the per-test alias needing it' \
    'push @al, $h;' '' \
    --files p1/hl.txt -- @p1/runtest-viahelper @p1/runtest-viastanza $probe_blind
  # shellcheck disable=SC2086
  control 'asking dune for an (alias) stanza'"'"'s dependencies is what lists it' \
    'source_query($_) for @al;' '' \
    --files p1/hl.txt -- @p1/runtest-viahelper @p1/runtest-viastanza $probe_blind
  printf '(test (name pc) (modules pc))\n' >>"$repo/p3/dune"
  printf 'let () = ()\n' >"$repo/p3/pc.ml"
  control 'comparing the dune file across revisions is what scopes a new test stanza (the base reach)' \
    'else scoped+=("$f"); fi' 'fi' \
    HEAD -- @p3/runtest-pc $p3_scan
  probe_restore
  probe_edit 's/\(echo a\)\)\)/(echo b)))/' &&
    control 'comparing rules across the probes is what finds an action-only edit' \
      'my @delta = grep {' 'my @delta = (); 0 && grep {' \
      HEAD -- @p3/runtest-act3 $p3_scan
  probe_restore
  probe_edit 's/\(echo m\)/(echo n)/' &&
    control 'a stanza'"'"'s link rule is what reaches a link_deps consumer' \
      'for (@{$stanza_rule{$t}}) { $hit{$_} = $whole_hit{$_} = 1; $hit_why{$_} //= $why }' '' \
      HEAD -- @p3/runtest-reads3 @p3/runtest-lk $p3_scan
  probe_restore
  probe_edit 's/\(test \(name pa\) \(modules pa\)\)/(test (name pa) (modules pa) (flags (:standard -w -a)))/' &&
    control 'a changed stanza named as dune lists it is what reaches a flags edit' \
      'if (@ix) { $whole{$_} = 1 for @ix; next }' 'if (@ix) { next }' \
      HEAD -- @p3/runtest-pa $p3_scan
  probe_restore
  probe_edit 's/\z/(env (_ (env-vars (P3 x))))\n/' &&
    control 'a changed form outside the allowlist is what keeps the whole directory' \
      'push @why, $h eq "alias" ? "a changed (alias) with an action" : "a changed ($h) stanza";' '' \
      HEAD -- $p3_all $p3_scan
  probe_restore
  probe_edit 's/\(deps h4\.txt\)/(deps h4b.txt)/' &&
    control 'a changed helper (alias) stanza is what lists the aliases needing it' \
      '$direct_here{in_dir($d, $x)} = "its (alias) stanza changed";' '' \
      HEAD -- @p3/runtest-needh4 @p3/runtest-viah4 $p3_scan
  probe_restore
  probe_edit 's/\(echo h\)/(echo g)/' &&
    control 'a changed twin of a helper alias is what lists the alias needing it' \
      '$direct_here{in_dir($td, $_)} = "a rule of it changed" for @{$twin_of{$t} // []};' '' \
      HEAD -- @p3/runtest-needh3 $p3_scan
  probe_restore
  probe_edit 's/\(rule \(alias runtest-two\) \(targets two2\.txt\)/(rule (alias runtest-dirt3) (targets two2.txt)/' &&
    control 'the aliases a changed form attaches to are what list a moved target rule'"'"'s aliases' \
      '$direct_here{$x} //= "the stanzas attached to it changed"' 'my $unused = 1' \
      HEAD -- @p3/runtest-dirt3 @p3/runtest-two $p3_scan
  probe_restore
  probe_edit 's/\(rule \(alias runtest-two\) \(targets two2\.txt\)[^\n]*\n//' &&
    control 'the old side'"'"'s owners are what list the alias of a removed target' \
      '$direct_here{in_dir($td, $a->{alias})} = "a target it may have owned is gone";' '' \
      HEAD -- @p3/runtest-two $p3_scan
  probe_restore
  perl -0pi -e 's/\(action \(progn\)\)\)/(action (echo i)))/' "$repo/p5/dune"
  control 'the converter'"'"'s verdict is what keeps a changed undescribed rule'"'"'s directory whole' \
    'unless ($r->{vouched}) { push @why, "a changed ($h) the probe does not describe"; next }' '' \
    HEAD -- @p5/runtest-inc5 @p5/runtest-p5ok
  probe_restore
  probe_p6 '(include gen/x.inc)
(rule (alias runtest-p6) (action (progn)))' '(rule (alias runtest-inc6) (action (progn)))' || exit 2
  # The shipping reader exits 0 here (the 'probe fallback' leg); without the
  # alias comparison it accepts the probe, and its listing stops the run.
  if FROM='cmp -s "$tmp/worktree-aliases" "$tmp/probe-aliases"' TO='true' \
       perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
       "$repo/tools/affected-tests.sh" >"$repo/tools/affected-mutant.sh"; then
    rc=0
    (cd "$TMP" && bash "$repo/tools/affected-mutant.sh" --files p6/p6.txt) >"$out" 2>"$err" || rc=$?
    if [ "$rc" = 2 ]; then report 0 'negative control: comparing the probe'"'"'s aliases with the worktree'"'"'s is what keeps a probe dune does not load from stopping the run'
    else report 1 'negative control: comparing the probe'"'"'s aliases with the worktree'"'"'s is what keeps a probe dune does not load from stopping the run' "mutant exit $rc"; fi
  else
    report 1 'negative control: comparing the probe'"'"'s aliases with the worktree'"'"'s is what keeps a probe dune does not load from stopping the run' 'the mutation anchor does not occur exactly once'
  fi
  rm -f "$repo/tools/affected-mutant.sh"
  probe_restore
fi

finish

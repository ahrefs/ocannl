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
# H, which gen names; nothing else names either.
printf 'let used = 1\nlet unused = 2\n' >"$repo/lib/helper.ml"
printf 'module H = Helper\n' >"$repo/lib/core.ml"
printf 'let () = () (* uses Helper.used *)\n' >"$repo/t1/t_core.ml"
printf 'let () = ()\n' >"$repo/t1/t_mid.ml"
printf 'let () = ()\n' >"$repo/lib/plain_core.ml"
printf '(rule (alias runtest-inc) (action (progn)))\n' >"$repo/t6/extra.inc"
printf 'let () = () (* H *)\n' >"$repo/t2/gen.ml"
git_q() { git -C "$repo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
git_q init -q && git_q add -A && git_q commit -qm fixture || exit 2

out=$TMP/out err=$TMP/err
affected() { # ARGS... -> $out, $err, $rc
  rc=0
  (cd "$TMP" && "$repo/tools/affected-tests.sh" "$@") >"$out" 2>"$err" || rc=$?
}
expect() { # LABEL RC ALIAS... : the exact list, in any order, and the exit status
  local label=$1 want_rc=$2
  shift 2
  printf '%s\n' "$@" | sed '/^$/d' | sort >"$TMP/want"
  # The printed order is ranked (link-only last); the 'link-only' leg pins it.
  sort "$out" >"$TMP/got"
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
  # t_core links mid, whose (select) module is code not all known: never link-only.
  ordered 'a change to one definition ranks first only what names it' \
    @t1/runtest-t_core @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen
  printf 'let used = 2\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected HEAD
  ordered '... and t_core names `used`' \
    @t1/runtest-t_core @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen
  mutant=$repo/tools/affected-mutant.sh
  if FROM='return 0 if !$defs || grep { $tok->{$_} } keys %$defs;' TO='return 0;' \
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
  printf 'let used = 1\nlet unused = 2\ntype t = A | B\n' >"$repo/lib/helper.ml"
  affected HEAD
  ordered 'a changed type definition counts as the whole module (its constructors are what users name)' \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh @t1/runtest-t_mid @t1/runtest-t_mid-blind
  git_q checkout -q -- lib/helper.ml || exit 2
  # A new interface can hide any member: the whole module changed, whatever its hunks show.
  printf 'val used : int\n' >"$repo/lib/helper.mli"
  git_q add lib/helper.mli && git_q commit -qm helper-mli || exit 2
  affected --referenced-only HEAD~1..HEAD
  if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then
    report 0 'a file new on one side changes its module as a whole'
  else report 1 'a file new on one side changes its module as a whole' "exit $rc; $(cat "$out")"; fi
  git_q reset -q --hard HEAD~1 || exit 2
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
    '$as{$_}{$as_name} = 1 for grep { $_ ne $as_name } @names;' '1;' \
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
      '$hit{$_} = $whole_hit{$_} = 1 for @{$stanza_rule{$t}};' '' \
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

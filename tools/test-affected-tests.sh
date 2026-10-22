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

repo=$TMP/proj
mkdir -p "$repo/tools" "$repo/lib" "$repo/mid" "$repo/t1" "$repo/t2" "$repo/t3" "$repo/t4/sub" "$repo/t5" "$repo/t6" \
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
cat >"$repo/t6/dune" <<'DUNE'
(rule (target word.txt) (action (progn (system "echo noise-from-an-action >&2") (with-stdout-to %{target} (echo dyn)))))
(rule (target dyn.actual) (action (with-stdout-to %{target} (echo %{read:word.txt}))))
(rule (alias runtest-dyn) (action (diff dyn.expected dyn.actual)))
DUNE
cat >"$repo/lib/dune" <<'DUNE'
(library (name core) (modules core helper lx))
(ocamllex lx)
(library (name leaf) (modules leaf))
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
(test (name t_leaf) (modules t_leaf) (libraries leaf))
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
git_q() { git -C "$repo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false "$@"; }
git_q init -q && git_q add -A && git_q commit -qm fixture || exit 2

out=$TMP/out err=$TMP/err
affected() { # ARGS... -> $out, $err, $rc
  rc=0
  (cd "$TMP" && "$repo/tools/affected-tests.sh" "$@") >"$out" 2>"$err" || rc=$?
}
expect() { # LABEL RC ALIAS... : the exact sorted list and the exit status
  local label=$1 want_rc=$2
  shift 2
  printf '%s\n' "$@" | sed '/^$/d' | sort >"$TMP/want"
  if [ "$rc" = "$want_rc" ] && cmp -s "$TMP/want" "$out"; then
    report 0 "$label"
  else
    report 1 "$label" "exit $rc (want $want_rc); got [$(tr '\n' ' ' <"$out")] want [$(tr '\n' ' ' <"$TMP/want")]; stderr: $(tr '\n' '|' <"$err")"
  fi
}
reached_by_core='@t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen
  @t5/slow-usesh'
# Every alias whose action dune does not dump and whose stanzas this reader
# cannot read in full (no golden among its own targets): listed for any changed
# data file. runtest-needs_pkg needs it, and is not listed for that reason.
opaque_all='@t2/runtest-pkg'

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
    if [ "$rc" = 2 ] && grep -q 'unexpected atom in dune rules output: "noise-from-an-action" at byte 0' "$err" &&
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
fi

if harness_leg 'select choice'; then
  affected --files mid/choice.core.ml
  expect 'a (select) choice maps to its module' 0 @t1/runtest-t_core
fi

if harness_leg 'tests helper module'; then
  affected --files t1/shared.ml
  expect 'a module of a tests stanza (undescribed) reaches every tests stanza of its directory' 0 \
    @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb
fi

if harness_leg 'library module alias'; then
  affected --files t1/m_one.ml
  expect 'a per-module alias of a library is attributed by the module name' 0 @t1/runtest-m_one
fi

if harness_leg 'ppx rewriter'; then
  affected --files lib/myppx.ml
  expect 'a ppx rewriter reaches a library it preprocesses (an edge dune describe workspace omits)' 0 \
    @t1/runtest-m_one @t2/runtest-scan
fi

if harness_leg 'generated module'; then
  affected --files lib/lx.mll
  expect 'an .mll reaches the stanza of the module its rule generates' 0 \
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
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
    @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb \
    @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan \
    @t2/runtest-scenario_x @t2/slow-gen @t3/runtest-cfgread @t3/runtest-esc @t3/runtest-neg @t3/runtest-py @t3/runtest-t_plain @t3/runtest-t_stub \
    @t3/runtest-uses @t4/runtest-gated @t4/runtest-objs @t4/runtest-t_gate @t4/runtest-t_gate-pre \
    @t4/sub/runtest-t_sub @t5/slow-usesh @t6/runtest-dyn
fi

if harness_leg 'dune file elsewhere'; then
  # t1/dune makes made.txt, which a t3 rule reads.
  affected --files t1/dune
  expect 'a dune file reaches the rules elsewhere that read the files it makes' 0 \
    @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb @t3/runtest-uses
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
  expect 'a data file lists the test declaring it, and every alias whose action is unknown' 0 @t3/runtest-t_plain $opaque_all
fi

if harness_leg 'foreign stubs'; then
  affected --files lib/stubbed_c.c
  expect 'a C stub reaches the library whose stub archive bundles its object' 0 @t3/runtest-t_stub
  affected --files t3/cexe_c.c
  expect 'a C stub no archive bundles reaches every executable and test stanza of its directory' 0 \
    @t3/runtest-cfgread @t3/runtest-esc @t3/runtest-neg @t3/runtest-t_plain @t3/runtest-t_stub \
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
  expect 'a file copied into a directory reaches the rules reading the copy' 0 @t3/runtest-cfgread @t3/runtest-t_plain $opaque_all
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
  # gated.actual needs the alias runtest-t_gate, listed for its changed config.
  affected --files t4/gate.cfg
  # shellcheck disable=SC2086
  expect 'an alias whose rules need a listed alias is listed too' 0 @t4/runtest-gated @t4/runtest-t_gate $opaque_all
  # runtest-pkg is listed only because its action is unknown; needs_pkg reads nothing from it.
  affected --files docs/other.md
  # shellcheck disable=SC2086
  expect 'an alias needing one listed only blind is not listed for that' 0 $opaque_all
fi

if harness_leg 'declared deps'; then
  # runtest-py runs and diffs in its own action; its golden made it look like a
  # diff of its own targets, and its (deps) were never read (gh-ocannl-1431).
  affected --files cfg/extra.py
  # shellcheck disable=SC2086
  expect 'an alias rule'"'"'s own (deps) reach it, outside its directory too' 0 @t3/runtest-py $opaque_all
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
    @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb
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
  if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then
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
    ' || $opaque_listed) {' ') {' \
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
    'for my $w (obj_owners($o)) { push @{$st[$w]{to}}, keys %{$runs{$o}} }' '' \
    --files lib/helper.ml -- $reached_by_core
  control 'UNIT_ targets are what tie an alias to the executable making them' \
    'my @near = grep { $_ !~ $artifact && index($_, "${unit}_") == 0 } @{$targets->{$d} // []};' 'my @near;' \
    --files t4/pre.ml -- @t4/runtest-gated @t4/runtest-t_gate @t4/runtest-t_gate-pre
  control 'a compiled-object glob is what reaches the stanza it reads' \
    'if ($where eq "In_build_dir" and my @o = compiled_owners("$path/")) { $r{$_} = 1 for @o; next }' '' \
    --files t4/inert.ml -- @t4/runtest-objs
  control 'the alias fixpoint is what lists an alias needing a listed one' \
    'next unless grep { my $t = $_; grep { $out{$_} && !$guess{$_} } keys %{$adeps{$t} // {}} } @{$a->{own}};' 'next;' \
    --files t4/gate.cfg -- @t4/runtest-gated @t4/runtest-t_gate $opaque_all
  # shellcheck disable=SC2086
  control 'an alias listed only blind is what keeps one needing it off the list' \
    'grep { $out{$_} && !$guess{$_} }' 'grep { $out{$_} }' \
    --files docs/other.md -- $opaque_all
  # shellcheck disable=SC2086
  control 'reading an alias rule'"'"'s own (deps) is what reaches it' \
    '$reads{$k} = { %{$acc->{files}} };' '$reads{$k} = {};' \
    --files cfg/extra.py -- @t3/runtest-py $opaque_all
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

finish

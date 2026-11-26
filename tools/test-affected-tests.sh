#!/usr/bin/env bash
# Legs for tools/affected-tests.sh (`tools/test-run.sh affected`), on a small
# real Dune project of its own: which per-test aliases a change reaches, read
# from dune. No Dune alias: the fixture owns its build, and a dune nested in
# `dune runtest` is a poor bargain (its own lock, the outer run's DUNE_*).
# Usage: tools/test-affected-tests.sh [--self-check] [--keep] [--only REGEX]
#
# Put --self-check before the shared options. It replaces each selected control's
# literal edits to the fixture reader with identity edits and requires the same
# assertions to reject each no-op.
# Fixture setup, mutation setup and reader
# completion failures that invalidate a control remain failures. Ordinary
# legs keep their assertions. The sweep follows calls to run_affected_mutant.
# It does not scan shell text or mutate fixture inputs.
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
#
# The isolated-addition leg queries tracked test-suite stanza paths without
# opening them; synthetic readers pin conservative handling of printed paths.
# The link_deps control matches one literal loop prefix through its reach
# assignment. The diagnostic call and the rest of the loop body may vary.
set -u
# The expected lists are sorted with `sort` and compared with the helper's
# bytewise order: one collation, whatever the caller's locale (gh-ocannl-1431).
export LC_ALL=C
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
self_check=0
if [ "${1:-}" = --self-check ]; then self_check=1; shift; fi
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
# Written from inside lib/, so the text names no repository path: a fixture's
# `lib/dune` spelled whole would select this harness on every edit of the
# real one (gh-ocannl-2292).
(cd "$repo/lib" && cat >dune) <<'DUNE'
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
# Every negative control writes its mutant reader into the fixture as
# tools/affected-mutant.sh (gh-ocannl-2356). Visible, it is an untracked change
# of its own, so a HEAD-relative query adds the aliases it reaches and a control
# reads that for the reach change it claims. Each fixture repository ignores it
# from its first commit; a leg that leans on the exclusion asserts it.
mutant_ignored() { # DIR
  mkdir -p "$1/.git/info" && printf '%s\n' '/tools/affected-mutant.sh' >>"$1/.git/info/exclude"
}
git_q init -q && mutant_ignored "$repo" && git_q add -A && git_q commit -qm fixture || exit 2

out=$TMP/out err=$TMP/err
affected() { # ARGS... -> $out, $err, $rc
  rc=0
  (cd "$TMP" && harness_dune_colour "$repo/tools/affected-tests.sh" "$@") >"$out" 2>"$err" || rc=$?
}
trailer() { # -> $verdict: the one `affected: verdict=` line when it is stderr's last, else <none>
  verdict=$(sed -n 's/^affected: verdict=//p' "$err")
  [ -n "$verdict" ] && [ "$(tail -n 1 "$err")" = "affected: verdict=$verdict" ] || verdict='<none>'
}
alias_reader_completed() {
  # Every alias list ends with this trailer, incomplete exactly on exit 3.
  trailer
  case $verdict/$rc in
    ok/0 | cross-cutting/0 | incomplete/3 | cross-cutting,incomplete/3) return 0 ;;
    *) return 1 ;;
  esac
}
harness_reader_completed() { # QUERY: both harness queries have their own closing summary
  [ "$rc" = 0 ] && [ ! -e "$TMP/dune-called" ] || return 1
  local summary
  summary=$(tail -n 1 "$err")
  case $1 in
    --harnesses) [[ $summary =~ ^affected:\ [0-9]+\ harness\ command\(s\)\ from\ [0-9]+\ changed\ path\(s\)$ ]] ;;
    --harness-census) [[ $summary =~ ^affected:\ harness\ census:\ [0-9]+\ line\(s\)\ over\ [0-9]+\ member\(s\)$ ]] ;;
    *) return 1 ;;
  esac
}
reader_completed() { # EXPECTED_REFUSAL QUERY
  if [ -n "$1" ]; then
    [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -qF -- "$1" "$err"
  else
    case $2 in
      --harnesses | --harness-census) harness_reader_completed "$2" ;;
      *) alias_reader_completed ;;
    esac
  fi
}
pending_mutant= mutants_checked=0
run_affected_mutant() { # LABEL FROM TO [--edit FROM TO]... [--noop-refuses REASON] ARGS... -> $out, $err, $rc
  local label=$1 mutant=$repo/tools/affected-mutant.sh edits=("$2" "$3") noop_refusal=
  shift 3
  while [ "${1:-}" = --edit ]; do
    edits+=("$2" "$3")
    shift 3
  done
  if [ "${1:-}" = --noop-refuses ]; then
    noop_refusal=$2
    shift 2
  fi
  if [ -n "$pending_mutant" ]; then
    report 1 "$pending_mutant" 'no verdict was given for the mutant'
  fi
  pending_mutant=
  if ! git_q check-ignore -q tools/affected-mutant.sh; then
    report 1 "$label" 'the fixture does not ignore the scratch mutant'
    return 1
  fi
  if ! perl -0777 -e '
    my ($path, $noop) = splice @ARGV, 0, 2;
    open my $reader, "<", $path or die "cannot open $path: $!\n";
    my $text = <$reader>;
    while (@ARGV) {
      my ($from, $to) = splice @ARGV, 0, 2;
      my $replacement = $noop ? $from : $to;
      my $count = $text =~ s/\Q$from\E/$replacement/g;
      die "expected one match for [$from], got ", $count || 0, "\n" unless $count == 1;
    }
    print $text or die "cannot write mutant: $!\n";
    close STDOUT or die "cannot close mutant: $!\n";
  ' "$repo/tools/affected-tests.sh" "$self_check" "${edits[@]}" >"$mutant"; then
    report 1 "$label" 'could not construct the scratch mutant'
    return 1
  fi
  rc=0
  (cd "$TMP" && bash "$mutant" "$@") >"$out" 2>"$err" || rc=$?
  if ! rm -- "$mutant"; then
    report 1 "$label" 'could not remove the scratch mutant'
    return 1
  fi
  if [ "$self_check" = 1 ] && ! reader_completed "$noop_refusal" "${1:-}"; then
    report 1 "self-check: $label" "unusable no-op result, exit $rc; stderr: $(tail -n 3 "$err" | tr '\n' '|')"
    return 1
  fi
  pending_mutant=$label
}
report_mutant() { # RC LABEL [DETAIL]: 1 means the oracle rejected the result, 2 means it could not judge
  if [ "$2" != "$pending_mutant" ] || [ -z "$pending_mutant" ]; then
    report 1 "$2" 'no matching mutant run for this verdict'
    return
  fi
  pending_mutant=
  mutants_checked=$((mutants_checked + 1))
  if [ "$self_check" = 0 ] || [ "$1" = 2 ]; then
    report "$@"
  elif [ "$1" = 1 ]; then
    report 0 "self-check: $2 (no-op rejected)"
  else
    report 1 "self-check: $2" 'the control accepted a no-op mutant'
  fi
}
expect() { expect_aliases report "$@"; }
expect_mutant() { expect_aliases report_mutant "$@"; }
expect_aliases() { # REPORTER LABEL RC ALIAS... : the exact list, in any order, and the exit status
  local reporter=$1
  shift
  local label=$1 want_rc=$2
  shift 2
  printf '%s\n' "$@" | sed '/^$/d' | sort >"$TMP/want"
  # The printed order is ranked (link-only last); the 'link-only' leg pins it.
  sort "$out" >"$TMP/got"
  if ! alias_reader_completed; then
    "$reporter" 2 "$label" "exit $rc with the verdict trailer $verdict; stderr: $(tr '\n' '|' <"$err")"
    return
  fi
  if [ "$rc" = "$want_rc" ] && cmp -s "$TMP/want" "$TMP/got"; then
    "$reporter" 0 "$label"
  else
    "$reporter" 1 "$label" "exit $rc (want $want_rc); got [$(tr '\n' ' ' <"$out")] want [$(tr '\n' ' ' <"$TMP/want")]; stderr: $(tr '\n' '|' <"$err")"
  fi
}
control() { # [--stops REASON] LABEL FROM TO ARGS... -- WANT...: --stops, a cut whose mutant must refuse (exit 2, REASON)
  local stops=
  if [ "$1" = --stops ]; then stops=$2; shift 2; fi
  local label="negative control: $1" from=$2 to=$3
  shift 3
  local args=()
  while [ "$1" != -- ]; do args+=("$1"); shift; done
  shift
  # WANT is the shipping list: one stale here would pass a no-op mutant (gh-ocannl-2356).
  affected "${args[@]}"
  expect "$label (the shipping list)" 0 "$@"
  run_affected_mutant "$label" "$from" "$to" "${args[@]}" || return
  printf '%s\n' "$@" | sort >"$TMP/want"
  sort "$out" >"$TMP/got"
  # A mutant that stops before its verdict trailer (a cut that breaks the
  # reader) is no evidence of the reach it names.
  if [ -n "$stops" ]; then
    if [ "$rc" = 2 ] && grep -qF -- "$stops" "$err"; then report_mutant 0 "$label"
    else report_mutant 1 "$label" "want the mutant to refuse (exit 2) naming: $stops; exit $rc; stderr: $(tail -n 3 "$err" | tr '\n' '|')"; fi
  elif ! alias_reader_completed; then
    report_mutant 2 "$label" "the mutant did not run to its end: exit $rc, verdict trailer $verdict; stderr: $(tail -n 3 "$err" | tr '\n' '|')"
  elif [ "$rc" = 0 ] && cmp -s "$TMP/want" "$TMP/got"; then
    report_mutant 1 "$label" "the mutant printed the shipping list"
  else
    report_mutant 0 "$label"
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
  mkdir -p "$TMP/kept" || exit 2
  rm -rf "$repo/_build"
  cold_label='negative control: reading dune rules'"'"' streams as its answer fails on the cold build, naming the atom'
  if TMPDIR=$TMP/kept DUNE_CACHE=disabled run_affected_mutant "$cold_label" \
      'my $text = slurp("$tmp/rules-out");' 'my $text = slurp("$tmp/rules-log") . slurp("$tmp/rules-out");' \
      --files t6/dyn.expected; then
    # The refusal names the atom and keeps the answer it refused (gh-ocannl-1431).
    # Which atom comes first is dune's: 3.24 prints the noise line alone, the 3.20
    # floor echoes the command first, wrapped in ANSI dim codes.
    if [ "$rc" = 2 ] && grep -Eq 'unexpected atom in dune rules output: ".+" at byte [0-9]+: ' "$err" &&
       grep -q noise-from-an-action "$TMP"/kept/affected-dune-rules.*.txt; then
      report_mutant 0 "$cold_label"
    else report_mutant 1 "$cold_label" "exit $rc; $(cat "$err")"; fi
  fi
fi

if harness_leg 'library chain'; then
  affected --files lib/helper.ml
  # shellcheck disable=SC2086 # the list is words by design
  expect 'with forced dune colour, a library module reaches tests through a library chain, an executable of another name, a variant, a scan reading it' 0 $reached_by_core
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
  if [ "$rc" = 0 ] && grep -q '^affected: CROSS-CUTTING: 31 of the 31 per-test aliases rank first, so this change is cross-cutting: a full-suite run on a CPU-only box before the merge owns the full list but its 2 slow- alias(es), which no CI job runs: run those locally too;' "$err"; then
    affected --files lib/leaf.ml
    if [ "$rc" = 0 ] && [ -s "$out" ] && ! grep -q 'CROSS-CUTTING' "$err"; then report 0 "$cross_label"
    else report 1 "$cross_label" "a narrow change: exit $rc; $(tr '\n' '|' <"$err")"; fi
    # The slow- aliases ranked link-only count too (local review): with core's helper changed, 10 of
    # the 31 rank first by a mutant threshold, slow-gen among the link-only ones.
    local_slow_label='negative control: the slow- aliases kept local include the link-only ones'
    printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
    if run_affected_mutant "$local_slow_label" \
        'return $n_all && 2 * $n_first > $n_all;' 'return $n_all && $n_first;' HEAD -- lib; then
      if [ "$rc" = 0 ] && grep -qx 'affected: link-only: @t2/slow-gen' "$err" && grep -q 'but its 2 slow- alias(es)' "$err"; then
        report_mutant 0 "$local_slow_label"
      else report_mutant 1 "$local_slow_label" "exit $rc; $(tr '\n' '|' <"$err")"; fi
    fi
    git_q checkout -q -- lib/helper.ml || exit 2
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

if harness_leg 'OS-gated Metal probe module'; then
  # On Linux this is the actual Metal gate. On macOS reverse the comparison
  # value so the same fixture exercises both Dune enablement outcomes.
  gate=macosx
  [ "$(ocamlc -config-var system)" != macosx ] || gate=linux
  probe_dir=test/metal_probe
  ml_ext=.ml
  probe=$probe_dir/metal_hang_probe$ml_ext
  mkdir -p "$repo/$probe_dir" || exit 2
  printf 'let () = ()\n' >"$repo/$probe"
  printf '(executable (name metal_hang_probe) (optional) (modules metal_hang_probe) (enabled_if (= %%{system} %s)) (libraries fixture_probe_library_missing_everywhere))\n' \
    "$gate" >"$repo/$probe_dir/dune"
  git_q add -A test && git_q commit -qm os-gated-probe || exit 2
  affected --files "$probe"
  if alias_reader_completed && [ "$verdict" = ok ] && [ ! -s "$out" ] &&
     grep -q "^affected: $probe: stanzas disabled on system .*: executable metal_hang_probe$" "$err"; then
    report 0 'a wholly OS-disabled directory owns its module without owing aliases'
  else report 1 'a wholly OS-disabled directory owns its module without owing aliases' "exit $rc; $(cat "$out" "$err")"; fi

  printf '(executable (name other) (modules other))\n' >>"$repo/$probe_dir/dune"
  printf 'let () = ()\n' >"$repo/$probe_dir/other.ml"
  git_q add -A test && git_q commit -qm mixed-probe || exit 2
  affected --files "$probe"
  if alias_reader_completed && [ "$verdict" = incomplete ] && grep -qF "UNMAPPED: $probe" "$err"; then
    report 0 'a directory with mixed enablement keeps the ordinary mapping requirement'
  else report 1 'a directory with mixed enablement keeps the ordinary mapping requirement' "exit $rc; $(cat "$out" "$err")"; fi
  # Leave only the false guard in the probe's own file. The enabled
  # executable comes from a parent include and still makes the directory mixed.
  git_q restore --source=HEAD~1 "$probe_dir/dune" || exit 2
  parent=${probe_dir%/*}
  printf '(include extra.inc)\n' >"$repo/$parent/dune"
  printf '(subdir metal_probe (executable (name other) (modules other)))\n' >"$repo/$parent/extra.inc"
  git_q add -A test && git_q commit -qm included-mixed-probe || exit 2
  affected --files "$probe"
  if alias_reader_completed && [ "$verdict" = incomplete ] && grep -qF "UNMAPPED: $probe" "$err"; then
    report 0 'a parent include can make an apparently disabled directory mixed'
  else report 1 'a parent include can make an apparently disabled directory mixed' "exit $rc; $(cat "$out" "$err")"; fi
  git_q reset -q --hard HEAD~3 || exit 2
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
  if run_affected_mutant "$vt_label" \
      'return $n_all && 2 * $n_first > $n_all;' 'return $n_all && $n_first;' --files lib/leaf.ml; then
    trailer
    if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] && grep -q '^affected: CROSS-CUTTING:' "$err"; then report_mutant 0 "$vt_label"
    else report_mutant 1 "$vt_label" "mutant exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  fi
  # A list spanning over half the per-test aliases over a narrower first tier
  # (gh-ocannl-1996): 16 of the 31 listed, 10 first, where 15 listed is ok above
  # and the first tier alone (--referenced-only) is too. Its line names the
  # first tier as where the local run starts, and keeps the slow- aliases local.
  vt_label='a list over half the per-test aliases is cross-cutting by its list, and says how many rank first'
  affected --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml
  trailer
  if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] && [ "$(wc -l <"$out" | tr -d ' ')" = 16 ] &&
     grep -q '^affected: CROSS-CUTTING: 16 of the 31 per-test aliases are listed, 6 of them link-only, so this change is cross-cutting by its list, though only 10 rank first: a full-suite run on a CPU-only box before the merge owns the full list but its 2 slow- alias(es), which no CI job runs: run those locally too; locally run the tests that exercise the change, named in the PR body with that run -- the 10 that rank first (--referenced-only) are where to start -- and `dune build @check`$' "$err"; then
    report 0 "$vt_label"
  else report 1 "$vt_label" "exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  # Without the list's threshold that change reads ok, as it did before it.
  vt_label='negative control: without the threshold on the list, the wide list reads ok'
  if run_affected_mutant "$vt_label" \
      'return $n_all && 2 * $n_listed > $n_all;' 'return 0;' --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml; then
    trailer
    if [ "$rc" = 0 ] && [ "$verdict" = ok ] && ! grep -q 'CROSS-CUTTING' "$err"; then report_mutant 0 "$vt_label"
    else report_mutant 1 "$vt_label" "mutant exit $rc, trailer $verdict; $(tr '\n' '|' <"$err")"; fi
  fi
  # The rule itself, on the counts this repository gave when gh-ocannl-1996 was
  # decided (825 per-test aliases): the fixture is too small to hold a list
  # that is mostly link-only. Each case: first tier, printed, all, UNMAPPED.
  vt_label='the verdict on a low_level.ml- and an autotune.ml-shaped list, their --referenced-only forms and the boundaries'
  vt_detail=$(perl -e '
    local $/; open my $fh, "<", $ARGV[0] or die "unreadable\n"; my $s = <$fh>;
    $s =~ /^(sub first_tier_wide \{.*?^sub verdict \{.*?^\})/ms or die "no verdict in the script\n";
    eval "$1; 1" or die "verdict does not load: $@";
    for (["low_level: whole module", 379, 815, 825, 0, "cross-cutting"],
         ["low_level: one function", 27, 815, 825, 0, "cross-cutting"],
         ["low_level --referenced-only", 379, 379, 825, 0, "ok"],
         ["autotune", 307, 636, 825, 0, "cross-cutting"],
         ["autotune --referenced-only", 307, 307, 825, 0, "ok"],
         ["metal_backend, nearly all link-only", 27, 815, 825, 0, "cross-cutting"],
         ["test/support/ll_test, none link-only", 329, 329, 825, 0, "ok"],
         ["shape, by its first tier", 544, 633, 825, 0, "cross-cutting"],
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

if harness_leg 'cross-cutting opens stderr and leaves the verdict trailer last'; then
  affected --files dune-project lib/leaf.ml
  trailer
  if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] &&
     head -n 1 "$err" | grep -q '^affected: CROSS-CUTTING:' &&
     grep -q '^affected: CROSS-CUTTING reason: dune-project (' "$err"; then
    report 0 'the cross-cutting banner comes first, its contributor is named, and the only trailer stays last'
  else report 1 'the cross-cutting banner comes first, its contributor is named, and the only trailer stays last' \
    "exit $rc; first: $(head -n 1 "$err"); last: $(tail -n 1 "$err")"; fi
  rc=0
  (cd "$TMP" && harness_dune_colour "$repo/tools/affected-tests.sh" \
    --files lib/core.ml lib/leaf.ml lib/stubbed.ml lib/myppx.ml) >"$TMP/combined" 2>&1 || rc=$?
  if [ "$rc" = 0 ] && head -n 1 "$TMP/combined" | grep -q '^affected: CROSS-CUTTING: full list' &&
     grep -q '^affected: link-only: ' "$TMP/combined" &&
     [ "$(tail -n 1 "$TMP/combined")" = 'affected: verdict=cross-cutting' ]; then
    report 0 'the wide-list banner precedes even stdout aliases when both streams are combined'
  else report 1 'the wide-list banner precedes even stdout aliases when both streams are combined' \
    "exit $rc; first: $(head -n 1 "$TMP/combined"); last: $(tail -n 1 "$TMP/combined")"; fi
fi

if harness_leg 'cross-cutting names the ranked source path'; then
  mkdir -p "$repo/ranked"
  for i in $(seq 1 32); do
    printf '(test (name uses_%s) (modules uses_%s) (libraries core))\n' "$i" "$i" >>"$repo/ranked/dune"
    printf 'let () = ignore Core.H.used\n' >"$repo/ranked/uses_$i.ml"
  done
  git_q add ranked && git_q commit -qm rank-fixture || exit 2
  printf '\n(* A comment-only edit in the same library. *)\n' >>"$repo/lib/core.ml"
  printf 'let used = 4\nlet unused = 2\n' >"$repo/lib/helper.ml"
  affected --referenced-only --since HEAD
  trailer
  if [ "$rc" = 0 ] && [ "$verdict" = cross-cutting ] &&
     grep -q '^affected: CROSS-CUTTING reason: lib/helper.ml (' "$err"; then
    report 0 'the definition named by the first tier outranks a comment-only change to its library'
  else report 1 'the definition named by the first tier outranks a comment-only change to its library' \
    "exit $rc, verdict $verdict; $(head -n 2 "$err")"; fi
  git_q reset -q --hard HEAD~1 || exit 2
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
  inert_mutant_label="negative control: $inert_label"
  if run_affected_mutant "$inert_mutant_label" '        next if $inert{$which}{$n};' '' --referenced-only HEAD; then
    if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then report_mutant 0 "$inert_mutant_label"
    else report_mutant 1 "$inert_mutant_label" "exit $rc; $(cat "$out")"; fi
  fi
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
  changed_defs_label='negative control: the changed definitions are what rank a test naming only the module'
  printf 'let used = 1\nlet unused = 3\n' >"$repo/lib/helper.ml"
  if run_affected_mutant "$changed_defs_label" \
      'my @d = sort grep { $tok->{$_} } keys %$defs;' 'my @d = sort keys %$defs;' --referenced-only HEAD; then
    if [ "$rc" = 0 ] && grep -qx @t2/runtest-gen-v1 "$out"; then
      report_mutant 0 "$changed_defs_label"
    else report_mutant 1 "$changed_defs_label" "exit $rc; $(cat "$out")"; fi
  fi
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
  flag_refusal='--referenced-only is a flag and takes no value: --referenced-only=1'
  affected --files lib/helper.ml --referenced-only=1
  if reader_completed "$flag_refusal" --files; then
    report 0 "$flag_label"
  else report 1 "$flag_label" "exit $rc (want 2); got [$(tr '\n' ' ' <"$out")]; $(tr '\n' '|' <"$err")"; fi
  flag_mutant_label="negative control: $flag_label"
  if run_affected_mutant "$flag_mutant_label" \
      '    --referenced-only=*) die' '    --referenced-only-cut=*) die' \
      --noop-refuses "$flag_refusal" --files lib/helper.ml --referenced-only=1; then
    if [ "$rc" != 2 ] || ! grep -qF -- '--referenced-only is a flag' "$err"; then report_mutant 0 "$flag_mutant_label"
    else report_mutant 1 "$flag_mutant_label" "the mutant refused it too: exit $rc; $(tr '\n' '|' <"$err")"; fi
  fi
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
    local label="negative control: $1"
    if run_affected_mutant "$label" "$2" "$3" --referenced-only --files lib/helper.ml; then
      if [ "$rc" = 0 ] && grep -qx "$4" "$out"; then report_mutant 0 "$label"
      else report_mutant 1 "$label" "exit $rc; $(cat "$out")"; fi
    fi
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
  # On trunk the merge base keeps the core edit and leaves the side's leaf edit out.
  # shellcheck disable=SC2086
  control 'the merge-base is what --since compares with' \
    'revs=("$base")' 'revs=("$since")' \
    --since side -- $reached_by_core
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
  if [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -q "a run or query holds this worktree's lock" "$err"; then
    report 0 'tools/test-run.sh affected refuses while a run holds the worktree'
  else report 1 'tools/test-run.sh affected refuses while a run holds the worktree' "exit $rc; $(cat "$out" "$err")"; fi
fi

# Negative controls: each mutant drops one source of reach, and the leg that
# pins it must see the list change. Without them a leg could pass on a
# fixture too small to exercise the path it names.
# The stronger form, for a control over a revision-relative query: the shipping
# reader and the mutant both read the one fixture edit, and each must print its
# exact list, the two different. "Not the shipping list" alone passes a mutant
# that fails to run, or one whose list moved for a reason other than the rule
# it cuts (gh-ocannl-2356: three such controls passed a no-op mutant).
reach_control() { # LABEL FROM TO ARGS... -- SHIPPING... -- MUTANT...
  local label="negative control: $1" from=$2 to=$3
  shift 3
  local args=() ship=()
  while [ "$1" != -- ]; do args+=("$1"); shift; done
  shift
  while [ "$1" != -- ]; do ship+=("$1"); shift; done
  shift
  if [ "$(printf '%s\n' ${ship[@]+"${ship[@]}"} | sort)" = "$(printf '%s\n' "$@" | sort)" ]; then
    report 1 "$label" "the shipping and mutant lists it names are the same: it names no reach change"
    return
  fi
  affected "${args[@]}"
  expect "$label (the shipping list)" 0 ${ship[@]+"${ship[@]}"}
  run_affected_mutant "$label" "$from" "$to" "${args[@]}" || return
  expect_mutant "$label" 0 "$@"
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
    --files t1/shared.ml -- @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb
  control 'external-lib-deps'"'"' edges are what carry a ppx rewriter to a described library' \
    'push @edges, [$i, field($v, "internal_deps")];' 'push @edges, [$i, field($v, "internal_deps")] if $kind eq "tests";' \
    --files lib/myppx.ml -- @t1/runtest-m_one @t2/runtest-scan
  control 'a generated module file is what carries an .mll to its stanza' \
    'push @{$generated{$rel}}, $i unless -e $rel;' '' \
    --files lib/lx.mll -- @t1/runtest-t_core @t1/runtest-t_mid @t1/runtest-t_mid-blind @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  # shellcheck disable=SC2086
  control 'an opaque action is what lists the aliases for a changed config' \
    'my $blind = $unknown_touched || $opaque_listed;' 'my $blind = $unknown_touched;' \
    --files t3/plain.cfg -- $opaque_all
  control 'a stub archive is what carries a C stub to its library' \
    '$obj_libs{$o}{$_} = 1 for @libs' '1' \
    --files lib/stubbed_c.c -- @t3/runtest-t_stub
  control 'following generated inputs is what reaches a copied file' \
    '$v{$rel} = 1 if !-e $rel' '$v{$rel} = 1 if 0 && !-e $rel' \
    --files cfg/base.cfg -- @t3/runtest-cfgread $opaque_all
  control 'reading the files a changed dune file makes is what reaches a consumer elsewhere' \
    '$n_made++ unless $read_by{$p}{$a->{name}}++;' '' \
    --files t1/dune -- @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-t_mid @t1/runtest-t_mid-blind @t1/runtest-ta @t1/runtest-tb @t3/runtest-uses
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
  # Both rankings read the fixture's re-export of Helper from Core, which the
  # git-forms leg's trunk commit replaces: without it, gen-v1 is link-only for
  # either reader and the re-export control proves nothing (gh-ocannl-2356).
  printf 'module H = Helper\n' >"$repo/lib/core.ml"
  # shellcheck disable=SC2086
  control 'the ranking is what leaves a link-only alias off --referenced-only' \
    'return 0 unless $rank_links;' 'return 0;' \
    --referenced-only --files lib/helper.ml -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  control 'a re-export is what ranks a test naming only the re-exported name' \
    '$as{$_}{$as_name}{$x} = 1 for grep { $_ ne $as_name } @names;' '1;' \
    --referenced-only --files lib/helper.ml -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-needs_pkg @t2/runtest-orphan @t2/runtest-pkg @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen @t5/slow-usesh
  git_q checkout -q -- lib/core.ml || exit 2
  # shellcheck disable=SC2086
  control 'a rule-made target beside a golden is what makes a golden pair' \
    'grep { !/\.expected$/ && !-e in_dir($d, $_) } @mine, @near' '1' \
    --files cfg/extra.py -- $opaque_all
  # shellcheck disable=SC2086
  control --stops 'library fx.offb (opt) uses project library offa, which dune does not describe' \
    'skipping an off-build stanza'"'"'s unresolved edge is what lets the run finish' \
    'elsif ($st[$i]{undescribed}) {' 'elsif (0) {' \
    --files lib/helper.ml -- $reached_by_core
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
mkdir -p "$hrepo/tools/pkg" "$hrepo/data" "$hrepo/a/one" "$hrepo/dir" "$TMP/nodune" || exit 2
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
  tools/sib.py tools/gen.ml data/one.tbl data/two.tbl data/three.tbl data/all.cfg data/x.txt a/one/b.py x.py dir/x.txt; do : >"$hrepo/$f"; done
git -C "$hrepo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false init -q && mutant_ignored "$hrepo" &&
  git -C "$hrepo" add -A && git -C "$hrepo" -c user.name=t -c user.email=t@t -c commit.gpgsign=false commit -qm fixture ||
  exit 2
cp "$hrepo/tools/test-c.sh" "$TMP/test-c.sh.orig" || exit 2
haffected() { # ARGS... -> $out, $err, $rc, run in the fixture with the recording dune first on PATH
  rc=0
  rm -f "$TMP/dune-called"
  (cd "$TMP" && PATH=$TMP/nodune:$PATH "$hrepo/tools/affected-tests.sh" "$@") >"$out" 2>"$err" || rc=$?
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
    "# affected-reads: ../data/x.txt" "# affected-reads: --only '^1:'" "# affected-reads: data/x.txt --only '^1:${tab}x'" \
    "# affected-reads: a/**/b.py" "# affected-reads: **.py" "# affected-reads: dir/**"; do
    { head -n 2 "$TMP/test-c.sh.orig"; printf '%s\n' "$bad"; tail -n +3 "$TMP/test-c.sh.orig"; } >"$hrepo/tools/test-c.sh"
    haffected --harnesses --files data/x.txt a/one/b.py x.py dir/x.txt
    hexpect "a malformed declaration stops the run: $bad" 2
    case "$bad" in
      *'**'*)
        herr "naming its line: $bad" 'tools/test-c.sh:3:'
        herr "naming the refused pattern: $bad" 'contains unsupported `**`'
        herr "naming the supported patterns: $bad" 'use `*` within one segment and an explicit list of directories for depth'
        ;;
    esac
  done
  { head -n 2 "$TMP/test-c.sh.orig"; printf '%s\n' "# affected-reads: data/*.txt"; tail -n +3 "$TMP/test-c.sh.orig"; } >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files data/x.txt
  hexpect 'a single star within one segment stays accepted' 0 tools/test-c.sh
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

hcontrol() { # LABEL FROM TO ARGS... -- WANT...: the mutant runs to completion and changes the list
  local label="negative control: $1" from=$2 to=$3 repo=$hrepo
  shift 3
  local args=()
  while [ "$1" != -- ]; do args+=("$1"); shift; done
  shift
  haffected "${args[@]}"
  hexpect "$label (the shipping list)" 0 "$@"
  if ! harness_reader_completed "${args[0]}"; then
    report 1 "$label (the shipping reader)" "the reader did not complete the query; exit $rc; $(tail -n 3 "$err" | tr '\n' '|')"
    return
  fi
  rm -f "$TMP/dune-called"
  PATH=$TMP/nodune:$PATH run_affected_mutant "$label" "$from" "$to" "${args[@]}" || return
  if ! harness_reader_completed "${args[0]}"; then
    report_mutant 2 "$label" "the mutant did not complete the query; exit $rc; $(tail -n 3 "$err" | tr '\n' '|')"
  elif cmp -s "$TMP/want" "$out"; then
    report_mutant 1 "$label" 'the mutant printed the shipping list'
  else
    report_mutant 0 "$label"
  fi
}
# A basename with several owners (gh-ocannl-2342): three twins, harness c
# naming them in one form at a time, each twin asked alone.
twins=(tools/twin.py data/twin.py a/one/twin.py)
twin_text() { printf '%s\n' '#!/usr/bin/env bash' "$@" >"$hrepo/tools/test-c.sh"; }
twin_reach() { # LABEL TWIN...: of the twins, exactly these reach harness c
  local label=$1 t r want miss=
  shift
  for t in "${twins[@]}"; do
    haffected --harnesses --files "$t"
    want=
    for r in "$@"; do [ "$r" != "$t" ] || want=tools/test-c.sh; done
    [ "$rc" = 0 ] && [ "$(cat "$out")" = "$want" ] || miss="$miss $t: exit $rc [$(tr '\n' '|' <"$out")];"
  done
  if [ -z "$miss" ]; then report 0 "$label"; else report 1 "$label" "want only [$*]:$miss"; fi
}
twin_setup() { for t in "${twins[@]}" data/twinlib.sh; do : >"$hrepo/$t" || exit 2; done; }
twin_teardown() {
  hc_restore
  for t in "${twins[@]}" data/twinlib.sh; do unlink "$hrepo/$t" || exit 2; done
  : >"$hrepo/tools/clib.sh" || exit 2
}
self0='here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 2'
if harness_leg 'harness shared basenames'; then
  twin_setup
  twin_text 'python3 data/twin.py'
  twin_reach 'a run with a directory part reaches the owner it spells, not its namesakes' data/twin.py
  twin_text 'cat one/twin.py'
  twin_reach 'an owner ending with the whole run is the one it spells, however deep' a/one/twin.py
  twin_text 'python3 twin.py'
  twin_reach 'a bare basename reaches every owner' "${twins[@]}"
  twin_text 'python3 "$ROOT/twin.py"'
  twin_reach 'so does one under a variable the reader cannot place' "${twins[@]}"
  twin_text 'cat "${x}one/twin.py"'
  twin_reach 'a first part continuing an expansion is no path part' "${twins[@]}"
  twin_text 'cat `pwd`data/twin.py'
  twin_reach 'nor is one after a closing backquote' "${twins[@]}"
  twin_text 'x=$(echo `cat data/twin.py`) y=`data/twin.py`'
  twin_reach 'while one an opening backquote starts is' data/twin.py
  twin_text "$self0" 'python3 "$here/twin.py"'
  twin_reach "a self-directory \$here/ reaches the owner beside the harness" tools/twin.py
  twin_text "$self0" 'python3 "${here}/twin.py"'
  twin_reach 'so does ${here}/' tools/twin.py
  twin_text 'cat "$(dirname "$0")/twin.py"'
  twin_reach 'so does $(dirname "$0")/' tools/twin.py
  twin_text "$self0" 'here=$TMP' 'cat "$here/twin.py"'
  twin_reach 'a self-directory variable assigned twice places nothing' "${twins[@]}"
  twin_text "$self0; here=data" 'cat "$here/twin.py"'
  twin_reach 'nor one assigned again on its own line' "${twins[@]}"
  for form in '$S' '${S}'; do
    twin_text 'S=data' "cat \"$form/twin.py\""
    twin_reach "a variable named like the reader's own anchors places nothing: $form" "${twins[@]}"
  done
  twin_text "$self0" 'cat "data/$here/twin.py" "data/$(dirname "$0")/twin.py"'
  twin_reach 'nor does an anchor inside a path' "${twins[@]}"
  printf '%s\n' 'here=$TMP' >"$hrepo/tools/clib.sh"
  twin_text "$self0" '. tools/clib.sh' 'cat "$here/twin.py"'
  twin_reach 'nor one another file of the program assigns' "${twins[@]}"
  : >"$hrepo/tools/clib.sh"
  printf '%s\n' 'd=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)' 'cat "$d/twin.py"' >"$hrepo/data/twinlib.sh"
  twin_text '. data/twinlib.sh'
  twin_reach "in a loaded file, BASH_SOURCE's directory is that file's" data/twin.py
  printf '%s\n' 'd=$(cd "$(dirname "$0")" && pwd)' 'cat "$d/twin.py"' >"$hrepo/data/twinlib.sh"
  twin_reach "and \$0's the harness's" tools/twin.py
  # A split quoted name: the part a blank ends may end an owner's part.
  mkdir -p "$hrepo/fixture data" && : >"$hrepo/fixture data/twin.py" || exit 2
  twins+=('fixture data/twin.py')
  twin_text 'cat "fixture data/twin.py"'
  twin_reach 'a run a quoted blank splits keeps the owner it splits' data/twin.py 'fixture data/twin.py'
  unset 'twins[3]'
  unlink "$hrepo/fixture data/twin.py" && rmdir "$hrepo/fixture data" || exit 2
  twin_teardown
fi

if harness_leg 'harness fixture and heredoc literals'; then
  # gh-ocannl-2380: a harness whose only mention of a real script is inside a
  # fixture string, data heredoc, or output redirection target does not reach
  # it, while running/reading the script (and fail-closed interpreter heredocs,
  # command substitutions, pipelines to interpreters, printf -v, eval, < redirs,
  # and unclosed heredocs) still reaches it.
  fx_reach() { # LABEL WANT... : tools/sub.sh reaches exactly WANT...
    local label=$1
    shift
    haffected --harnesses --files tools/sub.sh
    hexpect "$label" 0 "$@"
  }
  # Data heredocs (quoted, double-quoted, backslash, tab-stripped, unquoted, multiple).
  twin_text 'cat >"$TMP/fixture.txt" <<'\''EOF'\''' 'affected: tools/sub.sh: named by @x/runtest-y' 'EOF'
  fx_reach 'a quoted data heredoc mentioning a script does not reach it' tools/test-a.sh
  twin_text 'cat >"$TMP/fixture.txt" <<"EOF"' 'tools/sub.sh' 'EOF'
  fx_reach 'a double-quoted data heredoc does not reach it' tools/test-a.sh
  twin_text 'cat >"$TMP/fixture.txt" <<\EOF' 'tools/sub.sh' 'EOF'
  fx_reach 'a backslash-quoted data heredoc does not reach it' tools/test-a.sh
  twin_text 'cat >"$TMP/fixture.txt" <<-'\''EOF'\''' $'\ttools/sub.sh' $'\tEOF'
  fx_reach 'a tab-indented <<- data heredoc does not reach it' tools/test-a.sh
  twin_text 'cat >"$TMP/fixture.txt" <<EOF' 'literal tools/sub.sh line' 'EOF'
  fx_reach 'an unquoted data heredoc with literal text does not reach it' tools/test-a.sh
  twin_text 'cat <<'\''A'\'' >/dev/null; cat <<'\''B'\'' >/dev/null' 'tools/sub.sh' 'A' 'tools/sub.sh' 'B'
  fx_reach 'multiple data heredocs on one line do not reach it' tools/test-a.sh
  # echo / printf fixture strings (including continuations and multiline quotes).
  twin_text "printf 'affected: tools/sub.sh: named by @x/runtest-y\\n'"
  fx_reach 'a printf fixture string mentioning a script does not reach it' tools/test-a.sh
  twin_text 'echo "tools/sub.sh"'
  fx_reach 'an echo fixture string mentioning a script does not reach it' tools/test-a.sh
  twin_text 'builtin echo tools/sub.sh; LC_ALL=C command printf -- '\''-v %s\n'\'' tools/sub.sh'
  fx_reach 'wrapped echo and printf -- fixture strings do not reach it' tools/test-a.sh
  twin_text 'printf '\''%s\n'\'' \' '  '\''tools/sub.sh'\'''
  fx_reach 'a backslash-continued printf fixture string does not reach it' tools/test-a.sh
  twin_text 'printf '\''first line' 'tools/sub.sh' 'last line'\'''
  fx_reach 'a multiline quoted printf fixture string does not reach it' tools/test-a.sh
  # Truncating output redirections writing fixture/stub files (> / >| / &>).
  twin_text 'cat >"$TMP/tools/sub.sh" <<'\''EOF'\''' '#!/bin/sh' 'echo fake' 'EOF'
  fx_reach 'writing a stub via cat >"$TMP/..." <<EOF does not reach the real script' tools/test-a.sh
  twin_text 'tee "$TMP/fixture.txt" <<'\''EOF'\''' 'tools/sub.sh' 'EOF'
  fx_reach 'an unpiped tee data heredoc does not reach the script' tools/test-a.sh
  twin_text 'printf '\''echo fake\n'\'' >"$TMP/tools/sub.sh"' ': >|"$TMP/tools/sub.sh"' 'echo fake &>"$TMP/tools/sub.sh"'
  fx_reach 'truncating output redirection targets (>, >|, &>) do not reach the real script' tools/test-a.sh
  twin_text ': >>"$TMP/tools/sub.sh"'
  fx_reach 'fail-closed: append redirection >> still reaches its target' tools/test-a.sh tools/test-c.sh
  twin_text 'echo fake &>>"$TMP/tools/sub.sh"'
  fx_reach 'fail-closed: append redirection &>> still reaches its target' tools/test-a.sh tools/test-c.sh
  twin_text 'cat >>"$TMP/tools/sub.sh" <<'\''EOF'\''' 'fake' 'EOF'
  fx_reach 'fail-closed: cat >>target <<EOF still reaches the appended target' tools/test-a.sh tools/test-c.sh
  # Data heredoc does not falsely trigger loads or clobber self-directory anchors.
  twin_text 'cat >"$TMP/fake.sh" <<'\''EOF'\''' '. tools/lib.sh' '# shellcheck source=tools/dyn.sh' 'EOF'
  haffected --harnesses --files tools/lib.sh
  hexpect 'a source line inside a data heredoc does not trigger loads' 0 tools/test-a.sh
  haffected --harnesses --files tools/dyn.sh
  hexpect 'a shellcheck source directive inside a data heredoc does not trigger loads' 0 tools/test-a.sh
  twin_setup
  twin_text "$self0" 'cat >"$TMP/fake.sh" <<'\''EOF'\''' 'here=/tmp' 'EOF' 'python3 "$here/twin.py"'
  twin_reach 'an assignment inside a data heredoc does not clobber outer $here' tools/twin.py
  twin_teardown
  # Control: running or reading the script (alone or beside a fixture) STILL reaches it.
  twin_text 'cat >"$TMP/fixture.txt" <<'\''EOF'\''' 'tools/sub.sh' 'EOF' 'bash tools/sub.sh'
  fx_reach 'control: running the script beside a fixture heredoc still reaches it' tools/test-a.sh tools/test-c.sh
  twin_text 'cat >"$TMP/f" <<'\''EOF'\'' && bash tools/sub.sh' 'plain' 'EOF'
  fx_reach 'control: a command chained after && on a heredoc opener line still reaches it' tools/test-a.sh tools/test-c.sh
  twin_text 'cat tools/sub.sh'
  fx_reach 'control: reading a script with cat still reaches it' tools/test-a.sh tools/test-c.sh
  # Fail-closed controls: executed heredocs, command substitutions, pipelines, printf -v, eval, < redirs.
  twin_text 'python3 - <<'\''PY'\''' 'open("tools/sub.sh").read()' 'PY'
  fx_reach 'fail-closed: a python3 interpreter heredoc still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'out=$(python3 - "$(pwd)" <<'\''PY'\''' 'open("tools/sub.sh").read()' 'PY' ')'
  fx_reach 'fail-closed: a python3 heredoc inside $(...) with an earlier $(pwd) still reaches it' tools/test-a.sh tools/test-c.sh
  twin_text 'bash <<'\''SH'\''' 'tools/sub.sh' 'SH'
  fx_reach 'fail-closed: a bash interpreter heredoc still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''EOF'\'' | bash' 'tools/sub.sh' 'EOF'
  fx_reach 'fail-closed: a heredoc piped to bash still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''EOF'\'' | xargs cat' 'tools/sub.sh' 'EOF'
  fx_reach 'fail-closed: a heredoc piped to xargs still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<EOF' 'res=$(tools/sub.sh)' 'EOF'
  fx_reach 'fail-closed: $(...) inside an unquoted data heredoc still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<EOF' 'res=`tools/sub.sh`' 'EOF'
  fx_reach 'fail-closed: backticks inside an unquoted data heredoc still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' "$(tools/sub.sh)"'
  fx_reach 'fail-closed: $(...) inside printf double quotes still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'echo '\''tools/sub.sh'\'' | bash'
  fx_reach 'fail-closed: echo piped to bash still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' tools/sub.sh | xargs cat'
  fx_reach 'fail-closed: printf piped to xargs still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'printf -v cmd '\''%s'\'' tools/sub.sh'
  fx_reach 'fail-closed: printf -v still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'eval "$(printf '\''%s'\'' tools/sub.sh)"'
  fx_reach 'fail-closed: eval over printf still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'p=$(printf '\''%s'\'' tools/sub.sh)'
  fx_reach 'fail-closed: variable assignment from $(printf ...) still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'echo hi < tools/sub.sh'
  fx_reach 'fail-closed: input redirection on echo still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'echo hi >"$(tools/sub.sh)"'
  fx_reach 'fail-closed: $(...) inside an output redirection target still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'grep -q x <<<"$out"' 'bash tools/sub.sh'
  fx_reach 'fail-closed: a <<< here-string does not open a heredoc' tools/test-a.sh tools/test-c.sh
  twin_text 'msg="cat <<'\''EOF'\''"' 'bash tools/sub.sh'
  fx_reach 'fail-closed: << inside quotes does not open a heredoc' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''NEVER_CLOSED'\''' 'bash tools/sub.sh'
  fx_reach 'fail-closed: an unclosed heredoc leaves its lines scanned' tools/test-a.sh tools/test-c.sh
  # Local review findings: compound &>/&>> redirects, multiline logical commands/pipelines,
  # multiline and `#`-led substitutions in unquoted heredocs, sourced scripts in unquoted
  # heredoc substitutions, and overridden echo/printf functions.
  twin_text 'bash &>"$TMP/log" <<'\''SH'\''' 'tools/sub.sh' 'SH'
  fx_reach 'fail-closed: bash &>log <<SH keeps the interpreter heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'bash \' '  <<'\''SH'\''' 'tools/sub.sh' 'SH'
  fx_reach 'fail-closed: bash continued across a backslash onto <<SH keeps the body scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''tools/sub.sh\n'\'' |' '  bash'
  fx_reach 'fail-closed: printf piped across a newline to bash still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''EOF'\'' |' 'tools/sub.sh' 'EOF' 'bash'
  fx_reach 'fail-closed: a data heredoc piped across its body to bash still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<EOF' 'x=$(' '  tools/sub.sh' ')' 'EOF'
  fx_reach 'fail-closed: multiline $(...) inside an unquoted heredoc still reaches the script' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<EOF' '# $(tools/sub.sh)' 'EOF'
  fx_reach 'fail-closed: a #-led line with $(...) inside an unquoted heredoc still reaches it' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'cat data/x.txt' >"$hrepo/tools/clib.sh"
  twin_text 'cat <<EOF' '$(. tools/clib.sh)' 'EOF'
  haffected --harnesses --files data/x.txt
  hexpect 'fail-closed: a source inside $(...) of an unquoted heredoc is followed by loads' 0 tools/test-c.sh
  : >"$hrepo/tools/clib.sh"
  twin_text 'echo() { cat "$1"; }' 'echo tools/sub.sh'
  fx_reach 'fail-closed: an echo function override keeps echo arguments scanned' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'echo() { cat "$1"; }' >"$hrepo/tools/clib.sh"
  twin_text '. tools/clib.sh' 'echo tools/sub.sh'
  fx_reach 'fail-closed: an echo function override in a sourced helper keeps echo arguments scanned' tools/test-a.sh tools/test-c.sh
  : >"$hrepo/tools/clib.sh"
  # Round 1 review findings: closed allowlist for heredocs and pipelines,
  # process substitutions <(...) / >(...), and cat function override.
  twin_text 'source /dev/stdin <<'\''SH'\''' 'tools/sub.sh' 'SH'
  fx_reach 'fail-closed: source /dev/stdin <<SH keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text '. /dev/stdin <<'\''SH'\''' 'tools/sub.sh' 'SH'
  fx_reach 'fail-closed: . /dev/stdin <<SH keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'while read -r s; do bash "$s"; done <<'\''EOF'\''' 'tools/sub.sh' 'EOF'
  fx_reach 'fail-closed: while read ... done <<EOF keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'timeout 60 python3 - <<'\''PY'\''' 'open("tools/sub.sh").read()' 'PY'
  fx_reach 'fail-closed: timeout-wrapped python3 heredoc stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''EOF'\'' | grep -q x' 'tools/sub.sh' 'EOF'
  fx_reach 'fail-closed: any piped cat heredoc stays scanned (closed allowlist)' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' tools/sub.sh | while read -r s; do bash "$s"; done'
  fx_reach 'fail-closed: printf piped to while read stays scanned (closed allowlist)' tools/test-a.sh tools/test-c.sh
  twin_text 'cat() { bash "$@"; }' 'cat <<'\''EOF'\''' 'tools/sub.sh' 'EOF'
  fx_reach 'fail-closed: a cat function override keeps cat heredocs scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'bash <(printf '\''bash tools/sub.sh\n'\'')'
  fx_reach 'fail-closed: bash <(printf ...) process substitution keeps nested printf scanned' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'cat data/x.txt' >"$hrepo/tools/clib.sh"
  twin_text 'source <(echo '\''source tools/clib.sh'\'')'
  haffected --harnesses --files data/x.txt
  hexpect 'fail-closed: source <(echo ...) process substitution keeps nested source followed by loads' 0 tools/test-c.sh
  : >"$hrepo/tools/clib.sh"
  twin_text 'echo hi >(cat tools/sub.sh)'
  fx_reach 'fail-closed: >(cat ...) process substitution inside echo stays scanned' tools/test-a.sh tools/test-c.sh
  # Round 2 review findings: multiline substitutions, output process substitutions
  # on cat/tee/echo/printf, and single-/multi-line piped compound commands.
  twin_text 'code=$(' '  printf '\''bash tools/sub.sh\n'\''' ')' 'bash -c "$code"'
  fx_reach 'fail-closed: printf inside multiline $(...) substitution stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code="$(' '  printf '\''bash tools/sub.sh\n'\''' ')"' 'bash -c "$code"'
  fx_reach 'fail-closed: printf inside double-quoted multiline "$(...)" stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(' '  cat <<'\''EOF'\''' 'tools/sub.sh' 'EOF' ')'
  fx_reach 'fail-closed: cat heredoc inside multiline $(...) stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'cat > >(bash) <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: cat > >(bash) <<SH keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'tee >(bash) <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: tee >(bash) <<SH keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'echo '\''bash tools/sub.sh'\'' > >(bash)'
  fx_reach 'fail-closed: echo redirected into >(bash) keeps its arguments scanned' tools/test-a.sh tools/test-c.sh
  twin_text '{ printf '\''bash tools/sub.sh\n'\''; } | bash'
  fx_reach 'fail-closed: { printf ...; } | bash keeps printer arguments scanned' tools/test-a.sh tools/test-c.sh
  twin_text '( echo '\''bash tools/sub.sh'\'' ) | bash'
  fx_reach 'fail-closed: ( echo ... ) | bash keeps printer arguments scanned' tools/test-a.sh tools/test-c.sh
  twin_text '{ cat <<'\''SH'\''; } | bash' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: { cat <<SH; } | bash keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text '{' '  printf '\''bash tools/sub.sh\n'\''' '} | bash'
  fx_reach 'fail-closed: multiline { ... } | bash keeps printer arguments scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'if true; then' '  printf '\''bash tools/sub.sh\n'\''' 'fi | bash'
  fx_reach 'fail-closed: multiline if ... fi | bash keeps printer arguments scanned' tools/test-a.sh tools/test-c.sh
  twin_text '{' '  cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH' '} > >(bash)'
  fx_reach 'fail-closed: multiline { cat <<SH; } > >(bash) keeps the heredoc scanned' tools/test-a.sh tools/test-c.sh
  # Round 3 review findings: unredirected printers/heredocs inside shell functions,
  # multiline quoted `cat <<EOF` not opening a bogus heredoc, and sourced cat/tee
  # overrides propagated into loads_of discovery.
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\''; }' 'emit | bash'
  fx_reach 'fail-closed: unredirected printf inside a single-line function stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() {' '  printf '\''bash tools/sub.sh\n'\''' '}' 'emit | bash'
  fx_reach 'fail-closed: unredirected printf inside a multiline function stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'function emit {' '  cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH' '}' 'emit | bash'
  fx_reach 'fail-closed: unredirected cat heredoc inside a function stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\'' 2>/dev/null; }' 'emit | bash'
  fx_reach 'fail-closed: stderr-only redirection inside a function still leaves stdout scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'make_fx() {' '  cat >"$TMP/fixture.txt" <<'\''EOF'\''' 'tools/sub.sh' 'EOF' '  printf '\''tools/sub.sh\n'\'' >"$TMP/fx2.txt"' '}'
  fx_reach 'a file-redirected fixture heredoc and printf inside a function remain excluded' tools/test-a.sh
  twin_text 'text='\''' 'cat <<EOF' ''\''' 'bash tools/sub.sh' ': '\''' 'EOF' ''\'''
  fx_reach 'fail-closed: cat <<EOF inside a multiline quoted string does not open a heredoc' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'cat() { eval "$(/bin/cat)"; }' >"$hrepo/tools/clib.sh"
  printf '%s\n' 'bash tools/sub.sh' >"$hrepo/tools/dyn.sh"
  twin_text '. tools/clib.sh' 'cat <<'\''SH'\''' 'source tools/dyn.sh' 'SH'
  fx_reach 'fail-closed: sourced cat override propagates into loads_of for executable heredocs' tools/test-a.sh tools/test-c.sh
  : >"$hrepo/tools/clib.sh"
  : >"$hrepo/tools/dyn.sh"
  # Round 4 review findings: multi-argument alias overrides, ${var:=word}
  # assignments in printer args, and variable writes inside executable heredocs.
  twin_text 'alias unused=true cat=bash' 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: multi-argument alias override (alias unused=true cat=bash) keeps heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' "${helper:=tools/sub.sh}"' 'bash "$helper"'
  fx_reach 'fail-closed: ${var:=word} assignment inside printf argument stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' "${x:-${helper=tools/sub.sh}}"' 'bash "$helper"'
  fx_reach 'fail-closed: nested ${var=word} assignment inside printf argument stays scanned' tools/test-a.sh tools/test-c.sh
  twin_setup
  twin_text "$self0" 'source /dev/stdin <<'\''SH'\''' 'here=data' 'SH' 'python3 "$here/twin.py"'
  twin_reach 'fail-closed: assignment inside an executable heredoc invalidates outer $here' "${twins[@]}"
  twin_teardown
  # Round 5 review findings: command overrides inside executable heredocs and
  # backslash-newline continuations splitting substitution openers.
  twin_text 'source /dev/stdin <<'\''SH'\''' 'cat() { bash; }' 'SH' 'cat <<'\''SH2'\''' 'bash tools/sub.sh' 'SH2'
  fx_reach 'fail-closed: cat() override defined inside an executable heredoc keeps subsequent cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<SH' '$\' '(bash tools/sub.sh)' 'SH'
  fx_reach 'fail-closed: backslash-newline splitting $(...) inside an unquoted heredoc stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' "$\' '(bash tools/sub.sh)"'
  fx_reach 'fail-closed: backslash-newline splitting $(...) inside double-quoted printf stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' $\' '(bash tools/sub.sh)'
  fx_reach 'fail-closed: backslash-newline splitting unquoted $(...) on printf stays scanned' tools/test-a.sh tools/test-c.sh
  # Round 6 review findings: procfs/variable stdout redirects in functions, and
  # case-pattern `)` inside $(...) command substitutions.
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\'' >/proc/self/fd/1; }' 'emit | bash'
  fx_reach 'fail-closed: /proc/self/fd/1 redirect inside a function keeps printer scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\'' >"$OUT"; }' 'OUT=/dev/stdout; emit | bash'
  fx_reach 'fail-closed: bare-variable redirect inside a function keeps printer scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(case x in x) printf '\''bash tools/sub.sh\n'\'';; esac)' 'bash -c "$code"'
  fx_reach 'fail-closed: case-pattern ) inside $(...) does not expose nested printf' tools/test-a.sh tools/test-c.sh
  # Round 7 review findings: persistent `exec` stdout redirects, expanded printf
  # options (`printf "$option" ...`), and command boundaries around extracted
  # substitutions (`printf '%s\n' "$(source ...)"`).
  twin_text 'exec > >(bash)' 'printf '\''bash tools/sub.sh\n'\'''
  fx_reach 'fail-closed: persistent exec > >(bash) keeps subsequent printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&3'
  fx_reach 'fail-closed: custom-fd redirect >&3 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'option=-v' 'printf "$option" code '\''bash tools/sub.sh'\''' 'bash -c "$code"'
  fx_reach 'fail-closed: expanded printf option position keeps arguments scanned' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'bash tools/sub.sh' >"$hrepo/tools/dyn.sh"
  twin_text 'printf '\''%s\n'\'' "$(source tools/dyn.sh)"'
  fx_reach 'fail-closed: $(source ...) inside printf preserves command boundary for loads_of' tools/test-a.sh tools/test-c.sh
  : >"$hrepo/tools/dyn.sh"
  # Round 8 review findings: custom fds >= 10, env option operands, and
  # backslash-continued alias declarations.
  twin_text 'exec 10> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&10'
  fx_reach 'fail-closed: custom-fd redirect >&10 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'env --unset printf bash tools/sub.sh'
  fx_reach 'fail-closed: env --unset printf consumes operand and keeps bash tools/sub.sh scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'alias \' '  cat=bash' 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: backslash-continued alias declaration keeps cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  # Round 9 review findings: persistent `exec 2>...` with `>&2`, `/dev/fd/N`
  # targets on `tee` and stdout redirections, and backslash-continued heredoc
  # openers piped on the next line.
  twin_text 'exec 2> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&2'
  fx_reach 'fail-closed: persistent exec 2> >(bash) with >&2 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' 'tee /dev/fd/3 >/dev/null <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: tee /dev/fd/3 >/dev/null <<SH keeps heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' 'cat >/dev/fd/3 <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: cat >/dev/fd/3 <<SH keeps heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<'\''SH'\'' \' '  | bash' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: backslash-continued heredoc opener piped on the next line keeps heredoc scanned' tools/test-a.sh tools/test-c.sh
  # Round 10 review findings: descriptor redirections on compound commands,
  # whitespace-separated/quoted fd duplication targets, and output context of
  # sourced shell files.
  twin_text 'exec 3> >(bash)' '{ printf '\''bash tools/sub.sh\n'\''; } >&3'
  fx_reach 'fail-closed: compound command redirected to >&3 keeps inner printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' '{' '  printf '\''bash tools/sub.sh\n'\''' '} >&3'
  fx_reach 'fail-closed: multiline compound command redirected to >&3 keeps inner printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >& 3'
  fx_reach 'fail-closed: whitespace-separated fd dup (>& 3) keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >& '\''3'\'''
  fx_reach 'fail-closed: quoted fd dup (>& '\''3'\'') keeps printf scanned' tools/test-a.sh tools/test-c.sh
  printf '%s\n' 'printf '\''bash tools/sub.sh\n'\''' >"$hrepo/tools/clib.sh"
  twin_text 'code=$(source tools/clib.sh)' 'bash -c "$code"'
  fx_reach 'fail-closed: unredirected printf inside a sourced shell file stays scanned' tools/test-a.sh tools/test-c.sh
  : >"$hrepo/tools/clib.sh"
  # Round 11 review finding: enclosing double-quote state inside `${...}` recursion.
  twin_text 'printf '\''%s\n'\'' "${missing:-'\''$(bash tools/sub.sh)'\''}"'
  fx_reach 'fail-closed: single quotes inside double-quoted ${var:-...} do not suppress $(...)' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''%s\n'\'' "${a:-'\''}$(bash tools/sub.sh)${b:-'\''}"'
  fx_reach 'fail-closed: unpaired apostrophe inside double-quoted ${...} does not hide following $(...)' tools/test-a.sh tools/test-c.sh
  # Round 12 review findings: array-element parameter-expansion assignments,
  # descriptor 0 output duplication (`>&0`), and non-brace function bodies.
  twin_text 'printf '\''%s\n'\'' "${helpers[0]:=tools/sub.sh}"' 'bash "${helpers[0]}"'
  fx_reach 'fail-closed: array-element ${arr[0]:=word} assignment inside printf stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 0> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&0'
  fx_reach 'fail-closed: descriptor-0 output duplication (>&0) keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() if true; then printf '\''bash tools/sub.sh\n'\''; fi' 'emit | bash'
  fx_reach 'fail-closed: non-brace function body (if ... fi) keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  # gh-ocannl-2432: coproc compound-command bodies, persistent `exec <>` / `1<>`,
  # multiline function/coproc opening-line commands, and `#` comments inside `$(...)`.
  twin_text 'coproc NAME { printf '\''bash tools/sub.sh\n'\''; }' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: named coproc brace body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc { printf '\''bash tools/sub.sh\n'\''; }' 'bash <&"${COPROC[0]}"'
  fx_reach 'fail-closed: unnamed coproc brace body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc "NAME" { printf '\''bash tools/sub.sh\n'\''; }' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: quoted-name coproc brace body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME ( printf '\''bash tools/sub.sh\n'\'' )' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: coproc subshell body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME if true; then printf '\''bash tools/sub.sh\n'\''; fi' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: coproc non-brace compound body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME {' '  printf '\''bash tools/sub.sh\n'\''' '}' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: multiline coproc body keeps unredirected printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME { printf '\''bash tools/sub.sh\n'\''' '  echo ok' '}' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: opening-line printf in multiline coproc body stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\''' '  echo ok' '}' 'emit | bash'
  fx_reach 'fail-closed: opening-line printf in multiline function body stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME {' '  cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH' '}' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: unredirected cat heredoc inside coproc body stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME { cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH' '}' 'bash <&"${NAME[0]}"'
  fx_reach 'fail-closed: opening-line cat heredoc inside multiline coproc body stays scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'coproc NAME { printf '\''tools/sub.sh\n'\'' >"$TMP/fx.txt"; }'
  fx_reach 'a file-redirected printf inside a coproc body remains excluded' tools/test-a.sh
  twin_text 'coproc NAME {' '  echo ok' '}' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'a top-level printf after a closed multiline coproc body remains excluded' tools/test-a.sh
  twin_text 'exec 1<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'''
  fx_reach 'fail-closed: persistent exec 1<> >(bash) keeps subsequent printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 1<> >(bash)' 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: persistent exec 1<> >(bash) keeps subsequent cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 2<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&2'
  fx_reach 'fail-closed: persistent exec 2<> >(bash) with >&2 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 3<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >"$OUT"'
  fx_reach 'fail-closed: persistent exec 3<> >(bash) with bare-variable redirect keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec {fd}<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >"$OUT"'
  fx_reach 'fail-closed: persistent exec {fd}<> >(bash) with bare-variable redirect keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''bash tools/sub.sh\n'\'' 1<>/dev/fd/3'
  fx_reach 'fail-closed: unsafe 1<>/dev/fd/3 redirection keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text '{ printf '\''bash tools/sub.sh\n'\''; } 1<>/dev/fd/3'
  fx_reach 'fail-closed: compound command redirected with 1<>/dev/fd/3 keeps inner printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'emit() { printf '\''bash tools/sub.sh\n'\'' >"$TMP/f" 1<>/dev/fd/3; }' 'emit | bash'
  fx_reach 'fail-closed: 1<> after >file inside a function cancels safe-file stdout status' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 01<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'''
  fx_reach 'fail-closed: zero-padded exec 01<> >(bash) keeps subsequent printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'exec 02<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'' >&02'
  fx_reach 'fail-closed: zero-padded exec 02<> >(bash) with >&02 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'printf '\''bash tools/sub.sh\n'\'' 01<>/dev/fd/3'
  fx_reach 'fail-closed: zero-padded 01<>/dev/fd/3 keeps printf scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$( # )' '  printf '\''bash tools/sub.sh\n'\''' ')' 'bash -c "$code"'
  fx_reach 'fail-closed: # ) comment inside $(...) does not close substitution early' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(# )' '  printf '\''bash tools/sub.sh\n'\''' ')' 'bash -c "$code"'
  fx_reach 'fail-closed: adjacent # ) comment inside $(# ) does not close substitution early' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$( (true)# )' '  printf '\''bash tools/sub.sh\n'\''' ')' 'bash -c "$code"'
  fx_reach 'fail-closed: # ) comment immediately after ) inside $(...) does not close substitution early' tools/test-a.sh tools/test-c.sh
  twin_text 'code="$( # )' '  printf '\''bash tools/sub.sh\n'\''' ')"' 'bash -c "$code"'
  fx_reach 'fail-closed: # ) comment inside double-quoted "$(...)" does not close substitution early' tools/test-a.sh tools/test-c.sh
  twin_text 'bash <( # )' '  printf '\''bash tools/sub.sh\n'\''' ')'
  fx_reach 'fail-closed: # ) comment inside <(...) process substitution does not close it early' tools/test-a.sh tools/test-c.sh
  twin_text 'cat <<EOF' '$( # )' 'bash tools/sub.sh' ')' 'EOF'
  fx_reach 'fail-closed: # ) comment inside multiline $(...) in an unquoted heredoc keeps body scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(echo $#)' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'control: $# inside $(...) is not treated as a comment' tools/test-a.sh
  twin_text 'code=$(echo ${#foo})' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'control: ${#foo} inside $(...) is not treated as a comment' tools/test-a.sh
  twin_text 'code=$(printf '\''%s'\'' foo\ #)' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'control: escaped space before # inside $(...) stays in the same word' tools/test-a.sh
  twin_text 'code=$(echo $(echo a)#)' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'control: # immediately after nested $(...) stays in the same word' tools/test-a.sh
  twin_text 'code=$( # )' '  echo ok' ')' 'printf '\''tools/sub.sh\n'\'''
  fx_reach 'control: top-level printf after closed $( # ... ) remains excluded' tools/test-a.sh
  twin_text 'echo ok # comment \' 'alias cat=bash' 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: trailing backslash in a # comment does not swallow the next line override' tools/test-a.sh tools/test-c.sh
  twin_text 'shopt -s expand_aliases' 'builtin alias cat="ba\' 'sh"' 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: double-quoted backslash-continued alias definition keeps cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'shopt -s expand_aliases' "builtin alias cat='ba\\" "sh'" 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: single-quoted backslash-continued builtin alias keeps cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'shopt -s expand_aliases' "builtin alias unused='a" "b' cat='bash'" 'cat <<'\''SH'\''' 'bash tools/sub.sh' 'SH'
  fx_reach 'fail-closed: multiline quoted alias before cat= keeps cat heredoc scanned' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(case y in' '  x) : >esac;;' '  y) printf '\''bash tools/sub.sh\n'\'';;' 'esac)' 'bash -c "$code"'
  fx_reach 'fail-closed: >esac redirection operand inside multiline $(case ...) does not pop case early' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(case y in' '  x) : > esac;;' '  y) printf '\''bash tools/sub.sh\n'\'';;' 'esac)' 'bash -c "$code"'
  fx_reach 'fail-closed: spaced > esac redirection operand inside multiline $(case ...) does not pop case early' tools/test-a.sh tools/test-c.sh
  twin_text 'code=$(case y in x) : >esac;; y) printf '\''bash tools/sub.sh\n'\'';; esac)' 'bash -c "$code"'
  fx_reach 'fail-closed: >esac redirection operand inside single-line $(case ...) does not close substitution early' tools/test-a.sh tools/test-c.sh
  hc_restore
fi

py_harness_setup() {
  twin_setup
  cp "$hrepo/tools/test-harnesses.sh" "$TMP/test-harnesses.sh.pyorig" || exit 2
  cat >"$hrepo/tools/test-harnesses.sh" <<'STUB'
#!/usr/bin/env bash
[ "$*" = --list ] || exit 2
printf '%s\n' tools/test-a.sh tools/test-b.sh tools/test-c.sh tools/test_p.py
STUB
}
py_harness_teardown() {
  cp "$TMP/test-harnesses.sh.pyorig" "$hrepo/tools/test-harnesses.sh" || exit 2
  rm -f "$hrepo/tools/test_p.py" "$hrepo/data/py_helper.py"
  twin_teardown
}
py_text() { printf '%s\n' "$@" >"$hrepo/tools/test_p.py"; }
py_twin_reach() { # LABEL TWIN...: of the twins, exactly these reach tools/test_p.py
  local label=$1 t r want miss=
  shift
  for t in "${twins[@]}"; do
    haffected --harnesses --files "$t"
    want=
    for r in "$@"; do [ "$r" != "$t" ] || want=tools/test_p.py; done
    [ "$rc" = 0 ] && [ "$(cat "$out")" = "$want" ] || miss="$miss $t: exit $rc [$(tr '\n' '|' <"$out")];"
  done
  if [ -z "$miss" ]; then report 0 "$label"; else report 1 "$label" "want only [$*]:$miss"; fi
}
if harness_leg 'harness python __file__ placement'; then
  # gh-ocannl-2380: a Python __file__-relative join in a harness program places
  # the path at that .py file's own directory, resolving shared basenames to
  # one owner and failing closed to every tied owner outside the allowlist.
  py_harness_setup
  py_text 'import os' 'p = os.path.join(os.path.dirname(__file__), "twin.py")'
  py_twin_reach 'os.path.join(os.path.dirname(__file__), "twin.py") resolves to one owner' tools/twin.py
  haffected --harness-census tools/test_p.py
  hexpect 'and the census records OWNERS=1 for it' 0 \
    "$(printf 'tools/test_p.py\ttools/test_p.py\tself\t-\t-\t-')" \
    "$(printf 'tools/test_p.py\ttools/twin.py\tnames\ttools/test_p.py:2\ttwin.py\t1')"
  py_text 'import os' 'p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "twin.py")'
  py_twin_reach 'os.path.dirname(os.path.abspath(__file__)) resolves to one owner' tools/twin.py
  py_text 'import os' 'p = os.path.join(os.path.dirname(os.path.realpath(__file__)), "twin.py")'
  py_twin_reach 'os.path.dirname(os.path.realpath(__file__)) resolves to one owner' tools/twin.py
  py_text 'import os' 'p = os.path.join(os.path.dirname(__file__), "..", "data", "twin.py")'
  py_twin_reach 'os.path.join with ".." segments resolves relative to __file__'"'"'s directory' data/twin.py
  py_text 'import os' 'p = os.path.join(os.path.dirname(os.path.dirname(__file__)), "a", "one", "twin.py")'
  py_twin_reach 'nested os.path.dirname(os.path.dirname(__file__)) resolves one directory up' a/one/twin.py
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'a single-assigned HERE = os.path.dirname(__file__) places os.path.join(HERE, ...)' tools/twin.py
  py_text 'from pathlib import Path' 'p = Path(__file__).resolve().parent / "twin.py"'
  py_twin_reach 'Path(__file__).resolve().parent / "twin.py" resolves to one owner' tools/twin.py
  py_text 'from pathlib import Path' 'HERE = Path(__file__).resolve().parent' 'p = HERE / "twin.py"'
  py_twin_reach 'a single-assigned HERE = Path(__file__).resolve().parent places HERE / ...' tools/twin.py
  py_text 'from pathlib import Path' 'p = Path(__file__).resolve().parents[1] / "data" / "twin.py"'
  py_twin_reach 'Path(__file__).resolve().parents[1] / "data" / "twin.py" resolves to one owner' data/twin.py
  py_text 'import pathlib' 'p = pathlib.Path(__file__).parent.parent / "a" / "one" / "twin.py"'
  py_twin_reach 'pathlib.Path(__file__).parent.parent / ... resolves to one owner' a/one/twin.py
  # Imported module's __file__ is its own directory (data/), not the harness's (tools/).
  printf '%s\n' 'import os' 'p = os.path.join(os.path.dirname(__file__), "twin.py")' >"$hrepo/data/py_helper.py"
  py_text 'from data import py_helper'
  py_twin_reach 'in an imported Python module, __file__ places relative to that module'"'"'s directory' data/twin.py
  rm -f "$hrepo/data/py_helper.py"
  # Fail-closed cases: keep every tied owner.
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'HERE = "/tmp"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: a reassigned HERE keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'def f(HERE):' '    return os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: a parameter shadowing HERE keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'p = os.path.join(os.path.dirname(__file__), "/twin.py")'
  py_twin_reach 'fail-closed: a leading slash in os.path.join keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'p = os.path.join(os.path.dirname(__file__), sub, "twin.py")'
  py_twin_reach 'fail-closed: a non-literal join argument keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'p = os.path.join(os.path.dirname(other.__file__), "twin.py")'
  py_twin_reach 'fail-closed: another module'"'"'s __file__ keeps every tied owner' "${twins[@]}"
  py_text 'import fake as os' 'p = os.path.join(os.path.dirname(__file__), "twin.py")'
  py_twin_reach 'fail-closed: a rebound os keeps every tied owner' "${twins[@]}"
  py_text 'class Path: pass' 'p = Path(__file__).parent / "twin.py"'
  py_twin_reach 'fail-closed: a rebound Path keeps every tied owner' "${twins[@]}"
  py_text 'from pathlib import Path' 'HERE = Path(__file__).parent' 'def f(' '    x,' '    HERE,' '):' '    return HERE / "twin.py"'
  py_twin_reach 'fail-closed: a multiline def parameter shadowing HERE keeps every tied owner' "${twins[@]}"
  py_text 'from pathlib import Path' 'HERE = Path(__file__).parent' 'if flag: HERE = other_dir' 'p = HERE / "twin.py"'
  py_twin_reach 'fail-closed: a single-line suite reassignment (if flag: HERE = ...) keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'HERE: str = "data"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: an annotated reassignment (HERE: str = ...) keeps every tied owner' "${twins[@]}"
  py_text 'from pathlib import Path' 'HERE = Path(__file__).parent' 'if flag: HERE: str = other_dir' 'p = HERE / "twin.py"'
  py_twin_reach 'fail-closed: an annotated reassignment inside a single-line suite keeps every tied owner' "${twins[@]}"
  py_text 'from pathlib import Path' 'HERE = Path(__file__).parent' 'if d[1:2]: HERE = other_dir' 'p = HERE / "twin.py"'
  py_twin_reach 'fail-closed: a single-line suite with a slice colon (if d[1:2]: HERE = ...) keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' '[HERE] = ["data"]' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: list-unpacking reassignment ([HERE] = ...) keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'a, [b, *HERE] = [1, [2, "data"]]' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: nested/starred unpacking reassignment keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'for [HERE] in [["data"]]:' '    p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: list-unpacking for target keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' '(HERE,' ' other) = ("data", 1)' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: multiline tuple unpacking reassignment keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' '[' '  HERE,' '] = ["data"]' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: multiline list unpacking reassignment keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'import sys; HERE = "data"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: assignment after semicolon following import keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'match "data":' '    case HERE: pass' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: match/case capture rebinding HERE keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'str = lambda p: "data"' 'HERE = str(os.path.dirname(__file__))' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: shadowed str wrapper keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'with ctx() as (HERE, _): pass' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: tuple-unpacked with-as rebinding HERE keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'def read_twin(): HERE = "data"; return os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: single-line def suite rebinding HERE keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'msg = f'\''{(HERE := "data")}'\''' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: walrus binding inside an f-string keeps every tied owner' "${twins[@]}"
  py_text 'import os' '__file__ = "data/driver.py"' 'p = os.path.join(os.path.dirname(__file__), "twin.py")'
  py_twin_reach 'fail-closed: a rebound __file__ keeps every tied owner' "${twins[@]}"
  # gh-ocannl-2432: Python backslash-newline string continuations preserve open
  # quote and f-string state across the escaped newline.
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'message = "hello\' '"; HERE = "data"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: reassignment after double-quoted backslash-newline string keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' "message = 'hello\\" "'; HERE = 'data'" 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: reassignment after single-quoted backslash-newline string keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'message = r"hello\' '"; HERE = "data"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: reassignment after raw backslash-newline string keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'message = "hello\\\' '"; HERE = "data"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: odd-backslash string continuation keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'msg = f"hello\' '{(HERE := '\''data'\'')}"' 'p = os.path.join(HERE, "twin.py")'
  py_twin_reach 'fail-closed: walrus binding on second line of backslash-continued f-string keeps every tied owner' "${twins[@]}"
  py_text 'import os' 's = "prefix \' 'os.path.join(os.path.dirname(__file__), '\''twin.py'\'') \' 'suffix"'
  py_twin_reach 'fail-closed: os.path.join inside a backslash-continued string keeps every tied owner' "${twins[@]}"
  py_text 'import os' 'message = "hello\' '"; p = os.path.join(os.path.dirname(__file__), "twin.py")'
  py_twin_reach 'path placement still applies to code after a backslash-continued string closes on the same line' tools/twin.py
  # Symlinked Python file: `.resolve()` / `os.path.realpath(__file__)` dereference
  # to the symlink target's directory (`data/`), while plain `dirname(__file__)`
  # stays at the symlink's own directory (`tools/`).
  printf '%s\n' 'from pathlib import Path' 'p = Path(__file__).resolve().parent / "twin.py"' >"$hrepo/data/py_helper.py"
  rm -f "$hrepo/tools/test_p.py" && ln -s ../data/py_helper.py "$hrepo/tools/test_p.py"
  py_twin_reach 'symlinked .py: Path(__file__).resolve().parent places at symlink target directory' data/twin.py
  printf '%s\n' 'import os' 'p = os.path.join(os.path.dirname(os.path.realpath(__file__)), "twin.py")' >"$hrepo/data/py_helper.py"
  py_twin_reach 'symlinked .py: os.path.dirname(os.path.realpath(__file__)) places at symlink target directory' data/twin.py
  printf '%s\n' 'import os' 'p = os.path.join(os.path.dirname(__file__), "twin.py")' >"$hrepo/data/py_helper.py"
  py_twin_reach 'symlinked .py: os.path.dirname(__file__) stays at symlink directory' tools/twin.py
  rm -f "$hrepo/tools/test_p.py" "$hrepo/data/py_helper.py"
  py_harness_teardown
fi

if harness_leg 'harness census'; then
  haffected --harness-census
  hexpect 'the census lists every member reach, the step, the line and the token naming it, sorted' 0 \
    "$(printf 'tools/test-a.sh\ttools/dyn.sh\tloads\ttools/sub.sh:3\tdyn.sh\t1')" \
    "$(printf 'tools/test-a.sh\ttools/late.sh\tloads\ttools/sub.sh:5\ttools/late.sh\t1')" \
    "$(printf 'tools/test-a.sh\ttools/lib.sh\tloads\ttools/sub.sh:2\t/lib.sh\t1')" \
    "$(printf 'tools/test-a.sh\ttools/sub.sh\tnames\ttools/test-a.sh:3\t/sub.sh\t1')" \
    "$(printf 'tools/test-a.sh\ttools/test-a.sh\tself\t-\t-\t-')" \
    "$(printf 'tools/test-b.sh\ttools/drive.py\tnames\ttools/test-b.sh:4\ttools/drive.py\t1')" \
    "$(printf 'tools/test-b.sh\ttools/gen.ml\tnames\ttools/test-b.sh:5\t./gen.exe\t1')" \
    "$(printf 'tools/test-b.sh\ttools/helper.py\tloads\ttools/drive.py:1\thelper.py\t1')" \
    "$(printf 'tools/test-b.sh\ttools/pkg/mod.py\tloads\ttools/drive.py:2\tpkg/mod.py\t1')" \
    "$(printf 'tools/test-b.sh\ttools/sib.py\tloads\ttools/drive.py:3\tsib.py\t1')" \
    "$(printf 'tools/test-b.sh\ttools/test-b.sh\tself\t-\t-\t-')" \
    "$(printf "tools/test-c.sh\tdata/*.tbl\tdeclares\ttools/test-c.sh:3\t--only '^(1|3):'\t-")" \
    "$(printf 'tools/test-c.sh\tdata/all.cfg\tdeclares\ttools/test-c.sh:5\t-\t-')" \
    "$(printf 'tools/test-c.sh\tdata/three.tbl\tnames\ttools/test-c.sh:6\tdata/three.tbl\t1')" \
    "$(printf "tools/test-c.sh\tdata/two.tbl\tdeclares\ttools/test-c.sh:4\t--only '^5 two:'\t-")" \
    "$(printf 'tools/test-c.sh\ttools/clib.sh\tloads\ttools/test-c.sh:7\ttools/clib.sh\t1')" \
    "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')"
  herr 'and counts it' 'affected: harness census: 17 line(s) over 3 member(s)'
  haffected --harness-census tools/test-a.sh
  if [ "$rc" = 0 ] && [ "$(cut -f1 "$out" | sort -u)" = tools/test-a.sh ] && [ "$(wc -l <"$out" | tr -d ' ')" = 5 ]; then
    report 0 'a census of one member lists that member alone'
  else report 1 'a census of one member lists that member alone' "exit $rc; [$(tr '\n' '|' <"$out")]"; fi
  haffected --harness-census=tools/test-b.sh
  if [ "$rc" = 0 ] && [ "$(cut -f1 "$out" | sort -u)" = tools/test-b.sh ]; then report 0 'and so does --harness-census=MEMBER'
  else report 1 'and so does --harness-census=MEMBER' "exit $rc; [$(tr '\n' '|' <"$out")]"; fi
  # The ties a census shows, and what the rule placed (the twins' leg).
  twin_setup
  twin_text 'python3 data/twin.py' 'python3 twin.py'
  haffected --harness-census tools/test-c.sh
  hexpect 'a run naming several owners counts them' 0 \
    "$(printf 'tools/test-c.sh\ta/one/twin.py\tnames\ttools/test-c.sh:3\ttwin.py\t3')" \
    "$(printf 'tools/test-c.sh\tdata/twin.py\tnames\ttools/test-c.sh:2\tdata/twin.py\t1')" \
    "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')" \
    "$(printf 'tools/test-c.sh\ttools/twin.py\tnames\ttools/test-c.sh:3\ttwin.py\t3')"
  twin_teardown
  nochange='--harness-census reads no change, so it takes no revision, --since or --files:'
  for form in "--harnesses|--harness-census and --harnesses print different lists" \
    "--gpu-gated|--harness-census prints no alias, so --gpu-gated has none to filter" \
    "--referenced-only|--harness-census prints no alias, so --referenced-only has nothing to leave off" \
    "--why all|--harness-census prints no alias, so --why has none to explain" \
    "--files data/x.txt|$nochange --files data/x.txt" "--since HEAD|$nochange --since HEAD" \
    "tools/test-a.sh HEAD|$nochange HEAD" "-- tools/test-a.sh|$nochange -- tools/test-a.sh" \
    "--harness-census=|--harness-census=: name a member, or give none" \
    "tools/test-gone.sh|--harness-census tools/test-gone.sh: not a member of tools/test-harnesses.sh --list"; do
    IFS='|' read -r opts why <<<"$form"
    # shellcheck disable=SC2086
    haffected --harness-census $opts
    hexpect "--harness-census refuses $opts" 2
    herr "saying why: $opts" "$why"
  done
  haffected --files data/x.txt --harness-census
  hexpect '--harness-census refuses a change named before it' 2
  herr 'saying it reads none' '--harness-census reads no change'
fi

if harness_leg 'harness negative controls'; then
  hcontrol 'a declaration is what reaches a path its harness code does not name' \
    '    } elsif (@d) {' '    } elsif (0) {' --harnesses --files data/one.tbl -- "tools/test-c.sh --only '^(1|3):'"
  hcontrol 'the text reaching a path is what keeps a declaration from narrowing it' \
    '    if (exists $reach{$h}{$p}) {' '    if (0) {' --harnesses --files data/three.tbl -- tools/test-c.sh
  hcontrol 'the loads of a named script are what reach the file it sources' \
    'next unless is_script($named[$i]);' 'next;' --harnesses --files tools/lib.sh -- tools/test-a.sh
  hcontrol 'comment lines are what name nothing' \
    '    next if $l[$i] =~ /^\s*#/;' '    next if 0;' --harnesses --files tools/sub.sh -- tools/test-a.sh
  hcontrol 'the boundary of a name is what keeps a longer name from naming a basename' \
    '$t =~ s{\A[^/]*/}{} or last;' '$t =~ s{\A.}{} or last;' --harnesses --files tools/sub.sh -- tools/test-a.sh
  # The shared-basename rule's parts (gh-ocannl-2342), on the twins' leg.
  twin_setup
  twin_text 'python3 data/twin.py'
  hcontrol 'the most shared trailing parts are what keep a spelled owner from its namesakes' \
    '  return @c if @c < 2;' '  return @c;' --harness-census tools/test-c.sh -- \
    "$(printf 'tools/test-c.sh\tdata/twin.py\tnames\ttools/test-c.sh:2\tdata/twin.py\t1')" \
    "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')"
  twin_text "$self0" 'python3 "$here/twin.py"'
  hcontrol "a self-directory anchor is what places \$here/twin.py" \
    '    return ($p) if defined $p && $r{$p};' '    return ($p) if 0;' --harness-census tools/test-c.sh -- \
    "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')" \
    "$(printf 'tools/test-c.sh\ttools/twin.py\tnames\ttools/test-c.sh:3\there/twin.py\t1')"
  twin_text 'cat "${x}one/twin.py"'
  hcontrol 'a fragment is what keeps an expansion-led part from counting' \
    'my ($frag, $anchor) = (1, undef);' 'my ($frag, $anchor) = (0, undef);' --harness-census tools/test-c.sh -- \
    "$(printf 'tools/test-c.sh\ta/one/twin.py\tnames\ttools/test-c.sh:2\tone/twin.py\t3')" \
    "$(printf 'tools/test-c.sh\tdata/twin.py\tnames\ttools/test-c.sh:2\tone/twin.py\t3')" \
    "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')" \
    "$(printf 'tools/test-c.sh\ttools/twin.py\tnames\ttools/test-c.sh:2\tone/twin.py\t3')"
  twin_teardown
  # Fixture/heredoc literal exclusion and Python __file__ placement controls (gh-ocannl-2380).
  twin_text 'cat >"$TMP/fixture.txt" <<'\''EOF'\''' 'tools/sub.sh' 'EOF'
  hcontrol 'a data heredoc body is excluded from reaches' \
    'if ($kind[$i] eq "hd_quoted" || $kind[$i] eq "hd_delim") { $scan[$i] = ""; $i++; next }' \
    'if (0) { $scan[$i] = ""; $i++; next }' --harnesses --files tools/sub.sh -- tools/test-a.sh
  twin_text "printf 'affected: tools/sub.sh: named\\n'"
  hcontrol 'echo and printf literal arguments are excluded from reaches' \
    'next unless $hu eq "echo" || $hu eq "printf";' 'next;' --harnesses --files tools/sub.sh -- tools/test-a.sh
  twin_text 'cat >"$TMP/tools/sub.sh" <<'\''EOF'\''' 'fake' 'EOF'
  hcontrol 'an output redirection target is excluded from reaches' \
    '} elsif ($it->[0] eq "redir_out" && defined $it->[2]) {' '} elsif (0) {' --harnesses --files tools/sub.sh -- tools/test-a.sh
  # Residual grammar edge-case controls (gh-ocannl-2432).
  twin_text 'coproc NAME { printf '\''bash tools/sub.sh\n'\''; }' 'bash <&"${NAME[0]}"'
  hcontrol 'coproc compound-command tracking keeps unredirected printf scanned' \
    'if ($w eq "coproc") {' 'if (0 && $w eq "coproc") {' \
    --harnesses --files tools/sub.sh -- tools/test-a.sh tools/test-c.sh
  twin_text 'exec 1<> >(bash)' 'printf '\''bash tools/sub.sh\n'\'''
  hcontrol 'exec <> tracking keeps subsequent printf scanned' \
    '|| ($it->[0] eq "redir_in" && $it->[1][3] =~ /\A\d*<>\z/)) {' \
    '|| 0) {' \
    --harnesses --files tools/sub.sh -- tools/test-a.sh tools/test-c.sh
  twin_text 'code=$( # )' '  printf '\''bash tools/sub.sh\n'\''' ')' 'bash -c "$code"'
  hcontrol 'comment skipping inside sh_skip_balanced keeps # ) from closing $(...) early' \
    'if (!$isq && !$idq && !$in_word && $c eq "#") {' \
    'if (0 && !$isq && !$idq && !$in_word && $c eq "#") {' \
    --harnesses --files tools/sub.sh -- tools/test-a.sh tools/test-c.sh
  hc_restore
  py_harness_setup
  py_text 'import os' 'p = os.path.join(os.path.dirname(__file__), "twin.py")'
  hcontrol 'Python __file__ placement narrows os.path.join(os.path.dirname(__file__), ...) to one owner' \
    'my ($rem, @placed) = py_placed_runs($line, $f);' 'my ($rem, @placed) = ($line);' \
    --harness-census tools/test_p.py -- \
    "$(printf 'tools/test_p.py\ttools/test_p.py\tself\t-\t-\t-')" \
    "$(printf 'tools/test_p.py\ttools/twin.py\tnames\ttools/test_p.py:2\ttwin.py\t1')"
  py_text 'import os' 'HERE = os.path.dirname(__file__)' 'message = "hello\' '"; HERE = "data"' 'p = os.path.join(HERE, "twin.py")'
  hcontrol 'Python backslash-newline string continuation exposes trailing reassignment' \
    'if (!$triple && !$was_esc) { $q = ""; $is_fstr = 0 }' 'if (!$triple) { $q = ""; $is_fstr = 0 }' \
    --harnesses --files data/twin.py -- tools/test_p.py
  py_harness_teardown
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
  if run_affected_mutant "$hs_label" '$n_harness++ if $f eq "harness-cmds";' '' --files stubs/s.mli lib/helper.ml; then
    if [ "$rc" = 0 ] && [ "$(hs_summary)" = "${hs_plain/ 1 changed path(s)/ 2 changed path(s)}" ]; then report_mutant 0 "$hs_label"
    else report_mutant 1 "$hs_label" "mutant exit $rc; $(tr '\n' '|' <"$err")"; fi
  fi
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
  h_self=$(printf 'tools/test-h.sh\ttools/test-h.sh\tself\t-\t-\t-')
  rc=0
  (cd "$TMP" && PATH=$nodune_path "$repo/tools/test-run.sh" affected --harness-census) >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 0 ] && grep -qxF -- "$h_self" "$out"; then report 0 'nor does --harness-census'
  else report 1 'nor does --harness-census' "exit $rc; [$(tr '\n' '|' <"$out")]; $(tr '\n' '|' <"$err")"; fi
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
  rc=0 rc2=0 rc3=0
  if [ "$state" = held ]; then
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --harness-census=tools/test-h.sh) >"$TMP/hlock/census" 2>&1 || rc3=$?
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --harnesses --files stubs/s.mli) >"$out" 2>"$err" || rc=$?
    (cd "$TMP" && "$repo/tools/test-run.sh" affected --files stubs/s.mli) >/dev/null 2>"$TMP/hlock/err" || rc2=$?
    echo >"$TMP/hlock/release"
  else
    rc=holder-$state
  fi
  wait "$holder"
  if [ "$rc3" = 0 ] && grep -qxF -- "$h_self" "$TMP/hlock/census"; then report 0 '--harness-census=MEMBER runs while a run holds the worktree too'
  else report 1 '--harness-census=MEMBER runs while a run holds the worktree too' "exit $rc3; $(tr '\n' '|' <"$TMP/hlock/census")"; fi
  rm -rf "${TMP:?}/hlock"
  if [ "$rc" = 0 ] && [ "$(cat "$out")" = "tools/test-h.sh --only '^1:'" ] && [ "$rc2" = 2 ]; then
    report 0 'tools/test-run.sh affected --harnesses runs while a run holds the worktree, the alias query refusing'
  else report 1 'tools/test-run.sh affected --harnesses runs while a run holds the worktree, the alias query refusing' "exit $rc, alias query $rc2; $(tr '\n' '|' <"$err")"; fi
  # The mutant must report this stub as UNMAPPED, rather than fail for another reason.
  stub_label='negative control: the harness reading it is what keeps that stub from UNMAPPED'
  if run_affected_mutant "$stub_label" 'unless (!@ix && $harness_read{$p}) {' 'unless (0) {' --files stubs/s.mli; then
    if [ "$rc" = 3 ] && grep -qF 'UNMAPPED: stubs/s.mli' "$err"; then report_mutant 0 "$stub_label"
    else report_mutant 1 "$stub_label" "mutant exit $rc; $(tr '\n' '|' <"$err")"; fi
  fi
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
if harness_leg 'isolated test addition'; then
  # Query each tracked test-suite stanza, including nested test directories.
  # The query does not read the stanza or the new module as data.
  compiler=arrayjit
  git -C "$root" ls-files 'test/**/dune' "$compiler/test/dune" >"$TMP/suite-dunes" || exit 2
  n=0 addition=isolated_2231
  while IFS= read -r stanza; do
    suite=${stanza%/dune}
    raffected --files "$stanza" "$suite/$addition.ml"
    # Independent expectations for intentional source/fixture inputs. These
    # are judgment exceptions, not a copy of the reader's computed reach;
    # a newly added suite defaults to no harnesses.
    case $suite in
      test/config) printf '%s\n' tools/test-test-run-devices.sh >"$TMP/want" ;;
      arrayjit/test) printf '%s\n' tools/test-machine-verify.sh >"$TMP/want" ;;
      *) : >"$TMP/want" ;;
    esac
    if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then
      report 0 "an isolated addition in $suite reaches only its intentional harness inputs"
    else report 1 "an isolated addition in $suite reaches only its intentional harness inputs" "exit $rc; $(cat "$out" "$err")"; fi
    n=$((n + 1))
  done <"$TMP/suite-dunes"
  [ "$n" -gt 0 ] || report 1 'tracked test-suite stanzas were queried' 'no stanzas found'
  operations=test/operations
  # A real read still reaches its harness; a printf fixture string does not (gh-ocannl-2380).
  mkdir -p "$hrepo/$operations" || exit 2
  : >"$hrepo/$operations/dune"
  printf '%s\n' '#!/usr/bin/env bash' "cat '$operations/dune'" >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files "$operations/dune" "$operations/$addition.ml"
  hexpect 'cat reading the stanza file still reaches its harness' 0 tools/test-c.sh
  printf '%s\n' '#!/usr/bin/env bash' "printf '$operations/dune'" >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files "$operations/dune" "$operations/$addition.ml"
  hexpect 'printf printing the stanza path as fixture text does not reach its harness' 0
  hc_restore
  unlink "$hrepo/$operations/dune" && rmdir "$hrepo/$operations" "$hrepo/test" || exit 2
fi

if harness_leg 'source-only reach'; then
  # gh-ocannl-2292: a source edit selects no harness that only spells its path
  # in fixture text. Fake diagnostics, trace locations and executables, table
  # labels, fixture repositories' dune files and PATH stubs spelled each of
  # these; they now use fixture-only paths and basenames. Each is spelled here
  # in two parts (`DIR/STEM EXT`), so this harness's own text names none of
  # them. Independent expectations, as in the leg above: a path defaults to
  # no harness, the judgment that none reads it; a harness that reads one
  # (a declaration included) gets its line in the case below.
  ld=lib'/dune' bd=bin'/dune' rw=tensor/row'.ml' sj=test/einsum/surjectivity'.ml'
  for path in "$sj" "$rw" "$ld" "$bd"; do printf '%s\n' "$path"; done >"$TMP/source-only"
  for src in 'tensor/tensor .ml' 'tensor/shape .ml' 'arrayjit/lib/low_level .ml' 'arrayjit/lib/autotune .ml' \
    'arrayjit/lib/metal_backend .ml' 'arrayjit/lib/backends .ml' 'arrayjit/lib/online_softmax .ml' \
    'test/support/ll_test .ml' 'test/support/generated .ml' 'test/training/cifar_conv .ml' \
    'test/training/fsm_transformer .ml' 'test/operations/online_softmax .ml' 'test/operations/env_var_deps .ml' \
    'test/operations/cc_march_census .ml' 'test/operations/cc_temp_ownership .ml' \
    'test/operations/sweep_harness_driver .ml' 'test/operations/placement_store .ml' \
    'benchmarks/test_orchestrate .py'; do
    printf '%s\n' "${src% *}${src#* }"
  done >>"$TMP/source-only"
  while IFS= read -r path; do
    raffected --files "$path"
    case $path in
      benchmarks/*.py) printf '%s\n' "tools/test-merge-build-check.sh --only '#compile-output'" >"$TMP/want" ;;
      *) : >"$TMP/want" ;;
    esac
    if [ "$rc" = 0 ] && cmp -s "$TMP/want" "$out"; then report 0 "$path alone selects only the harnesses reading it"
    else report 1 "$path alone selects only the harnesses reading it" "exit $rc; got [$(tr '\n' '|' <"$out")]"; fi
  done <"$TMP/source-only"
  # gh-ocannl-2380 / gh-ocannl-2432: printf fixture strings and direct
  # truncating output-redirected fixture/stub writers stop reaching the named
  # source paths (covering direct-redirect forms of #2303's fixture lib/dune +
  # test/operations/dune writes and an inline-redirect rewrite of #2295's
  # dispatcher manifest), while variable assignments (#2295's actual
  # `local manifest=$target/tools/test-harnesses.sh` form) and non-fixture calls
  # intentionally stay fail-closed.
  od=test/operations'/dune' th=tools/test-'harnesses.sh'
  stem=${sj##*/} stem=${stem%.ml}
  mkdir -p "$hrepo/${ld%/*}" "$hrepo/${bd%/*}" "$hrepo/${rw%/*}" "$hrepo/${sj%/*}" "$hrepo/${od%/*}" || exit 2
  for f in "$ld" "$bd" "$rw" "$sj" "$od"; do : >"$hrepo/$f" || exit 2; done
  for form in \
    "$ld|a quoted diagnostic location|printf 'File \"$ld\", line 3, characters 1-20:\\n'" \
    "$ld|a fixture repository's file under a variable prefix|cat >\"\$repo/$ld\" <<'DUNE'" \
    "$ld|a direct-redirect fixture library dune (#2303)|printf '(library (name gate))\\n' >\"\$real_co/$ld\"" \
    "$od|a direct-redirect fixture operations dune (#2303)|printf '(alias (name scans))\\n' >\"\$real_co/$od\"" \
    "$th|an inline-redirect fixture dispatcher manifest (inline form of #2295)|printf 'printf \"%%s\\\\n\" tools/first.sh\\n' >\"\$target/$th\"" \
    "$rw|a trace location|printf 'debug solve_row_ineq begin \"$rw\":4164:31:\\n'" \
    "$sj|a bare basename|printf '(diff? $stem.ml $stem.ml.corrected)\\n'" \
    "$bd|a PATH stub|cat >\"\$TMP/$bd\" <<'SH'"; do
    IFS='|' read -r path what text <<<"$form"
    printf '%s\n' '#!/usr/bin/env bash' "$text" >"$hrepo/tools/test-c.sh"
    haffected --harnesses --files "$path"
    hexpect "$what no longer reaches its harness ($path)" 0
    haffected --harness-census tools/test-c.sh
    hexpect "and the census records only self for $what ($path)" 0 \
      "$(printf 'tools/test-c.sh\ttools/test-c.sh\tself\t-\t-\t-')"
  done
  printf '%s\n' '#!/usr/bin/env bash' \
    "local manifest=\$target/$th" \
    'printf '\''printf "%%s\\n" tools/first.sh\n'\'' >"$manifest"' >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files "$th"
  hexpect "fail-closed: #2295's variable-assigned manifest path still reaches its harness ($th)" 0 tools/test-c.sh
  printf '%s\n' '#!/usr/bin/env bash' "finish('./$stem.exe', b + 'test', 6, [])" >"$hrepo/tools/test-c.sh"
  haffected --harnesses --files "$sj"
  hexpect "the module's executable in a non-fixture call still reaches its harness ($sj)" 0 tools/test-c.sh
  hc_restore
  for f in "$ld" "$bd" "$rw" "$sj" "$od"; do unlink "$hrepo/$f" || exit 2; done
  rmdir "$hrepo/${ld%/*}" "$hrepo/${bd%/*}" "$hrepo/${rw%/*}" "$hrepo/${sj%/*}" "$hrepo/${od%/*}" "$hrepo/test" || exit 2
fi

if harness_leg 'harness legs of this repository'; then
  raffected --files tools/test-run.sh
  rhas 'tools/test-run.sh reaches a member of its harness whole' tools/test-test-run-bisect.sh
  raffected --files benchmarks/fixture_digest.py
  rhas 'benchmarks/fixture_digest.py reaches the harness probing it' benchmarks/test-check-box.sh
  raffected --files tools/typecheck-scan-cases.sh
  rhas 'the scan-case type check reaches its harness whole' tools/test-typecheck-scan-cases.sh
  # Two owners of one basename (gh-ocannl-2342), spelled in two parts so this
  # harness's own text names neither: the driver harness copies the one beside
  # it (`$here/NAME`), never the summary's golden test of the same name.
  gs=gh1133_summary'.py'
  raffected --files "benchmarks/$gs"
  rhas "benchmarks/$gs reaches the harness copying it" benchmarks/test-gh1133-cells.sh
  raffected --files "test/operations/$gs"
  if [ "$rc" = 0 ] && ! grep -qxF benchmarks/test-gh1133-cells.sh "$out"; then
    report 0 "test/operations/$gs, its namesake, does not"
  else report 1 "test/operations/$gs, its namesake, does not" "exit $rc; got [$(tr '\n' '|' <"$out")]"; fi
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
    rhas "the ppx module $m reaches the #project legs" "tools/test-typecheck-scan-cases.sh --only '^(10|11|16):'"
  done
  # The legs that selector picks are exactly the ones over the repository's
  # tables, as their names say: a new such leg must join the declaration.
  if "$root/tools/test-typecheck-scan-cases.sh" --list-legs >"$TMP/legs" 2>&1; then
    sed -n 's/^LEG  //p' "$TMP/legs" >"$TMP/leg-names"
    picked=$(while IFS= read -r l; do [[ $l =~ ^(1|10): ]] && printf '%s\n' "$l"; done <"$TMP/leg-names")
    named=$(grep -F 'the repository tables' "$TMP/leg-names")
    if [ -n "$picked" ] && [ "$picked" = "$named" ]; then report 0 "'^(1|10):' selects exactly the legs over the repository tables"
    else report 1 "'^(1|10):' selects exactly the legs over the repository tables" "picked [$picked] named [$named]"; fi
    picked=$(while IFS= read -r l; do [[ $l =~ ^(10|11|16): ]] && printf '%s\n' "$l"; done <"$TMP/leg-names")
    named=$(grep -F '#project' "$TMP/leg-names")
    if [ -n "$picked" ] && [ "$picked" = "$named" ]; then report 0 "'^(10|11|16):' selects exactly the #project legs"
    else report 1 "'^(10|11|16):' selects exactly the #project legs" "picked [$picked] named [$named]"; fi
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
  expect 'a (package) dependency is read through what dune installs for it (its alias, or from 3.25 its staged files)' 0 \
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
  # Preprocessed tests are omitted by `dune describe workspace`; their
  # generated rules must still have the owner declared by the compared forms.
  printf '(test (name pc) (modules pc) (preprocess (action (cat %%{input-file}))))\n' >>"$repo/p3/dune"
  printf 'let unused = ()\n' >"$repo/p3/pc.ml"
  printf 'val unused : unit\n' >"$repo/p3/pc.mli"
  # Its shipping list (a new preprocessed test reaches only its alias and the
  # dune reader) is the first check of the control.
  # shellcheck disable=SC2086
  reach_control 'literal test module ownership is what scopes new preprocessing rules' \
    'my $mods = dfield($x->[0], "modules") // $r->{names};' 'my $mods = [];' \
    HEAD -- @p3/runtest-pc $p3_scan -- $p3_all @p3/runtest-pc $p3_scan
  probe_restore
  probe_edit 's/\(modules pa\)/(modules pa extra) (preprocess (action (cat %{input-file})))/' && {
    printf 'let unused = ()\n' >"$repo/p3/extra.ml"
    affected HEAD
    expect 'a new module in an existing preprocessed test still reaches its owner and dune reader' 0 @p3/runtest-pa "$p3_scan"
    probe_restore
  }
  # A preprocessing-looking rule with no declared owner keeps the fail-safe.
  printf '(rule (target stray.pp.ml) (action (with-stdout-to %%{target} (echo x))))\n' >>"$repo/p3/dune"
  probe_leg 'an ownerless preprocessing rule keeps the whole directory' "$p3_all $p3_scan"
  probe_edit 's/\(echo a\)\)\)/(echo b)))/' &&
    probe_leg 'an action-only edit lists the alias whose action it is' "@p3/runtest-act3 $p3_scan"
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
  printf '(library (name big) (modules big moved added))\n' >"$repo/p7/dune"
  printf 'let h () = ()\n' >"$repo/p7/added.ml"
  affected HEAD
  expect 'a module added to an existing library keeps all its linking dependents' 0 @p7/t/runtest-p7a @p7/t/runtest-p7b @p7/t/runtest-p7c
  git_q clean -qf -- p7/added.ml
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
  membership_label='negative control: reading the membership change is what ranks the split'
  split_to ''
  if run_affected_mutant "$membership_label" \
      'if (@ix && $kind eq "library" && $membership_only->($r)) {' 'if (0) {' --referenced-only HEAD; then
    if [ "$rc" = 0 ] && grep -qx @p7/t/runtest-p7c "$out"; then report_mutant 0 "$membership_label"
    else report_mutant 1 "$membership_label" "exit $rc; $(cat "$out")"; fi
  fi
  git_q checkout -q -- p7/dune || exit 2
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
  siblings_label='negative control: the name and the executable are what list a sibling'
  if run_affected_mutant "$siblings_label" \
      'grep { $st[$_]{units}{$unit} } @in }, own' 'grep { !@variant && $st[$_]{units}{$unit} } @in }, own' \
      --edit 'map { %{$runs{$_} // {}} } @{$a->{own}});' '$a->{alias} =~ /-sib-/ ? () : map { %{$runs{$_} // {}} } @{$a->{own}});' \
      --files p8/sl.ml; then
    if [ "$rc" = 0 ] && ! grep -qx @p8/runtest-sib-alt "$out" && ! grep -qx @p8/runtest-sib-low "$out"; then
      report_mutant 0 "$siblings_label"
    else report_mutant 1 "$siblings_label" "exit $rc; $(cat "$out")"; fi
  fi
  git_q reset -q --hard HEAD~1 && rm -rf "${repo:?}/p8" || exit 2
fi
# Literal dependencies (gh-ocannl-2304, gh-ocannl-2379): one stanza both names the script and
# makes dune read it. A glob or source tree can also reach the script without naming it.
if harness_leg 'literal dependencies'; then
  probe_restore
  mkdir -p "$repo/p9/tree" "$repo/p9/alias-tree" "$repo/p9/wild" || exit 2
  cat >"$repo/p9/dune" <<'DUNE'
(rule
 (alias runtest-named)
 (deps lit.sh (:drv drv.sh) (:grp (file grp.sh)) (file filed.sh) (source_tree tree) (glob_files glob*.sh))
 (action (run %{dep:act.sh})))
(rule (alias runtest-tree) (deps (source_tree tree)) (action (progn)))
(rule (alias runtest-globbed) (deps (glob_files glob*.sh)) (action (progn)))
(rule (alias runtest-split) (enabled_if false) (deps tree/t.sh) (action (progn)))
(rule (alias runtest-split) (deps (source_tree tree)) (action (progn)))
(alias (name runtest-plain) (deps plain.sh))
(alias (name runtest-via-plain) (deps (alias runtest-plain)))
(alias (name runtest-alias-tree) (enabled_if false) (deps alias-tree/t.sh))
(alias (name runtest-alias-tree) (deps (source_tree alias-tree)))
DUNE
  cat >"$repo/p9/wild/dune" <<'DUNE'
(alias (name runtest-parent) (deps ../from-parent.sh))
(alias (name runtest-globbing) (deps (glob_files *)))
DUNE
  for f in lit drv grp filed act tree/t globa plain alias-tree/t from-parent; do printf '#!/bin/sh\n' >"$repo/p9/$f.sh"; done
  git_q add -A p9 && git_q commit -qm literal-fixture || exit 2
  affected --files p9/lit.sh p9/drv.sh p9/grp.sh p9/filed.sh p9/act.sh p9/tree/t.sh p9/globa.sh
  for f in act drv filed grp lit; do
    printf 'affected: p9/%s.sh: named as a dependency by the rule of @p9/runtest-named\n' "$f"
  done >"$TMP/want-named"
  grep '^affected: p9/.*: named as a dependency by the rule of ' "$err" | LC_ALL=C sort >"$TMP/got-named"
  # Each of the reached aliases is listed, so what keeps a line away is the rule, not the reach.
  if [ "$rc" = 0 ] && grep -qx '@p9/runtest-tree' "$out" && grep -qx '@p9/runtest-globbed' "$out" &&
     grep -qx '@p9/runtest-split' "$out" &&
     cmp -s "$TMP/want-named" "$TMP/got-named"; then
    report 0 'a deps atom, a named group, a (file) and a %{dep:} name a script; a glob, a source tree and a disabled rule do not'
  else report 1 'a deps atom, a named group, a (file) and a %{dep:} name a script; a glob, a source tree and a disabled rule do not' \
    "exit $rc; got: $(cat "$TMP/got-named"); stdout: $(cat "$out")"; fi
  affected --files p9/plain.sh
  if [ "$rc" = 0 ] && grep -qxF '@p9/runtest-plain' "$out" && grep -qxF '@p9/runtest-via-plain' "$out" &&
     [ "$(grep -F 'named as a dependency by the rule of ' "$err")" = \
       'affected: p9/plain.sh: named as a dependency by the rule of @p9/runtest-plain' ]; then
    report 0 'an alias stanza naming the changed script covers it, including it through another alias gives no credit to that alias'
  else report 1 'an alias stanza naming the changed script covers it, including it through another alias gives no credit to that alias' \
    "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  affected --files p9/alias-tree/t.sh
  if [ "$rc" = 0 ] && grep -qxF '@p9/runtest-alias-tree' "$out" &&
     ! grep -qF 'named as a dependency by the rule of ' "$err"; then
    report 0 'a source tree still reaches the alias but gives no script coverage, even with a disabled stanza naming it'
  else report 1 'a source tree still reaches the alias but gives no script coverage, even with a disabled stanza naming it' \
    "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  affected --files p9/from-parent.sh
  if [ "$rc" = 0 ] && [ "$(grep '^@p9/wild/' "$out")" = '@p9/wild/runtest-parent' ] &&
     grep -qxF 'affected: p9/from-parent.sh: named as a dependency by the rule of @p9/wild/runtest-parent' "$err"; then
    report 0 'a neighboring wildcard cannot select an alias through the private probe target'
  else report 1 'a neighboring wildcard cannot select an alias through the private probe target' \
    "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  # Dune's reading alone names the source tree's script too: the stanza is what tells them apart.
  literal_label='negative control: without the converter'"'"'s names, dune reads a source tree'"'"'s script as named'
  if run_affected_mutant "$literal_label" \
      'grep { $names_of{$_} && $names_of{$_}{$p} && $reads{$_}' 'grep { $reads{$_}' --files p9/tree/t.sh; then
    if [ "$rc" = 0 ] && grep -qx 'affected: p9/tree/t.sh: named as a dependency by the rule of @p9/runtest-tree' "$err"; then
      report_mutant 0 "$literal_label"
    else report_mutant 1 "$literal_label" "exit $rc; $(cat "$err")"; fi
  fi
  owner_label='negative control: dropping the owner guard also credits the alias that includes the declaring alias'
  if run_affected_mutant "$owner_label" 'next unless $alias eq $a->{name};' '' --files p9/plain.sh; then
    if ! alias_reader_completed; then
      report_mutant 2 "$owner_label" "exit $rc; $(cat "$err")"
    elif [ "$rc" = 0 ] && grep -qxF 'affected: p9/plain.sh: named as a dependency by the rule of @p9/runtest-plain' "$err" &&
       grep -qxF 'affected: p9/plain.sh: named as a dependency by the rule of @p9/runtest-via-plain' "$err"; then
      report_mutant 0 "$owner_label"
    else report_mutant 1 "$owner_label" "exit $rc; $(cat "$err")"; fi
  fi
  mkdir -p "$repo/p9/repeated" "$repo/p9/inc-first" "$repo/p9/inc-second" || exit 2
  cat >>"$repo/p9/dune" <<'DUNE'
(subdir repeated (alias (name runtest-pair) (deps first.sh)))
(subdir repeated (alias (name runtest-pair) (deps second.sh)))
(subdir repeated (rule (alias runtest-rule-pair) (deps rule-first.sh) (action (progn))))
(subdir repeated (rule (alias runtest-rule-pair) (deps rule-second.sh) (action (progn))))
DUNE
  printf '; The two subdir stanzas contribute to this directory.\n' >"$repo/p9/repeated/dune"
  for f in first second rule-first rule-second; do printf '#!/bin/sh\n' >"$repo/p9/repeated/$f.sh"; done
  : >"$repo/p9/repeated/rule-unused.txt"
  printf '(alias (name runtest-include) (deps shared.sh))\n' >"$repo/p9/shared.inc"
  for d in inc-first inc-second; do
    printf '(include ../shared.inc)\n' >"$repo/p9/$d/dune"
    printf '#!/bin/sh\n' >"$repo/p9/$d/shared.sh"
  done
  git_q add -A p9 && git_q commit -qm repeated-subdir-fixture || exit 2
  affected --files p9/repeated/first.sh p9/repeated/second.sh
  if [ "$rc" = 0 ] && grep -qxF '@p9/repeated/runtest-pair' "$out" &&
     grep -qxF 'affected: p9/repeated/first.sh: named as a dependency by the rule of @p9/repeated/runtest-pair' "$err" &&
     grep -qxF 'affected: p9/repeated/second.sh: named as a dependency by the rule of @p9/repeated/runtest-pair' "$err"; then
    report 0 'both subdir blocks can give the same alias its own literal dependencies'
  else report 1 'both subdir blocks can give the same alias its own literal dependencies' \
    "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  for f in rule-first rule-second; do
    affected --files "p9/repeated/$f.sh"
    # shellcheck disable=SC2086
    expect "subdir targetless rule reading $f.sh reaches only its alias and blind aliases" 0 \
      @p9/repeated/runtest-rule-pair $probe_blind
    if grep -qxF "affected: p9/repeated/$f.sh: named as a dependency by the rule of @p9/repeated/runtest-rule-pair" "$err" &&
       ! grep -qE '^affected: (the probe failed|dune could not print a rule in the probe)' "$err"; then
      report 0 "subdir targetless rule credits $f.sh without probe collision"
    else report 1 "subdir targetless rule credits $f.sh without probe collision" \
      "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  done
  affected --files p9/repeated/rule-unused.txt
  # shellcheck disable=SC2086
  expect 'an unread file beside repeated subdir targetless rules lists only the blind aliases' 0 $probe_blind
  subdir_rule_label='negative control: sharing the occurrence counter across subdir blocks keeps targetless rule twins unique'
  if run_affected_mutant "$subdir_rule_label" \
      '$occ //= {};
  my @out;' \
      '$occ //= {};
  my (@out, %rule_occ);' \
      --edit \
      'my $twin = "__affected_probe_" . substr(md5_hex(join("\0", $file, $key, ++$occ->{$key})), 0, 12);
      push @out, ["rule", map {' \
      'my $twin = "__affected_probe_" . substr(md5_hex(join("\0", $file, $key, ++$rule_occ{$key})), 0, 12);
      push @out, ["rule", map {' \
      --files p9/repeated/rule-first.sh p9/repeated/rule-second.sh; then
    if ! alias_reader_completed; then
      report_mutant 2 "$subdir_rule_label" "exit $rc; $(cat "$err")"
    elif [ "$rc" = 0 ] && grep -qF 'affected: the probe failed' "$err" &&
       ! grep -qF 'named as a dependency by the rule of @p9/repeated/runtest-rule-pair' "$err"; then
      report_mutant 0 "$subdir_rule_label"
    else report_mutant 1 "$subdir_rule_label" "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  fi
  affected --files p9/inc-first/shared.sh p9/inc-second/shared.sh
  for d in inc-first inc-second; do
    if [ "$rc" = 0 ] && grep -qxF "@p9/$d/runtest-include" "$out" &&
       grep -qxF "affected: p9/$d/shared.sh: named as a dependency by the rule of @p9/$d/runtest-include" "$err"; then
      report 0 "a shared include gives credit in $d"
    else report 1 "a shared include gives credit in $d" "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"; fi
  done
  git_q reset -q --hard HEAD~2 && rm -rf "${repo:?}/p9" || exit 2
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
  # Without twins every alias action is read blind, so what lists runtest-chdir
  # for chd.txt is no longer dune's answer: runtest-tchdir, whose action reads
  # only tch.txt, is listed beside it. (The rest of that list is every alias
  # action of the fixture, too wide to name here.) The probe itself must
  # still have answered: its fallback reads every alias action blind too.
  twin_label='negative control: a twin is what makes an alias action dune'"'"'s answer'
  affected --files p1/chd.txt
  # shellcheck disable=SC2086
  expect "$twin_label (the shipping list)" 0 @p1/runtest-chdir $probe_blind
  if run_affected_mutant "$twin_label" 'my @act = grep { named($_, "action") } @fl;' 'my @act;' \
      --files p1/chd.txt; then
    if [ "$rc" = 0 ] && grep -qx @p1/runtest-chdir "$out" && grep -qx @p1/runtest-tchdir "$out" &&
       ! grep -qE '^affected: (the probe failed|dune could not print a rule in the probe)' "$err"; then report_mutant 0 "$twin_label"
    else report_mutant 1 "$twin_label" "exit $rc; got [$(tr '\n' ' ' <"$out")]; stderr: $(tr '\n' '|' <"$err")"; fi
  fi
  # Both (:include) rules of the fixture stop being blind, and inc.txt, which
  # dune does not print among their dependencies, then reaches neither.
  # shellcheck disable=SC2086
  reach_control 'an (:include) is what leaves its alias blind' \
    'if ($canon =~ /(?:^|[\s(])(:include|dynamic-run)[\s)]/) {' 'if (0) {' \
    --files p1/inc.txt -- $probe_blind -- @p2/runtest-p2blind
  # The rule needing the helper loses it; the (alias) stanza needing it has no
  # target of its own and stays listed for its directory.
  # shellcheck disable=SC2086
  reach_control 'a helper alias'"'"'s own entry is what lists the per-test alias needing it' \
    'push @al, $h;' '' \
    --files p1/hl.txt -- @p1/runtest-viahelper @p1/runtest-viastanza $probe_blind -- @p1/runtest-viastanza $probe_blind
  # An (alias) stanza depending on a file outside its directory: only the query
  # finds that dependency (one inside it is listed for its directory anyway).
  printf '(alias (name runtest-viacfg) (deps %%{workspace_root}/cfg/st.txt))\n' >>"$repo/p1/dune"
  : >"$repo/cfg/st.txt"
  git_q add -A p1 cfg && git_q commit -qm viacfg || exit 2
  # shellcheck disable=SC2086
  reach_control 'asking dune for an (alias) stanza'"'"'s dependencies is what lists it' \
    'source_query($_) for @al;' '' \
    --files cfg/st.txt -- @p1/runtest-viacfg $probe_blind -- $probe_blind
  git_q reset -q --hard HEAD~1 || exit 2
  # The controls below query HEAD, so each one's edit is the whole change the
  # two readers see (the fixture ignores the scratch mutant), and each names
  # both lists.
  printf '(test (name pc) (modules pc))\n' >>"$repo/p3/dune"
  printf 'let () = ()\n' >"$repo/p3/pc.ml"
  # shellcheck disable=SC2086
  reach_control 'comparing the dune file across revisions is what scopes a new test stanza (the base reach)' \
    'else scoped+=("$f"); fi' 'fi' \
    HEAD -- @p3/runtest-pc $p3_scan -- $p3_all @p3/runtest-pc $p3_scan
  probe_restore
  probe_edit 's/\(echo a\)\)\)/(echo b)))/' &&
    # shellcheck disable=SC2086
    reach_control 'comparing rules across the probes is what finds an action-only edit' \
      'my @delta = grep {' 'my @delta = (); 0 && grep {' \
      HEAD -- @p3/runtest-act3 $p3_scan -- $p3_scan
  probe_restore
  # A field only the test's own alias action reads: no rule of a target moves,
  # in either compiler mode, so the stanza named in the changed form is the only
  # route to its alias. A `(flags)` edit, as the 'probe dune edits' leg makes,
  # changes pa.exe's link rule and reaches it a second way.
  probe_edit 's/\(test \(name pa\) \(modules pa\)\)/(test (name pa) (modules pa) (locks p3-lock))/' &&
    # shellcheck disable=SC2086
    reach_control 'a changed stanza named as dune lists it is what reaches an edit no compared rule shows' \
      'if (@ix) { $whole{$_} = 1 for @ix; next }' 'if (@ix) { next }' \
      HEAD -- @p3/runtest-pa $p3_scan -- $p3_scan
  probe_restore
  probe_edit 's/\z/(env (_ (env-vars (P3 x))))\n/' &&
    # shellcheck disable=SC2086
    reach_control 'a changed form outside the allowlist is what keeps the whole directory' \
      'push @why, $h eq "alias" ? "a changed (alias) with an action" : "a changed ($h) stanza";' '' \
      HEAD -- $p3_all $p3_scan -- $p3_scan
  probe_restore
  probe_edit 's/\(deps h4\.txt\)/(deps h4b.txt)/' &&
    # The rule needing the helper is reached through the helper's own entry; the
    # (alias) stanza needing it also by mentioning it, which this cut keeps.
    # shellcheck disable=SC2086
    reach_control 'a changed helper (alias) stanza'"'"'s own entry is what lists the rule needing it' \
      '$direct_here{in_dir($d, $x)} = "its (alias) stanza changed";' '' \
      HEAD -- @p3/runtest-needh4 @p3/runtest-viah4 $p3_scan -- @p3/runtest-viah4 $p3_scan
  probe_restore
  probe_edit 's/\(echo h\)/(echo g)/' &&
    # shellcheck disable=SC2086
    reach_control 'a changed twin of a helper alias is what lists the alias needing it' \
      '$direct_here{in_dir($td, $_)} = "a rule of it changed" for @{$twin_of{$t} // []};' '' \
      HEAD -- @p3/runtest-needh3 $p3_scan -- $p3_scan
  probe_restore
  probe_edit 's/\(rule \(alias runtest-two\) \(targets two2\.txt\)/(rule (alias runtest-dirt3) (targets two2.txt)/' &&
    # shellcheck disable=SC2086
    reach_control 'the aliases a changed form attaches to are what list a moved target rule'"'"'s aliases' \
      '$direct_here{$x} //= "the stanzas attached to it changed"' 'my $unused = 1' \
      HEAD -- @p3/runtest-dirt3 @p3/runtest-two $p3_scan -- $p3_scan
  probe_restore
  # A removed target rule attached to no alias, owned only by its name: removing
  # runtest-two's own two2.txt rule, as the 'probe dune edits' leg does, also
  # changes what its alias attaches, which lists the alias a second way.
  printf '(rule (target pa.log) (action (with-stdout-to %%{target} (echo l))))\n' >>"$repo/p3/dune"
  git_q commit -qam pa-log || exit 2
  probe_edit 's/\(rule \(target pa\.log\)[^\n]*\n//' &&
    # shellcheck disable=SC2086
    reach_control 'the old side'"'"'s owners are what list the alias of a removed target' \
      '$direct_here{in_dir($td, $a->{alias})} = "a target it may have owned is gone";' '' \
      HEAD -- @p3/runtest-pa $p3_scan -- $p3_scan
  git_q reset -q --hard HEAD~1 || exit 2
  probe_restore
  perl -0pi -e 's/\(action \(progn\)\)\)/(action (echo i)))/' "$repo/p5/dune"
  # Its blind alias stays listed; the rest of the directory is what the verdict adds.
  # shellcheck disable=SC2086
  reach_control 'the converter'"'"'s verdict is what keeps a changed undescribed rule'"'"'s directory whole' \
    'unless ($r->{vouched}) { push @why, "a changed ($h) the probe does not describe"; next }' '' \
    HEAD -- @p5/runtest-inc5 @p5/runtest-p5ok -- @p5/runtest-inc5
  probe_restore
  probe_p6 '(include gen/x.inc)
(rule (alias runtest-p6) (action (progn)))' '(rule (alias runtest-inc6) (action (progn)))' || exit 2
  # The shipping reader exits 0 here (the 'probe fallback' leg); without the
  # alias comparison it accepts the probe, and its listing stops the run.
  probe_compare_label='negative control: comparing the probe'"'"'s aliases with the worktree'"'"'s is what keeps a probe dune does not load from stopping the run'
  if run_affected_mutant "$probe_compare_label" 'cmp -s "$tmp/worktree-aliases" "$tmp/probe-aliases"' 'true' --files p6/p6.txt; then
    if [ "$rc" = 2 ] && grep -qF 'p6/dune' "$err" &&
      grep -qE 'unexpected line from dune show aliases:|dune show aliases failed:' "$err"; then
      report_mutant 0 "$probe_compare_label"
    else report_mutant 1 "$probe_compare_label" "mutant exit $rc; $(tr '\n' '|' <"$err")"; fi
  fi
  probe_restore
fi

if harness_leg 'link_deps reach control'; then
  link_label='negative control: without its link rule reach assignment, the link_deps consumer disappears'
  link_loop='for (@{$stanza_rule{$t}}) {'
  link_assignment='$hit{$_} = $whole_hit{$_} = 1;'
  if probe_edit 's/\(echo m\)/(echo n)/'; then
    affected HEAD
    expect 'a rule action edit lists its direct and link_deps consumers' 0 @p3/runtest-reads3 @p3/runtest-lk "$p3_scan"
    if run_affected_mutant "$link_label" "$link_loop $link_assignment" "$link_loop" HEAD; then
      expect_mutant "$link_label" 0 @p3/runtest-reads3 "$p3_scan"
    fi
  fi
  probe_restore
fi


if harness_leg 'GPU-gated reach'; then
  saved_repo=$repo
  repo=$TMP/gpu-project
  mkdir -p "$repo/tools" "$repo/lib" "$repo/tests"
  cp "$HERE/affected-tests.sh" "$repo/tools/"
  printf '(lang dune 3.20)\n' >"$repo/dune-project"
  (cd "$repo/lib" && printf '(library (name gpu_core) (modules gpu_core))\n' >dune)
  printf 'let used = 1\nlet other = 2\n' >"$repo/lib/gpu_core.ml"
  cat >"$repo/tests/dune" <<'DUNE'
(test (name gpu) (modules gpu) (libraries gpu_core))
(test (name shared) (modules shared) (libraries gpu_core))
(test (name cuda_leg) (modules cuda_leg) (libraries gpu_core))
(test (name skipped_gpu) (modules skipped_gpu) (libraries gpu_core))
(test (name cpu_leg) (modules cpu_leg) (libraries gpu_core))
(test (name leg) (modules leg) (libraries gpu_core))
(test (name marked) (modules marked)
 (libraries gpu_core
  ; ocannl-backend: cc, cuda, hip -- nested field and spaced backend list
 ))
(test (name plain) (modules plain) (libraries gpu_core)
 ; ocannl-backend: cc -- CPU marker is a negative control
)
(test (name unrelated) (modules unrelated) (libraries gpu_core))
(executable (name runner) (modules runner) (libraries gpu_core))
(rule (alias runtest-rule) (action (run %{exe:runner.exe}))
 ; ocannl-backend: metal -- marker on the alias's own rule
)
DUNE
  printf 'let on_gpu = false\nlet () = if on_gpu then ignore Gpu_core.used\n' >"$repo/tests/gpu.ml"
  printf 'let leg ~when_:_ ~on:_ () = ()\nlet backend = "cc"\nlet () = leg ~when_:(backend = "cuda") ~on:backend (); ignore Gpu_core.used\n' >"$repo/tests/leg.ml"
  cat >"$repo/tests/shared.ml" <<'ML'
module Ir = struct
  module Backend_intf = struct let binds_hardware_grid () = true end
end
let has_shared = Ir.Backend_intf.binds_hardware_grid ()
let () = if has_shared then ignore Gpu_core.used
ML
  printf 'module Verdict = struct let leg ~when_:_ ~on:_ () = () end\nlet on_cuda = true\nlet () = Verdict.leg ~when_:on_cuda ~on:"runtime" (); ignore Gpu_core.used\n' >"$repo/tests/cuda_leg.ml"
  printf 'module Verdict = struct let leg ~when_:_ ~on:_ () = () end\nlet on_cpu = true\nlet () = Verdict.leg ~when_:on_cpu ~on:"runtime" (); ignore Gpu_core.used\n' >"$repo/tests/cpu_leg.ml"
  printf 'module Verdict = struct let skipped ~backend:_ _ = () end\nlet backend = "cc"\nlet () = if backend <> "cuda" then Verdict.skipped ~backend "GPU claim" else ignore Gpu_core.used\n' >"$repo/tests/skipped_gpu.ml"
  printf 'let () = ignore Gpu_core.used\n' >"$repo/tests/marked.ml"
  cp "$repo/tests/marked.ml" "$repo/tests/runner.ml"
  printf '(* on_gpu; binds_hardware_grid; leg in_leg skipped "cuda" *)\nlet text = "on_gpu binds_hardware_grid leg in_leg skipped cuda"\nlet () = ignore Gpu_core.used\n' >"$repo/tests/plain.ml"
  printf 'let on_gpu = false\nlet () = if on_gpu then ignore Gpu_core.other\n' >"$repo/tests/unrelated.ml"
  git_q init -q && mutant_ignored "$repo" && git_q add -A && git_q commit -qm fixture || exit 2
  printf 'let used = 3\nlet other = 2\n' >"$repo/lib/gpu_core.ml"
  gpu_evidence() {
    sed -n 's/^affected: gpu-gated: \([^:]*\):.*/\1/p' "$err" | sort >"$TMP/evidence"
    sort "$out" >"$TMP/selected"
    if cmp -s "$TMP/evidence" "$TMP/selected"; then
      report 0 'GPU evidence names exactly the selected aliases'
    else report 1 'GPU evidence names exactly the selected aliases' "$(cat "$TMP/evidence")"; fi
  }
  affected --gpu-gated --referenced-only HEAD
  gpu_evidence
  expect 'a changed function lists its GPU-gated callers, not CPU or unrelated callers' 0 \
    @tests/runtest-cuda_leg @tests/runtest-gpu @tests/runtest-leg @tests/runtest-marked @tests/runtest-rule @tests/runtest-shared @tests/runtest-skipped_gpu
  affected HEAD --gpu-gated
  expect 'default GPU reach retains the link-only GPU consumer' 0 \
    @tests/runtest-cuda_leg @tests/runtest-gpu @tests/runtest-leg @tests/runtest-marked @tests/runtest-rule @tests/runtest-shared @tests/runtest-skipped_gpu @tests/runtest-unrelated
  # Remove the call, keeping the gate: reach must come from the changed function.
  printf 'let on_gpu = false\nlet () = if on_gpu then ignore Gpu_core.other\n' >"$repo/tests/gpu.ml"
  affected --gpu-gated --referenced-only HEAD -- lib/gpu_core.ml
  expect 'removing the changed function call removes that gated test from the first tier' 0 \
    @tests/runtest-cuda_leg @tests/runtest-leg @tests/runtest-marked @tests/runtest-rule @tests/runtest-shared @tests/runtest-skipped_gpu
  gpu_label='negative control: disabling gate discovery loses the expected GPU aliases'
  # Keep the later reporting branch intact so the cut isolates discovery.
  gpu_discovery='
  for my $a (grep { !$hidden{$_->{name}} && $tier{$_->{name}} } @al) {'
  if run_affected_mutant "$gpu_label" 'if ($gpu_gated) {'"$gpu_discovery" 'if (0) {'"$gpu_discovery" \
      --gpu-gated --referenced-only HEAD -- lib/gpu_core.ml; then
    expect_mutant "$gpu_label" 0
  fi
  repo=$saved_repo
fi

if harness_leg 'backend-default markers'; then
  saved_repo=$repo
  repo=$TMP/backend-default-project
  mkdir -p "$repo/tools" "$repo/lib" "$repo/other_lib" "$repo/tests" || exit 2
  cp "$HERE/affected-tests.sh" "$repo/tools/" || exit 2
  printf '(lang dune 3.20)\n' >"$repo/dune-project"
  cat >"$repo/lib/dune" <<'DUNE'
(library
 ; ocannl-backend-default: plan = cuda,hip Gpu_core.gate -- GPU default seeding gate
 ; ocannl-backend-default: cpu-plan = cc Gpu_core.gate -- CPU default plan
 ; ocannl-backend-default: mixed-plan = cc,cuda Gpu_core.gate -- mixed CPU/CUDA default plan
 (name gpu_core)
 (modules gpu_core))
DUNE
  printf 'let gate = 1\nlet used = 2\n' >"$repo/lib/gpu_core.ml"
  cat >"$repo/other_lib/dune" <<'DUNE'
(library
 (name other_core)
 (modules gpu_core))
DUNE
  printf 'let gate = 100\nlet other_used = 200\n' >"$repo/other_lib/gpu_core.ml"
  cat >"$repo/tests/dune" <<'DUNE'
(
 ; ocannl-backend-default: plan -- marker immediately after opening paren, before head atom
 test
 (name declared)
 (modules declared)
 (libraries gpu_core other_core))
(test
 (name undeclared)
 (modules undeclared)
 (libraries gpu_core))
(test
 ; ocannl-backend-default: cpu-plan -- reads CPU default plan
 (name declared_cc)
 (modules declared_cc)
 (libraries gpu_core))
(test
 ; ocannl-backend: cc -- CPU-pinned reader of a cc,cuda default stays out of --gpu-gated
 ; ocannl-backend-default: mixed-plan -- reads mixed plan on cc only
 (name cc_mixed)
 (modules cc_mixed)
 (libraries gpu_core))
(executable
 ; ocannl-backend-default: plan -- slow runner inherits from executable owner
 (name exec_reader)
 (modules exec_reader)
 (libraries gpu_core))
(rule
 (alias slow-exec_reader)
 (action (run %{exe:exec_reader.exe})))
(test
 ; ocannl-backend-default: plan -- default test reads GPU plan
 (name variant_owner)
 (modules variant_owner)
 (libraries gpu_core))
(rule
 ; CPU variant: ocannl-backend: cc -- prose-prefixed CPU-only override does not inherit GPU default marker
 (target variant_owner-cc.actual)
 (action (with-stdout-to %{target} (run %{exe:variant_owner.exe}))))
(rule
 (alias runtest-variant_owner-cc)
 (action (diff variant_owner-cc.expected variant_owner-cc.actual)))
DUNE
  for f in declared undeclared declared_cc cc_mixed exec_reader variant_owner; do
    printf 'let () = ignore Gpu_core.used\n' >"$repo/tests/$f.ml"
  done
  : >"$repo/tests/variant_owner-cc.expected"
  git_q init -q && mutant_ignored "$repo" && git_q add -A && git_q commit -qm fixture || exit 2
  # 1. Diff inside `gate`: declared GPU readers route under --gpu-gated --referenced-only;
  # undeclared, CPU-only default, and CPU-pinned variant rule stay out.
  printf 'let gate = 10\nlet used = 2\n' >"$repo/lib/gpu_core.ml"
  affected --gpu-gated --referenced-only HEAD
  sed -n 's/^affected: gpu-gated: \([^:]*\):.*/\1/p' "$err" | sort >"$TMP/bd-evidence"
  sort "$out" >"$TMP/bd-selected"
  if cmp -s "$TMP/bd-evidence" "$TMP/bd-selected"; then
    report 0 'backend-default GPU evidence names exactly the selected aliases'
  else report 1 'backend-default GPU evidence names exactly the selected aliases' "$(cat "$TMP/bd-evidence")"; fi
  expect 'a diff inside a declared gate routes declared GPU readers (not undeclared, cc-default, or cc-pinned variant)' 0 \
    @tests/runtest-declared @tests/runtest-variant_owner @tests/slow-exec_reader
  # 2. --why explains the backend-default promotion.
  affected --why @tests/runtest-declared HEAD
  if [ "$rc" = 0 ] && grep -q '^affected: why @tests/runtest-declared: REFERENCED:' "$err" &&
     grep -q 'backend-default marker `plan` (cuda,hip) matches changed definition(s) Gpu_core\.gate' "$err"; then
    report 0 '--why explains a backend-default promotion with the default name and matched definition'
  else report 1 '--why explains a backend-default promotion with the default name and matched definition' "$(cat "$err")"; fi
  # 3. Plain --referenced-only promotes both GPU and CPU default readers (including cc-pinned reader of cc,cuda), keeping undeclared link-only.
  affected --referenced-only HEAD
  expect 'a CPU-only backend-default and cc-pinned mixed-default reader promote to first tier without entering --gpu-gated' 0 \
    @tests/runtest-cc_mixed @tests/runtest-declared @tests/runtest-declared_cc @tests/runtest-variant_owner @tests/slow-exec_reader
  # 4. Unfiltered --gpu-gated still omits undeclared, declared_cc, and the cc-pinned variant.
  affected --gpu-gated HEAD
  expect 'unfiltered --gpu-gated omits undeclared, CPU-only default, and cc-pinned variant' 0 \
    @tests/runtest-declared @tests/runtest-variant_owner @tests/slow-exec_reader
  # 5. Mutant negative control: neutralizing declared_touch drops the routed readers.
  bd_label='negative control: disabling declared_touch drops the backend-default GPU readers'
  if run_affected_mutant "$bd_label" \
      'sub declared_touch {   # reached owner index -> arrayref of matched default records' \
      'sub declared_touch { return [];' \
      --gpu-gated --referenced-only HEAD; then
    expect_mutant "$bd_label" 0
  fi
  # 6. Diff inside sibling definition `used` only: plan only lists `gate`, so no GPU routing.
  printf 'let gate = 1\nlet used = 20\n' >"$repo/lib/gpu_core.ml"
  affected --gpu-gated --referenced-only HEAD
  expect 'a diff in an undeclared sibling definition routes no backend-default reader under --gpu-gated' 0
  # 7. Comment-only diff in gpu_core.ml: touches no definition, so no routing.
  printf '(* harmless comment *)\nlet gate = 1\nlet used = 2\n' >"$repo/lib/gpu_core.ml"
  affected --gpu-gated --referenced-only HEAD
  expect 'a comment-only diff routes no backend-default reader' 0
  git_q checkout -q -- lib/gpu_core.ml || exit 2
  # 7b. Same-named module in another library (`other_lib/gpu_core.ml`): must NOT route `lib`'s default.
  printf 'let gate = 999\nlet other_used = 200\n' >"$repo/other_lib/gpu_core.ml"
  affected --gpu-gated --referenced-only HEAD
  expect 'a same-named module edit in another library routes no reader of the declaring library default' 0
  git_q checkout -q -- other_lib/gpu_core.ml || exit 2
  # 8. Whole-module --files: conservatively covers every definition of Gpu_core.
  affected --gpu-gated --referenced-only --files lib/gpu_core.ml
  expect 'a whole-module --files change routes declared GPU readers and still excludes undeclared and cc-only' 0 \
    @tests/runtest-declared @tests/runtest-variant_owner @tests/slow-exec_reader
  # 9. Malformed or undefined backend-default marker refuses fail-closed (exit 2).
  printf '(test\n ; ocannl-backend-default: unknown-plan -- reads undefined default\n (name bad)\n (modules bad))\n' >>"$repo/tests/dune"
  printf 'let () = ()\n' >"$repo/tests/bad.ml"
  affected --files lib/gpu_core.ml
  if [ "$rc" = 2 ] && [ ! -s "$out" ] && grep -q 'names `unknown-plan`, which no `(library)` stanza defines' "$err"; then
    report 0 'an undefined backend-default reader name fails closed (exit 2)'
  else report 1 'an undefined backend-default reader name fails closed (exit 2)' "exit $rc; $(cat "$err")"; fi
  git_q checkout -q -- tests/dune && rm -f "$repo/tests/bad.ml" || exit 2
  repo=$saved_repo
fi

if harness_leg 'sweep-driver rules emit reached-by-sweep-driver lines for literal .sh deps alone #sweep-driver'; then
  saved_repo=$repo
  repo=$TMP/sweep-driver-repo
  mkdir -p "$repo/tools" "$repo/test/operations" || exit 2
  cp "$HERE/affected-tests.sh" "$repo/tools/" || exit 2
  printf '(lang dune 3.20)\n' >"$repo/dune-project"
  cat >"$repo/test/operations/dune" <<'DUNE'
(executable (name sweep_harness_driver) (modules sweep_harness_driver))
(rule
 (alias runtest-sweep_probe)
 (deps probe.sh data.txt)
 (action (run %{dep:sweep_harness_driver.exe} %{dep:probe.sh})))
(rule
 (alias runtest-other)
 (deps probe.sh)
 (action (progn)))
DUNE
  printf 'let () = ()\n' >"$repo/test/operations/sweep_harness_driver.ml"
  printf '#!/bin/sh\n:\n' >"$repo/test/operations/probe.sh"
  printf 'data\n' >"$repo/test/operations/data.txt"
  git_q init -q && mutant_ignored "$repo" && git_q add -A && git_q commit -qm sweep-driver-fixture || exit 2
  affected --files test/operations/probe.sh
  if [ "$rc" = 0 ] &&
     grep -qxF 'affected: test/operations/probe.sh: reached by sweep driver alias @test/operations/runtest-sweep_probe' "$err" &&
     ! grep -qF 'reached by sweep driver alias @test/operations/runtest-other' "$err" &&
     grep -qxF 'affected: test/operations/probe.sh: named as a dependency by the rule of @test/operations/runtest-other' "$err"; then
    report 0 'a sweep-driver rule emits reached-by-sweep-driver for its literal .sh dep; a non-sweep rule does not'
  else
    report 1 'a sweep-driver rule emits reached-by-sweep-driver for its literal .sh dep; a non-sweep rule does not' \
      "exit $rc; stdout: $(cat "$out"); stderr: $(cat "$err")"
  fi
  affected --files test/operations/data.txt
  if [ "$rc" = 0 ] && ! grep -qF 'reached by sweep driver alias' "$err"; then
    report 0 'a non-shell dep of a sweep-driver rule emits no reached-by-sweep-driver line'
  else
    report 1 'a non-shell dep of a sweep-driver rule emits no reached-by-sweep-driver line' \
      "exit $rc; stderr: $(cat "$err")"
  fi
  OCANNL_TOOL_AFFECTED_PROBE=0 affected --files test/operations/probe.sh
  if [ "$rc" = 0 ] &&
     grep -qxF 'affected: test/operations/probe.sh: sweep-driver reach unavailable (dune probe did not run)' "$err"; then
    report 0 'disabling the Dune probe reports sweep-driver reach unavailable for a changed .sh file'
  else
    report 1 'disabling the Dune probe reports sweep-driver reach unavailable for a changed .sh file' \
      "exit $rc; stderr: $(cat "$err")"
  fi
  sweep_label='negative control: disabling runs_sweep_driver drops the sweep-driver reach line'
  if run_affected_mutant "$sweep_label" \
      'if runs_sweep_driver($by_name{$n});' \
      'if 0;' --files test/operations/probe.sh; then
    if [ "$rc" = 0 ] && ! grep -qF 'reached by sweep driver alias' "$err"; then
      report_mutant 0 "$sweep_label"
    else
      report_mutant 1 "$sweep_label" "exit $rc; $(cat "$err")"
    fi
  fi
  repo=$saved_repo
fi

if [ -n "$pending_mutant" ]; then
  report 1 "$pending_mutant" 'no verdict was given for the mutant'
fi
if [ "$self_check" = 1 ] && [ "$harness_list" = 0 ] && [ "$mutants_checked" = 0 ]; then
  report 1 'self-check' 'no mutant controls were checked'
fi
finish

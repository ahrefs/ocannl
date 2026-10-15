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
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
HERE=$(cd "$(dirname "$0")" && pwd)
command -v dune >/dev/null 2>&1 || . "$HERE/opam-env.sh"
harness_require dune perl git
harness_scratch test-affected-tests
unset DUNE_BUILD_DIR DUNE_ROOT
export OCANNL_TOOL_FLEET_WORKER=none OCANNL_TOOL_TEST_RUNS=$TMP/runs

repo=$TMP/proj
mkdir -p "$repo/tools" "$repo/lib" "$repo/mid" "$repo/t1" "$repo/t2" "$repo/t3" "$repo/docs" || exit 2
cp "$HERE/affected-tests.sh" "$HERE/test-run.sh" "$repo/tools/" || exit 2
# Whatever test-run.sh sources, derived from it rather than listed here.
staged=0
while IFS= read -r rel; do
  mkdir -p "$repo/$(dirname "$rel")" && cp "$HERE/../$rel" "$repo/$rel" || exit 2
  staged=$((staged + 1))
done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$HERE/test-run.sh")
[ "$staged" -ge 3 ] || exit 2
printf '(lang dune 3.20)\n' >"$repo/dune-project"
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
(rule (alias scans) (deps (alias runtest-scan)) (action (progn)))
DUNE
cat >"$repo/t3/dune" <<'DUNE'
(test (name t_plain) (modules t_plain) (deps plain.cfg))
(test (name t_stub) (modules t_stub) (libraries stubbed))
(executable (name cexe) (modules cexe) (foreign_stubs (language c) (names cexe_c)))
(executable (name lister) (modules lister))
(rule
 (target neg.actual)
 (deps (glob_files %{project_root}/mid/[!c]*.ml))
 (action (with-stdout-to %{target} (run %{exe:lister.exe}))))
(rule (alias runtest-neg) (action (diff neg.expected neg.actual)))
DUNE
for f in lib/core.ml lib/helper.ml lib/leaf.ml lib/myppx.ml lib/optlib.ml lib/stubbed.ml \
  t3/t_plain.ml t3/t_stub.ml t3/cexe.ml t3/lister.ml mid/mid.ml mid/choice.core.ml mid/choice.none.ml \
  t1/t_core.ml t1/t_leaf.ml t1/ta.ml t1/tb.ml t1/shared.ml t1/m_one.ml t1/m_two.ml \
  t2/gen.ml t2/scan.ml; do
  printf 'let () = ()\n' >"$repo/$f"
done
for f in t2/gen-v1.expected t2/scenario_x.expected t2/scan.expected t3/neg.expected t3/plain.cfg \
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
reached_by_core='@t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-orphan @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen'

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
    @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-orphan @t2/runtest-scenario_x @t2/slow-gen
fi

if harness_leg 'golden'; then
  # The diff reading a golden is the alias's own action, which dune does not
  # dump; the golden's copy into the build is one of the alias's own targets.
  affected --files t2/gen-v1.expected
  expect 'a golden lists the alias diffing it (and its directory'"'"'s aliases with undumped actions)' 0 \
    @t2/runtest-gen-v1 @t2/runtest-orphan @t2/slow-gen
fi

if harness_leg 'dune-project'; then
  affected --files dune-project
  expect 'dune-project reaches every per-test alias beneath it' 0 \
    @t1/runtest-m_one @t1/runtest-t_core @t1/runtest-t_leaf @t1/runtest-ta @t1/runtest-tb \
    @t2/runtest-gen-v1 @t2/runtest-orphan @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen \
    @t3/runtest-neg @t3/runtest-t_plain @t3/runtest-t_stub
fi

if harness_leg 'opaque action'; then
  # t_plain has no golden: dune runs it as the alias's own action under the
  # stanza's (deps), which `dune rules` does not show.
  affected --files t3/plain.cfg
  expect 'a non-module path lists the aliases of its directory whose actions dune does not dump' 0 \
    @t3/runtest-t_plain @t3/runtest-t_stub
fi

if harness_leg 'foreign stubs'; then
  affected --files lib/stubbed_c.c
  expect 'a C stub reaches the library whose stub archive bundles its object' 0 @t3/runtest-t_stub
  affected --files t3/cexe_c.c
  expect 'a C stub no archive bundles reaches every executable and test stanza of its directory' 0 \
    @t3/runtest-neg @t3/runtest-t_plain @t3/runtest-t_stub
fi

if harness_leg 'negated glob'; then
  affected --files mid/mid.ml
  expect 'a [!set] glob matches outside its set (the select-choice leg pins the inside)' 0 \
    @t1/runtest-t_core @t3/runtest-neg
fi

if harness_leg 'undescribed stanza'; then
  affected --files lib/optlib.ml
  if [ "$rc" = 3 ] && grep -q 'UNMAPPED: lib/optlib.ml' "$err" &&
     grep -q 'does not build: library optlib (lib)' "$err" && [ "$(cat "$out")" = @t2/runtest-scan ]; then
    report 0 'a module of a library this build does not compile is UNMAPPED, exit 3, its rule readers still listed'
  else report 1 'a module of a library this build does not compile is UNMAPPED, exit 3, its rule readers still listed' "exit $rc; $(cat "$out" "$err")"; fi
fi

if harness_leg 'dune file'; then
  affected --files t2/dune
  expect 'a changed dune file lists every per-test alias of its directory, never an aggregate' 0 \
    @t2/runtest-gen-v1 @t2/runtest-orphan @t2/runtest-scan @t2/runtest-scenario_x @t2/slow-gen
fi

if harness_leg 'data read'; then
  affected --files docs/notes.md
  expect 'a non-OCaml file a rule reads lists that rule'"'"'s alias' 0 @t2/runtest-scan
  affected --files docs/other.md
  if [ "$rc" = 0 ] && [ ! -s "$out" ] && grep -q 'unread: docs/other.md' "$err"; then
    report 0 'a file nothing reads lists nothing and says so'
  else report 1 'a file nothing reads lists nothing and says so' "exit $rc; $(cat "$out" "$err")"; fi
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
  rc=0
  (cd "$TMP" && SHIM_LOCK_PROBE=$TMP/lock-probe PATH="$TMP/shim:$PATH" "$repo/tools/test-run.sh" affected --files lib/helper.ml) \
    >"$out" 2>"$err" || rc=$?
  if [ "$rc" = 0 ] && [ "$(sort -u "$TMP/lock-probe")" = held ]; then
    report 0 'the worktree lock is held while affected'"'"'s dune runs, so a run starting then refuses'
  else report 1 'the worktree lock is held while affected'"'"'s dune runs, so a run starting then refuses' "exit $rc; probe: $(cat "$TMP/lock-probe" 2>&1)"; fi
  lock=$("$repo/tools/test-run.sh" paths lock) || exit 2
  mkdir -p "$(dirname "$lock")" || exit 2
  # Hold the worktree's run lock as a run would, until the fifo is written.
  mkfifo "$TMP/release" || exit 2
  perl -e 'use Fcntl ":flock"; open my $fh, ">>", $ARGV[0] or exit 1; flock($fh, LOCK_EX) or exit 1;
    print "held\n"; $| = 1; open my $r, "<", $ARGV[1]; <$r>;' "$lock" "$TMP/release" >"$TMP/holder" &
  holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10; do grep -q held "$TMP/holder" 2>/dev/null && break; sleep 0.2; done
  rc=0
  (cd "$TMP" && "$repo/tools/test-run.sh" affected --files lib/helper.ml) >"$out" 2>"$err" || rc=$?
  echo >"$TMP/release"
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
    --files lib/lx.mll -- @t1/runtest-t_core @t2/runtest-gen-v1 @t2/runtest-orphan @t2/runtest-scenario_x @t2/slow-gen
  control 'an opaque action is what lists the aliases beside a changed config' \
    ' || ($opaque && $data_dirs{$a->{dir}})' '' \
    --files t3/plain.cfg -- @t3/runtest-t_plain @t3/runtest-t_stub
  control 'a stub archive is what carries a C stub to its library' \
    '$obj_libs{$o}{$_} = 1 for @libs' '1' \
    --files lib/stubbed_c.c -- @t3/runtest-t_stub
fi

finish

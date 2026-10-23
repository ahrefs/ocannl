#!/usr/bin/env bash
# Which per-test aliases can a change reach? Read from dune's own description
# of the build, never from a hand-kept map (gh-ocannl-1235). The entry point is
# `tools/test-run.sh affected`, which refuses while a run holds this worktree
# and then runs this script. It prints aliases and runs none of them.
#
# Usage:
#   tools/test-run.sh affected [--referenced-only] --since BASE [-- PATHSPEC...]
#   tools/test-run.sh affected [--referenced-only] REV | REV..REV | REV...REV [-- PATHSPEC...]
#   tools/test-run.sh affected [--referenced-only] --files PATH...
#
# The changed paths come from `git diff --name-only` with the given revision
# argument, in git's own forms: one REV compares it with the working tree, so
# uncommitted edits count, and files git does not track yet are added to that
# form. `--since BASE` is that form with REV the merge-base of HEAD and BASE
# (the base branch, e.g. the staging remote's `master`): a branch's own
# changes, whatever BASE gained since it forked. A REV that is not an ancestor
# of HEAD draws a warning, since its own later commits then count as changes.
# There is no default base: remote names are not the contract here.
# `--files` names repository-relative paths directly, without git.
#
# stdout: one alias per line, `@DIR/FAMILY-NAME[-VARIANT]`, ready for
# `tools/test-run.sh run build ...` or `tools/test-run.sh fresh ...`: sorted,
# then the link-only ones, sorted (the ranking is described below);
# `--referenced-only` leaves those off. stderr: what each changed path reached,
# a per-directory count, the link-only aliases, and what this script could
# not place. The contract: LINK-ONLY is a heuristic ranking, never a
# verdict that an alias cannot break. `--referenced-only` is a deliberate
# trade, faster with a possible miss, for iterating mid-change; the run before
# a push takes the full default list (AGENTS.md). The graph is always the CURRENT worktree's, whatever
# revisions were named.
#
# What dune is asked (each answer is checked against the forms listed below,
# and an answer of any other form stops the run, exit 2):
#   - `dune describe workspace --with-pps`: each library and `executables`
#     stanza this build compiles, its modules, and the libraries it requires.
#     It omits `tests` stanzas and leaves preprocessing out of `requires`.
#   - `dune describe external-lib-deps`: every stanza with the project
#     libraries it names, ppx rewriters included (`tests` ones and those this
#     build does not compile too, such as an optional backend library; their
#     modules stay unknown). Its edges are added to the first answer's. A
#     library it names that neither answer lists is skipped when the stanza
#     naming it is one this build does not compile either (the listing gives
#     public names only, and such a stanza may use a local one); from any other
#     stanza it stops the run.
#   - `dune show aliases DIR...` and `dune show targets DIR...`: the aliases
#     and the targets of every directory holding a `dune` file. A `NAME.exe`
#     target no stanza there names is a `tests` stanza naming no library,
#     which neither describe answer lists.
#   - `dune rules TARGET...`, for each per-test alias's own targets (below),
#     each generated module file, and every C stub object (`X.o`) and stub
#     archive (`libNAME_stubs.a`): the
#     executables those rules run and the files, or globs of files, they read.
#     A generated file such a rule reads (a `copy_files` copy, a file list) is
#     asked in turn, and what its rule reads counts as read; compiled objects
#     and executables are not followed (an executable is reached through its
#     stanza). Queries are batched to stay within any command-line limit.
#     `dune rules` prints a glob with its escapes dropped, so a glob also
#     matches with its braces, brackets and commas read literally, and as its
#     own text. A compiled artifact a rule reads
#     (a stanza's `.NAME.objs` files, an installed library's) makes that
#     stanza one of the rule's owners, like an executable the rule runs; an
#     alias a rule needs (`(deps (alias ...))`) is recorded as well. Only the
#     file `-o` names is read: printing a rule can build what its dependencies
#     are computed from, and those actions' output goes to dune's own streams.
#     An answer of another form is kept in $TMPDIR/affected-dune-rules.PID.txt,
#     and the refusal names the offending atom and its byte offset.
#   - The probe (gh-ocannl-1541): `dune rules @ALIAS` evaluates an alias's
#     actions lazily, which records no dependency, so what an alias rule's own
#     action reads dune never prints. The listings and `dune rules` are
#     therefore asked of a scratch copy of the sources (the probe) in which
#     each dune file is dune's own canonical text (`dune format-dune-file`),
#     each rule attached to an alias that declares no targets is a TWIN -- the
#     same rule writing a file of its own, its aliases kept defined by an
#     `(alias (name X) (deps TWIN))` -- and each test `dune show tests` lists
#     has a golden (an empty one where it has none), so dune makes its output
#     rule. The twin's rule is what the action reads, as dune evaluates it.
#     An alias some stanza makes depend on files (a rule with targets, an
#     `(alias)` stanza) is also asked as `dune rules @ALIAS`, which expands
#     dependencies, helper aliases included. On this repository the probe
#     costs about 10 s a run (its build directory starts cold; the dune cache
#     serves the compiles), and a compared dune edit about 30 s in all.
#     OCANNL_TOOL_AFFECTED_PROBE=0 skips it, at the price of every alias
#     action read blind (and the reach only a twin shows, such as an
#     executable of another name a run-and-diff rule runs); a probe dune does
#     not load falls back the same way, and says so.
#   - For a changed `dune` file, the old probe: the probe with that file at
#     the old revision. Both answer `dune show aliases`/`targets` and `dune
#     rules` for every target of its directory, and their forms (dune's
#     canonical text) are compared too.
#
# Boundary, as an allowlist. Whatever it does not cover is reported on stderr,
# never silently dropped. THE CONVERTER IS A STRICT ALLOWLIST: it reads dune's
# canonical text only as top-level forms, their heads, the `alias`/`aliases`/
# `targets`/`target`/`action` fields, literal `name`/`names`, and literal
# `include`/`subdir` paths; everything else passes to dune verbatim. It
# rewrites only a rule with literal alias names, no declared targets and one
# action field. Any form it neither rewrites nor knows to attach no action to
# an alias falls back to the reach without the probe for the aliases it can
# affect (the alias, or every alias of its directory), named on stderr.
#   - A per-test alias is `@DIR/FAMILY-NAME` or `@DIR/FAMILY-NAME-VARIANT`
#     that `dune show aliases` lists, FAMILY one of runtest, train, slow. Other
#     families (`@scans`, `@lifecycle`, `@bin-smoke`, ...) are aggregates and
#     never printed.
#   - Its owners are the stanzas of DIR with NAME as a name, as a library's
#     local name or as a module name, plus every executable that a rule of its
#     own targets runs. Its own targets are DIR's targets named `NAME-VARIANT.*`
#     or `NAME.VARIANT.*` (`NAME.*` when there is no variant), or `NAME_*` (a
#     rule-made `NAME_actual.ml` beside `runtest-NAME-ppx`), other than
#     compiled objects, executables and preprocessed copies: `fresh` attributes
#     a test's time by the same names, `NAME_*` aside. A golden alias is one
#     whose own targets, `NAME_*` aside, include an `.expected` file. An alias
#     with neither owners nor own targets is unattributed: it is printed
#     whenever a reached stanza or a changed path lies in its directory, since
#     what it reads is unknown.
#   - A changed `.ml`/`.mli` reaches the stanza whose module it is (by
#     directory and module name: `foo.ml`, `foo.mli`, a `(select)` choice
#     `foo.cudajit.ml` are `Foo`); failing that, in a directory with `tests`
#     stanzas, the one named after it or else every one there, since dune does
#     not describe their modules; failing that, the stanzas of its directory
#     this build does not compile, and the path is UNMAPPED. A module file
#     with no source of its name is generated: any changed path its rule
#     reads (a `.mll`, a `.mly`, a `(select)` choice) reaches its stanza, and
#     so does the executable the rule runs. A changed path a stub object's rule
#     reads (a `.c`, a header) reaches the library whose stub archive bundles
#     that object, or else every executable and `tests` stanza of its
#     directory, since which one links it dune does not say without computing
#     link rules; so does a reached executable the object's rule runs (a
#     generator writing a header it includes). A changed `dune` file not
#     compared across revisions (below), a `dune-project` or `dune-workspace`
#     file, or a fragment a dune file
#     `(include ...)`s, reaches every stanza and every per-test alias of the
#     directories beneath it (an `env` stanza applies there too), and every
#     alias elsewhere whose own targets' rules read a file or glob of those
#     directories (a file their rules make). Reach then spreads to each stanza
#     requiring a reached library, to a fixpoint.
#   - An alias is printed when its directory's `dune` file changed, when an
#     owner is reached, or when a rule of its own targets reads a changed path
#     (a file, or a glob over source files): the repository scans are found
#     this way, and so is a golden, whose copy into the build is one of the
#     own targets of the alias that diffs it. A changed path reaching nothing
#     else is listed as unread.
#   - An alias's own action is not in `dune rules` output. With the probe an
#     alias is DESCRIBED when every action attached to it is a twin or a
#     test's golden diff (dune's documented `(test)` action: its output rule,
#     made in the probe, shows the stanza's `(deps)` and `(action)`): its own
#     targets then include its twins, and it is printed only for what their
#     rules read. It stays BLIND -- an inline-test or cram alias (`dune show
#     tests` names an alias as its target), an alias a rule the converter
#     does not rewrite attaches to (one with an `(:include)` or a `dynamic-run`,
#     whose dependencies `dune rules` does not print), every alias of a
#     directory holding a form outside the allowlist -- and is printed as
#     without the probe. Without the probe, a golden alias's action diffs its
#     golden against an output a rule makes, both its own targets, whose rules
#     are read: an alias is golden when its own targets hold a `.expected` and a
#     rule-made file beside it. Any other alias's action is opaque -- a `tests`
#     stanza with no golden runs its executable under the stanza's `(deps)`; a
#     rule that runs and diffs in one action owns a golden but no rule-made
#     output (gh-ocannl-1431: taken for a golden pair,
#     `runtest-kernel_attribution` was never listed for the benchmarks file its
#     `(deps)` name). A blind or opaque alias is printed whenever a changed path
#     other than an owned module source lies in its directory or is copied
#     there, and whenever any changed path is no build input (a script, a
#     config, a golden).
#     An alias whose rules need a printed alias is printed too, to a fixpoint:
#     a golden pair's output rule carries the ambient
#     `runtest-env_spelling_gate` for this reason (gh-ocannl-1378) -- in the
#     needed alias's tier, one printed only blind included: a broken needed
#     alias breaks the needing one's build. A needed alias that is no per-test
#     alias (a helper) is an entry of its own, never printed, read as a
#     per-test one; one dune makes itself (`.PKG-files`, what `(package PKG)`
#     needs) is read through `dune rules @ALIAS`, its installed files followed
#     to the build's.
#   - A changed `dune` file compared across revisions (git forms only; both
#     sides have it; the worktree's copy is the right-hand side's; both probes
#     load) reaches what dune says changed: a rule of its directory that differs
#     between the probes -- a changed twin lists its aliases; a changed `.exe`,
#     `.pp.ml`, object or archive rule reaches its stanza as a whole; any other
#     changed target is read like a changed path, by the aliases' rules, a
#     generated module's, a stub's, and the link and preprocessing rules of
#     every stanza (`link_deps`, `preprocessor_deps`) -- plus a per-test alias
#     new on this side, plus, from the forms that differ, a changed library,
#     executables or tests stanza (compile flags live in rules not asked) by its
#     literal names as dune lists them here, and a changed `(alias)` stanza's
#     per-test alias. A changed form of any other kind (`env`, `subdir`,
#     `include`, `dirs`, an unknown head, an `(alias)` with an action), a
#     changed rule this reader cannot map to an owner, or a failed comparison
#     keeps the reach below for that file, and says why.
#
# Exit: 0 every changed `.ml`/`.mli` was mapped; 3 the list is printed, but a
# changed `.ml`/`.mli` is UNMAPPED (see stderr); 2 usage, git or dune failure,
# nothing printed.
set -u
die() { echo "affected: $*" >&2; exit 2; }
# One collation everywhere: `sort` and Perl's `sort` then agree bytewise, so
# the printed order never depends on the caller's locale (gh-ocannl-1431).
export LC_ALL=C

CDPATH= cd -P "$(dirname "$0")/.." || die "cannot cd to the repository root"
command -v dune >/dev/null 2>&1 || . tools/opam-env.sh
command -v dune >/dev/null 2>&1 || die "dune not found (opam environment not set up?)"
command -v perl >/dev/null 2>&1 || die "perl not found"

tmp=$(mktemp -d "${TMPDIR:-/tmp}/affected.XXXXXX") || die "cannot create a temporary directory"
tmp=$(cd "$tmp" && pwd -P) || die "cannot resolve $tmp"
trap 'rm -rf "${tmp:?}"' EXIT
trap "exit 130" INT TERM

usage="usage: affected [--referenced-only] --since BASE | REV | REV..REV | REV...REV [-- PATHSPEC...] | affected [--referenced-only] --files PATH..."
# Link-only aliases (see the Perl below) are printed last, and named on stderr;
# --referenced-only leaves them off stdout.
referenced_only=0
if [ "${1:-}" = --referenced-only ]; then referenced_only=1; shift; fi
mode=git
old_rev= new_rev=
revs=()
paths=()
since=
if [ "${1:-}" = --files ]; then
  shift
  [ $# -ge 1 ] || die "--files needs at least one path ($usage)"
  mode=files
  paths=("$@")
else
  while [ $# -gt 0 ]; do
    case $1 in
      --) shift; paths=("$@"); break ;;
      --since)
        [ $# -ge 2 ] || die "--since needs the base branch, e.g. --since <remote>/master ($usage)"
        [ -z "$since" ] || die "--since given twice ($usage)"
        since=$2; shift 2 ;;
      --since=*) [ -z "$since" ] || die "--since given twice ($usage)"; since=${1#--since=}; shift ;;
      -*) die "unknown option: $1 ($usage)" ;;
      *) revs+=("$1"); shift ;;
    esac
  done
  [ -z "$since" ] || [ ${#revs[@]} -eq 0 ] || die "--since BASE and a revision argument are exclusive ($usage)"
  [ ${#revs[@]} -le 1 ] || die "one revision argument at most, got ${#revs[@]} ($usage)"
fi

in_git=0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 && in_git=1
if [ "$mode" = files ]; then
  printf '%s\n' "${paths[@]}" >"$tmp/changed"
else
  [ "$in_git" = 1 ] || die "not a git checkout; use --files"
  # No default base: which remote carries the base branch is the checkout's
  # business (AGENTS.md: remote names are not the contract), so it is named.
  if [ -n "$since" ]; then
    # The branch's own changes: from where it left BASE, not from BASE's tip,
    # which would also count every commit BASE gained since (gh-ocannl-1379).
    git rev-parse --verify -q "${since}^{commit}" >/dev/null || die "not a revision: $since"
    base=$(git merge-base HEAD "$since") || die "HEAD and $since share no history"
    echo "affected: --since $since: comparing with the merge-base ${base}" >&2
    revs=("$base")
  fi
  [ ${#revs[@]} -eq 1 ] ||
    die "name the base: affected --since <base-branch>, REV, REV..REV, or --files PATH... ($usage)"
  case ${revs[0]} in
    *..*) ;;
    *)
      git rev-parse --verify -q "${revs[0]}^{commit}" >/dev/null || die "not a revision: ${revs[0]}"
      git merge-base --is-ancestor "${revs[0]}" HEAD ||
        echo "affected: warning: ${revs[0]} is not an ancestor of HEAD, so its own later commits count as changes too; --since ${revs[0]} compares with the merge-base instead" >&2 ;;
  esac
  git diff --name-only --no-renames "${revs[0]}" -- ${paths[@]+"${paths[@]}"} >"$tmp/changed" ||
    die "git diff ${revs[0]} failed"
  case ${revs[0]} in
    *..*) ;;
    *) git ls-files --others --exclude-standard -- ${paths[@]+"${paths[@]}"} >>"$tmp/changed" ||
         die "git ls-files failed" ;;
  esac
  # Both sides of each changed `.ml` and `.mli` file: a module edit is ranked by
  # the definitions it touched (the hunks, below). OLD is the revision compared
  # with; NEW the right-hand revision of a range, else the worktree.
  old_rev= new_rev=
  case ${revs[0]} in
    *...*) new_rev=${revs[0]#*...}; old_rev=$(git merge-base "${revs[0]%%...*}" "${new_rev:-HEAD}") ||
             die "no merge-base for ${revs[0]}" ;;
    *..*) old_rev=${revs[0]%%..*}; new_rev=${revs[0]#*..}; old_rev=${old_rev:-HEAD}; new_rev=${new_rev:-HEAD} ;;
    *) old_rev=${revs[0]} ;;
  esac
  while IFS= read -r f; do
    case $f in *.ml | *.mli) ;; *) continue ;; esac
    mkdir -p "$tmp/old/$(dirname -- "$f")" "$tmp/new/$(dirname -- "$f")" || die "cannot create $tmp/old"
    # A file the old side does not have is new: it gets no scoping.
    git show "$old_rev:$f" >"$tmp/old/$f" 2>/dev/null || rm -f "$tmp/old/$f"
    if [ -n "$new_rev" ]; then
      git show "$new_rev:$f" >"$tmp/new/$f" 2>/dev/null || rm -f "$tmp/new/$f"
    elif [ -f "$f" ]; then
      cp -- "$f" "$tmp/new/$f" || die "cannot copy $f"
    fi
  done <"$tmp/changed"
  git diff -U0 --no-color --no-renames --no-ext-diff "${revs[0]}" -- '*.ml' '*.mli' >"$tmp/ml-diff" ||
    die "git diff ${revs[0]} failed"
fi

dune describe workspace --root . --with-pps >"$tmp/workspace" 2>"$tmp/err" ||
  die "dune describe workspace failed:
$(cat "$tmp/err")"
dune describe external-lib-deps --root . >"$tmp/lib-deps" 2>"$tmp/err" ||
  die "dune describe external-lib-deps failed:
$(cat "$tmp/err")"
# Every directory holding a dune file may define per-test aliases.
if [ "$in_git" = 1 ]; then
  git ls-files --cached --others --exclude-standard -- dune '*/dune' >"$tmp/dune-files" ||
    die "git ls-files failed"
else
  find . -name _build -prune -o -name dune -type f -print | sed 's|^\./||' >"$tmp/dune-files"
fi
while IFS= read -r f; do dirname -- "$f"; done <"$tmp/dune-files" | sort -u >"$tmp/dirs"
[ -s "$tmp/dirs" ] || die "no dune files found"
dirs=()
while IFS= read -r d; do dirs+=("$d"); done <"$tmp/dirs"

# The probe (gh-ocannl-1541). What an alias rule's own action reads, dune
# never prints: `dune rules @ALIAS` evaluates an alias's actions lazily, which
# records no dependency. So the rules are handed to dune in a form it does
# print, in a scratch copy of the sources: each dune file is replaced by
# dune's own canonical text (`dune format-dune-file`: dune's lexer, so `#|`,
# block strings and escapes are its), and each rule attached to an alias that
# declares no targets becomes a TWIN, the same rule writing a file of its own
# (`__affected_probe_<hash>`) with an `(alias (name X) (deps <twin>))` keeping
# its aliases defined. `dune rules` on a twin is what the action reads, as dune
# evaluates it: `%{dep:}`, what `diff` infers, `(:include)`, `chdir`, globs,
# needed aliases. A test without a golden gets an empty one, so dune makes its
# output rule (which shows the test's `(deps)` and `(action)`). The converter
# is an allowlist (convert.pl below); what it does not rewrite is named in the
# manifest and stays as blind as before. No probe (it failed, or
# OCANNL_TOOL_AFFECTED_PROBE=0) means every alias action is read as before.
# Dune here never sees the caller's build directory.
probe_dune() { # DIR ARGS...: dune run at the root of a probe tree
  local d=$1
  shift
  (cd "$d" && env -u DUNE_BUILD_DIR -u DUNE_ROOT dune "$@" --root .)
}
cat >"$tmp/convert.pl" <<'CONVERT'
# convert.pl ROOT MANIFEST FORMS RECURSE LIST: rewrite each dune file LIST names in the
# probe ROOT into dune's canonical text with alias rules turned into twins;
# append to MANIFEST what was rewritten and what was not, to FORMS each
# top-level form of FILE (for a dune-file edit's comparison). RECURSE=0 leaves
# `(include ...)`d fragments alone (they are converted already).
use strict; use warnings; use Digest::MD5 qw(md5_hex); use File::Spec;
my ($root, $manifest, $forms_out, $recurse, $list) = @ARGV;
open my $lh, "<", $list or die "cannot read $list\n";
my @files = grep { $_ ne "" } map { s/\r?\n\z//r } <$lh>;
open my $man, ">>", $manifest or die "cannot write $manifest\n";
open my $fo, ">>", $forms_out or die "cannot write $forms_out\n";
sub canonical {   # dune's own text of a dune file, or undef
  my ($path) = @_;
  my $pid = open(my $fh, "-|") // die "cannot fork\n";
  if (!$pid) { open STDERR, ">", File::Spec->devnull; exec "dune", "format-dune-file", $path; exit 127 }
  local $/; my $s = <$fh> // ""; close $fh;
  return $? == 0 ? $s : undef;
}
# dune's printed text read as lists of raw tokens: atoms and quoted strings kept
# as printed (re-emitted verbatim), comments dropped. Anything else (a raw
# newline in a string, say, which dune's printer escapes): undef.
sub forms_of {
  my ($s) = @_;
  my @stack = ([]);
  pos($s) = 0;
  while (pos($s) < length $s) {
    next if $s =~ /\G\s+/gc || $s =~ /\G;[^\n]*/gc;
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) { @stack > 1 or return undef; my $l = pop @stack; push @{$stack[-1]}, $l; next }
    if ($s =~ /\G("(?:[^"\\\n]|\\.)*")/gc) { push @{$stack[-1]}, $1; next }
    if ($s =~ /\G([^\s()";]+)/gc) { push @{$stack[-1]}, $1; next }
    return undef;
  }
  return @stack == 1 ? $stack[0] : undef;
}
sub text { my ($x) = @_; ref $x ? "(" . join(" ", map { text($_) } @$x) . ")" : $x }
sub named { my ($x, $n) = @_; ref $x && @$x && !ref $x->[0] && $x->[0] eq $n }
sub is_field { my ($x) = @_; ref $x && @$x && !ref $x->[0] }
sub name_lit { my ($x) = @_; !ref $x && $x =~ /^[A-Za-z0-9_.+-]+$/ }
sub path_in {   # DIR/P normalized, for a literal relative P staying in the tree, else undef
  my ($dir, $p) = @_;
  return undef if ref $p || $p !~ m{^[A-Za-z0-9_.+-]+(?:/[A-Za-z0-9_.+-]+)*$};
  my @parts = $dir eq "." ? () : split m{/}, $dir;
  for my $c (split m{/}, $p) {
    next if $c eq ".";
    if ($c eq "..") { @parts or return undef; pop @parts } else { push @parts, $c }
  }
  return @parts ? join("/", @parts) : ".";
}
sub note { print $man join("\t", @_), "\n" }
# Heads of stanzas that attach no action to an alias. One with an `(alias ...)`
# field (copy_files has one) makes the alias depend on what it makes: a
# `source`, whose rules `dune rules @ALIAS` prints. Tests are read from
# `dune show tests` (their goldens), inline tests and cram there too.
my %plain = map { $_ => 1 } "executable", "executables", "library", "test", "tests", "copy_files", "copy_files#",
  "ocamllex", "ocamlyacc", "menhir", "install", "env", "dirs", "data_only_dirs", "vendored_dirs", "include_subdirs",
  "documentation", "generate_sites_module", "foreign_library";
my %written;
sub convert {   # DIR FILE FORMS TOP -> the rewritten forms
  my ($dir, $file, $forms, $top) = @_;
  my (@out, %occ);
  for my $f (@$forms) {
    unless (is_field($f)) { note("blind", $dir, "$file holds a top-level form with no stanza name"); push @out, $f; next }
    my ($h, @fl) = @$f;
    my $canon = text($f);
    if ($top) {
      my @names = grep { name_lit($_) } map { named($_, "name") || named($_, "names") ? @$_[1 .. $#$_] : () } @fl;
      print $fo join("\t", $file, $dir, $h, join(",", @names), (grep { named($_, "action") } @fl) ? 1 : 0, $canon), "\n";
    }
    if ($h eq "subdir") {
      my $sub = @fl ? path_in($dir, $fl[0]) : undef;
      if (defined $sub) { push @out, [$h, $fl[0], convert($sub, $file, [@fl[1 .. $#fl]], 0)]; next }
      note("blindtree", $dir, "$file has a (subdir) whose directory is not a literal path below it"); push @out, $f; next;
    }
    if ($h eq "include") {
      my $frag = @fl == 1 && $top ? path_in($dir, $fl[0]) : undef;
      unless (defined $frag && -f "$root/$frag") { note("blind", $dir, "$file has an (include) this reader cannot follow"); push @out, $f; next }
      if ($recurse) {
        my $t = canonical("$root/$frag");
        my $ff = defined $t ? forms_of($t) : undef;
        unless ($ff) { note("blind", $dir, "dune's text of $frag is not readable here"); push @out, $f; next }
        my @c = convert($dir, $frag, $ff, 0);
        unless ($written{$frag}++) {
          open my $o, ">", "$root/$frag" or die "cannot write $root/$frag\n";
          print $o text($_), "\n\n" for @c;
          close $o;
        }
      }
      push @out, $f; next;
    }
    my @af = grep { named($_, "alias") || named($_, "aliases") } @fl;
    my @al = map { @$_[1 .. $#$_] } @af;
    if (grep { !name_lit($_) } @al) { note("blind", $dir, "$file has a $h whose alias name is not literal"); push @out, $f; next }
    # Dependencies found while the rule runs -- what an `(:include FILE)` adds,
    # read from FILE, or what a `dynamic-run` reports -- `dune rules` does not
    # print: a form with one is not vouched for, as an alias action or as the
    # rules of its directory.
    if ($canon =~ /(?:^|[\s(])(:include|dynamic-run)[\s)]/) {
      my $what = $1;
      if ($h eq "rule" && @al) { note("undescribed", $dir, $_, "a rule of $file with a $what, whose dependencies dune rules does not print") for @al }
      else { note("blind", $dir, "$file has a ($h) with a $what, whose dependencies dune rules does not print") }
      push @out, $f; next;
    }
    if ($h eq "rule") {
      unless (@al) { push @out, $f; next }
      if (grep { named($_, "targets") || named($_, "target") } @fl) { note("source", $dir, $_) for @al; push @out, $f; next }
      my @act = grep { named($_, "action") } @fl;
      if (@act != 1 || @{$act[0]} != 2 || grep { !is_field($_) } @fl) {
        note("undescribed", $dir, $_, "a rule of $file with no single action field") for @al;
        push @out, $f; next;
      }
      # Named by the file, the aliases and the rule's rank among theirs, never by
      # its text: an edit dune reads the same (quoting, layout) changes no rule.
      my $key = join(" ", @al);
      my $twin = "__affected_probe_" . substr(md5_hex(join("\0", $file, $key, ++$occ{$key})), 0, 12);
      push @out, ["rule", map { named($_, "action") ? ["action", ["progn", ["write-file", $twin, '""'], $_->[1]]] : $_ }
                          grep { !named($_, "alias") && !named($_, "aliases") } @fl];
      push @out, ["alias", ["name", $_], ["deps", $twin]] for @al;
      note("twin", $dir, $twin, @al);
      next;
    }
    if ($h eq "alias") {
      my @n = map { @$_[1 .. $#$_] } grep { named($_, "name") } @fl;
      if (@n != 1 || !name_lit($n[0])) { note("blind", $dir, "$file has an (alias) whose name is not literal"); push @out, $f; next }
      if (grep { named($_, "action") } @fl) { note("undescribed", $dir, $n[0], "an (alias) of $file with an action") }
      else { note("source", $dir, $n[0]) }
      push @out, $f; next;
    }
    unless ($plain{$h}) { note("blind", $dir, "$file has a ($h) stanza, which this reader does not model"); push @out, $f; next }
    note("source", $dir, $_) for @al;
    push @out, $f;
  }
  return @out;
}
for my $file (@files) {
  next unless -f "$root/$file";
  my ($dir) = $file =~ m{^(.*)/[^/]+$} ? ($1) : (".");
  my $t = canonical("$root/$file");
  my $forms = defined $t ? forms_of($t) : undef;
  unless ($forms) { note("blind", $dir, "dune's text of $file is not readable here"); next }
  my @c = convert($dir, $file, $forms, 1);
  open my $o, ">", "$root/$file" or die "cannot write $root/$file\n";
  print $o text($_), "\n\n" for @c;
  close $o;
}
CONVERT
# Tests, as dune lists them: one with an executable gets an empty golden where
# it has none (and no rule makes one), so dune makes its output rule; one dune
# runs as an alias (inline tests, cram) stays undescribed.
cat >"$tmp/tests.pl" <<'TESTS'
use strict; use warnings;
my ($root, $tests, $targets, $manifest) = @ARGV;
open my $man, ">>", $manifest or die "cannot write $manifest\n";
local $/; open my $fh, "<", $tests or die; my $s = <$fh>;
open my $th, "<", $targets or die; my $tl = <$th>;
my (%made, $cur);
for (split /\n/, $tl) { if (/^(.*):$/) { $cur = $1; next } $made{$cur eq "." ? $_ : "$cur/$_"} = 1 if defined $cur }
# ((RECORD) ...), each RECORD a list of (KEY VALUE) pairs.
my @stack = ([]);
pos($s) = 0;
while (pos($s) < length $s) {
  next if $s =~ /\G\s+/gc;
  if ($s =~ /\G\(/gc) { push @stack, []; next }
  if ($s =~ /\G\)/gc) { @stack > 1 or die "unbalanced dune show tests\n"; my $l = pop @stack; push @{$stack[-1]}, $l; next }
  if ($s =~ /\G"((?:[^"\\]|\\.)*)"/gcs) { push @{$stack[-1]}, $1 =~ s/\\(.)/$1/gr; next }
  if ($s =~ /\G([^\s()"]+)/gc) { push @{$stack[-1]}, $1; next }
  die "unreadable dune show tests at byte " . pos($s) . "\n";
}
@stack == 1 && @{$stack[0]} == 1 && ref $stack[0][0] or die "dune show tests printed no list of tests\n";
for my $rec (@{$stack[0][0]}) {
  my %f;
  ref $rec or die "dune show tests printed an atom where a test was expected\n";
  for (@$rec) { ref $_ && @$_ == 2 && !ref $_->[0] or die "unexpected field in dune show tests\n"; $f{$_->[0]} = $_->[1] }
  my ($name, $dir, $target) = @f{qw(name source_dir target)};
  defined $name && defined $dir && defined $target && !ref $name && !ref $dir && !ref $target
    or die "a dune show tests record without a name, source_dir and target\n";
  if ($target =~ m{^\@(?:(.*)/)?([^/]+)$}) { print $man join("\t", "undescribed", $1 // ".", $2, "dune runs it as an inline-test or cram action"), "\n"; next }
  $target =~ m{^_build/[^/]+/(?:\Q$dir\E/)?\Q$name\E\.(?:exe|bc)$} or die "unexpected test target $target\n";
  print $man join("\t", "test", $dir, $name), "\n";
  my $golden = ($dir eq "." ? "" : "$dir/") . "$name.expected";
  next if -e "$root/$golden" || $made{$golden};
  open my $o, ">", "$root/$golden" or die "cannot write $root/$golden\n"; close $o;
}
TESTS
make_tree() { # PROBE: a scratch copy of the sources, converted (dune not run there yet)
  local p=$1
  mkdir -p "$p" || return 1
  if [ "$in_git" = 1 ]; then
    { git ls-files -z --cached --others --exclude-standard &&
      git ls-files -z --others --ignored --exclude-standard -- ocannl_config '*/ocannl_config'; } >"$tmp/probe-files" || return 1
  else
    find . -name _build -prune -o \( -type f -o -type l \) -print0 >"$tmp/probe-files" || return 1
  fi
  # A path git still indexes but the worktree deleted is not copied.
  perl -0ne 'chomp; print "$_\0" if -e $_ || -l $_' "$tmp/probe-files" >"$tmp/probe-list" || return 1
  tar --null -T "$tmp/probe-list" -cf - | (cd "$p" && tar -xf -) || return 1
  perl "$tmp/convert.pl" "$p" "$tmp/manifest" "$tmp/forms" 1 "$tmp/dune-files" || return 1
}
finish_probe() { # PROBE SUFFIX: its tests' goldens, and its listings
  local p=$1 s=$2
  probe_dune "$p" show tests >"$tmp/tests$s" 2>"$tmp/err" || return 1
  probe_dune "$p" show targets "${dirs[@]}" >"$tmp/probe-targets$s" 2>&1 || return 1
  perl "$tmp/tests.pl" "$p" "$tmp/tests$s" "$tmp/probe-targets$s" "$tmp/manifest$s" 2>>"$tmp/err" || return 1
}
# A changed dune file is compared across revisions (the old probe below) when
# both sides have it and the worktree's copy is the right-hand side's; any other
# keeps the reach of its whole directory, and says why.
scoped=()
: >"$tmp/unscoped"
while IFS= read -r f; do
  case $f in dune | */dune) ;; *) continue ;; esac
  why=
  if [ "$mode" = files ]; then why="--files names no revision to compare it with"
  elif [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" = 0 ]; then why="no probe"
  elif [ ! -f "$f" ]; then why="it is deleted"
  elif ! git cat-file -e "$old_rev:$f" 2>/dev/null; then why="$old_rev has no such file"
  elif [ -n "$new_rev" ] && ! git show "$new_rev:$f" 2>/dev/null | cmp -s - "$f"; then
    why="the worktree's copy is not $new_rev's"
  fi
  if [ -n "$why" ]; then printf '%s\t%s\n' "$f" "$why" >>"$tmp/unscoped"; else scoped+=("$f"); fi
done <"$tmp/changed"
qroot=.
probe_ok=0
old_ok=0
for x in manifest forms manifest-old forms-old old-targets old-aliases scoped; do : >"$tmp/$x"; done
if [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" = 0 ]; then
  echo "affected: OCANNL_TOOL_AFFECTED_PROBE=0: no probe, so every alias action dune does not dump stays blind" >&2
elif make_tree "$tmp/probe"; then
  # The old probe: this one with each compared dune file at the old revision.
  if [ ${#scoped[@]} -gt 0 ]; then
    if cp -R "$tmp/probe" "$tmp/probe-old" && printf '%s\n' "${scoped[@]}" >"$tmp/scoped" &&
       while IFS= read -r f; do git show "$old_rev:$f" >"$tmp/probe-old/$f" || exit 1; done <"$tmp/scoped" &&
       perl "$tmp/convert.pl" "$tmp/probe-old" "$tmp/manifest-old" "$tmp/forms-old" 0 "$tmp/scoped" &&
       finish_probe "$tmp/probe-old" -old; then
      old_dirs=()
      while IFS= read -r f; do old_dirs+=("$(dirname -- "$f")"); done <"$tmp/scoped"
      if probe_dune "$tmp/probe-old" show targets "${old_dirs[@]}" >"$tmp/old-targets" 2>&1 &&
         probe_dune "$tmp/probe-old" show aliases "${old_dirs[@]}" >"$tmp/old-aliases" 2>&1; then
        old_ok=1
      fi
    fi
    if [ "$old_ok" = 0 ]; then
      while IFS= read -r f; do printf '%s\t%s\n' "$f" "the old probe failed: $(tail -1 "$tmp/err" 2>/dev/null)" >>"$tmp/unscoped"; done <"$tmp/scoped"
      : >"$tmp/scoped"
    fi
  fi
  if finish_probe "$tmp/probe" ""; then
    probe_ok=1
    qroot=$tmp/probe
  fi
fi
if [ "${OCANNL_TOOL_AFFECTED_PROBE:-1}" != 0 ] && [ "$probe_ok" = 0 ]; then
  echo "affected: the probe failed, so every alias action dune does not dump stays blind: $(tail -5 "$tmp/err" 2>/dev/null)" >&2
  if [ -s "$tmp/scoped" ]; then
    while IFS= read -r f; do printf '%s\t%s\n' "$f" "the probe failed" >>"$tmp/unscoped"; done <"$tmp/scoped"
  fi
  for x in manifest forms manifest-old forms-old scoped; do : >"$tmp/$x"; done
fi

# Dune 3.24 prints these listings on stderr, and exits 0 even after an error,
# so both streams are read together, and every line must have one of the forms
# the reader below accepts.
for what in aliases targets; do
  probe_dune "$qroot" show "$what" "${dirs[@]}" >"$tmp/$what" 2>&1 ||
    die "dune show $what failed:
$(cat "$tmp/$what")"
done

perl - "$tmp" "$referenced_only" "$probe_ok" "$qroot" "$tmp/probe-old" <<'PERL'
use strict; use warnings;
my ($tmp, $referenced_only, $probe_ok, $qroot, $old_root) = @ARGV;
# While a dune answer is being read, a refusal keeps a copy of it (see keep_raw).
our ($raw_text, $raw_what);
sub fail {
  my $m = $_[0];
  $m .= " (the whole answer is kept in " . keep_raw($raw_text, $raw_what) . ")" if defined $raw_text;
  print STDERR "affected: $m\n"; exit 2;
}
sub slurp { local $/; open my $fh, "<", $_[0] or fail("cannot read $_[0]"); my $s = <$fh>; $s }

# A reader for the sexps dune prints: lists, atoms, quoted strings. $at, when
# given, collects the byte offset of each top-level item; a failure names the
# offset and the text around it.
sub around { my ($s, $p) = @_; my $from = $p > 60 ? $p - 60 : 0; (substr($s, $from, 120) =~ s/\n/\\n/gr) }
sub parse_sexp {
  my ($s, $what, $at) = @_;
  my @stack = ([]);
  pos($s) = 0;
  while (pos($s) < length $s) {
    if ($s =~ /\G\s+/gc) { next }
    push @$at, pos($s) if $at && @stack == 1;
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) {
      @stack > 1 or fail("unbalanced output from $what at byte " . (pos($s) - 1) . ": ..." . around($s, pos($s) - 1) . "...");
      my $l = pop @stack; push @{$stack[-1]}, $l; next;
    }
    if ($s =~ /\G"((?:[^"\\]|\\.)*)"/gcs) {
      my $a = $1; $a =~ s/\\(.)/$1 eq "n" ? "\n" : $1/ge;
      push @{$stack[-1]}, $a; next;
    }
    if ($s =~ /\G([^\s()"]+)/gc) { push @{$stack[-1]}, $1; next }
    fail("unreadable output from $what at byte " . pos($s) . ": ..." . around($s, pos($s)) . "...");
  }
  @stack == 1 or fail("unbalanced output from $what (" . (@stack - 1) . " list(s) left open)");
  return $stack[0];
}
sub field { # a value from a list of (key value) pairs
  my ($fields, $key) = @_;
  for my $f (@$fields) { return $f->[1] if ref $f && @$f == 2 && !ref $f->[0] && $f->[0] eq $key }
  return undef;
}
sub items { my $v = $_[0]; defined $v && ref $v ? @$v : () }
sub atom { my ($v, $what) = @_; defined $v && !ref $v or fail("expected an atom in $what"); $v }
sub pair { my ($e, $what) = @_; ref $e && @$e == 2 && !ref $e->[0] or fail("unexpected entry in $what"); @$e }
sub dir_of { $_[0] =~ m{^(.*)/([^/]+)$} ? ($1, $2) : (".", $_[0]) }
sub nsort { sort { $a <=> $b } @_ }   # numeric; outside any lexical $a or $b

# Libraries and executables: ((root R) (build_context C) (KIND FIELDS)...).
my $ws = parse_sexp(slurp("$tmp/workspace"), "dune describe workspace");
@$ws == 1 && ref $ws->[0] or fail("dune describe workspace printed no workspace");
my $ctx;
my @st;          # { kind, dir, names, units => {NAME => 1}, to => [stanza index...] }
my %by_modkey;   # "DIR\0Module" -> [stanza indices]
my %lib_by_uid;  # local library uid -> stanza index
my %lib_by_name; # public or local library name -> stanza index
my %generated;   # generated module file -> [stanza indices owning it]
my @requires;
for my $e (@{$ws->[0]}) {
  my ($kind, $v) = pair($e, "dune describe workspace");
  next if $kind eq "root";
  if ($kind eq "build_context") { $ctx = atom($v, "build_context"); next }
  $kind =~ /^(executables|library)$/
    or fail("dune describe workspace entry \"$kind\" is outside this helper's boundary (executables, library)");
  defined $ctx or fail("dune describe workspace gave no build_context before its stanzas");
  next if $kind eq "library" && (field($v, "local") // "") ne "true";
  my ($inc) = items(field($v, "include_dirs"));
  # Dune names a stanza's object directory after it: DIR/.NAME.objs for a
  # library (NAME its local name), DIR/.NAME.eobjs for executables.
  my ($dir, $objname) = atom($inc, "include_dirs") =~ m{^\Q$ctx\E/(?:(.+)/)?\.([^/]+)\.e?objs/}
    or fail("include_dir \"$inc\" is not an object directory under $ctx");
  $dir //= ".";
  my @names = $kind eq "library" ? (atom(field($v, "name"), "name")) : map { atom($_, "names") } items(field($v, "names"));
  @names or fail("an executables entry without names in dune describe workspace");
  push @st, { kind => $kind, dir => $dir, names => \@names, units => { map { $_ => 1 } @names } };
  my $i = $#st;
  push @requires, [items(field($v, "requires"))];
  if ($kind eq "library") {
    $lib_by_uid{atom(field($v, "uid"), "uid")} = $i;
    $st[$i]{units}{$objname} = 1;
    for my $n ($names[0], $objname) {
      !defined $lib_by_name{$n} || $lib_by_name{$n} == $i or fail("two libraries answer to the name $n");
      $lib_by_name{$n} = $i;
    }
  }
  for my $m (items(field($v, "modules"))) {
    my $name = atom(field($m, "name"), "module name");
    next if $name =~ /__$/;   # alias modules dune generates
    $st[$i]{units}{lcfirst $name} = 1;
    for my $key ("impl", "intf") {
      for my $p (items(field($m, $key))) {
        my $rel = atom($p, $key) =~ s{^\Q$ctx\E/}{}r;
        my ($d) = dir_of($rel);
        # A module file with no source of that name is generated (ocamllex,
        # menhir, a (select) copy, a rule's output): its rule says from what.
        push @{$generated{$rel}}, $i unless -e $rel;
        # Its sources, for the link-only ranking: a preprocessed `X.pp.ml` stands for
        # `X.ml`; any other file with no source makes its code not all known.
        my $src = -e $rel ? $rel : $rel =~ s/\.pp(\.mli?)$/$1/r;
        # (A wrapped library's alias module, `NAME.ml-gen`, is dune's and names only its modules.)
        if (-e $src) { push @{$st[$i]{mods}{$name}}, $src } elsif ($rel !~ /\.ml-gen$/) { $st[$i]{gen_mod} = 1 }
        my $k = "$d\0$name";
        push @{$by_modkey{$k}}, $i unless grep { $_ == $i } @{$by_modkey{$k} // []};
      }
    }
  }
}
for my $i (0 .. $#st) {   # a uid outside %lib_by_uid is an installed library
  $st[$i]{to} = [map { $lib_by_uid{$_} } grep { defined $lib_by_uid{$_} } @{$requires[$i]}];
}

# Every stanza with its project-library edges, ppx rewriters included:
# ((CONTEXT (ENTRY...))), each ENTRY (KIND FIELDS). `dune describe workspace`
# leaves preprocessing out of `requires` and omits `tests` stanzas, and this
# listing leaves out stanzas with no dependency at all, so the two are merged:
# an entry here adds its edges to the described stanza of the same kind,
# directory and names, and one with no such stanza is added -- a `tests`
# stanza, or one this build does not compile (an optional backend library),
# whose modules are then unknown.
my $ext = parse_sexp(slurp("$tmp/lib-deps"), "dune describe external-lib-deps");
@$ext == 1 or fail("dune describe external-lib-deps printed " . scalar(@$ext) . " contexts, expected one");
my (undef, $entries) = pair($ext->[0], "dune describe external-lib-deps");
my %described = map { join("\0", $st[$_]{kind}, $st[$_]{dir}, sort @{$st[$_]{names}}) => $_ } 0 .. $#st;
my @edges;   # [stanza index, internal_deps]
for my $e (@$entries) {
  my ($kind, $v) = pair($e, "dune describe external-lib-deps");
  $kind =~ /^(executables|tests|library)$/
    or fail("dune describe external-lib-deps entry \"$kind\" is outside this helper's boundary (executables, tests, library)");
  my @names = map { atom($_, "names") } items(field($v, "names"));
  my $dir = atom(field($v, "source_dir"), "source_dir");
  @names or fail("a $kind entry without names in $dir");
  my $i = $described{join("\0", $kind, $dir, sort @names)};
  unless (defined $i) {
    push @st, { kind => $kind, dir => $dir, names => \@names, units => { map { $_ => 1 } @names }, to => [],
                undescribed => $kind ne "tests" };
    $i = $#st;
    if ($kind eq "library") {
      !defined $lib_by_name{$names[0]} or fail("two libraries answer to the name $names[0]");
      $lib_by_name{$names[0]} = $i;
    }
  }
  push @edges, [$i, field($v, "internal_deps")];
}
my (%skipped_optional, %skipped_off_build);
for my $edge (@edges) {
  my ($i, $deps) = @$edge;
  for my $d (items($deps)) {
    my ($n, $k) = pair($d, "internal_deps");
    if (defined $lib_by_name{$n}) { push @{$st[$i]{to}}, $lib_by_name{$n} }
    elsif ($k eq "optional") { $skipped_optional{$n} = 1 }
    # A stanza this build does not compile (an optional backend library whose
    # platform package is absent) may name another such library by its local
    # name, which this listing does not give -- only the public one: on Linux,
    # `metal_backend` requires `metal_math_api_runtime` (gh-ocannl-1389). No
    # stanza this build compiles links either one, so the edge carries no reach.
    elsif ($st[$i]{undescribed}) { $skipped_off_build{"$n (required by " . label($i) . ")"} = 1 }
    else { fail(label($i) . " uses project library $n, which dune does not describe") }
  }
}

# Per-directory listings: "DIR:" headers, one name per line.
my @dir_list = grep { $_ ne "" } split /\n/, slurp("$tmp/dirs");
my %is_dir = map { $_ => 1 } @dir_list;
our $listing_soft = 0;   # an unexpected line returns undef instead of stopping the run
sub listing {   # FILE [DIR...]: the listing of the directories asked (all, by default)
  my ($what, @asked) = @_;
  @asked = @dir_list unless @asked;
  my (%by, $cur);
  $cur = $asked[0] if @asked == 1;
  for my $l (split /\n/, slurp("$tmp/$what")) {
    next if $l eq "";
    # A directory dune does not build (a data-only one, say) defines nothing.
    if ($l =~ /^Error: Directory (.*) does not exist\.$/ && $is_dir{$1}) { $cur = undef; next }
    if ($l =~ /^(.*):$/ && $is_dir{$1}) { $cur = $1; next }
    # A directory target is listed with a trailing slash.
    unless ($l =~ s{^([^\s/]+)/?$}{$1}) { return undef if $listing_soft; fail("unexpected line from dune show $what: $l") }
    defined $cur or fail("dune show $what printed $l outside any directory");
    push @{$by{$cur}}, $l;
  }
  return \%by;
}
my $aliases = listing("aliases");
my $targets = listing("targets");
sub in_dir { my ($d, $f) = @_; $d eq "." ? $f : "$d/$f" }
# A `tests` stanza naming no library is in neither describe answer, but its
# executable is a target of its directory: one named by no stanza there is
# taken as such a stanza, linking nothing of the project.
for my $d (sort keys %$targets) {
  for my $x (map { /^(.+)\.exe$/ ? $1 : () } @{$targets->{$d}}) {
    next if grep { $st[$_]{dir} eq $d && $st[$_]{kind} ne "library" && grep { $_ eq $x } @{$st[$_]{names}} } 0 .. $#st;
    push @st, { kind => "tests", dir => $d, names => [$x], units => { $x => 1 }, to => [] };
  }
}

# What the probe's converter rewrote and what it did not (see convert.pl): per
# directory, the twins of each alias, the aliases with a stanza that makes them
# depend on files (`source`), and what stays blind.
my (%twins, %sources, %undescribed, %blind, %blindtree, %tested);
for (split /\n/, slurp("$tmp/manifest")) {
  my ($k, $d, @r) = split /\t/;
  if ($k eq "twin") { my $t = shift @r; push @{$twins{$d}{$_}}, in_dir($d, $t) for @r }
  elsif ($k eq "source") { $sources{$d}{$r[0]} = 1 }
  elsif ($k eq "undescribed") { $undescribed{$d}{$r[0]} //= $r[1] }
  elsif ($k eq "blind") { $blind{$d} //= $r[0] }
  elsif ($k eq "blindtree") { $blindtree{$d} //= $r[0] }
  elsif ($k eq "test") { $tested{$d}{"runtest-$r[0]"} = 1 }
  else { fail("unexpected manifest line from the probe: $_") }
}
sub blind_reason {   # why an alias of DIR named NAME is not described, or undef
  my ($d, $name) = @_;
  return $blind{$d} if defined $blind{$d};
  for my $r (sort keys %blindtree) { return $blindtree{$r} if under($d, $r) }
  return $undescribed{$d}{$name} if defined $undescribed{$d}{$name};
  return undef if $twins{$d}{$name} || $sources{$d}{$name} || $tested{$d}{$name};
  return "no stanza this reader read attaches it";
}

# Per-test aliases, their name owners and their own targets.
my @al;   # { name, dir, owners => {index => 1}, own => [target...] }
my $artifact = qr/\.(?:exe|bc|bc-for-jsoo|js|cmi|cmo|cmx|cmt|cmti|cma|cmxa|cmxs|a|o|so|dll|pp\.ml|pp\.mli|ml-gen|d)$/;
for my $d (sort keys %$aliases) {
  my @in = grep { $st[$_]{dir} eq $d } 0 .. $#st;
  my %listed = map { $_ => 1 } @{$targets->{$d} // []};
  for my $a (@{$aliases->{$d}}) {
    my ($rest) = $a =~ /^(?:runtest|train|slow)-(.+)$/ or next;
    my ($unit, @variant) = split /-/, $rest;
    my @prefix = @variant ? ("$rest.", join(".", $unit, @variant) . ".") : ("$rest.");
    my @mine = grep { my $t = $_; $t !~ $artifact && grep { index($t, $_) == 0 } @prefix } @{$targets->{$d} // []};
    # `UNIT_*` targets count too (a rule-made `X_actual.ml` beside `runtest-X-ppx`,
    # test/ppx), though only a golden named as above makes the action a known diff.
    my @near = grep { $_ !~ $artifact && index($_, "${unit}_") == 0 } @{$targets->{$d} // []};
    my %own = map { in_dir($d, $_) => 1 } @mine, @near;
    # A golden pair diffs its golden against an output a rule makes. A golden
    # with no rule-made target beside it is diffed inside the alias's own action
    # (the run-and-diff shape: `runtest-kernel_attribution`, a `slow-` rule),
    # whose `(deps)` dune never dumps -- so that alias is opaque, not golden
    # (gh-ocannl-1431: an edit to the benchmarks/gh1002_cells.py such a rule
    # names in its `(deps)` missed it).
    my $golden = grep({ /\.expected$/ } @mine) && grep { !/\.expected$/ && !-e in_dir($d, $_) } @mine, @near;
    # With the probe, an alias's actions are its twins (a disabled rule makes
    # none) and its tests' golden diffs: described, unless the converter said
    # otherwise. Without it, a golden pair is the one action known.
    my $why;
    if ($probe_ok) {
      $why = blind_reason($d, $a);
      $own{$_} = 1 for grep { $listed{(dir_of($_))[1]} } @{$twins{$d}{$a} // []};
    }
    push @al, { name => "\@" . in_dir($d, $a), dir => $d, alias => $a,
                described => $probe_ok ? !defined $why : $golden ? 1 : 0, blind_why => $why,
                owners => { map { $_ => 1 } grep { $st[$_]{units}{$unit} } @in }, own => [sort keys %own] };
  }
}

# The rules of those targets: what they run and what they read.
# dune's own build directory is never a source, even where git does not ignore it.
my @changed = grep { $_ ne "" && !m{^_build(?:/|$)} } map { s{^\./}{}r } split /\n/, slurp("$tmp/changed");
# C-family stubs: the objects dune compiles (`X.o`) and the libraries' stub
# archives (`libNAME_stubs.a`) that bundle them, in every directory: an
# object's inputs can be generated elsewhere (a header a generator writes).
my @stub_targets = map { my $d = $_; map { in_dir($d, $_) } grep { /\.o$/ || /^lib.+_stubs\.a$/ } @{$targets->{$d} // []} }
  sort keys %$targets;
my @own_targets = sort { $a cmp $b } keys %{{ map { $_ => 1 } (map { @{$_->{own}} } @al), keys %generated, @stub_targets }};
my %exe;   # "DIR\0NAME" -> stanza indices that build that executable
for my $i (grep { $st[$_]{kind} ne "library" } 0 .. $#st) {
  push @{$exe{"$st[$i]{dir}\0$_"}}, $i for @{$st[$i]{names}};
}
sub glob_regex {   # dune's glob syntax: * ? [set] [!set] {a,b} \x; anything else is literal
  my ($g) = @_;
  my ($re, $depth) = ("", 0);
  while (length $g) {
    if ($g =~ s/^\[(!?)([^\]]+)\]//) {   # a class; [!set] is its complement
      my ($neg, $set) = ($1, $2);
      $re .= "[" . ($neg ? "^" : "") . ($set =~ s/([\\\[\]^])/\\$1/gr) . "]";
      next;
    }
    my $c = substr($g, 0, 1, "");
    if ($c eq "\\" && length $g) { $re .= quotemeta substr($g, 0, 1, "") }   # an escaped character
    elsif ($c eq "*") { $re .= "[^/]*" } elsif ($c eq "?") { $re .= "[^/]" }
    elsif ($c eq "{") { $re .= "(?:"; $depth++ } elsif ($c eq "}" && $depth) { $re .= ")"; $depth-- }
    elsif ($c eq "," && $depth) { $re .= "|" }
    else { $re .= quotemeta $c }
  }
  return $depth ? qr/./ : qr/^$re$/;   # unbalanced: matches anything
}
# A glob, or (and (P...)), (or (P...)), (not P), as a pair of matchers over
# file names: [may, must], what some reading of the printed predicate matches
# and what every reading does. The list wants "may"; `not` swaps the two.
# `dune rules` prints a glob with its escapes dropped (`\{x\}*.md` as
# `{x}*.md`), so which of its metacharacters were literal is lost. The
# readings of a printed glob: as a glob; with its braces, brackets and commas
# literal and its wildcards live; and as its own text, every character
# literal. An escaped wildcard is covered by the live one, an escaped brace or
# bracket by the second reading.
sub predicate_pair {
  my ($p) = @_;
  if (!ref $p) {
    my $re = glob_regex($p);
    my $plain = glob_regex($p =~ s/([{}\[\],])/\\$1/gr);
    return [sub { $_[0] eq $p || $_[0] =~ $re || $_[0] =~ $plain },
            sub { $_[0] eq $p && $_[0] =~ $re && $_[0] =~ $plain }];
  }
  my ($op, $arg) = pair($p, "a glob predicate");
  if ($op eq "not") { my ($may, $must) = @{predicate_pair($arg)}; return [sub { !$must->($_[0]) }, sub { !$may->($_[0]) }] }
  $op =~ /^(?:and|or)$/ && ref $arg or fail("a glob predicate \"$op\" is outside this helper's boundary");
  my @m = map { predicate_pair($_) } @$arg;
  my $all = sub { my ($k, $f) = @_; !grep { !$_->[$k]->($f) } @m };
  my $any = sub { my ($k, $f) = @_; grep { $_->[$k]->($f) } @m };
  return $op eq "and" ? [sub { $all->(0, $_[0]) }, sub { $all->(1, $_[0]) }]
                      : [sub { $any->(0, $_[0]) }, sub { $any->(1, $_[0]) }];
}
sub predicate { predicate_pair($_[0])->[0] }
sub strip_ctx { my ($p) = @_; $p =~ s{^\Q$ctx\E/}{} ? $p : undef }
# `dune rules` over many targets, in batches whose command line stays far below
# the smallest limit (Windows' 32767 characters for the whole command line).
# OCANNL_TOOL_AFFECTED_ARGS overrides the 16000-character budget, for tests.
my $arg_budget = $ENV{OCANNL_TOOL_AFFECTED_ARGS} // 16000;
$arg_budget =~ /^[1-9][0-9]*$/ or fail("OCANNL_TOOL_AFFECTED_ARGS must be a positive integer");
# The tree queried is $rules_root: the probe when there is one.
our $rules_root = $qroot;
our $rules_soft = 0;   # a failure returns undef instead of stopping the run
sub dune_rules {
  my @todo = @_;
  my @rules;
  while (@todo) {
    my ($len, @batch) = (0);
    while (@todo && (!@batch || $len + length($todo[0]) + 3 < $arg_budget)) { $len += length($todo[0]) + 3; push @batch, shift @todo }
    # Only the file `-o` names is the answer. To print a rule, dune builds what
    # the rule's dependencies are computed from (a `%{read:...}` file, say), and
    # what those actions print reaches dune's own streams: on a cold build
    # directory that was `Wrote value of 'backend_extension' to
    # ocannl_backend_extension.txt` in the middle of the sexps (gh-ocannl-1389).
    # Both streams are kept for the failure message, never parsed.
    unlink "$tmp/rules-out";
    my $pid = fork // fail("cannot fork for dune rules");
    if (!$pid) {
      open STDOUT, ">", "$tmp/rules-log" or exit 126;
      open STDERR, ">&", \*STDOUT or exit 126;
      chdir $rules_root or exit 126;
      delete @ENV{qw(DUNE_BUILD_DIR DUNE_ROOT)};
      exec "dune", "rules", "--root", ".", "-o", "$tmp/rules-out", @batch; exit 127;
    }
    unless (waitpid($pid, 0) == $pid && $? == 0) {
      return undef if $rules_soft;
      fail("dune rules failed:\n" . slurp("$tmp/rules-log"));
    }
    my $text = slurp("$tmp/rules-out");
    # A refusal from here on keeps what dune wrote, so its cause can be read
    # rather than guessed.
    local ($raw_text, $raw_what) = ($text, "dune-rules");
    my @at;
    my @got = @{parse_sexp($text, "dune rules", \@at)};
    for my $k (0 .. $#got) {
      ref $got[$k] or fail("unexpected atom in dune rules output: \"$got[$k]\" at byte $at[$k]: ..." . around($text, $at[$k]) . "...");
    }
    push @rules, @got;
  }
  return @rules;
}
# A copy of an answer this reader refused, outside the temporary directory the
# exit trap removes.
sub keep_raw {
  my ($text, $what) = @_;
  my $dir = $ENV{TMPDIR} // "/tmp";
  my $f = "$dir/affected-$what.$$.txt";
  open my $o, ">", $f or return "(could not write $f)";
  print $o $text; close $o;
  return $f;
}
# The dune files themselves, read as text: what an alias's own rule needs, which
# `dune rules` never dumps (it accepts no alias), and which stanzas a dune-file
# edit changed. A reader for dune's syntax, as its lexer has it
# (src/dune_sexp/lexer.mll): `;` end-of-line comments only (`#|` and `#;` are
# atom text), quoted strings, block strings, atoms. It dies on text it cannot
# read; the caller decides whether that stops the run. A block string comes
# back OPAQUE (\x01-prefixed: its escapes and its leading space are not
# decoded), which every reader of a stanza's needs takes as unknown.
sub parse_dune {
  my ($s, $what) = @_;
  my @stack = ([]);
  my $put = sub { push @{$stack[-1]}, $_[0] };
  pos($s) = 0;
  while (pos($s) < length $s) {
    next if $s =~ /\G\s+/gc || $s =~ /\G;[^\n]*/gc;
    if ($s =~ /\G\(/gc) { push @stack, []; next }
    if ($s =~ /\G\)/gc) {
      @stack > 1 or die "unbalanced parentheses in $what at byte " . (pos($s) - 1) . "\n";
      my $l = pop @stack; $put->($l); next;
    }
    if ($s =~ /\G"\\[|>]([^\n]*)/gc) {   # block string: consecutive "\| lines
      my $str = $1;
      $str .= "\n$1" while $s =~ /\G\n[ \t]*"\\[|>]([^\n]*)/gc;
      $put->("\x01" . $str); next;
    }
    if ($s =~ /\G"((?:[^"\\]|\\.)*)"/gcs) {
      my $a = $1 =~ s/\\\n[ \t]*//gr;
      $a =~ s/\\(x[0-9A-Fa-f]{2}|[0-9]{3}|.)/length $1 > 1 ? ($1 =~ m{^x} ? chr hex substr($1, 1) : chr $1)
                                            : $1 eq "n" ? "\n" : $1 eq "t" ? "\t" : $1 eq "r" ? "\r" : $1/gse;
      $put->($a); next;
    }
    if ($s =~ /\G([^\s()";]+)/gc) { $put->($1); next }
    die "unreadable text in $what at byte " . pos($s) . "\n";
  }
  @stack == 1 or die "unbalanced parentheses in $what\n";
  return $stack[0];
}
sub has_opaque { my ($x) = @_; ref $x ? scalar(grep { has_opaque($_) } @$x) : $x =~ /^\x01/ }
sub canon {   # a stanza or form as text, for messages
  my ($x) = @_;
  return "(" . join(" ", map { canon($_) } @$x) . ")" if ref $x;
  return "\"" . ($x =~ s/(["\\])/\\$1/gr =~ s/\n/\\n/gr) . "\"";
}
sub dfield {   # the arguments of a stanza's first (NAME ...) field
  my ($x, $name) = @_;
  for my $f (@$x[1 .. $#$x]) { return [@$f[1 .. $#$f]] if ref $f && @$f && !ref $f->[0] && $f->[0] eq $name }
  return undef;
}
# Stanzas of DIR, `(subdir ...)` flattened into the directory it names.
sub dune_stanzas {
  my ($dir, $items) = @_;
  my @out;
  for my $x (@$items) {
    next unless ref $x && @$x && !ref $x->[0];
    if ($x->[0] eq "subdir" && @$x >= 2 && !ref $x->[1]) {
      my $sub = rpath($dir, $x->[1]);
      if (defined $sub) { push @out, dune_stanzas($sub, [@$x[2 .. $#$x]]); next }
    }
    push @out, { dir => $dir, kind => $x->[0], x => $x };
  }
  return @out;
}
sub rpath {   # BASE/P normalized, repository-relative; undef when it leaves the repository
  my ($base, $p) = @_;
  return undef if $p =~ m{^/};
  my @parts = $base eq "." ? () : split m{/}, $base;
  for my $c (split m{/}, $p) {
    next if $c eq "" || $c eq ".";
    if ($c eq "..") { @parts or return undef; pop @parts } else { push @parts, $c }
  }
  return @parts ? join("/", @parts) : ".";
}
my @dstanzas;   # every stanza of every dune file, as read from the worktree
my %dune_text;  # dune file -> its text
for my $f (grep { $_ ne "" } split /\n/, slurp("$tmp/dune-files")) {
  next unless -f $f;
  $dune_text{$f} = slurp($f);
  my $items = eval { parse_dune($dune_text{$f}, $f) } or fail($@ =~ s/\n$//r);
  push @dstanzas, dune_stanzas((dir_of($f))[0], $items);
}
# An edit to a fragment a dune file `(include ...)`s (or `dynamic_include`s)
# reaches like an edit to that dune file.
my %fragment_dirs;   # fragment -> every directory whose dune file includes it
for my $s (grep { $_->{kind} =~ /^(?:dynamic_)?include$/ } @dstanzas) {
  my $frag = $s->{x}[1];
  my $p = !ref $frag && $frag !~ /%\{/ ? rpath($s->{dir}, $frag) : undef;
  push @{$fragment_dirs{$p}}, $s->{dir} if defined $p;
}

# Each queried target's rule: the executables it runs, the files and globs it
# reads, and the generated files it reads (`via`), whose rules are queried in
# turn -- a `copy_files` copy, a file list -- so what a rule reaches through
# them counts as its own. Compiled objects and executables are not followed:
# an executable is reached through its stanza.
# A compiled artifact a rule reads directly -- a stanza's `.NAME.objs` or
# `.NAME.eobjs` directory, or an installed library's files -- makes that
# stanza an owner of the reader, as an executable it runs does.
sub compiled_owners {
  my ($path) = @_;   # a build path, context prefix included
  if ($path =~ m{^_build/install/[^/]+/lib/([^/]+)(?:/([^/]+))?/}) {
    my @n = grep { defined $lib_by_name{$_} } (defined $2 ? ("$1.$2") : ()), $1;
    return @n ? ($lib_by_name{$n[0]}) : ();
  }
  my $rel = strip_ctx($path) // return ();
  my ($d, $x, $e) = $rel =~ m{^(?:(.*)/)?\.([^/]+)\.(e?)objs(?:/|$)} or return ();
  $d //= ".";
  return grep { $st[$_]{dir} eq $d && ($e ? $st[$_]{kind} ne "library" : $st[$_]{kind} eq "library")
                && $st[$_]{units}{$x} } 0 .. $#st;
}
my (%runs, %reads, %globs, %via, %adeps);
my %queried;
# An alias some stanza makes depend on files (a rule with targets, an
# `(alias)` stanza, `copy_files`): `dune rules @ALIAS` expands those
# dependencies (not actions), helper aliases included, so what it prints is
# the alias's own too. One query per alias, since one answer for several does
# not say whose each rule is.
sub source_query {   # ALIAS [FORCE]: FORCE asks dune whatever the manifest says
  my ($a, $force) = @_;
  return unless $probe_ok && $a->{described} && ($force || $sources{$a->{dir}}{$a->{alias}});
  local $rules_soft = 1;
  my @got = dune_rules($a->{name});
  if (grep { !defined } @got) {
    $a->{described} = 0;
    $a->{blind_why} = "dune rules $a->{name} failed";
    return;
  }
  my %own = map { $_ => 1 } @{$a->{own}};
  for my $rule (@got) {
    my $tg = field($rule, "targets");
    my @t = map { atom($_, "targets") } map { items(field($tg // [], $_)) } "files", "directories";
    if (my @rel = grep { defined } map { strip_ctx($_) } @t) { $own{$_} = 1 for @rel; next }
    # An installed file (what `(package PKG)` needs) is a link to a file of the
    # build: that file is the alias's own, or a library's compiled artifact.
    for my $dep (items(field($rule, "deps"))) {
      my ($k, $v) = ref $dep ? @$dep : ($dep);
      next if $k =~ /^(?:Env|Universe|Sandbox_config)$/;
      my ($where, $path) = $k eq "File" ? pair($v, "a File dependency") : ("", "");
      if ($where eq "In_build_dir" && (my @o = compiled_owners($path))) { $a->{owners}{$_} = 1 for @o }
      elsif ($where eq "In_build_dir" && defined(my $rel = strip_ctx($path))) { $own{$rel} = 1 }
      elsif ($where ne "External") {
        $a->{described} = 0;
        $a->{blind_why} = "$a->{name} needs @t, whose rule this reader does not follow";
      }
    }
  }
  $a->{own} = [sort keys %own];
}
source_query($_) for @al;
@own_targets = sort { $a cmp $b } keys %{{ map { $_ => 1 } @own_targets, map { @{$_->{own}} } @al }};
sub query_rules {   # TARGET...: 1, or 0 when dune could not print them (under $rules_soft)
my @queue = @_;
while (@queue) {
  my @got = dune_rules(@queue);
  return 0 if grep { !defined } @got;
  $queried{$_} = 1 for @queue;
  my %next;
  for my $rule (@got) {
    my $tg = field($rule, "targets");
    my @files = map { atom($_, "targets") } map { items(field($tg // [], $_)) } "files", "directories";
    my (%r, %p, @g, %v, %ad);
    for my $dep (items(field($rule, "deps"))) {
      next if !ref $dep && $dep eq "Universe";   # always reruns; reads nothing in particular
      ref $dep && @$dep && !ref $dep->[0] or fail("unexpected dependency in dune rules output");
      my $k = $dep->[0];
      if ($k eq "File") {
        my ($where, $path) = pair($dep->[1], "a File dependency");
        if ($where eq "In_build_dir") {
          if (my @o = compiled_owners($path)) { $r{$_} = 1 for @o; next }
          my $rel = strip_ctx($path) // next;   # another context's, or _build/install
          if ($rel =~ m{^(?:(.*)/)?([^/]+)\.(?:exe|bc)$}) { $r{$_} = 1 for @{$exe{($1 // ".") . "\0$2"} // []} }
          else {
            $p{$rel} = 1;
            $v{$rel} = 1 if !-e $rel && $rel !~ $artifact && $rel !~ m{(?:^|/)\.[^/]+\.e?objs/};
          }
        } elsif ($where eq "In_source_tree") { $p{$path} = 1 }
        elsif ($where ne "External") { fail("a File dependency \"$where\" is outside this helper's boundary") }
      } elsif ($k eq "glob") {
        my $spec = $dep->[1];
        my ($where, $path) = pair(field($spec, "dir"), "a glob dependency");
        next if $where eq "External";
        if ($where eq "In_build_dir" and my @o = compiled_owners("$path/")) { $r{$_} = 1 for @o; next }
        next if (field($spec, "only_generated_files") // "false") eq "true";
        my $gd = $where eq "In_build_dir" ? strip_ctx($path) : $path;
        next unless defined $gd;
        push @g, [$gd, predicate(field($spec, "predicate"))];
      } elsif ($k eq "Alias") {
        # Another alias this rule needs: when that one is listed, so is this.
        my $spec = $dep->[1];
        my ($where, $path) = pair(field($spec, "dir"), "an Alias dependency");
        my $ad = $where eq "In_build_dir" ? strip_ctx($path) : $path;
        defined $ad or ($path eq $ctx and $ad = ".") or next;
        $ad{"\@" . in_dir($ad, atom(field($spec, "name"), "alias name"))} = 1;
      } elsif ($k !~ /^(?:Env|Universe|Sandbox_config)$/) {
        fail("a dependency \"$k\" in dune rules output is outside this helper's boundary");
      }
    }
    for my $f (@files) {
      my $t = strip_ctx($f) // next;
      $runs{$t} = \%r; $reads{$t} = \%p; $globs{$t} = \@g; $via{$t} = [sort keys %v]; $adeps{$t} = \%ad;
    }
    $next{$_} = 1 for grep { !$queried{$_} } keys %v;
  }
  exists $runs{$_} or fail("dune rules printed no rule for $_") for @queue;
  @queue = sort keys %next;
}
return 1;
}
query_rules(@own_targets);
# A helper alias a queried rule needs (`(deps (alias NAME))`, NAME no per-test
# alias) is an entry of its own, never printed: its twins are queried and read
# like a per-test alias's, and an alias needing it is listed when it would be
# (see the tiers below), to a fixpoint.
{
  my %is_al = map { $_->{name} => 1 } @al;
  for (my $grew = 1; $grew;) {
    $grew = 0;
    my %need = map { %{$adeps{$_} // {}} } sort keys %queried;
    for my $n (sort grep { !$is_al{$_} } keys %need) {
      $is_al{$n} = 1;
      my ($d, $x) = $n =~ m{^\@(?:(.*)/)?([^/]+)$} or next;
      $d //= ".";
      my %listed = map { $_ => 1 } @{$targets->{$d} // []};
      # One dune makes itself (`.PKG-files`, what `(package PKG)` needs) is
      # dependencies only, which `dune rules @ALIAS` expands; one dune does not
      # list otherwise is not known.
      my $internal = $x =~ /^\./;
      my $why = !$probe_ok ? "no probe" : $internal ? undef
              : grep({ $_ eq $x } @{$aliases->{$d} // []}) ? blind_reason($d, $x) : "dune does not list it";
      my $h = { name => $n, dir => $d, alias => $x, hidden => 1, described => !defined $why, blind_why => $why,
                owners => {}, own => [grep { $listed{(dir_of($_))[1]} } @{$twins{$d}{$x} // []}] };
      push @al, $h;
      source_query($h, $internal);
      query_rules(grep { !$queried{$_} } @{$h->{own}});
      $grew = 1;
    }
  }
}
# A rule-made file with the name of a file it reads, in another directory, is
# a copy of it (`copy_files`): an opaque action there may read the copy.
my %copies;   # path -> [directories holding a copy of it]
for my $t (sort keys %queried) {
  my ($td, $tb) = dir_of($t);
  for my $src (keys %{$reads{$t}}) {
    my ($sd, $sb) = dir_of($src);
    push @{$copies{$src}}, $td if $sb eq $tb && $sd ne $td;
  }
}
{ # Fold each target's generated inputs into it (cycle-safe: a file is folded once).
  my %done;
  sub fold {
    my ($t) = @_;
    return if $done{$t}++;
    for my $u (@{$via{$t}}) {
      fold($u);
      $runs{$t} = { %{$runs{$t}}, %{$runs{$u}} };
      $reads{$t} = { %{$reads{$t}}, %{$reads{$u}} };
      $globs{$t} = [@{$globs{$t}}, @{$globs{$u}}];
      $adeps{$t} = { %{$adeps{$t}}, %{$adeps{$u}} };
    }
  }
  fold($_) for sort keys %queried;
}

# A stub object belongs to the library whose stub archive (`libNAME_stubs.a`,
# NAME its local name) reads it. One no archive reads is linked into an
# executable, and which one dune does not say without computing link rules.
my %obj_libs;   # object path -> {library stanza index}
for my $t (grep { /(?:^|\/)lib.+_stubs\.a$/ } @stub_targets) {
  my ($d, $b) = dir_of($t);
  my ($name) = $b =~ /^lib(.+)_stubs\.a$/;
  my @libs = grep { $st[$_]{kind} eq "library" && $st[$_]{dir} eq $d && $st[$_]{units}{$name} } 0 .. $#st;
  for my $o (grep { /\.o$/ } keys %{$reads{$t} // {}}) { $obj_libs{$o}{$_} = 1 for @libs }
}

# A generated module's stanza is reached through what its rule reads (a
# `.mll`, a `.mly`, a (select) choice) and through the executable it runs.
for my $g (keys %generated) {
  for my $o (@{$generated{$g}}) { push @{$st[$o]{to}}, keys %{$runs{$g}}; push @{$st[$o]{gen_to}}, keys %{$runs{$g}} }
}
# A stub object's owners: the library whose stub archive bundles it, or else
# every executable and `tests` stanza of its directory. They are reached
# through the executables its rule runs, generated inputs folded in (a
# generator writing a header the object includes).
sub obj_owners {
  my ($o) = @_;
  my @libs = sort keys %{$obj_libs{$o} // {}};
  my ($od) = dir_of($o);
  return @libs ? @libs : grep { $st[$_]{kind} ne "library" && $st[$_]{dir} eq $od } 0 .. $#st;
}
for my $o (grep { /\.o$/ } @stub_targets) {
  for my $w (obj_owners($o)) { push @{$st[$w]{to}}, keys %{$runs{$o}}; push @{$st[$w]{gen_to}}, keys %{$runs{$o}} }
}

my (%hit, %dune_dirs, %data_dirs, %read_by, @unmapped, @unread);
# How a stanza was hit: by changed modules it owns (NAME => 1), or as a whole
# (a generated module, a stub, a changed stanza, a tests directory's modules).
my (%mod_hit, %whole_hit, %mod_paths);
my $any_data = 0;
sub label { my $s = $st[$_[0]]; "$s->{kind} @{$s->{names}} ($s->{dir})" }
# A glob entry is [DIR, matcher over a file name, recursive?]; a recursive one
# (glob_files_rec, source_tree) matches in DIR and beneath it.
sub reads {
  my ($t, $p, $d, $b) = @_;
  $reads{$t}{$p} || grep { ($_->[2] ? under($d, $_->[0]) : $_->[0] eq $d) && $_->[1]->($b) } @{$globs{$t}};
}
sub under { my ($x, $root) = @_; $root eq "." || $x eq $root || index($x, "$root/") == 0 }

# A changed dune file compared across revisions (gh-ocannl-1541): the old
# probe is this one with the file at the old revision. What it changed is
# what dune says changed -- the rules of its directory's targets, twins
# included, that differ between the probes -- plus, from dune's canonical text
# of both sides, the stanzas whose compilation the rules shown do not cover
# (flags live in the object rules): a changed library, executables or tests
# stanza, named by its literal names as dune lists them here. Any other
# changed form (`env`, `subdir`, `include`, `dirs`, an unknown head, an
# `(alias)` with an action), or a changed rule this reader cannot map to an
# owner, keeps the reach of the whole directory beneath the file, and says why.
my (%covered, %direct, %delta_paths);
my %unscoped = map { my ($f, $w) = split /\t/; ($f => $w) } grep { $_ ne "" } split /\n/, slurp("$tmp/unscoped");
my @scoped = grep { $_ ne "" } split /\n/, slurp("$tmp/scoped");
if (@scoped) {
  my @sdirs = map { (dir_of($_))[0] } @scoped;
  my ($old_aliases, $old_targets);
  { local $listing_soft = 1; $old_aliases = listing("old-aliases", @sdirs); $old_targets = listing("old-targets", @sdirs) }
  unless ($old_aliases && $old_targets) {
    $unscoped{$_} = "dune does not load the old revision's dune file beside these sources (see dune show in the old probe)" for @scoped;
    @scoped = ();
  }
  my %twin_of;   # twin path -> [aliases], either side
  for my $m ("manifest", "manifest-old") {
    for (split /\n/, slurp("$tmp/$m")) {
      my ($k, $d, $t, @a) = split /\t/;
      push @{$twin_of{in_dir($d, $t)}}, @a if $k eq "twin";
    }
  }
  my %forms;   # side -> file -> [form records]
  for my $side ("new", "old") {
    for (split /\n/, slurp($side eq "new" ? "$tmp/forms" : "$tmp/forms-old")) {
      my ($file, $d, $head, $names, $act, $canon) = split /\t/, $_, 6;
      push @{$forms{$side}{$file}}, { dir => $d, head => $head, names => [grep { $_ ne "" } split /,/, $names],
                                      action => $act, canon => $canon };
    }
  }
  # The `(alias)` stanzas of this side, by directory and name: what an alias a
  # stanza makes depend on files may need is read off its stanzas' tokens, to a
  # fixpoint (a mention errs towards "needs it"; a template means anything).
  my (%alias_forms, %alias_dirs);
  for my $file (sort keys %{$forms{new} // {}}) {
    for my $r (grep { $_->{head} eq "alias" } @{$forms{new}{$file}}) {
      for my $n (@{$r->{names}}) { push @{$alias_forms{"$r->{dir}\0$n"}}, $r->{canon}; $alias_dirs{$n}{$r->{dir}} = 1 }
    }
  }
  sub mentions {   # DIR X H SEEN: may an (alias) stanza of X in DIR need the alias H?
    my ($sd, $x, $h, $seen) = @_;
    return 0 if $seen->{"$sd\0$x"}++;
    my @texts = @{$alias_forms{"$sd\0$x"} // []};
    # No (alias) stanza: its dependencies come from rules, whose needs dune prints.
    return 0 unless @texts;
    my @tok = grep { $_ ne "" } map { split /[\s()]+/ } @texts;
    return 1 if grep { /%\{/ || $_ eq $h || m{/\Q$h\E$} } @tok;
    for my $t (@tok) {
      my ($b) = $t =~ m{([^/]+)$} or next;
      for my $od (sort keys %{$alias_dirs{$b} // {}}) { return 1 if mentions($od, $b, $h, $seen) }
    }
    return 0;
  }
  my %rules_shown = map { $_ => 1 } "rule", "copy_files", "copy_files#", "ocamllex", "ocamlyacc", "menhir";
  my %kind_of = (test => "tests", tests => "tests", executable => "executables", executables => "executables", library => "library");
  my $per_test = qr/^(?:runtest|train|slow)-/;
  # A rule as text, the probe roots made one.
  my @roots = map { quotemeta } grep { defined && $_ ne "." } $qroot, $old_root;
  my $roots = @roots ? qr/(?:@{[join "|", @roots]})/ : undef;
  sub rule_text { my ($x) = @_; ref $x ? "(" . join(" ", map { rule_text($_) } @$x) . ")" : $x }
  FILE: for my $f (@scoped) {
    my ($d) = dir_of($f);
    my (@why, @parts, %whole, %direct_here, @paths);
    # The changed forms: each side's canonical forms not matched on the other.
    my (%count, @changed_forms);
    $count{$_->{canon}}++ for @{$forms{new}{$f} // []};
    $count{$_->{canon}}-- for @{$forms{old}{$f} // []};
    for my $side ("new", "old") {
      my %left = map { $_ => ($side eq "new" ? $count{$_} : -$count{$_}) } keys %count;
      for my $r (@{$forms{$side}{$f} // []}) { push @changed_forms, { %$r, side => $side } if $left{$r->{canon}}-- > 0 }
    }
    for my $r (@changed_forms) {
      my $h = $r->{head};
      next if $rules_shown{$h};
      if ($h eq "alias" && !$r->{action}) {
        # The alias itself, per-test or a helper (whose entry carries the change to
        # the aliases whose rules need it); and, since `dune rules @ALIAS` folds a
        # helper's dependencies into an alias some stanza makes depend on files
        # without saying which came through it, every such alias too.
        for my $x (@{$r->{names}}) {
          $direct_here{in_dir($d, $x)} = "its (alias) stanza changed";
          next if $x =~ $per_test;
          for my $sd (sort keys %sources) {
            $direct_here{in_dir($sd, $_)} = "it may depend on the changed (alias) $x"
              for grep { $_ =~ $per_test && mentions($sd, $_, $x, {}) } keys %{$sources{$sd}};
          }
        }
        next;
      }
      if (my $kind = $kind_of{$h}) {
        unless (@{$r->{names}}) { push @why, "a changed ($h) with no literal name"; next }
        for my $n (@{$r->{names}}) {
          my @ix = grep { $st[$_]{dir} eq $d && $st[$_]{kind} eq $kind
                          && ($kind eq "library" ? $st[$_]{units}{$n} : grep { $_ eq $n } @{$st[$_]{names}}) } 0 .. $#st;
          if (@ix) { $whole{$_} = 1 for @ix; next }
          # Gone from this side: nothing of it is left to run.
          my $gone = $r->{side} eq "old" && !grep({ $_ eq "$n.exe" } @{$targets->{$d} // []})
                     && !grep { $_ eq "runtest-$n" } @{$aliases->{$d} // []};
          push @why, "a changed ($h) named $n, which dune does not list here" unless $gone;
        }
        next;
      }
      push @why, $h eq "alias" ? "a changed (alias) with an action" : "a changed ($h) stanza";
    }
    unless (@why) {
      # The rules of every target of the directory, on both sides.
      my %text;
      for my $side ("old", "new") {
        my @t = map { in_dir($d, $_) } @{($side eq "old" ? $old_targets : $targets)->{$d} // []};
        local $rules_root = $side eq "old" ? $old_root : $qroot;
        local $rules_soft = 1;
        my @got = @t ? dune_rules(@t) : ();
        if (grep { !defined } @got) { push @why, "dune rules failed on the $side side"; last }
        for my $rule (@got) {
          my $s = rule_text($rule);
          $s =~ s/$roots/<probe>/g if $roots;
          my $tg = field($rule, "targets");
          $text{$side}{$_} = $s for grep { defined } map { strip_ctx(atom($_, "targets")) } map { items(field($tg // [], $_)) } "files", "directories";
        }
      }
      my @delta = grep { ($text{old}{$_} // "") ne ($text{new}{$_} // "") } sort keys %{{ %{$text{old} // {}}, %{$text{new} // {}} }};
      for my $t (@why ? () : @delta) {
        my ($td, $tb) = dir_of($t);
        my $new_side = exists $text{new}{$t};
        if ($tb =~ /^__affected_probe_/) {
          $direct_here{in_dir($td, $_)} = "a rule of it changed" for @{$twin_of{$t} // []};
        } elsif ($tb =~ /^(.+)\.(?:exe|bc)$/) {
          my @ix = @{$exe{"$td\0$1"} // []};
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, which no stanza here builds" if !@ix && $new_side;
        } elsif ($tb =~ /^([^.]+)\.pp\.mli?$/) {
          my @ix = @{$by_modkey{"$td\0" . ucfirst $1} // []};
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, whose module no stanza here owns" if !@ix && $new_side;
        } elsif ($tb =~ /\.o$/) {
          $whole{$_} = 1 for obj_owners($t);
        } elsif ($tb =~ /^(?:lib(.+)_stubs\.a|(.+)\.(?:cmxa|cma|cmxs|a))$/) {
          my $n = $1 // $2;
          my @ix = grep { $st[$_]{kind} eq "library" && $st[$_]{dir} eq $td && $st[$_]{units}{$n} } 0 .. $#st;
          $whole{$_} = 1 for @ix;
          push @why, "a changed rule for $t, which no library here makes" if !@ix && $new_side;
        } elsif ($tb =~ $artifact) {
          push @why, "a changed rule for $t, an artifact this reader does not map" if $new_side;
        } else {
          push @paths, $t;
        }
      }
      # A per-test alias new on this side is listed: it never ran.
      my %was = map { $_ => 1 } @{$old_aliases->{$d} // []};
      $direct_here{in_dir($d, $_)} = "it is new" for grep { $_ =~ $per_test && !$was{$_} } @{$aliases->{$d} // []};
      push @parts, scalar(@delta) . " changed rule(s)";
    }
    if (@why) {
      $unscoped{$f} = join("; ", @why);
      next FILE;
    }
    $covered{$f} = 1;
    $hit{$_} = $whole_hit{$_} = 1 for keys %whole;
    $direct{"\@$_"} //= $direct_here{$_} for keys %direct_here;
    $delta_paths{$_} = 1 for @paths;
    push @parts, scalar(keys %whole) . " stanza(s) changed as a whole" if %whole;
    push @parts, scalar(keys %direct_here) . " alias(es) whose own rules changed" if %direct_here;
    print STDERR "affected: $f: compared with the old revision: " . join(", ", @parts) . "\n";
  }
}
print STDERR "affected: $_: its whole directory, because $unscoped{$_}\n" for sort keys %unscoped;

for my $p (@changed) {
  my ($d, $b) = dir_of($p);
  for my $a (@al) {
    for my $t (@{$a->{own}}) { if (reads($t, $p, $d, $b)) { $read_by{$p}{$a->{name}} = 1; last } }
  }
  my $n_read = keys %{$read_by{$p} // {}};
  my (@ix, @parts, $unmapped, $by_module);
  my @gen = grep { reads($_, $p, $d, $b) } sort keys %generated;
  if (@gen) {
    my %seen_g;
    my @g = grep { !$seen_g{$_}++ } map { @{$generated{$_}} } @gen;
    $hit{$_} = $whole_hit{$_} = 1 for @g;
    push @parts, "the rule generating " . join(", ", @gen) . " reads it: " . join(", ", map { label($_) } @g);
  }
  for my $o (grep { /\.o$/ && reads($_, $p, $d, $b) } @stub_targets) {
    my @libs = sort keys %{$obj_libs{$o} // {}};
    my @o = obj_owners($o);
    $hit{$_} = $whole_hit{$_} = 1 for @o;
    push @parts, "compiled into $o, " . (@libs ? "bundled by " . join(", ", map { label($_) } @libs)
      : "which no stub archive bundles, so every executable and test stanza of its directory (" . scalar(@o) . ")");
  }
  # An included fragment counts as the dune file of every directory including it.
  my @frag_of = @{$fragment_dirs{$p} // []};
  if ($covered{$p}) {
    push @parts, "compared with the old revision (above)";
  } elsif ($b eq "dune" || $b eq "dune-project" || $b eq "dune-workspace" || @frag_of) {
    for my $root (@frag_of ? @frag_of : ($d)) {
      $data_dirs{$root} = 1;
      # A `dune` file's settings (`env`, `(dirs)`, ...) apply beneath it as well,
      # as `dune-project`'s language and extensions do: every directory under it.
      $dune_dirs{$_} = 1 for grep { under($_, $root) } $root, keys %$aliases;
      my @here = grep { under($st[$_]{dir}, $root) } 0 .. $#st;
      push @ix, @here;
      # Its rules make the files there: a rule elsewhere reading one of them (a
      # fixture's generated sources) reads the change.
      my $n_made = 0;
      for my $a (grep { !under($_->{dir}, $root) } @al) {
        for my $t (@{$a->{own}}) {
          next unless grep({ under((dir_of($_))[0], $root) } keys %{$reads{$t}}) || grep { under($_->[0], $root) } @{$globs{$t}};
          $n_made++ unless $read_by{$p}{$a->{name}}++;
          last;
        }
      }
      push @parts, "every per-test alias and all " . scalar(@here) . " stanza(s) " . ($root eq "." ? "of the project" : "under $root")
        . ($n_made ? "; files its rules make are read by the rules of $n_made alias(es) elsewhere" : "");
    }
  } elsif ($b =~ /^([^.]+)(?:\.[^.]+)*\.mli?$/) {
    my $mod = ucfirst $1;
    my $how = "module $mod of";
    @ix = @{$by_modkey{"$d\0$mod"} // []};
    $by_module = $mod if @ix;
    unless (@ix) {
      my @tests = grep { $st[$_]{kind} eq "tests" && $st[$_]{dir} eq $d } 0 .. $#st;
      my @named = grep { $st[$_]{units}{lcfirst $mod} } @tests;
      @ix = @named ? @named : @tests;
      $how = @named ? "test module $mod of" : "a module of a tests stanza here (dune does not describe theirs), so all of";
    }
    unless (@ix || @gen) {
      # No stanza of this build owns it. One that dune lists but does not
      # compile here (an optional backend) may: those are reached, unknown
      # modules and all, and the path stays UNMAPPED either way.
      @ix = grep { $st[$_]{undescribed} && $st[$_]{dir} eq $d } 0 .. $#st;
      $how = "no stanza of this build owns $mod; reaching the stanzas dune lists here but does not build:";
      $unmapped = 1;
    }
    push @parts, "$how " . (@ix > 3 ? scalar(@ix) . " stanzas" : join(", ", map { label($_) } @ix)) if @ix;
  }
  $hit{$_} = 1 for @ix;
  if (defined $by_module) { $mod_hit{$_}{$by_module} = 1 for @ix; $mod_paths{$by_module}{$p} = 1 } else { $whole_hit{$_} = 1 for @ix }
  push @parts, "read by the rules of $n_read alias(es)" if $n_read;
  print STDERR "affected: $p: " . join("; ", @parts) . "\n" if @parts;
  # Any path but a module source a stanza owns may be an input of an alias
  # action dune does not dump (see `opaque` below).
  unless ($b =~ /\.mli?$/ && @ix && !$unmapped) { $data_dirs{$_} = 1 for $d, @{$copies{$p} // []} }
  # A path no compiler or dune reads as a build input (a script, a config, a
  # golden, a note) may be any opaque action's `(deps)`, wherever it lies.
  $any_data = 1 unless $b =~ /\.(?:mli?|mll|mly|c|h|cc|cpp|cxx|hh|hpp)$/ || $b =~ /^dune(?:-project|-workspace)?$/;
  if ($unmapped) { push @unmapped, $p } elsif (!@parts) { push @unread, $p }
}
# The files whose rules a compared dune edit changed (above), as the build has
# them: what reads one is reached as by a changed path -- an alias's own rules,
# a generated module's rule, a stub object's, a copy elsewhere -- and so is a
# stanza whose link or preprocessing rule reads one (`link_deps`,
# `preprocessor_deps`), which those rules are asked for here.
if (%delta_paths) {
  my %stanza_rule;   # .exe or .pp.ml(i) target -> stanza indices
  for my $d (sort keys %$targets) {
    for my $t (@{$targets->{$d}}) {
      my @ix = $t =~ /^(.+)\.(?:exe|bc)$/ ? @{$exe{"$d\0$1"} // []}
             : $t =~ /^([^.]+)\.pp\.mli?$/ ? @{$by_modkey{"$d\0" . ucfirst $1} // []} : ();
      $stanza_rule{in_dir($d, $_)} = [@ix] for @ix ? ($t) : ();
    }
  }
  # One dune cannot print (a rewriter it cannot drive, say) is one dune cannot
  # build either: its stanza has nothing to run, which is said.
  {
    local $rules_soft = 1;
    my @todo = grep { !$queried{$_} } sort keys %stanza_rule;
    unless (query_rules(@todo)) {
      for my $t (@todo) {
        next if $queried{$t} || query_rules($t);
        print STDERR "affected: dune rules $t failed (dune cannot build it either), so what it reads is not known\n";
      }
    }
  }
  fold($_) for sort keys %queried;
  for my $p (sort keys %delta_paths) {
    my ($d, $b) = dir_of($p);
    for my $a (@al) {
      for my $t (@{$a->{own}}) { if ($t eq $p || reads($t, $p, $d, $b)) { $read_by{$p}{$a->{name}} = 1; last } }
    }
    for my $g (grep { reads($_, $p, $d, $b) } sort keys %generated) { $hit{$_} = $whole_hit{$_} = 1 for @{$generated{$g}} }
    for my $o (grep { /\.o$/ && reads($_, $p, $d, $b) } @stub_targets) { $hit{$_} = $whole_hit{$_} = 1 for obj_owners($o) }
    # A stanza's own module sources are its modules, not inputs of this kind.
    for my $t (grep { reads($_, $p, $d, $b) } sort keys %stanza_rule) {
      next if $b =~ /\.mli?$/ && grep { $_ eq $p } map { @$_ } values %{$st[$stanza_rule{$t}[0]]{mods} // {}};
      $hit{$_} = $whole_hit{$_} = 1 for @{$stanza_rule{$t}};
    }
    $data_dirs{$_} = 1 for @{$copies{$p} // []};
  }
}

# Reach, to a fixpoint: a stanza requiring a reached library is reached.
my %reached = %hit;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $i (0 .. $#st) {
    next if $reached{$i};
    if (grep { $reached{$_} } @{$st[$i]{to}}) { $reached{$i} = 1; $grew = 1 }
  }
}

# The link-only ranking (gh-ocannl-1431). Link reach is sound and blunt: an
# edit to one module of arrayjit reaches every test linking arrayjit, ~600
# aliases. A reached alias is ranked "link-only" when no reached owner of it
# links a library hit as a whole (by a stub, a generated module, a stanza
# edit: no name says what changed there), and none names the changed part of
# a changed module in its own code. Its own code: its stanza's sources,
# the sources of a project ppx rewriter it is preprocessed with, and the
# modules of the private (unpublished) libraries it links that that code names,
# to a fixpoint. Names the changed part: mentions the module, or a name
# re-exporting it (`module A = ...M` with A not M, or an `include ...M`), and,
# when the git hunks place every changed line in a top-level definition, one of
# those definitions too (see %path_defs). Mentions are tokens, comments and
# strings included, so a name only ever errs towards "names it". What this does
# NOT see: what the owner reaches through other library code (every compile
# goes through Low_level.optimize_proc, which no test names; gh-ocannl-1235's
# Register_tile miss was of this kind). So link-only aliases stay printed, LAST,
# and named on stderr; `--referenced-only` leaves them off stdout.
my %tok_cache;
sub tokens_of {
  my ($f) = @_;
  return $tok_cache{$f} //= do {
    my %t;
    if (open my $fh, "<", $f) { local $/; my $s = <$fh> // ""; $t{$1} = 1 while $s =~ /\b([A-Za-z_][A-Za-z0-9_']*)/g }
    \%t;
  };
}
sub dnames { my ($s) = @_; map { ref $_ ? () : $_ } map { @{dfield($s->{x}, $_) // []} } qw(name names public_name public_names) }
sub static_of {   # the dune stanza an @st entry was read from, or undef
  my ($i) = @_;
  my $s = $st[$i];
  return $s->{static} if exists $s->{static};
  my %n = map { $_ => 1 } @{$s->{names}};
  my $lib = $s->{kind} eq "library";
  my @m = grep {
    $_->{dir} eq $s->{dir} && $_->{kind} =~ /^(?:library|executables?|tests?)$/
      && ($_->{kind} eq "library") == $lib && grep { $n{$_} } dnames($_)
  } @dstanzas;
  return $s->{static} = @m == 1 ? $m[0] : undef;
}
sub stanza_sources {   # an @st entry's source files, or undef when they are not known
  my ($i) = @_;
  my $s = $st[$i];
  return undef if $s->{gen_mod};   # a generated module (a select, a rule's output): unknown, not empty
  return [map { @$_ } values %{$s->{mods}}] if $s->{mods};
  my $ds = static_of($i) // return undef;
  # Under `(include_subdirs ...)` its sources extend into subdirectories: unknown.
  return undef if grep { $_->{kind} eq "include_subdirs" && under($s->{dir}, $_->{dir}) } @dstanzas;
  my $mods = dfield($ds->{x}, "modules");
  my @names;
  if ($mods) {
    return undef if grep { ref $_ || !/^[A-Za-z_][A-Za-z0-9_']*$/ } @$mods;   # :standard, \, (:include ...)
    @names = map { lcfirst } @$mods;
  } else {   # every module of the directory no other stanza there claims
    my %claimed;
    for my $o (grep { $_ != $ds && $_->{dir} eq $s->{dir} } @dstanzas) {
      my $om = dfield($o->{x}, "modules") // next;
      $claimed{lcfirst $_} = 1 for grep { !ref } @$om;
    }
    opendir my $dh, $s->{dir} or return undef;
    @names = grep { !$claimed{$_} } map { /^([A-Za-z_][A-Za-z0-9_']*)\.mli?$/ ? lcfirst $1 : () } readdir $dh;
  }
  my @files;
  for my $n (@names) {
    my @f = grep { -f } map { in_dir($s->{dir}, "$n.$_") } "ml", "mli";
    @f or return undef;   # a module with no source here: generated, so unknown
    push @files, @f;
  }
  return \@files;
}
sub is_private { my $ds = static_of($_[0]); $ds && !grep { defined dfield($ds->{x}, $_) } qw(public_name public_names) }
sub flag_atoms {   # every atom of some flags fields, undef for a part not read as atoms
  map { !defined $_ ? () : ref $_ ? (@$_ && !ref $_->[0] && $_->[0] =~ /^:include$/ ? (undef) : flag_atoms(@$_)) : $_ } @_;
}
sub own_tokens {   # the names an owner's own code mentions, or undef when its code is not known
  my ($i) = @_;
  return $st[$i]{own_tokens} if exists $st[$i]{own_tokens};
  my $files = stanza_sources($i);
  my %tok;
  if ($files) {
    my @more = @$files;
    if (my $ds = static_of($i)) {
      my $pp = dfield($ds->{x}, "preprocess");
      if ($pp && @$pp) {
        # (pps NAME... -- flags): the project rewriters' own sources count.
        if (ref $pp->[0] && ($pp->[0][0] // "") eq "pps" && @$pp == 1) {
          for my $n (map { ref $_ ? () : $_ } @{$pp->[0]}[1 .. $#{$pp->[0]}]) {
            last if $n eq "--";
            next unless defined $lib_by_name{$n};
            # A rewriter that changed (or links what changed) rewrites this code
            # under no name it shows: unknown.
            $files = undef if $reached{$lib_by_name{$n}};
            push @more, @{stanza_sources($lib_by_name{$n}) // ($files = undef, [])};
          }
        } elsif (!(!ref $pp->[0] && $pp->[0] eq "no_preprocessing")) { $files = undef }   # per_module, action: not read
      }
      # `-open M` in its flags, or in an `env` stanza above it, lets its code use M's
      # names bare: unknown.
      # Flags with an `(:include ...)` or any other non-literal element are not
      # known either.
      $files = undef if grep { !defined || $_ eq "-open" || /^:(?!standard$)/ || /%\{/ } flag_atoms(map { dfield($ds->{x}, $_) } qw(flags ocamlc_flags ocamlopt_flags));
      $files = undef if grep { $_->{kind} eq "env" && under($ds->{dir}, $_->{dir}) && canon($_->{x}) =~ /"-open"/ } @dstanzas;
    }
    if ($files) {
      $tok{$_} = 1 for map { keys %{tokens_of($_)} } @more;
      my (%priv, %seen);
      my @q = @{$st[$i]{to}};
      while (defined(my $j = shift @q)) {
        # A changed library is not the owner's own code, even when private: its modules
        # name their own definitions.
        next if $seen{$j}++ || $st[$j]{kind} ne "library" || !is_private($j) || $mod_hit{$j};
        # The same uncertainty checks as the owner's own stanza: a helper whose code
        # is not all known leaves the owner's unknown too.
        defined own_tokens($j) or return $st[$i]{own_tokens} = undef;
        $priv{$j} = 1;
        push @q, @{$st[$j]{to}};
      }
      my %done;
      for (my $grew = 1; $grew;) {
        $grew = 0;
        for my $j (sort keys %priv) {
          for my $m (sort keys %{$st[$j]{mods} // {}}) {
            next if $done{"$j\0$m"} || !$tok{$m};
            $done{"$j\0$m"} = $grew = 1;
            $tok{$_} = 1 for map { keys %{tokens_of($_)} } @{$st[$j]{mods}{$m}};
          }
        }
      }
    }
  }
  return $st[$i]{own_tokens} = $files ? \%tok : undef;
}
# Which top-level definitions a changed `.ml`/`.mli` touched, from the git
# hunks and both sides of the file: each changed line belongs to the nearest
# definition starting at column 0 above it (`let`, `and`, `type`, `val`,
# `module`, `external`, `exception`, `class`), on the side the line is on. A
# line under `open`, `include`, a `let () =`/`let _ =` effect or a pattern
# binding, or above every definition, touches the module as a whole; so does
# any file without hunks (`--files`, a new file).
my %path_defs;   # path -> {NAME => 1}, or undef: the whole module
sub enclosing_def {
  my ($lines, $n) = @_;   # 1-based line number
  for (my $k = $n; $k >= 1; $k--) {
    my $l = $lines->[$k - 1] // next;
    next if $l =~ /^\s/ || $l eq "" || $l =~ /^(?:\(\*|\[\@|\)|;;|\||end\b|\})/;
    # Whitelisted forms only: a `let` value, a `val`/`external`, a module, class
    # or exception name. A `type` (its constructors and fields are what users
    # name) and an `and` (of a type or of a value) touch the whole module.
    return undef if $l =~ /^(?:type|and)\b/;
    return $1 if $l =~ /^let\s+(?:rec\s+)?(?:%\w+\s+)?([a-z_][\w']*)\b(?!\s*,)/ && $1 ne "_";
    return $1 if $l =~ /^let\s+(?:rec\s+)?\(\s*([^\s()]+)\s*\)/;   # an operator
    return $1 if $l =~ /^(?:val|external)\s+([a-z_][\w']*)/ || $l =~ /^val\s+\(\s*([^\s()]+)\s*\)/;
    return $1 if $l =~ /^(?:module|class|exception)\s+(?:rec\s+|type\s+)?([A-Za-z_][\w']*)/;
    return undef;   # open, include, let () =, a pattern: the whole module
  }
  return undef;
}
if (-f "$tmp/ml-diff") {
  my (%hunks, $cur, $gone);
  for (split /\n/, slurp("$tmp/ml-diff")) {
    if (/^diff /) { $cur = $gone = undef; next }
    if (/^--- (?:a\/(.*)|\/dev\/null)$/) { $gone = $1; next }
    if (/^\+\+\+ (?:b\/(.*)|\/dev\/null)$/) { $cur = $1 // $gone; next }
    if (/^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@/ && defined $cur) {
      push @{$hunks{$cur}}, [old => $1, $2 // 1], [new => $3, $4 // 1];
    }
  }
  for my $p (keys %hunks) {
    my %defs;
    my %side = map { my $f = "$tmp/$_/$p"; ($_ => [-f $f ? split(/\n/, slurp($f), -1) : ()]) } qw(old new);
    # A file new on this side (an `.mli` added beside its `.ml`, say) or gone from
    # it can hide or expose any member: the whole module.
    if (!-f "$tmp/old/$p" || !-f "$tmp/new/$p") { $path_defs{$p} = undef; next }
    for my $h (@{$hunks{$p}}) {
      my ($which, $start, $count) = @$h;
      next unless $count;
      @{$side{$which}} or do { %defs = ("\0whole" => 1); last };
      for my $n ($start .. $start + $count - 1) {
        my $name = enclosing_def($side{$which}, $n);
        if (!defined $name) { %defs = (); $defs{"\0whole"} = 1; last }
        # An operator or other non-identifier: tokens hold identifiers only, so the
        # whole module counts.
        if ($name !~ /^[A-Za-z_][\w']*$/) { %defs = ("\0whole" => 1); last }
        $defs{$name} = 1;
      }
      last if $defs{"\0whole"};
    }
    $path_defs{$p} = %defs && !$defs{"\0whole"} ? \%defs : undef;
  }
}
# The names a changed module answers to: its own, and those re-exporting it.
# A stanza reached from one hit as a whole is never link-only: no name says
# what changed there.
# So is one whose generated module or stub a reached generator writes: what the
# generator changed is code the stanza compiles, under no name it would show.
my %reached_whole = %whole_hit;
$reached_whole{$_} = 1 for grep { my $i = $_; grep { $reached{$_} } @{$st[$i]{gen_to} // []} } 0 .. $#st;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $i (0 .. $#st) {
    next if $reached_whole{$i} || !grep { $reached_whole{$_} } @{$st[$i]{to}};
    $reached_whole{$i} = $grew = 1;
  }
}
my $rank_links = 1;
my %changed_names;   # NAME -> {changed module it answers for}
my %mod_defs;        # changed module -> {definition => 1}, or undef: the whole module
if ($rank_links) {
  for my $m (map { keys %{$mod_hit{$_}} } grep { $st[$_]{kind} eq "library" } keys %mod_hit) {
    $changed_names{$m}{$m} = 1;
    my @p = keys %{$mod_paths{$m}};
    $mod_defs{$m} = (grep { !$path_defs{$_} } @p) ? undef : { map { %{$path_defs{$_}} } @p };
  }
  $rank_links = 0 unless %changed_names;
}
if ($rank_links) {
  my %as;   # M -> names under which some library module re-exports M
  for my $j (grep { $st[$_]{kind} eq "library" } 0 .. $#st) {
    for my $x (keys %{$st[$j]{mods} // {}}) {
      for my $f (@{$st[$j]{mods}{$x}}) {
        open my $fh, "<", $f or next;
        my $s = do { local $/; <$fh> // "" };
        # What a top-level `module NAME ... = PATH` or `include PATH` re-exports: the
        # module PATH names (its last component), read up to the first `struct`,
        # `sig`, `end`, `let`, `val`, blank or unindented line. A signature
        # constraint is read past (`module G : module type of M = M`, `include (M :
        # S)`). A functor application makes a new module whose code calls its
        # arguments: reached through other code, like any caller, so not read.
        while ($s =~ /^(module|include)\b/mg) {
          my ($kw, $at) = ($1, pos($s));
          my $head = substr($s, $at, 400);
          $head =~ s/(?:\b(?:struct|sig|end|let|val)\b|\n[ \t]*\n|\n(?=\S)).*\z//s;
          my ($as_name, $rhs);
          if ($kw eq "module") {
            next unless $head =~ /^\s*(?:rec\s+)?([A-Z][\w']*)/;
            $as_name = $1;
            ($rhs) = $head =~ /=([^=]*)\z/s or next;   # a declaration: no right-hand side
          } else { ($as_name, $rhs) = ($x, $head =~ s/:.*\z//sr) }
          $rhs =~ s/\(\*.*?\*\)//gs;   # comments
          my @names = $rhs =~ /^\s*\(?\s*([A-Z][\w']*)\s*(?:[:)]|\z)/s ? ($1) : ();
          # A right-hand side not read as one plain module name (a qualified path,
          # a functor application, anything else): uncertain, so every module name
          # in it counts as re-exported, and the wrapper's users are never
          # link-only on its account.
          @names = $rhs =~ /\b([A-Z][\w']*)/g if !@names && $rhs =~ /\S/;
          $as{$_}{$as_name} = 1 for grep { $_ ne $as_name } @names;
        }
      }
    }
  }
  my @q = keys %changed_names;
  while (defined(my $m = shift @q)) {
    for my $n (keys %{$as{$m} // {}}) {
      my $had = keys %{$changed_names{$n} // {}};
      $changed_names{$n}{$_} = 1 for keys %{$changed_names{$m}};
      push @q, $n if keys %{$changed_names{$n}} > $had;
    }
  }
}
sub link_only {   # reached owners -> true when none of them names a changed module's changed part
  return 0 unless $rank_links;
  for my $o (@_) {
    return 0 if $hit{$o} || $reached_whole{$o};
    my $tok = own_tokens($o) // return 0;
    for my $n (grep { $tok->{$_} } keys %changed_names) {
      for my $m (keys %{$changed_names{$n}}) {
        my $defs = $mod_defs{$m};
        return 0 if !$defs || grep { $tok->{$_} } keys %$defs;
      }
    }
  }
  return 1;
}

# What an alias's own action reads, dune does not dump. A golden alias's
# action is a diff of two of its own targets (`NAME.expected` and the output
# beside it), whose rules are read above; any other alias's action is opaque
# -- a `(test)` stanza with no golden runs its executable under the stanza's
# `(deps)`, a hand-written rule runs whatever it names under its own `(deps)`
# -- and is printed for a changed path other than an owned module source in
# its directory, and for a changed path anywhere that is no build input (a
# script, a config, a golden: what such `(deps)` name, like the sweep
# harness's `tools/sweep.sh`). One with no owner and no targets of its own
# is unattributed and printed whenever its directory is touched at all, a
# reached stanza included, since not even its executable is known.
my %touched = map { $st[$_]{dir} => 1 } keys %reached;
$touched{(dir_of($_))[0]} = 1 for @changed;
my (%out, %unattributed, %opaque_hit, %total, %listed, %linked, %aliased, %blind_why);
my %hidden = map { $_->{name} => 1 } grep { $_->{hidden} } @al;
my %read_any = map { %{$read_by{$_}} } keys %read_by;
# The tier of each listed alias: "main"; "link", reached only by linking a
# library a changed module belongs to, never naming that module (see
# link_only); "guess", listed only because its action is unknown.
my %tier;
for my $a (@al) {
  my %owners = (%{$a->{owners}}, map { %{$runs{$_} // {}} } @{$a->{own}});
  $aliased{$_} = 1 for keys %owners;
  $total{$a->{dir}}++ unless $a->{hidden};
  my $unknown = !%owners && !@{$a->{own}};
  my $opaque = !$a->{described};
  my $unknown_touched = $unknown && $touched{$a->{dir}};
  $unattributed{$a->{name}} = 1 if $unknown_touched && !$a->{hidden};
  my $opaque_listed = $opaque && ($any_data || $data_dirs{$a->{dir}});
  if ($opaque_listed && !$a->{hidden}) { $opaque_hit{$a->{dir}}++; $blind_why{$a->{blind_why} // "dune does not dump its action"}++ }
  my @by_owner = grep { $reached{$_} } nsort(keys %owners);
  # Uncertainty ranks first: an alias whose action is unknown and might read a
  # changed path is never link-only, whatever its owners name.
  my $blind = $unknown_touched || $opaque_listed;
  if ($dune_dirs{$a->{dir}} || $read_any{$a->{name}} || $direct{$a->{name}}) { $tier{$a->{name}} = "main" }
  elsif ($blind) { $tier{$a->{name}} = @by_owner ? "main" : "guess" }
  elsif (@by_owner) { $tier{$a->{name}} = link_only(@by_owner) ? "link" : "main" }
}
# An alias whose own rules need a listed alias (an ambient gate such as
# `runtest-env_spelling_gate`) is listed too, in that alias's tier, to a
# fixpoint: if the needed alias breaks, building the needing one fails. That
# holds for one listed only blind as well (gh-ocannl-1431 review), so a blind
# alias carries its dependents in its own tier.
my %rank = (main => 3, guess => 2, link => 1);
my %tier_of_rank = reverse %rank;
my %by_alias;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $a (@al) {
    my $now = $rank{$tier{$a->{name}} // ""} // 0;
    next if $now == 3;
    my $best = 0;
    for my $t (@{$a->{own}}) {
      for my $n (keys %{$adeps{$t} // {}}) { my $r = $rank{$tier{$n} // ""} // 0; $best = $r if $r > $best }
    }
    next unless $best > $now;
    $tier{$a->{name}} = $tier_of_rank{$best};
    $by_alias{$a->{name}} = 1;
    $grew = 1;
  }
}
for my $n (grep { !$hidden{$_} } keys %tier) {
  my ($d) = $n =~ m{^\@(?:(.*)/)?[^/]+$};
  if ($tier{$n} eq "link") { $linked{$d // "."}++; next }
  $out{$n} = 1;
  $listed{$d // "."}++;
}
my @link_only = sort grep { $tier{$_} eq "link" && !$hidden{$_} } keys %tier;

print "$_\n" for sort keys %out;
print "$_\n" for $referenced_only ? () : @link_only;
my $n_by_alias = grep { $out{$_} } keys %by_alias;
printf STDERR "affected: %d alias(es) listed because their rules need a listed alias\n", $n_by_alias if $n_by_alias;
printf STDERR "affected: %s: %d of its %d per-test aliases%s\n", $_, $listed{$_} // 0, $total{$_},
  ($linked{$_} ? " (and $linked{$_} link-only)" : "") for sort keys %{{ %listed, %linked }};
my %bare;
$bare{$st[$_]{dir}}++ for grep { !$aliased{$_} && $st[$_]{kind} ne "library" } keys %reached;
printf STDERR "affected: %s: %d reached executable stanza(s) no per-test alias runs\n", $_, $bare{$_} for sort keys %bare;
# Only worth saying when a changed module has no stanza here: it may belong to one of these.
if (@unmapped) {
  print STDERR "affected: optional library $_ is not in this build; edges to it skipped\n" for sort keys %skipped_optional;
  print STDERR "affected: library $_ is not in this build, nor is the stanza requiring it; edge skipped\n" for sort keys %skipped_off_build;
}
print STDERR "affected: listed as unattributed: $_ (no stanza of its directory has that name, no targets of its own; its directory is touched)\n" for sort keys %unattributed;
printf STDERR "affected: %s: %d blind per-test alias(es) (actions not described here), listed for the changed paths there\n", $_, $opaque_hit{$_} for sort keys %opaque_hit;
printf STDERR "affected: %d of them blind because %s\n", $blind_why{$_}, $_ for sort keys %blind_why;
print STDERR "affected: unread: $_ (no stanza owns it and no rule dune shows reads it; only blind aliases are listed for it)\n" for @unread;
print STDERR "affected: UNMAPPED: $_ (no stanza of this build owns this module)\n" for @unmapped;
if (@link_only) {
  my $what = join("; ", map { $_ . ($mod_defs{$_} ? " (" . join(", ", sort keys %{$mod_defs{$_}}) . ")" : "") }
                        sort keys %mod_defs);
  printf STDERR "affected: %d LINK-ONLY alias(es), %s: each links the library of a changed module, %s, but no code of"
    . " its own names the changed part, so it is reached only through other library code. A change in what that"
    . " code does for its callers still breaks them (gh-ocannl-1235: Register_tile); run them before pushing such a"
    . " change. %s\n",
    scalar(@link_only), $referenced_only ? "NOT printed (--referenced-only)" : "printed LAST, after the others",
    $what, $referenced_only ? "They are:" : "--referenced-only leaves them off. They are:";
  print STDERR "affected: link-only: $_\n" for @link_only;
}
printf STDERR "affected: %d alias(es) from %d changed path(s)%s%s\n", scalar(keys %out) + ($referenced_only ? 0 : @link_only),
  scalar(@changed), (@link_only ? sprintf(", %d of them link-only%s", scalar(@link_only), $referenced_only ? " and left off" : "") : ""),
  @unmapped ? "; INCOMPLETE, see UNMAPPED above" : "";
exit(@unmapped ? 3 : 0);
PERL

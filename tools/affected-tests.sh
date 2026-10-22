#!/usr/bin/env bash
# Which per-test aliases can a change reach? Read from dune's own description
# of the build, never from a hand-kept map (gh-ocannl-1235). The entry point is
# `tools/test-run.sh affected`, which refuses while a run holds this worktree
# and then runs this script. It prints aliases and runs none of them.
#
# Usage:
#   tools/test-run.sh affected --since BASE [-- PATHSPEC...]
#   tools/test-run.sh affected REV | REV..REV | REV...REV [-- PATHSPEC...]
#   tools/test-run.sh affected --files PATH...
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
# stdout: one alias per line, `@DIR/FAMILY-NAME[-VARIANT]`, sorted, ready for
# `tools/test-run.sh run build ...` or `tools/test-run.sh fresh ...`. stderr:
# what each changed path reached, a per-directory count, and what this script
# could not place. The graph is always the CURRENT worktree's, whatever
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
#
# Boundary, as an allowlist. Whatever it does not cover is reported on stderr,
# never silently dropped:
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
#     generator writing a header it includes). A changed `dune`,
#     `dune-project` or `dune-workspace` file reaches every stanza and every
#     per-test alias of the directories beneath it (an `env` stanza applies
#     there too), and every alias elsewhere whose own targets' rules read a
#     file or glob of those directories (a file their rules make). Reach then
#     spreads to each stanza requiring a reached library, to a fixpoint.
#   - An alias is printed when its directory's `dune` file changed, when an
#     owner is reached, or when a rule of its own targets reads a changed path
#     (a file, or a glob over source files): the repository scans are found
#     this way, and so is a golden, whose copy into the build is one of the
#     own targets of the alias that diffs it. A changed path reaching nothing
#     else is listed as unread.
#   - An alias's own action is not in `dune rules` output. A golden alias's
#     action diffs two of its own targets, whose rules are read; any other
#     alias's action is opaque (a `tests` stanza with no golden runs its
#     executable under the stanza's `(deps)`), so such an alias is also
#     printed whenever a changed path other than an owned module source lies
#     in its directory or is copied there, and whenever any changed path is no
#     build input (a script, a config, a golden: what an alias rule's own
#     `(deps)` name, as `runtest-sweep_harness` names `tools/sweep.sh`).
#     An alias whose own targets' rules need a printed alias is printed too,
#     to a fixpoint: a golden pair's output rule carries the ambient
#     `runtest-env_spelling_gate` for this reason (gh-ocannl-1378). Not seen:
#     what an opaque action reads outside its directory otherwise, and an
#     alias an alias rule's own `(deps)` name (a run-and-diff rule's gate),
#     which `dune rules` does not dump in any form it accepts (`@DIR/NAME`,
#     `@@DIR/NAME`).
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
trap 'rm -rf "${tmp:?}"' EXIT

usage="usage: affected --since BASE | REV | REV..REV | REV...REV [-- PATHSPEC...] | affected --files PATH..."
mode=git
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
# Dune 3.24 prints these listings on stderr, and exits 0 even after an error,
# so both streams are read together, and every line must have one of the forms
# the reader below accepts.
for what in aliases targets; do
  dune show "$what" --root . "${dirs[@]}" >"$tmp/$what" 2>&1 ||
    die "dune show $what failed:
$(cat "$tmp/$what")"
done

perl - "$tmp" <<'PERL'
use strict; use warnings;
my ($tmp) = @ARGV;
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
sub listing {
  my ($what) = @_;
  my (%by, $cur);
  $cur = $dir_list[0] if @dir_list == 1;
  for my $l (split /\n/, slurp("$tmp/$what")) {
    next if $l eq "";
    # A directory dune does not build (a data-only one, say) defines nothing.
    if ($l =~ /^Error: Directory (.*) does not exist\.$/ && $is_dir{$1}) { $cur = undef; next }
    if ($l =~ /^(.*):$/ && $is_dir{$1}) { $cur = $1; next }
    $l =~ /^[A-Za-z0-9_.+-]+$/ or fail("unexpected line from dune show $what: $l");
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

# Per-test aliases, their name owners and their own targets.
my @al;   # { name, dir, owners => {index => 1}, own => [target...] }
my $artifact = qr/\.(?:exe|bc|bc-for-jsoo|js|cmi|cmo|cmx|cmt|cmti|cma|cmxa|cmxs|a|o|so|dll|pp\.ml|pp\.mli|ml-gen|d)$/;
for my $d (sort keys %$aliases) {
  my @in = grep { $st[$_]{dir} eq $d } 0 .. $#st;
  for my $a (@{$aliases->{$d}}) {
    my ($rest) = $a =~ /^(?:runtest|train|slow)-(.+)$/ or next;
    my ($unit, @variant) = split /-/, $rest;
    my @prefix = @variant ? ("$rest.", join(".", $unit, @variant) . ".") : ("$rest.");
    my @mine = grep { my $t = $_; $t !~ $artifact && grep { index($t, $_) == 0 } @prefix } @{$targets->{$d} // []};
    # `UNIT_*` targets count too (a rule-made `X_actual.ml` beside `runtest-X-ppx`,
    # test/ppx), though only a golden named as above makes the action a known diff.
    my @near = grep { $_ !~ $artifact && index($_, "${unit}_") == 0 } @{$targets->{$d} // []};
    my %own = map { in_dir($d, $_) => 1 } @mine, @near;
    push @al, { name => "\@" . in_dir($d, $a), dir => $d, golden => scalar(grep { /\.expected$/ } @mine),
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
      exec "dune", "rules", "--root", ".", "-o", "$tmp/rules-out", @batch; exit 127;
    }
    waitpid($pid, 0) == $pid && $? == 0 or fail("dune rules failed:\n" . slurp("$tmp/rules-log"));
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
my @queue = @own_targets;
while (@queue) {
  $queried{$_} = 1 for @queue;
  my %next;
  for my $rule (dune_rules(@queue)) {
    my $tg = field($rule, "targets");
    my @files = map { atom($_, "targets") } items(field($tg // [], "files"));
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
  for my $o (@{$generated{$g}}) { push @{$st[$o]{to}}, keys %{$runs{$g}} }
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
  for my $w (obj_owners($o)) { push @{$st[$w]{to}}, keys %{$runs{$o}} }
}

# Changed paths -> stanzas, and the aliases whose own targets read them.
my (%hit, %dune_dirs, %data_dirs, %read_by, @unmapped, @unread);
my $any_data = 0;
sub label { my $s = $st[$_[0]]; "$s->{kind} @{$s->{names}} ($s->{dir})" }
sub reads { my ($t, $p, $d, $b) = @_; $reads{$t}{$p} || grep { $_->[0] eq $d && $_->[1]->($b) } @{$globs{$t}} }
sub under { my ($x, $root) = @_; $root eq "." || $x eq $root || index($x, "$root/") == 0 }
for my $p (@changed) {
  my ($d, $b) = dir_of($p);
  for my $a (@al) {
    for my $t (@{$a->{own}}) { if (reads($t, $p, $d, $b)) { $read_by{$p}{$a->{name}} = 1; last } }
  }
  my $n_read = keys %{$read_by{$p} // {}};
  my (@ix, @parts, $unmapped);
  my @gen = grep { reads($_, $p, $d, $b) } sort keys %generated;
  if (@gen) {
    my %seen_g;
    my @g = grep { !$seen_g{$_}++ } map { @{$generated{$_}} } @gen;
    $hit{$_} = 1 for @g;
    push @parts, "the rule generating " . join(", ", @gen) . " reads it: " . join(", ", map { label($_) } @g);
  }
  for my $o (grep { /\.o$/ && reads($_, $p, $d, $b) } @stub_targets) {
    my @libs = sort keys %{$obj_libs{$o} // {}};
    my @o = obj_owners($o);
    $hit{$_} = 1 for @o;
    push @parts, "compiled into $o, " . (@libs ? "bundled by " . join(", ", map { label($_) } @libs)
      : "which no stub archive bundles, so every executable and test stanza of its directory (" . scalar(@o) . ")");
  }
  if ($b eq "dune" || $b eq "dune-project" || $b eq "dune-workspace") {
    # A `dune` file's settings (`env`, `(dirs)`, ...) apply beneath it as well,
    # as `dune-project`'s language and extensions do: every directory under it.
    $dune_dirs{$_} = 1 for grep { under($_, $d) } $d, keys %$aliases;
    @ix = grep { under($st[$_]{dir}, $d) } 0 .. $#st;
    # Its rules make the files there: a rule elsewhere reading one of them (a
    # fixture's generated sources) reads the change.
    my $n_made = 0;
    for my $a (grep { !under($_->{dir}, $d) } @al) {
      for my $t (@{$a->{own}}) {
        next unless grep({ under((dir_of($_))[0], $d) } keys %{$reads{$t}}) || grep { under($_->[0], $d) } @{$globs{$t}};
        $n_made++ unless $read_by{$p}{$a->{name}}++;
        last;
      }
    }
    push @parts, "every per-test alias and all " . scalar(@ix) . " stanza(s) " . ($d eq "." ? "of the project" : "under $d")
      . ($n_made ? "; files its rules make are read by the rules of $n_made alias(es) elsewhere" : "");
  } elsif ($b =~ /^([^.]+)(?:\.[^.]+)*\.mli?$/) {
    my $mod = ucfirst $1;
    my $how = "module $mod of";
    @ix = @{$by_modkey{"$d\0$mod"} // []};
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

# Reach, to a fixpoint: a stanza requiring a reached library is reached.
my %reached = %hit;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $i (0 .. $#st) {
    next if $reached{$i};
    if (grep { $reached{$_} } @{$st[$i]{to}}) { $reached{$i} = 1; $grew = 1 }
  }
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
my (%out, %unattributed, %opaque_hit, %total, %listed, %aliased);
my %read_any = map { %{$read_by{$_}} } keys %read_by;
for my $a (@al) {
  my %owners = (%{$a->{owners}}, map { %{$runs{$_} // {}} } @{$a->{own}});
  $aliased{$_} = 1 for keys %owners;
  $total{$a->{dir}}++;
  my $unknown = !%owners && !@{$a->{own}};
  my $opaque = !$a->{golden};
  $unattributed{$a->{name}} = 1 if $unknown && $touched{$a->{dir}};
  my $opaque_listed = $opaque && ($any_data || $data_dirs{$a->{dir}});
  $opaque_hit{$a->{dir}}++ if $opaque_listed;
  if ($dune_dirs{$a->{dir}} || $read_any{$a->{name}} || grep({ $reached{$_} } keys %owners)
      || $unattributed{$a->{name}} || $opaque_listed) {
    $out{$a->{name}} = 1;
    $listed{$a->{dir}}++;
  }
}
# An alias whose own targets' rules need a listed alias (an ambient gate such
# as `runtest-env_spelling_gate`) is listed too, to a fixpoint.
my $by_alias = 0;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $a (grep { !$out{$_->{name}} } @al) {
    next unless grep { my $t = $_; grep { $out{$_} } keys %{$adeps{$t} // {}} } @{$a->{own}};
    $out{$a->{name}} = 1;
    $listed{$a->{dir}}++;
    $by_alias++;
    $grew = 1;
  }
}

print "$_\n" for sort keys %out;
printf STDERR "affected: %d alias(es) listed because their rules need a listed alias\n", $by_alias if $by_alias;
printf STDERR "affected: %s: %d of its %d per-test aliases\n", $_, $listed{$_}, $total{$_} for sort keys %listed;
my %bare;
$bare{$st[$_]{dir}}++ for grep { !$aliased{$_} && $st[$_]{kind} ne "library" } keys %reached;
printf STDERR "affected: %s: %d reached executable stanza(s) no per-test alias runs\n", $_, $bare{$_} for sort keys %bare;
# Only worth saying when a changed module has no stanza here: it may belong to one of these.
if (@unmapped) {
  print STDERR "affected: optional library $_ is not in this build; edges to it skipped\n" for sort keys %skipped_optional;
  print STDERR "affected: library $_ is not in this build, nor is the stanza requiring it; edge skipped\n" for sort keys %skipped_off_build;
}
print STDERR "affected: listed as unattributed: $_ (no stanza of its directory has that name, no targets of its own; its directory is touched)\n" for sort keys %unattributed;
printf STDERR "affected: %s: %d per-test alias(es) whose actions dune does not dump, listed for the changed paths there\n", $_, $opaque_hit{$_} for sort keys %opaque_hit;
print STDERR "affected: unread: $_ (no stanza owns it and no rule dune shows reads it; only aliases whose actions dune does not dump are listed for it)\n" for @unread;
print STDERR "affected: UNMAPPED: $_ (no stanza of this build owns this module)\n" for @unmapped;
printf STDERR "affected: %d alias(es) from %d changed path(s)%s\n", scalar(keys %out), scalar(@changed),
  @unmapped ? "; INCOMPLETE, see UNMAPPED above" : "";
exit(@unmapped ? 3 : 0);
PERL

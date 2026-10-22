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
# not place. The graph is always the CURRENT worktree's, whatever
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
#     generator writing a header it includes). A changed `dune` file whose
#     two sides the revision forms give (not `--files`) reaches only the
#     stanzas whose text changed, comments and layout aside (gh-ocannl-1431): a
#     rule the aliases it declares and the files it makes, through what reads
#     them; a test the aliases it declares and its stanza; a library or
#     executables its stanza; an `(alias)` the per-test aliases it names, and
#     a helper alias every per-test alias needing it. Only forms modeled in
#     full are scoped: a changed stanza of any other kind (`env`, `dirs`,
#     `copy_files`, ...), a file holding a `(subdir ...)`, a block string, a
#     new file, `--files`, and a changed `dune-project` or `dune-workspace`
#     file reach every stanza and every per-test alias of the directories
#     beneath it (an `env` stanza applies there too), and every alias
#     elsewhere whose own targets' rules read a file or glob of those
#     directories (a file their rules make). Reach then spreads to each stanza
#     requiring a reached library, to a fixpoint.
#   - An alias is printed when a changed dune stanza reaches it, when an
#     owner is reached, or when a rule of its own targets reads a changed path
#     (a file, or a glob over source files): the repository scans are found
#     this way, and so is a golden, whose copy into the build is one of the
#     own targets of the alias that diffs it. A changed path reaching nothing
#     else is listed as unread.
#   - An alias's own action is not in `dune rules` output, so the dune files
#     are read as text too: the stanzas declaring each alias (a rule's `(alias
#     ...)`, an `(alias (name ...))`, the `runtest-NAME` of a `(test)`/`(tests)`
#     name or of a library's `(inline_tests)`), and what they need: `(deps)`
#     files, globs, `source_tree`s and aliases, the `%{dep:}`/`%{exe:}`/
#     `%{read:}`/`%{lib:}`/`%{bin:}` forms, and what dune infers from `diff`,
#     `cmp`, `cat`, `copy`, `with-stdin-from` and a `run` of a path, outside
#     `no-infer` and relative to a `chdir`. A generated file among them is
#     queried like an own target. An alias whose stanzas are read in full is
#     printed only for what they need (gh-ocannl-1431: a run-and-diff rule
#     owning a golden was taken for a golden pair, its `(deps)` never read).
#     Any other alias -- no stanza declares it, or one uses a form outside
#     that list, such as `(package ...)` -- keeps the blind treatment: a golden
#     alias's action diffs two of its own targets, whose rules are read; any
#     other is opaque, printed whenever a changed path other than an owned
#     module source lies in its directory or is copied there, and whenever any
#     changed path is no build input (a script, a config, a golden).
#     An alias whose rules need a printed alias is printed too, to a fixpoint:
#     a golden pair's output rule carries the ambient
#     `runtest-env_spelling_gate` for this reason (gh-ocannl-1378) -- but not
#     for one printed only blind: the needing alias reads no file of it.
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

usage="usage: affected [--referenced-only] --since BASE | REV | REV..REV | REV...REV [-- PATHSPEC...] | affected [--referenced-only] --files PATH..."
# Link-only aliases (see the Perl below) are printed last, and named on stderr;
# --referenced-only leaves them off stdout.
referenced_only=0
if [ "${1:-}" = --referenced-only ]; then referenced_only=1; shift; fi
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
  # Both sides of each changed `dune`, `.ml` and `.mli` file: a dune edit then
  # reaches only the stanzas whose text changed, and a module edit is ranked by
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
    case $f in dune | */dune | *.ml | *.mli) ;; *) continue ;; esac
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
# Dune 3.24 prints these listings on stderr, and exits 0 even after an error,
# so both streams are read together, and every line must have one of the forms
# the reader below accepts.
for what in aliases targets; do
  dune show "$what" --root . "${dirs[@]}" >"$tmp/$what" 2>&1 ||
    die "dune show $what failed:
$(cat "$tmp/$what")"
done

perl - "$tmp" "$referenced_only" <<'PERL'
use strict; use warnings;
my ($tmp, $referenced_only) = @ARGV;
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
        push @{$st[$i]{mods}{$name}}, $rel if -e $rel;   # its sources, for the link-only ranking
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
# The dune files themselves, read as text: what an alias's own rule needs, which
# `dune rules` never dumps (it accepts no alias), and which stanzas a dune-file
# edit changed. A reader for dune's syntax, as its lexer has it
# (src/dune_sexp/lexer.mll): `;` end-of-line comments only (`#|` and `#;` are
# atom text), quoted strings, block strings, atoms. It dies on text it cannot
# read; the caller decides whether that stops the run. A block string comes
# back OPAQUE (\x01-prefixed: its escapes and its leading space are not
# decoded), which every reader of a stanza's needs takes as unknown.
# With $quoted, a quoted or block string comes back NUL-prefixed instead: what
# the stanza comparison reads, since `"%{deps}"` is one argument and `%{deps}`
# many.
sub parse_dune {
  my ($s, $what, $quoted) = @_;
  my $q = $quoted ? "\0" : "";
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
      $put->(($quoted ? "\0" : "\x01") . $str); next;
    }
    if ($s =~ /\G"((?:[^"\\]|\\.)*)"/gcs) {
      my $a = $1 =~ s/\\\n[ \t]*//gr;
      $a =~ s/\\(x[0-9A-Fa-f]{2}|[0-9]{3}|.)/length $1 > 1 ? ($1 =~ m{^x} ? chr hex substr($1, 1) : chr $1)
                                            : $1 eq "n" ? "\n" : $1 eq "t" ? "\t" : $1 eq "r" ? "\r" : $1/gse;
      $put->($q . $a); next;
    }
    if ($s =~ /\G([^\s()";]+)/gc) { $put->($1); next }
    die "unreadable text in $what at byte " . pos($s) . "\n";
  }
  @stack == 1 or die "unbalanced parentheses in $what\n";
  return $stack[0];
}
sub has_opaque { my ($x) = @_; ref $x ? scalar(grep { has_opaque($_) } @$x) : $x =~ /^\x01/ }
sub canon {   # a quoted string (NUL-prefixed) and an atom of the same text stay apart
  my ($x) = @_;
  return "(" . join(" ", map { canon($_) } @$x) . ")" if ref $x;
  my $quoted = $x =~ s/^\0//;
  return ($quoted ? "q" : "a") . "\"" . ($x =~ s/(["\\])/\\$1/gr =~ s/\n/\\n/gr) . "\"";
}
sub dfield {   # the arguments of a stanza's first (NAME ...) field
  my ($x, $name) = @_;
  for my $f (@$x[1 .. $#$x]) { return [@$f[1 .. $#$f]] if ref $f && @$f && !ref $f->[0] && $f->[0] eq $name }
  return undef;
}
# Stanzas of DIR, `(subdir ...)` flattened into the directory it names.
# QITEMS is the same text read with quotation kept (parse_dune's $quoted): the
# canonical text compared across a dune-file edit.
sub dune_stanzas {
  my ($dir, $items, $qitems) = @_;
  my @out;
  for my $k (0 .. $#$items) {
    my ($x, $qx) = ($items->[$k], $qitems->[$k]);
    next unless ref $x && @$x && !ref $x->[0];
    if ($x->[0] eq "subdir" && @$x >= 2 && !ref $x->[1]) {
      my $sub = rpath($dir, $x->[1]);
      if (defined $sub) { push @out, dune_stanzas($sub, [@$x[2 .. $#$x]], [@$qx[2 .. $#$qx]]); next }
    }
    push @out, { dir => $dir, kind => $x->[0], x => $x, text => canon($qx) };
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
  push @dstanzas, dune_stanzas((dir_of($f))[0], $items, parse_dune($dune_text{$f}, $f, 1));
}
# Public executables, for `%{bin:NAME}`.
my %public_exe;
for my $s (grep { $_->{kind} =~ /^(?:executables?|tests?)$/ } @dstanzas) {
  my @names = map { ref $_ ? () : $_ } @{dfield($s->{x}, "names") // dfield($s->{x}, "name") // []};
  my @pub = map { ref $_ ? () : $_ } @{dfield($s->{x}, "public_names") // dfield($s->{x}, "public_name") // []};
  for my $k (0 .. $#pub) {
    my $n = $names[$k] // next;
    $public_exe{$pub[$k]} = $exe{"$s->{dir}\0$n"} if $pub[$k] ne "-" && $exe{"$s->{dir}\0$n"};
  }
}
# What a stanza attached to an alias needs, declared: files, globs, executables,
# other aliases. Anything this reader does not know marks the stanza unknown,
# and its alias then keeps the treatment of an undumped action (below).
my %targets_of = map { my $d = $_; ($d => { map { $_ => 1 } @{$targets->{$d}} }) } keys %$targets;
sub decl_new { return { files => {}, globs => [], runs => {}, adeps => {}, adeps_rec => {}, via => [], unknown => [] } }
sub decl_file {
  my ($acc, $p) = @_;
  return unless defined $p;   # outside the repository: an external file
  # The runner dune builds for a library's inline tests links that library.
  if ($p =~ m{^(?:(.*)/)?\.([^/]+)\.inline-tests/[^/]+$}) {
    my ($ld, $ln) = ($1 // ".", $2);
    $acc->{runs}{$_} = 1 for grep { $st[$_]{kind} eq "library" && $st[$_]{dir} eq $ld && $st[$_]{units}{$ln} } 0 .. $#st;
    return;
  }
  if ($p =~ m{^(?:(.*)/)?([^/]+)\.(?:exe|bc)$}) { $acc->{runs}{$_} = 1 for @{$exe{($1 // ".") . "\0$2"} // []}; return }
  if (my @o = compiled_owners("$ctx/$p")) { $acc->{runs}{$_} = 1 for @o; return }
  $acc->{files}{$p} = 1;
}
sub decl_root {   # a %{project_root}/%{workspace_root} prefix, as a path from DIR
  my ($dir, $s) = @_;
  $s =~ s{^%\{(?:project_root|workspace_root)\}(?=/|$)}{$dir eq "." ? "." : join "/", map { ".." } split m{/}, $dir}e;
  return $s;
}
sub decl_vars {   # the %{...} forms in a string, relative to the stanza's DIR
  my ($acc, $dir, $s) = @_;
  while ($s =~ /%\{([^}]*)\}/g) {
    my $v = $1;
    my ($k, $arg) = $v =~ /^([^:]+):(.*)$/ or next;   # no argument: %{target}, %{deps}, %{test}, ...
    if ($k =~ /^(?:dep|exe|read|read-lines|read-strings)$/) { decl_file($acc, rpath($dir, decl_root($dir, $arg))) }
    elsif ($k =~ /^lib(?:-private)?$/) {
      my ($lib) = split /:/, $arg;
      if (defined $lib_by_name{$lib}) { $acc->{runs}{$lib_by_name{$lib}} = 1 } else { push @{$acc->{unknown}}, "%{$v}" }
    } elsif ($k eq "bin") {
      if ($public_exe{$arg}) { $acc->{runs}{$_} = 1 for @{$public_exe{$arg}} } else { push @{$acc->{unknown}}, "%{$v}" }
    } elsif ($k !~ /^(?:env|ocaml-config|lib-available|bin-available|version|coq)$/) { push @{$acc->{unknown}}, "%{$v}" }
  }
}
sub decl_deps {
  my ($acc, $dir, $items) = @_;
  for my $d (@$items) {
    if (!ref $d) {
      next if $d =~ /^:/;   # a (:name ...) binding's name
      my $r = decl_root($dir, $d);
      if ($r =~ /^%\{[^}]*\}$/) { decl_vars($acc, $dir, $r) }
      elsif ($r =~ /%\{/) { push @{$acc->{unknown}}, $d }
      else { decl_file($acc, rpath($dir, $r)) }
      next;
    }
    my ($h, @args) = @$d;
    if (!defined $h || ref $h) { push @{$acc->{unknown}}, canon($d); next }
    if ($h =~ /^:/) { decl_deps($acc, $dir, \@args) }
    elsif ($h eq "file") { decl_deps($acc, $dir, \@args) }
    elsif ($h =~ /^(glob_files(?:_rec)?|source_tree)$/ && @args == 1 && !ref $args[0]) {
      my $g = decl_root($dir, $args[0]);
      if ($g =~ /%\{/) { push @{$acc->{unknown}}, canon($d); next }
      my $rec = $h ne "glob_files";
      if ($h eq "source_tree") { my $t = rpath($dir, $g); push @{$acc->{globs}}, [$t, sub { 1 }, 1] if defined $t; next }
      my ($gd, $pat) = $g =~ m{^(.*)/([^/]*)$} ? ($1, $2) : (".", $g);
      my $t = rpath($dir, $gd);
      push @{$acc->{globs}}, [$t, predicate($pat), $rec] if defined $t;
    } elsif ($h =~ /^alias(?:_rec)?$/ && @args == 1 && !ref $args[0]) {
      my $is_rec = $h eq "alias_rec";
      my ($ad, $an) = dir_of($args[0]);
      my $t = rpath($dir, $ad) // next;
      $acc->{adeps}{"\@" . in_dir($t, $an)} = 1;
      $acc->{adeps_rec}{"$t\0$an"} = 1 if $is_rec;
    } elsif ($h !~ /^(?:env_var|universe|sandbox)$/) { push @{$acc->{unknown}}, canon($d) }
  }
}
my %action_walk = map { $_ => 1 } qw(progn concurrent ignore-stdout ignore-stderr ignore-outputs pipe-stdout pipe-stderr
  pipe-outputs);
sub decl_action {
  my ($acc, $dir, $cwd, $x, $infer) = @_;
  if (!ref $x) { decl_vars($acc, $dir, $x); return }
  my ($h, @args) = @$x;
  if (!defined $h || ref $h) { push @{$acc->{unknown}}, canon($x); return }
  decl_vars($acc, $dir, $_) for grep { !ref } @args;
  my $here = sub { defined $cwd ? rpath($cwd, decl_root($cwd, $_[0])) : undef };
  my $plain = sub { !ref $_[0] && $_[0] !~ /%\{/ };
  if ($h eq "no-infer") { decl_action($acc, $dir, $cwd, $_, 0) for @args }
  elsif ($h eq "chdir" && @args == 2 && !ref $args[0]) {
    my $to = decl_root($dir, $args[0]);
    # A chdir inside a chdir is not modeled: unknown, so its alias stays blind.
    if (!defined $cwd || $cwd ne $dir) { push @{$acc->{unknown}}, "a nested chdir"; return }
    decl_action($acc, $dir, $to =~ /%\{/ ? undef : rpath($dir, $to), $args[1], $infer);
  } elsif ($action_walk{$h}) { decl_action($acc, $dir, $cwd, $_, $infer) for @args }
  elsif ($h =~ /^(?:with-(?:stdout|stderr|outputs)-to|write-file|with-accepted-exit-codes|setenv)$/) {
    decl_action($acc, $dir, $cwd, $_, $infer) for grep { ref } @args;
  } elsif ($h eq "with-stdin-from") {
    decl_file($acc, $here->($args[0])) if $infer && $plain->($args[0]);
    decl_action($acc, $dir, $cwd, $_, $infer) for grep { ref } @args;
  } elsif ($h eq "run") {
    decl_file($acc, $here->($args[0])) if $infer && @args && $plain->($args[0]) && $args[0] =~ m{/};
  } elsif ($h =~ /^(?:diff|cmp|cat)$/) {
    $infer && $plain->($_) and decl_file($acc, $here->($_)) for @args;
  } elsif ($h eq "diff?") {   # the second file is one the action itself may have written
    decl_file($acc, $here->($args[0])) if $infer && @args && $plain->($args[0]);
  } elsif ($h =~ /^copy#?$/) {
    decl_file($acc, $here->($args[0])) if $infer && @args && $plain->($args[0]);
  } elsif ($h !~ /^(?:echo|system|bash)$/) { push @{$acc->{unknown}}, "($h ...)" }
}
my %rule_fields = map { $_ => 1 } qw(targets target deps action mode fallback locks alias aliases package enabled_if);
sub decl_stanza {   # what an alias-attached stanza needs
  my ($s) = @_;
  my $acc = decl_new();
  my $x = $s->{x};
  push @{$acc->{unknown}}, "a block string" if has_opaque($x);
  if ($s->{kind} eq "library") {   # runtest-NAME of its (inline_tests (deps ...)): the runner links the library
    my $it = dfield($x, "inline_tests") // [];
    decl_deps($acc, $s->{dir}, dfield(["inline_tests", @$it], "deps") // []);
    return $acc;
  }
  decl_deps($acc, $s->{dir}, dfield($x, "deps") // []);
  my $act = dfield($x, "action");
  decl_action($acc, $s->{dir}, $s->{dir}, $_, 1) for @{$act // []};
  if ($s->{kind} eq "rule") {
    for my $f (@$x[1 .. $#$x]) {
      next if ref $f && @$f && !ref $f->[0] && $rule_fields{$f->[0]};
      if (!$act && ref $f) { decl_action($acc, $s->{dir}, $s->{dir}, $f, 1) }   # (rule <action>), the short form
      else { push @{$acc->{unknown}}, canon($f) }
    }
  }
  return $acc;
}
# Which stanzas declare each alias: a rule's (alias ...) field, an (alias (name
# ...)) stanza, and the `runtest-NAME` dune generates for a (test)/(tests) name
# and for a library with (inline_tests).
my %declared_by;   # "DIR\0ALIAS" -> [stanza...]
for my $s (@dstanzas) {
  my @names;
  if ($s->{kind} eq "rule") { @names = map { ref $_ ? () : $_ } @{dfield($s->{x}, "alias") // []}, @{dfield($s->{x}, "aliases") // []} }
  elsif ($s->{kind} eq "alias") { @names = grep { !ref } @{dfield($s->{x}, "name") // []} }
  elsif ($s->{kind} =~ /^tests?$/) { @names = map { ref $_ ? () : "runtest-$_" } @{dfield($s->{x}, "names") // dfield($s->{x}, "name") // []} }
  elsif ($s->{kind} eq "library" && dfield($s->{x}, "inline_tests")) { @names = map { ref $_ ? () : "runtest-$_" } @{dfield($s->{x}, "name") // []} }
  push @{$declared_by{"$s->{dir}\0$_"}}, $s for @names;
}
my %is_per_test = map { $_->{name} => 1 } @al;
# What a declared alias needs, its own stanzas merged. An alias it needs that is
# no per-test alias (a helper: `(alias helper)` in its deps) is inlined, to a
# fixpoint: what the helper reads, the needing alias reads (gh-ocannl-1431
# review). A helper no stanza declares, or an `(alias_rec ...)` of one, is
# unknown: the needing alias keeps the blind treatment.
my %alias_acc;
sub alias_acc {
  my ($dir, $name, $seen) = @_;
  my $key = "$dir\0$name";
  return $alias_acc{$key} if $alias_acc{$key};
  my $acc = decl_new();
  $seen //= {};
  return $acc if $seen->{$key}++;
  my @by = @{$declared_by{$key} // []};
  push @{$acc->{unknown}}, "alias \@" . in_dir($dir, $name) . ", which no stanza declares" unless @by;
  for my $one (map { decl_stanza($_) } @by) {
    $acc->{files}{$_} = 1 for keys %{$one->{files}};
    $acc->{runs}{$_} = 1 for keys %{$one->{runs}};
    push @{$acc->{globs}}, @{$one->{globs}};
    push @{$acc->{unknown}}, @{$one->{unknown}};
    for my $ad (keys %{$one->{adeps}}) {
      if ($is_per_test{$ad}) { $acc->{adeps}{$ad} = 1; next }
      my ($hd, $hn) = $ad =~ m{^\@(?:(.*)/)?([^/]+)$};
      $acc->{helpers}{$ad} = 1;
      merge_acc($acc, alias_acc($hd // ".", $hn, $seen));
    }
    for my $r (keys %{$one->{adeps_rec} // {}}) {
      my ($rd, $rn) = split /\0/, $r;
      if ($rn =~ /^(?:runtest|train|slow)-/) { $acc->{adeps_rec}{$r} = 1 }
      else { push @{$acc->{unknown}}, "(alias_rec $rn)" }
    }
  }
  for my $f (sort keys %{$acc->{files}}) {
    next if -e $f;
    my ($fd, $fb) = dir_of($f);
    if ($targets_of{$fd}{$fb}) { $acc->{gen}{$f} = 1 }
    else { push @{$acc->{unknown}}, "$f (neither a source file nor a target dune shows)" }
  }
  return $alias_acc{$key} = $acc;
}
sub merge_acc {
  my ($into, $from) = @_;
  for my $k (qw(files runs adeps adeps_rec gen helpers)) { $into->{$k}{$_} = 1 for keys %{$from->{$k} // {}} }
  push @{$into->{globs}}, @{$from->{globs}};
  push @{$into->{unknown}}, @{$from->{unknown}};
}
my @declared_generated;   # generated files a declared stanza needs: their rules are queried too
for my $a (@al) {
  my ($an) = $a->{name} =~ m{([^/]+)$};
  @{$declared_by{"$a->{dir}\0$an"} // []} or next;
  my $acc = alias_acc($a->{dir}, $an);
  $acc->{via} = [sort keys %{$acc->{gen} // {}}];
  push @declared_generated, @{$acc->{via}};
  $a->{decl} = $acc;
}
@declared_generated = sort keys %{{ map { $_ => 1 } @declared_generated }};

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
my %blind_target;   # a target whose rule needs a helper alias this reader cannot read
my %target_helpers; # target -> {helper alias its rule needs, inlined}
my %queried;
my @queue = (@own_targets, @declared_generated);
while (@queue) {
  $queried{$_} = 1 for @queue;
  my %next;
  for my $rule (dune_rules(@queue)) {
    my $tg = field($rule, "targets");
    my @files = map { atom($_, "targets") } items(field($tg // [], "files"));
    my (%r, %p, @g, %v, %ad, %hl, $blind_rule);
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
        my $an = atom(field($spec, "name"), "alias name");
        my $full = "\@" . in_dir($ad, $an);
        if ($is_per_test{$full}) { $ad{$full} = 1; next }
        # A helper alias: what it needs, this rule needs (its stanzas read as text).
        my $h = alias_acc($ad, $an);
        $hl{$full} = 1;
        $hl{$_} = 1 for keys %{$h->{helpers} // {}};
        $p{$_} = 1 for keys %{$h->{files}};
        $v{$_} = 1 for keys %{$h->{gen} // {}};
        $r{$_} = 1 for keys %{$h->{runs}};
        $ad{$_} = 1 for keys %{$h->{adeps}};
        push @g, @{$h->{globs}};
        $blind_rule = 1 if @{$h->{unknown}} || %{$h->{adeps_rec}};
      } elsif ($k !~ /^(?:Env|Universe|Sandbox_config)$/) {
        fail("a dependency \"$k\" in dune rules output is outside this helper's boundary");
      }
    }
    for my $f (@files) {
      my $t = strip_ctx($f) // next;
      $runs{$t} = \%r; $reads{$t} = \%p; $globs{$t} = \@g; $via{$t} = [sort keys %v]; $adeps{$t} = \%ad;
      $blind_target{$t} = 1 if $blind_rule;
      $target_helpers{$t} = \%hl;
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
      $blind_target{$t} = 1 if $blind_target{$u};
      $target_helpers{$t} = { %{$target_helpers{$t} // {}}, %{$target_helpers{$u} // {}} };
    }
  }
  fold($_) for sort keys %queried;
  # A declared alias: what its own stanzas need, as one more own target of the
  # alias, read like a rule dune dumped (its generated inputs folded in).
  for my $a (grep { $_->{decl} } @al) {
    my $acc = $a->{decl};
    my $k = "$a->{name}\0declared";
    my %ad = %{$acc->{adeps}};
    for my $r (keys %{$acc->{adeps_rec}}) {
      my ($rd, $rn) = split /\0/, $r;
      $ad{$_->{name}} = 1 for grep { under($_->{dir}, $rd) && $_->{name} =~ m{/\Q$rn\E$} } @al;
    }
    $runs{$k} = { %{$acc->{runs}} }; $reads{$k} = { %{$acc->{files}} }; $globs{$k} = [@{$acc->{globs}}];
    $via{$k} = [@{$acc->{via}}]; $adeps{$k} = \%ad;
    fold($k);
    push @{$a->{own}}, $k;
    $a->{declared} = !@{$acc->{unknown}};
  }
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

# A changed `dune` file, scoped to the stanzas whose text changed (comments and
# layout aside), compared as a multiset between its two sides. Each changed
# stanza, old or new version, reaches: the aliases it declares; the stanza it
# is (a library, executables, a test), whose reach then spreads by linking; and
# the files a rule makes (its targets, or what its action writes), through
# whatever reads them. Any other kind of stanza (`env`, `dirs`, `copy_files`,
# `ocamllex`, ...), a side this reader cannot read, a new file and `--files`
# mode return undef: the whole directory and those beneath it, as before.
my %dune_alias;   # per-test aliases a changed stanza declares
sub dune_side {
  my ($which, $p) = @_;
  my $f = "$tmp/$which/$p";
  return undef unless -f $f;
  my $text = slurp($f);
  my $items = eval { parse_dune($text, "the $which side of $p") } or return undef;
  return [dune_stanzas((dir_of($p))[0], $items, parse_dune($text, $p, 1))];
}
sub stanza_index {   # the @st entries a stanza of KIND and NAMES in DIR is
  my ($kind, $dir, @names) = @_;
  my %n = map { $_ => 1 } @names;
  return grep {
    my $s = $st[$_];
    $s->{dir} eq $dir && ($kind eq "library" ? $s->{kind} eq "library" : $s->{kind} ne "library")
      && grep({ $n{$_} } @{$s->{names}}, $kind eq "library" ? keys %{$s->{units}} : ())
  } 0 .. $#st;
}
sub stanza_mentions {   # does any text of stanza S name the repository path M (a file, a %{...:M}, a glob)?
  my ($s, $m) = @_;
  my ($md, $mb) = dir_of($m);
  my @todo = ($s->{x});
  while (defined(my $y = shift @todo)) {
    if (ref $y) {
      if (@$y == 2 && !ref $y->[0] && $y->[0] =~ /^(glob_files(?:_rec)?|source_tree)$/ && !ref $y->[1]) {
        my $form = $y->[0];
        my $g = decl_root($s->{dir}, $y->[1]);
        if ($form eq "source_tree") { my $t = rpath($s->{dir}, $g); return 1 if defined $t && under($m, $t); next }
        my ($gd, $pat) = $g =~ m{^(.*)/([^/]*)$} ? ($1, $2) : (".", $g);
        my $t = rpath($s->{dir}, $gd) // next;
        return 1 if ($form eq "glob_files" ? $md eq $t : under($md, $t)) && predicate($pat)->($mb);
        next;
      }
      push @todo, @$y;
      next;
    }
    for my $c ($y, map { s/^[^:]*://r } $y =~ /%\{([^}]*)\}/g) {
      my $r = rpath($s->{dir}, decl_root($s->{dir}, $c));
      return 1 if defined $r && $r eq $m;
    }
  }
  return 0;
}
sub dune_scope {
  my ($p) = @_;
  my ($old, $new) = (dune_side("old", $p), dune_side("new", $p));
  return undef unless $old && $new;
  # Only forms modeled in full are scoped; a `(subdir ...)` on either side is not.
  my ($pd) = dir_of($p);
  return undef if grep { $_->{dir} ne $pd } @$old, @$new;
  my %count;
  $count{$_->{text}}++ for @$old;
  my @diff;
  for my $s (@$new) { if ($count{$s->{text}}) { $count{$s->{text}}-- } else { push @diff, $s } }
  my %left;
  $left{$_->{text}}++ for @$new;
  for my $s (@$old) { if ($left{$s->{text}}) { $left{$s->{text}}-- } else { push @diff, $s } }
  my $r = { aliases => {}, stanzas => {}, targets => {}, changed => [] };
  my $atoms = sub { map { ref $_ ? () : $_ } @{$_[0] // []} };
  for my $s (@diff) {
    my ($k, $x, $dir) = ($s->{kind}, $s->{x}, $s->{dir});
    return undef if has_opaque($x);   # a block string: not read, so the whole directory
    my @names = ($atoms->(dfield($x, "name")), $atoms->(dfield($x, "names")));
    my $alias = sub { $r->{aliases}{"\@" . in_dir($dir, $_[0])} = 1 };
    my $label = "$k" . (@names ? " @names" : "") . ($dir eq (dir_of($p))[0] ? "" : " in $dir");
    push @{$r->{changed}}, $label unless grep { $_ eq $label } @{$r->{changed}};
    if ($k eq "rule") {
      my @declares = ($atoms->(dfield($x, "alias")), $atoms->(dfield($x, "aliases")));
      $alias->($_) for @declares;
      my @t = ($atoms->(dfield($x, "target")), $atoms->(dfield($x, "targets")));
      unless (@t) {   # targets dune infers from the action
        my @todo = @{dfield($x, "action") // [grep { ref } @$x[1 .. $#$x]]};
        while (defined(my $y = shift @todo)) {
          next unless ref $y && @$y && !ref $y->[0];
          my ($h, @args) = @$y;
          push @t, $args[0] if $h =~ /^(?:with-(?:stdout|stderr|outputs)-to|write-file)$/ && @args && !ref $args[0];
          push @t, $args[1] if $h =~ /^copy#?$/ && @args > 1 && !ref $args[1];
          push @todo, grep { ref } @args unless $h eq "no-infer";
        }
      }
      @t || @declares or return undef;
      for my $t (@t) {
        return undef if $t =~ /%\{(?!target)/;   # a target name this reader cannot expand
        my $rel = rpath($dir, $t) // return undef;
        $r->{targets}{$rel} = 1 unless $t =~ /%\{/;
      }
    } elsif ($k =~ /^tests?$/) {
      $alias->("runtest-$_") for @names;
      $r->{stanzas}{$_} = 1 for stanza_index("tests", $dir, @names);
    } elsif ($k =~ /^executables?$/) {
      my @i = stanza_index("executables", $dir, @names) or return undef;
      $r->{stanzas}{$_} = 1 for @i;
    } elsif ($k eq "library") {
      my @i = stanza_index("library", $dir, @names) or return undef;
      $r->{stanzas}{$_} = 1 for @i;
      $alias->("runtest-$_") for dfield($x, "inline_tests") ? @names : ();
    } elsif ($k eq "alias") {
      $alias->($_) for @names;
      $alias->($_) for map { ref $_ && @$_ == 2 && !ref $_->[1] && ($_->[0] // "") eq "alias" ? $_->[1] : () } @{dfield($x, "deps") // []};
    } else { return undef }
  }
  return $r;
}

# Changed paths -> stanzas, and the aliases whose own targets read them.
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
  my $scoped = $b eq "dune" ? dune_scope($p) : undef;
  if ($scoped) {
    # Only the stanzas whose text changed (either side of the edit) reach anything.
    my %al_of = map { $_->{name} => $_ } @al;
    my @named = grep { $al_of{$_} } sort keys %{$scoped->{aliases}};
    # A changed helper alias (no per-test name): every per-test alias needing it,
    # through its own stanzas or a rule dune dumps.
    for my $h (grep { !$al_of{$_} } keys %{$scoped->{aliases}}) {
      for my $a (@al) {
        push @named, $a->{name}
          if ($a->{decl} && $a->{decl}{helpers}{$h}) || grep { $target_helpers{$_}{$h} } @{$a->{own}};
      }
    }
    @named = sort keys %{{ map { $_ => 1 } @named }};
    $dune_alias{$_} = 1 for @named;
    @ix = nsort(keys %{$scoped->{stanzas}});
    my @made = sort keys %{$scoped->{targets}};
    # A file a changed rule makes: an alias whose own targets' rules read it, or
    # are it; a stanza whose generated module it is, or whose generated
    # module's rule reads it; the owners of a stub object compiled from it (a
    # generated header), as for a changed path.
    my $n_made = 0;
    for my $m (@made) {
      my ($md, $mb) = dir_of($m);
      for my $g (grep { $_ eq $m || reads($_, $m, $md, $mb) } sort keys %generated) {
        $hit{$_} = $whole_hit{$_} = 1 for @{$generated{$g}};
      }
      for my $o (grep { /\.o$/ && ($_ eq $m || reads($_, $m, $md, $mb)) } @stub_targets) {
        $hit{$_} = $whole_hit{$_} = 1 for obj_owners($o);
      }
      # A compiled stanza naming it anywhere in its text (`preprocessor_deps`,
      # `embed`, a `(deps)` of its own, ...): which field means a dependency is
      # not decided here, so any mention reaches the stanza; one this build
      # does not list reaches its whole directory.
      for my $s (grep { $_->{kind} =~ /^(?:library|executables?|tests?)$/ && stanza_mentions($_, $m) } @dstanzas) {
        my @i = stanza_index($s->{kind} eq "library" ? "library" : "executables", $s->{dir}, dnames($s));
        if (@i) { $hit{$_} = $whole_hit{$_} = 1 for @i }
        else { $dune_dirs{$_} = 1 for grep { under($_, $s->{dir}) } $s->{dir}, keys %$aliases }
      }
      for my $a (@al) {
        next unless grep { $_ eq $m || reads($_, $m, $md, $mb) } @{$a->{own}};
        $read_by{$p}{$a->{name}}++ or $n_made++;
      }
    }
    push @parts, scalar(@{$scoped->{changed}}) . " changed stanza(s)"
      . (@{$scoped->{changed}} ? " (" . join(", ", @{$scoped->{changed}}) . ")" : ", comments or layout only")
      . (@named ? "; naming " . scalar(@named) . " alias(es)" : "")
      . (@made ? "; making " . scalar(@made) . " file(s), read by the rules of $n_made alias(es)" : "");
  } elsif ($b eq "dune" || $b eq "dune-project" || $b eq "dune-workspace") {
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
  return [map { @$_ } values %{$s->{mods}}] if $s->{mods};
  my $ds = static_of($i) // return undef;
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
  return [grep { -f } map { my $n = $_; map { in_dir($s->{dir}, "$n.$_") } "ml", "mli" } @names];
}
sub is_private { my $ds = static_of($_[0]); $ds && !grep { defined dfield($ds->{x}, $_) } qw(public_name public_names) }
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
            push @more, @{stanza_sources($lib_by_name{$n}) // ($files = undef, [])};
          }
        } elsif (!(!ref $pp->[0] && $pp->[0] eq "no_preprocessing")) { $files = undef }   # per_module, action: not read
      }
    }
    if ($files) {
      $tok{$_} = 1 for map { keys %{tokens_of($_)} } @more;
      my (%priv, %seen);
      my @q = @{$st[$i]{to}};
      while (defined(my $j = shift @q)) {
        # A changed library is not the owner's own code, even when private: its modules
        # name their own definitions.
        next if $seen{$j}++ || $st[$j]{kind} ne "library" || !is_private($j) || $mod_hit{$j};
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
my %reached_whole = %whole_hit;
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
        # Every module name in the head of a `module NAME ...` or `include ...` --
        # up to the first `struct`, `sig`, `end`, `let`, `val`, blank or unindented line,
        # `module type of` read through -- may be what it re-exports: `module G : module type of M = M`,
        # `include (M : S)`, a functor application. Over-reading only ranks more
        # aliases as naming the change.
        while ($s =~ /\b(module|include)\b/g) {
          my ($kw, $at) = ($1, pos($s));
          my $head = substr($s, $at, 400);
          $head =~ s/(?:\b(?:struct|sig|end|let|val)\b|\n[ \t]*\n|\n(?=\S)).*\z//s;
          my @names = $head =~ /\b([A-Z][\w']*)/g;
          next if $kw eq "module" && (!@names || $head !~ /^\s*(?:rec\s+)?\Q$names[0]\E\b/);
          my $as_name = $kw eq "module" ? shift @names : $x;
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
my (%out, %unattributed, %opaque_hit, %total, %listed, %linked, %aliased);
my %read_any = map { %{$read_by{$_}} } keys %read_by;
# The tier of each listed alias: "main"; "link", reached only by linking a
# library a changed module belongs to, never naming that module (see
# link_only); "guess", listed only because its action is unknown.
my %tier;
for my $a (@al) {
  my %owners = (%{$a->{owners}}, map { %{$runs{$_} // {}} } @{$a->{own}});
  $aliased{$_} = 1 for keys %owners;
  $total{$a->{dir}}++;
  my $unknown = !%owners && !@{$a->{own}};
  # A declared alias (its stanzas read in full above) is no longer opaque.
  my $opaque = (!$a->{golden} && !$a->{declared}) || grep { $blind_target{$_} } @{$a->{own}};
  $unattributed{$a->{name}} = 1 if $unknown && $touched{$a->{dir}};
  my $opaque_listed = $opaque && ($any_data || $data_dirs{$a->{dir}});
  $opaque_hit{$a->{dir}}++ if $opaque_listed;
  my @by_owner = grep { $reached{$_} } nsort(keys %owners);
  if ($dune_dirs{$a->{dir}} || $dune_alias{$a->{name}} || $read_any{$a->{name}}) { $tier{$a->{name}} = "main" }
  elsif (@by_owner) { $tier{$a->{name}} = link_only(@by_owner) ? "link" : "main" }
  elsif ($unattributed{$a->{name}} || $opaque_listed) { $tier{$a->{name}} = "guess" }
}
# An alias whose own rules need a listed alias (an ambient gate such as
# `runtest-env_spelling_gate`) is listed too, in that alias's tier, to a
# fixpoint -- unless that one is listed only because its undumped action might
# read a changed path: the needing alias reads nothing from it (an alias makes
# no file), so what that action reads changes only whether it, the needed one,
# passes.
my %rank = (main => 2, link => 1);
my %by_alias;
for (my $grew = 1; $grew;) {
  $grew = 0;
  for my $a (@al) {
    my $now = $rank{$tier{$a->{name}} // ""} // 0;
    next if $now == 2;
    my $best = 0;
    for my $t (@{$a->{own}}) {
      for my $n (keys %{$adeps{$t} // {}}) { my $r = $rank{$tier{$n} // ""} // 0; $best = $r if $r > $best }
    }
    next unless $best > $now;
    $tier{$a->{name}} = $best == 2 ? "main" : "link";
    $by_alias{$a->{name}} = 1;
    $grew = 1;
  }
}
for my $n (keys %tier) {
  my ($d) = $n =~ m{^\@(?:(.*)/)?[^/]+$};
  if ($tier{$n} eq "link") { $linked{$d // "."}++; next }
  $out{$n} = 1;
  $listed{$d // "."}++;
}
my @link_only = sort grep { $tier{$_} eq "link" } keys %tier;

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
printf STDERR "affected: %s: %d per-test alias(es) whose actions dune does not dump, listed for the changed paths there\n", $_, $opaque_hit{$_} for sort keys %opaque_hit;
print STDERR "affected: unread: $_ (no stanza owns it and no rule dune shows reads it; only aliases whose actions dune does not dump are listed for it)\n" for @unread;
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

#!/usr/bin/env bash
# Check a merge commit: the command `.github/ship-pr-merge-build` gives ship-pr's merge build,
# which runs it at the root of the PR head merged with master's tip, before the merge call.
# Run it by hand the same way, at the root of a checkout whose HEAD is a merge you built.
# Usage: tools/merge-build-check.sh [--plan [--base REV] | --plan-merge] [--all] | --compile-only
#
# HEAD is read as the merge to judge: first parent the base, second parent the PR head (the
# order `git merge` and ship-pr's `commit-tree -p <base tip> -p <head>` give). Four tiers, run in
# this order, the first failure ending the run:
#   format   always: `tools/fmt-check.sh` with its scans off (the compile tier runs them): the
#            formatter and the invalid-odoc check CI's `Formatting` job makes.
#   compile  always: `dune build @check @test/operations/scans` (about 140 s from a fresh tree
#            on mac-studio: the compile and the repository scans, which read the repository
#            itself -- dune stanzas, sources, scripts, Markdown notes and docs, goldens -- so
#            the config makes every shared path an input).
#   tests    always: the per-test aliases the PR's change reaches, as the pre-push run lists them
#            (`tools/test-run.sh prepush --since MERGE_BASE..HEAD_SIDE --list`: affected's default
#            list without its `slow-` and `train-` members), built by `dune build -j 4`. The
#            change is read from the two REVISIONS, never from the tree, and the reach from the
#            tree: so ship-pr's build of the base alone (below) asks about the same change and
#            runs the base's own copies of those tests. Three answers run less than the list:
#              - no per-test alias reached: nothing to run, the tier passes;
#              - CROSS-CUTTING (the list is most of the suite): its first tier runs
#                (`prepush --referenced-only`) when that is narrow, nothing when it is wide too.
#                The rest is OWED, said on a line of its own: the suite is longer than this
#                build's bound, so a cross-cutting change runs it before the merge on the PR's
#                head and records that run (OWED RECORD, below);
#              - INCOMPLETE (a changed path is UNMAPPED): what the list holds runs, and the gap
#                is OWED the same way, with every `train-` alias too: what the unmapped path
#                reaches is unknown. A mapping gap is `affected`'s to close; a record covers it
#                meanwhile;
#              - a list of more than TESTS_MAX aliases (200, sized below): it does not run, and
#                is OWED: such a build would not fit the bound. A cross-cutting change's first
#                tier is held to the same length.
#            The `train-` aliases the change reaches (prepush leaves them off its list) never
#            run here: they serialize on the training lock, and a cross-cutting change reaches
#            most of them, past the bound. They are OWED too, whatever the list's answer.
#            prepush names them on anchored `test-run: prepush: train: ALIAS` stderr lines.
#            Only canonical @DIR/train-NAME operands are read; each must exist in Dune's
#            alias listing, be distinct, and agree with the printed count, else the tier fails.
#            With no names, a nonzero or missing count asks the legacy `affected
#            MERGE_BASE..HEAD_SIDE` route for count-only protocol compatibility.
#            A count must match that answer too. Other diagnostic text is not a name.
#            The merged tree ships both tools together; the base-alone comparison also runs
#            both base versions. An older base alone does not create a mixed-version case.
#   OWED RECORD  what is OWED must be recorded for the PR's head, or the tier FAILS before it
#            builds anything: `tools/owed-run.sh check HEAD_SIDE NEED...`, NEED `runtest` for
#            an OWED list, `train` for an INCOMPLETE one, each `train-` alias reached, and on
#            non-Linux hosts (`uname -s` != Linux) the scoped Linux shell leg (gh-ocannl-2414):
#            when the head side changed any `*.sh` path reached by the harness census or a
#            sweep-driver Dune rule (`affected: PATH: reached by sweep driver alias @ALIAS`),
#            each reached harness member (`linux:member:<MEMBER>`, or `linux:harnesses` when
#            the whole manifest is reached or the census ask failed on a `*.sh`) and each
#            reached sweep-driver alias (`linux:@DIR/runtest-NAME`). A merge build running on
#            Linux satisfies the Linux shell leg by itself; changes touching no census- or
#            sweep-reached `*.sh` owe nothing new. The record is the ref
#            refs/owed-runs/<head sha> on the PR repository, fetched with a
#            bound and failing closed; tools/owed-run.sh's header is its boundary (what it
#            admits: a finished, clean, exit-0 `--expect-backend cc` run of the full suite or
#            of the `train-` aliases, a finished, clean, exit-0 `os=Linux` `tools/linux-shell-leg.sh`
#            run covering the Linux shell targets, on the head itself, or a waiver for exactly that
#            head; what it refuses: a missing, unreadable, stale, foreign, dirty, failed, non-cc,
#            non-Linux, filtered or first-tier record). The record is keyed by the head, not the merged
#            tree: a base that moved since leaves it standing (gh-ocannl-861: a green head
#            merges as it is), and a push after the run stales it. The refusal names the
#            MERGED tree, which ship-pr's build of the base alone (the base's tree, the same
#            parents, so the same change and the same record) never prints: a missing record
#            is the merge's own failure, which `--override` does not pass; a waiver is the way
#            through. A merge owing nothing reads no record and needs no network.
#            A list that cannot be read at all (prepush refusing for another reason, a failed
#            or unanswered ask of either list, a list both cross-cutting and incomplete, of
#            which no first tier can be asked, a HEAD with no two sides) fails the tier: nothing
#            here guesses what to run.
#   harness  always asked: the shell harnesses the PR's change reaches, of every group, the
#            members four at once (SHELL_JOBS, gh-ocannl-2002): what the scheduled `Formatting`
#            job's harness batch and the `Shell harnesses` jobs read. The change is the paths
#            the HEAD side changed since the merge base, both names of a rename, a path the
#            base side renamed counting by its other name too, scripts or not. What runs is
#            `tools/test-harnesses.sh --affected --files PATH...`, the mapping `tools/test-run.sh
#            affected --harnesses` owns, read from the merged tree. A change reaching no member runs
#            none. An exact `affected: SCRIPT: verified hand-run harness marker` line from the
#            successful single-path ask covers a script with empty harness reach. The checker in
#            affected-tests.sh owns the marker grammar and conflicts. Any other changed script no
#            member reaches leaves the tier PENDING until the tests tier's
#            list is read (gh-ocannl-2304): it counts as covered when an alias of the list the tests
#            tier runs here names it as a dependency, as an exact `affected: SCRIPT: named as a
#            dependency by the rule of ALIAS` line of the ask that printed that list says
#            (tools/affected-tests.sh's "Literal dependencies": one rule of the alias names its
#            path, and dune reads it as a File of that rule). Every such script covered, the tier
#            narrows to what the paths reach, nothing when that is no member; any one not covered
#            runs the whole tier. Until 2026-11-13 this was a shell tier that ran only when BOTH
#            sides changed a script, the `--shell` group alone: remote CI read a PR's own change
#            then, and this build only the clash of two.
# The harness tier's scope is a fail-closed allowlist: it narrows only when HEAD is a two-parent
# merge, its merge base and both sides' changes were read, git listed every changed path
# unquoted, each changed path matching `*.sh` (SHELL_GLOB), asked alone, reaches a member, has a verified marker, or is
# covered (above: named as a dependency by an alias in the list the tests tier runs here, read
# whole-line from that ask), and the ask for all the paths answers. Anything else -- a one-parent
# HEAD, a failed git read, a path git had to quote, an ask that fails (test-harnesses.sh refuses a
# line of `affected --harnesses` it does not recognize, a path reading as an option, a failed
# `affected`), an unmarked script no member reaches and no alias run here names (one an alias outside the
# list names, a list OWED rather than run, a glob or source-tree read), or a scoped run refused
# before any member started (exit 2) -- runs the whole tier, every group, never less. A script is
# asked alone because a reach the other paths make up for would hide one nothing maps. Coverage
# is no claim that the alias drives every path through the script: it is the dune file's own
# statement that the alias's rule depends on it, and the tests tier ran that alias green.
# The shared-red comparison (ship-pr's build of the base alone) stays like with like: that build
# carries the merge's two parents on the base's tree, so it reads the same two sides here, asks
# about the same change and narrows to the same paths, the reach read from the base's text. A
# reach the two trees answer differently prints lines the base's log lacks, which ship-pr reads
# as the merge's own failure: a refusal, never a red waved through as shared. So what this
# script prints of a failed ship-pr run is the same for the same failure: the timing summary
# is emitted at exit only on success or on a standalone run (gh-ocannl-1950, gh-ocannl-2433).
# The summary reports whole seconds for each completed tier and each finished harness member.
# It covers the tests tier's alias and record reads and the harness tier's scoped attempt and
# fallback. Initial git and harness selection, toolchain setup and timeout kills are outside it.
# Use ordinary successful runs to re-measure the tiers when the harness set or per-alias cost
# grows. The tests tier prints neither its list nor a count (a PR that adds a test lists one
# alias more than the base alone does); `--plan` shows the command that lists it.
# The coverage is read the same way: from the tree's own ask, and the alias it names is printed (a
# dune file the PR changed can name a script the base's does not, a difference that refuses). --all
# runs the whole local harness tier whatever HEAD is, while still reading both sides' changed paths
# on a two-parent merge so --all cannot bypass the owed Linux shell leg on non-Linux hosts. --plan alone prints the tier selection for HEAD
# without asking Dune or reading records, a PENDING tier as such. On a branch, --plan --base REV
# reads the locally resolved base tip (fetch first to refresh a remote-tracking ref) and committed
# HEAD. It asks the tests tier on their candidate merge in a scratch checkout outside the
# repository, with parents in ship-pr's order. The caller's index and working files are ignored and
# its refs stay untouched. The merged copy of this script handles the `--plan-merge` query, so
# changes the base brought to the gate apply too. It runs Dune queries and reads the head's owed-run
# record when there is a need.
# Only the record check's exit status decides the plan verdict (no prose of `owed-run.sh` is
# parsed): `merged_plan` (`--plan-merge`, configured as `.github/ship-pr-merge-build`'s `plan` and
# called by `--plan --base REV`) prints exactly one `ship-pr-merge-plan:` line beside its
# `owed-run needs:` line: `nothing-owed` when nothing is owed, `admitted <needs>` when
# `tools/owed-run.sh check` exits 0, or `unadmitted <needs>` when it exits 1 with output
# (definitive record refusal: missing, foreign, invalid or uncovered). Transport-unreadable (exit 3
# from `tools/owed-run.sh check`), a diagnostics-file setup/read error, an empty-output exit 1, any
# other `owed-run.sh` status (usage or execution failure), or an unreadable tests tier fails the
# plan query (exit 1) with no verdict line, so ship-pr treats it as no verdict and falls back to the
# build (which remains fail-closed on any nonzero `owed-run.sh` status). `--plan` alone does not
# ask Dune or read records and prints no verdict line. A plan makes no build or readiness verdict
# and writes no record. A conflict or an unreadable plan fails the query.
# The only extra affected text read is one anchored, printable `affected: CROSS-CUTTING reason:`
# line from the full-list ask. It is display-only; a missing, duplicate or malformed reason
# contributes nothing. The trailers, exit statuses and existing list protocol still govern.
# --compile-only runs the existing Dune setup and the compile tier alone (`dune build --root .
# @check @test/operations/scans` at the root of the merged tree), with the same command contract,
# timing summary policy and exit statuses as the full run, skipping format, tests, owed-run record
# reads and harnesses. Configured as `.github/ship-pr-merge-build`'s `recheck` for a moved-base
# refresh (ship-pr re-runs `plan` before `recheck`, advisory on no-verdict fallback). Cannot be
# combined with `--plan`, `--plan-merge`, `--base` or `--all` (exit 2).
#
# Test-reach cache (gh-ocannl-2488): successful `read_tests` answers (the test alias list, verdict,
# owed flags/reasons, training/Linux reach needs, `CROSS-CUTTING reason:` lines, and literal-
# dependency coverage lines) may be reused across `--plan --base`, `--plan-merge` and full merge
# builds for the same clean merge candidate and Dune inputs. Failures (`tests=unread`), owed-run
# record checks (`tools/owed-run.sh check`), builds and harnesses are never cached: the published
# owed-run record is always checked afresh.
# Cache location: `OCANNL_TOOL_MERGE_BUILD_REACH_CACHE` when set (`""`, `off` or `0` disables
# caching; a relative path resolves against the caller's initial working directory), otherwise
# `$XDG_CACHE_HOME/ocannl/merge-build-reach` (when `XDG_CACHE_HOME` is absolute) or
# `$HOME/.cache/ocannl/merge-build-reach` (when `HOME` is absolute); disabled if neither is set.
# Cache-key allowlist (fail-closed): schema version (`merge-build-reach-v1`), `TESTS_MAX`, two-
# parent `HEAD` (`HEAD^1`, `HEAD^2`, no `HEAD^3`), `merge-base`, `HEAD^{tree}`, clean working tree
# (`git status --porcelain -z --untracked-files=normal` empty and `git ls-files -v -z` showing only
# `H ` entries so `assume-unchanged`/`skip-worktree` and dirty/untracked files bypass both read
# and write), hashes of all gitignored non-directory files that `affected-tests.sh` copies into its
# probe (regular readable files only; any symlink or non-regular ignored file bypasses the cache),
# `uname -s`/`uname -m`, resolved `dune` and `ocamlc` executables (`realpath`, stat metadata, plus
# content hash when `<= 65536` bytes), active opam switch state (`.opam-switch/{switch-state,
# switch-config,environment}` and content hashes of `lib` `dune-package`/`META`/`META.*` and
# `findlib.conf`), `OCAMLPATH` and `findlib.conf` `path` package metadata hashes, `OCAMLFIND_CONF`,
# `DUNE_WORKSPACE` and user Dune config (`${XDG_CONFIG_HOME:-$HOME/.config}/dune/config`) file
# contents, relevant environment variables (`DUNE_*` excluding `DUNE_BUILD_DIR`, `DUNE_ROOT`,
# `DUNE_TRACE` and `DUNE_CACHE*`; `OCANNL_*` excluding `OCANNL_TOOL_*`, plus
# `OCANNL_TOOL_AFFECTED_PROBE` and `OCANNL_TOOL_AFFECTED_ARGS`; `OPAM_SWITCH_PREFIX`, `OPAMSWITCH`,
# `OPAMROOT`, `OCAMLFIND_CONF`, `OCAMLPATH`, `CAML_LD_LIBRARY_PATH`, `OCAMLPARAM`, `INSIDE_DUNE`),
# and `select_tiers`'s Linux-scope inputs (`linux_scope_err`, `linux_harness_all`, `scripts`,
# `renamed_scripts`, `linux_harness_members`).
# Storage boundary: keys are revalidated after `read_tests` completes so mid-query mutations never
# publish under a stale key. Entries are written atomically (`0600`) as `reach-<key>.v1` (via
# `.ocannl-reach-tmp.<key>.<pid>.<6-digits>` staging files) with an embedded key and `end` trailer;
# retention prunes only owner-matching files in those two exact shapes. Symlinks, wrong ownership,
# group/world-writable files, malformed entries, key mismatches or unwritable cache directories are
# treated as cache misses / skipped writes and never fail a run.
# On `--plan-merge`/`--plan --base`, a cache hit prints `merge-build-check: tests plan: reused cached reach`;
# on full merge builds, live output up to `passed` is unchanged for ship-pr's base-alone log
# comparison, and `merge-build-check: tests reach: reused cached result` joins the exit summary.
#
# Toolchain: a non-interactive shell can lack the profile that puts the opam switch on PATH
# (on mac-studio `zsh -c` resolves opam but not dune). When `dune` does not resolve, this sources
# tools/opam-env.sh, first adding Homebrew's prefix when `opam` does not resolve either.
#
# Exit status: 0 every selected tier passed (or the plan answered, even if a record is unadmitted),
# 1 a tier failed or the plan could not be read, including a merge conflict or record-check error;
# 2 a usage error.
set -u
# Name a locale before any subshell forks (gh-ocannl-2002, gh-ocannl-2462): with neither LANG
# nor LC_ALL set (e.g. macOS Terminal/SSH with only LC_CTYPE=UTF-8), Homebrew Bash's libintl
# calls CoreFoundation's preferred-languages lookup when tearing down a temporary or local
# locale assignment, which segfaults inside a forked subshell or under concurrent runs.
# LANG=C matches unset LANG in POSIX while letting caller-set categories (LC_CTYPE) win.
[ -n "${LANG:-}" ] || [ -n "${LC_ALL:-}" ] || export LANG=C
caller_pwd=$(pwd -P) || exit 2
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
me=merge-build-check
plan=0 all=0 plan_base= plan_merge=0 compile_only=0
while [ $# -gt 0 ]; do
  case $1 in
    --plan) plan=1 ;;
    --plan-merge) plan_merge=1 ;;
    --compile-only) compile_only=1 ;;
    --base)
      if [ -z "${2:-}" ] || [ -n "$plan_base" ]; then
        echo "$me: --base requires one revision" >&2; exit 2
      fi
      plan_base=$2
      shift ;;
    --all) all=1 ;;
    -h|--help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "$me: unknown argument '$1'" >&2; exit 2 ;;
  esac
  shift
done
if [ "$compile_only" = 1 ] &&
   { [ "$plan" = 1 ] || [ "$plan_merge" = 1 ] || [ -n "$plan_base" ] || [ "$all" = 1 ]; }; then
  echo "$me: --compile-only cannot be combined with --plan, --plan-merge, --base or --all" >&2
  exit 2
fi
[ -z "$plan_base" ] || { [ "$plan" = 1 ] && [ "$plan_merge" = 0 ]; } ||
  { echo "$me: --base requires --plan" >&2; exit 2; }
SHELL_GLOB='*.sh'
host_os=$(uname -s 2>/dev/null || echo unknown)
host_arch=$(uname -m 2>/dev/null || echo unknown)
# The harness tier's members run this many at once (tools/test-harnesses.sh --jobs): the width
# of the `-j 4` build the slot this runs in is sized for, and what the bound was measured at.
SHELL_JOBS=4
# The tests tier's build width: the same `-j 4`.
TEST_JOBS=4
# The longest list the tests tier builds. Sized on 2026-11-13 from a fresh clone at `-j 4`: the
# 411 aliases of `arrayjit/lib/low_level.ml`'s first tier, the widest list `affected` still calls
# narrow, took 661 s on mac-studio and 731 s on minix-amd-linux, about 1.7 s an alias, which alone
# is most of a 30-minute bound. 200 aliases is under six minutes, and a longer list is OWED.
TESTS_MAX=200

# read_side FROM TO sets side_names (every path the diff names, both names of a rename, one per
# line) and side_pairs (each rename as OLD<TAB>NEW, one per line). Status 1: git did not answer;
# 3: git quoted a path, which this does not decode.
read_side() {
  local out st a b
  out=$(git -C "$root" -c core.quotePath=false diff -M --name-status "$1" "$2") || return 1
  side_names= side_pairs=
  while IFS=$'\t' read -r st a b; do
    [ -n "$st" ] || continue
    case $a in '"'*) return 3 ;; esac
    case $b in '"'*) return 3 ;; esac
    case $st in
      R*) side_names+=$a$'\n'$b$'\n' side_pairs+=$a$'\t'$b$'\n' ;;
      *) side_names+=$a$'\n' ;;
    esac
  done <<EOF
$out
EOF
}

# read_linux_scripts_nul MB BASE HEAD: NUL-delimited fallback used when read_side hits a quoted
# path (rc=3), so Linux shell scope never falsely blocks non-shell changes and stays satisfiable
# for shell changes (gh-ocannl-2414).
read_linux_scripts_nul() {
  local out tag p line m rc=0
  out=$(perl -e '
    use strict; use warnings;
    my ($root, $mb, $base, $head) = @ARGV;
    sub read_z {
      my ($a, $b) = @_;
      open my $fh, "-|", "git", "-C", $root, "diff", "-z", "-M", "--name-status", $a, $b or exit 1;
      local $/; my $raw = <$fh>; close $fh or exit 1;
      my @toks = split /\0/, (defined $raw ? $raw : "");
      my (@paths, @pairs);
      my $i = 0;
      while ($i < @toks) {
        my $st = $toks[$i++]; last if !defined $st || $st eq "";
        if ($st =~ /^[RC]/) {
          my ($old, $new) = ($toks[$i++], $toks[$i++]);
          push @paths, $old, $new;
          push @pairs, [$old, $new];
        } else {
          push @paths, $toks[$i++];
        }
      }
      return (\@paths, \@pairs);
    }
    my (undef, $bpairs) = read_z($mb, $base);
    my ($hpaths, undef) = read_z($mb, $head);
    my (%seen_h, %seen_r, $unaskable);
    my $clean = sub { my ($x) = @_; $x =~ m{\A[A-Za-z0-9_./-]+\z} && $x !~ m{\A[-/]} && $x !~ m{(?:\A|/)\.\.(?:/|\z)} };
    for my $p (@$hpaths) {
      if ($p =~ /\.sh\z/) {
        if ($clean->($p)) { $seen_h{$p} = 1; } else { $unaskable = 1; }
      }
      for my $pr (@$bpairs) {
        my $other = $pr->[0] eq $p ? $pr->[1] : $pr->[1] eq $p ? $pr->[0] : undef;
        next unless defined $other && $other =~ /\.sh\z/;
        if ($clean->($other)) { $seen_r{$other} = 1; } else { $unaskable = 1; }
      }
    }
    print "UNASKABLE\n" if $unaskable;
    print "HEAD\t$_\n" for sort keys %seen_h;
    print "RENAME\t$_\n" for sort keys %seen_r;
  ' "$root" "$1" "$2" "$3") || return 1
  while IFS=$'\t' read -r tag p; do
    [ -n "$tag" ] || continue
    case $tag in
      UNASKABLE) linux_harness_all=1 ;;
      HEAD) scripts+=("$p") ;;
      RENAME) scripts+=("$p"); renamed_scripts+=("$p") ;;
    esac
  done <<EOF
$out
EOF
  for p in ${scripts[@]+"${scripts[@]}"}; do
    ask_reach "$p" || { linux_harness_all=1; break; }
    if [ -n "$reach" ]; then
      while IFS= read -r line; do
        [ -n "$line" ] || continue
        m=${line%% *}
        linux_harness_members+=("$m")
      done <<EOF
$reach
EOF
    fi
  done
}

# ask_reach PATH... sets `reach`, the commands of the harness members PATH reaches, one per line.
# It fails when the ask does; `--list` takes no slot and runs nothing.
ask_reach() {
  reach=$("${BASH:-bash}" "$root/tools/test-harnesses.sh" --affected --list --files "$@" \
    2>"$tests_dir/harness-ask" </dev/null)
}

# scope_tier narrows the tier to what `asked` reaches (`scoped` 1, `reach` its commands, empty
# when they reach no member), or says in `scope_why` why the whole tier runs. `scripts` is every
# asked path matching SHELL_GLOB: a reach the others make up for would hide one nothing maps, so
# each is asked alone.
# A script no member reaches leaves the tier PENDING (`pending` 1, the scripts in `unreached`)
# unless the ask verifies a hand-run marker. cover_tier settles PENDING once the tests tier is read.
scope_tier() {
  local s rc line m
  for s in ${scripts[@]+"${scripts[@]}"}; do
    ask_reach "$s" || {
      rc=$?
      linux_harness_all=1
      scope_why="what $s reaches could not be asked (test-harnesses.sh exit $rc)"
      return
    }
    if [ -n "$reach" ]; then
      while IFS= read -r line; do
        [ -n "$line" ] || continue
        m=${line%% *}
        linux_harness_members+=("$m")
      done <<EOF
$reach
EOF
    elif ! grep -qxF -- "affected: $s: verified hand-run harness marker" "$tests_dir/harness-ask"; then
      unreached+=("$s")
    fi
  done
  ask_reach "${asked[@]}" || {
    rc=$?
    scope_why="what the paths the head side changed reach could not be asked (test-harnesses.sh exit $rc)"
    return
  }
  if [ ${#unreached[@]} -gt 0 ]; then pending=1; else scoped=1; fi
}

# narrowed [SUFFIX] sets `harness` and `why` for a tier narrowed to `reach`.
narrowed() {
  if [ -z "$reach" ]; then
    harness=0 why="the $n_paths path(s) the head side changed reach no harness${1:-}"
  else
    why="scoped to what the $n_paths path(s) the head side changed reach${1:-}:"
  fi
}

# cover_tier settles a PENDING tier from the tests tier's answer (read_tests): a script no member
# reaches is covered by an alias of the list this build runs ($tests_dir/list, the tests tier
# being `list`) whose rule names that script as a dependency, as one exact line
# `affected: SCRIPT: named as a dependency by the rule of ALIAS` of the ask that printed the list
# says (tools/affected-tests.sh, "Literal dependencies"). Every unreached script covered, the tier
# narrows to what the paths reach; one not covered, or no list run here, leaves it whole.
# `covered` holds SCRIPT<TAB>ALIAS lines for show_cover.
cover_tier() {
  local s a f found
  covered=
  [ "$pending" = 1 ] || return 0
  pending=0
  if [ "$tests" != list ]; then
    why="whole -- no harness reaches ${unreached[0]}, and the tests tier runs no list here to cover it"
    return
  fi
  f=$(list_ask)
  for s in "${unreached[@]}"; do
    found=
    while IFS= read -r a; do
      [ -n "$a" ] || continue
      if grep -qxF -- "affected: $s: named as a dependency by the rule of $a" "$f"; then found=$a; break; fi
    done <"$tests_dir/list"
    if [ -z "$found" ]; then
      why="whole -- no harness reaches $s, and no alias the tests tier runs names it as a dependency"
      return
    fi
    covered+=$s$'\t'$found$'\n'
  done
  scoped=1
  narrowed ' beyond the aliases the tests tier runs'
}

# select_tiers sets `harness` (1 = run the harness tier), `why`, `script_line` (whether the head
# side changed a script, which tools/merge-build-replay.py's shell-tier set replays) and, for a
# tier it could narrow, `scoped`, `reach` and `asked` (or `scope_why`).
select_tiers() {
  harness=1 scoped=0 pending=0 scope_why= reach= asked=() scripts=() renamed_scripts=() unreached=() script_line= n_paths=0
  linux_harness_members=() linux_harness_all=0 linux_scope_err=
  local head_names base_pairs p old new other n=0 rc=0 hit=
  if [ -z "$base" ]; then
    [ "$all" = 0 ] || { why='--all'; return; }
    why='HEAD is not a two-parent merge, so the whole harness tier runs'
    return
  fi
  [ -n "$mb" ] || rc=1
  # The base side is read for its renames alone: an edit of a path the base renamed lands in the
  # merged tree under the other name.
  [ "$rc" != 0 ] || read_side "$mb" "$base" || rc=$?
  base_pairs=${side_pairs-}
  [ "$rc" != 0 ] || read_side "$mb" "$head" || rc=$?
  case $rc in
    0) ;;
    3)
      # Preserve the whole local harness tier on quoted paths, while decoding NUL-delimited Git
      # diffs for Linux shell scope so a quoted non-shell path (especially on the base side) owes
      # nothing new and a quoted shell path remains satisfiable (gh-ocannl-2414).
      read_linux_scripts_nul "$mb" "$base" "$head" ||
        linux_scope_err='the paths each side changed could not be read, so non-Linux cannot verify whether a shell source changed'
      if [ "$all" = 1 ]; then why='--all'; else why='git quoted a path one side changed, so the whole harness tier runs'; fi
      return ;;
    *)
      linux_scope_err='the paths each side changed could not be read, so non-Linux cannot verify whether a shell source changed'
      if [ "$all" = 1 ]; then why='--all'; else why='the paths each side changed could not be read, so the whole harness tier runs'; fi
      return ;;
  esac
  head_names=$(printf '%s' "$side_names" | env LC_ALL=C sort -u)
  while IFS= read -r p; do
    [ -n "$p" ] || continue
    n=$((n + 1))
    asked+=("$p")
    case $p in
      $SHELL_GLOB)
        scripts+=("$p")
        [ -n "$hit" ] || hit="$p matches $SHELL_GLOB" ;;
    esac
    # A path counts by its other name when a rename on the base side touched it (the head side's
    # own renames are among its names already).
    while IFS=$'\t' read -r old new; do
      [ -n "$old" ] || continue
      if [ "$old" = "$p" ]; then other=$new; elif [ "$new" = "$p" ]; then other=$old; else continue; fi
      asked+=("$other")
      case $other in
        $SHELL_GLOB)
          scripts+=("$other")
          renamed_scripts+=("$other")
          [ -n "$hit" ] || hit="$p was renamed from or to $other, which matches $SHELL_GLOB" ;;
      esac
    done <<EOF
$base_pairs
EOF
  done <<EOF
$head_names
EOF
  if [ -n "$hit" ]; then
    script_line="a script changed on the head side -- $hit"
  else
    script_line="no script changed on the head side -- none of the $n path(s) it changed matches $SHELL_GLOB, by either name of a rename"
  fi
  if [ "$n" = 0 ]; then
    if [ "$all" = 1 ]; then
      harness=1 why='--all'
    else
      harness=0 why='the head side changed no path'
    fi
    return
  fi
  n_paths=$n
  scope_tier
  if [ "$all" = 1 ]; then
    harness=1 scoped=0 pending=0 why='--all'
    return
  fi
  if [ "$scoped" = 1 ]; then
    narrowed
  elif [ "$pending" = 1 ]; then
    why="no harness reaches ${unreached[*]}: whole unless an alias the tests tier runs names each as a dependency, read with that tier's list (--plan --base REV asks it)"
  else
    why="whole -- $scope_why"
  fi
}

# read_parents sets `base` and `head`, the merge's two parents, and `mb`, their merge base; each is
# empty when it could not be read (`mb` alone: git did not answer).
read_parents() {
  base= head= mb=
  base=$(git -C "$root" rev-parse --verify --quiet 'HEAD^1^{commit}') &&
    head=$(git -C "$root" rev-parse --verify --quiet 'HEAD^2^{commit}') || { base= head=; return; }
  echo "$me: HEAD merges ${head:0:8} into ${base:0:8}"
  mb=$(git -C "$root" merge-base "$base" "$head") || mb=
}

# list_ask prints the ask whose stderr goes with $tests_dir/list: the first-tier ask when a
# cross-cutting change asked one (it repeats the default ask's stderr, then its own), else the
# default ask. The train- names (owed_needs) and the coverage lines (cover_tier) are read from it,
# so both describe the list this build runs.
list_ask() {
  if [ -f "$tests_dir/ask-first" ]; then echo "$tests_dir/ask-first"; else echo "$tests_dir/ask"; fi
}

# ask_tests asks the pre-push list of the head's change and sets `tests`: `list` (run
# $tests_dir/list, with `tests_owed` saying what it leaves out, if anything), `owed` (nothing to
# run here), or `unread`, with `tests_why`; `tests_incomplete` 1 when a list read was INCOMPLETE. The change is the two revisions' (MERGE_BASE..HEAD_SIDE),
# so the build of the base alone asks about the same one. The ask's own output is kept in
# $tests_dir/ask: it names the list and its size, which differ between a merge and its base.
ask_tests() {
  local rc=0 verdict
  tests= tests_verdict= tests_owed= tests_why= tests_incomplete=0
  if [ -z "$mb" ]; then
    tests=unread tests_why='HEAD has no two sides with a merge base, so what the change reaches cannot be asked'
    return
  fi
  "${BASH:-bash}" "$root/tools/test-run.sh" prepush --since "$mb..$head" --list \
    >"$tests_dir/list" 2>"$tests_dir/ask" </dev/null || rc=$?
  verdict=$(sed -n 's/^affected: verdict=//p' "$tests_dir/ask" | tail -n 1)
  tests_verdict=$verdict
  case $rc/$verdict in
    0/ok) tests=list ;;
    3/incomplete) tests=list tests_incomplete=1 tests_owed='the list is INCOMPLETE (a changed path is UNMAPPED), so what that path reaches' ;;
    2/cross-cutting,incomplete)
      # prepush asks no first tier of an incomplete list, and its refusal (2) is the status of its
      # other refusals too: nothing in this answer says the refusal was that one, so nothing here
      # says what to run or that running nothing is right.
      tests=unread tests_why='the change is CROSS-CUTTING and its list INCOMPLETE (a changed path is UNMAPPED), so no first tier can be asked: map the path in tools/affected-tests.sh, which tools/test-run.sh affected names as UNMAPPED' ;;
    2/cross-cutting)
      rc=0
      "${BASH:-bash}" "$root/tools/test-run.sh" prepush --referenced-only --since "$mb..$head" --list \
        >"$tests_dir/list" 2>"$tests_dir/ask-first" </dev/null || rc=$?
      # The first tier's own verdict: narrow (ok), incomplete, or wide (cross-cutting again, which
      # prepush refuses with 2). This ask prints the default list's trailer before the first
      # tier's, so the tier answered only when there are TWO trailers; with one, the last trailer
      # is the default list's `cross-cutting` again, and a query that failed before the tier was
      # asked would read as a wide tier. Any other answer -- a failed query, a lock refusal, a
      # missing trailer -- is no reading of the tier and fails it. This second ask is also what
      # vouches for the first one's refusal: a first ask that exited 2 for another reason fails
      # here the same way, before the tier's trailer.
      verdict=$(sed -n 's/^affected: verdict=//p' "$tests_dir/ask-first" | tail -n 1)
      [ "$(grep -c '^affected: verdict=' "$tests_dir/ask-first")" = 2 ] || verdict='no trailer of its own'
      case $rc/$verdict in
        0/ok) tests=list tests_owed='the change is CROSS-CUTTING, so every per-test alias past its first tier' ;;
        3/incomplete) tests=list tests_incomplete=1 tests_owed='the change is CROSS-CUTTING and its first tier INCOMPLETE (a changed path is UNMAPPED), so every per-test alias past what that tier holds' ;;
        2/cross-cutting | 2/cross-cutting,incomplete)
          tests=owed tests_owed='the change is CROSS-CUTTING with no narrow first tier, so the whole list' ;;
        *)
          cat "$tests_dir/ask-first" >>"$tests_dir/ask"
          tests=unread tests_why="tools/test-run.sh prepush --referenced-only --since ${mb:0:8}..${head:0:8} --list exited $rc with no verdict this script reads" ;;
      esac ;;
    *) tests=unread tests_why="tools/test-run.sh prepush --since ${mb:0:8}..${head:0:8} --list exited $rc with no verdict this script reads" ;;
  esac
  # The bound's share: a list longer than TESTS_MAX does not run, whichever answer it came from.
  if [ "$tests" = list ] && [ "$(grep -c . "$tests_dir/list")" -gt "$TESTS_MAX" ]; then
    case $tests_owed in
      *CROSS-CUTTING*) tests_owed="the change is CROSS-CUTTING and its first tier is longer than the $TESTS_MAX aliases this build runs, so the whole list" ;;
      *) tests_owed="the list is longer than the $TESTS_MAX aliases this build runs, so the whole list" ;;
    esac
    tests=owed
  fi
}

# owed_needs sets `needs`, what tools/owed-run.sh must find recorded for the head (OWED RECORD in
# the header): `runtest` for every OWED answer, `train` too for an INCOMPLETE one, and each
# `train-` alias the change reaches, with `train_why` saying why. A reach it cannot read sets
# `tests` to unread.
owed_needs() {
  local f n counted rc=0 verdict line
  needs=() train_why=
  [ -z "$tests_owed" ] || needs+=(runtest)
  if [ "$tests_incomplete" = 1 ]; then
    needs+=(train)
    train_why='a changed path is UNMAPPED, so what it reaches of the train- aliases is unknown: every one of them is'
    return
  fi
  # Read only the anchored train-name lines and the existing count trailer. Names
  # require a count, exact cardinality, distinct canonical @DIR/train-NAME operands
  # and aliases Dune lists in this tree. Other diagnostics are not parsed as names.
  # With no names, retain the legacy affected ask as count-only protocol compatibility.
  # Normal merged and base-alone trees each ship their own pair of tools together.
  f=$(list_ask)
  counted=$(sed -n -E 's/.*left off: ([0-9]+) train-.*/\1/p' "$f" | tail -n 1)
  sed -n 's/^test-run: prepush: train: //p' "$f" >"$tests_dir/train-names"
  if [ -s "$tests_dir/train-names" ]; then
    n=$(grep -c . "$tests_dir/train-names")
    if [ -z "$counted" ] || [ "$n" != "$counted" ] ||
       [ "$(sort -u "$tests_dir/train-names" | grep -c .)" != "$n" ]; then
      tests=unread tests_why="prepush train- names disagree with its count (${counted:-missing})"
      return
    fi
    local dir last_dir= name
    while IFS= read -r line; do
      case $line in
        *[!a-zA-Z0-9_@./-]* | *..* | *//* | @/* | *@*@* | '')
          tests=unread tests_why="prepush printed an invalid train- alias: $line"; return ;;
        @?*/train-?*) ;;
        *) tests=unread tests_why="prepush printed an invalid train- alias: $line"; return ;;
      esac
      dir=${line%/*} dir=${dir#@} name=${line##*/}
      case $name in train-?*) ;; *)
        tests=unread tests_why="prepush printed an invalid train- alias: $line"; return ;;
      esac
      if [ "$dir" != "$last_dir" ]; then
        if ! dune show aliases --root . "$dir" >"$tests_dir/train-aliases" 2>&1 </dev/null; then
          tests=unread tests_why="Dune could not list prepush's train- aliases in $dir"; return
        fi
        last_dir=$dir
      fi
      if ! grep -qxF "$name" "$tests_dir/train-aliases"; then
        tests=unread tests_why="prepush named a train- alias this tree does not have: $line"; return
      fi
      needs+=("$line")
    done <"$tests_dir/train-names"
    train_why='the change reaches train- aliases, which this build does not run'
    return
  fi
  [ "$counted" != 0 ] || return 0
  "${BASH:-bash}" "$root/tools/test-run.sh" affected "$mb..$head" \
    >"$tests_dir/affected" 2>"$tests_dir/train-ask" </dev/null || rc=$?
  verdict=$(sed -n 's/^affected: verdict=//p' "$tests_dir/train-ask" | tail -n 1)
  case $rc/$verdict in
    0/ok | 0/cross-cutting) ;;
    *)
      tests=unread tests_why="tools/test-run.sh affected ${mb:0:8}..${head:0:8} exited $rc with no verdict this script reads, so the train- aliases the change reaches are not known"
      return ;;
  esac
  n=0
  while IFS= read -r line; do
    case $line in
      *[[:space:]]* | '') ;;
      @*/train-?*) needs+=("$line") n=$((n + 1)) ;;
    esac
  done <"$tests_dir/affected"
  # A legacy count must agree exactly with affected; nothing guesses which is right.
  # With no count read, none named is the answer.
  if [ -n "$counted" ] && [ "$n" != "$counted" ]; then
    tests=unread tests_why="prepush counted $counted train- alias(es) the change reaches, but tools/test-run.sh affected ${mb:0:8}..${head:0:8} named $n"
    return
  fi
  # No count: a merge that adds a training test reaches one alias more than its base alone.
  [ "$n" = 0 ] || train_why='the change reaches train- aliases, which this build does not run'
}

# linux_shell_needs appends scoped Linux shell-leg NEEDs (gh-ocannl-2414) to `needs` on non-Linux
# hosts when the head side changed any `*.sh` path reached by the harness census or a sweep-driver
# Dune rule, and on Linux when a reached sweep-driver alias will not be built locally by this
# merge build's tests tier (e.g. an OWED list or cross-cutting first tier that drops it). A Linux
# merge build always runs the harness tier locally and satisfies built sweep-driver aliases by
# itself. Sets `linux_why` and `linux_leg_args` when any Linux need is added; fails closed
# (`tests=unread`) when Linux shell reach cannot be read (quoted path, disabled Dune probe,
# failed rename ask, or an invalid/absent sweep-driver alias).
linux_shell_needs() {
  local s line a dir last_dir= name m sorted_h sorted_d manifest_all af rrc=0
  local ask_files=() sweep_aliases=() uniq_h=() uniq_d=() local_missing_d=()
  linux_why= linux_leg_args=
  if [ "$host_os" != Linux ] && [ -n "${linux_scope_err:-}" ]; then
    tests=unread tests_why="$linux_scope_err"
    return
  fi
  [ ${#scripts[@]} -gt 0 ] || [ "$linux_harness_all" = 1 ] || return 0
  if [ ${#scripts[@]} -gt 0 ] && [ -f "$tests_dir/ask" ]; then
    ask_files+=("$tests_dir/ask")
  fi
  if [ ${#renamed_scripts[@]} -gt 0 ]; then
    "${BASH:-bash}" "$root/tools/test-run.sh" affected --files "${renamed_scripts[@]}" \
      >/dev/null 2>"$tests_dir/sweep-rename-ask" </dev/null || rrc=$?
    if [ "$rrc" != 0 ] && [ "$rrc" != 3 ]; then
      tests=unread tests_why="could not query sweep-driver reach for base-renamed script(s) (${renamed_scripts[*]}; exit $rrc)"
      return
    fi
    ask_files+=("$tests_dir/sweep-rename-ask")
  fi
  for af in ${ask_files[@]+"${ask_files[@]}"}; do
    for s in ${scripts[@]+"${scripts[@]}"}; do
      while IFS= read -r line; do
        case $line in
          "affected: $s: sweep-driver reach unavailable"*)
            tests=unread tests_why="sweep-driver reach for $s could not be determined (the Dune probe did not run)"
            return ;;
          "affected: $s: reached by sweep driver alias "*)
            a=${line#"affected: $s: reached by sweep driver alias "}
            sweep_aliases+=("$a") ;;
        esac
      done <"$af"
    done
  done
  if [ ${#sweep_aliases[@]} -gt 0 ]; then
    sorted_d=$(printf '%s\n' "${sweep_aliases[@]}" | env LC_ALL=C sort -u)
    while IFS= read -r a; do
      [ -n "$a" ] || continue
      if ! "${BASH:-bash}" "$root/tools/owed-run.sh" valid-target --alias "$a"; then
        tests=unread tests_why="affected printed an invalid sweep-driver alias: $a"; return
      fi
      dir=${a%/*} dir=${dir#@} name=${a##*/}
      if [ "$dir" != "$last_dir" ]; then
        if ! dune show aliases --root . "$dir" >"$tests_dir/sweep-aliases" 2>&1 </dev/null; then
          tests=unread tests_why="Dune could not list sweep-driver aliases in $dir"; return
        fi
        last_dir=$dir
      fi
      if ! grep -qxF "$name" "$tests_dir/sweep-aliases"; then
        tests=unread tests_why="affected named a sweep-driver alias this tree does not have: $a"; return
      fi
      uniq_d+=("$a")
    done <<EOF
$sorted_d
EOF
  fi
  if [ "$host_os" = Linux ]; then
    for a in ${uniq_d[@]+"${uniq_d[@]}"}; do
      if [ "$tests" = list ] && grep -qxF -- "$a" "$tests_dir/list"; then
        continue
      fi
      local_missing_d+=("$a")
    done
    [ ${#local_missing_d[@]} -gt 0 ] || return 0
    for a in "${local_missing_d[@]}"; do
      needs+=("linux:$a")
      linux_leg_args+="${linux_leg_args:+ }--dune $a"
    done
    linux_why="the change reaches sweep-driver alias(es) omitted from this Linux build's tests tier"
    return
  fi
  if [ "$linux_harness_all" = 0 ] && [ ${#linux_harness_members[@]} -gt 0 ]; then
    sorted_h=$(printf '%s\n' "${linux_harness_members[@]}" | env LC_ALL=C sort -u)
    while IFS= read -r m; do
      [ -n "$m" ] || continue
      if "${BASH:-bash}" "$root/tools/owed-run.sh" valid-target --member "$m"; then
        uniq_h+=("$m")
      else
        linux_harness_all=1; uniq_h=(); break
      fi
    done <<EOF
$sorted_h
EOF
    if [ "$linux_harness_all" = 0 ] && [ ${#uniq_h[@]} -gt 0 ]; then
      # If the base added a harness member that does not exist at $head, --harness MEMBER
      # would be unrunnable at $head; collapse to --all-harnesses (which covers every member).
      for m in "${uniq_h[@]}"; do
        if git -C "$root" cat-file -e "HEAD:$m" 2>/dev/null &&
           ! git -C "$root" cat-file -e "$head:$m" 2>/dev/null; then
          linux_harness_all=1
          uniq_h=()
          break
        fi
      done
    fi
    if [ "$linux_harness_all" = 0 ] && [ ${#uniq_h[@]} -gt 0 ]; then
      if manifest_all=$("${BASH:-bash}" "$root/tools/test-harnesses.sh" --list 2>/dev/null </dev/null) &&
         [ -n "$manifest_all" ] &&
         [ "$(printf '%s\n' "$manifest_all" | env LC_ALL=C sort -u)" = "$sorted_h" ]; then
        linux_harness_all=1
        uniq_h=()
      fi
    fi
  fi
  if [ "$linux_harness_all" = 0 ] && [ ${#uniq_h[@]} -eq 0 ] && [ ${#uniq_d[@]} -eq 0 ]; then
    return 0
  fi
  if [ "$linux_harness_all" = 1 ]; then
    needs+=(linux:harnesses)
    linux_leg_args+="${linux_leg_args:+ }--all-harnesses"
  else
    for m in ${uniq_h[@]+"${uniq_h[@]}"}; do
      needs+=("linux:member:$m")
      linux_leg_args+="${linux_leg_args:+ }--harness $m"
    done
  fi
  for a in ${uniq_d[@]+"${uniq_d[@]}"}; do
    needs+=("linux:$a")
    linux_leg_args+="${linux_leg_args:+ }--dune $a"
  done
  linux_why="the change touches shell source(s) reached by the harness census or sweep driver on non-Linux ($host_os)"
}

show_linux_leg() {
  [ -n "${linux_leg_args:-}" ] || return 0
  printf '%s: linux-shell leg: fleet-worker.sh execution leg --request <id> --rev %s --checkout <linux-clone> --worktree <linux-worktree> --cpu --wait-window -- tools/linux-shell-leg.sh %s\n' \
    "$me" "$head" "$linux_leg_args"
}

show_tiers() {
  echo "$me: format tier: tools/fmt-check.sh, its scans off (always)"
  echo "$me: compile tier: dune build @check @test/operations/scans (always)"
  if [ -n "$mb" ]; then
    echo "$me: tests tier: the per-test aliases the head's change reaches, by tools/test-run.sh prepush --since ${mb:0:8}..${head:0:8} --list (always)"
  else
    echo "$me: tests tier: UNREAD -- HEAD has no two sides with a merge base, so the tier fails"
  fi
  [ -z "$script_line" ] || echo "$me: harness tier: $script_line"
  show_harness
}

show_harness() {
  if [ "$pending" = 1 ]; then
    echo "$me: harness tier: PENDING -- $why"
  elif [ "$harness" = 1 ]; then
    echo "$me: harness tier: SELECTED -- $why"
    if [ "$scoped" = 1 ]; then
      while IFS= read -r line; do echo "$me:   $line"; done <<EOF
$reach
EOF
    fi
  else
    echo "$me: harness tier: skipped -- $why"
  fi
}

# show_cover WAS_PENDING prints what cover_tier settled of a PENDING tier: each covered script
# with the alias naming it, then the selection.
show_cover() {
  local s a
  [ "$1" = 1 ] || return 0
  while IFS=$'\t' read -r s a; do
    [ -z "$s" ] || echo "$me: harness tier: $s is named as a dependency by $a, which the tests tier runs"
  done <<EOF
$covered
EOF
  show_harness
}

find_dune() {
  if ! command -v dune >/dev/null 2>&1; then
    if ! command -v opam >/dev/null 2>&1; then
      for prefix in /opt/homebrew/bin /usr/local/bin; do
        if [ -x "$prefix/opam" ]; then PATH=$prefix:$PATH; break; fi
      done
    fi
    echo "$me: dune is not on PATH; sourcing tools/opam-env.sh"
    # shellcheck source=tools/opam-env.sh
    . "$root/tools/opam-env.sh" || exit 1
    command -v dune >/dev/null 2>&1 || { echo "$me: no dune on PATH even after opam env" >&2; exit 1; }
  fi
}

# Keep the live verdict lines stable and hold timings until the run's outcome is known.
finish_tier() { # TIER STARTED [FAILURE]
  local tier=$1 started=$2
  local elapsed=$((SECONDS - started))
  if [ "$#" = 3 ]; then
    timing_rows+=("$me: $tier tier FAILED in ${elapsed}s ($3)")
    echo "$me: $tier tier FAILED ($3)" >&2
    exit 1
  fi
  timing_rows+=("$me: $tier tier passed in ${elapsed}s")
  echo "$me: $tier tier passed"
}

finish_run() {
  local row
  if [ "$build_passed" = 1 ] || [ -z "${SHIP_PR_MERGE_BASE_SHA:-}" ] || [ -z "${SHIP_PR_MERGE_HEAD_SHA:-}" ]; then
    for row in ${timing_rows[@]+"${timing_rows[@]}"}; do printf '%s\n' "$row"; done
    if [ -n "$tests_dir" ] && [ -s "$tests_dir/harness-timings" ]; then
      cat "$tests_dir/harness-timings"
    fi
  fi
  [ -z "$tests_dir" ] || rm -rf "$tests_dir"
}

# A signal can enter EXIT with the previous command's zero status.
pass_build() {
  echo "$me: passed" || exit $?
  build_passed=1
  exit 0
}

run_compile_tier() {
  local started=$SECONDS
  dune build --root . @check @test/operations/scans || finish_tier compile "$started" "dune exit $?"
  finish_tier compile "$started"
}

# resolve_reach_cache_dir sets `reach_cache_dir` (empty when disabled) and `reach_cache_implicit`
# (1 when derived from XDG_CACHE_HOME/HOME, 0 when explicitly supplied via env).
resolve_reach_cache_dir() {
  reach_cache_dir= reach_cache_implicit=0
  if [ -n "${OCANNL_TOOL_MERGE_BUILD_REACH_CACHE+x}" ]; then
    case $OCANNL_TOOL_MERGE_BUILD_REACH_CACHE in
      '' | off | 0) return 0 ;;
      /*) reach_cache_dir=$OCANNL_TOOL_MERGE_BUILD_REACH_CACHE ;;
      *) reach_cache_dir=$caller_pwd/$OCANNL_TOOL_MERGE_BUILD_REACH_CACHE ;;
    esac
    return 0
  fi
  case ${XDG_CACHE_HOME:-} in
    /*)
      reach_cache_dir=$XDG_CACHE_HOME/ocannl/merge-build-reach
      reach_cache_implicit=1
      return 0 ;;
  esac
  case ${HOME:-} in
    /*)
      reach_cache_dir=$HOME/.cache/ocannl/merge-build-reach
      reach_cache_implicit=1 ;;
  esac
}

# reach_cache_key prints a 64-hex SHA-256 key for the current clean merge candidate and Dune
# inputs, or returns 1 to bypass the cache (disabled, non-merge, dirty checkout, unreadable
# external input, or missing dune).
reach_cache_key() {
  local dune_bin= ocamlc_bin=
  [ -n "$reach_cache_dir" ] && [ -n "$base" ] && [ -n "$head" ] && [ -n "$mb" ] || return 1
  dune_bin=$(command -v dune 2>/dev/null || true)
  [ -n "$dune_bin" ] || return 1
  ocamlc_bin=$(command -v ocamlc 2>/dev/null || true)
  perl -e '
    use strict; use warnings;
    use Cwd qw(realpath);
    use Digest::SHA ();
    use File::Basename qw(dirname);

    my ($root, $base, $head, $mb, $tests_max, $host_os, $host_arch,
        $dune_bin, $ocamlc_bin, $linux_scope_err, $linux_harness_all, @rest) = @ARGV;

    sub take_list {
      my $n = shift @rest;
      exit 1 unless defined $n && $n =~ /\A[0-9]+\z/ && @rest >= $n;
      return splice @rest, 0, $n;
    }
    my @scripts = take_list();
    my @renamed_scripts = take_list();
    my @linux_members = take_list();
    exit 1 if @rest;

    sub git_capture {
      my (@args) = @_;
      open my $fh, "-|", "git", "-C", $root, @args or exit 1;
      binmode $fh;
      local $/;
      my $out = <$fh>;
      close $fh or return (0, "");
      return (1, defined $out ? $out : "");
    }

    my ($ok_tree, $tree) = git_capture("rev-parse", "--verify", "--quiet", "HEAD^{tree}");
    exit 1 unless $ok_tree;
    $tree =~ s/\r?\n\z//;
    exit 1 unless $tree =~ /\A[0-9a-f]{40}\z/;

    # Refuse octopus merges (more than two parents).
    my ($has_p3) = git_capture("rev-parse", "--verify", "--quiet", "HEAD^3^{commit}");
    exit 1 if $has_p3;

    # Working tree and index must match HEAD^{tree} cleanly (no staged, modified, or untracked
    # non-ignored files, and no assume-unchanged/skip-worktree index flags that hide edits).
    my ($ok_st, $st_out) = git_capture("status", "--porcelain", "-z", "--untracked-files=normal");
    exit 1 unless $ok_st && length($st_out) == 0;
    my ($ok_ls, $ls_out) = git_capture("ls-files", "-v", "-z");
    exit 1 unless $ok_ls;
    for my $ent (split /\0/, $ls_out) {
      next unless length($ent);
      exit 1 unless $ent =~ /\AH /;
    }

    # Ignored non-directory files that affected-tests.sh copies into its probe (e.g.
    # dune-workspace, /ocannl_config). Any symlink or non-regular/unreadable ignored file bypasses.
    my ($ok_ig, $ig_out) = git_capture("ls-files", "-z", "--others", "--ignored", "--exclude-standard", "--directory");
    exit 1 unless $ok_ig;
    my @ig_files = sort grep { length($_) && !m{/\z} } split /\0/, $ig_out;

    my $sha = Digest::SHA->new(256);
    sub feed { $sha->add(join("\0", @_), "\n"); }
    sub hash_file_bytes {
      my ($path) = @_;
      open my $fh, "<", $path or exit 1;
      binmode $fh;
      my $fsha = Digest::SHA->new(256);
      $fsha->addfile($fh);
      close $fh or exit 1;
      return $fsha->hexdigest;
    }

    feed("schema", "merge-build-reach-v1");
    feed("tests_max", $tests_max);
    feed("base", $base);
    feed("head", $head);
    feed("mb", $mb);
    feed("tree", $tree);
    feed("host_os", $host_os);
    feed("host_arch", $host_arch);
    feed("linux_scope_err", $linux_scope_err);
    feed("linux_harness_all", $linux_harness_all);
    feed("scripts", scalar(@scripts), @scripts);
    feed("renamed_scripts", scalar(@renamed_scripts), @renamed_scripts);
    feed("linux_members", scalar(@linux_members), @linux_members);

    for my $rel (@ig_files) {
      my $p = "$root/$rel";
      my @lst = lstat($p);
      exit 1 unless @lst && -f _ && !-l _ && -r $p;
      feed("ignored", $rel, hash_file_bytes($p));
    }

    my %prefixes;
    sub add_prefix {
      my ($dir) = @_;
      return unless defined $dir && length($dir) && -d $dir;
      my $rp = realpath($dir);
      $prefixes{$rp} = 1 if defined $rp && length($rp);
    }

    sub fingerprint_tool {
      my ($tag, $bin) = @_;
      if (!defined $bin || !length($bin)) {
        feed("tool", $tag, "absent");
        return;
      }
      my $rp = realpath($bin);
      exit 1 unless defined $rp && -f $rp && -r $rp;
      my @st = stat($rp);
      exit 1 unless @st;
      my $content = $st[7] <= 65536 ? hash_file_bytes($rp) : "-";
      feed("tool", $tag, $rp, $st[0], $st[1], $st[7], $st[9], $content);
      if ($rp =~ m{\A(.+)/bin/[^/]+\z}) {
        my $cand = $1;
        add_prefix($cand) if -d "$cand/.opam-switch" || -d "$cand/lib";
      }
    }

    fingerprint_tool("dune", $dune_bin);
    fingerprint_tool("ocamlc", $ocamlc_bin);
    add_prefix($ENV{OPAM_SWITCH_PREFIX}) if defined $ENV{OPAM_SWITCH_PREFIX};

    sub fingerprint_ext_file {
      my ($tag, $path) = @_;
      if (!defined $path || !length($path)) {
        feed("ext", $tag, "unset");
        return;
      }
      if (!-e $path && !-l $path) {
        feed("ext", $tag, "absent");
        return;
      }
      exit 1 unless -f $path && -r $path;
      feed("ext", $tag, hash_file_bytes($path));
    }

    my %scanned_lib_dirs;
    sub scan_lib_dir {
      my ($tag, $dir) = @_;
      if (!-d $dir) {
        feed("libdir", $tag, $dir, "absent");
        return;
      }
      my $rp = realpath($dir);
      exit 1 unless defined $rp;
      if ($scanned_lib_dirs{$rp}++) {
        feed("libdir_dup", $tag, $rp);
        return;
      }
      opendir my $dh, $rp or exit 1;
      my @entries = sort grep { $_ ne "." && $_ ne ".." } readdir($dh);
      closedir $dh;
      feed("libdir", $tag, $rp, scalar(@entries));
      for my $e (@entries) {
        my $ep = "$rp/$e";
        if ($e eq "dune-package" || $e eq "findlib.conf" || $e =~ /\AMETA(?:\..+)?\z/) {
          exit 1 unless -f $ep && -r $ep;
          feed("pkgflat", $tag, $e, hash_file_bytes($ep));
          next;
        }
        next unless -d $ep;
        for my $name ("dune-package", "META") {
          my $f = "$ep/$name";
          next unless -e $f || -l $f;
          exit 1 unless -f $f && -r $f;
          feed("pkgmeta", $tag, $e, $name, hash_file_bytes($f));
        }
      }
    }

    sub scan_findlib_conf {
      my ($tag, $path) = @_;
      fingerprint_ext_file($tag, $path);
      return unless defined $path && length($path) && -f $path && -r $path;
      my $conf_rp = realpath($path);
      exit 1 unless defined $conf_rp;
      my $conf_dir = dirname($conf_rp);
      open my $fh, "<", $conf_rp or exit 1;
      binmode $fh;
      local $/;
      my $body = <$fh>;
      close $fh or exit 1;
      return unless defined $body;
      my $i = 0;
      while ($body =~ /^\s*path\s*=\s*"([^"]*)"/mg) {
        for my $d (split /:/, $1, -1) {
          $i++;
          next unless length($d);
          $d = "$conf_dir/$d" unless $d =~ m{\A/};
          scan_lib_dir("$tag:path:$i", $d);
        }
      }
    }

    for my $pref (sort keys %prefixes) {
      feed("switch_prefix", $pref);
      for my $sf ("switch-state", "switch-config", "environment") {
        fingerprint_ext_file("opam:$pref:$sf", "$pref/.opam-switch/$sf");
      }
      scan_lib_dir("switch:$pref", "$pref/lib");
      scan_findlib_conf("switch_findlib:$pref", "$pref/lib/findlib.conf") if -e "$pref/lib/findlib.conf" || -l "$pref/lib/findlib.conf";
    }

    if (defined $ENV{OCAMLPATH} && length($ENV{OCAMLPATH})) {
      my $i = 0;
      for my $dir (split /:/, $ENV{OCAMLPATH}, -1) {
        $i++;
        next unless length($dir);
        scan_lib_dir("ocamlpath:$i", $dir);
      }
    }

    scan_findlib_conf("ocamlfind_conf", $ENV{OCAMLFIND_CONF});
    if (defined $ENV{DUNE_WORKSPACE} && length($ENV{DUNE_WORKSPACE})) {
      my $ws = $ENV{DUNE_WORKSPACE};
      $ws = "$root/$ws" unless $ws =~ m{\A/};
      fingerprint_ext_file("dune_workspace_env", $ws);
    } else {
      feed("ext", "dune_workspace_env", "unset");
    }

    my $dune_user_cfg;
    if (defined $ENV{XDG_CONFIG_HOME} && $ENV{XDG_CONFIG_HOME} =~ m{\A/}) {
      $dune_user_cfg = "$ENV{XDG_CONFIG_HOME}/dune/config";
    } elsif (defined $ENV{HOME} && $ENV{HOME} =~ m{\A/}) {
      $dune_user_cfg = "$ENV{HOME}/.config/dune/config";
    }
    fingerprint_ext_file("dune_user_config", $dune_user_cfg);

    my %keep_exact = map { $_ => 1 } qw(
      OPAM_SWITCH_PREFIX OPAMSWITCH OPAMROOT OCAMLFIND_CONF OCAMLPATH
      CAML_LD_LIBRARY_PATH OCAMLPARAM INSIDE_DUNE
      OCANNL_TOOL_AFFECTED_PROBE OCANNL_TOOL_AFFECTED_ARGS
    );
    my %skip_dune = map { $_ => 1 } qw(
      DUNE_BUILD_DIR DUNE_ROOT DUNE_TRACE
      DUNE_CACHE DUNE_CACHE_ROOT DUNE_CACHE_STORAGE_MODE DUNE_CACHE_TRANSPORT
    );
    for my $k (sort keys %ENV) {
      my $keep = 0;
      if ($keep_exact{$k}) {
        $keep = 1;
      } elsif ($k =~ /\ADUNE_/ && !$skip_dune{$k}) {
        $keep = 1;
      } elsif ($k =~ /\AOCANNL_/ && $k !~ /\AOCANNL_TOOL_/) {
        $keep = 1;
      }
      feed("env", $k, $ENV{$k}) if $keep;
    }

    print $sha->hexdigest, "\n";
  ' "$root" "$base" "$head" "$mb" "$TESTS_MAX" "$host_os" "$host_arch" \
    "$dune_bin" "$ocamlc_bin" "${linux_scope_err:-}" "$linux_harness_all" \
    "${#scripts[@]}" ${scripts[@]+"${scripts[@]}"} \
    "${#renamed_scripts[@]}" ${renamed_scripts[@]+"${renamed_scripts[@]}"} \
    "${#linux_harness_members[@]}" ${linux_harness_members[@]+"${linux_harness_members[@]}"} \
    2>/dev/null
}

# reach_cache_load KEY restores a validated cached `read_tests` result into `$tests_dir` and the
# shell state variables, returning 0 on a valid hit and 1 on any miss, permission/symlink refusal,
# key mismatch or format violation.
reach_cache_load() {
  local key=$1 line tag val
  [ -n "$reach_cache_dir" ] || return 1
  perl -e '
    use strict; use warnings;
    use Fcntl qw(O_RDONLY);

    my ($cache_dir, $implicit, $key, $tests_dir) = @ARGV;
    exit 1 unless defined $key && $key =~ /\A[0-9a-f]{64}\z/;

    my @dst = lstat($cache_dir);
    exit 1 unless @dst;
    if ($implicit) {
      exit 1 unless -d _ && !-l _ && $dst[4] == $< && (($dst[2] & 0022) == 0);
    } else {
      exit 1 unless -d $cache_dir;
    }

    my $entry = "$cache_dir/reach-$key.v1";
    my @lst = lstat($entry);
    exit 1 unless @lst && -f _ && !-l _ && $lst[4] == $< && (($lst[2] & 0022) == 0);

    my $flags = O_RDONLY;
    my $nofollow = eval { Fcntl::O_NOFOLLOW() };
    $flags |= $nofollow if defined $nofollow;
    sysopen(my $fh, $entry, $flags) or exit 1;
    binmode $fh;
    my @fst = stat($fh);
    exit 1 unless @fst && -f _ && $fst[0] == $lst[0] && $fst[1] == $lst[1]
               && $fst[4] == $< && (($fst[2] & 0022) == 0);

    local $/;
    my $raw = <$fh>;
    close $fh or exit 1;
    exit 1 unless defined $raw && length($raw) > 0 && index($raw, "\0") < 0;
    exit 1 unless $raw =~ /\n\z/;

    my @lines = split /\n/, $raw, -1;
    pop @lines;
    exit 1 unless @lines >= 3 && $lines[0] eq "schema\tv1" && $lines[1] eq "key\t$key"
               && $lines[-1] eq "end\tv1";

    my %scalar_tags = map { $_ => 1 } qw(
      tests tests_verdict tests_incomplete has_ask_first
      tests_owed train_why linux_why linux_leg_args
    );
    my %scalars;
    my (@needs, @list, @cross, @deps);

    for my $i (2 .. $#lines - 1) {
      my $ln = $lines[$i];
      my $tab = index($ln, "\t");
      exit 1 if $tab < 1;
      my $tag = substr($ln, 0, $tab);
      my $val = substr($ln, $tab + 1);
      if ($scalar_tags{$tag}) {
        exit 1 if exists $scalars{$tag};
        $scalars{$tag} = $val;
      } elsif ($tag eq "need") {
        exit 1 unless $val eq "runtest" || $val eq "train" || $val eq "linux:harnesses"
                   || $val =~ m{\A@[^\s]+/train-[^\s]+\z}
                   || $val =~ m{\Alinux:member:[^\s]+\z}
                   || $val =~ m{\Alinux:@[^\s]+\z};
        push @needs, $val;
      } elsif ($tag eq "list") {
        exit 1 unless $val =~ m{\A@[^\s]+\z};
        push @list, $val;
      } elsif ($tag eq "cross_line") {
        exit 1 unless index($val, "affected: CROSS-CUTTING reason: ") == 0;
        push @cross, $val;
      } elsif ($tag eq "dep") {
        exit 1 unless $val =~ m{\Aaffected: .+: named as a dependency by the rule of @[^\s]+\z};
        push @deps, $val;
      } else {
        exit 1;
      }
    }

    for my $req (keys %scalar_tags) {
      exit 1 unless exists $scalars{$req};
    }
    exit 1 unless $scalars{tests} eq "list" || $scalars{tests} eq "owed";
    exit 1 unless $scalars{tests_verdict} eq "ok"
               || $scalars{tests_verdict} eq "incomplete"
               || $scalars{tests_verdict} eq "cross-cutting"
               || $scalars{tests_verdict} eq "cross-cutting,incomplete";
    exit 1 unless $scalars{tests_incomplete} eq "0" || $scalars{tests_incomplete} eq "1";
    exit 1 unless $scalars{has_ask_first} eq "0" || $scalars{has_ask_first} eq "1";

    sub write_lines {
      my ($path, $arr) = @_;
      open my $oh, ">", $path or exit 1;
      binmode $oh;
      for my $x (@$arr) { print $oh $x, "\n" or exit 1; }
      close $oh or exit 1;
    }

    write_lines("$tests_dir/list", \@list);
    if ($scalars{has_ask_first} eq "1") {
      write_lines("$tests_dir/ask", \@cross);
      write_lines("$tests_dir/ask-first", \@deps);
    } else {
      unlink "$tests_dir/ask-first";
      write_lines("$tests_dir/ask", [@cross, @deps]);
    }

    open my $mh, ">", "$tests_dir/cache-meta" or exit 1;
    binmode $mh;
    for my $k (qw(tests tests_verdict tests_incomplete tests_owed train_why linux_why linux_leg_args)) {
      print $mh "$k=$scalars{$k}\n" or exit 1;
    }
    for my $n (@needs) {
      print $mh "need=$n\n" or exit 1;
    }
    close $mh or exit 1;

    # Best-effort LRU touch so actively reused entries stay fresh.
    utime(undef, undef, $entry);
  ' "$reach_cache_dir" "$reach_cache_implicit" "$key" "$tests_dir" 2>/dev/null || {
    rm -f "$tests_dir/list" "$tests_dir/ask" "$tests_dir/ask-first" "$tests_dir/cache-meta"
    return 1
  }
  tests= tests_verdict= tests_owed= tests_why= tests_incomplete=0
  needs=() train_why= linux_why= linux_leg_args=
  while IFS= read -r line; do
    tag=${line%%=*}
    val=${line#*=}
    case $tag in
      tests) tests=$val ;;
      tests_verdict) tests_verdict=$val ;;
      tests_incomplete) tests_incomplete=$val ;;
      tests_owed) tests_owed=$val ;;
      train_why) train_why=$val ;;
      linux_why) linux_why=$val ;;
      linux_leg_args) linux_leg_args=$val ;;
      need) needs+=("$val") ;;
      *)
        rm -f "$tests_dir/list" "$tests_dir/ask" "$tests_dir/ask-first" "$tests_dir/cache-meta"
        return 1 ;;
    esac
  done <"$tests_dir/cache-meta"
  rm -f "$tests_dir/cache-meta"
}

# reach_cache_store KEY atomically writes the successful `read_tests` result to the cache and
# best-effort prunes expired/excess entries. Any error is ignored so caching never fails a run.
reach_cache_store() {
  local key=$1 has_ask_first=0
  [ -n "$reach_cache_dir" ] || return 0
  [ ! -f "$tests_dir/ask-first" ] || has_ask_first=1
  perl -e '
    use strict; use warnings;
    use Fcntl qw(O_WRONLY O_CREAT O_EXCL);

    my ($cache_dir, $implicit, $key, $tests_dir, $has_ask_first,
        $tests, $tests_verdict, $tests_incomplete, $tests_owed,
        $train_why, $linux_why, $linux_leg_args, @needs) = @ARGV;

    exit 0 unless defined $key && $key =~ /\A[0-9a-f]{64}\z/;
    exit 0 unless $tests eq "list" || $tests eq "owed";
    exit 0 unless $tests_verdict eq "ok"
               || $tests_verdict eq "incomplete"
               || $tests_verdict eq "cross-cutting"
               || $tests_verdict eq "cross-cutting,incomplete";
    exit 0 unless $tests_incomplete eq "0" || $tests_incomplete eq "1";
    exit 0 unless $has_ask_first eq "0" || $has_ask_first eq "1";

    for my $s ($tests_owed, $train_why, $linux_why, $linux_leg_args, @needs) {
      exit 0 if !defined $s || index($s, "\n") >= 0 || index($s, "\0") >= 0;
    }
    for my $n (@needs) {
      exit 0 unless $n eq "runtest" || $n eq "train" || $n eq "linux:harnesses"
                 || $n =~ m{\A@[^\s]+/train-[^\s]+\z}
                 || $n =~ m{\Alinux:member:[^\s]+\z}
                 || $n =~ m{\Alinux:@[^\s]+\z};
    }

    my @list;
    if (-f "$tests_dir/list") {
      open my $lh, "<", "$tests_dir/list" or exit 0;
      binmode $lh;
      while (my $ln = <$lh>) {
        $ln =~ s/\r?\n\z//;
        next unless length($ln);
        exit 0 if index($ln, "\0") >= 0 || $ln !~ m{\A@[^\s]+\z};
        push @list, $ln;
      }
      close $lh;
    }

    my @cross;
    if (-f "$tests_dir/ask") {
      open my $ah, "<", "$tests_dir/ask" or exit 0;
      binmode $ah;
      while (my $ln = <$ah>) {
        $ln =~ s/\r?\n\z//;
        next unless index($ln, "affected: CROSS-CUTTING reason: ") == 0;
        exit 0 if index($ln, "\0") >= 0;
        push @cross, $ln;
      }
      close $ah;
    }

    my @deps;
    my $dep_file = $has_ask_first eq "1" ? "$tests_dir/ask-first" : "$tests_dir/ask";
    if (-f $dep_file) {
      open my $dh, "<", $dep_file or exit 0;
      binmode $dh;
      while (my $ln = <$dh>) {
        $ln =~ s/\r?\n\z//;
        next unless $ln =~ m{\Aaffected: .+: named as a dependency by the rule of @[^\s]+\z};
        exit 0 if index($ln, "\0") >= 0;
        push @deps, $ln;
      }
      close $dh;
    }

    # Ensure cache_dir exists; newly created directories get 0700.
    umask 0077;
    if (!-e $cache_dir) {
      my @parts = split m{/+}, $cache_dir;
      my $cur = ($cache_dir =~ m{\A/}) ? "" : ".";
      for my $p (@parts) {
        next unless length($p);
        $cur = ($cur eq "") ? "/$p" : "$cur/$p";
        if (!-e $cur) {
          mkdir($cur, 0700) or (-d $cur) or exit 0;
        }
      }
    }
    my @dst = lstat($cache_dir);
    exit 0 unless @dst;
    if ($implicit) {
      exit 0 unless -d _ && !-l _ && $dst[4] == $< && (($dst[2] & 0022) == 0);
    } else {
      exit 0 unless -d $cache_dir;
    }

    my ($fh, $tmp_path);
    for my $try (1 .. 16) {
      my $cand = sprintf("%s/.ocannl-reach-tmp.%s.%d.%06d", $cache_dir, $key, $$, int(rand(1000000)));
      if (sysopen($fh, $cand, O_WRONLY | O_CREAT | O_EXCL, 0600)) {
        $tmp_path = $cand;
        last;
      }
    }
    exit 0 unless defined $tmp_path;
    binmode $fh;
    my $ok = 1;
    $ok &&= print $fh "schema\tv1\n";
    $ok &&= print $fh "key\t$key\n";
    $ok &&= print $fh "tests\t$tests\n";
    $ok &&= print $fh "tests_verdict\t$tests_verdict\n";
    $ok &&= print $fh "tests_incomplete\t$tests_incomplete\n";
    $ok &&= print $fh "has_ask_first\t$has_ask_first\n";
    $ok &&= print $fh "tests_owed\t$tests_owed\n";
    $ok &&= print $fh "train_why\t$train_why\n";
    $ok &&= print $fh "linux_why\t$linux_why\n";
    $ok &&= print $fh "linux_leg_args\t$linux_leg_args\n";
    for my $n (@needs) { $ok &&= print $fh "need\t$n\n"; }
    for my $l (@list) { $ok &&= print $fh "list\t$l\n"; }
    for my $c (@cross) { $ok &&= print $fh "cross_line\t$c\n"; }
    for my $d (@deps) { $ok &&= print $fh "dep\t$d\n"; }
    $ok &&= print $fh "end\tv1\n";
    $ok &&= close $fh;
    if (!$ok || !rename($tmp_path, "$cache_dir/reach-$key.v1")) {
      unlink $tmp_path;
      exit 0;
    }

    # Best-effort retention: prune only owner-matching files in this tool'\''s exact staging and
    # entry filename formats (stale staging >1h, entries >14d, cap at 256 entries).
    if (opendir my $cdh, $cache_dir) {
      my $now = time();
      my @entries;
      while (my $name = readdir($cdh)) {
        if ($name =~ /\A\.ocannl-reach-tmp\.[0-9a-f]{64}\.[0-9]+\.[0-9]{6}\z/) {
          my @st = lstat("$cache_dir/$name");
          unlink "$cache_dir/$name" if @st && -f _ && !-l _ && $st[4] == $< && ($now - $st[9]) > 3600;
        } elsif ($name =~ /\Areach-[0-9a-f]{64}\.v1\z/) {
          my @st = lstat("$cache_dir/$name");
          next unless @st && -f _ && !-l _ && $st[4] == $<;
          if (($now - $st[9]) > 14 * 86400) {
            unlink "$cache_dir/$name";
          } else {
            push @entries, [$st[9], $name];
          }
        }
      }
      closedir $cdh;
      if (@entries > 256) {
        @entries = sort { $b->[0] <=> $a->[0] } @entries;
        for my $i (256 .. $#entries) {
          unlink "$cache_dir/$entries[$i][1]";
        }
      }
    }
  ' "$reach_cache_dir" "$reach_cache_implicit" "$key" "$tests_dir" "$has_ask_first" \
    "$tests" "$tests_verdict" "$tests_incomplete" "$tests_owed" \
    "$train_why" "$linux_why" "$linux_leg_args" ${needs[@]+"${needs[@]}"} \
    2>/dev/null || true
}

read_tests() {
  local cache_key= post_key=
  reach_reused=0
  resolve_reach_cache_dir
  if cache_key=$(reach_cache_key); then
    if reach_cache_load "$cache_key"; then
      reach_reused=1
      return 0
    fi
  fi
  ask_tests
  if [ "$tests" = unread ]; then
    [ ! -s "$tests_dir/ask" ] || cat "$tests_dir/ask" >&2
    return 1
  fi
  owed_needs
  if [ "$tests" = unread ]; then
    [ ! -s "$tests_dir/train-ask" ] || cat "$tests_dir/train-ask" >&2
    return 1
  fi
  linux_shell_needs
  if [ "$tests" = unread ]; then
    [ ! -s "$tests_dir/ask" ] || cat "$tests_dir/ask" >&2
    return 1
  fi
  # Revalidate the key after computing reach so mid-query mutations never publish under stale inputs.
  if [ -n "$cache_key" ] && post_key=$(reach_cache_key) && [ "$post_key" = "$cache_key" ]; then
    reach_cache_store "$cache_key"
  fi
}

show_cross_reason() {
  local reason
  case $tests_verdict in cross-cutting | cross-cutting,incomplete) ;; *) return 0 ;; esac
  reason=$(sed -n 's/^affected: CROSS-CUTTING reason: //p' "$tests_dir/ask")
  # Pass LC_ALL=C through env, not a shell prefix assignment, so a forked subshell (merged_plan)
  # never triggers Homebrew Bash's dispose_temporary_env -> libintl_setlocale path (gh-ocannl-2462).
  if [ -n "$reason" ] && env LC_ALL=C grep -qx '[[:print:]][[:print:]]*' <<<"$reason" &&
     [ "$(grep -c '^affected: CROSS-CUTTING reason: ' "$tests_dir/ask")" = 1 ]; then
    printf '%s: tests tier: CROSS-CUTTING reason: %s\n' "$me" "$reason"
  fi
}

check_record() {
  "${BASH:-bash}" "$root/tools/owed-run.sh" check "$head" "${needs[@]}"
}

die_plan() { printf '%s: plan: %s\n' "$me" "$*" >&2; exit 1; }

branch_plan() (
  base=$(git -C "$root" rev-parse --verify --quiet --end-of-options "${plan_base}^{commit}") ||
    die_plan "base revision not found: $plan_base"
  head=$(git -C "$root" rev-parse --verify HEAD) || die_plan 'cannot resolve HEAD'
  [ "$base" != "$head" ] || die_plan 'base and HEAD are the same commit'
  echo "$me: plan: committed head $head against base $base ($plan_base)"
  git_vars=$(git -C "$root" rev-parse --local-env-vars) || die_plan 'cannot read Git environment names'
  # The candidate repository must not inherit a caller's alternate index or gitdir.
  # shellcheck disable=SC2086
  unset $git_vars
  unset DUNE_BUILD_DIR DUNE_ROOT
  scratch_parent=$(cd "${TMPDIR:-/tmp}" && pwd -P) || die_plan 'cannot resolve temporary directory'
  if git -C "$scratch_parent" rev-parse --git-dir >/dev/null 2>&1; then
    die_plan 'temporary directory is inside a Git checkout; set TMPDIR outside it'
  fi
  scratch=$(mktemp -d "$scratch_parent/merge-build-plan.XXXXXX") || die_plan 'cannot make a scratch directory'
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  trap 'rm -rf "$scratch"' EXIT
  git clone --shared --no-checkout --quiet -- "$root" "$scratch/checkout" ||
    die_plan 'cannot create the candidate repository'
  cd "$scratch/checkout" || die_plan 'cannot enter the candidate repository'
  rc=0
  tree=$(git merge-tree --write-tree "$base" "$head") || rc=$?
  case $rc in
    0) ;;
    1) printf '%s\n' "$tree" >&2; die_plan 'merge conflict' ;;
    *) die_plan "git merge-tree exited $rc" ;;
  esac
  commit=$(git -c user.name=merge-build-plan -c user.email=merge-build-plan@example.invalid \
    commit-tree "$tree" -p "$base" -p "$head" -m 'Candidate merge for the merge-build plan') ||
    die_plan 'cannot create the candidate commit'
  git -c core.hooksPath=/dev/null checkout --quiet --detach "$commit" ||
    die_plan 'cannot check out the candidate merge'
  # A local opam switch is visible from the original checkout only.
  cd "$root" || die_plan 'cannot enter the head checkout'
  find_dune
  cd "$scratch/checkout" || die_plan 'cannot enter the candidate repository'
  set -- --plan-merge
  [ "$all" = 0 ] || set -- "$@" --all
  rc=0
  resolve_reach_cache_dir
  # Affected retains rejected Dune output in TMPDIR for inspection.
  if [ -n "${OCANNL_TOOL_MERGE_BUILD_REACH_CACHE+x}" ] && [ -n "$reach_cache_dir" ]; then
    env OCANNL_TOOL_MERGE_BUILD_REACH_CACHE="$reach_cache_dir" TMPDIR="$scratch_parent" \
      "${BASH:-bash}" tools/merge-build-check.sh "$@" </dev/null || rc=$?
  else
    TMPDIR="$scratch_parent" "${BASH:-bash}" tools/merge-build-check.sh "$@" </dev/null || rc=$?
  fi
  [ "$rc" != 2 ] || die_plan 'the candidate merge does not support plan queries'
  exit "$rc"
)

merged_plan() (
  cd "$root" || exit 1
  read_parents
  tests_dir=$(mktemp -d "${TMPDIR:-/tmp}/merge-build-query.XXXXXX") || die_plan 'cannot make a query directory'
  trap 'rm -rf "$tests_dir"' EXIT
  select_tiers
  show_tiers
  find_dune
  if ! read_tests; then die_plan "tests tier unreadable: $tests_why"; fi
  was_pending=$pending
  cover_tier
  show_cover "$was_pending"
  [ "$reach_reused" = 0 ] || echo "$me: tests plan: reused cached reach"
  case $tests_verdict/$tests/$tests_incomplete in
    cross-cutting/list/0) echo "$me: tests plan: CROSS-CUTTING (first tier selected)" ;;
    cross-cutting/list/1) echo "$me: tests plan: CROSS-CUTTING (first tier selected, INCOMPLETE)" ;;
    cross-cutting/owed/*) echo "$me: tests plan: CROSS-CUTTING (no first tier selected)" ;;
    incomplete/*) echo "$me: tests plan: INCOMPLETE" ;;
    ok/owed/*) echo "$me: tests plan: narrow, over the $TESTS_MAX-alias merge build limit" ;;
    ok/list/*) echo "$me: tests plan: narrow" ;;
  esac
  show_cross_reason
  if [ ${#needs[@]} -eq 0 ]; then
    echo "$me: owed-run needs: none"
    echo 'ship-pr-merge-plan: nothing-owed'
  else
    exec 3>"$tests_dir/owed" || die_plan 'cannot open record-check output file'
    rc=0
    check_record >&3 2>&3 </dev/null || rc=$?
    exec 3>&-
    needs_text=$(IFS=' '; printf '%s' "${needs[*]}")
    case $rc in
      0) echo "$me: owed-run needs: $needs_text (record admits)" ;;
      1) echo "$me: owed-run needs: $needs_text (record does not admit; owed-run.sh exit $rc)" ;;
      3) echo "$me: owed-run needs: $needs_text (record unreadable; owed-run.sh exit $rc)" ;;
      *) echo "$me: owed-run needs: $needs_text (record check failed; owed-run.sh exit $rc)" ;;
    esac
    cat "$tests_dir/owed" || die_plan 'cannot read record-check output'
    show_linux_leg
    case $rc in
      0) echo "ship-pr-merge-plan: admitted $needs_text" ;;
      1)
        [ -s "$tests_dir/owed" ] || die_plan 'owed-run.sh check exited 1 with no output'
        echo "ship-pr-merge-plan: unadmitted $needs_text" ;;
      3) die_plan "owed-run.sh check could not read the record repository (exit $rc)" ;;
      *) die_plan "owed-run.sh check failed (exit $rc)" ;;
    esac
  fi
)

if [ -n "$plan_base" ]; then
  branch_plan
  exit $?
fi
if [ "$plan_merge" = 1 ]; then
  merged_plan
  exit $?
fi
tests_dir= timing_rows=() build_passed=0
trap finish_run EXIT
if [ "$compile_only" = 1 ]; then
  read_parents
  echo "$me: compile tier: dune build @check @test/operations/scans (--compile-only)"
  find_dune
  cd "$root" || exit 1
  run_compile_tier
  pass_build
fi
started=$SECONDS
tests_dir=$(mktemp -d "${TMPDIR:-/tmp}/merge-build-check.XXXXXX") ||
  finish_tier tests "$started" 'cannot make a scratch directory'
read_parents
select_tiers
show_tiers
[ "$plan" = 0 ] || exit 0

find_dune
cd "$root" || exit 1
started=$SECONDS
OCANNL_TOOL_FMT_CHECK_SCANS=0 tools/fmt-check.sh </dev/null ||
  finish_tier format "$started" "tools/fmt-check.sh exit $?"
finish_tier format "$started"
run_compile_tier
started=$SECONDS
read_tests || finish_tier tests "$started" "$tests_why"
[ "$reach_reused" = 0 ] || timing_rows+=("$me: tests reach: reused cached result")
if [ ${#needs[@]} -gt 0 ]; then
  [ -z "$tests_owed" ] || echo "$me: tests tier: OWED -- $tests_owed did not run here"
  [ "$train_why" = '' ] || echo "$me: tests tier: OWED -- $train_why"
  [ "$linux_why" = '' ] || echo "$me: tests tier: OWED -- $linux_why"
  rc=0
  check_record >"$tests_dir/owed" 2>&1 </dev/null || rc=$?
  while IFS= read -r line; do [ -z "$line" ] || echo "$me:   $line"; done <"$tests_dir/owed"
  if [ "$rc" != 0 ]; then
    show_cross_reason
    show_linux_leg
    # The merged tree, which ship-pr's build of the base alone (the base's tree) never prints: a
    # missing record is the merge's own failure, never a shared red `--override` would pass.
    finish_tier tests "$started" "no owed-run record for ${head:0:8} admits the merged tree $(git -C "$root" rev-parse 'HEAD^{tree}'): record the run with tools/owed-run.sh record, or waive it with tools/owed-run.sh waive; tools/owed-run.sh exit $rc"
  fi
fi
if [ "$tests" = list ] && [ -s "$tests_dir/list" ]; then
  targets=()
  while IFS= read -r line; do [ -z "$line" ] || targets+=("$line"); done <"$tests_dir/list"
  dune build --root . -j "$TEST_JOBS" "${targets[@]}" ||
    finish_tier tests "$started" "dune exit $?"
elif [ "$tests" = list ]; then
  echo "$me: tests tier: the change reaches no per-test alias"
fi
finish_tier tests "$started"
was_pending=$pending
cover_tier
show_cover "$was_pending"
if [ "$harness" = 1 ]; then
  started=$SECONDS
  timing_file=$tests_dir/harness-timings
  rc=0
  if [ "$scoped" = 1 ]; then
    OCANNL_TOOL_HARNESS_TIMINGS_FILE=$timing_file \
      tools/test-harnesses.sh --jobs "$SHELL_JOBS" --affected --files "${asked[@]}" </dev/null || rc=$?
    if [ "$rc" = 2 ]; then
      # Every refusal comes before a member starts, so the whole tier still fits the bound.
      echo "$me: harness tier: the scoped run was refused (exit 2), no member run; running the whole tier"
      scoped=0 rc=0
    fi
  fi
  [ "$scoped" = 1 ] || OCANNL_TOOL_HARNESS_TIMINGS_FILE=$timing_file \
    tools/test-harnesses.sh --jobs "$SHELL_JOBS" </dev/null || rc=$?
  [ "$rc" = 0 ] || finish_tier harness "$started" "tools/test-harnesses.sh exit $rc"
  finish_tier harness "$started"
fi
pass_build

#!/usr/bin/env bash
# Tests for tools/test-run.sh: the shared `group_alive` predicate its callers ask "is anything in this
# process group still running?", the sentences `stop` prints about a surviving
# process group (gh-ocannl-742), the read-only queries, and the launchers'
# options, targeted batches and `prepush`.
#
#   tools/test-test-run-options.sh          # run every leg of this member
#   tools/test-test-run-options.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-options.sh --only '^94:'   # only the legs whose names match
#   tools/test-test-run-options.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #group-alive (1, 2-3, 4, 5), #stop (6-8, 67, 101), #queries (39), #options (26-28, 37, 89, 103, 94, 95, 96, 98), #hermetic (70).
#
# Legs:
#   1. extraction -- the functions came out of the shipping script.
#  1b. each Perl program the script embeds (every NAME_perl='...' in it and in
#      the files it sources) holds no single quote and compiles under
#      `perl -c`; a lone quote, a quoted word and an open brace planted in a
#      copy each fail it (gh-ocannl-1945).
#   2. a genuinely live group reads ALIVE (the other side of leg 4: without
#      this, a group_alive that answered "dead" always would pass leg 4).
#   3. a group with no members at all reads DEAD, and so does the bare signal.
#   4. a group holding nothing but a ZOMBIE reads dead, where the bare
#      `kill -0 -- -PGID` this replaced reads it as alive. That misreading is
#      what made `stop` able to announce an orphaned group holding only corpses.
#   5. a pgid that is not a positive decimal integer is refused -- 0 and
#      negatives are kill specials (caller's own group, broadcast).
#   6. `stop` on a group whose leader IGNORES TERM says so and escalates --
#      and the escalation kills the whole group. Every fixture group holds two
#      processes, or a leader-only kill would pass for a group kill.
#   7. `stop` on a group whose leader exits on TERM says the TERM went out and
#      asks for a re-run, rather than claiming the group ignored it.
#   8. `stop` on a run recorded by the version that kept its lock BESIDE the
#      worktree (no `runs` root on record) still attributes a leftover holding
#      that in-tree lock to the run, reaps it, and says so.
#  26. an option written AFTER the dune arguments is refused, having run
#      nothing -- the misplacement dune would otherwise digest as a red run.
#  27. the same option in its documented place is consumed, not forwarded.
#  28. past dune's own `--` the same word is a program argument, untouched.
#  37. the `--cap` guard still refuses BEFORE dune is spawned: the two exit-2
#      refusals are told apart by whether dune was ever invoked.
#  39. read-only paths and lock-status: absent state, invalid input, physical
#      paths, recorded and legacy paths, held/released and uninspectable locks.
#  89-90 sit after leg 79 (gh-ocannl-1448).
#  89. a `dune build` naming an alias dune does not know is ALIAS REFUSED
#      (exit 2) before the backends' resolution, the slot and dune's build --
#      under @ and @@, in its own directory or below it, at the root, and a
#      directory dune does not know beside a known one; known ones run, and
#      nothing past dune's `--` is asked about; `plan` says run would refuse
#      it and `repeat` runs no iteration; a listing that is not dune's leaves
#      the aliases to dune; and where a real dune is at hand, its own listing.
#      Where dune is made to color (CLICOLOR_FORCE=1), its escaped `Error`
#      tag reads as the plain one, from the fixture and from a real dune.
# 103. a named test's sibling goldens: `plan`, the launch and the digest name
#      each alias `runtest-NAME-VARIANT` with a golden `NAME-VARIANT[.*].expected`
#      of its own in the test's directory, under any per-test family, in the
#      directory a recursive operand found it; the batch reaches dune as given.
#      A variant with no golden, one the batch names, a variant named by itself
#      and an argv whose listing is not read say nothing, and mutants cutting
#      the golden test, its `.` or the batch's own names turn those probes red.
#  94. targeted batches (gh-ocannl-1740): `--targets-file` hands dune the
#      listed aliases after its own words (comments, blank lines, CRLF and
#      padding skipped; a relative file read from the caller's directory;
#      `build` by default; repeatable; before dune's `--`), and refuses, dune
#      never invoked, an empty list, a line that is not one `@` operand, a
#      missing file, a subcommand other than build, the option after the dune
#      arguments; repeat (plain, and --direct through fresh's operand check)
#      and bisect read the file too, and bisect wants a target from one side
#      (gh-ocannl-1797); `--cpu`/`--gpu` are refused pointing at the
#      slot; a refused `runtest-NAME` names the directory's `slow-NAME`, and a
#      refusal with no sibling names none; `aliases PATH...` lists a test's
#      aliases by family from dune's listing (a source, a golden, a variant, a
#      qualified golden by its longest test prefix, a directory, a dune file; a helper and an unknown directory exit 1 with
#      the rest listed; outside the worktree exit 2), and its output read
#      back as a targets file is the batch it lists.
#  95. value options take `--opt V` and `--opt=V` alike in every parser (run's
#      cap, targets file and backend check, plan's test cap, repeat's two caps,
#      wait's timeout, bisect's caps and builds), an empty value is refused,
#      and `--cpu`/`--gpu` are refused with the slot's spelling by run, plan,
#      fresh, repeat and bisect, before the dune arguments or after them, but
#      not past dune's `--` (gh-ocannl-1740, gh-ocannl-1734); a mutant cutting
#      the `=` arms, and one cutting the slot-kind arms, turns every probe of
#      its set red.
#  96. a flag refuses a value by name (`repeat --alone=1`, `--direct=1`,
#      `bisect --no-first-parent=x`, before GOOD and BAD or after), and a
#      query (status, wait, stop, paths, lock-status) refuses an option-shaped
#      word as unknown instead of looking it up as a run (`wait --timout=60
#      last`); a second run, an option repeat does not take and list's
#      arguments are refused, while a run's directory, `last` and wait's
#      timeout in either spelling still answer; a mutant cutting opt_flag's
#      refusal, and one cutting take_run_ref's option arm, turns every probe
#      of its set red. The launchers (run, start, plan, fresh) refuse an
#      unknown leading option by name before dune (gh-ocannl-1797), and a
#      leading -j with the order to write it in, except fresh's, which reaches
#      its operands; a mutant cutting that arm turns its probes red.
#  98. `prepush --since BASE` (gh-ocannl-1829), against a stand-in
#      `affected`: its list's runtest- members reach dune as run's batch, in
#      order (a root alias, a NAME holding `slow-`, a `train-` directory kept),
#      its slow- and train- members left off and the slow- ones named; run
#      options pass through; --list prints the list and runs nothing (and exits
#      3, still printing, on an incomplete list); usage errors run nothing and
#      ask no affected; an affected failure, a cross-cutting verdict (either
#      form), an incomplete one, and a line of no known family run nothing; a
#      list with no runtest- member runs nothing, exit 0; no scratch list
#      outlives any of them. The verdict is affected's trailer, the last line
#      of its stderr (gh-ocannl-1921), each of the four pinned: its prose alone
#      decides nothing either way, and a stderr with no trailer, two, one that
#      is not last, an unknown one, or one at odds with the exit status runs
#      nothing; a mutant reading the prose instead turns those probes red.
#      The harness commands `affected --harnesses --since BASE` prints
#      (gh-ocannl-1929) are named as the last lines of stderr and never run,
#      the very last naming the one command that runs them for this BASE:
#      after a run, after --list's list, after a cross-cutting or incomplete
#      refusal and a list with nothing to run, the exit status unchanged; a
#      change reaching none (an `.ml`-only diff) gets no line, and a failed
#      ask runs nothing; a mutant keeping no command, and one closing with
#      another BASE, turn each naming probe red.
# 101. one rendering of a start token (gh-ocannl-1944): ps_token, the launch
#      chain's bulk join and census_token, read from the script under test,
#      render one string for this shell and each ancestor; a ps_token or a
#      bulk join that renders differently is told apart (the negative
#      controls); and the script holds no renderer beside them.
#  67 sits after leg 8: a lock whose acquirer exited, held through the
#      description it shared, is found by `stop`'s census and reaped, though
#      /proc/locks names only the dead acquirer, or nobody (gh-ocannl-1107).
#  67b. its negative control: the census going by that name leaves the holder
#      running. Not applicable where /proc holds neither lock table (macOS):
#      the census reads lsof's openers there, and no name (gh-ocannl-1930).
#  67c. that probe's own control: only a /proc observed without both tables
#      makes 67b not applicable.

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"


# The decision behind this issue is one definition, not two copies tested in
# parallel. Count definitions across the shared helper and both production
# callers so either copy coming back fails the harness that CI runs.
leg_1() {
group_definition_count() { awk '/^group_alive\(\)/ { n++ } END { print n + 0 }' "$@"; }
one_group_definition() { [ "$(group_definition_count "$@")" = 1 ]; }
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC"; then
  report 0 "group_alive has one shared production definition"
else
  report 1 "group_alive has one shared production definition" \
    "found $(group_definition_count "$GROUP_SRC" "$SRC" "$HOOK_SRC") definitions across $GROUP_SRC, $SRC and $HOOK_SRC"
fi
# Synthetic duplicate: the clean production tree alone cannot distinguish a
# working uniqueness check from one that always answers yes.
cp "$GROUP_SRC" "$TMP/duplicate-process-group.sh"
if one_group_definition "$GROUP_SRC" "$SRC" "$HOOK_SRC" "$TMP/duplicate-process-group.sh"; then
  report 1 "the shared-definition check rejects a duplicate" \
    "the production definition plus a copied definition still read as unique"
else
  report 0 "the shared-definition check rejects a duplicate"
fi
}
harness_leg_run '1: group_alive has one shared production definition #group-alive' leg_1

# A copy of the script with whole lines changed, for the negative controls of
# legs 1b and 101.
token_mutant() { # <out> [FROM TO]...: $SRC with each whole line FROM replaced by TO; 0 iff each FROM was one line
  perl -e '
    my ($src, $out, %to) = @ARGV;
    my %n;
    open(my $in, "<", $src) or exit 1;
    open(my $o, ">", $out) or exit 1;
    while (my $l = <$in>) {
      chomp $l;
      if (exists $to{$l}) { $n{$l}++; $l = $to{$l} }
      print $o "$l\n";
    }
    close $o or exit 1;
    exit((grep { ($n{$_} || 0) != 1 } keys %to) ? 1 : 0);
  ' "$SRC" "$@"
}

# ---------------------------------------------------------------------------
# Leg 1b: each embedded Perl program compiles (gh-ocannl-1945)
# ---------------------------------------------------------------------------
# The script carries its Perl programs as single-quoted assignments,
# `NAME_perl='` through a closing `'` on a line of its own, and `bash -n`
# reads none of that text: a `'` landing in one closes the string where it
# stands. So each program -- every NAME_perl assigned at a line's start in the
# script and in the files it sources, the list derived as stage_sourced
# derives its own -- is held to two things. Its text holds no `'` at all: the
# one check that sees a PAIR of them, which leaves a script `bash -n` accepts
# and, where the unquoted word still reads as Perl, a program `perl -c`
# accepts too. And the value the shell assigns compiles under `perl -c`, the
# assignment evaluated in a subshell only once it is known to hold no quote,
# so a program cut short never runs as shell. A program another one is
# prepended to (supervisor_perl opens with $lock_census_perl) is compiled as
# the shell joins them, so it fails with its prefix. The names the script
# expands are the other side: each must be one of the programs checked, so an
# assignment respelled out of the scan's reach fails here and is not left out.
# The negative controls are the script with one line of lock_census_perl
# changed: a lone quote, a quoted word (which `bash -n` must still accept, or
# the control shows nothing bash could not), and a brace left open with no
# quote in sight (`perl -c` alone can fail that one).
perl_program_rows() { # <script>: "NAME<TAB>STATE<TAB>PREFIX" per NAME_perl assigned at a line's start, in file order
  awk -v q="'" '
    function row() { if (name != "") print name "\t" state "\t" prefix; name = "" }
    /^[A-Za-z_][A-Za-z0-9_]*_perl=/ {
      if (name != "") { state = "unclosed"; row() }
      name = $0; sub(/=.*/, "", name)
      prefix = ""
      if ($0 ~ ("^[A-Za-z_][A-Za-z0-9_]*=(\\$[A-Za-z_][A-Za-z0-9_]*)?" q "$")) {
        state = "ok"
        if ($0 ~ /=\$/) { prefix = $0; sub(/^[^$]*\$/, "", prefix); sub(/.$/, "", prefix) }
      } else { state = "form"; row() }
      next
    }
    name != "" && $0 == q { row(); next }
    name != "" && state == "ok" && index($0, q) { state = "quote:" NR }
    END { if (name != "") { state = "unclosed"; row() } }
  ' "$1"
}
perl_program_compile() { # <script> <name> <its rows>: 0 iff the value the shell assigns compiles; prints why not
  (
    local m state prefix out
    while IFS=$'\t' read -r m state prefix; do unset "$m"; done <<<"$3"
    while IFS=$'\t' read -r m state prefix; do
      if [ "$state" = ok ]; then
        if [ -n "$prefix" ] && [ -z "${!prefix:-}" ]; then
          [ "$m" != "$2" ] || { printf 'it prepends $%s, which this file does not assign cleanly before it' "$prefix"; exit 1; }
        else
          # No quote inside, so this is the assignment and nothing else.
          eval "$(awk -v q="'" -v open="$m=" 'index($0, open) == 1 { f = 1 } f { print } f && $0 == q { exit }' "$1")" ||
            { [ "$m" != "$2" ] || { printf 'its assignment does not evaluate'; exit 1; }; }
        fi
      fi
      [ "$m" != "$2" ] || break
    done <<<"$3"
    [ -n "${!2:-}" ] || { printf 'its assignment leaves it empty'; exit 1; }
    out=$(perl -c -e "${!2}" 2>&1) || { printf 'perl -c: %s' "$(tr '\n' ' ' <<<"$out")"; exit 1; }
  )
}
perl_program_verdicts() { # <script>...: "NAME<TAB>pass", or "NAME<TAB>fail<TAB>WHY", per program
  local f rows name state prefix why
  for f in "$@"; do
    rows=$(perl_program_rows "$f") || return 1
    while IFS=$'\t' read -r name state prefix; do
      [ -n "$name" ] || continue
      case $state in
        ok) why=$(perl_program_compile "$f" "$name" "$rows") && why= ;;
        quote:*) why="line ${state#quote:} of ${f##*/} holds a single quote, which ends the program there" ;;
        unclosed) why="no closing quote on a line of its own in ${f##*/}" ;;
        *) why="its assignment in ${f##*/} does not open a single-quoted program at the line's end" ;;
      esac
      if [ -z "$why" ]; then printf '%s\tpass\n' "$name"; else printf '%s\tfail\t%s\n' "$name" "$why"; fi
    done <<<"$rows"
  done
}
perl_program_control() { # <label> <bash -n must accept it: 0|1> <why pattern> FROM TO
  local mutant=$TMP/perl-program-mutant.sh rows failed why
  token_mutant "$mutant" "$4" "$5" ||
    { harness_abandon "the line '$4' is not in $SRC exactly once; the mutant is broken"; return 1; }
  rows=$(perl_program_verdicts "$mutant")
  failed=$(awk -F '\t' '$2 != "pass" { printf "%s%s", sep, $1; sep = " " }' <<<"$rows")
  why=$(awk -F '\t' '$1 == "lock_census_perl" { print $3 }' <<<"$rows")
  if [ "$2" = 1 ] && ! bash -n "$mutant" 2>/dev/null; then
    report 1 "$1" "bash -n refuses this mutant, so it shows nothing bash -n could not"
  elif [ "$failed" = "lock_census_perl supervisor_perl" ] && [[ $why == $3 ]]; then
    report 0 "$1"
  else
    report 1 "$1" "failed: ${failed:-none} (want lock_census_perl and supervisor_perl, which opens with it); lock_census_perl: ${why:-passed} (want $3)"
  fi
}
leg_1b() {
local l_all="each embedded Perl program of the script and of the files it sources holds no single quote and compiles under perl -c"
local l_refs="every NAME_perl the script expands is one of the programs checked"
local files=("$SRC") rel rows bad refs missing=
while IFS= read -r rel; do files+=("$HERE/../$rel"); done < <(sed -n 's/^\. \([A-Za-z0-9_./-]*\)$/\1/p' "$SRC")
rows=$(perl_program_verdicts "${files[@]}")
bad=$(awk -F '\t' '$2 != "pass" { printf "%s: %s; ", $1, $3 }' <<<"$rows")
if [ -n "$rows" ] && [ -z "$bad" ]; then
  report 0 "$l_all ($(grep -c . <<<"$rows") programs)"
else
  report 1 "$l_all" "${bad:-no program found in ${files[*]}}"
fi
refs=$(grep -v '^[[:space:]]*#' "$SRC" | grep -oE '\$\{?[A-Za-z_][A-Za-z0-9_]*_perl' | tr -d '${' | sort -u)
for rel in $refs; do
  grep -q "^$rel"$'\t' <<<"$rows" || missing="$missing $rel"
done
if [ -n "$refs" ] && [ -z "$missing" ]; then
  report 0 "$l_refs"
else
  report 1 "$l_refs" "$([ -n "$refs" ] && echo "no NAME_perl=' at a line's start assigns:$missing" || echo 'the script expands no NAME_perl; the scan is broken')"
fi
perl_program_control "negative control: a lone quote in a program fails it, and the program that opens with it" 0 \
  'line * holds a single quote*' 'sub census_token {' "sub census_token { # it's" || return
perl_program_control "negative control: a quoted word in a program, which bash -n accepts, fails it" 1 \
  'line * holds a single quote*' 'sub census_token {' "sub 'census_token' {" || return
perl_program_control "negative control: a program that is not Perl, which bash -n accepts, fails perl -c" 1 \
  'perl -c: *' 'sub census_token {' 'sub census_token {{' || return
}
harness_leg_run '1b: each embedded Perl program compiles and holds no quote #group-alive' leg_1b

# ---------------------------------------------------------------------------
# Leg 2: a live group reads alive
# ---------------------------------------------------------------------------
# `set -m` puts the child in a process group of its own, so its pid IS a pgid
# holding exactly one live process -- the shape all four call sites signal.
# That the child really leads its own group has to be CHECKED (a shell whose
# setpgrp does not take would otherwise have this leg quietly testing the
# harness's own group), so both legs need the pgid reader.
leg_2_3() {
live_label="a group holding a running process reads as alive"
empty_label="a group with no members left reads as dead"
if [ "$have_pgid" = 0 ]; then
  skip "$live_label" "no way to read a process's group on this system"
  skip "$empty_label" "no way to read a process's group on this system"
else
  set -m
  sleep 30 >/dev/null 2>&1 </dev/null &
  livepid=$!
  set +m
  lpgid="$(ppgid "$livepid")"
  if [ "$lpgid" != "$livepid" ]; then
    report 1 "$live_label" \
      "the child did not lead its own group (pgid '${lpgid:-gone}' vs pid $livepid)"
  elif ! kill -0 -- "-$livepid" 2>/dev/null; then
    report 1 "$live_label" "the group is not even signal-reachable; the leg tested nothing"
  elif group_alive "$livepid"; then
    report 0 "$live_label"
  else
    report 1 "$live_label" "group_alive said dead for a group with a live sleep in it"
  fi

  # -------------------------------------------------------------------------
  # Leg 3: an empty group reads dead
  # -------------------------------------------------------------------------
  kill -KILL -- "-$livepid" 2>/dev/null
  wait "$livepid" 2>/dev/null   # reaped here, so the group really is empty
  livepid=""
  if kill -0 -- "-$lpgid" 2>/dev/null; then
    report 1 "$empty_label" "pgid $lpgid is still signal-reachable after the kill and reap"
  elif group_alive "$lpgid"; then
    report 1 "$empty_label" "group_alive said alive for an empty group"
  else
    report 0 "$empty_label"
  fi
fi
}
harness_leg_run '2-3: a live group reads alive, an emptied one dead #group-alive' leg_2_3

# ---------------------------------------------------------------------------
# Leg 4: a zombie-only group reads dead
# ---------------------------------------------------------------------------
# The negative control for the whole change: `kill -0` must still say YES here,
# or the leg would pass without exercising the difference. Under an init that
# reaps, such a corpse is transient and the bare probe merely lost a race with
# it; under one that does not -- the ordinary container case -- it is PERMANENT,
# so waiting it out was never the fix.
#
# Making a zombie that is reliably still a zombie when looked at, and that leaves
# nothing behind afterwards, takes care. The parent puts the child in a process
# group of ITS OWN (so the group holds the zombie and nothing else), then STOPs
# itself before the child exits: a stopped shell runs no SIGCHLD handler, so it
# cannot reap, and the child stays a zombie for as long as the leg needs.
#
# All of that needs a state reader, and needs it INDEPENDENT of the function
# under test: without one, "is it a zombie yet" reads the same as "it is gone",
# the leg would sit out its whole retry budget and then judge `group_alive` on a
# premise it never established -- and the cleanup assertion at the end would
# read an empty state as "reaped" and pass on a system that cannot see the
# corpse at all. So the whole leg, zombie maker included, is skipped there.
leg_4() {
zlabel="a group holding nothing but a zombie reads as dead"
clabel="the state reader alone rejects a zombie-only group (signal probe forced to say alive)"
rlabel="the zombie leg reaps its own zombie, leaving no process-table entry"
ilabel="a zombie retains identity without being reported alive"
if [ "$have_state" = 0 ]; then
  skip "$zlabel" "no way to read a process's state on this system"
  skip "$clabel" "no way to read a process's state on this system"
  skip "$rlabel" "the zombie leg did not run, so it left nothing to reap"
  skip "$ilabel" "the zombie leg did not run, so identity was not testable"
else
zpidfile="$TMP/zombie.pid"; rm -f "$zpidfile"
bash -c 'set -m
         sleep 0.5 >/dev/null 2>&1 </dev/null &
         echo $! >"$1"
         set +m
         kill -STOP $$
         wait' _ "$zpidfile" >/dev/null 2>&1 </dev/null &
zparent=$!
zpid=""
for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
  [ -s "$zpidfile" ] && zpid="$(cat "$zpidfile")" && break
  sleep 0.1
done
# Wait for real zombiehood rather than guessing at it, through `pstate`, which
# is this harness's own reader and shares no code with `group_alive`.
zstate=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30; do
    zstate="$(pstate "$zpid")"
    case "$zstate" in Z*) break ;; esac
    sleep 0.1
  done
fi
if [ -z "$zpid" ]; then
  report 1 "$zlabel" "could not start the zombie maker"
  report 1 "$clabel" "could not start the zombie maker"
else
  case "$zstate" in
    Z*)
      # Whether the BARE probe over-reports here is a property of the kernel:
      # Linux (and every container on it) counts a zombie as a group member and
      # says alive, which is the bug; Darwin's killpg answers ESRCH once the
      # group holds only corpses, so the bare probe happens to agree there. The
      # claim itself holds on both, and the control below runs on both.
      if kill -0 -- "-$zpid" 2>/dev/null; then
        zwhere="the bare group signal probe says ALIVE here -- the misreading this fixes"
      else
        zwhere="this kernel's killpg already answers dead for a zombie-only group"
      fi
      if group_alive "$zpid"; then
        report 1 "$zlabel" \
          "group_alive counted zombie group $zpid as work -- stop can report a phantom orphaned group"
      else
        report 0 "$zlabel ($zwhere)"
      fi
      printf '%s\n' "$zpid" >"$TMP/zombie-identity.pid"
      ps_token "$zpid" >"$TMP/zombie-identity.token"
      if proc_identity_matches "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token" \
         && ! proc_alive "$TMP/zombie-identity.pid" "$TMP/zombie-identity.token"; then
        report 0 "$ilabel"
      else
        report 1 "$ilabel" \
          "the zombie either lost its recorded identity or was incorrectly reported live"
      fi
      # The negative control, run everywhere: shadow `kill` so the signal probe
      # inside group_alive succeeds, which is exactly what the bare check did on
      # the kernel where this was reproduced. The state reader must still refuse
      # the group -- otherwise this platform's pass above was the gate's doing
      # alone and the ladder itself is untested here.
      (
        kill() { case "$*" in -0*) return 0 ;; *) command kill "$@" ;; esac; }
        if ! kill -0 -- "-$zpid"; then
          exit 3   # the shadow did not take; the control tested nothing
        fi
        group_alive "$zpid" && exit 1
        exit 0
      )
      case $? in
        0) report 0 "$clabel" ;;
        3) report 1 "$clabel" 'the `kill` shadow did not take effect' ;;
        *) report 1 "$clabel" \
             "with the signal probe forced alive, group_alive called a zombie-only group alive" ;;
      esac
      ;;
    *)
      report 1 "$zlabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$clabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so the leg tested nothing"
      report 1 "$ilabel" \
        "the child never reached state Z (saw '${zstate:-gone}'), so identity was not tested"
      ;;
  esac
fi
# Let the parent reap its own child rather than orphaning the zombie, then check
# we really left nothing behind.
kill -CONT "$zparent" 2>/dev/null
wait "$zparent" 2>/dev/null
zparent=""
zleft=""
if [ -n "$zpid" ]; then
  for _ in 1 2 3 4 5 6 7 8 9 10; do
    zleft="$(pstate "$zpid")"
    [ -z "$zleft" ] && break
    sleep 0.1
  done
fi
if [ -z "$zleft" ]; then
  report 0 "$rlabel"
else
  report 1 "$rlabel" "pid $zpid is still present as '$zleft'"
fi
fi
}
harness_leg_run '4: a zombie-only group reads dead #group-alive' leg_4

# ---------------------------------------------------------------------------
# Leg 5: only a positive decimal integer is a pgid
# ---------------------------------------------------------------------------
# A corrupted or forged pgid file must never reach kill: `kill -0 -- -0` targets
# the CALLER's own group (which is alive, so the answer would be a confident
# yes) and a negative reading broadcasts.
leg_5() {
bad=""
for candidate in "" "0" "12x" "-1" "abc" " 7"; do
  if group_alive "$candidate"; then bad="$bad '$candidate'"; fi
done
if [ -z "$bad" ]; then
  report 0 "a pgid that is not a positive decimal integer is refused"
else
  report 1 "a pgid that is not a positive decimal integer is refused" \
    "accepted:$bad"
fi
}
harness_leg_run '5: only a positive decimal integer is a pgid #group-alive' leg_5


shared_through stop
leg_6_8() {
[ -n "$stop_last" ] || { harness_abandon "stop: paths last named no pointer under $STOP_RUNS"; return; }
if [ -n "$stop_skip" ]; then
  skip "$l_ignored" "$stop_skip"
  skip "$l_killed" "$stop_skip"
  skip "$l_took" "$stop_skip"
else
  # -------------------------------------------------------------------------
  # Leg 6: the leader ignores TERM
  # -------------------------------------------------------------------------
  if stop_probe ignores "$body_ignores" "$SRC" "$stop_last"; then
    said "$l_ignored" \
      "orphaned process group $stop_pg survived TERM (possibly only as unreaped exited processes); escalated to KILL"
    # The sentence is a claim about what stop DID, so the doing is checked too:
    # an escalation that only announced itself would leave the group holding the
    # worktree lock, which is the whole reason stop reaches for KILL here.
    if [ "$have_state" = 0 ]; then
      skip "$l_killed" "no way to read a process's state on this system"
    else
      # Both members, so a KILL that reached only the leader fails here.
      k_left=""
      for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
        k_left=""
        for k_pid in $stop_pg $stop_member; do
          case "$(pstate "$k_pid")" in '' | Z*) ;; *) k_left="$k_left $k_pid" ;; esac
        done
        [ -z "$k_left" ] && break
        sleep 0.1
      done
      if [ -z "$k_left" ]; then
        report 0 "$l_killed"
      else
        report 1 "$l_killed" \
          "still running two seconds after the escalation:$k_left (group $stop_pg)"
      fi
    fi
  else
    report 1 "$l_ignored" "$stop_err"
    report 1 "$l_killed" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 7: the leader takes the TERM
  # -------------------------------------------------------------------------
  # The difference from leg 6 is one `trap` in the leader and nothing else, so
  # a stop that reported both the same way would fail exactly one of them.
  if stop_probe takes "$body_takes" "$SRC" "$stop_last"; then
    said "$l_took" "sent TERM to the orphaned process group $stop_pg; re-run stop to confirm"
  else
    report 1 "$l_took" "$stop_err"
  fi
  end_leader

  # -------------------------------------------------------------------------
  # Leg 8: a legacy run's leftover is reaped through its in-tree lock
  # -------------------------------------------------------------------------
  # The version before gh-ocannl-606 kept the lock and its owner pointer
  # BESIDE the worktree, and a run directory it recorded has no `runs` root.
  # Its leftovers -- a descendant dune holding the inherited lock fd -- hold
  # THAT lock, so `stop` must read the run under those paths or it reports
  # nothing to reap while the lock stays held. The fixture: a run directory
  # with no `runs`, no pid and no pgid (only the lock can attribute
  # anything), the in-tree owner pointer naming it, and a perl holding an
  # flock on the in-tree lock file, which stop's census must find and TERM.
  l_legacy="stop: a legacy run's leftover holding the in-tree lock is attributed and reaped"
  legacy_dir=$STOP_RUNS/19700101-000000-legacy
  mkdir -p "$legacy_dir" "$STOP_WT"
  printf 'runtest (the harness fixture legacy)\n' >"$legacy_dir/cmd"
  printf '0\n' >"$legacy_dir/cap"
  printf '%s\n' "$STOP_WT" >"$legacy_dir/wt"
  : >"$legacy_dir/log"
  printf '%s\n' "$legacy_dir" >"$STOP_WT/.test-run.lock.owner"
  printf '%s\n' "$legacy_dir" >"$stop_last"
  rm -f "$TMP/legacy.pid"
  perl -e 'use Fcntl ":flock";
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           flock($fh, LOCK_EX | LOCK_NB) or exit 1;
           $| = 1; print "$$\n"; sleep 600' \
    "$STOP_WT/.test-run.lock" >"$TMP/legacy.pid" 2>/dev/null </dev/null &
  legacy_holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$TMP/legacy.pid" ] && break
    sleep 0.1
  done
  # The premise, checked: the lock really is held before stop is asked.
  legacy_lock_out=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last)
  legacy_lock_rc=$?
  if [ ! -s "$TMP/legacy.pid" ] || [ "$legacy_lock_rc" != 3 ] || [ "$legacy_lock_out" != held ]; then
    report 1 "$l_legacy" "the fixture holder did not take the in-tree lock"
  else
    stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" stop last 2>"$TMP/legacy.stderr")"
    stop_rc=$?
    stop_diag="$(cat "$TMP/legacy.stderr" 2>/dev/null)"
    said "$l_legacy" \
      "run is dead without a verdict, but its leftover processes held the worktree lock; reaped them"
    # The claim is about what stop DID: the holder must be gone and the lock free.
    for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
      kill -0 "$legacy_holder" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$legacy_holder" 2>/dev/null; then
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "pid $legacy_holder still holds $STOP_WT/.test-run.lock after stop"
    elif OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status last >"$TMP/legacy-lock-after" \
         && [ "$(cat "$TMP/legacy-lock-after")" = idle ]; then
      report 0 "$l_legacy (the holder is gone and the lock is free)"
    else
      report 1 "$l_legacy (the holder is gone and the lock is free)" \
        "$STOP_WT/.test-run.lock is still locked after stop"
    fi
  fi
  if kill -0 "$legacy_holder" 2>/dev/null; then kill -KILL "$legacy_holder" 2>/dev/null; fi
  wait "$legacy_holder" 2>/dev/null
  legacy_holder=""
  rm -f "$STOP_WT/.test-run.lock" "$STOP_WT/.test-run.lock.owner"
fi
}
harness_leg_run '6-8: the sentences stop prints, and a legacy leftover reaped #stop' leg_6_8

# ---------------------------------------------------------------------------
# Leg 67: a lock its acquirer left behind, held through an inherited descriptor
# ---------------------------------------------------------------------------
# The shape every run's lock has (gh-ocannl-1107): the launcher opens fd 9,
# take_lock's perl -- a child sharing that description -- takes the flock and
# exits, and the lock lives on in the description the supervisor, dune and
# dune's leftovers hold. On Linux /proc/locks may keep naming the ACQUIRER --
# dead, or 0 from inside a pid namespace -- and a census that trusted any name
# it gave returned that pid, never swept fdinfo, and `stop` reaped none of the
# live holders; other kernels drop the row once the acquirer exits. The census
# `stop` shares with the supervisor (lock_census_perl, gh-ocannl-1906) reads
# no name: it sweeps fdinfo on Linux, and elsewhere asks lsof for the file's
# openers, so the reap is asserted wherever either answers. The fixture is
# that shape: a run directory with no pid, no pgid and no verdict (only the
# lock can attribute anything), the owner pointer naming it, and a perl
# HOLDER -- this shell's own child, so it stays reapable under a PID 1 that
# reaps nothing -- that opens the lock file, forks the ACQUIRER to flock the
# shared description, and reaps it. The negative control, leg 67b, runs the
# census that went by the name -- the shipping script with the sweep's
# processes replaced by the pids /proc/locks names -- and must leave the
# holder running. Both legs build the fixture, so each passes run alone.
inherit_fixture() {
l_inherit="stop: a leftover holding the lock through an inherited descriptor, its acquirer gone, is reaped"
l_inherit_nc="negative control: a census that goes by /proc/locks' named acquirer misses that holder"
inherit_skip= inherit_nc_skip=
if [ ! -d /proc/self/fdinfo ] && ! command -v lsof >/dev/null 2>&1; then
  inherit_skip="neither /proc fdinfo nor lsof here: \`stop\` has no lock census"
elif [ "$have_state" = 0 ]; then
  inherit_skip="no way to read a process's state, so a reaped holder cannot be told from a zombie"
elif [ ! -r /proc/locks ] || [ ! -d /proc/self/fdinfo ]; then
  inherit_nc_skip="/proc/locks or /proc fdinfo cannot be read here, so the census that goes by the name cannot be run"
fi
inherit_dir=$STOP_RUNS/19700101-000000-inherited
mkdir -p "$inherit_dir" "$STOP_WT"
printf 'runtest (the harness fixture inherited)\n' >"$inherit_dir/cmd"
printf '0\n' >"$inherit_dir/cap"
printf '%s\n' "$STOP_WT" >"$inherit_dir/wt"
printf '%s\n' "$STOP_RUNS" >"$inherit_dir/runs"
: >"$inherit_dir/log"
inherit_lock=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths lock "$inherit_dir") || inherit_lock=
inherit_owner=$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" paths owner "$inherit_dir") || inherit_owner=

# The pids /proc/locks names for the fixture lock, matched by device AND inode
# as the shipping census matches them: inode numbers repeat across
# filesystems, and another mount's flock must not pass for the acquirer.
inherit_named() {
  perl -e '
    my @st = stat($ARGV[0]) or exit 1;
    my $dev = $st[0];
    my ($maj, $min) = (($dev >> 8) & 0xfff, ($dev & 0xff) | (($dev >> 12) & ~0xff));
    open my $fh, "<", "/proc/locks" or exit 1;
    while (<$fh>) {
      my @f = split;
      next unless defined $f[1] && $f[1] eq "FLOCK";
      my ($lmaj, $lmin, $ino) = split /:/, $f[5];
      print "$f[4]\n"
        if defined $ino && $ino == $st[1] &&
           hex($lmaj) == $maj && hex($lmin) == $min;
    }
  ' "$inherit_lock" 2>/dev/null
}
inherit_kill() { # the holder is this shell's child: unreaped, its pid cannot be recycled
  [ -n "${inherit_holder:-}" ] || return 0
  kill -KILL "$inherit_holder" 2>/dev/null
  wait "$inherit_holder" 2>/dev/null
  inherit_holder=
  return 0
}
inherit_err= inherit_acquirer= inherit_named_pids=
inherit_start() { # sets inherit_holder, inherit_acquirer, inherit_named_pids, or inherit_err
  local line
  inherit_holder= inherit_acquirer= inherit_named_pids= inherit_err=
  printf '%s\n' "$inherit_dir" >"$inherit_owner" || { inherit_err="cannot write $inherit_owner"; return 1; }
  rm -f "$TMP/inherited.pids"
  perl -e 'use Fcntl ":flock"; use POSIX ();
           open(my $fh, ">>", $ARGV[0]) or exit 1;
           defined(my $kid = fork) or exit 1;
           POSIX::_exit(flock($fh, LOCK_EX | LOCK_NB) ? 0 : 1) if !$kid;
           waitpid($kid, 0) == $kid && $? == 0 or exit 1;
           $| = 1; print "$$ $kid\n"; sleep 600' \
    "$inherit_lock" >"$TMP/inherited.pids" 2>/dev/null </dev/null &
  inherit_holder=$!
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    [ -s "$TMP/inherited.pids" ] && break
    sleep 0.1
  done
  read -r line inherit_acquirer <"$TMP/inherited.pids" 2>/dev/null
  if [ "${line:-}" != "$inherit_holder" ] || [ -z "${inherit_acquirer:-}" ]; then
    inherit_err="the holder did not report taking the lock through a reaped acquirer"; return 1
  fi
  # The premise, checked: the lock is held, its acquirer is gone, and
  # /proc/locks does not name the live holder -- whatever else it names.
  if [ "$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status "$inherit_dir")" != held ]; then
    inherit_err="the fixture lock is not held"; return 1
  fi
  kill -0 "$inherit_acquirer" 2>/dev/null && { inherit_err="the acquirer $inherit_acquirer is still alive"; return 1; }
  inherit_named_pids=$(inherit_named)
  if printf '%s\n' "$inherit_named_pids" | grep -qx "$inherit_holder"; then
    inherit_err="/proc/locks names the live holder $inherit_holder, not its acquirer"; return 1
  fi
  return 0
}
inherit_stop() { # <script>
  stop_out="$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$1" stop "$inherit_dir" 2>"$TMP/inherited.stderr")"
  stop_rc=$?
  stop_diag="$(cat "$TMP/inherited.stderr" 2>/dev/null)"
}
inherit_gone() { # 0 iff the holder is dead (a zombie: it is ours to reap) within two seconds
  for _ in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
    case "$(pstate "$inherit_holder")" in '' | Z*) return 0 ;; esac
    sleep 0.1
  done
  return 1
}
}
leg_67() {
inherit_fixture
if [ -n "$inherit_skip" ]; then
  skip "$l_inherit" "$inherit_skip"
elif [ -z "$inherit_lock" ] || [ -z "$inherit_owner" ]; then
  report 1 "$l_inherit" "\`paths lock\`/\`paths owner\` did not name the fixture run's lock"
else
  if ! inherit_start; then
    report 1 "$l_inherit" "$inherit_err"
  else
    inherit_stop "$SRC"
    said "$l_inherit" \
      "run is dead without a verdict, but its leftover processes held the worktree lock; reaped them"
    if ! inherit_gone; then
      report 1 "$l_inherit (the holder is gone and the lock is free)" \
        "pid $inherit_holder still holds $inherit_lock after stop"
    elif [ "$(OCANNL_TOOL_TEST_RUNS="$STOP_RUNS" "$SRC" lock-status "$inherit_dir")" = idle ]; then
      report 0 "$l_inherit (the holder is gone and the lock is free)"
    else
      report 1 "$l_inherit (the holder is gone and the lock is free)" \
        "$inherit_lock is still locked after stop"
    fi
  fi
  inherit_kill
fi
[ -z "$inherit_lock" ] || rm -f "$inherit_lock" "$inherit_owner"
}
harness_leg_run '67: a lock held through an inherited descriptor is reaped #stop' leg_67

# Whether the /proc under ROOT can show what leg 67b pins: prints `absent`
# when it holds neither the lock table (ROOT/locks) nor the per-descriptor one
# (ROOT/self/fdinfo). That answer alone makes 67b not applicable
# (gh-ocannl-1930): with no fdinfo the shipping census never runs the sweep
# 67b mutates -- it asks lsof for the file's openers -- and with no
# /proc/locks there is no named acquirer for a census to go by, so the control
# has no subject there and nothing is lost. A /proc holding either table
# leaves 67b to run, or to skip where the other cannot be read.
proc_lock_tables_probe() { # ROOT
  [ -e "$1/locks" ] || [ -e "$1/self/fdinfo" ] || echo absent
}
leg_67b() {
if [ "$(proc_lock_tables_probe /proc)" = absent ]; then
  harness_not_applicable "this system has neither /proc/locks nor /proc fdinfo: \`stop\`'s census reads lsof's openers and no name, so a census that goes by /proc/locks' named acquirer does not exist here"
  return
fi
inherit_fixture
if [ -n "$inherit_skip" ]; then
  skip "$l_inherit_nc" "$inherit_skip"
elif [ -z "$inherit_lock" ] || [ -z "$inherit_owner" ]; then
  report 1 "$l_inherit_nc" "\`paths lock\`/\`paths owner\` did not name the fixture run's lock"
else
  # The census before gh-ocannl-1107, near enough: the processes /proc/locks
  # names for any FLOCK, each still asked for the FLOCK in its fdinfo. The
  # sweep's line is replaced whole, and checked to be the only change, so a
  # sweep that moved leaves this control failing rather than testing the
  # shipping code. A kernel that names nobody once the acquirer exits gives
  # that census nothing to ask, which is the miss all the same.
  inherit_root=$TMP/inherit-repo
  mkdir -p "$inherit_root/tools"
  stage_sourced "$inherit_root" || { harness_abandon "$stage_error"; return; }
  awk -v named='    for my $p (do { my @n; if (open my $lf, "<", "/proc/locks") { @n = map { (split)[4] } grep { ((split)[1] // "") eq "FLOCK" } <$lf> } @n }) {' '
    $0 == "    for my $p (grep { /\\A[0-9]+\\z/ && !$skip{$_} } readdir $pd) {" { print named; n++; next }
    { print }
    END { exit n == 1 ? 0 : 1 }
  ' "$SRC" >"$inherit_root/tools/test-run.sh" && chmod +x "$inherit_root/tools/test-run.sh"
  inherit_mut_rc=$?
  inherit_changed=$(diff "$SRC" "$inherit_root/tools/test-run.sh" | grep -c '^[<>]')
  if [ -n "$inherit_nc_skip" ]; then
    skip "$l_inherit_nc" "$inherit_nc_skip"
  elif [ "$inherit_mut_rc" != 0 ] || [ "$inherit_changed" != 2 ]; then
    report 1 "$l_inherit_nc" "the census sweep's line was not found once in $SRC"
  elif ! inherit_start; then
    report 1 "$l_inherit_nc" "$inherit_err"
  else
    inherit_stop "$inherit_root/tools/test-run.sh"
    if inherit_gone; then
      report 1 "$l_inherit_nc" \
        "the unfiltered census reaped the holder too (stop printed: ${stop_out:-<nothing>}), so the leg cannot tell the fix"
    else
      report 0 "$l_inherit_nc"
    fi
  fi
  inherit_kill
fi
[ -z "$inherit_lock" ] || rm -f "$inherit_lock" "$inherit_owner"
}
harness_leg_run "67b: a census that goes by /proc/locks' named acquirer misses that holder #stop" leg_67b

leg_67c() {
  local d=$TMP/67c ok=1
  # Stand-ins for the /proc a host shows: both tables (Linux), one of them, none.
  mkdir -p "$d/both/self/fdinfo" "$d/locks-only" "$d/fdinfo-only/self/fdinfo" "$d/neither"
  : >"$d/both/locks"
  : >"$d/locks-only/locks"
  for stand_in in "$d/both" "$d/locks-only" "$d/fdinfo-only"; do
    [ "$(proc_lock_tables_probe "$stand_in")" != absent ] || ok=0
  done
  [ "$(proc_lock_tables_probe "$d/neither")" = absent ] || ok=0
  if [ "$ok" = 1 ]; then
    report 0 "67c: only a /proc observed without both lock tables makes 67b not applicable (negative control)"
  else
    report 1 "67c: only a /proc observed without both lock tables makes 67b not applicable (negative control)" \
      "the probe answered a /proc holding a table as absent, or one holding neither as present"
  fi
}
harness_leg_run '67c: only a /proc observed without both lock tables makes 67b not applicable #stop' leg_67c

# ---------------------------------------------------------------------------
# Leg 101: one rendering of a start token (gh-ocannl-1944)
# ---------------------------------------------------------------------------
# Start tokens are compared as strings across writers: the supervisor and the
# group leader record their own (`ptoken`, `gtoken`) through census_token, the
# shell re-checks them through ps_token, and a launch chain (`launchers`) holds
# the tokens proc_table_perl joined out of one bulk process table. A rendering
# that differs in one of them makes a live run read as dead. So the three are
# read here from the script under test -- ps_token's function, and the two
# Perl programs by their assignments -- for the same pids: this shell and each
# ancestor above it, the rows `chain` prints (an ancestor started on a
# one-digit day carries the padded lstart the squeeze is for). The negative
# controls are that script with ps_token's rendering, then the bulk join's,
# changed on both of its branches (another stat field under /proc, another
# time zone for lstart): each must be told from census_token. The fourth
# renderer the issue counted, the supervisor's own, is gone: the last check
# is the inventory of every line of the script that renders a token, so one
# added beside these is pinned here or refused.
# lock_census's own bulk join needs no row: it pairs a pid only while
# census_same holds on the token it joined, so a rendering that differs finds
# no holder, which leg 67 decides wherever lsof is the census.
token_views() { # <script>: "PID<TAB>CENSUS<TAB>SHELL<TAB>BULK" for this shell and each ancestor above it
  (
    unset -f ps_token
    unset lock_census_perl proc_table_perl
    sed -n '/^ps_token() {/,/^}/p' "$1" >"$TMP/token-views.sh" || exit 2
    # shellcheck disable=SC1091
    . "$TMP/token-views.sh" || exit 2
    eval "$(sed -n "/^lock_census_perl='\$/,/^'\$/p" "$1")" || exit 2
    eval "$(sed -n "/^proc_table_perl='\$/,/^'\$/p" "$1")" || exit 2
    [ "$(type -t ps_token)" = function ] && [ -n "${lock_census_perl:-}" ] && [ -n "${proc_table_perl:-}" ] || exit 2
    perl -e "$proc_table_perl" chain >"$TMP/token-chain" 2>/dev/null || exit 2
    while read -r pid rest; do
      # "PID TOKEN ARGV-MD5": the token holds spaces where it is an lstart.
      case $rest in *' '*) bulk=${rest% *} ;; *) bulk= ;; esac
      census=$(perl -e "$lock_census_perl"'print census_token($ARGV[0])' "$pid" 2>/dev/null)
      shell=$(ps_token "$pid")
      # An ancestor that exited since the table was read has no token to compare.
      [ "$pid" = "$$" ] || [ -n "$census$shell" ] || continue
      printf '%s\t%s\t%s\t%s\n' "$pid" "$census" "$shell" "$bulk"
    done <"$TMP/token-chain"
  )
}
token_diff() { # <script>; sets token_rows, token_self (this shell's row), token_bad (rows not one string)
  token_rows=$(token_views "$1") || token_rows=
  token_self=$(awk -F '\t' -v p="$$" '$1 == p' <<<"$token_rows")
  token_bad=$(awk -F '\t' 'NF && !($2 == $3 && $3 == $4)' <<<"$token_rows")
}
leg_101() {
local l_one="start tokens: ps_token, the launch chain's bulk join and census_token render one string for this shell and each ancestor"
local l_shell_nc="negative control: a ps_token that renders differently is told from census_token"
local l_bulk_nc="negative control: a bulk join that renders differently is told from census_token"
local l_only="start tokens: the renderers are census_token, ps_token and the two bulk joins, and no other"
local tok_skip=
[ -n "$(ps_token $$)" ] || tok_skip="this system records no start token, so there is no rendering to compare"
if [ -n "$tok_skip" ]; then
  skip "$l_one" "$tok_skip"
  skip "$l_shell_nc" "$tok_skip"
  skip "$l_bulk_nc" "$tok_skip"
else
  token_diff "$SRC"
  if [ -n "$(cut -f2 <<<"$token_self")" ] && [ -z "$token_bad" ]; then
    report 0 "$l_one"
  else
    report 1 "$l_one" "this shell's row (pid, census_token, ps_token, bulk join): ${token_self:-none}; rows that differ: ${token_bad:-none}"
  fi
  token_mutant "$TMP/token-shell-mutant.sh" \
    '      print defined $f[19] ? $f[19] : "";' '      print defined $f[20] ? $f[20] : "";' \
    '    LC_ALL=C TZ=UTC ps -o lstart= -p "$1" 2>/dev/null |' '    LC_ALL=C TZ=UTC-13 ps -o lstart= -p "$1" 2>/dev/null |' ||
    { harness_abandon "ps_token's two renderings could not be changed in a copy of $SRC; the mutant is broken"; return; }
  token_diff "$TMP/token-shell-mutant.sh"
  if awk -F '\t' '$2 != "" && $2 == $4 && $2 != $3 { ok = 1 } END { exit !ok }' <<<"$token_self"; then
    report 0 "$l_shell_nc"
  else
    report 1 "$l_shell_nc" "this shell's row (pid, census_token, ps_token, bulk join): ${token_self:-none}"
  fi
  token_mutant "$TMP/token-bulk-mutant.sh" \
    '    $tok{$p} = defined $f[19] ? $f[19] : "";' '    $tok{$p} = defined $f[20] ? $f[20] : "";' \
    '  $ENV{TZ} = "UTC";' '  $ENV{TZ} = "UTC-13";' ||
    { harness_abandon "the bulk join's two renderings could not be changed in a copy of $SRC; the mutant is broken"; return; }
  token_diff "$TMP/token-bulk-mutant.sh"
  if awk -F '\t' '$2 != "" && $2 == $3 && $2 != $4 { ok = 1 } END { exit !ok }' <<<"$token_self"; then
    report 0 "$l_bulk_nc"
  else
    report 1 "$l_bulk_nc" "this shell's row (pid, census_token, ps_token, bulk join): ${token_self:-none}"
  fi
fi
# Every line that reads a start time, by the two things a token is made of:
# stat's starttime (census_token, ps_token, the chain's table) and an lstart
# (census_token, lock_census's table, ps_token, the chain's table).
local tok_stat tok_lstart
tok_stat=$(grep -c -F '$f[19] :' "$SRC")
tok_lstart=$(grep -c -E '^[^#]*lstart=' "$SRC")
if [ "$tok_stat" = 3 ] && [ "$tok_lstart" = 4 ]; then
  report 0 "$l_only"
else
  report 1 "$l_only" "$tok_stat line(s) render stat's starttime (want 3), $tok_lstart an lstart (want 4): $(grep -n -E '\$f\[19\] :|^[^#]*lstart=' "$SRC" | cut -c1-90 | tr '\n' ';')"
fi
}
harness_leg_run '101: one rendering of a start token #stop' leg_101

shared_through repeat_root
# Read-only query contract: no state store, no first run, and no lock file yet.
# The fixture root (not the caller's cwd) owns all omitted-RUN queries.
leg_39() {
query_runs=$TMP/query-state/runs
query_expected=$(cd "$TMP" && pwd -P)/query-state/runs
query() {
  OCANNL_TOOL_TEST_RUNS=$query_runs "$repeat_root/tools/test-run.sh" "$@"
}
query_rc=0
query_out=$(query paths runs) || query_rc=$?
query_wt=$(query paths worktree)
query_lock=$(query paths lock)
query_owner=$(query paths owner)
query_last=$(query paths last)
query_idle=$(query lock-status) || query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_expected" ] \
   && [ "$query_wt" = "$(cd "$repeat_root" && pwd -P)" ] \
   && [ "$query_idle" = idle ] && [ ! -e "$TMP/query-state" ] \
   && [ -n "$query_lock" ] && [ -n "$query_owner" ] && [ -n "$query_last" ]; then
  report 0 "queries: absent state resolves physically without creating files"
else
  report 1 "queries: absent state resolves physically without creating files" \
    "rc=$query_rc runs=$query_out worktree=$query_wt lock=$query_lock idle=$query_idle"
fi
query_bad=
for args in 'paths' 'paths bogus' 'paths lock last extra' 'paths run' \
            'paths run missing' 'paths lock missing' 'lock-status last' \
            'lock-status last extra'; do
  query_out=$(query $args 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$args: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ ! -e "$TMP/query-state" ]; then
  report 0 "queries: malformed arguments and missing runs fail without state mutation"
else
  report 1 "queries: malformed arguments and missing runs fail without state mutation" "$query_bad"
fi

# Three or more leading slashes are a local root, not a UNC host/share.
query_slash_root=ocannl-query-slashes-${TMP##*/}
query_out=$(OCANNL_TOOL_TEST_RUNS="///$query_slash_root/runs" \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
query_root=$(cd / && pwd -P)
if [ "$query_rc" = 0 ] && [ "$query_out" = "${query_root%/}/$query_slash_root/runs" ] \
   && [ ! -e "/$query_slash_root" ]; then
  report 0 "queries: redundant leading slashes preserve the local root without writes"
else
  report 1 "queries: redundant leading slashes preserve the local root without writes" \
    "rc=$query_rc runs=$query_out root=$query_root"
fi

case $(uname -s) in
  MSYS* | MINGW*) skip "queries: double slashes preserve the POSIX local root without writes" "Windows reserves UNC roots" ;;
  *)
    query_out=$(OCANNL_TOOL_TEST_RUNS="//$query_slash_root/runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "${query_root%/}/$query_slash_root/runs" ] \
       && [ ! -e "/$query_slash_root" ]; then
      report 0 "queries: double slashes preserve the POSIX local root without writes"
    else
      report 1 "queries: double slashes preserve the POSIX local root without writes" "rc=$query_rc runs=$query_out"
    fi ;;
esac

query_bad=
for query_break in $'\n' $'\r'; do
  query_break_root=$TMP/query-break${query_break}root
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_break_root" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] \
       || ! grep -q 'single-line paths' "$TMP/query-error" \
       || [ -e "$query_break_root" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ]; then
  report 0 "queries: CR and LF roots fail loudly without state writes"
else
  report 1 "queries: CR and LF roots fail loudly without state writes" "$query_bad"
fi

# A competing CDPATH location must neither redirect the identity nor leak cd's
# announcement into the one-path stdout contract.
mkdir -p "$repeat_root/query-relative" "$TMP/query-cdpath/query-relative"
query_out=$(CDPATH="$TMP/query-cdpath" OCANNL_TOOL_TEST_RUNS=query-relative/missing \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_wt/query-relative/missing" ] \
   && [ ! -e "$repeat_root/query-relative/missing" ] \
   && [ ! -e "$TMP/query-cdpath/query-relative/missing" ]; then
  report 0 "queries: CDPATH cannot redirect relative state roots or add stdout"
else
  report 1 "queries: CDPATH cannot redirect relative state roots or add stdout" \
    "rc=$query_rc runs=$query_out"
fi

# Physical normalization must agree with launch even when the caller uses a
# symlinked script or state root. MSYS may copy rather than create a symlink.
ln -s "$repeat_root" "$TMP/query-repo-link" 2>/dev/null
ln -s "$repeat_root" "$TMP/query-state-link" 2>/dev/null
if [ -L "$TMP/query-repo-link" ] && [ -L "$TMP/query-state-link" ]; then
  query_out=$(OCANNL_TOOL_TEST_RUNS="$TMP/query-state-link/nonexistent/runs" \
    "$TMP/query-repo-link/tools/test-run.sh" paths runs)
  query_rc=$?
  if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_wt/nonexistent/runs" ] \
     && [ ! -e "$repeat_root/nonexistent" ]; then
    report 0 "queries: symlink spellings identify the same physical worktree and state root"
  else
    report 1 "queries: symlink spellings identify the same physical worktree and state root" \
      "rc=$query_rc runs=$query_out expected=$query_wt/nonexistent/runs"
  fi
else
  skip "queries: symlink spellings identify the same physical worktree and state root" \
    "native symlinks unavailable"
fi

# Missing children of a symlink/.. prefix can differ between logical cd and
# mkdir's physical traversal. Refuse that prediction, but accept a resolved root.
mkdir -p "$TMP/query-parent-outside/child"
ln -s "$TMP/query-parent-outside/child" "$repeat_root/query-parent-link" 2>/dev/null
if [ -L "$repeat_root/query-parent-link" ]; then
  query_bad=
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS=query-parent-link/../fresh \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ ! -e "$TMP/query-parent-outside/fresh" ] && [ ! -e "$repeat_root/fresh" ] || query_bad="query created state"
  mkdir "$TMP/query-parent-outside/fresh"
  query_out=$(OCANNL_TOOL_TEST_RUNS=query-parent-link/../fresh \
    "$repeat_root/tools/test-run.sh" paths runs)
  query_rc=$?
  query_want=$(cd "$repeat_root" && cd query-parent-link/../fresh && pwd -P)
  if [ -z "$query_bad" ] && [ "$query_rc" = 0 ] && [ "$query_out" = "$query_want" ]; then
    report 0 "queries: ambiguous missing symlink parents refuse while resolved paths match launch"
  else
    report 1 "queries: ambiguous missing symlink parents refuse while resolved paths match launch" "$query_bad rc=$query_rc output=$query_out expected=$query_want"
  fi
else
  skip "queries: ambiguous missing symlink parents refuse while resolved paths match launch" "native symlinks unavailable"
fi

# A plain symlink name can hide a physical path ending in LF. Refuse it
# before command substitution trims that character into a different path.
case $(uname -s) in MSYS* | MINGW*) query_line_paths=0 ;; *) query_line_paths=1 ;; esac
if [ "$query_line_paths" = 1 ] && [ -L "$TMP/query-state-link" ]; then
  query_line_target=$TMP/query-physical$'\n'
  mkdir "$query_line_target"
  ln -s "$query_line_target" "$TMP/query-line-link"
  query_out=$(OCANNL_TOOL_TEST_RUNS="$TMP/query-line-link" \
    "$repeat_root/tools/test-run.sh" paths runs 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" = 2 ] && [ -z "$query_out" ] \
     && grep -q 'single-line paths' "$TMP/query-error"; then
    report 0 "queries: physical line breaks behind symlinks are refused before capture"
  else
    report 1 "queries: physical line breaks behind symlinks are refused before capture" "rc=$query_rc runs=$query_out"
  fi
else
  skip "queries: physical line breaks behind symlinks are refused before capture" "requires POSIX symlink filenames"
fi

# Exercise UNC preservation portably: the launch shell's physical-prefix
# reader reports a network root. The query must retain that exact prefix,
# not run it through a separate filesystem grammar that collapses // to /.
query_unc_root=$TMP/query-unc
mkdir "$query_unc_root"
query_unc_root=$(cd "$query_unc_root" && pwd -P)
cat >"$TMP/query-unc-env" <<'EOF'
pwd() {
  if [ "$PWD" = "$QUERY_TEST_UNC_ROOT" ]; then printf '//query-host/share\n';
  else builtin pwd "$@"; fi
}
EOF
query_out=$(BASH_ENV="$TMP/query-unc-env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
  OCANNL_TOOL_TEST_RUNS="$query_unc_root/missing/runs" \
  "$repeat_root/tools/test-run.sh" paths runs)
query_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = //query-host/share/missing/runs ] \
   && [ ! -e "$query_unc_root/missing" ]; then
  report 0 "queries: the launch shell's UNC prefix survives missing-suffix normalization"
else
  report 1 "queries: the launch shell's UNC prefix survives missing-suffix normalization" \
    "rc=$query_rc runs=$query_out"
fi

# Parent traversal is resolved only by the filesystem, never by a second
# path grammar. Test both sides for local and shell-reported UNC prefixes.
mkdir "$query_unc_root/existing"
for query_style in local unc; do
  query_env=$TMP/query-empty-env
  : >"$query_env"
  [ "$query_style" != unc ] || query_env=$TMP/query-unc-env
  query_want=$query_unc_root/runs
  [ "$query_style" != unc ] || query_want=//query-host/share/runs
  query_good=$(BASH_ENV="$query_env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
    OCANNL_TOOL_TEST_RUNS="$query_unc_root/existing/../runs" \
    "$repeat_root/tools/test-run.sh" paths runs)
  query_good_rc=$?
  query_bad=
  for query_command in 'paths runs' 'lock-status'; do
    query_out=$(BASH_ENV="$query_env" QUERY_TEST_UNC_ROOT="$query_unc_root" \
      OCANNL_TOOL_TEST_RUNS="$query_unc_root/missing/../../target" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] \
       || ! grep -q 'missing state path containing ..' "$TMP/query-error"; then
      query_bad="$query_command: rc=$query_rc runs=$query_out"; break
    fi
  done
  if [ "$query_good_rc" = 0 ] && [ "$query_good" = "$query_want" ] \
     && [ -z "$query_bad" ] && [ ! -e "$query_unc_root/missing" ] \
     && [ ! -e "$query_unc_root/runs" ] && [ ! -e "$TMP/target" ]; then
    report 0 "queries: $query_style existing parent traversal resolves; missing traversal refuses without writes"
  else
    report 1 "queries: $query_style existing parent traversal resolves; missing traversal refuses without writes" \
      "resolved=$query_good_rc:$query_good expected=$query_want $query_bad"
  fi
done

# MSYS accepts both /c/... and C:/... at launch. Its Perl File::Spec does
# not necessarily share that interpretation, so compare both against Bash.
case $(uname -s) in
  MSYS* | MINGW*)
    # Reach the drive root itself, not just an existing temporary prefix.
    query_drive=$(cygpath -m "$TMP")
    query_drive=${query_drive%%/*}/
    query_drive_physical=$(cd "$query_drive" && pwd -P)
    query_drive_runs=${query_drive}ocannl-query-root-${TMP##*/}
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_drive_runs" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_drive_physical/ocannl-query-root-${TMP##*/}" ] \
       && [ ! -e "$query_drive_runs" ]; then
      report 0 "queries: missing state directly below a drive root matches the launch shell without writes"
    else
      report 1 "queries: missing state directly below a drive root matches the launch shell without writes" \
        "rc=$query_rc runs=$query_out root=$query_drive_physical"
    fi
    query_native=$(cygpath -m "$query_expected")
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_expected" ] \
       && [ ! -e "$TMP/query-state" ]; then
      report 0 "queries: absent Windows drive spelling matches the launch shell"
    else
      report 1 "queries: absent Windows drive spelling matches the launch shell" \
        "rc=$query_rc runs=$query_out expected=$query_expected"
    fi
    query_native=$(cygpath -m "$query_unc_root")
    query_good=$(OCANNL_TOOL_TEST_RUNS="$query_native/existing/../runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_good_rc=$?
    # Native C:/ paths resolve missing/.. through Windows before Bash cd.
    # Match that actual launch-shell identity, not POSIX lexical semantics.
    query_native_parent=$(cd "$query_native/missing/.." && pwd -P)
    query_native_parent_rc=$?
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native/missing/../runs" \
      "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_good_rc" = 0 ] && [ "$query_good" = "$query_unc_root/runs" ] \
       && [ "$query_native_parent_rc" = 0 ] && [ "$query_rc" = 0 ] \
       && [ "$query_out" = "$query_native_parent/runs" ] \
       && [ ! -e "$query_unc_root/missing" ] && [ ! -e "$query_unc_root/runs" ]; then
      report 0 "queries: drive parent traversal matches the native launch shell without writes"
    else
      report 1 "queries: drive parent traversal matches the native launch shell without writes" \
        "resolved=$query_good_rc:$query_good parent=$query_native_parent_rc:$query_native_parent runs=$query_rc:$query_out"
    fi
    ;;
  *) skip "queries: missing state directly below a drive root matches the launch shell without writes" "requires Windows drive paths"
     skip "queries: absent Windows drive spelling matches the launch shell" "requires Windows drive paths"
     skip "queries: drive parent traversal matches the native launch shell without writes" "requires Windows drive paths" ;;
esac

# The pointer returned by paths last must feed the SAME resolver as status and
# stop; the published fixture has deliberately no live pid and no verdict.
mkdir -p "$query_expected/fixture"
query_runs=$query_expected
query_dir=$query_expected/fixture
printf 'fixture\n' >"$query_dir/cmd"
printf '0\n' >"$query_dir/cap"
printf '%s\n' "$query_wt" >"$query_dir/wt"
printf '%s\n' "$query_runs" >"$query_dir/runs"
printf '%s\n' "$query_dir" >"$query_last"
case $(uname -s) in
  MSYS* | MINGW*)
    query_native=$(cygpath -m "$query_runs")
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_native" "$repeat_root/tools/test-run.sh" paths runs)
    query_rc=$?
    if [ "$query_rc" = 0 ] && [ "$query_out" = "$(cd "$query_native" && pwd -P)" ]; then
      report 0 "queries: existing Windows drive spelling matches the launch shell"
    else
      report 1 "queries: existing Windows drive spelling matches the launch shell" \
        "rc=$query_rc runs=$query_out"
    fi
    ;;
  *) skip "queries: existing Windows drive spelling matches the launch shell" "requires Windows drive paths" ;;
esac
query_bad=
for ref in last fixture "$query_dir"; do
  [ "$(query paths run "$ref")" = "$query_dir" ] \
    && [ "$(query paths lock "$ref")" = "$query_lock" ] \
    && [ "$(query paths owner "$ref")" = "$query_owner" ] \
    && [ "$(query paths worktree "$ref")" = "$query_wt" ] \
    && [ "$(query paths runs "$ref")" = "$query_runs" ] \
    && [ "$(query paths last "$ref")" = "$query_last" ] \
    || query_bad="$ref did not resolve the published fixture"
done
query_out=$(query lock-status last); query_rc=$?
if [ -z "$query_bad" ] && [ "$query_rc" = 0 ] && [ "$query_out" = idle ] \
   && [ ! -e "$query_lock" ] && [ ! -e "$query_owner" ]; then
  report 0 "queries: last, bare id and absolute run share recorded paths without creating locks"
else
  report 1 "queries: last, bare id and absolute run share recorded paths without creating locks" "$query_bad"
fi

# Absolute references remain useful after the caller's ambient root changes.
# Omitted and bare-id references still require that current root to be valid.
printf 'not a directory\n' >"$TMP/query-ambient-file"
query_bad=
for query_ambient in "$TMP/query-ambient-file" "$TMP/query-ambient-missing/../runs"; do
  for field in run worktree runs lock owner last; do
    case $field in
      run) query_want=$query_dir ;; worktree) query_want=$query_wt ;;
      runs) query_want=$query_runs ;; lock) query_want=$query_lock ;;
      owner) query_want=$query_owner ;; last) query_want=$query_last ;;
    esac
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
      "$repeat_root/tools/test-run.sh" paths "$field" "$query_dir" 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 0 ] || [ "$query_out" != "$query_want" ]; then
      query_bad="$field: rc=$query_rc output=$query_out expected=$query_want"; break
    fi
  done
  query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
    "$repeat_root/tools/test-run.sh" lock-status "$query_dir" 2>"$TMP/query-error")
  query_rc=$?
  [ "$query_rc" = 0 ] && [ "$query_out" = idle ] || query_bad="lock-status: $query_rc:$query_out"
  for query_command in 'paths runs' 'paths runs fixture' 'lock-status'; do
    query_out=$(OCANNL_TOOL_TEST_RUNS="$query_ambient" \
      "$repeat_root/tools/test-run.sh" $query_command 2>"$TMP/query-error")
    query_rc=$?
    [ "$query_rc" = 2 ] && [ -z "$query_out" ] || query_bad="$query_command: $query_rc:$query_out"
  done
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ -f "$TMP/query-ambient-file" ] \
   && [ ! -e "$TMP/query-ambient-missing" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: absolute runs ignore unrelated ambient state while current references validate it"
else
  report 1 "queries: absolute runs ignore unrelated ambient state while current references validate it" "$query_bad"
fi

# A missing absolute reference must not select a matching state-root suffix.
query_missing_absolute=/ocannl-query-missing-$$
mkdir -p "$query_runs/$query_missing_absolute"
cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$query_runs/$query_missing_absolute/"
query_bad=
for query_command in 'paths run' 'paths lock' 'lock-status'; do
  query_out=$(query $query_command "$query_missing_absolute" 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$query_command: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ ! -e "$query_missing_absolute" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: missing absolute references never fall back under the state root"
else
  report 1 "queries: missing absolute references never fall back under the state root" "$query_bad"
fi

# Explicit references use the same host-absolute grammar as recorded paths.
case $(uname -s) in
  MSYS* | MINGW*) query_drive_ref=$(cygpath -m "$query_dir"); query_drive_ref_rc=0 ;;
  *)
    query_drive_ref=C:/fixture
    mkdir -p "$repeat_root/$query_drive_ref"
    cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$repeat_root/$query_drive_ref/"
    query_drive_ref_rc=2 ;;
esac
query_out=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths run "$query_drive_ref" 2>"$TMP/query-error")
query_rc=$?
query_drive_lock=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" lock-status "$query_drive_ref" 2>"$TMP/query-error")
query_drive_lock_rc=$?
query_bad=
[ "$query_rc" = "$query_drive_ref_rc" ] && [ "$query_drive_lock_rc" = "$query_drive_ref_rc" ] || query_bad="rc=$query_rc lock=$query_drive_lock_rc"
if [ "$query_drive_ref_rc" = 0 ]; then
  [ "$query_out" = "$query_dir" ] && [ "$query_drive_lock" = idle ] || query_bad="native reference changed identity"
else
  [ -z "$query_out" ] && [ -z "$query_drive_lock" ] || query_bad="relative reference bypassed current state"
  [ "$(query paths run "$query_drive_ref")" = "$query_wt/$query_drive_ref" ] || query_bad="relative reference no longer resolves with valid state"
fi
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: explicit drive references follow host absolute-path semantics"
else
  report 1 "queries: explicit drive references follow host absolute-path semantics" "$query_bad"
fi

query_out=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths runs "$query_dir" 2>"$TMP/query-error")
query_rc=$?
query_home_lock=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" lock-status "$query_dir" 2>"$TMP/query-error")
query_home_lock_rc=$?
query_home_current=$(env -u HOME -u OCANNL_TOOL_TEST_RUNS \
  "$repeat_root/tools/test-run.sh" paths runs 2>"$TMP/query-error")
query_home_current_rc=$?
if [ "$query_rc" = 0 ] && [ "$query_out" = "$query_runs" ] \
   && [ "$query_home_lock_rc" = 0 ] && [ "$query_home_lock" = idle ] \
   && [ "$query_home_current_rc" = 2 ] && [ -z "$query_home_current" ] \
   && [ ! -e "$query_lock" ]; then
  report 0 "queries: HOME is required only when its current-root default is used"
else
  report 1 "queries: HOME is required only when its current-root default is used" \
    "runs=$query_rc:$query_out lock=$query_home_lock_rc:$query_home_lock current=$query_home_current_rc:$query_home_current"
fi

# A held lock needs no process metadata to be reported held. Run the holder
# synchronously; Perl closes its nonstandard descriptors when execing the query.
query_out=$(perl -MFcntl=:flock -e '
  open my $fh, ">>", $ARGV[0] or die $!;
  flock($fh, LOCK_EX | LOCK_NB) or die $!;
  system @ARGV[1..$#ARGV];
  exit($? >> 8);
' "$query_lock" env OCANNL_TOOL_TEST_RUNS="$query_runs" \
  "$repeat_root/tools/test-run.sh" lock-status last)
query_rc=$?
query_after=$(query lock-status); query_after_rc=$?
if [ "$query_rc" = 3 ] && [ "$query_out" = held ] \
   && [ "$query_after_rc" = 0 ] && [ "$query_after" = idle ] \
   && [ ! -s "$query_lock" ]; then
  report 0 "queries: occupied and released locks have distinct statuses without writes"
else
  report 1 "queries: occupied and released locks have distinct statuses without writes" \
    "held=$query_rc:$query_out released=$query_after_rc:$query_after"
fi
rm "$query_lock"
mkdir "$query_lock"
query_out=$(query lock-status 2>"$TMP/query-error"); query_rc=$?
if [ "$query_rc" = 2 ] && [ -z "$query_out" ] && [ -d "$query_lock" ]; then
  report 0 "queries: an uninspectable lock is an error, never idle"
else
  report 1 "queries: an uninspectable lock is an error, never idle" "rc=$query_rc output=$query_out"
fi
rmdir "$query_lock"
# Drive-form records are absolute only on Windows. Both recorded path fields
# follow host semantics rather than accepting a foreign relative spelling.
query_bad=
for query_record_field in wt runs; do
  case $query_record_field in wt) query_record_path=$query_wt; query_field=worktree ;; runs) query_record_path=$query_runs; query_field=runs ;; esac
  case $(uname -s) in
    MSYS* | MINGW*) query_drive_record=$(cygpath -m "$query_record_path"); query_drive_rc=0 ;;
    *) query_drive_record=C:/foreign-state; query_drive_rc=2 ;;
  esac
  printf '%s\n' "$query_drive_record" >"$query_dir/$query_record_field"
  query_out=$(query paths "$query_field" "$query_dir" 2>"$TMP/query-error")
  query_rc=$?
  query_drive_lock=$(query lock-status "$query_dir" 2>"$TMP/query-error")
  query_drive_lock_rc=$?
  if [ "$query_rc" != "$query_drive_rc" ] || [ "$query_drive_lock_rc" != "$query_drive_rc" ]; then
    query_bad="$query_record_field: path=$query_rc:$query_out lock=$query_drive_lock_rc:$query_drive_lock"
  elif [ "$query_drive_rc" = 0 ]; then
    [ "$query_out" = "$query_drive_record" ] && [ "$query_drive_lock" = idle ] || query_bad="native record changed identity"
  else
    [ -z "$query_out" ] && [ -z "$query_drive_lock" ] || query_bad="foreign record produced output"
  fi
  printf '%s\n' "$query_record_path" >"$query_dir/$query_record_field"
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: recorded drive paths follow host absolute-path semantics"
else
  report 1 "queries: recorded drive paths follow host absolute-path semantics" "$query_bad"
fi

# Missing runs metadata is legacy; present broken metadata is an error.
query_bad=
for query_metadata in empty relative multiline directory trailing_blank unterminated nul; do
  rm -f "$query_dir/runs"
  case $query_metadata in
    empty) : >"$query_dir/runs" ;;
    relative) printf 'relative/runs\n' >"$query_dir/runs" ;;
    multiline) printf '%s\nextra\n' "$query_runs" >"$query_dir/runs" ;;
    directory) mkdir "$query_dir/runs" ;;
    trailing_blank) printf '%s\n\n' "$query_runs" >"$query_dir/runs" ;;
    unterminated) printf '%s' "$query_runs" >"$query_dir/runs" ;;
    nul) printf '%s\000\n' "$query_runs" >"$query_dir/runs" ;;
  esac
  for query_ref in last "$query_dir"; do
    for query_command in 'paths runs' 'paths lock' 'lock-status'; do
      query_out=$(query $query_command "$query_ref" 2>"$TMP/query-error")
      query_rc=$?
      if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
        query_bad="$query_metadata $query_command $query_ref: rc=$query_rc output=$query_out"; break
      fi
    done
    [ -z "$query_bad" ] || break
  done
  [ ! -d "$query_dir/runs" ] || rmdir "$query_dir/runs"
  [ -z "$query_bad" ] || break
done
if [ -z "$query_bad" ] && [ ! -e "$query_lock" ] && [ ! -e "$query_owner" ]; then
  report 0 "queries: present malformed state metadata is never mistaken for a legacy run"
else
  report 1 "queries: present malformed state metadata is never mistaken for a legacy run" "$query_bad"
fi
printf '%s\n' "$query_runs" >"$query_dir/runs"

# The regular last pointer is a record too; shell capture must not repair it.
query_bad=
for query_pointer_kind in trailing_blank unterminated; do
  case $query_pointer_kind in
    trailing_blank) printf '%s\n\n' "$query_dir" >"$query_last" ;;
    unterminated) printf '%s' "$query_dir" >"$query_last" ;;
  esac
  for query_command in 'paths run' 'paths runs' 'lock-status'; do
    query_out=$(query $query_command last 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_pointer_kind $query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -z "$query_bad" ] || break
done
printf '%s\n' "$query_dir" >"$query_last"
if [ -z "$query_bad" ] && [ "$(query paths run last)" = "$query_dir" ] \
   && [ "$(query lock-status last)" = idle ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: last pointer records are validated before capture"
else
  report 1 "queries: last pointer records are validated before capture" "$query_bad"
fi

# A copied directory at LAST is not a legacy symlink or a regular pointer.
rm "$query_last"
mkdir "$query_last"
cp "$query_dir/cmd" "$query_dir/cap" "$query_dir/wt" "$query_dir/runs" "$query_last/"
query_bad=
for query_command in 'paths run' 'paths lock' 'lock-status'; do
  query_out=$(query $query_command last 2>"$TMP/query-error")
  query_rc=$?
  if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
    query_bad="$query_command: rc=$query_rc output=$query_out"; break
  fi
done
if [ -z "$query_bad" ] && [ -d "$query_last" ] && [ ! -e "$query_lock" ]; then
  report 0 "queries: a directory at the last pointer is refused without mutation"
else
  report 1 "queries: a directory at the last pointer is refused without mutation" "$query_bad"
fi
rm "$query_last/cmd" "$query_last/cap" "$query_last/wt" "$query_last/runs"
rmdir "$query_last"
printf '%s\n' "$query_dir" >"$query_last"

# Legacy pointer symlinks must preserve physical identity before capture too.
if [ "$query_line_paths" = 1 ] && [ -L "$TMP/query-state-link" ]; then
  query_bad=
  query_pointer_target=$query_dir$'\n'
  mkdir "$query_pointer_target"
  rm "$query_last"
  ln -s "$query_pointer_target" "$query_last"
  for query_command in 'paths run' 'paths lock' 'lock-status'; do
    query_out=$(query $query_command last 2>"$TMP/query-error")
    query_rc=$?
    if [ "$query_rc" != 2 ] || [ -n "$query_out" ] || [ ! -s "$TMP/query-error" ]; then
      query_bad="$query_command: rc=$query_rc output=$query_out"; break
    fi
  done
  [ -L "$query_last" ] || query_bad="query replaced the legacy pointer"
  rm "$query_last"
  ln -s "$query_dir" "$query_last"
  if [ -z "$query_bad" ] && [ "$(query paths run last)" = "$query_dir" ] \
     && [ "$(query lock-status last)" = idle ] && [ -L "$query_last" ] \
     && [ ! -e "$query_lock" ]; then
    report 0 "queries: legacy pointer symlinks retain physical identity before capture"
  else
    report 1 "queries: legacy pointer symlinks retain physical identity before capture" "$query_bad"
  fi
  rm "$query_last"
  printf '%s\n' "$query_dir" >"$query_last"
else
  skip "queries: legacy pointer symlinks retain physical identity before capture" "requires POSIX symlink filenames"
fi

# Legacy paths come from the recorded worktree, not the querying script's root.
rm "$query_dir/runs"
printf '%s\n' "$STOP_WT" >"$query_dir/wt"
if [ "$(query paths lock last)" = "$STOP_WT/.test-run.lock" ] \
   && [ "$(query paths owner last)" = "$STOP_WT/.test-run.lock.owner" ]; then
  report 0 "queries: legacy runs retain their recorded in-tree lock paths"
else
  report 1 "queries: legacy runs retain their recorded in-tree lock paths"
fi
}
harness_leg_run '39: read-only paths and lock-status queries #queries' leg_39

shared_through argv_probe

# Every subcommand that takes options, and both spellings of a value option:
# the guard sits on one code path, but a future refactor that splits it must
# not be able to leave one entry point forwarding.
leg_26_28() {
misplaced_label="an option after the dune arguments is refused before dune runs"
misplaced_detail=
for probe in "run:run build @cheap --cap 900" \
             "run-test-cap:run build @cheap --test-cap 9" \
             "start:start build @cheap --cap=900" \
             "repeat:repeat 2 build @cheap --alone"; do
  argv_probe "misplaced-${probe%%:*}" ${probe#*:}
  # Exit 2 is the usage code; the message must name the option AND the order,
  # or the reader is left where the unknown-option error left them.
  case $argv_rc in
    2) ;;
    *) misplaced_detail="${probe#*:}: exit $argv_rc (want 2): $argv_err" ; break ;;
  esac
  case $argv_err in
    *"belongs before the dune arguments"*"tools/test-run.sh"*) ;;
    *) misplaced_detail="${probe#*:}: unhelpful refusal: $argv_err" ; break ;;
  esac
  if [ -n "$argv_calls" ]; then
    misplaced_detail="${probe#*:}: dune was invoked anyway: $argv_calls"
    break
  fi
done
if [ -z "$misplaced_detail" ]; then
  report 0 "$misplaced_label"
else
  report 1 "$misplaced_label" "$misplaced_detail"
fi

# The control for the leg above: the refusal must be about PLACEMENT, not about
# the word. A guard that also rejected the documented form would pass leg 26.
argv_probe cap-placed run --cap 900 build @cheap
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ]; then
  report 0 "a correctly placed --cap is consumed and never reaches dune"
else
  report 1 "a correctly placed --cap is consumed and never reaches dune" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi

# Past dune's own separator the word is an executable's argument -- the caller's
# business, and the one place `--cap` after the target can be meant.
argv_probe cap-after-sep run exec ./prog.exe -- --cap 900
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec ./prog.exe -- --cap 900" ]; then
  report 0 "an option past dune's -- reaches the program untouched"
else
  report 1 "an option past dune's -- reaches the program untouched" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
}
harness_leg_run '26-28: options on the wrong side of the dune arguments #options' leg_26_28
shared_through synthetic_trace

# The guard from leg 26 is not made redundant by the digest: it knows the
# correct order and refuses without spawning dune. With a fixture that WOULD
# refuse, the guard's refusal is the one that arrives, and the calls file --
# empty here, one line for the digest's refusal -- is what separates the two
# exit-2 outcomes.
leg_37() {
argv_mode=usage_option argv_probe guard-before-dune run build @cheap --cap 900
argv_mode=
if [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] \
   && case $argv_err in *"belongs before the dune arguments"*) true ;; *) false ;; esac \
   && case $argv_out in *"INVOCATION REFUSED"*) false ;; *) true ;; esac; then
  report 0 "the --cap guard still refuses before dune is spawned"
else
  report 1 "the --cap guard still refuses before dune is spawned" \
    "exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err; stdout: $argv_out"
fi
}
harness_leg_run '37: the --cap guard refuses before dune is spawned #options' leg_37
shared_through renamed_dune

# ---------------------------------------------------------------------------
# Leg 89: an alias dune does not know is refused before the slot (gh-ocannl-1448)
# ---------------------------------------------------------------------------
# A misspelled per-test alias used to be found by dune itself, after the
# launch had resolved the batch's backends, taken the fleet's slot and probed
# the device. The fixture dune answers `dune show aliases` from a table, as
# dune 3.24 does (see its `show` branch), for a directory `al` with an alias of
# its own and a subdirectory `al/sub` with another; each probe runs under a
# GPU slot, so a check that came too late would show in the slot's calls, the
# reachability tool's (the backends' resolution) and dune's. The table empty
# -- an answer that is not dune's -- leaves every alias to dune. Where a real
# dune is at hand, the same refusals and acceptances against its own listing.
# Both, plain and under CLICOLOR_FORCE=1, whose escapes around dune's `Error`
# tag once made the answer inconclusive (gh-ocannl-1591).
leg_89() {
alias_label="an alias dune does not know is ALIAS REFUSED (exit 2) before the backends, the slot and dune's build; known ones, recursive or not, run"
alias_open_label="an answer that is not dune's leaves the aliases to dune"
alias_real_label="against a real dune: an unknown alias is refused, known ones run"
alias_detail=
alias_table=$TMP/alias-table
mkdir -p "$repeat_root/al/sub"
# `al/virt` has no dune file: a `(subdir ...)` stanza of al's, nested in
# another, behind a comment and through a quoted atom, gives it its rules;
# `al/alt` has a `dune-file`, the name a project may accept instead.
mkdir -p "$repeat_root/al/alt"
cat >"$repeat_root/al/dune" <<'DUNE'
; (subdir commented (rule (alias not_this)))
(rule (alias runtest-yes) (action (progn)))
(subdir "virt" (subdir inner (rule (alias vinner) (action (progn))))
 (rule (alias vdeep) (action (progn))))
DUNE
printf '(rule (alias deep) (action (progn)))\n' >"$repeat_root/al/sub/dune"
printf '(rule (alias altname) (action (progn)))\n' >"$repeat_root/al/alt/dune-file"
printf '%s\n' . 'al runtest-yes' 'al/alt altname' 'al/sub deep' 'al/virt vdeep' 'al/virt/inner vinner' >"$alias_table"
alias_probe() { # tag argv...: a run under a GPU slot, with the table answering
  REPEAT_TEST_ALIASES=$alias_table REPEAT_TEST_SHOW_CALLS=$TMP/$1.show slot_probe "$1" hold cuda cc "${@:2}"
  alias_reach=$(cat "$TMP/$1.reach")
}
: >"$TMP/alias-unknown.show"
alias_probe alias-unknown run build @al/runtest-no_such
{ [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && [ -z "$slot_calls" ] && [ -z "$alias_reach" ]; } ||
  alias_detail="unknown: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; reach calls: ${alias_reach:-<none>}"
if [ -z "$alias_detail" ]; then
  case $argv_out in
    *"verdict: ALIAS REFUSED (dune knows no such alias; nothing ran) (exit 2)"*"alias refused: @al/runtest-no_such: neither al nor any directory below it defines an alias runtest-no_such"*) ;;
    *) alias_detail="unknown: digest: $argv_out" ;;
  esac
fi
# It asked for `al` first, then for the directories below it, the subdir
# stanzas' included.
[ -n "$alias_detail" ] || [ "$(cat "$TMP/alias-unknown.show")" = "al
al/alt al/sub al/virt al/virt/inner" ] || alias_detail="unknown: the listings asked for: $(tr '\n' '|' <"$TMP/alias-unknown.show")"
# Refused: one under @@ that only a subdirectory defines, one whose directory
# dune does not know (beside a known one: dune then answers with nothing but
# the error, and the known one is asked again), and a root one nothing defines.
for probe in "@@al/deep|al defines no alias deep" \
             "-j 2 --force @al/runtest-no_such|neither al nor any directory below it" \
             "@nowhere/x @al/runtest-yes|@nowhere/x: dune knows no directory nowhere" \
             "@./al//runtest-no_such|neither al nor any directory below it" \
             "@runtest-no_such|neither . nor any directory below it defines an alias runtest-no_such"; do
  [ -z "$alias_detail" ] || break
  alias_args=${probe%%|*} alias_want=${probe#*|}
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  alias_probe alias-refused run build $alias_args
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && [ -z "$slot_calls" ] &&
    grep -qF -- "alias refused: " <<<"$argv_out" && grep -qF -- "$alias_want" <<<"$argv_out"; } ||
    alias_detail="$alias_args: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; digest: $argv_out"
  if [ -z "$alias_detail" ] && grep -q 'runtest-yes' <<<"$(grep 'alias refused' <<<"$argv_out")"; then
    alias_detail="$alias_args: a known alias was refused too: $argv_out"
  fi
done
# Where dune is made to color (CLICOLOR_FORCE=1, which setup-ocaml exports
# job-wide), its `Error` tag comes wrapped in escapes, and the answer is as
# conclusive as the plain one (gh-ocannl-1591): both, whatever the ambient
# setting.
for alias_color in 1 0; do
  [ -z "$alias_detail" ] || break
  CLICOLOR_FORCE=$alias_color alias_probe alias-color run build @nowhere/x @al/runtest-yes
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && [ -z "$slot_calls" ] &&
    [ "$(grep -c '^alias refused: ' <<<"$argv_out")" = 1 ] &&
    grep -qF -- "alias refused: @nowhere/x: dune knows no directory nowhere" <<<"$argv_out"; } ||
    alias_detail="CLICOLOR_FORCE=$alias_color @nowhere/x @al/runtest-yes: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; digest: $argv_out"
done
# Known: in its own directory, under @@ and @, below it, at the root, and
# past dune's own `--` nothing is asked about; nor where an option the check
# does not know may change dune's alias set (a profile), or the operand names
# a build context (`@_build/default/...`): those are dune's to judge.
for probe in "@al/runtest-yes" "@@al/runtest-yes" "@al/deep" "@deep" "@al/sub/deep @@al/sub/deep" "@al/./deep" \
             "@./al/runtest-yes" "@al/runtest-yes -- @not/asked" \
             "--profile release @al/runtest-no_such" "@_build/default/al/runtest-no_such" \
             "@al/vdeep" "@vinner" "@al/altname" "--build-dir=out @out/default/al/runtest-no_such"; do
  [ -z "$alias_detail" ] || break
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  alias_probe alias-known run build $probe
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build $probe" ] &&
    [ "$(grep -c '' <<<"$slot_calls")" = 1 ]; } ||
    alias_detail="$probe: exit $argv_rc (want 0); dune calls: ${argv_calls:-<none>}; slot calls: ${slot_calls:-<none>}; digest: $argv_out"
done
# `plan` says run would refuse, and a `repeat` runs no iteration.
if [ -z "$alias_detail" ]; then
  alias_probe alias-plan plan build @al/runtest-no_such
  case $argv_out in
    *"aliases: run would refuse this batch (ALIAS REFUSED, exit 2)"*"@al/runtest-no_such: neither al"*) ;;
    *) alias_detail="plan: exit $argv_rc; output: $argv_out" ;;
  esac
fi
if [ -z "$alias_detail" ]; then
  REPEAT_TEST_ALIASES=$alias_table repeat_probe alias-repeat stable 2 build @al/runtest-no_such
  { [ "$repeat_rc" = 2 ] && [ -z "$(cat "$TMP/alias-repeat.calls")" ] &&
    grep -qF "ALIAS REFUSED: @al/runtest-no_such" <<<"$repeat_out"; } ||
    alias_detail="repeat: exit $repeat_rc (want 2); dune calls: $(cat "$TMP/alias-repeat.calls"); output: $repeat_out"
fi
# A tree the census cannot list completely leaves the whole batch to dune
# before any query, even an @@ miss in the operand's own directory: a dune file
# the reader cannot read (unbalanced, or holding a `#|` or `#;`, which dune
# versions read differently), or one importing another file anywhere -- a
# `dynamic_include`, which loading its directory would build, or a static
# `include`, which can import one -- also behind a comment or as a quoted head.
for probe in "unread|(rule (alias odd)" "block|(rule (action (run echo #|x)))" \
             "include|(subdir x (include rules.inc))" "dynamic|(subdir x (dynamic_include gen/rules.inc))" \
             "commented|(; a comment
 dynamic_include gen/rules.inc)" "quoted|(\"include\" rules.inc)"; do
  [ -z "$alias_detail" ] || break
  mkdir -p "$repeat_root/al/odd"
  printf '%s\n' "${probe#*|}" >"$repeat_root/al/odd/dune"
  alias_probe "alias-${probe%%|*}" run build @al/runtest-no_such
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @al/runtest-no_such" ]; } ||
    alias_detail="${probe%%|*} below: exit $argv_rc (want 0); dune calls: ${argv_calls:-<none>}; digest: $argv_out"
  if [ -z "$alias_detail" ]; then
    alias_want=0
    : >"$TMP/alias-${probe%%|*}-own.show"
    alias_probe "alias-${probe%%|*}-own" run build @@al/runtest-no_such
    [ "$argv_rc" = "$alias_want" ] || alias_detail="${probe%%|*} below, @@: exit $argv_rc (want $alias_want): $argv_out"
    [ -n "$alias_detail" ] || [ ! -s "$TMP/alias-${probe%%|*}-own.show" ] ||
      alias_detail="${probe%%|*} below: dune was asked for a listing: $(cat "$TMP/alias-${probe%%|*}-own.show")"
  fi
  rm -rf "$repeat_root/al/odd"
done
if [ -z "$alias_detail" ]; then report 0 "$alias_label"; else report 1 "$alias_label" "$alias_detail"; fi
# The fixture dune with no table prints nothing for the listing: not dune's
# answer, so the check decides nothing and dune runs the batch.
slot_probe alias-open hold cuda cc run build @al/runtest-no_such
if [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @al/runtest-no_such" ]; then
  report 0 "$alias_open_label"
else
  report 1 "$alias_open_label" "exit $argv_rc (want 0); dune calls: ${argv_calls:-<none>}; digest: $argv_out"
fi
rm -rf "$repeat_root/al"
# A real dune's listing, in a project of its own: the format the check reads
# is dune's, not only the fixture's.
alias_real=
! harness_real_dune || alias_real=$harness_dune
if [ -z "$alias_real" ]; then
  skip "$alias_real_label" "$harness_dune_absent"
else
  alias_root=$TMP/alias-real
  mkdir -p "$alias_root/tools" "$alias_root/scripts" "$alias_root/al/sub"
  cp "$SRC" "$alias_root/tools/test-run.sh"
  if ! stage_sourced "$alias_root"; then
    report 1 "$alias_real_label" "cannot stage what test-run.sh sources: $stage_error"
  else
    printf '(lang dune 3.20)\n' >"$alias_root/dune-project"
    printf '(rule (alias runtest-yes) (action (progn)))\n(subdir virt (rule (alias vdeep) (action (progn))))\n' \
      >"$alias_root/al/dune"
    printf '(rule (alias deep) (action (progn)))\n' >"$alias_root/al/sub/dune"
    # A checkout, so the directories come from git's listing, one of them under
    # a name git quotes unless asked for it verbatim (Codex review round 5).
    mkdir -p "$alias_root/al/café"
    printf '(rule (alias cafealias) (action (progn)))\n' >"$alias_root/al/café/dune"
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git -C "$alias_root" init -q
    alias_real_detail=
    alias_real_run() { # tag color argv...: CLICOLOR_FORCE set to the color, whatever the ambient one
      local rc=0
      (cd "$alias_root" && export CLICOLOR_FORCE="$2" && PATH=$(dirname "$alias_real"):$PATH \
        OCANNL_TOOL_TEST_RUNS=$TMP/$1-runs tools/test-run.sh run "${@:3}") >"$TMP/$1.out" 2>&1 || rc=$?
      alias_real_rc=$rc
    }
    # Each probe plain and where dune is made to color, its `Error` tag then
    # wrapped in escapes (gh-ocannl-1591).
    for probe in "2|build @al/runtest-no_such @al/deep|alias refused: @al/runtest-no_such: neither al" \
                 "2|build @@al/deep|alias refused: @@al/deep: al defines no alias deep" \
                 "2|build @nowhere/x @al/runtest-yes|alias refused: @nowhere/x: dune knows no directory nowhere" \
                 "0|build @al/deep @@al/runtest-yes @deep @al/vdeep @vdeep @al/cafealias|verdict: pass"; do
      IFS='|' read -r want args text <<<"$probe"
      for alias_color in 0 1; do
        # shellcheck disable=SC2086 # the probe's operands, split on purpose
        alias_real_run alias-real "$alias_color" $args
        { [ "$alias_real_rc" = "$want" ] && grep -qF -- "$text" "$TMP/alias-real.out"; } ||
          { alias_real_detail="CLICOLOR_FORCE=$alias_color $args: exit $alias_real_rc (want $want): $(tr '\n' '|' <"$TMP/alias-real.out")"; break 2; }
        [ "$want" = 0 ] || [ "$(grep -c '^alias refused: ' "$TMP/alias-real.out")" = 1 ] ||
          { alias_real_detail="CLICOLOR_FORCE=$alias_color $args: not exactly one refusal: $(tr '\n' '|' <"$TMP/alias-real.out")"; break 2; }
      done
    done
    if [ -z "$alias_real_detail" ]; then report 0 "$alias_real_label"; else report 1 "$alias_real_label" "$alias_real_detail"; fi
  fi
fi
}
harness_leg_run '89: an alias dune does not know is refused before the slot #options' leg_89

# ---------------------------------------------------------------------------
# Leg 103: a named test's sibling goldens are named, and never added to the batch
# ---------------------------------------------------------------------------
# `runtest-NAME` does not reach `runtest-NAME-VARIANT`: promoting after a run
# naming only NAME left two goldens stale, which only CI showed. Directory `sb`
# holds the test `t` with two siblings (one an `.ll.expected`), a variant with no
# golden, a golden with no alias and an alias a golden's name merely extends; `q`
# with a golden-less variant only; `s` under slow; `sb/deep` a test only a
# recursive operand finds.
leg_103() {
sib_label="plan, the launch and the digest name a test's sibling goldens, and dune gets the batch as given"
sib_quiet_label="a variant with no golden, one the batch names, and a listing not read say nothing"
sib_control_label="negative controls: the golden test, its stem's dot and the batch's own names decide the note"
sib_note=$(sed -n "s/^alias_siblings_label='\(.*\)'$/\1/p" "$SRC")
[ -n "$sib_note" ] || { harness_abandon "no alias_siblings_label in $SRC"; return; }
sib_table=$TMP/sib-table
mkdir -p "$repeat_root/sb/deep"
printf '(rule (alias runtest-t) (action (progn)))\n' >"$repeat_root/sb/dune"
printf '(rule (alias runtest-d) (action (progn)))\n' >"$repeat_root/sb/deep/dune"
for f in t t-own t-ll.ll t-orphan s-own deep/d deep/d-own; do : >"$repeat_root/sb/$f.expected"; done
printf '%s\n' . 'sb runtest-t' 'sb runtest-t-own' 'sb runtest-t-ll' 'sb runtest-t-bare' 'sb runtest-t-ow' \
  'sb runtest-q' 'sb runtest-q-bare' 'sb slow-s' 'sb slow-s-own' 'sb/deep runtest-d' 'sb/deep runtest-d-own' >"$sib_table"
sib_probe() { REPEAT_TEST_ALIASES=$sib_table argv_probe "$@"; }
sib_detail=
for probe in "@@sb/runtest-t|@sb/runtest-t: @sb/runtest-t-ll @sb/runtest-t-own" \
             "@@sb/runtest-t @@sb/runtest-t-own|@sb/runtest-t: @sb/runtest-t-ll" \
             "@sb/runtest-d|@sb/deep/runtest-d: @sb/deep/runtest-d-own" \
             "@@sb/slow-s|@sb/slow-s: @sb/slow-s-own"; do
  IFS='|' read -r args want <<<"$probe"
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  sib_probe sib-run run build $args
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build $args" ]; } ||
    { sib_detail="run $args: exit $argv_rc; dune calls: ${argv_calls:-<none>}"; break; }
  grep -qxF -- "$sib_note: $want" <<<"$argv_out" ||
    { sib_detail="run $args: the digest does not carry '$want': $argv_out"; break; }
  grep -qxF -- "test-run: $sib_note: $want" <<<"$argv_err" ||
    { sib_detail="run $args: the launch does not carry '$want': $argv_err"; break; }
  [ "$(cat "$argv_dir/sibling-goldens" 2>/dev/null)" = "$want" ] ||
    { sib_detail="run $args: sibling-goldens: $(cat "$argv_dir/sibling-goldens" 2>/dev/null)"; break; }
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  sib_probe sib-plan plan build $args
  { grep -qxF -- "$sib_note:" <<<"$argv_out" && grep -qxF -- "  $want" <<<"$argv_out"; } ||
    { sib_detail="plan $args: exit $argv_rc: $argv_out"; break; }
done
if [ -z "$sib_detail" ]; then report 0 "$sib_label"; else report 1 "$sib_label" "$sib_detail"; fi

sib_detail=
for args in "@@sb/runtest-q" "@@sb/runtest-t @@sb/runtest-t-own @@sb/runtest-t-ll" "@sb/runtest-d @sb/runtest-d-own" \
            "@@sb/runtest-t-own" "--profile release @@sb/runtest-t"; do
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  sib_probe sib-quiet run build $args
  { [ "$argv_rc" = 0 ] && [ ! -e "$argv_dir/sibling-goldens" ] &&
    ! grep -qF -- "$sib_note" <<<"$argv_out$argv_err"; } ||
    { sib_detail="$args: exit $argv_rc; digest: $argv_out; stderr: $argv_err"; break; }
done
if [ -z "$sib_detail" ]; then report 0 "$sib_quiet_label"; else report 1 "$sib_quiet_label" "$sib_detail"; fi

sib_detail=
for probe in 'next unless grep { index($_, $stem) == 0 } @{ goldens($at) };|@@sb/runtest-q|runtest-q-bare' \
             'substr($alias, length("$name-")) . ".";|@@sb/runtest-t|runtest-t-ow ' \
             'next if grep { $_->[2] eq $alias && at_covers($_, $at) } @op;|@@sb/runtest-t @@sb/runtest-t-own @@sb/runtest-t-ll|runtest-t-ll'; do
  IFS='|' read -r from args want <<<"$probe"
  case $from in
    substr*) to='substr($alias, length("$name-"));' ;;
    *) to= ;;
  esac
  SIB_FROM=$from SIB_TO=$to perl -pe 's/\Q$ENV{SIB_FROM}\E/$ENV{SIB_TO}/' "$SRC" >"$repeat_root/tools/test-run.sh"
  if cmp -s "$SRC" "$repeat_root/tools/test-run.sh"; then
    sib_detail="the mutation matched nothing in $SRC: $from"
    break
  fi
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  sib_probe sib-mutant run build $args
  grep -F -- "$sib_note" <<<"$argv_err" | grep -qF -- "$want" ||
    sib_detail="the mutant '$from' said nothing of $want: $argv_err"
  [ -z "$sib_detail" ] || break
done
cp "$SRC" "$repeat_root/tools/test-run.sh"
if [ -z "$sib_detail" ]; then report 0 "$sib_control_label"; else report 1 "$sib_control_label" "$sib_detail"; fi
}
harness_leg_run '103: a named test'\''s sibling goldens are named, never added to the batch #options' leg_103

# ---------------------------------------------------------------------------
# Leg 94: targeted batches from a file, and a test's aliases by family
# (gh-ocannl-1740)
# ---------------------------------------------------------------------------
# A 75-alias batch once had to go through a wrapper script because the
# caller's shell expansion of the list was refused, and a batch built from the
# files a change touched asked for `runtest-NAME` where NAME was a `@slow`
# member. Every refusal here is asserted with dune's calls empty: "runs
# nothing" is what makes the exit 2 a usage verdict.
leg_94() {
tf_label="--targets-file: the listed aliases reach dune after its own words; bad lists are refused before dune"
tf_cpu_label="--cpu/--gpu are refused before dune, pointing at the fleet slot"
tf_hint_label="a refused runtest-NAME names the directory's slow-NAME; no sibling, no hint"
tf_aliases_label="aliases: a test's per-test aliases by family, from dune's listing, readable back as a targets file"
tf_detail=
tf_dir=$TMP/targets
mkdir -p "$tf_dir/sub"
printf '# a comment\n\n  @x/runtest-a  \r\n@x/slow-b\r\n# trailing\n' >"$tf_dir/list.txt"
printf '@y/train-c\n' >"$tf_dir/sub/more.txt"
# Accepted: argv order, default subcommand, a relative file, two files, and
# placement before dune's own `--`.
for probe in "with-opts|run --targets-file $tf_dir/list.txt build -j 2|build -j 2 @x/runtest-a @x/slow-b" \
             "default|run --targets-file $tf_dir/list.txt|build @x/runtest-a @x/slow-b" \
             "two|start --targets-file $tf_dir/list.txt --targets-file $tf_dir/sub/more.txt build|build @x/runtest-a @x/slow-b @y/train-c" \
             "sep|run --targets-file $tf_dir/sub/more.txt build @w -- @after|build @w @y/train-c -- @after" \
             "relative|run --targets-file more.txt build|build @y/train-c"; do
  IFS='|' read -r tag args want <<<"$probe"
  tf_old=$PWD
  [ "$tag" != relative ] || cd "$tf_dir/sub" || { harness_abandon "cannot enter $tf_dir/sub"; return; }
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe "tf-$tag" $args
  cd "$tf_old" || { harness_abandon "cannot return to $tf_old"; return; }
  if [ "$tag" = two ] && [ "$argv_rc" = 0 ]; then
    # `start` detaches: its dune runs after the launcher reports.
    OCANNL_TOOL_TEST_RUNS=$TMP/argv-runs-tf-two "$repeat_root/tools/test-run.sh" wait last --timeout 60 >/dev/null 2>&1
    argv_calls=$(cat "$TMP/tf-two.calls")
  fi
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "$want" ]; } ||
    { tf_detail="$tag: exit $argv_rc; dune calls: ${argv_calls:-<none>} (want $want); stderr: $argv_err"; break; }
done
# Refused, dune never invoked.
printf '# nothing but comments\n\n' >"$tf_dir/empty.txt"
printf '@x/runtest-a\n-j 4\n' >"$tf_dir/option.txt"
printf '@x/runtest-a\ntest/x\n' >"$tf_dir/bare.txt"
printf '@x/scans\n' >"$tf_dir/aggregate.txt"
for probe in "empty|run --targets-file $tf_dir/empty.txt build|no alias listed" \
             "option|run --targets-file $tf_dir/option.txt build|option.txt:2: one alias per line" \
             "bare|run --targets-file $tf_dir/bare.txt build|bare.txt:2: not an alias operand" \
             "missing|run --targets-file $tf_dir/absent.txt build|no readable file" \
             "runtest|run --targets-file $tf_dir/list.txt runtest|which only dune's build takes" \
             "valueless|run --targets-file|requires a file" \
             "after|run build @w --targets-file $tf_dir/list.txt|belongs before the dune arguments" \
             "repeat|repeat 2 build @w --targets-file $tf_dir/list.txt|belongs before the dune arguments" \
             "repeat-verb|repeat --targets-file $tf_dir/list.txt 2 runtest|which only dune's build takes" \
             "repeat-direct|repeat --direct --targets-file $tf_dir/aggregate.txt 2|repeat --direct: not a per-test alias: @x/scans" \
             "repeat-empty|repeat --targets-file $tf_dir/empty.txt 2 build|no alias listed" \
             "bisect-empty|bisect --targets-file $tf_dir/empty.txt good bad|no alias listed" \
             "bisect-none|bisect good bad|at least one TARGET"; do
  [ -z "$tf_detail" ] || break
  IFS='|' read -r tag args want <<<"$probe"
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe "tf-refused-$tag" $args
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -qF -- "$want" <<<"$argv_err"; } ||
    tf_detail="refused $tag: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
done
# Accepted by repeat too (gh-ocannl-1797): every iteration builds the listed
# aliases after dune's own words.
if [ -z "$tf_detail" ]; then
  fixture_probe tf-repeat stable "$TMP/tf-runs-repeat" repeat --targets-file "$tf_dir/sub/more.txt" 2 build -j 2
  { [ "$fixture_rc" = 0 ] && [ "$(grep -c '^build -j 2 @y/train-c' <<<"$fixture_calls")" = 2 ]; } ||
    tf_detail="repeat --targets-file F 2 build -j 2: exit $fixture_rc; dune calls: ${fixture_calls:-<none>}"
fi
if [ -z "$tf_detail" ]; then report 0 "$tf_label"; else report 1 "$tf_label" "$tf_detail"; fi

tf_detail=
for probe in "run --cpu build @w" "start --gpu build @w" "plan --cpu=1 build @w"; do
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe tf-cpu $probe
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
    grep -qF -- "not an option of tools/test-run.sh" <<<"$argv_err" &&
    grep -qF -- "fleet-worker.sh execution slot --" <<<"$argv_err"; } ||
    { tf_detail="$probe: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"; break; }
done
if [ -z "$tf_detail" ]; then report 0 "$tf_cpu_label"; else report 1 "$tf_cpu_label" "$tf_detail"; fi

# The family hint and the listing read one table: directory `fam` holds a
# test `foo` under runtest, `bar` under slow and train with a runtest variant,
# an aggregate, and `fam/deep` a slow member only a recursive refusal finds.
mkdir -p "$repeat_root/fam/deep" "$repeat_root/famless"
printf '(rule (alias runtest-foo) (action (progn)))\n' >"$repeat_root/fam/dune"
printf '(rule (alias slow-qux) (action (progn)))\n' >"$repeat_root/fam/deep/dune"
printf '(rule (alias other) (action (progn)))\n' >"$repeat_root/famless/dune"
tf_table=$TMP/targets-alias-table
printf '%s\n' 'fam runtest-foo' 'fam slow-bar' 'fam train-bar' 'fam runtest-bar-tf32' 'fam scans' 'fam runtest-qual' \
  'fam/deep slow-qux' 'famless other' >"$tf_table"
tf_detail=
for probe in "run build @fam/runtest-bar|@fam/runtest-bar: neither fam nor any directory below it defines an alias runtest-bar; the same test under another family: @fam/slow-bar @fam/train-bar" \
             "run build @@fam/slow-foo|@@fam/slow-foo: fam defines no alias slow-foo; the same test under another family: @fam/runtest-foo" \
             "run build @fam/runtest-qux|@fam/runtest-qux: neither fam nor any directory below it defines an alias runtest-qux; the same test under another family: @fam/deep/slow-qux"; do
  IFS='|' read -r args want <<<"$probe"
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  REPEAT_TEST_ALIASES=$tf_table argv_probe tf-hint $args
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -qF -- "alias refused: $want" <<<"$argv_out"; } ||
    { tf_detail="$args: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; digest: $argv_out"; break; }
done
if [ -z "$tf_detail" ]; then
  # The control: a refusal with no sibling under any family says nothing more,
  # and a non-recursive one does not look below its directory.
  for args in "build @fam/runtest-nothing" "build @@fam/runtest-qux"; do
    # shellcheck disable=SC2086 # the probe's operands, split on purpose
    REPEAT_TEST_ALIASES=$tf_table argv_probe tf-hint-none run $args
    { [ "$argv_rc" = 2 ] && grep -qF -- "alias refused: " <<<"$argv_out" &&
      ! grep -qF -- "another family" <<<"$argv_out"; } ||
      { tf_detail="$args: exit $argv_rc (want 2, no hint); digest: $argv_out"; break; }
  done
fi
if [ -z "$tf_detail" ]; then report 0 "$tf_hint_label"; else report 1 "$tf_hint_label" "$tf_detail"; fi

tf_detail=
tf_aliases() { # tag argv...: `aliases` against the table, from the fixture root
  local rc=0
  (cd "$repeat_root" && REPEAT_TEST_ALIASES=$tf_table PATH=$repeat_bin:$PATH \
    OCANNL_TOOL_TEST_RUNS=$TMP/tf-aliases-runs tools/test-run.sh aliases "${@:2}") \
    >"$TMP/$1.out" 2>"$TMP/$1.err" || rc=$?
  tf_rc=$rc tf_out=$(cat "$TMP/$1.out") tf_err=$(cat "$TMP/$1.err")
}
tf_want_bar='# runtest: run by dune runtest
@fam/runtest-bar-tf32
# slow: outside dune runtest; dune build @slow runs them
@fam/slow-bar
# train: outside dune runtest; dune build @train runs them
@fam/train-bar'
tf_want_dir='# runtest: run by dune runtest
@fam/runtest-bar-tf32
@fam/runtest-foo
@fam/runtest-qual
# slow: outside dune runtest; dune build @slow runs them
@fam/slow-bar
# train: outside dune runtest; dune build @train runs them
@fam/train-bar'
# A source, its golden and an absolute path to it are one test; a directory
# and its dune file are every per-test alias there, aggregates left out.
tf_aliases tf-al-bar fam/bar.ml fam/bar.tf32.expected "$repeat_root/fam/bar.ml"
{ [ "$tf_rc" = 0 ] && [ "$tf_out" = "$tf_want_bar" ] && [ -z "$tf_err" ]; } ||
  tf_detail="bar: exit $tf_rc (want 0); out: $tf_out; err: $tf_err"
# A qualified golden is its test's, by the longest `-` prefix that is a test:
# `bar-tf32` is a variant alias of its own, never `bar`'s.
if [ -z "$tf_detail" ]; then
  tf_aliases tf-al-qual fam/qual-cc-0-0.log.expected fam/bar-tf32.expected
  { [ "$tf_rc" = 0 ] && [ "$tf_out" = '# runtest: run by dune runtest
@fam/runtest-bar-tf32
@fam/runtest-qual' ]; } || tf_detail="qualified: exit $tf_rc (want 0); out: $tf_out; err: $tf_err"
fi
if [ -z "$tf_detail" ]; then
  tf_aliases tf-al-dir fam fam/dune
  { [ "$tf_rc" = 0 ] && [ "$tf_out" = "$tf_want_dir" ]; } || tf_detail="dir: exit $tf_rc (want 0); out: $tf_out; err: $tf_err"
fi
# A helper and a directory dune does not know exit 1, the rest still listed.
if [ -z "$tf_detail" ]; then
  tf_aliases tf-al-helper fam/foo.ml fam/helper.ml famless/x.ml
  { [ "$tf_rc" = 1 ] && [ "$tf_out" = '# runtest: run by dune runtest
@fam/runtest-foo' ] && grep -qF "fam/helper.ml: fam defines no per-test alias of a test helper" <<<"$tf_err" &&
    grep -qF "affected --files fam/helper.ml" <<<"$tf_err" &&
    grep -qF "famless/x.ml: famless defines no per-test alias of a test x" <<<"$tf_err"; } ||
    tf_detail="helper: exit $tf_rc (want 1); out: $tf_out; err: $tf_err"
fi
if [ -z "$tf_detail" ]; then
  mkdir -p "$repeat_root/unlisted"
  tf_aliases tf-al-unknown unlisted/x.ml
  rmdir "$repeat_root/unlisted"
  { [ "$tf_rc" = 1 ] && [ -z "$tf_out" ] && grep -qF "dune knows no directory unlisted" <<<"$tf_err"; } ||
    tf_detail="unknown directory: exit $tf_rc (want 1); out: $tf_out; err: $tf_err"
fi
# Refused before dune: outside the worktree, a directory that does not exist, no path.
for args in "$TMP/elsewhere.ml|outside this worktree" "nowhere/x.ml|no directory" "|usage: tools/test-run.sh aliases"; do
  [ -z "$tf_detail" ] || break
  IFS='|' read -r arg want <<<"$args"
  : >"$TMP/tf-al-refused.show"
  # shellcheck disable=SC2086 # an empty operand stands for none
  REPEAT_TEST_SHOW_CALLS=$TMP/tf-al-refused.show tf_aliases tf-al-refused $arg
  { [ "$tf_rc" = 2 ] && [ -z "$tf_out" ] && grep -qF -- "$want" <<<"$tf_err" && [ ! -s "$TMP/tf-al-refused.show" ]; } ||
    tf_detail="refused '$arg': exit $tf_rc (want 2); out: $tf_out; err: $tf_err; listings: $(cat "$TMP/tf-al-refused.show")"
done
# Read back as a targets file, the listing is the batch.
if [ -z "$tf_detail" ]; then
  printf '%s\n' "$tf_want_bar" >"$tf_dir/from-aliases.txt"
  REPEAT_TEST_ALIASES=$tf_table argv_probe tf-al-roundtrip run --targets-file "$tf_dir/from-aliases.txt" build
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @fam/runtest-bar-tf32 @fam/slow-bar @fam/train-bar" ]; } ||
    tf_detail="round trip: exit $argv_rc; dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
if [ -z "$tf_detail" ]; then report 0 "$tf_aliases_label"; else report 1 "$tf_aliases_label" "$tf_detail"; fi
rm -rf "$repeat_root/fam" "$repeat_root/famless"
}
harness_leg_run '94: targeted batches from a file, and aliases by family #options' leg_94

# ---------------------------------------------------------------------------
# Leg 95: value options in both spellings; the slot's kinds refused by every
# launching subcommand (gh-ocannl-1740, gh-ocannl-1734)
# ---------------------------------------------------------------------------
# `run --cap=900 @x` reached dune and was refused as a misplaced option, where
# it was already in its place; `repeat --cpu 2 ...` read the slot's kind as the
# repeat count, and `run build @x --cpu` handed it to dune. Each `=` probe is
# checked by what the value DID (the recorded cap, the test cap the plan names,
# the backend check, the aliases dune got, the bisect's next check), not by the
# absence of a refusal. Two mutants control them: with every `=` arm cut each
# `=` probe must go red, and with every slot-kind arm cut each slot probe must.
leg_95() {
eq_label="value options: the = form is the spaced one in every parser; an empty value is refused"
eq_nc_label="negative control: with the = arms cut, every = probe goes red"
sk_label="--cpu/--gpu: refused with the slot's spelling by run, plan, fresh, repeat and bisect, before or after the dune arguments"
sk_nc_label="negative control: with the slot-kind arms cut, every slot-kind probe goes red"
eq_dir=$TMP/eq-options
mkdir -p "$eq_dir"
printf '@cheap\n' >"$eq_dir/targets.txt"
unset OCANNL_BACKEND
eq_probes=(run-cap run-targets run-expect plan-test-cap repeat-caps wait-timeout bisect-opts empty-eq)
eq_check() { # PROBE: runs it; eq_detail is empty iff it holds
  eq_detail=
  case $1 in
    run-cap)
      argv_runs=$TMP/eq-runs-cap argv_probe eq-run-cap run --cap=900 build @cheap
      { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ] &&
        [ "$(cat "$argv_dir/cap" 2>/dev/null)" = 900 ]; } ||
        eq_detail="run --cap=900: exit $argv_rc; calls: ${argv_calls:-<none>}; recorded cap: $(cat "$argv_dir/cap" 2>/dev/null); stderr: $argv_err" ;;
    run-targets)
      argv_probe eq-run-targets run --targets-file="$eq_dir/targets.txt" build
      { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "build @cheap" ]; } ||
        eq_detail="run --targets-file=F: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err" ;;
    run-expect)
      argv_probe eq-run-expect run --expect-backend=vulkan build @cheap
      { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
        grep -qF -- "--expect-backend vulkan: not a backend" <<<"$argv_err"; } ||
        eq_detail="run --expect-backend=vulkan: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err" ;;
    plan-test-cap)
      argv_probe eq-plan-test-cap plan --test-cap=9 --cap=900 build @cheap
      { [ "$argv_rc" = 0 ] && grep -qF "test cap: 9s for each process dune starts, asked for" <<<"$argv_out"; } ||
        eq_detail="plan --test-cap=9: exit $argv_rc; stdout: $argv_out; stderr: $argv_err" ;;
    repeat-caps)
      fixture_probe eq-repeat-caps stable "$TMP/eq-runs-repeat" repeat --cap=900 --test-cap=9 2 build @cheap
      # The bounds' first line is the per-test cap every iteration is given.
      { [ "$fixture_rc" = 0 ] && [ "$(grep -c '^build @cheap ' <<<"$fixture_calls")" = 2 ] &&
        [ "$(cat "$fixture_dir/cap" 2>/dev/null)" = 900 ] &&
        [ "$(head -n 1 "$fixture_dir/bounds" 2>/dev/null)" = 9 ]; } ||
        eq_detail="repeat --cap=900 --test-cap=9: exit $fixture_rc; calls: ${fixture_calls:-<none>}; recorded cap: $(cat "$fixture_dir/cap" 2>/dev/null); bounds: $(tr '\n' ' ' <"$fixture_dir/bounds" 2>/dev/null)" ;;
    wait-timeout)
      # The run the run-cap probe left, finished and green. The reference
      # goes first: a later word would overwrite a `--timeout=60` misread as one.
      OCANNL_TOOL_TEST_RUNS=$TMP/eq-runs-cap "$repeat_root/tools/test-run.sh" wait last --timeout=60 \
        >"$TMP/eq-wait.out" 2>&1
      eq_rc=$?
      [ "$eq_rc" = 0 ] || eq_detail="wait last --timeout=60: exit $eq_rc: $(cat "$TMP/eq-wait.out")" ;;
    bisect-opts)
      # Every option parsed, the bisect stops at its first check after them.
      argv_probe eq-bisect bisect --cap=900 --test-cap=9 --build=@cheap good bad @cheap
      { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
        grep -qF "bisect passes OCANNL_BACKEND to every step explicitly" <<<"$argv_err"; } ||
        eq_detail="bisect --cap=900 --test-cap=9 --build=T: exit $argv_rc; stderr: $argv_err" ;;
    empty-eq)
      # Read as "no test cap given", it would leave the default in force.
      argv_probe eq-empty run --test-cap= build @cheap
      { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
        grep -qF -- "--test-cap requires a value" <<<"$argv_err"; } ||
        eq_detail="run --test-cap=: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err" ;;
  esac
}
eq_all=
for probe in "${eq_probes[@]}"; do
  eq_check "$probe"
  [ -z "$eq_detail" ] || { eq_all="$probe: $eq_detail"; break; }
done
# The spaced empty value, refused the same way (it is not in the mutant's set:
# no `=` arm reads it).
if [ -z "$eq_all" ]; then
  argv_probe eq-empty-spaced run --test-cap "" build @cheap
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
    grep -qF -- "--test-cap requires a value" <<<"$argv_err"; } ||
    eq_all="run --test-cap '': exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
if [ -z "$eq_all" ]; then report 0 "$eq_label"; else report 1 "$eq_label" "$eq_all"; fi

sk_probes=("run|run --cpu build @w" "fresh|fresh --gpu @w/runtest-a" "repeat|repeat --cpu 2 build @w"
           "repeat|repeat --gpu=1 --direct 2 @w/runtest-a" "bisect|bisect --cpu good bad @w"
           "run|run build @w --gpu" "plan|plan build @w --cpu=1" "repeat|repeat 2 build @w --cpu"
           "fresh|fresh @w/runtest-a --gpu" "bisect|bisect good bad @w --cpu")
sk_check() { # SUB|ARGV: runs it; sk_detail is empty iff it holds
  local sub=${1%%|*} args=${1#*|} word
  sk_detail=
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe sk-probe $args
  for word in $args; do case $word in --cpu* | --gpu*) break ;; esac; done
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
    grep -qF -- "${word} is not an option of tools/test-run.sh" <<<"$argv_err" &&
    grep -qF -- "fleet-worker.sh execution slot ${word%%=*} -- tools/test-run.sh $sub" <<<"$argv_err"; } ||
    sk_detail="$args: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
}
sk_all=
for probe in "${sk_probes[@]}"; do
  sk_check "$probe"
  [ -z "$sk_detail" ] || { sk_all=$sk_detail; break; }
done
# The control on the word's side: past dune's own `--` it is the program's.
if [ -z "$sk_all" ]; then
  argv_probe sk-after-sep run exec ./prog.exe -- --cpu
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "exec ./prog.exe -- --cpu" ]; } ||
    sk_all="run exec ./prog.exe -- --cpu: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
if [ -z "$sk_all" ]; then report 0 "$sk_label"; else report 1 "$sk_label" "$sk_all"; fi

# The mutants replace the fixture's copies (bisect's parser is bisect-steps.sh,
# read from beside SRC: the tree's, also where a subject is staged)
# and are put back after each set.
eq_mutate() { # PERL-0-PROGRAM: sets eq_made to 1 iff both copies were rewritten
  local f n=0
  for f in test-run.sh bisect-steps.sh; do
    perl -0pe "$1" "${SRC%/*}/$f" >"$repeat_root/tools/$f"
    cmp -s "${SRC%/*}/$f" "$repeat_root/tools/$f" || n=$((n + 1))
  done
  eq_made=0
  [ "$n" = 2 ] && bash -n "$repeat_root/tools/test-run.sh" && bash -n "$repeat_root/tools/bisect-steps.sh" && eq_made=1
}
eq_restore() { cp "$SRC" "${SRC%/*}/bisect-steps.sh" "$repeat_root/tools/"; }
eq_mutate 's/(--[a-z-]+) \| \1=\*\)(\s*opt_value)/$1)$2/g'
if [ "$eq_made" != 1 ]; then
  report 1 "$eq_nc_label" "the mutant was not made"
else
  eq_survivors=
  for probe in "${eq_probes[@]}"; do
    eq_check "$probe"
    [ -n "$eq_detail" ] || eq_survivors="$eq_survivors $probe"
  done
  if [ -z "$eq_survivors" ]; then report 0 "$eq_nc_label"; else report 1 "$eq_nc_label" "held under the mutant:$eq_survivors"; fi
fi
eq_restore
eq_mutate 's/--cpu \| --cpu=\* \| --gpu \| --gpu=\*\) refuse_slot_option/--no-slot-kind) refuse_slot_option/g'
if [ "$eq_made" != 1 ]; then
  report 1 "$sk_nc_label" "the mutant was not made"
else
  sk_survivors=
  for probe in "${sk_probes[@]}"; do
    sk_check "$probe"
    [ -n "$sk_detail" ] || sk_survivors="$sk_survivors [${probe#*|}]"
  done
  if [ -z "$sk_survivors" ]; then report 0 "$sk_nc_label"; else report 1 "$sk_nc_label" "held under the mutant:$sk_survivors"; fi
fi
eq_restore
}
harness_leg_run '95: value options spaced or with =; the slot kinds refused by every launcher #options' leg_95

# ---------------------------------------------------------------------------
# Leg 96: a flag refuses a value; a query refuses an option-shaped run
# ---------------------------------------------------------------------------
# Leg 95's `=` spelling left two misreadings. `repeat --alone=1 2 @x` read
# `--alone=1` as the repeat count (and `bisect --no-first-parent=x` met only
# "unknown option"); `wait --cpu` was looked up as a run, and `wait --timout=60
# last` waited on `last` with the default budget, the later word winning. Each
# refusal is checked by its exit, its silence toward dune and the words it
# names. A mutant cutting opt_flag's refusal turns every flag probe red, and
# one cutting take_run_ref's option arm every option-shaped query probe; the
# valid queries (a run's directory, `last`, a spaced or `=` timeout) must
# still answer, or a refusal of everything would pass.
leg_96() {
fl_label="flags: --alone=, --direct=, --no-first-parent= are refused by name, dune never invoked"
fl_nc_label="negative control: with opt_flag's refusal cut, every flag probe goes red"
qr_label="queries: an option-shaped word is refused as unknown by status, wait, stop, paths and lock-status, never looked up as a run"
qr_nc_label="negative control: with take_run_ref's option arm cut, every option-shaped query probe goes red"
qr_ok_label="queries: a run's directory, last, and wait's timeout spaced or with = still answer; a second run, an unknown repeat option and list's arguments are refused"
lo_label="launchers: an unknown leading option is refused by run, start, plan and fresh before dune; a leading -j is refused with the order, except by fresh"
lo_nc_label="negative control: with the launchers' unknown-option arm cut, every unknown-option probe goes red"
unset OCANNL_BACKEND
fl_probes=("--alone|repeat --alone=1 2 build @cheap" "--direct|repeat --direct=1 2 build @cheap"
           "--alone|repeat --cap 900 --alone=yes 2 build @cheap"
           "--no-first-parent|bisect --no-first-parent=x good bad @cheap")
fl_check() { # FLAG|ARGV: runs it; fl_detail is empty iff it holds
  local flag=${1%%|*} args=${1#*|}
  fl_detail=
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe fl-probe $args
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] &&
    grep -qF -- "$flag is a flag and takes no value: $flag=" <<<"$argv_err"; } ||
    fl_detail="$args: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
}
fl_all=
for probe in "${fl_probes[@]}"; do
  fl_check "$probe"
  [ -z "$fl_detail" ] || { fl_all=$fl_detail; break; }
done
# After GOOD and BAD the flag is misplaced, spelled either way.
if [ -z "$fl_all" ]; then
  argv_probe fl-bisect-late bisect good bad @cheap --no-first-parent=x
  { [ "$argv_rc" = 2 ] && grep -qF -- "--no-first-parent=x belongs before GOOD and BAD" <<<"$argv_err"; } ||
    fl_all="bisect good bad @cheap --no-first-parent=x: exit $argv_rc; stderr: $argv_err"
fi
if [ -z "$fl_all" ]; then report 0 "$fl_label"; else report 1 "$fl_label" "$fl_all"; fi

# A finished green run for the queries to name.
argv_runs=$TMP/qr-runs argv_probe qr-run run build @cheap
qr_dir=$argv_dir
qr_query() { # ARGV: runs the query against qr-runs; qr_rc, qr_out
  OCANNL_TOOL_TEST_RUNS=$TMP/qr-runs "$repeat_root/tools/test-run.sh" "$@" >"$TMP/qr.out" 2>&1
  qr_rc=$?
  qr_out=$(cat "$TMP/qr.out")
}
qr_probes=("status|--cpu" "wait|--timout=60 last" "wait|--cpu" "wait|last --timout 60"
           "stop|--cpu" "paths|run --cpu" "lock-status|-x")
qr_check() { # SUB|ARGS: runs it; qr_detail is empty iff it holds
  local sub=${1%%|*} args=${1#*|} word
  qr_detail=
  for word in $args; do case $word in -*) break ;; esac; done
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  qr_query "$sub" $args
  { [ "$qr_rc" = 2 ] && grep -qF -- "$sub: unknown option $word" <<<"$qr_out" &&
    ! grep -qF "no such run" <<<"$qr_out"; } ||
    qr_detail="$sub $args: exit $qr_rc (want 2): $qr_out"
}
qr_all=
if [ "$argv_rc" != 0 ] || [ ! -d "$qr_dir" ]; then
  qr_all="the run the queries name did not finish green: exit $argv_rc; stderr: $argv_err"
fi
if [ -z "$qr_all" ]; then
  for probe in "${qr_probes[@]}"; do
    qr_check "$probe"
    [ -z "$qr_detail" ] || { qr_all=$qr_detail; break; }
  done
fi
if [ -z "$qr_all" ]; then report 0 "$qr_label"; else report 1 "$qr_label" "$qr_all"; fi

qr_ok_all=
if [ "$argv_rc" != 0 ] || [ ! -d "$qr_dir" ]; then
  qr_ok_all="the run the queries name did not finish green: exit $argv_rc; stderr: $argv_err"
fi
if [ -z "$qr_ok_all" ]; then
  for args in "status $qr_dir" "status last" "wait $qr_dir --timeout=60" "wait --timeout 60 last" \
              "stop last" "lock-status $qr_dir" "paths run $qr_dir"; do
    # shellcheck disable=SC2086 # the query's operands, split on purpose
    qr_query $args
    [ "$qr_rc" = 0 ] || { qr_ok_all="$args: exit $qr_rc (want 0): $qr_out"; break; }
  done
fi
if [ -z "$qr_ok_all" ]; then
  qr_query wait last "$qr_dir"
  { [ "$qr_rc" = 2 ] && grep -qF "wait: one run at most, got last and $qr_dir" <<<"$qr_out"; } ||
    qr_ok_all="wait last DIR: exit $qr_rc (want 2): $qr_out"
fi
if [ -z "$qr_ok_all" ]; then
  qr_query list extra
  { [ "$qr_rc" = 2 ] && grep -qF "list takes no arguments" <<<"$qr_out"; } ||
    qr_ok_all="list extra: exit $qr_rc (want 2): $qr_out"
fi
if [ -z "$qr_ok_all" ]; then
  argv_probe qr-repeat-unknown repeat --alon 2 build @cheap
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -qF "repeat: unknown option --alon" <<<"$argv_err"; } ||
    qr_ok_all="repeat --alon 2: exit $argv_rc; calls: ${argv_calls:-<none>}; stderr: $argv_err"
fi
if [ -z "$qr_ok_all" ]; then report 0 "$qr_ok_label"; else report 1 "$qr_ok_label" "$qr_ok_all"; fi

# The launchers' intake (gh-ocannl-1797): `run --cpa=1 build @x` once reached
# dune as its subcommand ("unknown command") after the lock, the readers and
# the slot. Each unknown option is refused by name with no dune call; a
# leading -j is refused with the order to write it in, except by `fresh`,
# whose operands take -j anywhere -- the acceptance probe proves the -j
# reached fresh_argv by the refusal of the aggregate after it.
lo_probes=("run: unknown option --cpa=1|run --cpa=1 build @cheap"
           "start: unknown option --bogus|start --bogus build @cheap"
           "plan: unknown option -x|plan -x build @cheap"
           "fresh: unknown option --bogus|fresh --bogus @x/runtest-a")
lo_check() { # WANT|ARGV: runs it; lo_detail is empty iff it holds
  local want=${1%%|*} args=${1#*|}
  lo_detail=
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  argv_probe lo-probe $args
  { [ "$argv_rc" = 2 ] && [ -z "$argv_calls" ] && grep -qF -- "$want" <<<"$argv_err"; } ||
    lo_detail="$args: exit $argv_rc (want 2); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
}
lo_all=
for probe in "${lo_probes[@]}" "run: -j is dune's option|run -j 4 build @cheap" \
             "fresh: not a per-test alias: @x/scans|fresh -j 2 @x/scans"; do
  lo_check "$probe"
  [ -z "$lo_detail" ] || { lo_all=$lo_detail; break; }
done
if [ -z "$lo_all" ]; then report 0 "$lo_label"; else report 1 "$lo_label" "$lo_all"; fi

# The mutants replace the fixture's copies (bisect's parser is bisect-steps.sh)
# and are put back after each set; each must rewrite what it names.
fq_mutate() { # PERL-0-PROGRAM: sets fq_made to 1 iff it rewrote test-run.sh
  local f
  for f in test-run.sh bisect-steps.sh; do
    perl -0pe "$1" "${SRC%/*}/$f" >"$repeat_root/tools/$f"
  done
  fq_made=0
  ! cmp -s "$SRC" "$repeat_root/tools/test-run.sh" &&
    bash -n "$repeat_root/tools/test-run.sh" && bash -n "$repeat_root/tools/bisect-steps.sh" && fq_made=1
}
fq_restore() { cp "$SRC" "${SRC%/*}/bisect-steps.sh" "$repeat_root/tools/"; }
fq_mutate 's/(opt_flag\(\) \{[^\n]*\n)  case \$1 in --\*=\*\) die [^\n]*\n/$1  :\n/'
if [ "$fq_made" != 1 ]; then
  report 1 "$fl_nc_label" "the mutant was not made"
else
  fl_survivors=
  for probe in "${fl_probes[@]}"; do
    fl_check "$probe"
    [ -n "$fl_detail" ] || fl_survivors="$fl_survivors [${probe#*|}]"
  done
  if [ -z "$fl_survivors" ]; then report 0 "$fl_nc_label"; else report 1 "$fl_nc_label" "held under the mutant:$fl_survivors"; fi
fi
fq_restore
fq_mutate 's/(take_run_ref\(\) \{[^\n]*\n  case \$2 in\n)    -\*\) die/$1    --cut-option-arm) die/'
if [ "$fq_made" != 1 ]; then
  report 1 "$qr_nc_label" "the mutant was not made"
else
  qr_survivors=
  for probe in "${qr_probes[@]}"; do
    qr_check "$probe"
    [ -n "$qr_detail" ] || qr_survivors="$qr_survivors [${probe%%|*} ${probe#*|}]"
  done
  if [ -z "$qr_survivors" ]; then report 0 "$qr_nc_label"; else report 1 "$qr_nc_label" "held under the mutant:$qr_survivors"; fi
fi
fq_restore
fq_mutate 's/\n        -\*\)\n          case \$sub in\n            fresh\) form=/\n        --cut-unknown-arm)\n          case \$sub in\n            fresh) form=/'
if [ "$fq_made" != 1 ]; then
  report 1 "$lo_nc_label" "the mutant was not made"
else
  lo_survivors=
  for probe in "${lo_probes[@]}"; do
    lo_check "$probe"
    [ -n "$lo_detail" ] || lo_survivors="$lo_survivors [${probe#*|}]"
  done
  if [ -z "$lo_survivors" ]; then report 0 "$lo_nc_label"; else report 1 "$lo_nc_label" "held under the mutant:$lo_survivors"; fi
fi
fq_restore
}
harness_leg_run '96: a flag refuses a value; a query refuses an option-shaped run #options' leg_96

# ---------------------------------------------------------------------------
# Leg 98: `prepush`, the pre-push run as one command (gh-ocannl-1829)
# ---------------------------------------------------------------------------
# Every worker ran `affected --since BASE`'s list without its slow- and train-
# members by a hand-written grep or a targets file computed inside `bash -c`,
# which a fleet leg's backend-pin check refuses. Here `affected` is a stand-in
# (the fixture's tools/affected-tests.sh, which `affected` execs) printing a
# list, a stderr and a status the probe chooses, so what is pinned is prepush's
# own work: which members reach dune (both ways), what --list prints, and every
# case that runs nothing.
leg_98() {
pp_label="prepush: affected's list without slow-/train- reaches dune as run's; --list prints it"
pp_refuse_label="prepush: usage errors, a cross-cutting or incomplete verdict and an unknown line run nothing"
pp_verdict_label="prepush: the verdict is affected's trailer, not its prose; without one nothing runs"
pp_harness_label="prepush: the harness commands affected --harnesses prints are named last, never run; none, no line"
pp_stub=$repeat_root/tools/affected-tests.sh
if [ -e "$pp_stub" ]; then
  harness_abandon "the fixture root already has $pp_stub: it is no stand-in of this leg's"; return
fi
cat >"$pp_stub" <<'STUB'
case " $* " in
  *" --harnesses "*)
    printf '%s\n' "$*" >>"$PP_AFFECTED_CALLS.harnesses"
    cat "${PP_HARNESS_OUT:-/dev/null}"
    cat "${PP_HARNESS_ERR:-/dev/null}" >&2
    exit "${PP_HARNESS_RC:-0}" ;;
esac
printf '%s\n' "$*" >>"$PP_AFFECTED_CALLS"
cat "$PP_AFFECTED_OUT"
cat "$PP_AFFECTED_ERR" >&2
exit "${PP_AFFECTED_RC:-0}"
STUB
pp_tmp=$TMP/prepush-tmp
mkdir -p "$pp_tmp"
pp_out=$TMP/prepush-affected.out pp_err=$TMP/prepush-affected.err
# The scratch list prepush makes, which no probe may leave behind.
pp_left() { find "$pp_tmp" -name 'test-run-prepush.*' -prune -print; }
# Kept: runtest- members, a root one, one whose NAME holds `slow-`, one in a
# directory named `train-...`. Left off: slow- and train- members.
printf '%s\n' '@a/runtest-x' '@a/slow-y' '@b/train-z' '@@runtest-root' '@a/runtest-slow-ish' \
  '@train-dir/runtest-q' '@test/training/slow-w' >"$pp_out"
printf '%s\n' 'affected: 7 alias(es) from 2 changed path(s)' 'affected: verdict=ok' >"$pp_err"
# The same list as affected ends it when a changed path is UNMAPPED (exit 3).
pp_inc_err=$TMP/prepush-affected-incomplete.err
printf '%s\n' 'affected: 7 alias(es) from 2 changed path(s); INCOMPLETE, see UNMAPPED above' \
  'affected: verdict=incomplete' >"$pp_inc_err"
pp_probe() { # tag rc argv...: prepush against the stand-in's list, its stderr the exit status's
  local tag=$1 rc=$2 err=$pp_err
  shift 2
  [ "$rc" != 3 ] || err=$pp_inc_err
  : >"$TMP/$tag.affected"
  PP_AFFECTED_CALLS=$TMP/$tag.affected PP_AFFECTED_OUT=$pp_out PP_AFFECTED_ERR=$err PP_AFFECTED_RC=$rc \
    TMPDIR=$pp_tmp argv_probe "$tag" prepush "$@"
  pp_asked=$(cat "$TMP/$tag.affected")
}
pp_detail=
pp_want='build -j 2 @a/runtest-x @@runtest-root @a/runtest-slow-ish @train-dir/runtest-q'
pp_probe pp-run 0 --since base build -j 2
{ [ "$argv_rc" = 0 ] && [ "$argv_calls" = "$pp_want" ] && [ "$pp_asked" = "--since base" ] &&
  grep -qF 'prepush: 4 of the 7 alias(es) affected --since base lists; left off: 1 train-' <<<"$argv_err" &&
  grep -qxF 'test-run: prepush: slow: @a/slow-y' <<<"$argv_err" &&
  grep -qxF 'test-run: prepush: slow: @test/training/slow-w' <<<"$argv_err" &&
  grep -qxF 'affected: 7 alias(es) from 2 changed path(s)' <<<"$argv_err" &&
  [ -z "$(pp_left)" ]; } ||
  pp_detail="run: exit $argv_rc; dune calls: ${argv_calls:-<none>} (want $pp_want); affected asked: $pp_asked; scratch left: $(pp_left); stderr: $argv_err"
# Run options pass through in either spelling, `build` is the default verb,
# and the recorded command is the batch's.
if [ -z "$pp_detail" ]; then
  pp_probe pp-opts 0 --cap=900 --since=base --test-cap 50
  pp_want='build @a/runtest-x @@runtest-root @a/runtest-slow-ish @train-dir/runtest-q'
  { [ "$argv_rc" = 0 ] && [ "$argv_calls" = "$pp_want" ] && [ "$(cat "$argv_dir/cap" 2>/dev/null)" = 900 ] &&
    grep -qF '@train-dir/runtest-q' "$argv_dir/cmd"; } ||
    pp_detail="options: exit $argv_rc; dune calls: ${argv_calls:-<none>} (want $pp_want); cap: $(cat "$argv_dir/cap" 2>/dev/null); stderr: $argv_err"
fi
# --list prints the same list and runs nothing.
if [ -z "$pp_detail" ]; then
  pp_probe pp-list 0 --list --since base
  { [ "$argv_rc" = 0 ] && [ -z "$argv_calls" ] &&
    [ "$argv_out" = "$(printf '%s\n' '@a/runtest-x' '@@runtest-root' '@a/runtest-slow-ish' '@train-dir/runtest-q')" ]; } ||
    pp_detail="--list: exit $argv_rc; dune calls: ${argv_calls:-<none>}; out: $argv_out; stderr: $argv_err"
fi
# INCOMPLETE: --list still prints what was found and exits 3.
if [ -z "$pp_detail" ]; then
  pp_probe pp-list-incomplete 3 --since base --list
  { [ "$argv_rc" = 3 ] && [ -z "$argv_calls" ] && grep -qxF '@a/runtest-x' <<<"$argv_out" &&
    ! grep -qF '/slow-' <<<"$argv_out" && grep -qF 'INCOMPLETE' <<<"$argv_err"; } ||
    pp_detail="--list incomplete: exit $argv_rc (want 3); out: $argv_out; stderr: $argv_err"
fi
if [ -z "$pp_detail" ]; then report 0 "$pp_label"; else report 1 "$pp_label" "$pp_detail"; fi

# Refused, or nothing to run: dune never invoked; a usage error asks no affected.
pp_detail=
for probe in "usage|0|build -j 2|--since BASE is required" \
             "usage|0|--since base --list build|--list prints the list and runs nothing" \
             "usage|0|--since base --list --cap 9|--list prints the list and runs nothing" \
             "usage|0|--since base --targets-file f|the list is affected's" \
             "usage|0|--since base --referenced-only|unknown option --referenced-only" \
             "usage|0|--since base runtest|only dune's build takes" \
             "usage|0|--since base --cpu|fleet-worker.sh execution slot --" \
             "usage|0|--since base --list=1|is a flag and takes no value" \
             "usage|0|--since base build @w --cap 9|tools/test-run.sh prepush --since BASE [--list |" \
             "asked|2|--since base|exited 2" \
             "asked|3|--since base|the list is INCOMPLETE"; do
  IFS='|' read -r kind rc args want <<<"$probe"
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  pp_probe pp-refused "$rc" $args
  pp_want_rc=2
  [ "$rc" != 3 ] || pp_want_rc=3
  { [ "$argv_rc" = "$pp_want_rc" ] && [ -z "$argv_calls" ] && grep -qF -- "$want" <<<"$argv_err" &&
    { [ "$kind" = asked ] || [ -z "$pp_asked" ]; } && [ -z "$(pp_left)" ]; } ||
    { pp_detail="$args (affected exit $rc): exit $argv_rc (want $pp_want_rc); dune calls: ${argv_calls:-<none>}; affected asked: ${pp_asked:-<none>}; stderr: $argv_err"; break; }
done
# CROSS-CUTTING refuses both forms; a line of no known family refuses the list;
# a list with no runtest- member runs nothing, green.
pp_cc_err=$TMP/prepush-affected-cc.err
printf '%s\n' 'affected: CROSS-CUTTING: 9 of the 10 per-test aliases rank first' 'affected: verdict=cross-cutting' >"$pp_cc_err"
for probe in "cc|$pp_out|$pp_cc_err|--since base|2|is CROSS-CUTTING" \
             "cc-list|$pp_out|$pp_cc_err|--since base --list|2|is CROSS-CUTTING" \
             "aggregate|$TMP/prepush-aggregate.out|$pp_err|--since base|2|no per-test alias of runtest, slow or train: '@x/scans'" \
             "spaced|$TMP/prepush-spaced.out|$pp_err|--since base --list|2|no per-test alias of runtest, slow or train: '@x/runtest-a @x/runtest-b'" \
             "none|$TMP/prepush-none.out|$pp_err|--since base|0|no runtest- alias to run"; do
  [ -z "$pp_detail" ] || break
  IFS='|' read -r tag out err args want_rc want <<<"$probe"
  printf '%s\n' '@x/runtest-a' '@x/scans' >"$TMP/prepush-aggregate.out"
  printf '%s\n' '@x/runtest-a @x/runtest-b' >"$TMP/prepush-spaced.out"
  printf '%s\n' '@x/slow-a' '@y/train-b' >"$TMP/prepush-none.out"
  # shellcheck disable=SC2086 # the probe's operands, split on purpose
  PP_AFFECTED_OUT=$out PP_AFFECTED_ERR=$err PP_AFFECTED_CALLS=$TMP/pp-$tag.affected TMPDIR=$pp_tmp \
    argv_probe "pp-$tag" prepush $args
  { [ "$argv_rc" = "$want_rc" ] && [ -z "$argv_calls" ] && [ -z "$argv_out" ] && grep -qF -- "$want" <<<"$argv_err" &&
    [ -z "$(pp_left)" ]; } ||
    pp_detail="$tag: exit $argv_rc (want $want_rc); dune calls: ${argv_calls:-<none>}; out: $argv_out; stderr: $argv_err"
done
if [ -z "$pp_detail" ]; then report 0 "$pp_refuse_label"; else report 1 "$pp_refuse_label" "$pp_detail"; fi

# The verdict is the trailer (gh-ocannl-1921): `STDERR LINES ('~' between
# them)|affected's exit|prepush's exit|what its stderr says`, exit 0 a run of
# the four kept aliases. The prose decides nothing: CROSS-CUTTING or INCOMPLETE
# prose over an `ok` trailer runs, a cross-cutting trailer with no prose does
# not. No trailer, two, one that is not the last line, an unknown one, and one
# at odds with the exit status (incomplete is exit 3, and only it) run nothing.
pp_verdict_cases=(
  "affected: verdict=ok|0|0|4 of the 7 alias(es)"
  "affected: CROSS-CUTTING: 9 of the 10 per-test aliases rank first~affected: 7 alias(es); INCOMPLETE, see UNMAPPED above~affected: verdict=ok|0|0|4 of the 7 alias(es)"
  "affected: verdict=cross-cutting|0|2|is CROSS-CUTTING"
  "affected: verdict=incomplete|3|3|the list is INCOMPLETE"
  "affected: verdict=cross-cutting,incomplete|3|2|is CROSS-CUTTING"
  "affected: CROSS-CUTTING: 9 of the 10 per-test aliases rank first|0|2|printed no verdict trailer"
  "affected: 7 alias(es) from 2 changed path(s)|0|2|printed no verdict trailer"
  "affected: 7 alias(es); INCOMPLETE, see UNMAPPED above|3|2|printed no verdict trailer"
  "|0|2|printed no verdict trailer"
  "affected: verdict=ok~affected: a later line|0|2|printed no verdict trailer"
  "affected: verdict=ok~affected: verdict=ok|0|2|printed no verdict trailer"
  "affected: verdict=cross-cutting~affected: verdict=ok|0|2|printed no verdict trailer"
  " affected: verdict=ok|0|2|printed no verdict trailer"
  "affected: verdict=|0|2|a verdict this runner does not know: ''"
  "affected: verdict=fine|0|2|a verdict this runner does not know: 'fine'"
  "affected: verdict=OK|0|2|a verdict this runner does not know: 'OK'"
  "affected: verdict=incomplete,cross-cutting|3|2|a verdict this runner does not know: 'incomplete,cross-cutting'"
  "affected: verdict=ok|3|2|exited 3 with the verdict 'ok'"
  "affected: verdict=cross-cutting|3|2|exited 3 with the verdict 'cross-cutting'"
  "affected: verdict=incomplete|0|2|exited 0 with the verdict 'incomplete'"
  "affected: verdict=cross-cutting,incomplete|0|2|exited 0 with the verdict 'cross-cutting,incomplete'"
)
pp_verdict_check() { # CASE -> $pp_detail, empty when prepush answers it as written
  local lines rc want_rc want pp_kept
  pp_kept='build @a/runtest-x @@runtest-root @a/runtest-slow-ish @train-dir/runtest-q'
  pp_detail=
  IFS='|' read -r lines rc want_rc want <<<"$1"
  printf '%s' "$lines" | tr '~' '\n' >"$TMP/prepush-verdict.err"
  [ -z "$lines" ] || echo >>"$TMP/prepush-verdict.err"
  PP_AFFECTED_OUT=$pp_out PP_AFFECTED_ERR=$TMP/prepush-verdict.err PP_AFFECTED_RC=$rc \
    PP_AFFECTED_CALLS=$TMP/pp-verdict.affected TMPDIR=$pp_tmp argv_probe pp-verdict prepush --since base
  { [ "$argv_rc" = "$want_rc" ] && grep -qF -- "$want" <<<"$argv_err" && [ -z "$(pp_left)" ] &&
    { if [ "$want_rc" = 0 ]; then [ "$argv_calls" = "$pp_kept" ]; else [ -z "$argv_calls" ]; fi; }; } ||
    pp_detail="affected's stderr [$lines], exit $rc: exit $argv_rc (want $want_rc, saying '$want'); dune calls: ${argv_calls:-<none>}; stderr: $argv_err"
}
for probe in "${pp_verdict_cases[@]}"; do
  pp_verdict_check "$probe"
  [ -z "$pp_detail" ] || break
done
if [ -z "$pp_detail" ]; then report 0 "$pp_verdict_label"; else report 1 "$pp_verdict_label" "$pp_detail"; fi
# Negative control: prepush as it read the verdict before (the CROSS-CUTTING
# prose and the exit status). It answers only the two cases where a lone
# trailer and the exit status say what the prose would have: the first (ok)
# and the fourth (incomplete, exit 3). Every other case goes red.
pp_nc_label="negative control: reading affected's prose and exit status instead, every case but the two they also decide goes red"
perl -0pe 's/\n  pp_verdict=\$\(sed -n .*?\n  \[ "\$pp_cross" = 0 \] \|\|\n/\n  pp_incomplete=\$((pp_rc \/ 3))\n  ! grep -q "^affected: CROSS-CUTTING:" "\$pp_dir\/affected.err" ||\n/s' \
  "$SRC" >"$repeat_root/tools/test-run.sh"
if cmp -s "$SRC" "$repeat_root/tools/test-run.sh" || ! bash -n "$repeat_root/tools/test-run.sh" ||
   grep -qF 'verdict=' <(sed -n '/^if \[ "\$sub" = prepush \]; then$/,/^fi$/p' "$repeat_root/tools/test-run.sh" | grep -v '^ *#'); then
  report 1 "$pp_nc_label" "the mutant was not made"
else
  pp_held=
  for pp_i in "${!pp_verdict_cases[@]}"; do
    pp_verdict_check "${pp_verdict_cases[$pp_i]}"
    [ -n "$pp_detail" ] || pp_held="$pp_held $pp_i"
  done
  if [ "$pp_held" = " 0 3" ]; then report 0 "$pp_nc_label"; else report 1 "$pp_nc_label" "held under the mutant: cases${pp_held:- <none>} (want 0 3)"; fi
fi
cp "$SRC" "$repeat_root/tools/test-run.sh"

# The harness commands (gh-ocannl-1929): `TAG|affected --harnesses' STDOUT
# LINES ('~' between them)|its exit|affected's list|its stderr|its exit|
# prepush's operands|prepush's exit|dune reached (kept) or not (none)|the
# commands named (named) or no line about them (silent)`. They are asked of
# `affected --harnesses --since BASE`, named as the LAST lines of stderr, the
# very last the one command that runs them for this BASE --
# after a run, after --list's list (stdout is still the list alone), after a
# cross-cutting or incomplete refusal and after a list with nothing to run --
# and never reach dune or change the exit status. A change reaching none (an
# `.ml`-only diff: the stand-in prints nothing) gets no line; a failed ask
# runs nothing.
pp_h_cmds="tools/test-a.sh~tools/test-b.sh --only '^1:'"
printf '%s\n' '@x/slow-a' '@y/train-b' >"$TMP/prepush-none.out"
pp_harness_cases=(
  "run|$pp_h_cmds|0|$pp_out|$pp_err|0|--since base build|0|kept|named"
  "ml-only|||$pp_out|$pp_err|0|--since base build|0|kept|silent"
  "list|$pp_h_cmds|0|$pp_out|$pp_err|0|--since base --list|0|none|named"
  "list-ml-only||0|$pp_out|$pp_err|0|--since base --list|0|none|silent"
  "cross-cutting|$pp_h_cmds|0|$pp_out|$pp_cc_err|0|--since base|2|none|named"
  "incomplete|$pp_h_cmds|0|$pp_out|$pp_inc_err|3|--since base|3|none|named"
  "list-incomplete|$pp_h_cmds|0|$pp_out|$pp_inc_err|3|--since base --list|3|none|named"
  "nothing-to-run|$pp_h_cmds|0|$TMP/prepush-none.out|$pp_err|0|--since base|0|none|named"
  "ask-failed|$pp_h_cmds|2|$pp_out|$pp_err|0|--since base build|2|none|silent"
)
pp_harness_check() { # CASE -> $pp_detail, empty when prepush answers it as written
  local tag hlines hrc out err arc args want_rc calls named n_count n_named pp_kept
  pp_kept='build @a/runtest-x @@runtest-root @a/runtest-slow-ish @train-dir/runtest-q'
  pp_detail=
  IFS='|' read -r tag hlines hrc out err arc args want_rc calls named <<<"$1"
  printf '%s' "$hlines" | tr '~' '\n' >"$TMP/prepush-harness.out"
  [ -z "$hlines" ] || echo >>"$TMP/prepush-harness.out"
  : >"$TMP/pp-h.affected.harnesses"
  # shellcheck disable=SC2086 # the case's operands, split on purpose
  PP_AFFECTED_OUT=$out PP_AFFECTED_ERR=$err PP_AFFECTED_RC=$arc PP_AFFECTED_CALLS=$TMP/pp-h.affected \
    PP_HARNESS_OUT=$TMP/prepush-harness.out PP_HARNESS_RC=${hrc:-0} TMPDIR=$pp_tmp argv_probe pp-h prepush $args
  n_count=$(grep -c 'harness command(s) this change reaches' <<<"$argv_err")
  n_named=$(grep -c '^test-run: prepush: harness: ' <<<"$argv_err")
  { [ "$argv_rc" = "$want_rc" ] && [ -z "$(pp_left)" ] &&
    [ "$(cat "$TMP/pp-h.affected.harnesses")" = "--harnesses --since base" ] &&
    { if [ "$calls" = kept ]; then [ "$argv_calls" = "$pp_kept" ]; else [ -z "$argv_calls" ]; fi; } &&
    { case $args in *--list*) [ "$argv_out" = "$(grep -E '/runtest-|^@@runtest-' "$out")" ] ;; esac; } &&
    { if [ "$named" = named ]; then
        [ "$n_count" = 1 ] && [ "$n_named" = 2 ] &&
          [ "$(tail -n 4 <<<"$argv_err" | sed '$d')" = "$(grep '^test-run: prepush: 2 harness command(s) this change reaches, named here and NOT run' <<<"$argv_err")
test-run: prepush: harness: tools/test-a.sh
test-run: prepush: harness: tools/test-b.sh --only '^1:'" ] &&
          # The closing line's command is held by tools/test-test-harnesses.sh, which runs it as
          # printed: naming the runner here would make every edit of it reach this whole harness.
          case $(tail -n 1 <<<"$argv_err") in
            'test-run: prepush: run them before pushing, as one command that takes its own --cpu slot: '*' --affected --since base') ;;
            *) false ;;
          esac
      else
        [ "$n_count" = 0 ] && [ "$n_named" = 0 ]
      fi; } &&
    { [ "$tag" != ask-failed ] || grep -qF 'the harness commands this change reaches are not known; nothing run' <<<"$argv_err"; }; } ||
    pp_detail="$tag ($args): exit $argv_rc (want $want_rc); dune calls: ${argv_calls:-<none>} (want $calls); harnesses asked: $(tr '\n' '|' <"$TMP/pp-h.affected.harnesses"); out: $argv_out; stderr: $argv_err"
}
for probe in "${pp_harness_cases[@]}"; do
  pp_harness_check "$probe"
  [ -z "$pp_detail" ] || break
done
if [ -z "$pp_detail" ]; then report 0 "$pp_harness_label"; else report 1 "$pp_harness_label" "$pp_detail"; fi
pp_harness_control() { # LABEL FROM TO WANT: the cases WANT alone hold under the mutant
  local label="negative control: $1" want=$4
  FROM=$2 TO=$3 \
    perl -0pe 'BEGIN { $n = 0 } $n += s/\Q$ENV{FROM}\E/$ENV{TO}/g; END { exit($n == 1 ? 0 : 1) }' \
    "$SRC" >"$repeat_root/tools/test-run.sh"
  # shellcheck disable=SC2181 # perl's status, read after the redirection
  if [ $? != 0 ] || ! bash -n "$repeat_root/tools/test-run.sh"; then
    report 1 "$label" "the mutant was not made"
    return
  fi
  pp_held=
  for pp_i in "${!pp_harness_cases[@]}"; do
    pp_harness_check "${pp_harness_cases[$pp_i]}"
    [ -n "$pp_detail" ] || pp_held="$pp_held $pp_i"
  done
  if [ "$pp_held" = "$want" ]; then report 0 "$label"; else report 1 "$label" "held under the mutant: cases${pp_held:- <none>} (want$want)"; fi
}
# Prepush as it was (gh-ocannl-1829), keeping none of the commands it is told,
# and a closing line that does not carry the run's BASE. Only the cases with
# nothing to name hold: the two `.ml`-only ones (1, 3) and the failed ask (8).
pp_harness_control 'a prepush that names no harness command turns every case with one to name red' \
  '[ -z "$pp_line" ] || pp_harness+=("$pp_line")' ':' ' 1 3 8'
pp_harness_control 'a closing line naming another base than the run'\''s turns every case with one to name red' \
  '"$pp_base"
      } >&2' '"master"
      } >&2' ' 1 3 8'
cp "$SRC" "$repeat_root/tools/test-run.sh"
rm -f "$pp_stub"
}
harness_leg_run '98: prepush runs affected'\''s list without slow-/train- members #options' leg_98

test_run_close

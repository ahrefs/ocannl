#!/usr/bin/env bash
# Refusal checks match literal reasons for the named fixture cases; they do not
# parse complete usage output or classify arbitrary diagnostics.
# The shard checks read only the fixture member paths in HARNESS headers or on
# --list's stdout. Per-leg output and wording are outside their membership check.
# Opposing controls for tools/harness-legs-alone.sh (gh-ocannl-1367): on a
# fixture repository whose manifest names a gated and an ungated harness, a
# clean harness passes leg by leg and group by group; a leg that reads what an
# earlier leg's body made is ALONE FAIL "passes in the full run" while its
# group passes; a leg that skips alone what the full run decided is ALONE
# SKIP; a red full run is told apart; a naming defect fails the listing; and
# a checker that does not escape leg names loses on names with regular
# expression characters, failing one and over-selecting another.
# Usage: tools/test-harness-legs-alone.sh [--keep|--help]
set -u
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch test-harness-legs-alone
fixture="$TMP/repository with spaces"
mkdir -p "$fixture/tools" "$fixture/scripts" "$fixture/fixtures"
cp "$here/harness-legs-alone.sh" "$here/harness-legs-alone-impl.sh" "$fixture/tools/" &&
  cp "$here/../scripts/harness-support.sh" "$here/../scripts/process-group.sh" "$fixture/scripts/" || exit 2
cat >"$fixture/tools/test-harnesses.sh" <<'STUB'
#!/usr/bin/env bash
if [ -n "${FAKE_LOCALE_OUT:-}" ]; then
  printf 'LANG=%s LC_ALL=%s LC_CTYPE=%s\n' \
    "${LANG-<unset>}" "${LC_ALL-<unset>}" "${LC_CTYPE-<unset>}" >"$FAKE_LOCALE_OUT"
fi
[ "$*" = --list ] || exit 2
if [ "${DISCOVERY:-0}" = list ]; then
  printf '%s\n' fixtures/list-d.sh fixtures/commented.sh fixtures/list-b.sh fixtures/list-c.sh fixtures/list-a.sh
  exit 0
fi
printf '%s\n' fixtures/gated.sh fixtures/ungated.sh
[ "${DISCOVERY:-0}" = 0 ] || printf '%s\n' fixtures/continued.sh fixtures/commented.sh
STUB
cat >"$fixture/fixtures/gated.sh" <<'GATED'
#!/usr/bin/env bash
# Fixture harness: #a (1-2), #b (3-4) and #c (the dot legs).
set -u
. "$(dirname "$0")/../scripts/harness-support.sh"
harness_args "$@"
harness_scratch gated
leg_1() {
  if [ "${SHAREDSKIP:-0}" = 1 ]; then skip 'shared label' 'one never decides it'; shared=made; fi
  if [ -n "${HOLD:-}" ]; then
    echo $$ >"$HOLD"
    while :; do sleep 0.2; done
  fi
  report 0 'one'
  [ "${LEAK:-0}" = 0 ] || shared=made
  [ "${SKIPLEAK:-0}" = 0 ] || touch "$TMP/from-one"
}
leg_2() {
  if [ "${SHAREDSKIP:-0}" = 1 ] && [ "${shared:-}" != made ]; then skip 'shared label' 'two alone'
  elif [ "${LEAK:-0}" = 1 ] && [ "${shared:-}" != made ]; then report 1 'two reads what one made'
  else report 0 'two'; fi
}
leg_3() {
  if [ "${SKIPLEAK:-0}" = 1 ] && [ ! -f "$TMP/from-one" ]; then skip 'three check' 'nothing from one'
  else report 0 'three'; fi
}
leg_4() {
  [ "${SELFTERM:-0}" = 0 ] || exit 143
  [ "${LITTER:-0}" = 0 ] || : >"$(dirname "$0")/../litter"
  [ "${REWRITE:-0}" = 0 ] || echo rewritten >"$(dirname "$0")/../scribble"
  [ "${EMPTYDIR:-0}" = 0 ] || mkdir -p "$(dirname "$0")/../hollow"
  report "${RED:-0}" 'four'
}
dot() { report 0 "$1"; }
harness_leg_run '1: one makes things #a' leg_1
harness_leg_run '2: two (x.y) reads them #a' leg_2
harness_leg_run '3: three [z]+ #b' leg_3
harness_leg_run '4: four #b' leg_4
[ "${DUP:-0}" = 0 ] || harness_leg_run '4: four #b' leg_4
if harness_leg 'dot a.c #c'; then dot a.c; fi
if harness_leg 'dot abc #c'; then dot abc; fi
finish
GATED
cat >"$fixture/fixtures/ungated.sh" <<'UNGATED'
#!/usr/bin/env bash
# Fixture harness gating nothing on this host.
set -u
. "$(dirname "$0")/../scripts/harness-support.sh"
harness_args "$@"
[ "${BROKEN:-0}" = 0 ] || { echo 'FAIL  shared fixture: cannot build'; exit 1; }
if [ "${NEVER:-0}" = 1 ] && harness_leg 'never'; then report 0 'never'; fi
report 0 'ungated'
finish
UNGATED
cat >"$fixture/fixtures/continued.sh" <<'CONTINUED'
#!/usr/bin/env bash
# Fixture harness whose one gate continues onto the next line.
set -u
. "$(dirname "$0")/../scripts/harness-support.sh"
harness_args "$@"
only() { report 0 'only'; }
harness_leg_run\
  'only leg' only
finish
CONTINUED
cat >"$fixture/fixtures/commented.sh" <<'COMMENTED'
#!/usr/bin/env bash
# Fixture harness naming harness_leg_run only in this comment; running it fails.
exit 1
COMMENTED
for member in a b c d; do
  cat >"$fixture/fixtures/list-$member.sh" <<'LIST_PROBE'
#!/usr/bin/env bash
: >"$LIST_STARTED"
harness_leg never
exit 9
LIST_PROBE
done
chmod +x "$fixture/tools/"*.sh "$fixture/fixtures/"*.sh
# The checkout every run must leave as it found it.
git -C "$fixture" init -q && git -C "$fixture" add -A &&
  git -C "$fixture" -c user.name=t -c user.email=t@t commit -q -m fixture || exit 2
subject="$fixture/tools/harness-legs-alone.sh"
subject_impl="$fixture/tools/harness-legs-alone-impl.sh"
unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_SKIP_LEGS LEAK SKIPLEAK SHAREDSKIP RED DUP NEVER BROKEN HOLD \
  LITTER REWRITE EMPTYDIR SELFTERM DISCOVERY
run() { # [subject arguments] -- from an unrelated directory
  rc=0
  (cd "$TMP" && "${SUBJECT:-$subject}" "$@") >"$TMP/out" 2>&1 || rc=$?
}
has() { grep -qxF -- "$1" "$TMP/out"; }
refusal_matches() {
  [ "$rc" = 2 ] && grep -qF -- "$reason" "$TMP/out" && ! grep -q 'ALONE' "$TMP/out"
}

leg_isolation() {
  legs='1: one makes things #a
2: two (x.y) reads them #a
3: three [z]+ #b
4: four #b
dot a.c #c
dot abc #c'

  run
  ok=1
  for sel in '#a' '#b' '#c'; do has "  ALONE PASS  $sel" || ok=0; done
  while IFS= read -r leg; do has "  ALONE PASS  $leg" || ok=0; done <<<"$legs"
  if [ "$rc" = 0 ] && [ "$ok" = 1 ] && [ "$(grep -c '^  ALONE ' "$TMP/out")" = 9 ] \
     && has 'HARNESS fixtures/ungated.sh: gates no leg on this host; not checked' \
     && has 'HARNESS fixtures/gated.sh: 6 leg(s), 3 group(s)' \
     && has '  full run: PASS (0 check(s) skipped)' \
     && has 'harness-legs-alone: 1 harness(es) checked; 0 red full run(s), 0 failed listing(s), 0 finding(s) run alone' \
     && ! grep -q '^logs: ' "$TMP/out"; then
    report 0 'every leg and group of a clean harness passes alone; the manifest picks the gated member'
  else report 1 'every leg and group of a clean harness passes alone; the manifest picks the gated member' "exit $rc; $(cat "$TMP/out")"; fi

  LEAK=1 run
  if [ "$rc" = 1 ] && grep -q '^  ALONE FAIL  2: two (x.y) reads them #a (exit 1; passes in the full run): ' "$TMP/out" \
     && has '        FAIL  two reads what one made' && has '  ALONE PASS  #a' \
     && [ "$(grep -c '^  ALONE \(FAIL\|MISS\|SKIP\)' "$TMP/out")" = 1 ] && grep -q '^logs: ' "$TMP/out"; then
    report 0 'a leg reading an earlier leg'"'"'s state fails alone and passes in the full run and its group'
  else report 1 'a leg reading an earlier leg'"'"'s state fails alone and passes in the full run and its group' "exit $rc; $(cat "$TMP/out")"; fi

  SKIPLEAK=1 run
  if [ "$rc" = 1 ] && grep -q '^  ALONE SKIP  3: three \[z\]+ #b: skips alone what the full run decided: ' "$TMP/out" \
     && grep -q '^  ALONE SKIP  #b: ' "$TMP/out" && has '        SKIP  three check  (in leg: 3: three [z]+ #b)' \
     && [ "$(grep -c '^  ALONE \(FAIL\|MISS\|SKIP\)' "$TMP/out")" = 2 ]; then
    report 0 'a check skipped alone that the full run decided is a finding'
  else report 1 'a check skipped alone that the full run decided is a finding' "exit $rc; $(cat "$TMP/out")"; fi

  SHAREDSKIP=1 run
  if [ "$rc" = 1 ] && grep -q '^  ALONE SKIP  2: two (x.y) reads them #a: ' "$TMP/out" \
     && has '        SKIP  shared label  (in leg: 2: two (x.y) reads them #a)' \
     && [ "$(grep -c '^  ALONE \(FAIL\|MISS\|SKIP\)' "$TMP/out")" = 1 ]; then
    report 0 'a label one leg skips in the full run does not cover another leg skipping it alone'
  else report 1 'a label one leg skips in the full run does not cover another leg skipping it alone' "exit $rc; $(cat "$TMP/out")"; fi

  RED=1 run
  if [ "$rc" = 1 ] && grep -q '^  full run: FAIL (exit 1): ' "$TMP/out" \
     && grep -q '^  ALONE FAIL  4: four #b (exit 1; the full run is red too (exit 1)): ' "$TMP/out" \
     && has 'harness-legs-alone: 1 harness(es) checked; 1 red full run(s), 0 failed listing(s), 2 finding(s) run alone'; then
    report 0 'a red full run is reported, and a leg red in it too is told apart'
  else report 1 'a red full run is reported, and a leg red in it too is told apart' "exit $rc; $(cat "$TMP/out")"; fi

  DUP=1 run fixtures/gated.sh
  if [ "$rc" = 1 ] && grep -q '^  LISTING FAILED (exit 1): ' "$TMP/out" \
     && has "        FAIL  leg naming: two legs are named '4: four #b'"; then
    report 0 'a naming defect fails the listing'
  else report 1 'a naming defect fails the listing' "exit $rc; $(cat "$TMP/out")"; fi

  BROKEN=1 run
  if [ "$rc" = 1 ] && grep -q '^HARNESS fixtures/ungated.sh: LISTING FAILED (exit 1) before any gate: ' "$TMP/out" \
     && has '        FAIL  shared fixture: cannot build' \
     && has 'harness-legs-alone: 1 harness(es) checked; 0 red full run(s), 1 failed listing(s), 0 finding(s) run alone'; then
    report 0 'a listing that dies before its first gate is a failed listing, not a host without legs'
  else report 1 'a listing that dies before its first gate is a failed listing, not a host without legs' "exit $rc; $(cat "$TMP/out")"; fi

  LITTER=1 run fixtures/gated.sh
  rm -f "$fixture/litter"
  if [ "$rc" = 1 ] && grep -q '^  CHECKOUT CHANGED  by fixtures/gated.sh : ' "$TMP/out" \
     && has '        ?? litter' \
     && has 'harness-legs-alone: 1 harness(es) checked; 0 red full run(s), 0 failed listing(s), 1 finding(s) run alone'; then
    report 0 'a run that leaves a file in the checkout is a finding'
  else report 1 'a run that leaves a file in the checkout is a finding' "exit $rc; $(cat "$TMP/out")"; fi

  echo scribbled >"$fixture/scribble"
  REWRITE=1 run fixtures/gated.sh
  rm -f "$fixture/scribble"
  if [ "$rc" = 1 ] && [ "$(grep -c '^  CHECKOUT CHANGED  ' "$TMP/out")" = 1 ] \
     && grep -q '^  CHECKOUT CHANGED  by fixtures/gated.sh : ' "$TMP/out"; then
    report 0 'a run rewriting a file the checkout already had untracked is a finding'
  else report 1 'a run rewriting a file the checkout already had untracked is a finding' "exit $rc; $(cat "$TMP/out")"; fi

  # Git lists no FIFO, but it does list a symbolic link, and hashing one opens
  # its target: a link to a FIFO is recorded by its target, never opened.
  mkfifo "$TMP/pipe"
  ln -s "$TMP/pipe" "$fixture/pipe-link"
  rc=0
  (cd "$TMP" && perl -e 'alarm 60; exec @ARGV' "$subject" fixtures/gated.sh) >"$TMP/out" 2>&1 || rc=$?
  rm -f "$fixture/pipe-link" "$TMP/pipe"
  if [ "$rc" = 0 ] && has 'harness-legs-alone: 1 harness(es) checked; 0 red full run(s), 0 failed listing(s), 0 finding(s) run alone'; then
    report 0 'an untracked link to a FIFO is recorded without being opened'
  else report 1 'an untracked link to a FIFO is recorded without being opened' "exit $rc; $(cat "$TMP/out")"; fi

  SELFTERM=1 run fixtures/gated.sh
  if [ "$rc" = 1 ] && grep -q '^  full run: FAIL (exit 143): ' "$TMP/out" \
     && grep -q '^  ALONE FAIL  4: four #b (exit 143; ' "$TMP/out" && ! grep -q 'interrupted' "$TMP/out" \
     && grep -q '^harness-legs-alone: 1 harness(es) checked; 1 red full run(s)' "$TMP/out"; then
    report 0 'a harness exiting 143 on its own is a red run, not an interrupt of the check'
  else report 1 'a harness exiting 143 on its own is a red run, not an interrupt of the check' "exit $rc; $(cat "$TMP/out")"; fi

  EMPTYDIR=1 run fixtures/gated.sh
  rmdir "$fixture/hollow" 2>/dev/null
  if [ "$rc" = 1 ] && [ "$(grep -c '^  CHECKOUT CHANGED  ' "$TMP/out")" = 1 ] && has '        hollow/'; then
    report 0 'a run leaving an empty directory in the checkout is a finding'
  else report 1 'a run leaving an empty directory in the checkout is a finding' "exit $rc; $(cat "$TMP/out")"; fi

  run --logs "$fixture/logs" fixtures/gated.sh
  if [ "$rc" = 2 ] && grep -q 'is inside the checkout' "$TMP/out" && ! grep -q 'ALONE' "$TMP/out"; then
    report 0 'a log directory inside the checkout is refused'
  else report 1 'a log directory inside the checkout is refused' "exit $rc; $(cat "$TMP/out")"; fi
  rm -rf "$fixture/logs"
  mkdir -p "$fixture/logs"
  ln -s "$fixture/logs" "$TMP/logs-link"
  run --logs "$TMP/logs-link" fixtures/gated.sh
  if [ "$rc" = 2 ] && grep -q 'is inside the checkout' "$TMP/out" && ! grep -q 'ALONE' "$TMP/out"; then
    report 0 'a log directory reached through a symbolic link into the checkout is refused'
  else report 1 'a log directory reached through a symbolic link into the checkout is refused' "exit $rc; $(cat "$TMP/out")"; fi
  rm -rf "$fixture/logs" "$TMP/logs-link"

  DISCOVERY=1 run
  if [ "$rc" = 0 ] && has 'HARNESS fixtures/continued.sh: 1 leg(s), 0 group(s)' && has '  ALONE PASS  only leg' \
     && ! grep -q 'commented' "$TMP/out"; then
    report 0 'discovery: a continued gate line is found, a gate named in a comment is not'
  else report 1 'discovery: a continued gate line is found, a gate named in a comment is not' "exit $rc; $(cat "$TMP/out")"; fi

  # A TERM to the check stops the harness it is running, and waits for it.
  hold="$TMP/hold.pid"
  rm -f "$hold"
  (cd "$TMP" && exec env HOLD="$hold" "$subject" fixtures/gated.sh) >"$TMP/out" 2>&1 &
  checker=$!
  for _ in $(seq 100); do [ -s "$hold" ] && break; sleep 0.1; done
  if [ -s "$hold" ]; then
    member_pid=$(cat "$hold")
    kill -TERM "$checker"
    rc=0
    wait "$checker" || rc=$?
    if [ "$rc" = 143 ] && ! kill -0 "$member_pid" 2>/dev/null && grep -q '^harness-legs-alone: interrupted (exit 143) in ' "$TMP/out"; then
      report 0 'TERM to the check stops the running harness first and keeps its status'
    else
      kill -TERM "$member_pid" 2>/dev/null
      report 1 'TERM to the check stops the running harness first and keeps its status' "exit $rc; $(cat "$TMP/out")"
    fi
  else
    kill -TERM "$checker" 2>/dev/null
    wait "$checker" 2>/dev/null
    report 1 'TERM to the check stops the running harness first and keeps its status' "the held leg never started: $(cat "$TMP/out")"
  fi

  kept="$(cd "$TMP" && pwd -P)/kept"
  run --logs "$kept" fixtures/gated.sh
  if [ "$rc" = 0 ] && has "logs: $kept" && [ -s "$kept/fixtures_gated.sh/full" ] \
     && ! grep -q 'ungated' "$TMP/out"; then
    report 0 '--logs keeps every run, and a named harness replaces the manifest'
  else report 1 '--logs keeps every run, and a named harness replaces the manifest' "exit $rc; $(cat "$TMP/out")"; fi

  cat >"$TMP/unrelated-subject" <<'STUB'
#!/bin/bash
echo 'unrelated refusal control' >&2
exit 2
STUB
  chmod +x "$TMP/unrelated-subject"
  for refused in 'fixtures/ungated.sh' '--bogus' 'fixtures/missing.sh'; do
    case $refused in
      fixtures/ungated.sh) reason='fixtures/ungated.sh lists no leg (exit 0;' ;;
      --bogus) reason="unknown argument '--bogus'" ;;
      fixtures/missing.sh) reason='no harness fixtures/missing.sh' ;;
    esac
    SUBJECT="$TMP/unrelated-subject" run "$refused"
    if [ "$rc" != 2 ] || ! grep -qxF 'unrelated refusal control' "$TMP/out" || refusal_matches; then
      report 1 "negative control: refusal rejects unrelated exit 2: $refused"
    else report 0 "negative control: refusal rejects unrelated exit 2: $refused"; fi
    run "$refused"
    if refusal_matches; then
      report 0 "refused with exit 2: $refused"
    else report 1 "refused with exit 2: $refused" "exit $rc; $(cat "$TMP/out")"; fi
  done

  # Negative control: a checker that hands leg names to --only unescaped.
  mutant="$fixture/tools/harness-legs-alone-unescaped.sh"
  awk '/^ere_escape\(\) \{$/ { print; print "  printf %s \"$1\"; return"; next } { print }' "$subject_impl" >"$mutant"
  chmod +x "$mutant"
  if cmp -s "$subject_impl" "$mutant"; then
    report 1 'negative control: unescaped leg names' 'the escape could not be cut out of the checker'
  else
    SUBJECT=$mutant run fixtures/gated.sh
    if [ "$rc" = 1 ] && grep -q '^  ALONE MISS  2: two (x.y) reads them #a: --only .* selects 0 leg(s)' "$TMP/out" \
       && grep -q '^  ALONE MISS  dot a.c #c: --only .* selects 2 leg(s)' "$TMP/out"; then
      report 0 'negative control: unescaped names select nothing or more than the leg'
    else report 1 'negative control: unescaped names select nothing or more than the leg' "exit $rc; $(cat "$TMP/out")"; fi
  fi
}

fixture_members() {
  sed -n 's/^HARNESS \([^:]*\): .*/\1/p' "$TMP/out"
}

shard_members() { # SPEC... -- write one ordered list per shard and their sorted union
  local spec
  : >"$TMP/shards"
  for spec in "$@"; do
    DISCOVERY=1 run --shard "$spec"
    if [ "$rc" != 0 ]; then
      cat "$TMP/out" >&2
      return 1
    fi
    fixture_members >"$TMP/shard-${spec%/*}"
    cat "$TMP/shard-${spec%/*}" >>"$TMP/shards"
  done
  LC_ALL=C sort "$TMP/shards" >"$TMP/shards-sorted"
}

complete_shard_membership() {
  cmp -s "$TMP/unsharded-sorted" "$TMP/shards-sorted"
}

leg_shards() {
  local reason
  DISCOVERY=1 run
  if [ "$rc" != 0 ]; then harness_abandon "unsharded fixture exited $rc"; return; fi
  fixture_members >"$TMP/unsharded"
  LC_ALL=C sort "$TMP/unsharded" >"$TMP/unsharded-sorted"
  sed -n '1p;3p' "$TMP/unsharded" >"$TMP/want-first"
  sed -n '2p' "$TMP/unsharded" >"$TMP/want-second"

  if shard_members 1/2 2/2 && complete_shard_membership &&
     cmp -s "$TMP/want-first" "$TMP/shard-1" && cmp -s "$TMP/want-second" "$TMP/shard-2"; then
    report 0 'shards deal the unsharded members in manifest order, each once'
  else
    report 1 'shards deal the unsharded members in manifest order, each once'
  fi
  if shard_members 1/3 2/3 && ! complete_shard_membership &&
     [ "$(LC_ALL=C comm -13 "$TMP/shards-sorted" "$TMP/unsharded-sorted")" = fixtures/continued.sh ]; then
    report 0 'negative control: the membership check catches an omitted shard'
  else
    report 1 'negative control: the membership check catches an omitted shard'
  fi

  run --shard 3/2
  reason='--shard needs K/N with 1 <= K <= N'
  if refusal_matches; then report 0 'a shard index outside its total is refused'
  else report 1 'a shard index outside its total is refused' "exit $rc; $(cat "$TMP/out")"; fi

  run --shard 1/2 fixtures/gated.sh
  reason='--shard deals the manifest and takes no named harnesses'
  if refusal_matches; then report 0 'a shard and named harness together are refused'
  else report 1 'a shard and named harness together are refused' "exit $rc; $(cat "$TMP/out")"; fi
}

leg_list() {
  local started=$TMP/list-started shard reason
  DISCOVERY=list LIST_STARTED=$started run --list
  if [ "$rc" != 0 ] || ! cmp -s "$TMP/out" <(
    printf '%s\n' fixtures/list-d.sh fixtures/list-b.sh fixtures/list-c.sh fixtures/list-a.sh
  ); then
    harness_abandon "candidate list: exit $rc; $(cat "$TMP/out")"
    return
  fi
  cp "$TMP/out" "$TMP/list-all"
  : >"$TMP/list-shards"
  for shard in 1 2 3 4; do
    DISCOVERY=list LIST_STARTED=$started run --list --shard "$shard/4"
    if [ "$rc" != 0 ]; then
      harness_abandon "candidate list for shard $shard/4: exit $rc; $(cat "$TMP/out")"
      return
    fi
    cat "$TMP/out" >>"$TMP/list-shards"
  done
  # Four candidates across four shards also pin the dealing order.
  if cmp -s "$TMP/list-all" "$TMP/list-shards" && [ ! -e "$started" ]; then
    report 0 '--list shards enumerate the unsharded candidates once each without starting a harness'
  else
    report 1 '--list shards enumerate the unsharded candidates once each without starting a harness'
  fi

  run --list fixtures/gated.sh
  reason='--list selects from the manifest and takes no named harnesses'
  if refusal_matches; then report 0 '--list with a named harness is refused'
  else report 1 '--list with a named harness is refused' "exit $rc; $(cat "$TMP/out")"; fi

  rc=0
  LC_ALL=C DISCOVERY=list LIST_STARTED=$started "$subject" --list 1>&- 2>"$TMP/out" || rc=$?
  if [ "$rc" = 1 ] && grep -qF 'Bad file descriptor' "$TMP/out"; then
    report 0 '--list reports a failed write'
  else
    report 1 '--list reports a failed write' "exit $rc; $(cat "$TMP/out")"
  fi
}

leg_locale() {
  local seen=$TMP/hla-locale
  run_locale() { # ENV_ARGS... -- SUBJECT_ARGS...
    local env_args=()
    while [ $# -gt 0 ] && [ "$1" != -- ]; do env_args+=("$1"); shift; done
    shift
    rc=0
    rm -f "$seen"
    (cd "$TMP" && env ${env_args[@]+"${env_args[@]}"} FAKE_LOCALE_OUT="$seen" "$subject" "$@") >"$TMP/out" 2>&1 || rc=$?
  }
  run_locale -u LANG -u LC_ALL LC_CTYPE=UTF-8 --
  if [ "$rc" = 0 ] &&
     has 'harness-legs-alone: 1 harness(es) checked; 0 red full run(s), 0 failed listing(s), 0 finding(s) run alone' &&
     [ "$(cat "$seen" 2>/dev/null)" = 'LANG=C LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 'unset LANG/LC_ALL exports LANG=C while keeping LC_CTYPE and passing full+alone checks'
  else
    report 1 'unset LANG/LC_ALL exports LANG=C while keeping LC_CTYPE and passing full+alone checks' \
      "exit $rc; locale=$(cat "$seen" 2>/dev/null); $(cat "$TMP/out")"
  fi
  run_locale LANG= LC_ALL= LC_CTYPE=UTF-8 -- --list
  if [ "$rc" = 0 ] && [ "$(cat "$seen" 2>/dev/null)" = 'LANG=C LC_ALL= LC_CTYPE=UTF-8' ]; then
    report 0 'empty LANG/LC_ALL exports LANG=C while keeping LC_CTYPE'
  else
    report 1 'empty LANG/LC_ALL exports LANG=C while keeping LC_CTYPE' \
      "exit $rc; locale=$(cat "$seen" 2>/dev/null); $(cat "$TMP/out")"
  fi
  run_locale -u LC_ALL LANG=en_US.UTF-8 LC_CTYPE=UTF-8 -- --list
  if [ "$rc" = 0 ] && [ "$(cat "$seen" 2>/dev/null)" = 'LANG=en_US.UTF-8 LC_ALL=<unset> LC_CTYPE=UTF-8' ]; then
    report 0 'caller-supplied LANG is preserved untouched'
  else
    report 1 'caller-supplied LANG is preserved untouched' \
      "exit $rc; locale=$(cat "$seen" 2>/dev/null); $(cat "$TMP/out")"
  fi
  run_locale -u LANG LC_ALL=POSIX LC_CTYPE=UTF-8 -- --list
  if [ "$rc" = 0 ] && [ "$(cat "$seen" 2>/dev/null)" = 'LANG=<unset> LC_ALL=POSIX LC_CTYPE=UTF-8' ]; then
    report 0 'caller-supplied LC_ALL is preserved untouched'
  else
    report 1 'caller-supplied LC_ALL is preserved untouched' \
      "exit $rc; locale=$(cat "$seen" 2>/dev/null); $(cat "$TMP/out")"
  fi
}

harness_leg_run 'leg isolation controls' leg_isolation
harness_leg_run 'shard membership and dealing' leg_shards
harness_leg_run 'list candidates and shards without running members' leg_list
harness_leg_run 'unset-locale startup guard' leg_locale
finish

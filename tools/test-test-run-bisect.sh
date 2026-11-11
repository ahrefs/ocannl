#!/usr/bin/env bash
# Tests for tools/test-run.sh: `bisect`, the credentials and runner variables kept from
# dune, and a structural failure ending its own leg alone.
#
#   tools/test-test-run-bisect.sh          # run every leg of this member
#   tools/test-test-run-bisect.sh --keep   # keep the temp dir for inspection
#   tools/test-test-run-bisect.sh --only '^82:'   # only the legs whose names match
#   tools/test-test-run-bisect.sh --subject FILE   # test FILE, not tools/test-run.sh
#
# One member of the harness of tools/test-run.sh (gh-ocannl-2002), whose
# others `tools/test-test-run.sh --members` lists; that script runs them all,
# or the ones holding the legs an --only names. The conventions every member
# shares -- --subject, how a leg is named, gated and abandoned, what is tested
# and the fixtures the legs run against -- are tools/test-test-run-common.sh's
# header. Leg numbers are the harness's, not this file's: they stayed as they
# were when it was one file, so a member's are not consecutive.
#
# Groups: #credentials (79, 90), #bisect (82, 121), #harness (86, 105), #hermetic (70).
#
# Legs:
#  79. credentials never reach dune (gh-ocannl-1280): a run exporting GH_TOKEN
#      and FOO_API_KEY hands dune neither, while a plain variable and the
#      OCANNL_TOOL_* exports pass -- observed by a fixture dune, and where a
#      real dune is at hand in the run's own trace; a copy with the scrub
#      cut out hands dune both, the negative control for each observer. With
#      no dune on PATH, credentials the opam environment select_dune sources
#      sets are scrubbed too, against a copy without that second scrub; and
#      tools/dune-quiet.sh, run directly, scrubs on its own. The scrub text
#      itself fails closed, under sh, bash and dash where installed: an
#      unrunnable env and a deny-listed non-identifier refuse.
#  90. the runner's own OCANNL_TOOL_TESTRUN_BG, _RD and _OWN reach no dune: a
#      run, a probed GPU run under a slot, a probed repeat's iterations.
#  82 sits just before leg 70: `bisect` (gh-ocannl-1225), over fixture
#      repositories whose commits tell the fixture dune what to do. A red
#      build is a skip, never bad, and the first bad commit is found under
#      the runner frozen at the start -- the range's last commit carries one
#      that cannot run; a range left with only skipped commits lists them
#      (exit 1), and so is a test run that is BUILD FAILED (gh-ocannl-1547);
#      SLOT REFUSED and a cap's TIMEOUT in a step abort the bisect
#      (75, 142; git stopped by the step's 128) judging nothing; ends that do
#      not hold here bisect nothing; no backend, an unrelated GOOD or a
#      misplaced option is refused before any step; first-parent by default
#      never steps into a merged branch, and refuses a GOOD merged in through
#      a second parent (--no-first-parent takes it); a TERM reaps the step's dune and
#      exits 143; a descendant a step's run leaves holding the tree is
#      waited for before git moves the tree; a relative runs directory keeps
#      every step's runs under it; a dune-workspace above the runs directory
#      is refused leaving no worktree behind; a TERM during setup, or before a
#      step has a group to signal, unwinds with 143 starting nothing. Every
#      dune call sees the pinned backend and neither of the
#      runner's own variables, and the invoking checkout and its worktree
#      list end as they began.
#  86 sits after leg 70: a copy of this harness, its test-run.sh respelled so
#      leg 79's credential mutant cannot be cut, abandons leg 79 alone and
#      still runs leg 70 to the footer; the same copy with that check spelled
#      as the `exit 2` it was aborts inside leg 79, and leg 70 never reports
#      (gh-ocannl-1374).
# 105 sits last: tools/test-test-run.sh, beside a stand-in list of two members,
#      runs the members listing a leg its selector matches, each with every
#      word, and none for a selector no member holds (exit 2); with no
#      selector it runs them all, exiting 1 after a failed one, 2 after a
#      refused one, and at once with an interrupted one's status
#      (gh-ocannl-2002).

. "$(cd "$(dirname "$0")" && pwd)/test-test-run-common.sh"

shared_through renamed_dune

# ---------------------------------------------------------------------------
# Leg 79: credentials never reach dune (gh-ocannl-1280)
# ---------------------------------------------------------------------------
# Dune records the environment of every process it spawns in the worktree's
# `_build/trace.csexp`, so a token the session exports would sit inside the
# checkout after every build. The run below carries GH_TOKEN (a deny-listed
# name) and FOO_API_KEY (matched by a pattern), both with obviously fake
# values, beside CRED_TEST_PLAIN and the harness's own OCANNL_TOOL_* exports,
# which must still pass. Two observers: a fixture dune that writes the names of
# its environment -- decided on every host -- and, where a real dune is at hand,
# that dune's own trace. The negative control is the shipping script with its
# scrub cut out, which must hand dune both credentials: an observer that cannot
# see them proves nothing about the scrub.
leg_79() {
cred_label="credentials: dune runs without GH_TOKEN and FOO_API_KEY, a plain variable and OCANNL_TOOL_* still pass"
cred_nc_label="negative control: with the scrub cut out, dune sees both credentials"
cred_trace_label="credentials: the run's real dune trace names neither GH_TOKEN nor FOO_API_KEY, and does name the plain variable"
cred_trace_nc_label="negative control: with the scrub cut out, the real trace names both credentials"
cred_mutant=$(mutant no-credential-scrub \
  '/^  eval "\$\(credential_env_scrub_text\)" \|\|$/ { getline; print "  :"; next } { print }')
if [ -z "$cred_mutant" ] || cmp -s "$SRC" "$cred_mutant"; then
  harness_abandon "the credential scrub could not be cut out of $SRC; the mutant is broken"
  return
fi
cred_bin=$TMP/cred-bin
mkdir -p "$cred_bin"
cat >"$cred_bin/dune" <<'EOF'
#!/usr/bin/env bash
# The run's last phase asks for the version and the promotion list: an empty one.
case ${1:-} in
  --version) echo 3.24.2; exit 0 ;;
  promotion) exit 0 ;;
esac
env | sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)=.*/\1/p' >"$CRED_TEST_NAMES"
EOF
chmod +x "$cred_bin/dune"
cred_root() { # name script -> a fixture root running SCRIPT as tools/test-run.sh
  local root=$TMP/$1
  mkdir -p "$root/tools" "$root/scripts" "$root/_build"
  cp "$2" "$root/tools/test-run.sh"
  stage_sourced "$root" || return 1
  chmod +x "$root/tools/test-run.sh"
  printf '%s' "$root"
}
cred_run() { # root runs-dir names-file subcommand argv...: the launch, with the two credentials exported
  local root=$1 runs=$2 names=$3
  shift 3
  GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key CRED_TEST_PLAIN=kept \
    CRED_TEST_NAMES=$names OCANNL_TOOL_TEST_RUNS=$runs \
    "$root/tools/test-run.sh" "$@" >"$names.out" 2>&1
}
cred_has() { grep -qx -e "$1" "$2"; }
cred_ship=$(cred_root cred-ship "$SRC") && cred_cut=$(cred_root cred-cut "$cred_mutant") ||
  { harness_abandon "cannot stage the credential fixture roots (what test-run.sh sources)"; return; }
cred_rc=0
PATH=$cred_bin:$PATH cred_run "$cred_ship" "$TMP/cred-runs" "$TMP/cred-ship.names" run build @fixture ||
  cred_rc=$?
cred_detail=
if [ "$cred_rc" != 0 ]; then
  cred_detail="the run exited $cred_rc: $(tail -5 "$TMP/cred-ship.names.out")"
elif [ ! -s "$TMP/cred-ship.names" ]; then
  cred_detail="the fixture dune recorded no environment"
elif cred_has GH_TOKEN "$TMP/cred-ship.names"; then
  cred_detail="dune saw GH_TOKEN"
elif cred_has FOO_API_KEY "$TMP/cred-ship.names"; then
  cred_detail="dune saw FOO_API_KEY"
elif ! cred_has CRED_TEST_PLAIN "$TMP/cred-ship.names"; then
  cred_detail="dune did not see CRED_TEST_PLAIN"
elif ! cred_has OCANNL_TOOL_DEVICE_PROBE "$TMP/cred-ship.names"; then
  cred_detail="dune did not see the harness's OCANNL_TOOL_DEVICE_PROBE"
fi
report "$([ -z "$cred_detail" ] && echo 0 || echo 1)" "$cred_label" "$cred_detail"
cred_rc=0
PATH=$cred_bin:$PATH cred_run "$cred_cut" "$TMP/cred-cut-runs" "$TMP/cred-cut.names" run build @fixture ||
  cred_rc=$?
if [ "$cred_rc" = 0 ] && cred_has GH_TOKEN "$TMP/cred-cut.names" &&
  cred_has FOO_API_KEY "$TMP/cred-cut.names"; then
  report 0 "$cred_nc_label"
else
  report 1 "$cred_nc_label" "exit $cred_rc; names: $(tr '\n' ' ' <"$TMP/cred-cut.names" 2>/dev/null)"
fi
# And when dune is not on PATH, select_dune sources tools/opam-env.sh, whose switch environment
# can set a credential the startup scrub never saw: a fake opam answers `opam env` with both
# credentials and a PATH reaching the fixture dune. Through `repeat`, whose dune the launching
# process starts itself (a `run`'s dune is started by a fresh `_resolve`, whose own startup scrub
# also catches it). The PATH holds only that opam and the system directories; a host whose system
# directories hold a dune skips (the opam environment would never be asked for). The negative
# control cuts out select_dune's second scrub alone.
cred_opam_label="credentials: the opam environment select_dune sources is scrubbed too"
cred_opam_nc_label="negative control: without select_dune's scrub, dune sees the switch's credentials"
cred_opam_bin=$TMP/cred-opam-bin
mkdir -p "$cred_opam_bin"
cat >"$cred_opam_bin/opam" <<'EOF'
#!/usr/bin/env bash
[ "${1:-}" = env ] || exit 9
: >>"$CRED_TEST_OPAM_ASKED"
printf "GH_TOKEN='fixture-switch-token'; export GH_TOKEN; FOO_API_KEY='fixture-switch-key'; export FOO_API_KEY; PATH='%s':\"\$PATH\"; export PATH;\n" "$CRED_TEST_DUNE_DIR"
EOF
chmod +x "$cred_opam_bin/opam"
cred_opam_mutant=$(mutant no-select-dune-scrub '/^  scrub_credentials$/ { next } { print }')
cred_opam_run() { # tag script -> names file; 0 when the opam environment was asked for
  local root
  root=$(cred_root "$1" "$2")
  cp "$HERE/opam-env.sh" "$root/tools/opam-env.sh"
  rm -f "$TMP/$1.asked"
  PATH=$cred_opam_bin:/usr/bin:/bin:/usr/sbin:/sbin CRED_TEST_DUNE_DIR=$cred_bin \
    CRED_TEST_OPAM_ASKED=$TMP/$1.asked cred_run "$root" "$TMP/$1-runs" "$TMP/$1.names" repeat 2 build @fixture
  [ -e "$TMP/$1.asked" ]
}
if [ -z "$cred_opam_mutant" ] || cmp -s "$SRC" "$cred_opam_mutant"; then
  harness_abandon "select_dune's scrub could not be cut out of $SRC; the mutant is broken"
  return
elif PATH=/usr/bin:/bin:/usr/sbin:/sbin command -v dune >/dev/null 2>&1; then
  skip "$cred_opam_label" "a dune in the system directories: select_dune never sources the opam environment"
  skip "$cred_opam_nc_label" "a dune in the system directories: select_dune never sources the opam environment"
else
  if ! cred_opam_run cred-opam "$SRC"; then
    report 1 "$cred_opam_label" "the opam environment was not asked for: $(tail -3 "$TMP/cred-opam.names.out")"
  elif cred_has GH_TOKEN "$TMP/cred-opam.names" || cred_has FOO_API_KEY "$TMP/cred-opam.names" ||
    ! cred_has CRED_TEST_PLAIN "$TMP/cred-opam.names"; then
    report 1 "$cred_opam_label" "names: $(tr '\n' ' ' <"$TMP/cred-opam.names" 2>/dev/null)"
  else
    report 0 "$cred_opam_label"
  fi
  if cred_opam_run cred-opam-cut "$cred_opam_mutant" && cred_has GH_TOKEN "$TMP/cred-opam-cut.names" &&
    cred_has FOO_API_KEY "$TMP/cred-opam-cut.names"; then
    report 0 "$cred_opam_nc_label"
  else
    report 1 "$cred_opam_nc_label" "names: $(tr '\n' ' ' <"$TMP/cred-opam-cut.names" 2>/dev/null)"
  fi
fi
# The scrub text itself fails CLOSED, under every shell at hand (dash where installed): an `env`
# that cannot run is a refusal, never an empty environment that passes; a deny-listed name a parent
# passed that is not a shell identifier (`odd-name_TOKEN`) cannot be unset, so it refuses too,
# naming it. With PATH holding nothing but `env`, valid credentials are removed and a plain variable
# stays -- the text needs no other external program, so no parser stage can fail open -- and a
# value spanning lines with a glob character in it neither breaks the walk nor leaks the caller's
# IFS or `set -f`. A newline-named entry refuses through the second listing, and inherited
# functions named env, unset or printf are bypassed.
cred_text_label="credentials: the scrub text refuses when env cannot run, on a deny-listed non-identifier or newline-named entry, and bypasses inherited functions"
cred_text_ok_label="credentials: the scrub text, with only env on PATH, removes valid names and restores IFS and set -f"
cred_text_detail= cred_text_ok_detail=
cred_nl='
'
cred_env_only=$TMP/cred-env-only
mkdir -p "$cred_env_only"
ln -s "$(command -v env)" "$cred_env_only/env"
cred_helper=$HERE/credential-env.sh
for cred_shell in /bin/sh "$(command -v bash)" "$(command -v dash 2>/dev/null)"; do
  [ -n "$cred_shell" ] || continue
  cred_rc=0
  cred_out=$(env -i PATH=/nonexistent GH_TOKEN=fixture-not-a-token "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)"' cred "$cred_helper" 2>&1) || cred_rc=$?
  [ "$cred_rc" != 0 ] || cred_text_detail="$cred_text_detail; $cred_shell passed with env unrunnable"
  cred_rc=0
  # Refused by name -- or, where the shell itself drops such an entry at startup (dash), gone from
  # what its children inherit, which the trailing `env` shows.
  cred_out=$(env -i "PATH=$cred_env_only" 'odd-name_TOKEN=fixture-not-a-token' "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)" || { printf "%s\n" "$credential_env_left"; exit 1; }; env' \
    cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc:$cred_out in
    1:*"odd-name_TOKEN (not a shell identifier)"*) ;;
    0:*odd-name_TOKEN*) cred_text_detail="$cred_text_detail; $cred_shell passed odd-name_TOKEN to its children" ;;
    0:*) ;;
    *) cred_text_detail="$cred_text_detail; $cred_shell on odd-name_TOKEN: exit $cred_rc, said: $cred_out" ;;
  esac
  cred_rc=0
  cred_out=$(env -i "PATH=$cred_env_only" GH_TOKEN=fixture-not-a-token "FOO_API_KEY=fixture
BAR_TOKEN=* not-a-key" CRED_TEST_PLAIN=kept "$cred_shell" -c \
    'set -u; IFS=" "; . "$1"; eval "$(credential_env_scrub_text)" || exit 3
     printf "ifs=[%s] opts=%s\n" "$IFS" "$-"; env' cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc in 0) ;; *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell exited $cred_rc: $cred_out" ;; esac
  case $cred_out in *GH_TOKEN=* | *FOO_API_KEY=*) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell left a credential" ;; esac
  case $cred_out in *CRED_TEST_PLAIN=kept*) ;; *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell lost the plain variable" ;; esac
  cred_first=${cred_out%%"$cred_nl"*}
  case $cred_first in "ifs=[ ] opts="*f*) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell left set -f on" ;; "ifs=[ ] opts="*) ;;
    *) cred_text_ok_detail="$cred_text_ok_detail; $cred_shell did not restore IFS: $cred_first" ;; esac
  # A name holding a newline reads, line by line, as a plain `GH_TOKEN`: the second listing
  # still sees it, so it refuses -- or the shell dropped it at startup (dash) and its children
  # never see it.
  cred_rc=0
  cred_out=$(env -i "PATH=$cred_env_only" "ODD${cred_nl}GH_TOKEN=fixture-not-a-token" "$cred_shell" -c \
    '. "$1"; eval "$(credential_env_scrub_text)" || { printf "%s\n" "$credential_env_left"; exit 1; }; env' \
    cred "$cred_helper" 2>&1) || cred_rc=$?
  case $cred_rc:$cred_out in
    1:*"GH_TOKEN (still in the environment)"*) ;;
    0:*"${cred_nl}GH_TOKEN="* | 0:"GH_TOKEN="*) cred_text_detail="$cred_text_detail; $cred_shell passed a newline-named GH_TOKEN to its children" ;;
    0:*) ;;
    *) cred_text_detail="$cred_text_detail; $cred_shell on a newline-named GH_TOKEN: exit $cred_rc, said: $cred_out" ;;
  esac
done
# Inherited functions named env, unset and printf (bash imports exported ones) are bypassed: the
# scrub still removes the credential, rather than asking a function that answers nothing.
cred_bash=$(command -v bash)
cred_rc=0
cred_out=$(env -i "PATH=$cred_env_only" GH_TOKEN=fixture-not-a-token CRED_HELPER="$cred_helper" \
  CRED_BASH="$cred_bash" "$cred_bash" -c \
  'env() { :; }; unset() { :; }; printf() { :; }; export -f env unset printf
   exec "$CRED_BASH" -c '\''. "$CRED_HELPER"; eval "$(credential_env_scrub_text)" || exit 3; command env'\''' \
  2>&1) || cred_rc=$?
case $cred_rc:$cred_out in
  0:*GH_TOKEN=*) cred_text_detail="$cred_text_detail; inherited env/unset/printf functions let GH_TOKEN through" ;;
  0:*PATH=*) ;;
  *) cred_text_detail="$cred_text_detail; under inherited functions: exit $cred_rc, said: $cred_out" ;;
esac
report "$([ -z "$cred_text_detail" ] && echo 0 || echo 1)" "$cred_text_label" "${cred_text_detail#; }"
report "$([ -z "$cred_text_ok_detail" ] && echo 0 || echo 1)" "$cred_text_ok_label" "${cred_text_ok_detail#; }"
# The Windows wrapper test-run.sh starts dune through, which AGENTS.md also has a Git Bash session
# run by hand, scrubs on its own: a copy beside the helper, the fixture dune on PATH, and a copy
# with the scrub cut out as the negative control.
cred_quiet_label="credentials: tools/dune-quiet.sh, run directly, starts dune without them"
cred_quiet_nc_label="negative control: dune-quiet.sh without its scrub hands dune both credentials"
mkdir -p "$TMP/cred-quiet" "$TMP/cred-quiet-cut"
cp "$HERE/dune-quiet.sh" "$HERE/credential-env.sh" "$TMP/cred-quiet/"
cp "$HERE/credential-env.sh" "$TMP/cred-quiet-cut/"
awk '/^eval "\$\(credential_env_scrub_text\)" \|\| \{$/ { skip = 1 } skip { if ($0 == "}") skip = 0; next } { print }' \
  "$HERE/dune-quiet.sh" >"$TMP/cred-quiet-cut/dune-quiet.sh"
cred_quiet_run() { # dir
  rm -f "$1.names"
  GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key CRED_TEST_PLAIN=kept \
    CRED_TEST_NAMES=$1.names PATH=$cred_bin:$PATH bash "$1/dune-quiet.sh" build @fixture >"$1.out" 2>&1
}
if cmp -s "$HERE/dune-quiet.sh" "$TMP/cred-quiet-cut/dune-quiet.sh"; then
  harness_abandon "the scrub could not be cut out of tools/dune-quiet.sh; the mutant is broken"
  return
fi
cred_rc=0
cred_quiet_run "$TMP/cred-quiet" || cred_rc=$?
if [ "$cred_rc" = 0 ] && [ -s "$TMP/cred-quiet.names" ] && ! cred_has GH_TOKEN "$TMP/cred-quiet.names" &&
  ! cred_has FOO_API_KEY "$TMP/cred-quiet.names" && cred_has CRED_TEST_PLAIN "$TMP/cred-quiet.names"; then
  report 0 "$cred_quiet_label"
else
  report 1 "$cred_quiet_label" "exit $cred_rc; names: $(tr '\n' ' ' <"$TMP/cred-quiet.names" 2>/dev/null)"
fi
cred_rc=0
cred_quiet_run "$TMP/cred-quiet-cut" || cred_rc=$?
if [ "$cred_rc" = 0 ] && cred_has GH_TOKEN "$TMP/cred-quiet-cut.names" &&
  cred_has FOO_API_KEY "$TMP/cred-quiet-cut.names"; then
  report 0 "$cred_quiet_nc_label"
else
  report 1 "$cred_quiet_nc_label" "exit $cred_rc; see $TMP/cred-quiet-cut.out"
fi
# The trace itself, which is what the issue found tokens in. A host with no
# dune skips it (the toolchain-free CI job); a dune that writes no trace there
# skips it too, since there is then nothing to leak into.
cred_real=
! harness_real_dune || cred_real=$harness_dune
if [ -z "$cred_real" ]; then
  skip "$cred_trace_label" "$harness_dune_absent"
  skip "$cred_trace_nc_label" "$harness_dune_absent"
else
  cred_project() { # root: a one-rule dune project whose build spawns a process
    printf '(lang dune 3.20)\n' >"$1/dune-project"
    printf '(rule\n (with-stdout-to spawned.txt (run sh -c "echo spawned")))\n' >"$1/dune"
    rm -rf "$1/_build"
  }
  cred_project "$cred_ship"
  cred_project "$cred_cut"
  # The run hands dune a trace file of its own and deletes it once its
  # slowest actions are read (leg 80), so the trace is observed through the
  # reader seam: a stand-in reader that copies the run's trace to the root's
  # `_build/trace.csexp`, where the default trace would have been.
  cred_copier=$TMP/cred-trace-copier.sh
  printf '#!/usr/bin/env bash\ncp "$3" "$CRED_TRACE_COPY"\n' >"$cred_copier"
  cred_trace() { # root -> 0 when its trace exists, the three names' counts in cred_gh/foo/plain
    local trace=$1/_build/trace.csexp
    [ -f "$trace" ] || return 1
    cred_gh=$(grep -a -c GH_TOKEN "$trace")
    cred_foo=$(grep -a -c FOO_API_KEY "$trace")
    cred_plain=$(grep -a -c CRED_TEST_PLAIN "$trace")
    cred_counts="GH_TOKEN: $cred_gh, FOO_API_KEY: $cred_foo, CRED_TEST_PLAIN: $cred_plain"
  }
  cred_real_run() { # root tag
    (cd "$1" && PATH=$(dirname "$cred_real"):$PATH OCANNL_TOOL_SLOWEST_READER=$cred_copier \
      CRED_TRACE_COPY=$1/_build/trace.csexp cred_run "$1" "$TMP/$2-runs" "$TMP/$2.names" \
      run build --cache=disabled ./spawned.txt)
  }
  cred_rc=0
  cred_real_run "$cred_ship" cred-real || cred_rc=$?
  if [ "$cred_rc" != 0 ]; then
    report 1 "$cred_trace_label" "the run exited $cred_rc: $(tail -5 "$TMP/cred-real.names.out")"
  elif ! cred_trace "$cred_ship"; then
    skip "$cred_trace_label" "$("$cred_real" --version) wrote no trace for the run"
  elif [ "$cred_gh" = 0 ] && [ "$cred_foo" = 0 ] && [ "$cred_plain" != 0 ]; then
    report 0 "$cred_trace_label"
  else
    report 1 "$cred_trace_label" "$cred_counts"
  fi
  cred_rc=0
  cred_real_run "$cred_cut" cred-real-cut || cred_rc=$?
  if [ "$cred_rc" != 0 ]; then
    report 1 "$cred_trace_nc_label" "the run exited $cred_rc: $(tail -5 "$TMP/cred-real-cut.names.out")"
  elif ! cred_trace "$cred_cut"; then
    skip "$cred_trace_nc_label" "$("$cred_real" --version) wrote no trace for the run"
  elif [ "$cred_gh" != 0 ] && [ "$cred_foo" != 0 ]; then
    report 0 "$cred_trace_nc_label"
  else
    report 1 "$cred_trace_nc_label" "$cred_counts"
  fi
fi
}
harness_leg_run '79: credentials never reach dune #credentials' leg_79

# ---------------------------------------------------------------------------
# Leg 90: the runner's own variables never reach dune (gh-ocannl-1448)
# ---------------------------------------------------------------------------
# The supervisor and the run's phases are handed their directories through
# OCANNL_TOOL_TESTRUN_BG, _RD and _OWN; dune, and every test it starts, used to
# inherit all three (only _RUNNER was taken out). A fixture dune records the
# names of its environment on every path a batch's dune is started from: a
# plain run, a GPU run whose device probe execs dune inside the slot, and the
# iterations of a repeat with that probe. A planted plain variable must still
# pass, or the observer proves nothing.
leg_90() {
env_label="the runner's own variables (OCANNL_TOOL_TESTRUN_*) reach no dune: a run, a probed GPU run under a slot, a probed repeat"
env_detail=
env_names=$TMP/env-names
env_check() { # what: the names recorded since the last call, judged
  if [ ! -s "$env_names" ]; then
    env_detail="$1: the fixture dune recorded no environment"
  elif grep -q '^OCANNL_TOOL_TESTRUN_' "$env_names"; then
    env_detail="$1: dune saw $(grep '^OCANNL_TOOL_TESTRUN_' "$env_names" | sort -u | tr '\n' ' ')"
  elif ! grep -qx ENV_TEST_PLAIN "$env_names"; then
    env_detail="$1: dune did not see ENV_TEST_PLAIN"
  fi
  : >"$env_names"
}
: >"$env_names"
ENV_TEST_PLAIN=kept REPEAT_TEST_ENV_NAMES=$env_names argv_mode=env_names argv_probe env-run run build @cheap
[ "$argv_rc" = 0 ] || env_detail="run: exit $argv_rc: $argv_out"
[ -n "$env_detail" ] || env_check run
if [ -z "$env_detail" ]; then
  ENV_TEST_PLAIN=kept REPEAT_TEST_ENV_NAMES=$env_names argv_mode=env_names \
    probe_run env-probe hip cc "" "" run build @cheap
  { [ "$argv_rc" = 0 ] && [ "$probe_calls" = "$(cd "$repeat_root" && pwd -P)/test/config --ocannl_backend=hip" ]; } ||
    env_detail="probed run: exit $argv_rc; probes: ${probe_calls:-<none>}; digest: $argv_out"
  [ -n "$env_detail" ] || env_check "probed run"
fi
if [ -z "$env_detail" ]; then
  ENV_TEST_PLAIN=kept REPEAT_TEST_ENV_NAMES=$env_names argv_mode=env_names \
    probe_run env-repeat hip cc "" "" repeat 2 build @cheap
  [ "$argv_rc" = 0 ] || env_detail="probed repeat: exit $argv_rc: $argv_out"
  [ -n "$env_detail" ] || [ "$(grep -c '' <<<"$probe_calls")" = 2 ] ||
    env_detail="probed repeat: the device was not probed before each iteration: ${probe_calls:-<none>}"
  [ -n "$env_detail" ] || env_check "probed repeat"
fi
if [ -z "$env_detail" ]; then report 0 "$env_label"; else report 1 "$env_label" "$env_detail"; fi
}
harness_leg_run '90: the runner'\''s own variables reach no dune #credentials' leg_90

# ---------------------------------------------------------------------------
# Leg 82: bisect (gh-ocannl-1225)
# ---------------------------------------------------------------------------
# A fixture repository whose commits each carry a STATE the fixture dune
# reads from the tree it runs in: `pass`, `fail` (the test red), `nobuild`
# (`@check` red), `refuse` (the fake slot refuses), `hang` (the test outlives
# the cap) and `hold` (the test waits to be cancelled). The fixture dune
# records, per call, the STATE, the OCANNL_BACKEND it saw and whether the
# runner's own variables reached it. The invoking checkout sits on `master`
# at the base commit; the range is the `range` branch, whose LAST commit
# replaces tools/test-run.sh with one that exits 3 -- so a bisect that ran a
# tested commit's own runner, rather than its frozen copy, aborts at the bad
# end instead of finding anything.
leg_82() {
if ! command -v git >/dev/null 2>&1; then
  skip 'bisect: exit mapping, backend, isolation (leg 82)' 'no git on PATH'
else
  bi_git() { GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git "$@"; }
  bi_bin=$TMP/bisect-bin
  mkdir -p "$bi_bin"
  cat >"$bi_bin/dune" <<'EOF'
#!/usr/bin/env bash
# The alias check's listing (gh-ocannl-1448) is no call of the bisect's: an
# empty answer, which leaves every alias to dune.
[ "${1:-}" != show ] || exit 0
st=$(cat STATE 2>/dev/null)
printf '%s %s tree=%s runner=%s :: %s\n' "$st" "${OCANNL_BACKEND:-<unset>}" \
  "${OCANNL_TOOL_TESTRUN_TREE-<unset>}" "${OCANNL_TOOL_TESTRUN_RUNNER-<unset>}" "$*" >>"$BISECT_CALLS"
case " $* " in
  *" @check "*) [ "$st" != nobuild ] || { echo 'File "x.ml", line 1: Error: broken'; exit 1; }; exit 0 ;;
esac
case $st in
  fail) echo 'Error: the test failed'; exit 1 ;;
  # A build step that fails for the targets alone, @check green (a link): the
  # run's own trace records it, so test-run reads BUILD FAILED (gh-ocannl-1547).
  linkfail)
    for a do case $a in --trace-file=*) cp "$BISECT_LINKFAIL_TRACE" "${a#--trace-file=}" ;; esac; done
    echo 'Error: the link failed'; exit 1 ;;
  hang) exec sleep 60 ;;
  escape*)
    # A descendant that leaves dune's group, keeping the run's lock (fd 9),
    # and reports the tree's STATE as it found it two seconds later.
    perl -MPOSIX -e 'POSIX::setsid(); sleep 2; open my $f, "<", "STATE"; my $now = <$f>;
      chomp $now; open my $o, ">>", $ENV{BISECT_ESCAPE}; print $o "$ARGV[0] $now\n"' "$st" \
      </dev/null >/dev/null 2>&1 &
    # Out of the group before dune exits: the supervisor reaps what is left in it.
    sleep 0.5
    ;;
  hold)
    printf '%s\n' "$$" >"$BISECT_HOLD.pid"
    exec perl -e 'sleep 60' ;;
esac
echo ok
EOF
  # The slot: refused where the tree's STATE says so, held otherwise.
  cat >"$bi_bin/fleet-worker.sh" <<'EOF'
#!/usr/bin/env bash
if [ "$1 $2 $3" = "execution slot --probe" ]; then echo "EXECUTION SLOT PROBE fakebox 4 2"; exit 0; fi
while [ "$1" != -- ]; do shift; done
shift
if [ "$(cat STATE 2>/dev/null)" = refuse ]; then
  echo "EXECUTION SLOT REFUSED fakebox: all 2 GPU tokens (slots 1-2 of 4) busy after 0s"
  exit 1
fi
exec "$@"
EOF
  chmod +x "$bi_bin/dune" "$bi_bin/fleet-worker.sh"
  bi_n=0
  bi_repo() { # <states of the range...>; a fresh fixture, sets bi_root, bi_base, bi_shas
    local s
    bi_n=$((bi_n + 1))
    bi_root=$TMP/bisect-repo-$bi_n
    mkdir -p "$bi_root/tools" "$bi_root/scripts" && cp "$SRC" "$bi_root/tools/test-run.sh" || return 1
    stage_sourced "$bi_root" || return 1
    # The trace reader, frozen with the runner: (b2)'s BUILD FAILED is read from a run's trace.
    cp "$HERE/action-durations.sh" "$bi_root/tools/action-durations.sh" || return 1
    printf 'pass\n' >"$bi_root/STATE"
    { bi_git init -q "$bi_root" && bi_git -C "$bi_root" config user.name harness &&
      bi_git -C "$bi_root" config user.email harness@invalid &&
      bi_git -C "$bi_root" config commit.gpgsign false &&
      bi_git -C "$bi_root" add -A && bi_git -C "$bi_root" commit -q --no-verify -m base &&
      bi_git -C "$bi_root" branch -M master && bi_git -C "$bi_root" checkout -q -b range; } >/dev/null 2>&1 ||
      { report 1 'bisect: fixture repository'; return 1; }
    bi_base=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_shas=()
    for s in "$@"; do
      printf '%s\n' "$s" >"$bi_root/STATE"
      bi_git -C "$bi_root" commit -q --no-verify --allow-empty -am "$s" >/dev/null 2>&1 || return 1
      bi_shas+=("$(bi_git -C "$bi_root" rev-parse HEAD)")
    done
    # The range's last commit carries a runner that cannot run anything.
    printf '#!/usr/bin/env bash\nexit 3\n' >"$bi_root/tools/test-run.sh"
    bi_git -C "$bi_root" commit -q --no-verify --amend -a --no-edit >/dev/null 2>&1 || return 1
    bi_shas[${#bi_shas[@]}-1]=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q master >/dev/null 2>&1
  }
  bi_runs=$TMP/bisect-runs
  bi_run() { # [VAR=value...] -- bisect args...; sets bi_rc, bi_out, bi_err, bi_dir
    local envs=()
    while [ "$1" != -- ]; do envs+=("$1"); shift; done
    shift
    : >"$TMP/bisect.calls"
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_HOLD=$TMP/bisect-hold BISECT_ESCAPE=$TMP/bisect.escape \
      PATH=$bi_bin:$PATH ${envs[@]+"${envs[@]}"} "$bi_root/tools/test-run.sh" bisect "$@" \
      >"$TMP/bisect.out" 2>"$TMP/bisect.err"
    bi_rc=$?
    bi_out=$(cat "$TMP/bisect.out") bi_err=$(cat "$TMP/bisect.err")
    bi_dir=$(sed -n 's/^bisect: \(\/.*\)$/\1/p' "$TMP/bisect.out" | head -n 1)
  }
  bi_clean() { # the invoking checkout as the fixture left it, and no tree behind
    [ "$(bi_git -C "$bi_root" rev-parse HEAD)" = "$bi_base" ] &&
      [ -z "$(bi_git -C "$bi_root" status --porcelain)" ] &&
      [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" = 1 ] &&
      [ -n "$bi_dir" ] && [ ! -e "$bi_dir/tree" ]
  }
  bi_verdicts() { cut -f3 "$bi_dir/steps.tsv" 2>/dev/null | tr '\n' ' '; }

  # (a) the mapping: a red build is a skip, never bad; the first bad commit is
  # found; every dune call saw the pinned backend, none the runner's own
  # variables; the bad end ran under the frozen runner; the records stay.
  l_found="bisect: a red build is skipped, never bad, and the first bad commit is found under the frozen runner"
  bi_detail=
  bi_repo pass nobuild pass fail fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=hip OCANNL_TOOL_FLEET_WORKER=none -- --cap 60 "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ]; then bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif ! grep -q "^result: FIRST BAD COMMIT ${bi_shas[3]} fail\$" <<<"$bi_out"; then
    bi_detail="result: $(grep '^result:' <<<"$bi_out") (want ${bi_shas[3]})"
  elif ! grep -q "^bad-end	${bi_shas[4]}	bad	" "$bi_dir/steps.tsv" ||
       ! grep -q "^good-end	$bi_base	good	" "$bi_dir/steps.tsv"; then
    bi_detail="the ends were not run first: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q "	${bi_shas[1]}	skip	" "$bi_dir/steps.tsv"; then
    bi_detail="the commit whose build is red was not a skip: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q '^nobuild hip ' "$TMP/bisect.calls" ||
       grep -v '^[a-z]* hip tree=<unset> runner=<unset> :: ' "$TMP/bisect.calls" | grep -q .; then
    bi_detail="a dune call missed the pinned backend, or saw the runner's variables: $(cat "$TMP/bisect.calls")"
  elif grep -q '^nobuild .* @t$' "$TMP/bisect.calls"; then
    bi_detail="the test ran on a commit whose build is red: $(cat "$TMP/bisect.calls")"
  elif [ ! -s "$bi_dir/bisect-log" ] || [ ! -s "$bi_dir/steps/bad-end-test.log" ] ||
       [ ! -s "$bi_dir/steps/bad-end-test.out" ]; then
    bi_detail="the records are missing: $(ls "$bi_dir" "$bi_dir/steps" 2>&1 | tr '\n' ' ')"
  elif ! bi_clean; then
    bi_detail="the invoking checkout changed, or the tree was left: $(bi_git -C "$bi_root" worktree list)"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_found"; else report 1 "$l_found" "$bi_detail"; fi

  # (a2) the targets from a --targets-file (gh-ocannl-1797), with none in the
  # arguments: recorded as the bisect's targets and handed to every test run.
  l_tfile="bisect: --targets-file supplies the targets, recorded and run at every step"
  bi_detail=
  printf '# the bisect target\n@t\n' >"$TMP/bisect-targets.txt"
  bi_repo pass fail || bi_detail="fixture"
  [ -n "$bi_detail" ] ||
    bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- --targets-file "$TMP/bisect-targets.txt" "$bi_base" range
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ]; then bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif ! grep -q "^result: FIRST BAD COMMIT ${bi_shas[1]} fail\$" <<<"$bi_out"; then
    bi_detail="result: $(grep '^result:' <<<"$bi_out") (want ${bi_shas[1]})"
  elif [ "$(cat "$bi_dir/targets" 2>/dev/null)" != @t ]; then
    bi_detail="recorded targets: $(cat "$bi_dir/targets" 2>&1) (want @t)"
  elif ! grep -q ' @t$' "$TMP/bisect.calls"; then
    bi_detail="no test run was handed the listed target: $(cat "$TMP/bisect.calls")"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_tfile"; else report 1 "$l_tfile" "$bi_detail"; fi

  # (b) a range whose only untested commits did not build ends without a
  # first bad commit (exit 1, the candidates listed) -- a skip read as bad
  # would have named the red build.
  l_skipped="bisect: a range left with only skipped commits names the candidates, exit 1"
  bi_detail=
  bi_repo nobuild fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 1 ] || grep -q 'FIRST BAD COMMIT' <<<"$bi_out" ||
       ! grep -q "only skipped commits are left" <<<"$bi_out" ||
       ! grep -q "^  ${bi_shas[0]} nobuild" <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_skipped"; else report 1 "$l_skipped" "$bi_detail"; fi

  # (b2) a test run that is BUILD FAILED -- @check green, a build step of the
  # targets' own failed -- is a skip too, never bad (gh-ocannl-1547): the
  # range's only untested commit is listed as a candidate, exit 1.
  l_linkfail="bisect: a test run that is BUILD FAILED is a skip, never bad"
  if ! command -v python3 >/dev/null 2>&1; then
    skip "$l_linkfail" "no python3 on PATH (tools/action-durations.sh needs it)"
  else
    bi_detail=
    bi_repo linkfail fail || bi_detail="fixture"
    if [ -z "$bi_detail" ]; then
      synthetic_trace "$TMP/bisect-linkfail.csexp" "$TMP/bisect-linkfail-build" 'ocamlopt.opt|2|t/x.exe'
      bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none BISECT_LINKFAIL_TRACE=$TMP/bisect-linkfail.csexp \
        -- "$bi_base" range @t
    fi
    if [ -n "$bi_detail" ]; then :
    elif [ "$bi_rc" != 1 ] || grep -q 'FIRST BAD COMMIT' <<<"$bi_out" ||
         ! grep -q "only skipped commits are left" <<<"$bi_out" ||
         ! grep -q "^  ${bi_shas[0]} linkfail" <<<"$bi_out"; then
      bi_detail="exit $bi_rc: $bi_out $bi_err"
    elif ! grep -q "	${bi_shas[0]}	skip	" "$bi_dir/steps.tsv" ||
         ! grep -q '^verdict: BUILD FAILED' "$bi_dir/steps/"*-test.out; then
      bi_detail="the BUILD FAILED test run was not a skip: $(cat "$bi_dir/steps.tsv")"
    elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
    fi
    if [ -z "$bi_detail" ]; then report 0 "$l_linkfail"; else report 1 "$l_linkfail" "$bi_detail"; fi
  fi

  # (c) a refused slot mid-bisect ABORTS it: git bisect stops (the step's 128),
  # the bisect exits 75, nothing is called bad, and the tree goes.
  l_slot="bisect: SLOT REFUSED in a step aborts the bisect with exit 75, judging nothing"
  bi_detail=
  bi_repo refuse fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=$bi_bin/fleet-worker.sh -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 75 ] || ! grep -q '^result: ABORTED: the build run of .*SLOT REFUSED' <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ "$(bi_verdicts)" != "bad good ABORT " ]; then
    bi_detail="steps: $(cat "$bi_dir/steps.tsv")"
  elif ! grep -q 'exit code 128' "$bi_dir/git-bisect.out" 2>/dev/null &&
       ! grep -q 'exit code 128' <<<"$bi_err"; then
    bi_detail="git bisect run was not stopped by the step's 128: $(cat "$bi_dir/git-bisect.out") $bi_err"
  elif grep -q '^refuse ' "$TMP/bisect.calls"; then
    bi_detail="dune ran under a refused slot: $(cat "$TMP/bisect.calls")"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_slot"; else report 1 "$l_slot" "$bi_detail"; fi

  # (d) a test that outlives the cap is not a verdict either: exit 142, abort.
  l_cap="bisect: a cap's TIMEOUT in a step aborts the bisect with exit 142"
  bi_detail=
  bi_repo hang fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- --cap 3 "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 142 ] || ! grep -q '^result: ABORTED: the test run of .*TIMEOUT' <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "bad good ABORT " ]; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif ! grep -q 'exit code 128' "$bi_dir/git-bisect.out" 2>/dev/null &&
       ! grep -q 'exit code 128' <<<"$bi_err"; then
    bi_detail="git bisect run was not stopped by the step's 128: $(cat "$bi_dir/git-bisect.out") $bi_err"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_cap"; else report 1 "$l_cap" "$bi_detail"; fi

  # (e) ends that do not hold here bisect nothing: exit 1, said why.
  l_ends="bisect: a GOOD end that fails here, or a BAD one that passes, bisects nothing (exit 1)"
  bi_detail=
  bi_repo pass pass || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 1 ] || ! grep -q "^result: the BAD end ${bi_shas[1]} passes here" <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "good " ] || [ -s "$bi_dir/bisect-log" ]; then
    bi_detail="BAD passing: exit $bi_rc: $bi_out"
  fi
  if [ -z "$bi_detail" ]; then
    bi_repo fail fail || bi_detail="fixture"
    bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "${bi_shas[0]}" range @t
    if [ "$bi_rc" != 1 ] || ! grep -q "^result: the GOOD end ${bi_shas[0]} fails here" <<<"$bi_out" ||
       [ "$(bi_verdicts)" != "bad bad " ]; then
      bi_detail="GOOD failing: exit $bi_rc: $bi_out"
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_ends"; else report 1 "$l_ends" "$bi_detail"; fi

  # (f) refusals before any step: no backend, an unrelated GOOD, a misplaced
  # option -- exit 2, no dune, no bisect directory; and the tree variable is
  # refused by a subcommand that does not take it.
  l_usage="bisect: no backend, a GOOD that is not BAD's ancestor or a misplaced option is refused (exit 2) before any step"
  bi_detail=
  bi_repo fail || bi_detail="fixture"
  bi_before=$(ls "$bi_runs" 2>/dev/null | grep -c '^bisect-')
  for probe in "nobackend:OCANNL_BACKEND=:$bi_base range @t:passes OCANNL_BACKEND" \
                "badbackend:OCANNL_BACKEND=vulkan:$bi_base range @t:names no backend" \
                "unrelated:OCANNL_BACKEND=cc:range $bi_base @t:not an ancestor" \
                "misplaced:OCANNL_BACKEND=cc:$bi_base range @t --cap 9:belongs before"; do
    [ -z "$bi_detail" ] || break
    IFS=: read -r tag benv args want <<<"$probe"
    # shellcheck disable=SC2086  # the arguments are word-split on purpose
    bi_run "$benv" OCANNL_TOOL_FLEET_WORKER=none -- $args
    if [ "$bi_rc" != 2 ] || ! grep -q "$want" <<<"$bi_err" || [ -s "$TMP/bisect.calls" ]; then
      bi_detail="$tag: exit $bi_rc; stderr: $bi_err; calls: $(cat "$TMP/bisect.calls")"
    fi
  done
  if [ -z "$bi_detail" ] && [ "$(ls "$bi_runs" 2>/dev/null | grep -c '^bisect-')" != "$bi_before" ]; then
    bi_detail="a refused bisect left a bisect directory"
  fi
  if [ -z "$bi_detail" ]; then
    bi_tree_out=$(OCANNL_TOOL_TESTRUN_TREE=$TMP OCANNL_TOOL_TEST_RUNS=$bi_runs "$bi_root/tools/test-run.sh" list 2>&1)
    bi_tree_rc=$?
    [ "$bi_tree_rc" = 2 ] && grep -q 'does not take it' <<<"$bi_tree_out" ||
      bi_detail="list under OCANNL_TOOL_TESTRUN_TREE: exit $bi_tree_rc: $bi_tree_out"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_usage"; else report 1 "$l_usage" "$bi_detail"; fi

  # (g) first-parent by default: the commits of a merged branch are never
  # stepped on, though one of them is red.
  l_fp="bisect: first-parent by default never steps into a merged branch, and refuses a GOOD only a second parent reaches"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  if [ -z "$bi_detail" ]; then
    # range: base - pass - M(side: fail, pass) - fail(+broken runner)
    bi_git -C "$bi_root" checkout -q -b side master >/dev/null 2>&1
    printf 'fail\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am side-fail >/dev/null 2>&1
    bi_side1=$(bi_git -C "$bi_root" rev-parse HEAD)
    printf 'pass\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am side-pass >/dev/null 2>&1
    bi_side2=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q -b fp "${bi_shas[0]}" >/dev/null 2>&1
    bi_git -C "$bi_root" merge -q --no-ff --no-edit side >/dev/null 2>&1
    printf 'pass\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify --allow-empty -am merged >/dev/null 2>&1
    bi_merge=$(bi_git -C "$bi_root" rev-parse HEAD)
    printf 'fail\n' >"$bi_root/STATE"; bi_git -C "$bi_root" commit -q --no-verify -am fp-fail >/dev/null 2>&1
    bi_fpbad=$(bi_git -C "$bi_root" rev-parse HEAD)
    bi_git -C "$bi_root" checkout -q master >/dev/null 2>&1
    bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" fp @t
    if [ "$bi_rc" != 0 ] || ! grep -q "^result: FIRST BAD COMMIT $bi_fpbad " <<<"$bi_out"; then
      bi_detail="exit $bi_rc: $bi_out $bi_err"
    elif grep -q "$bi_side1\|$bi_side2" "$bi_dir/steps.tsv"; then
      bi_detail="a merged branch's commit was stepped on: $(cat "$bi_dir/steps.tsv")"
    elif [ -z "$bi_merge" ]; then
      bi_detail="fixture: no merge"
    fi
    # A GOOD merged in through a second parent bounds nothing on the
    # first-parent chain: refused, unless the walk takes every parent.
    if [ -z "$bi_detail" ]; then
      bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_side2" fp @t
      if [ "$bi_rc" != 2 ] || ! grep -q "not on BAD's first-parent chain" <<<"$bi_err" ||
         [ -s "$TMP/bisect.calls" ]; then
        bi_detail="a second-parent GOOD: exit $bi_rc: $bi_err"
      else
        bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- --no-first-parent "$bi_side2" fp @t
        [ "$bi_rc" = 0 ] && grep -q "^result: FIRST BAD COMMIT $bi_fpbad " <<<"$bi_out" ||
          bi_detail="a second-parent GOOD under --no-first-parent: exit $bi_rc: $bi_out $bi_err"
      fi
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_fp"; else report 1 "$l_fp" "$bi_detail"; fi

  # (i) a descendant a step's run leaves behind, holding the tree's lock,
  # finishes before git checks out the next commit: it reads its own commit.
  l_settle="bisect: a step's verdict waits for a descendant still holding the tree, so it never sees the next commit"
  bi_detail=
  rm -f "$TMP/bisect.escape"
  bi_repo escape1 escape2 fail fail || bi_detail="fixture"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ] || ! grep -q "^result: FIRST BAD COMMIT ${bi_shas[2]} fail\$" <<<"$bi_out"; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ ! -s "$TMP/bisect.escape" ] || awk '$1 != $2 { bad = 1 } END { exit !bad }' "$TMP/bisect.escape"; then
    bi_detail="a descendant saw another commit's tree (its STATE, then what it read): $(cat "$TMP/bisect.escape" 2>&1)"
  elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_settle"; else report 1 "$l_settle" "$bi_detail"; fi

  # (j) a relative runs directory is resolved once, by the bisect: every
  # step's runs land beside the bisect's records and outlive the tree.
  l_rel="bisect: a relative OCANNL_TOOL_TEST_RUNS keeps every step's runs under the one directory it names"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  rm -rf "${TMP:?}/bisect-relruns"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none OCANNL_TOOL_TEST_RUNS=../bisect-relruns \
    -- "$bi_base" range @t
  bi_relruns=$(cd -P "$TMP/bisect-relruns" 2>/dev/null && pwd -P)
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 0 ] || [ -z "$bi_relruns" ] || [ "$(dirname "$bi_dir")" != "$bi_relruns" ]; then
    bi_detail="exit $bi_rc; bisect directory ${bi_dir:-?} (want one under $TMP/bisect-relruns): $bi_out $bi_err"
  else
    while IFS=$'\t' read -r _ _ _ b t; do
      for r in "$b" "$t"; do
        [ "$r" = - ] && continue
        [ "$(dirname "$r")" = "$bi_relruns" ] && [ -s "$r/exit" ] ||
          bi_detail="a step's run is not under $bi_relruns, or did not outlive the tree: $r"
      done
    done <"$bi_dir/steps.tsv"
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_rel"; else report 1 "$l_rel" "$bi_detail"; fi

  # (k) a setup refusal leaves nothing behind: a dune-workspace above the
  # runs directory would root dune above the tree, and is refused before
  # any worktree is registered or bisect directory made.
  l_setup="bisect: a dune-workspace above the runs directory is refused (exit 2) leaving no worktree or bisect directory"
  bi_detail=
  bi_repo pass fail || bi_detail="fixture"
  mkdir -p "$TMP/bisect-wsanc/runs" && printf '(lang dune 3.18)\n' >"$TMP/bisect-wsanc/dune-workspace"
  [ -n "$bi_detail" ] || bi_run OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
    OCANNL_TOOL_TEST_RUNS=$TMP/bisect-wsanc/runs -- "$bi_base" range @t
  if [ -n "$bi_detail" ]; then :
  elif [ "$bi_rc" != 2 ] || ! grep -q 'would root dune above' <<<"$bi_err" || [ -s "$TMP/bisect.calls" ]; then
    bi_detail="exit $bi_rc: $bi_out $bi_err"
  elif [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" != 1 ] ||
       ls "$TMP/bisect-wsanc/runs" | grep -q '^bisect-'; then
    bi_detail="the refusal left a worktree or bisect directory: $(bi_git -C "$bi_root" worktree list; ls "$TMP/bisect-wsanc/runs")"
  fi
  rm -f "$TMP/bisect-wsanc/dune-workspace"
  if [ -z "$bi_detail" ]; then report 0 "$l_setup"; else report 1 "$l_setup" "$bi_detail"; fi

  # (l) a TERM that lands before a step has a process group to signal: during
  # setup (`git worktree add` slowed by a git wrapper), the bisect unwinds
  # leaving nothing; during an end's checkout, no step is started at all.
  # Either way the exit is 143.
  l_early="bisect: a TERM during setup or before a step starts unwinds with 143, starting nothing"
  bi_detail=
  bi_gitbin=$TMP/bisect-gitbin
  mkdir -p "$bi_gitbin"
  bi_realgit=$(command -v git)
  cat >"$bi_gitbin/git" <<'EOF'
#!/usr/bin/env bash
case " $* " in
  *" worktree add "*) [ "$BISECT_GIT_SLOW" != add ] || { : >"$BISECT_GIT_READY"; sleep 3; } ;;
  *" checkout --quiet --detach "*) [ "$BISECT_GIT_SLOW" != checkout ] || { : >"$BISECT_GIT_READY"; sleep 3; } ;;
esac
exec "$BISECT_REAL_GIT" "$@"
EOF
  chmod +x "$bi_gitbin/git"
  for slow in add checkout; do
    [ -z "$bi_detail" ] || break
    bi_repo pass fail || { bi_detail="fixture"; break; }
    rm -f "$TMP/bisect-git.ready"
    : >"$TMP/bisect.calls"
    bi_early_runs=$TMP/bisect-early-$slow
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_early_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_GIT_SLOW=$slow BISECT_GIT_READY=$TMP/bisect-git.ready \
      BISECT_REAL_GIT=$bi_realgit PATH=$bi_gitbin:$bi_bin:$PATH OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
      "$bi_root/tools/test-run.sh" bisect "$bi_base" range @t >"$TMP/bisect.out" 2>"$TMP/bisect.err" &
    bi_pid=$!
    for _ in $(seq 1 300); do [ -e "$TMP/bisect-git.ready" ] && break; sleep 0.1; done
    kill -TERM "$bi_pid" 2>/dev/null
    for _ in $(seq 1 300); do kill -0 "$bi_pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$bi_pid" 2>/dev/null; then
      kill -KILL "$bi_pid" 2>/dev/null
      bi_detail="$slow: the bisect outlived its TERM by 30s"
    fi
    wait "$bi_pid" 2>/dev/null
    bi_rc=$?
    [ -z "$bi_detail" ] || break
    if [ ! -e "$TMP/bisect-git.ready" ]; then
      bi_detail="$slow: the slowed git call never came: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    elif [ "$bi_rc" != 143 ]; then
      bi_detail="$slow: exit $bi_rc: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    elif [ -s "$TMP/bisect.calls" ]; then
      bi_detail="$slow: a step started after the TERM: $(cat "$TMP/bisect.calls")"
    elif [ "$(bi_git -C "$bi_root" worktree list | grep -c '')" != 1 ]; then
      bi_detail="$slow: a worktree was left registered: $(bi_git -C "$bi_root" worktree list)"
    elif [ "$slow" = add ] && ls "$bi_early_runs" 2>/dev/null | grep -q '^bisect-'; then
      bi_detail="add: the cancelled setup left its bisect directory: $(ls "$bi_early_runs")"
    elif [ "$slow" = checkout ] && ! grep -q '^result: CANCELLED (TERM)' "$TMP/bisect.out"; then
      bi_detail="checkout: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
    fi
  done
  if [ -z "$bi_detail" ]; then report 0 "$l_early"; else report 1 "$l_early" "$bi_detail"; fi

  # (h) a TERM to the bisect stops the step's run (its dune reaped), exits
  # 143 and removes the tree.
  l_cancel="bisect: TERM stops the running step, reaps its dune, exits 143 and removes the tree"
  bi_detail=
  bi_repo hold fail || bi_detail="fixture"
  rm -f "$TMP/bisect-hold.pid"
  if [ -z "$bi_detail" ]; then
    : >"$TMP/bisect.calls"
    env GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null OCANNL_TOOL_TEST_RUNS=$bi_runs \
      BISECT_CALLS=$TMP/bisect.calls BISECT_HOLD=$TMP/bisect-hold PATH=$bi_bin:$PATH \
      OCANNL_BACKEND=cc OCANNL_TOOL_FLEET_WORKER=none \
      "$bi_root/tools/test-run.sh" bisect "$bi_base" range @t >"$TMP/bisect.out" 2>"$TMP/bisect.err" &
    bi_pid=$!
    for _ in $(seq 1 300); do [ -s "$TMP/bisect-hold.pid" ] && break; sleep 0.1; done
    bi_hold=$(cat "$TMP/bisect-hold.pid" 2>/dev/null)
    if [ -z "$bi_hold" ]; then
      bi_detail="the held step never started: $(cat "$TMP/bisect.out" "$TMP/bisect.err")"
      kill -TERM "$bi_pid" 2>/dev/null
    else
      kill -TERM "$bi_pid"
    fi
    for _ in $(seq 1 300); do kill -0 "$bi_pid" 2>/dev/null || break; sleep 0.1; done
    if kill -0 "$bi_pid" 2>/dev/null; then
      bi_detail="the bisect outlived its TERM by 30s"
      kill -KILL "$bi_pid" 2>/dev/null
    fi
    wait "$bi_pid" 2>/dev/null
    bi_rc=$?
    bi_out=$(cat "$TMP/bisect.out")
    bi_dir=$(sed -n 's/^bisect: \(\/.*\)$/\1/p' "$TMP/bisect.out" | head -n 1)
    if [ -n "$bi_detail" ]; then :
    elif [ "$bi_rc" != 143 ] || ! grep -q '^result: CANCELLED (TERM)' <<<"$bi_out"; then
      bi_detail="exit $bi_rc: $bi_out $(cat "$TMP/bisect.err")"
    elif case "$(pstate "$bi_hold")" in '' | Z*) false ;; esac; then
      bi_detail="the held dune $bi_hold survived the cancellation"
      kill -KILL "$bi_hold" 2>/dev/null
    elif ! bi_clean; then bi_detail="the tree was left, or the checkout changed"
    fi
  fi
  if [ -z "$bi_detail" ]; then report 0 "$l_cancel"; else report 1 "$l_cancel" "$bi_detail"; fi
fi
}
harness_leg_run '82: bisect #bisect' leg_82

# A bisect must compare the full target set at every revision, even when the
# remaining targets produce an ordinary passing or failing run.
leg_121() {
  local label="bisect aborts on backend-disabled targets, for passing and failing partial runs"
  local real root state outcome gate action base bad rc dir detail= saved mutant
  if ! harness_real_dune; then skip "$label" "$harness_dune_absent"; return; fi
  command -v git >/dev/null 2>&1 || { skip "$label" 'no git on PATH'; return; }
  real=$harness_dune
  for outcome in pass fail; do
    root=$TMP/bisect-disabled-$outcome
    mkdir -p "$root/tools" "$root/scripts" "$root/al"
    cp "$SRC" "$root/tools/test-run.sh"
    stage_sourced "$root" || { harness_abandon "$stage_error"; return; }
    printf '(lang dune 3.20)\n' >"$root/dune-project"
    printf '(executable (name witness))\n' >"$root/dune"
    printf 'let () = ()\n' >"$root/witness.ml"
    printf '_build/\n' >"$root/.gitignore"
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null git -C "$root" init -q
    git -C "$root" config user.name harness
    git -C "$root" config user.email harness@invalid
    git -C "$root" config commit.gpgsign false
    for state in base partial bad; do
      gate= action='(progn)'
      if [ "$state" = partial ]; then
        gate='(enabled_if false)'
        [ "$outcome" != fail ] || action='(run false)'
      elif [ "$state" = bad ]; then action='(run false)'; fi
      printf '(rule (alias active) (action %s))\n(rule (alias dormant) %s (action (progn)))\n' \
        "$action" "$gate" >"$root/al/dune"
      git -C "$root" add -A && git -C "$root" -c core.hooksPath=/dev/null commit -qm "$state" ||
        { harness_abandon 'cannot commit the bisect fixture'; return; }
      if [ "$state" = base ]; then
        base=$(git -C "$root" rev-parse HEAD)
        git -C "$root" branch -M master
        git -C "$root" checkout -qb range
      fi
    done
    bad=$(git -C "$root" rev-parse HEAD)
    git -C "$root" checkout -q master
    disabled_bisect_run() {
      rc=0
      (cd "$root" && PATH="$(dirname "$real"):$PATH" OCANNL_BACKEND=cc \
        OCANNL_TOOL_TEST_RUNS="$TMP/bisect-disabled-runs" "${BASH:-bash}" tools/test-run.sh \
        bisect "$base" "$bad" @al/active @al/dormant) >"$TMP/bisect-disabled.out" 2>&1 || rc=$?
      dir=$(sed -n 's/^bisect: \(\/.*\)$/\1/p' "$TMP/bisect-disabled.out" | head -n 1)
    }
    disabled_bisect_run
    if [ "$rc" != 2 ] || ! grep -qF 'targets skipped as backend-disabled: @al/dormant' "$TMP/bisect-disabled.out" ||
       [ "$(cut -f3 "$dir/steps.tsv" | tr '\n' ' ')" != 'bad good ABORT ' ]; then
      detail="$outcome: partial run judged the step: exit $rc: $(cat "$TMP/bisect-disabled.out")"; break
    fi
    # Negative control: removing only the record check makes the SAME oracle
    # fail, and the bisect instead publishes a first-bad result on a partial set.
    saved=$TMP/bisect-helper-$outcome
    cp "$root/tools/bisect-steps.sh" "$saved"
    mutant=$TMP/bisect-helper-mutant-$outcome
    awk '/^    # A partial target set/ { cut = 1 }
      cut && /^    fi$/ { cut = 0; next }
      !cut { print }' "$saved" >"$mutant"
    if cmp -s "$saved" "$mutant" || grep -q 'targets skipped as backend-disabled' "$mutant"; then
      harness_abandon 'cannot remove the bisect partial-target guard'; return
    fi
    cp "$mutant" "$root/tools/bisect-steps.sh"
    disabled_bisect_run
    cp "$saved" "$root/tools/bisect-steps.sh"
    if [ "$rc" != 0 ] || ! grep -q '^result: FIRST BAD COMMIT ' "$TMP/bisect-disabled.out"; then
      detail="$outcome: negative control did not publish the partial verdict: exit $rc: $(cat "$TMP/bisect-disabled.out")"; break
    fi
    if [ -n "$(git -C "$root" status --porcelain)" ] ||
       [ "$(git -C "$root" worktree list | wc -l | tr -d ' ')" != 1 ]; then
      detail="$outcome: the fixture checkout changed or a bisect tree was left"; break
    fi
  done
  if [ -z "$detail" ]; then report 0 "$label"; else report 1 "$label" "$detail"; fi
}
harness_leg_run '121: bisect rejects backend-disabled targets #bisect' leg_121

# ---------------------------------------------------------------------------
# Leg 86: a structural failure ends only its own leg (gh-ocannl-1374)
# ---------------------------------------------------------------------------
# Each leg's body is a function, and a check a leg cannot get past -- here,
# leg 79's mutant that must cut the credential scrub out of test-run.sh -- ends
# it through harness_abandon, not `exit`: the legs after it still run and are
# reported. A copy of this harness runs against a copy of test-run.sh whose
# scrub line is spelled so the mutant's pattern no longer matches it (the drift
# the check exists for), with leg 79 and the file's last leg, 70, selected. The
# copy must report leg 79 abandoned, report leg 70 and reach its footer. The
# negative control is the same copy with that check's `return` put back as the
# `exit 2` it was: the copy then aborts inside leg 79, saying so, and leg 70
# never reports, so the oracle tells the two apart.
self_label="a leg's structural failure ends that leg alone: leg 79's broken mutant is abandoned, and leg 70 after it still runs and reports"
self_nc_label="negative control: the same check spelled as an exit aborts the harness inside leg 79, and leg 70 never reports"
# The name leg 79 runs under, read from its own call below rather than restated.
self_leg79=$(sed -n "s/^harness_leg_run '\\(79: [^']*\\)' leg_79\$/\\1/p" "$SELF")
self_run() { # tag: the copy under $TMP/self-<tag>, run on legs 79 and 70; sets self_rc, self_out
  local root=$TMP/self-$1
  self_rc=0
  (
    unset OCANNL_TOOL_HARNESS_STRICT OCANNL_TOOL_HARNESS_EXPECTED_SKIPS \
      OCANNL_TOOL_HARNESS_NAME OCANNL_TOOL_HARNESS_SKIP_RECORD OCANNL_TOOL_HARNESS_DUNE_SKIP_RECORD \
      OCANNL_TOOL_HARNESS_SUBJECT # the copy tests the script it was staged with
    mkdir -p "$root/tmp"
    TMPDIR=$root/tmp bash "$root/tools/${SELF##*/}" --only '^(79|70):'
  ) >"$root.out" 2>&1 || self_rc=$?
  self_out=$(cat "$root.out")
}
self_stage() { # tag: a copy of tools/ and scripts/ with the scrub line respelled
  local root=$TMP/self-$1
  mkdir -p "$root" && cp -R "$HERE" "$root/tools" && cp -R "$HERE/../scripts" "$root/scripts" &&
    awk '$0 == "  eval \"$(credential_env_scrub_text)\" ||" { n++; print "  eval \"$(credential_env_scrub_text)\"  ||"; next }
         { print } END { exit n == 1 ? 0 : 1 }' "$SRC" >"$root/tools/test-run.sh" &&
    bash -n "$root/tools/test-run.sh"
}
leg_86() {
[ -n "$self_leg79" ] || { harness_abandon "no call runs leg 79 as leg_79 in $SELF"; return; }
local abandoned="FAIL  $self_leg79: abandoned, its later checks did not run"
self_stage ship || { harness_abandon "the harness copy, or its respelled scrub line, could not be staged"; return; }
self_run ship
if [ "$self_rc" = 1 ] && grep -qxF "$abandoned" <<<"$self_out" &&
   { grep -qxF "PASS  $herm_label" <<<"$self_out" || grep -qxF "FAIL  $herm_label" <<<"$self_out"; } &&
   grep -qF "ran 2 leg(s):" <<<"$self_out" && ! grep -qF "aborted before finish" <<<"$self_out"; then
  report 0 "$self_label"
else
  report 1 "$self_label" "exit $self_rc (want 1): $self_out"
fi
self_stage exit || { harness_abandon "the negative control's copy could not be staged"; return; }
awk '$0 == "  harness_abandon \"the credential scrub could not be cut out of $SRC; the mutant is broken\"" {
       print; getline; if ($0 == "  return") { n++; print "  exit 2"; next } }
     { print } END { exit n == 1 ? 0 : 1 }' "$SELF" >"$TMP/self-exit/tools/${SELF##*/}" ||
  { harness_abandon "the negative control's exit could not be spliced into the harness copy"; return; }
self_run exit
if [ "$self_rc" = 2 ] && grep -qF "aborted before finish (rc=2) inside leg: $self_leg79" <<<"$self_out" &&
   ! grep -qF "$herm_label" <<<"$self_out"; then
  report 0 "$self_nc_label"
else
  report 1 "$self_nc_label" "exit $self_rc (want 2): $self_out"
fi
}
harness_leg_run '86: a structural failure ends only its own leg #harness' leg_86

# ---------------------------------------------------------------------------
# Leg 105: tools/test-test-run.sh runs the members holding a selected leg
# (gh-ocannl-2002)
# ---------------------------------------------------------------------------
# The harness is several members, and a leg's number names it across them, so
# the one command a person iterates with has to find the member. The script,
# copied beside a stand-in for the list it reads, naming two members and a
# harness that is none, each member a stub that lists its legs and records the
# words it was run with: a selector runs exactly the members listing a leg it
# matches, handing each every word; one matching no leg anywhere runs nothing
# (exit 2); with none, every member runs, a failure among them exits 1 after
# the rest, a refusal 2, and an interrupted member ends the run with its
# status.
front_label="the one command runs the members holding a selected leg, each with every word, and no member for a selector none holds"
front_nc_label="negative control: without the selector's match, the member holding no selected leg runs too"
front_all_label="with no selector every member runs: a failed one exits 1 after the rest, a refused one 2, an interrupted one ends the run"
front_run() { # [argv...]: sets front_rc, front_out, front_calls
  : >"$TMP/front/calls"
  front_rc=0
  front_out=$(FRONT_CALLS=$TMP/front/calls bash "$TMP/front/tools/test-test-run.sh" "$@" 2>&1) || front_rc=$?
  front_calls=$(tr '\n' '|' <"$TMP/front/calls")
}
leg_105() {
local detail= m
mkdir -p "$TMP/front/tools"
cp "$HERE/test-test-run.sh" "$TMP/front/tools/" || { harness_abandon "no $HERE/test-test-run.sh to copy"; return; }
# The list the script asks, by the name it asks it under: read from the script,
# never spelled here, where it would hand this member every edit of that list.
local lister
lister=$(sed -n 's/^done < <("${BASH:-bash}" "$here\/\([a-z-]*\.sh\)" --list)$/\1/p' "$HERE/test-test-run.sh")
[ -n "$lister" ] || { harness_abandon "$HERE/test-test-run.sh reads its members from no script's --list"; return; }
printf '%s\n' '#!/usr/bin/env bash' '[ "$*" = --list ] || exit 9' \
  'printf "%s\n" tools/test-other.sh tools/test-test-run-a.sh tools/test-test-run-b.sh' >"$TMP/front/tools/$lister"
for m in a b; do
  cat >"$TMP/front/tools/test-test-run-$m.sh" <<'STUB'
#!/usr/bin/env bash
case " $* " in *' --list-legs '*) cat "$0.legs"; exit 0 ;; esac
printf '%s:%s\n' "${0##*/}" "$*" >>"$FRONT_CALLS"
exit "$(cat "$0.rc")"
STUB
  echo 0 >"$TMP/front/tools/test-test-run-$m.sh.rc"
done
printf 'LEG  %s\n' '1: one #x' '2: two #y' >"$TMP/front/tools/test-test-run-a.sh.legs"
printf 'LEG  %s\n' '3: three #x' >"$TMP/front/tools/test-test-run-b.sh.legs"
front_run --keep --only '^3:'
[ "$front_rc $front_calls" = "0 test-test-run-b.sh:--keep --only ^3:|" ] ||
  detail="--only '^3:': exit $front_rc, ran: $front_calls"
if [ -z "$detail" ]; then
  front_run --only='#x'
  [ "$front_rc $front_calls" = "0 test-test-run-a.sh:--only=#x|test-test-run-b.sh:--only=#x|" ] ||
    detail="--only='#x': exit $front_rc, ran: $front_calls"
fi
if [ -z "$detail" ]; then
  front_run --only '^9:'
  { [ "$front_rc $front_calls" = "2 " ] && grep -qF "selected no leg of any member" <<<"$front_out"; } ||
    detail="--only '^9:': exit $front_rc (want 2), ran: $front_calls; $front_out"
fi
if [ -z "$detail" ]; then
  front_run --members
  [ "$front_rc $front_calls $front_out" = "0  tools/test-test-run-a.sh"$'\n'"tools/test-test-run-b.sh" ] ||
    detail="--members: exit $front_rc, ran: $front_calls; $front_out"
fi
if [ -z "$detail" ]; then report 0 "$front_label"; else report 1 "$front_label" "$detail"; fi
# Negative control: a copy that keeps every member, whatever its legs, runs
# the member holding no selected leg too, which the first probe above refuses.
detail=
awk '$0 == "      [[ ${leg#LEG  } =~ $only ]] || continue" { n++; next } { print } END { exit n == 1 ? 0 : 1 }' \
  "$HERE/test-test-run.sh" >"$TMP/front/tools/test-test-run.sh" && bash -n "$TMP/front/tools/test-test-run.sh" ||
  { harness_abandon "the selector's match is not in $HERE/test-test-run.sh exactly once; the mutant is broken"; return; }
front_run --keep --only '^3:'
[ "$front_rc $front_calls" = "0 test-test-run-a.sh:--keep --only ^3:|test-test-run-b.sh:--keep --only ^3:|" ] ||
  detail="exit $front_rc, ran: $front_calls"
if [ -z "$detail" ]; then report 0 "$front_nc_label"; else report 1 "$front_nc_label" "$detail"; fi
cp "$HERE/test-test-run.sh" "$TMP/front/tools/"
detail=
echo 1 >"$TMP/front/tools/test-test-run-a.sh.rc"
front_run
[ "$front_rc $front_calls" = "1 test-test-run-a.sh:|test-test-run-b.sh:|" ] ||
  detail="a failed member: exit $front_rc (want 1), ran: $front_calls"
if [ -z "$detail" ]; then
  echo 2 >"$TMP/front/tools/test-test-run-a.sh.rc"
  front_run
  [ "$front_rc $front_calls" = "2 test-test-run-a.sh:|test-test-run-b.sh:|" ] ||
    detail="a refused member: exit $front_rc (want 2), ran: $front_calls"
fi
if [ -z "$detail" ]; then
  echo 143 >"$TMP/front/tools/test-test-run-a.sh.rc"
  front_run
  [ "$front_rc $front_calls" = "143 test-test-run-a.sh:|" ] ||
    detail="an interrupted member: exit $front_rc (want 143), ran: $front_calls"
fi
if [ -z "$detail" ]; then report 0 "$front_all_label"; else report 1 "$front_all_label" "$detail"; fi
}
harness_leg_run '105: the one command runs the members holding a selected leg #harness' leg_105

test_run_close

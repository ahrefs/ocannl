#!/usr/bin/env bash
# The owed-run record (gh-ocannl-2218): evidence, read by the merge build, that a run the merge
# build does not make itself -- a cross-cutting change's full CPU suite, the `train-` aliases a
# change reaches -- ran on the PR's head.
# Usage:
#   tools/owed-run.sh record [RUN|last] [--print]  # publish a finished tools/test-run.sh run
#   tools/owed-run.sh add LINE                     # publish a line `record --print` printed
#   tools/owed-run.sh waive --head SHA --reason TEXT  # publish a waiver for exactly that head
#   tools/owed-run.sh check HEAD NEED...           # the merge build's reader (NEED below)
#   tools/owed-run.sh prune [HEAD...]              # delete the records of merged heads
#   tools/owed-run.sh url                          # the repository the records live in
#
# Where a record lives: the ref refs/owed-runs/<head sha> on OWED_RUN_URL (below; the environment
# variable OCANNL_TOOL_OWED_RUN_URL overrides it, for the test harness and for a fork). The ref
# names a commit whose ONLY parent is the head and whose tree is the head's tree, so it neither
# moves the PR's head (no CI run, review round or checkpoint is re-triggered) nor carries code;
# its message holds the records, one line each:
#   owed-run: v2 run head=<sha> writer=<writer> exit=0 clean=yes backend=cc source=<host>:<run dir> cmd=<dune argv>
#   owed-run: v2 waiver head=<sha> writer=<writer> by=<who> reason=<text>
# Run writers are record, relay, or leg:<hostname> (ASCII letters, digits, dot, hyphen and
# underscore); waiver writers are waive or relay. Existing v1 lines omit the writer and are
# admitted as writer=legacy. `add` rewrites either version as v2 with writer=relay.
# Whoever can push to that repository can write one (the decision recorded on gh-ocannl-2218:
# attestation, as the PR-body citation it replaces was, now machine-checked); any box with network
# can read one, anonymously. Each write keeps the head's earlier records and adds its own.
#
# What `record` certifies, read from the run's own record under ~/.ocannl-test-runs (written by
# tools/test-run.sh): `exit` present (the run finished) and 0; `dirty` present and empty (the
# checkout was clean at launch); `head`, the commit the record is for; `expect-backend` exactly
# `cc` (test-run.sh refused the batch unless it held cc: `--expect-backend cc`); and `cmd`, dune's
# argv, one of the allowlisted forms (FULL, TRAIN below). Refused otherwise, nothing published.
#
# What `check HEAD NEED...` admits (exit 0, one `owed-run: admitted ...` line per record it
# used), as a fail-closed allowlist. NEED is `runtest` (the full suite), `train` (every `train-`
# alias: `@train`) or one alias `@test/training/train-NAME`. The ref must exist on OWED_RUN_URL
# and be fetched within FETCH_CAP seconds; its commit must have HEAD as its only parent and
# HEAD's tree; every `owed-run:` line in its message must parse as v1 or v2 and name HEAD; and together
# the run records must cover every NEED, or one waiver must stand:
#   FULL   cmd `runtest [-j N]`, or `build @runtest [@train] [-j N]`: covers `runtest`;
#          (in every form, `-j N` and `--no-buffer` may stand anywhere after the subcommand, as
#          tools/test-run.sh injects them: neither filters what runs; a `fresh` run's injected
#          `--build-dir=...` is refused like any other option, so record a `run`);
#   TRAIN  cmd `build @train [-j N]` (with or without @runtest): covers `train` and every alias;
#          cmd `build @test/training/train-NAME... [-j N]` covers the aliases it names.
# Refused (exit 1, saying why and naming OWED_RUN_URL): no ref (missing); a ref the fetch could not
# read in time (unreadable); a commit whose parent is not HEAD or whose tree is another (foreign:
# a record of another head, i.e. of an older push or another PR, is under that head's own name and
# is never read for this one); a malformed or unknown line, a record of another head; a run that
# is not exit 0, clean and cc, or whose cmd is any other form (a target list, a directory, a
# `runtest-` alias, an option: a filtered or first-tier run); a NEED no record covers.
# Writer and source are informational. The writer identifies the publication path. Anyone with
# push access can still craft a line claiming any writer. The record certifies the checkout at
# launch and the backend held. Other ambient settings and edits undone mid-run fall outside it.
#
# The waiver is the way through for a head no record can certify (a run on a box the record
# cannot reach, a release hotfix): `waive --head SHA --reason '<the run it cites, or the decision
# and who made it>'`. It covers every NEED for exactly that head, and the merge build prints it,
# so the merge log carries the decision. ship-pr's `--override` does not pass a refusal of this
# reader: the merge build names the merged tree in it, which the base alone never shares.
#
# `prune` deletes refs/owed-runs/<head> for each head already reachable from the repository's
# master (all of them when no HEAD is named): the landing recipe runs it after a merge, so the
# refs do not accumulate. An unmerged head's record is kept.
#
# Exit status: 0 done or admitted, 1 refused, 2 usage error.
set -u
me=owed-run
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd) || exit 2
# The repository the records live in: the PR repository (AGENTS.md, Pull Requests).
OWED_RUN_URL=${OCANNL_TOOL_OWED_RUN_URL:-https://github.com/lukstafi/ocannl-staging.git}
# The longest a read or write of the repository may take, in seconds
# (OCANNL_TOOL_OWED_RUN_FETCH_CAP overrides it, for the harness's hanging-transport row).
FETCH_CAP=${OCANNL_TOOL_OWED_RUN_FETCH_CAP:-60}
case $FETCH_CAP in '' | *[!0-9]* | 0) echo "$me: OCANNL_TOOL_OWED_RUN_FETCH_CAP must be a positive integer" >&2; exit 2 ;; esac
SCHEMA=v2

die() { echo "$me: $*" >&2; exit 2; }
refuse() { echo "$me: REFUSED -- $* (records on $OWED_RUN_URL)"; exit 1; }
is_sha() { case $1 in *[!0-9a-f]* | '') return 1 ;; esac; [ "${#1}" = 40 ]; }
# bounded CMD...: CMD under FETCH_CAP (macOS has no timeout(1); perl's alarm survives exec).
bounded() { perl -e 'alarm shift; exec @ARGV or exit 127' "$FETCH_CAP" "$@"; }
# A read never prompts and reads no ambient credential helper's prompt.
rgit() { GIT_TERMINAL_PROMPT=0 bounded git -C "$root" "$@"; }

# remote_ref REF: sets `remote_sha` to what REF names on OWED_RUN_URL, empty when it names nothing.
# Status 1 when the repository could not be read.
remote_ref() {
  local out
  remote_sha=
  out=$(rgit ls-remote "$OWED_RUN_URL" "$1" 2>/dev/null) || return 1
  remote_sha=$(printf '%s\n' "$out" | awk -v r="$1" '$2 == r { print $1 }')
  return 0
}
# fetch_objects REF...: the objects of REF... into this repository; no ref or FETCH_HEAD written.
fetch_objects() {
  rgit fetch --quiet --no-tags --no-write-fetch-head --refmap= "$OWED_RUN_URL" "$@" >/dev/null 2>&1
}

# cmd_cover CMD: sets `cover` to what a run of dune argv CMD covers (`runtest`, `train`, aliases,
# space-separated), or fails when CMD is not an allowlisted form.
cmd_cover() {
  local w full=0 train=0 names= sub= jwant=0
  cover=
  # No token may carry a quote, an escape or an expansion: `cmd` is printf %q output, never eval'd.
  case $1 in *[\'\"\\\$\`]*) return 1 ;; esac
  set -f
  # shellcheck disable=SC2086 # the argv, split on purpose
  set -- $1
  set +f
  [ $# -gt 0 ] || return 1
  sub=$1
  shift
  for w do
    if [ "$jwant" = 1 ]; then
      case $w in '' | *[!0-9]*) return 1 ;; esac
      jwant=0
      continue
    fi
    case $sub/$w in
      # What tools/test-run.sh injects into the argv it records: a width, and --no-buffer for a
      # single per-test build (capped_stderr). Neither filters what runs.
      */--no-buffer) ;;
      */-j) jwant=1 ;;
      */-j[0-9]*) case ${w#-j} in *[!0-9]*) return 1 ;; esac ;;
      build/@runtest) full=1 ;;
      build/@train) train=1 ;;
      build/@test/training/train-?*)
        case ${w#@test/training/train-} in *[!A-Za-z0-9_]*) return 1 ;; esac
        names="$names $w" ;;
      *) return 1 ;;
    esac
  done
  [ "$jwant" = 0 ] || return 1
  case $sub in
    runtest) full=1 ;;
    build) [ "$full$train" != 00 ] || [ -n "$names" ] || return 1 ;;
    *) return 1 ;;
  esac
  [ "$full" = 0 ] || cover="$cover runtest"
  [ "$train" = 0 ] || cover="$cover train"
  cover="$cover$names"
  cover=${cover# }
}

# parse_line LINE: sets kind, head, writer, and for a run exit_, clean, backend, source, cmd;
# for a waiver by, reason. Fails on anything else.
parse_line() {
  local rest schema
  kind= head= writer=legacy exit_= clean= backend= source= cmd= by= reason=
  case $1 in *$'\n'*) return 1 ;; esac
  case $1 in
    'owed-run: v1 run head='* | 'owed-run: v1 waiver head='*) schema=v1 ;;
    "owed-run: $SCHEMA run head="* | "owed-run: $SCHEMA waiver head="*) schema=$SCHEMA ;;
    *) return 1 ;;
  esac
  rest=${1#"owed-run: $schema "}
  kind=${rest%% *} rest=${rest#* }
  head=${rest%% *} head=${head#head=} rest=${rest#* }
  is_sha "$head" || return 1
  if [ "$schema" = "$SCHEMA" ]; then
    case $rest in writer=*' '*) ;; *) return 1 ;; esac
    writer=${rest%% *} writer=${writer#writer=} rest=${rest#* }
    case $kind/$writer in
      run/record | run/relay | waiver/waive | waiver/relay) ;;
      run/leg:*) valid_host "${writer#leg:}" || return 1 ;;
      *) return 1 ;;
    esac
  fi
  case $kind in
    run)
      case $rest in exit=*' clean='*' backend='*' source='*' cmd='*) ;; *) return 1 ;; esac
      exit_=${rest%% *} exit_=${exit_#exit=} rest=${rest#* }
      clean=${rest%% *} clean=${clean#clean=} rest=${rest#* }
      backend=${rest%% *} backend=${backend#backend=} rest=${rest#* }
      source=${rest%% cmd=*} source=${source#source=}
      cmd=${rest#* cmd=}
      [ -n "$source" ] && [ -n "$cmd" ] ;;
    waiver)
      case $rest in by=*' reason='?*) ;; *) return 1 ;; esac
      by=${rest%% reason=*} by=${by#by=}
      reason=${rest#* reason=}
      [ -n "$by" ] ;;
    *) return 1 ;;
  esac
}

valid_host() {
  local LC_ALL=C
  case $1 in '' | *[!A-Za-z0-9._-]*) return 1 ;; esac
}

run_line() {
  printf 'owed-run: %s run head=%s writer=%s exit=0 clean=yes backend=cc source=%s cmd=%s' \
    "$SCHEMA" "$1" "$2" "$3" "$4"
}

waiver_line() {
  printf 'owed-run: %s waiver head=%s writer=%s by=%s reason=%s' "$SCHEMA" "$1" "$2" "$3" "$4"
}

# publish HEAD LINE: adds LINE to refs/owed-runs/HEAD on OWED_RUN_URL, keeping the head's earlier
# lines; the push is leased on the ref as it was read, so two writers do not drop each other's.
publish() {
  local h=$1 line=$2 ref="refs/owed-runs/$1" old msg tree commit
  git -C "$root" cat-file -e "$h^{commit}" 2>/dev/null || die "$h is not a commit in this checkout"
  remote_ref "$ref" || die "could not read $ref on $OWED_RUN_URL"
  old=$remote_sha
  msg="owed-run record for $h"$'\n\n'
  if [ -n "$old" ]; then
    fetch_objects "$ref" || die "could not fetch $ref from $OWED_RUN_URL"
    msg+=$(git -C "$root" log -1 --format=%B "$old" | grep '^owed-run: ' | grep -vxF -- "$line")$'\n'
  fi
  msg+=$line
  tree=$(git -C "$root" rev-parse "$h^{tree}") || die "no tree for $h"
  commit=$(printf '%s\n' "$msg" | git -C "$root" commit-tree "$tree" -p "$h") || die "could not write the record commit"
  bounded git -C "$root" push --quiet --force-with-lease="$ref:$old" "$OWED_RUN_URL" "$commit:$ref" ||
    die "the push of $ref to $OWED_RUN_URL failed (retry: another writer may have moved it)"
  echo "$me: published $ref on $OWED_RUN_URL: $line"
}

read_pid_field() {
  local id
  id=$(ps -o "$1=" -p "$2") || return 1
  id=${id//[[:space:]]/}
  case $id in '' | 0 | *[!0-9]*) return 1 ;; esac
  printf '%s\n' "$id"
}

# Both the fleet runner and report get a bounded nested slot, each with its own process group.
# The parent of this group's leader belongs to the enclosing leg's group. Its live leader's
# PID and start token must appear in test-run.sh's stored launchers. The argv digest is outside
# this check.
fleet_receipt() {
  local run=$1 inner parent scope token row
  inner=$(read_pid_field pgid "$$") && parent=$(read_pid_field ppid "$inner") &&
    scope=$(read_pid_field pgid "$parent") || return 1
  [ "$scope" != "$inner" ] || return 1
  if [ -r /proc/self/stat ]; then
    token=$(awk '{ sub(/^.*\) /, ""); print $20 }' "/proc/$scope/stat") || return 1
  else
    token=$(LC_ALL=C TZ=UTC ps -o lstart= -p "$scope" | awk '{$1=$1; print}') || return 1
  fi
  [ -n "$token" ] && [ -r "$run/launchers" ] || return 1
  while IFS= read -r row; do
    case $row in "$scope $token "*) return 0 ;; esac
  done <"$run/launchers"
  return 1
}

on_record_master() {
  local h=$1 master rc
  remote_ref refs/heads/master || die "record: could not read master on $OWED_RUN_URL"
  master=$remote_sha
  [ -n "$master" ] || return 1
  if ! git -C "$root" cat-file -e "$master^{commit}" 2>/dev/null; then
    fetch_objects refs/heads/master && git -C "$root" cat-file -e "$master^{commit}" 2>/dev/null ||
      die "record: could not fetch master on $OWED_RUN_URL"
  fi
  git -C "$root" merge-base --is-ancestor "$h" "$master"
  rc=$?
  [ "$rc" -le 1 ] || die "record: could not compare $h with master on $OWED_RUN_URL"
  return "$rc"
}

record_run() {
  local run=$1 print=$2 requested_cover=${3-} h c host line checkout writer=record
  case $run in
    '' | last) run=$(cd "$root" && "${BASH:-bash}" tools/test-run.sh paths run last) || die "record: no last run of this worktree" ;;
  esac
  [ -d "$run" ] || die "record: no run directory $run"
  [ -f "$run/exit" ] || die "record: $run has no exit status: it has not finished"
  [ "$(cat "$run/exit")" = 0 ] || die "record: $run exited $(cat "$run/exit"), not 0"
  [ -f "$run/dirty" ] || die "record: $run recorded no checkout state"
  [ ! -s "$run/dirty" ] || die "record: $run ran a dirty checkout"
  h=$(sed -n 1p "$run/head" 2>/dev/null)
  is_sha "$h" || die "record: $run recorded no head commit"
  [ "$(cat "$run/expect-backend" 2>/dev/null)" = cc ] ||
    die "record: $run was not run with --expect-backend cc (its expect-backend: '$(cat "$run/expect-backend" 2>/dev/null)')"
  c=$(sed -n 1p "$run/cmd" 2>/dev/null) c=${c% }
  cmd_cover "$c" || die "record: $run ran '$c', none of the forms a record covers (tools/owed-run.sh's header)"
  host=$(hostname -s 2>/dev/null || hostname)
  if [ -n "$requested_cover" ]; then
    [ "$cover" = "$requested_cover" ] || die "record: $run does not cover the fleet leg's requested targets"
    [ "$h" = "$(git -C "$root" rev-parse HEAD)" ] || die "record: $run is for a different head"
    checkout=$(git -C "$root" status --porcelain --untracked-files=all --ignore-submodules=none) ||
      die "record: could not read the fleet worktree's checkout state"
    [ -z "$checkout" ] || die "record: the fleet worktree is dirty"
    fleet_receipt "$run" || die "record: $run was not launched by the current fleet leg"
    # Integration runs at merged heads owe no record and can leave a ref that no PR will prune.
    if on_record_master "$h"; then
      printf '%s\n' "$me: recorded nothing (head is on master at $OWED_RUN_URL)"
      return 0
    fi
    valid_host "$host" || die "record: cannot use '$host' as a leg writer hostname"
    writer=leg:$host
  fi
  line=$(run_line "$h" "$writer" "$host:$run" "$c")
  if [ "$print" = 1 ]; then printf '%s\n' "$line"; else publish "$h" "$line"; fi
}

leg_command() {
  if [ "${1-}" = env ] && [ "${2-}" = OCANNL_BACKEND=cc ]; then shift 2; fi
  case ${1-} in tools/test-run.sh | ./tools/test-run.sh) shift ;; *) return 1 ;; esac
  [ "${1-}" = run ] && [ "${2-}" = --expect-backend ] && [ "${3-}" = cc ] || return 1
  shift 3
  cmd_cover "$(printf '%q ' "$@")"
}

verb=${1-}
[ $# -gt 0 ] && shift
case $verb in
  url) printf '%s\n' "$OWED_RUN_URL" ;;
  record)
    run= print=0
    for a do
      case $a in
        --print) print=1 ;;
        -*) die "record: unknown option '$a'" ;;
        *) [ -z "$run" ] || die "record: one RUN at most"; run=$a ;;
      esac
    done
    record_run "$run" "$print" ;;
  leg-report)
    if ! leg_command "$@"; then
      printf '%s\n' "$me: recorded nothing (not an owed-run argv)"
    elif ! ( export GIT_TERMINAL_PROMPT=0; record_run last 0 "$cover" ) </dev/null 2>&1; then
      printf '%s\n' "$me: recorded nothing (record refused)"
    fi ;;
  add)
    [ $# = 1 ] || die "add: one LINE"
    parse_line "$1" || die "add: not an owed-run line: $1"
    if [ "$kind" = run ]; then
      [ "$exit_/$clean/$backend" = 0/yes/cc ] && cmd_cover "$cmd" || die "add: a record no check admits: $1"
      line=$(run_line "$head" relay "$source" "$cmd")
    else
      line=$(waiver_line "$head" relay "$by" "$reason")
    fi
    publish "$head" "$line" ;;
  waive)
    h= why=
    while [ $# -gt 0 ]; do
      case $1 in
        --head) h=${2-}; shift ;;
        --reason) why=${2-}; shift ;;
        *) die "waive: unknown argument '$1'" ;;
      esac
      shift
    done
    is_sha "$h" || die "waive: --head takes a full 40-hex commit"
    case $why in '' | *$'\n'*) die "waive: --reason takes one non-empty line" ;; esac
    who=$(git -C "$root" var GIT_AUTHOR_IDENT 2>/dev/null) || who=
    who=${who%% <*} who=${who:-$(id -un)} who=${who// /_}
    publish "$h" "$(waiver_line "$h" waive "$who" "$why")" ;;
  check)
    [ $# -ge 2 ] || die "check: HEAD and at least one NEED"
    h=$1
    shift
    is_sha "$h" || die "check: HEAD must be a full 40-hex commit"
    for n do
      case $n in runtest | train | @test/training/train-?*) ;; *) die "check: unknown NEED '$n'" ;; esac
    done
    ref=refs/owed-runs/$h
    remote_ref "$ref" || refuse "$ref could not be read within ${FETCH_CAP}s: unreadable"
    [ -n "$remote_sha" ] || refuse "no $ref: the head has no record (missing)"
    fetch_objects "$ref" && git -C "$root" cat-file -e "$remote_sha^{commit}" 2>/dev/null ||
      refuse "$ref (${remote_sha:0:8}) could not be fetched within ${FETCH_CAP}s: unreadable"
    [ "$(git -C "$root" rev-list --parents -n 1 "$remote_sha")" = "$remote_sha $h" ] ||
      refuse "$ref's commit is not a child of ${h:0:8} alone: foreign"
    [ "$(git -C "$root" rev-parse "$remote_sha^{tree}")" = "$(git -C "$root" rev-parse "$h^{tree}" 2>/dev/null)" ] ||
      refuse "$ref's commit carries another tree than ${h:0:8}'s: foreign"
    got=" " admitted= waived= nlines=0
    while IFS= read -r l; do
      case $l in owed-run:*) ;; *) continue ;; esac
      nlines=$((nlines + 1))
      parse_line "$l" || refuse "$ref holds a line no v1 or $SCHEMA record reads: $l"
      [ "$head" = "$h" ] || refuse "$ref holds a record of another head (${head:0:8}): foreign"
      if [ "$kind" = waiver ]; then
        waived="owed-run: admitted waiver for ${h:0:8} by $by (writer=$writer): $reason"
        continue
      fi
      [ "$exit_" = 0 ] || refuse "$ref holds a run that exited $exit_"
      [ "$clean" = yes ] || refuse "$ref holds a run of a dirty checkout"
      [ "$backend" = cc ] || refuse "$ref holds a run on backend $backend, not cc"
      cmd_cover "$cmd" || refuse "$ref holds a run of '$cmd', which is no full or train- run (a filtered or first-tier run)"
      got="$got$cover "
      admitted+="owed-run: admitted run of '$cmd' for ${h:0:8} ($source, writer=$writer)"$'\n'
    done <<EOF
$(git -C "$root" log -1 --format=%B "$remote_sha")
EOF
    [ "$nlines" -gt 0 ] || refuse "$ref holds no record line"
    if [ -n "$waived" ]; then printf '%s\n' "$waived"; exit 0; fi
    missing=
    for n do
      case $got in *" $n "*) continue ;; esac
      case $n in @test/training/train-*) case $got in *' train '*) continue ;; esac ;; esac
      missing="$missing $n"
    done
    [ -z "$missing" ] || refuse "$ref covers no run of:$missing"
    printf '%s' "$admitted" ;;
  prune)
    remote_ref refs/heads/master || die "prune: could not read $OWED_RUN_URL"
    tip=$remote_sha
    is_sha "$tip" || die "prune: $OWED_RUN_URL has no master"
    if [ $# = 0 ]; then
      out=$(rgit ls-remote "$OWED_RUN_URL" 'refs/owed-runs/*') || die "prune: could not list refs/owed-runs on $OWED_RUN_URL"
      set -- $(printf '%s\n' "$out" | awk '{ sub("^refs/owed-runs/", "", $2); print $2 }')
    fi
    fetch_objects refs/heads/master || die "prune: could not fetch master from $OWED_RUN_URL"
    dels=()
    for h do
      is_sha "$h" || die "prune: '$h' is not a full 40-hex commit"
      if git -C "$root" cat-file -e "$h^{commit}" 2>/dev/null && git -C "$root" merge-base --is-ancestor "$h" "$tip"; then
        dels+=(":refs/owed-runs/$h")
      else
        echo "$me: kept refs/owed-runs/$h: ${h:0:8} is not on master at ${tip:0:8} (or not in this checkout)"
      fi
    done
    if [ ${#dels[@]} -gt 0 ]; then
      bounded git -C "$root" push --quiet "$OWED_RUN_URL" "${dels[@]}" || die "prune: the delete push to $OWED_RUN_URL failed"
      echo "$me: pruned ${#dels[@]} record(s) of merged heads on $OWED_RUN_URL"
    fi ;;
  -h | --help) sed -n '2,${/^#/!q;p;}' "$0" | sed 's/^# \{0,1\}//' ;;
  *) die "usage: tools/owed-run.sh record|add|waive|check|prune|url ... (--help)" ;;
esac

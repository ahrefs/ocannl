#!/usr/bin/env bash
# Hermetic regression and mutation tests for tools/ci-shard.sh.
#
# The subject reads two things from its surroundings: which directories hold a
# tracked dune file (git) and what `dune show aliases` says each defines. Both
# are faked here -- a throwaway git repository holding a COPY of the subject
# and a handful of dune files, and a fake dune (through the subject's
# OCANNL_TOOL_CI_SHARD_DUNE hook) serving canned listings the way the real one
# does: on stderr, and with exit 0 even for a directory it does not know. The
# copy carries a fixture WEIGHTS table, so the legs pin the dealing itself
# rather than whatever the shipping table measures this month.
#
#   tools/test-ci-shard.sh          # run every leg
#   tools/test-ci-shard.sh --keep   # keep the scratch directory
#
# What the legs hold, in the order the shards would lose coverage without them:
# for every N the shards partition the unit targets exactly (none lost, none
# twice); the split directory's own `runtest`/`default` never appear as
# targets (either would build every test in every shard); `@rest` reaches each
# suite alias, and only those, in every other directory; the dealing is the
# documented longest-first one and deterministic; and every listing the
# subject cannot trust is refused with exit 2 and a reason. `weigh` is pinned
# on a synthetic trace, including the executable whose name begins with `cc`
# and must not be mistaken for the C compiler. The coverage legs have
# fault-injected twins -- the subject with the split-directory exclusion
# removed, and with `@rest` naming `default` (recursive in dune, so every
# test in one shard) instead of `all` -- which the same oracle must reject.
#
# On no dune alias, like its siblings in CI's shell-harness job: the subject
# runs `python3` under that exact name, which a host without one skips.

set -u

. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/ci-shard.sh"
[ -f "$SRC" ] || { echo "no $SRC" >&2; exit 2; }
harness_require python3 git
harness_scratch test-ci-shard

# The fixture repository: the root, three plain directories, the split
# directory and a subdirectory of it that defines a same-named per-test alias
# (the `@@` form is what keeps it out of the split units).
repo="$TMP/repo"
mkdir -p "$repo/tools" "$repo/lib" "$repo/bin" "$repo/test/operations/sub" "$TMP/aliases"
for d in . lib bin test/operations test/operations/sub; do : >"$repo/$d/dune"; done
# Untracked: not part of the tree the shards cover.
mkdir -p "$repo/scratch" && : >"$repo/scratch/dune"

# The subject, with the fixture table spliced over the shipping one.
fixture_weights='
heavy 10
@rest 3
ghost 5
'
install_subject() { # SOURCE
  python3 - "$1" "$repo/tools/ci-shard.sh" "$fixture_weights" <<'PY'
import re
import sys
source, target, weights = sys.argv[1:4]
text = open(source).read()
new, count = re.subn(r"^WEIGHTS='\n.*?^'\n", lambda m: "WEIGHTS='" + weights + "'\n", text, flags=re.M | re.S)
assert count == 1, 'WEIGHTS block not found exactly once'
open(target, 'w').write(new)
PY
  chmod +x "$repo/tools/ci-shard.sh"
}
install_subject "$SRC"
# The credential deny-list it sources beside itself (gh-ocannl-1280).
cp "$HERE/credential-env.sh" "$repo/tools/credential-env.sh"
git -C "$repo" init -q
git -C "$repo" add -- tools lib bin test dune

# One fixture file per directory, `/` spelled `_` and the root `ROOT`.
listing_file() { # DIR
  case $1 in
    .) printf '%s/ROOT' "$TMP/aliases" ;;
    *) printf '%s/%s' "$TMP/aliases" "$(printf '%s' "$1" | tr / _)" ;;
  esac
}
listing() { # DIR ALIAS...
  local file
  file=$(listing_file "$1")
  shift
  printf '%s\n' "$@" >"$file"
}
default_listings() {
  rm -f -- "$TMP/aliases"/*
  listing . all default runtest
  listing lib all default
  listing bin all default bin-smoke
  listing test/operations all default runtest runtest-a runtest-b runtest-c runtest-heavy bin-smoke scans
  listing test/operations/sub all default runtest runtest-a
  # The split directory's file targets: test executables, their outputs and
  # sources, a helper and a slow-only executable that no unit is named after.
  printf '%s\n' a.exe a.exe.output a.ml b.exe c.exe heavy.exe helper.exe slowonly.exe slowonly.ml \
    >"$TMP/targets"
}
default_listings

# The fake dune: `show aliases DIR...` answers like dune 3.24 -- the listing on
# STDERR, a `DIR:` header per directory, blank lines between blocks, and for an
# unknown directory an `Error:` line with exit status 0. FAKE_DUNE_NOISE puts
# chatter ahead of the first header; FAKE_DUNE_RC fails the command outright.
# `show targets test/operations` serves the one split-directory listing, on
# stderr too, with no header (one directory asked).
cat >"$TMP/dune" <<'FAKE_DUNE'
#!/usr/bin/env bash
# Which credential fixtures reached this call (gh-ocannl-1280): names only.
[ -z "${FAKE_DUNE_CREDENTIALS:-}" ] ||
  printf '%s|%s|%s\n' "${GH_TOKEN+GH_TOKEN}" "${FOO_API_KEY+FOO_API_KEY}" "${CRED_PLAIN+plain}" >>"$FAKE_DUNE_CREDENTIALS"
if [ "$1 $2 $3" = "show targets test/operations" ]; then
  cat "$FAKE_DUNE_TARGETS" >&2
  exit "${FAKE_DUNE_RC:-0}"
fi
[ "$1 $2" = "show aliases" ] || { echo "unsupported fake dune call: $*" >&2; exit 64; }
shift 2
[ -z "${FAKE_DUNE_NOISE:-}" ] || echo "$FAKE_DUNE_NOISE" >&2
first=1
for dir in "$@"; do
  case $dir in
    .) file="$FAKE_DUNE_ALIASES/ROOT" ;;
    *) file="$FAKE_DUNE_ALIASES/$(printf '%s' "$dir" | tr / _)" ;;
  esac
  if [ ! -f "$file" ]; then
    echo "Error: Directory $dir does not exist." >&2
    continue
  fi
  [ "$first" = 1 ] || echo >&2
  first=0
  echo "$dir:" >&2
  cat "$file" >&2
done
exit "${FAKE_DUNE_RC:-0}"
FAKE_DUNE
chmod +x "$TMP/dune"
export OCANNL_TOOL_CI_SHARD_DUNE="$TMP/dune" FAKE_DUNE_ALIASES="$TMP/aliases" FAKE_DUNE_TARGETS="$TMP/targets"

subject() { "$repo/tools/ci-shard.sh" "$@"; }

ops=test/operations
units_expected="@@$ops/runtest-a
@@$ops/runtest-b
@@$ops/runtest-c
@@$ops/runtest-heavy
@@all
@@runtest
@@bin/all
@@bin/bin-smoke
@@lib/all
@@$ops/bin-smoke
@@$ops/sub/all
@@$ops/sub/runtest
$ops/helper.exe
$ops/slowonly.exe"

# The coverage oracle, shared with the mutation twin: for N in 1..4 every
# shard is nonempty, and the shards' targets together are exactly the units'.
partition_holds() { # LOG
  local n k all
  : >"$1"
  for n in 1 2 3 4; do
    all=
    for ((k = 1; k <= n; k++)); do
      out=$(subject targets "$k/$n" 2>>"$1") || { echo "targets $k/$n failed" >>"$1"; return 1; }
      [ -n "$out" ] || { echo "shard $k/$n is empty" >>"$1"; return 1; }
      all="$all$out"$'\n'
    done
    if [ "$(printf '%s' "$all" | sort)" != "$(printf '%s\n' "$units_expected" | sort)" ]; then
      { echo "N=$n targets differ from the units:"; printf '%s' "$all" | sort; } >>"$1"
      return 1
    fi
  done
}
partition_holds "$TMP/partition.log"
report $? "for N=1..4 the shards partition the units exactly, every shard nonempty" "see $TMP/partition.log"

# Each absence is claimed over a listing that holds the units, so an empty
# one cannot pass these vacuously.
all_targets=$(subject targets 1/1 2>/dev/null)
for absent in "@@$ops/runtest" "@@$ops/default" "@@$ops/all" "@@default" "@@lib/default" "@@$ops/scans" "@@lib/runtest" "@@$ops/sub/runtest-a" "@@scratch/all" "$ops/a.exe" "$ops/heavy.exe"; do
  ok=0
  printf '%s\n' "$all_targets" | grep -qxF -- "@@$ops/runtest-heavy" || ok=1
  printf '%s\n' "$all_targets" | grep -qxF -- "$absent" && ok=1
  report $ok "no shard targets $absent"
done

# The dealing: heavy (10) alone on shard 1; @rest (3) then a, b, c (1 each,
# by name) onto the lighter shard 2. The ghost weight names no unit.
shard1=$(subject targets 1/2 2>"$TMP/deal.err")
shard2=$(subject targets 2/2 2>/dev/null)
[ "$shard1" = "@@$ops/runtest-heavy" ]
report $? "the heaviest unit is dealt first, alone on shard 1/2" "got: $shard1"
ok=0
[ "$(printf '%s\n' "$shard2" | head -1)" = "@@all" ] || ok=1
[ "$(printf '%s\n' "$shard2" | tail -3 | tr '\n' ' ')" = "@@$ops/runtest-a @@$ops/runtest-b @@$ops/runtest-c " ] || ok=1
report $ok "the rest go to the lighter shard, @rest first, equal weights by name" "got: $shard2"
grep -q 'weight for ghost, which is no unit any more' "$TMP/deal.err"
report $? "a weight naming no unit warns on stderr" "see $TMP/deal.err"
ok=0
[ "$(subject targets 2/2 2>/dev/null)" = "$shard2" ] || ok=1
[ "$(subject plan 2 2>/dev/null)" = "$(subject plan 2 2>/dev/null)" ] || ok=1
report $ok "targets and plan are deterministic"
subject plan 2 2>/dev/null | grep -q '^shard 1/2: 1 units, load 10 '
report $? "plan reports each shard's units and load"
[ "$(subject aliases)" = "default
runtest
bin-smoke" ]
report $? "aliases names the suite being sharded"

FAKE_DUNE_NOISE='Warning: something dune wanted said' subject targets 1/1 >"$TMP/noise.out" 2>"$TMP/noise.err"
ok=0
[ "$(sort "$TMP/noise.out")" = "$(printf '%s\n' "$units_expected" | sort)" ] || ok=1
grep -q '^Warning: something' "$TMP/noise.err" || ok=1
report $ok "chatter before the first directory header passes through to stderr, unparsed"

# Refusals: each listing the subject cannot trust exits 2 with its reason.
refused() { # LABEL PATTERN COMMAND...
  local label=$1 pattern=$2
  shift 2
  harness_rejected 2 "$pattern" "$TMP/refusal.log" "$@"
  report $? "refused: $label" "$(cat "$TMP/refusal.log")"
}
rm -f -- "$(listing_file lib)"
refused "dune reports an Error line" 'dune show aliases refused' subject targets 1/2
listing lib all
refused "a directory block without default" 'reported nothing for lib' subject targets 1/2
default_listings
listing test/operations all default runtest bin-smoke
refused "the split directory lists no per-test aliases" 'lists no runtest-<name> aliases' subject targets 1/2
default_listings
printf '(alias\n (name default)\n (deps x))\n' >"$repo/lib/dune"
refused "a dune file defining default itself" 'lib/dune defines the default alias' subject targets 1/2
printf '(rule\n (aliases runtest default)\n (action (progn)))\n' >"$repo/lib/dune"
refused "a rule attaching to default" 'lib/dune defines the default alias' subject targets 1/2
printf '; (alias default) in a comment is prose\n(rule (alias runtest) (action (progn)))\n' >"$repo/lib/dune"
subject targets 1/1 >/dev/null 2>"$TMP/comment.err"
report $? "default named only in a comment is not a definition" "$(cat "$TMP/comment.err")"
: >"$repo/lib/dune"
printf 'Error: Directory test/operations does not exist.\n' >"$TMP/targets"
refused "dune show targets reports an Error line" 'dune show targets refused' subject targets 1/2
printf '%s\n' a.ml a.expected >"$TMP/targets"
refused "the split directory lists no executables" 'lists no executables' subject targets 1/2
default_listings
FAKE_DUNE_RC=1 refused "dune show aliases exits nonzero" 'dune show aliases failed' subject targets 1/2
for spec in 0/2 3/2 1/0 x 1/2/3; do
  refused "shard spec $spec" 'shard' subject targets "$spec"
done
refused "plan without N" 'usage' subject plan
refused "weigh without a trace" 'usage' subject weigh "$TMP/no-such-trace"

# weigh: one synthetic trace, written as canonical S-expressions. CPU is
# rusage nanoseconds. Expected: compilers and the toolchain excluded (but not
# cc_march_census.exe); fast-math runs credited to the _fast_math twin; a
# sandboxed run of the split directory counted there; elsewhere -> @rest.
python3 - "$TMP/trace" <<'PY'
import sys


def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s


def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'


def finish(prog, directory, cpu_s, args=()):
    ns = int(cpu_s * 1e9)
    return sexp(['process', 'finish', ['1', '2'], ['process_args', list(args)], ['pid', '7'],
                 ['prog', prog], ['dir', directory], ['exit', '0'],
                 ['rusage', [['user_cpu_time', str(ns - ns // 4)], ['system_cpu_time', str(ns // 4)]]]])


b = '/x/_build/default/'
events = [
    sexp(['config', 'init', '1']),
    finish('/opt/bin/ocamlopt.opt', b + 'lib', 50),
    finish('/usr/bin/cc', b + 'lib', 40),
    finish('cc_march_census.exe', b + 'test/operations', 2),
    finish('online_softmax.exe', b + 'test/operations', 3, ['--ocannl_backend=cc', '--ocannl_cc_backend_fast_math=true']),
    finish('online_softmax.exe', '/x/_build/.sandbox/0123abcd/default/test/operations', 1),
    finish('./einsum_test.exe', b + 'test/einsum', 4),
    finish('placement_store.exe', b + 'test/operations/sub', 0.4),
]
open(sys.argv[1], 'wb').write(b''.join(events))
PY
subject weigh "$TMP/trace" >"$TMP/weigh.out" 2>"$TMP/weigh.err"
printf '%s\n' '@rest 4' 'online_softmax_fast_math 3' 'cc_march_census 2' 'online_softmax 1' >"$TMP/weigh.expected"
diff "$TMP/weigh.expected" "$TMP/weigh.out" >"$TMP/weigh.diff"
report $? "weigh credits CPU per unit from a dune trace" "$(cat "$TMP/weigh.diff" "$TMP/weigh.err")"
printf 'not a trace' >"$TMP/garbage"
harness_rejected 1 'not a canonical-S-expression dune trace' "$TMP/weigh.log" subject weigh "$TMP/garbage"
report $? "weigh refuses a file that is not a dune trace" "$(cat "$TMP/weigh.log")"
printf '(6:config4:init1:1)' >"$TMP/empty-trace"
harness_rejected 1 'no finished processes' "$TMP/weigh.log" subject weigh "$TMP/empty-trace"
report $? "weigh refuses a trace that recorded no process" "$(cat "$TMP/weigh.log")"

# The fault-injected twins. Without the split-directory exclusion, @rest would
# carry test/operations' own runtest and all; naming `default` instead of
# `all`, it would carry dune's recursive implicit default. Either puts every
# test in whichever shard holds @rest, and the partition oracle must reject
# each for exactly that target.
twin() { # LABEL AWK_PROGRAM EXPECTED_TARGET
  local path
  path=$(mutant "$(printf '%s' "$1" | tr ' ' '-')" "$2") \
    || { report 1 "negative control: $1" "could not build the mutant"; return; }
  install_subject "$path"
  if partition_holds "$TMP/twin.log"; then
    report 1 "negative control: $1" "the partition oracle accepted the mutant"
  elif ! grep -qxF -- "$3" "$TMP/twin.log"; then
    report 1 "negative control: $1" "rejected for an unrelated reason; see $TMP/twin.log"
  else
    report 0 "negative control: $1"
  fi
  install_subject "$SRC"
}
twin "no split exclusion" \
  '/if d == split_dir and a in \(.runtest., .default.\):/ { print "        if False:"; next } { print }' \
  "@@$ops/runtest"
# Credentials never reach its `dune show` calls (gh-ocannl-1280), while a plain variable does; the
# negative control is a copy with the scrub cut out of both subshells.
credentials_clean() { # SUBJECT-LABEL -> 0 when every recorded call saw only the plain variable
  rm -f "$TMP/credentials.log"
  GH_TOKEN=fixture-not-a-token FOO_API_KEY=fixture-not-a-key CRED_PLAIN=kept \
    FAKE_DUNE_CREDENTIALS="$TMP/credentials.log" subject targets 1/1 >/dev/null 2>&1 || return 2
  [ "$(grep -c . "$TMP/credentials.log")" -ge 2 ] || return 2
  ! grep -qvxF '||plain' "$TMP/credentials.log"
}
credentials_clean
report $? "dune show runs without GH_TOKEN and FOO_API_KEY, the plain variable kept" "see $TMP/credentials.log"
cred_rc=0
path=$(mutant no-credential-scrub '{ gsub(/scrub_credentials && /, ""); print }') && install_subject "$path" ||
  cred_rc=3
if [ "$cred_rc" = 0 ]; then credentials_clean || cred_rc=$?; fi
install_subject "$SRC"
if [ "$cred_rc" = 1 ]; then
  report 0 "negative control: without the scrub, dune show sees the credentials"
else
  report 1 "negative control: without the scrub, dune show sees the credentials" "exit $cred_rc; see $TMP/credentials.log"
fi
twin "rest names default" \
  "/share = 'all' if a == 'default' else a/ { print \"        share = a\"; next } { print }" \
  "@@default"

finish

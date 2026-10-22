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
# on a synthetic trace -- each process credited to the unit its TARGET names,
# not to its executable (gh-ocannl-1503), with a twin keyed by executable that
# the same oracle must reject -- including the executable whose name begins
# with `cc` and must not be mistaken for the C compiler; the same trace plants a canary
# in every field `weigh` must not decode, with one decode-and-print twin per
# field that the canary check must reject (gh-ocannl-1287). The coverage legs have
# fault-injected twins -- the subject with the split-directory exclusion
# removed, and with `@rest` naming `default` (recursive in dune, so every
# test in one shard) instead of `all` -- which the same oracle must reject.
# The `train` suite's legs hold the same partition over the dune files'
# `train` aggregates with no dune at all, the reader's refusals, and a twin
# that skips `subdir` stanzas.
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
# (the `@@` form is what keeps it out of the split units), and a training
# directory whose dune file is the `train` suite's real input.
repo="$TMP/repo"
mkdir -p "$repo/tools" "$repo/lib" "$repo/bin" "$repo/test/operations/sub" "$repo/test/training" "$TMP/aliases"
for d in . lib bin test/operations test/operations/sub; do : >"$repo/$d/dune"; done
# The `train` suite reads dune files, not dune: an aggregate whose members
# include a rule on two aliases and one whose action quotes `;`, parens and
# `(alias train)`; a train-<name> rule nobody aggregates (no unit: `@train`
# does not build it); the same `(alias train-gate)` as a DEPENDENCY, which
# defines nothing; and a `subdir` stanza in the root holding an aggregate of
# its own. bin/dune names a module `train`, which is no alias at all.
train_dune='; (alias train) in a comment is prose
(rule
 (alias train-gate)
 (action (progn)))
(rule
 (alias train-heavy)
 (deps (alias train-gate))
 (action (progn)))
(rule
 (aliases train-b runtest-b)
 (action (progn)))
(rule
 (alias train-c)
 (action (system "echo \"; (alias train)\" ) (")))
(rule
 (alias train-orphan)
 (action (progn)))
(alias
 (name train)
 (package p)
 (deps
  (alias train-gate)
  (alias train-heavy)
  (alias train-b)
  (alias train-c)))
'
reset_train_dune() { printf '%s' "$train_dune" >"$repo/test/training/dune"; }
reset_train_dune
printf '(subdir sub2\n (rule (alias train-sub) (action (progn)))\n (alias (name train) (deps (alias train-sub))))\n' >"$repo/dune"
printf '(executable (name train) (modules train))\n' >"$repo/bin/dune"
# Untracked: not part of the tree the shards cover.
mkdir -p "$repo/scratch" && : >"$repo/scratch/dune"

# The subject, with the fixture tables spliced over the shipping ones.
fixture_weights='
heavy 10
@rest 3
ghost 5
'
fixture_train_weights='
test/training/train-heavy 10
test/training/train-ghost 5
'
# Wall-clock: `a`, light by CPU, is shard 2/2's critical path.
fixture_wall_weights='
heavy 30
a 50
'
install_subject() { # SOURCE
  python3 - "$1" "$repo/tools/ci-shard.sh" "$fixture_weights" "$fixture_train_weights" "$fixture_wall_weights" <<'PY'
import re
import sys
source, target, weights, train_weights, wall_weights = sys.argv[1:6]
text = open(source).read()
for name, table in (('WEIGHTS', weights), ('TRAIN_WEIGHTS', train_weights), ('WALL_WEIGHTS', wall_weights)):
    text, count = re.subn(r"^" + name + r"='\n.*?^'\n", lambda m: name + "='" + table + "'\n", text, flags=re.M | re.S)
    assert count == 1, name + ' block not found exactly once'
open(target, 'w').write(text)
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
  listing test/training all default train train-gate train-heavy train-b train-c train-orphan
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
@@test/training/all
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
grep -q 'WEIGHT NAMES NO UNIT: ghost 5' "$TMP/deal.err"
report $? "a weight naming no unit is reported loudly on stderr" "see $TMP/deal.err"
ok=0
[ "$(subject targets 2/2 2>/dev/null)" = "$shard2" ] || ok=1
[ "$(subject plan 2 2>/dev/null)" = "$(subject plan 2 2>/dev/null)" ] || ok=1
report $ok "targets and plan are deterministic"
subject plan 2 2>/dev/null >"$TMP/plan.out"
grep -q '^shard 1/2: 1 units, load 10 ([0-9]*%), longest by wall-clock heavy 30s$' "$TMP/plan.out" &&
  grep -q '^shard 2/2: .*, longest by wall-clock a 50s$' "$TMP/plan.out"
report $? "plan reports each shard's units, load and longest unit by WALL-clock, not by CPU" \
  "$(cat "$TMP/plan.out")"
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

# weigh: one synthetic trace, written as canonical S-expressions. A unit's
# seconds are its finish events' CPU times (rusage's user_cpu_time plus
# system_cpu_time, in nanoseconds; never the wall-clock duration), credited
# by the process's first target (gh-ocannl-1503): an
# alias action of the split directory's runtest-<name> -> <name>, its other
# aliases (bin-smoke) -> @rest; a file <name>.<ext...> there -> <name>;
# anything elsewhere -> @rest. Expected: compilers and the toolchain excluded
# (but not cc_march_census.exe); online_softmax.exe credited to each of the
# three tests it runs -- its own (`.exe.output`), the fast-math and the
# fp-licences twins (`.actual`, the flag on its command line read by nobody);
# sweep_harness_driver.exe and inline-test-runner.exe to the aliases whose
# actions they are; the cc_temp_ownership-success variant to itself; a
# sandboxed run of the split directory counted there; a process with no target
# by its directory and program; elsewhere -> @rest.
#
# The trace is also the canary fixture (gh-ocannl-1287): weigh decodes only
# `process finish`'s prog, dir, target_files, digit-only duration and its
# rusage's digit-only CPU times. Every entry of the inventory below carries
# CANARY -- a field of a finish event (process_args among them, and maxrss
# beside the CPU times inside rusage), of a start event (a kind weigh never
# opens), of config init (where dune records its environment and argv), of
# other kinds; a time slot that is not all digits is refused; and every
# weigh run here, refusals included, must print none of it. Each entry also
# gets a twin that decodes it and prints what it decoded, which the canary
# check alone must reject, so an entry the fixture forgot to seed fails its
# twin.
CANARY=CANARY-5e1d-not-for-output
# KIND/NAME:FIELD, the fields the fixture seeds and the twins decode.
WEIGH_SECRET="process/finish:env process/finish:process_args process/finish:rusage
process/finish:pid process/finish:exit process/finish:categories process/finish:queued
process/finish:target_dirs process/finish:name
process/finish:stdout process/finish:stderr process/finish:error
process/start:prog process/start:dir process/start:env process/start:process_args
config/init:env config/init:argv config/init:build_dir config/init:root
config/init:initial_cwd config/init:version config/init:pid
log/info:message action/write-file:file"
python3 - "$TMP" "$CANARY" "$WEIGH_SECRET" <<'PY'
import sys
from pathlib import Path

root, canary, secret = Path(sys.argv[1]), sys.argv[2], sys.argv[3].split()


def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s


def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'


def seeded(kind, written):
    # KIND's inventory fields, other than those the caller WRITTEN itself, each a
    # list where dune writes a list.
    out = []
    for entry in secret:
        k, field = entry.split(':')
        if k != kind or field in written:
            continue
        value = {'env': ['FAKE_API_TOKEN=' + canary, 'HOME=/home/' + canary],
                 'argv': ['dune', 'build', '--x=' + canary],
                 'categories': [canary], 'target_files': [canary], 'target_dirs': [canary]}.get(field, canary)
        out.append([field, value])
    return out


def finish(prog, directory, seconds, targets=None, args=(), time=None, cpu=None, rusage=True):
    # SECONDS of CPU, user and system together, in the rusage beside a canary
    # maxrss; the wall-clock duration is three times that, as on a loaded
    # runner, so a weigh crediting wall-clock reads every unit 3x. TARGETS
    # None: a process the trace records no target_files for.
    time = time or ['1', str(int(seconds * 3e9))]
    cpu = cpu or [str(int(seconds * 0.75e9)), str(int(seconds * 0.25e9))]
    own = [] if targets is None else [['target_files', list(targets)]]
    if rusage:
        own.append(['rusage', [['user_cpu_time', cpu[0]], ['maxrss', canary], ['system_cpu_time', cpu[1]]]])
    return sexp(['process', 'finish', list(time), ['process_args', list(args) + ['--token=' + canary]],
                 ['prog', prog], ['dir', directory]] + own
                + seeded('process/finish', ('process_args', 'rusage', 'target_files')))


def start(prog):
    return sexp(['process', 'start', '1', ['prog', canary + prog], ['dir', canary],
                 ['process_args', ['--token=' + canary]]]
                + seeded('process/start', ('prog', 'dir', 'process_args')))


b = '/x/_build/default/'
ops = '_build/default/test/operations/'


def act(alias, d='test/operations'):
    # An alias action's target, as dune names it.
    return '_build/.actions/default/%s/%s-0123456789abcdef0123456789abcdef' % (d, alias)


events = [
    sexp(['config', 'init', '1', ['build_dir', '_build/' + canary]] + seeded('config/init', ('build_dir',))),
    sexp(['log', 'info', '1', ['message', canary]]),
    start('ocamlopt.opt'),
    finish('/opt/bin/ocamlopt.opt', b + 'lib', 50, ['_build/default/lib/a.cmx', '_build/default/lib/a.o']),
    finish('/usr/bin/cc', b + 'lib', 40, ['_build/default/lib/a_stubs.o']),
    finish('cc_march_census.exe', b + 'test/operations', 6, [ops + 'cc_march_census.exe.output']),
    finish('./online_softmax.exe', b + 'test/operations', 7, [ops + 'online_softmax.exe.output']),
    finish('./online_softmax.exe', b + 'test/operations', 3, [ops + 'online_softmax_fast_math.actual'],
           ['--ocannl_backend=cc', '--ocannl_cc_backend_fast_math=true']),
    finish('./online_softmax.exe', '/x/_build/.sandbox/0123abcd/default/test/operations', 5,
           [ops + 'online_softmax_fp_licences.actual']),
    finish('./sweep_harness_driver.exe', b + 'test/operations', 9, [act('runtest-sweep_components')]),
    finish('.tut.inline-tests/inline-test-runner.exe', '/x/_build/.sandbox/4567cdef/default/test/operations', 4,
           [act('runtest-operations_tutorials')]),
    finish('.tut.inline-tests/inline-test-runner.exe', b + 'test/operations', 4, [act('runtest-operations_tutorials')]),
    finish('./cc_temp_ownership.exe', b + 'test/operations', 2, [ops + 'cc_temp_ownership-success.actual']),
    finish('./smoke.exe', b + 'test/operations', 1, [act('bin-smoke')]),
    finish('./helper.exe', b + 'test/operations', 1, ['_build/default/test/config/generated']),
    finish('./untargeted.exe', b + 'test/operations', 1),
    finish('./einsum_test.exe', b + 'test/einsum', 4, ['_build/default/test/einsum/einsum_test.exe.output']),
    finish('placement_store.exe', b + 'test/operations/sub', 0.4, [act('runtest-placement_store', 'test/operations/sub')]),
    sexp(['action', 'write-file', ['1', '2'], ['file', canary], ['size', '3']]),
]
(root / 'trace').write_bytes(b''.join(events))
# A duration that is not all digits, and a time slot that is no `(start
# duration)`: refused by offset, quoting none of either.
(root / 'bad-time-trace').write_bytes(finish('x.exe', b + 'lib', 1, time=['1', canary]))
(root / 'bare-time-trace').write_bytes(finish('x.exe', b + 'lib', 1, time=[canary]))
# A CPU time that is not all digits: refused the same way.
(root / 'bad-cpu-trace').write_bytes(finish('x.exe', b + 'lib', 1, cpu=['1', canary]))
# A process with no rusage weighs nothing, and says so.
(root / 'no-rusage-trace').write_bytes(finish('x.exe', b + 'test/operations', 5, [ops + 'x.exe.output'])
                                       + finish('y.exe', b + 'test/operations', 2, [ops + 'y.exe.output'],
                                                rusage=False))
# Cut short inside a finish event, after the canary fields began.
(root / 'cut-trace').write_bytes(events[0] + events[3][:-20])
PY
canary_free() { # FILE...: no file carries the canary
  ! grep -qF -- "$CANARY" "$@"
}
subject weigh "$TMP/trace" >"$TMP/weigh.out" 2>"$TMP/weigh.err"
weigh_rc=$?
printf '%s\n' 'sweep_components 9' 'operations_tutorials 8' 'online_softmax 7' '@rest 6' 'cc_march_census 6' \
  'online_softmax_fp_licences 5' 'online_softmax_fast_math 3' 'cc_temp_ownership-success 2' 'untargeted 1' \
  >"$TMP/weigh.expected"
diff "$TMP/weigh.expected" "$TMP/weigh.out" >"$TMP/weigh.diff"
report $? "weigh credits each unit the CPU seconds of the processes its targets name" \
  "exit $weigh_rc; $(cat "$TMP/weigh.diff" "$TMP/weigh.err")"
subject weigh --wall "$TMP/trace" >"$TMP/weigh-wall.out" 2>"$TMP/weigh-wall.err"
printf '%s\n' 'sweep_components 27' 'operations_tutorials 24' 'online_softmax 21' '@rest 19' 'cc_march_census 18' \
  'online_softmax_fp_licences 15' 'online_softmax_fast_math 9' 'cc_temp_ownership-success 6' 'untargeted 3' \
  >"$TMP/weigh-wall.expected"
diff "$TMP/weigh-wall.expected" "$TMP/weigh-wall.out" >"$TMP/weigh-wall.diff" && [ ! -s "$TMP/weigh-wall.err" ] &&
  canary_free "$TMP/weigh-wall.out"
report $? "weigh --wall credits the same units their wall-clock durations" \
  "$(cat "$TMP/weigh-wall.diff" "$TMP/weigh-wall.err")"
subject weigh "$TMP/no-rusage-trace" >"$TMP/no-rusage.out" 2>"$TMP/no-rusage.err"
[ "$?" = 0 ] && [ "$(cat "$TMP/no-rusage.out")" = 'x 5
y 0' ] && grep -q '1 finished processes carry no rusage; each weighs 0' "$TMP/no-rusage.err"
report $? "weigh credits a process without rusage nothing, with a warning" \
  "$(cat "$TMP/no-rusage.out" "$TMP/no-rusage.err")"
[ "$weigh_rc" = 0 ] && [ ! -s "$TMP/weigh.err" ] && canary_free "$TMP/weigh.out" "$TMP/weigh.err"
report $? "weigh decodes none of the canary fields: exit 0, empty stderr, no canary" \
  "exit $weigh_rc; see $TMP/weigh.out $TMP/weigh.err"
printf 'not a trace' >"$TMP/garbage"
harness_rejected 1 'not a canonical-S-expression dune trace' "$TMP/weigh.log" subject weigh "$TMP/garbage"
report $? "weigh refuses a file that is not a dune trace" "$(cat "$TMP/weigh.log")"
printf '(6:config4:init1:1)' >"$TMP/empty-trace"
harness_rejected 1 'no finished processes' "$TMP/weigh.log" subject weigh "$TMP/empty-trace"
report $? "weigh refuses a trace that recorded no process" "$(cat "$TMP/weigh.log")"
for t in bad-time-trace bare-time-trace bad-cpu-trace cut-trace; do
  harness_rejected 1 'malformed or cut short at byte [0-9]+$' "$TMP/weigh.log" subject weigh "$TMP/$t" &&
    canary_free "$TMP/weigh.log"
  report $? "weigh refuses $t by offset, quoting none of its bytes" "$(cat "$TMP/weigh.log")"
done

# The twins: one per inventory entry, the subject with that field added to
# what it decodes and every decoded event printed on each row. Each must
# change the subject (else the control is vacuous) and be rejected by the
# canary alone. They run from the fixture repository's tools/, beside the
# credential deny-list the subject sources.
weigh_twin() { # ENTRY -> 0 when the twin ran and leaked the canary
  local entry=$1 kind field twin="$repo/tools/weigh-twin.sh"
  kind=${entry%%:*} field=${entry#*:}
  python3 - "$SRC" "$twin" "${kind%%/*}" "${kind#*/}" "$field" <<'PY' || return 3
import sys
src, dst, cat, name, field = sys.argv[1:6]
text = open(src).read()
pairs = [("TEXT = {FINISH: {b'prog', b'dir', b'target_files'}}\n",
          "TEXT = {FINISH: {b'prog', b'dir', b'target_files'}}\nTEXT.setdefault((b'%s', b'%s'), set()).add(b'%s')\n"
          % (cat, name, field)),
         ("    print('%s %.0f' % (name, seconds))\n", "    print('%s %.0f %s' % (name, seconds, events))\n")]
for old, new in pairs:
    if text.count(old) != 1:
        sys.exit('twin: expected one occurrence of %r' % old)
    text = text.replace(old, new)
open(dst, 'w').write(text)
PY
  bash "$twin" weigh "$TMP/trace" >"$TMP/twin.out" 2>"$TMP/twin.err" || return 4
  ! canary_free "$TMP/twin.out" "$TMP/twin.err"
}
for entry in $WEIGH_SECRET; do
  label="negative control: a weigh that decodes and prints $entry leaks the canary"
  weigh_twin "$entry"
  case $? in
    0) report 0 "$label" ;;
    1) report 1 "$label" "the canary check accepted the twin: is $entry seeded? see $TMP/twin.out" ;;
    *) report 1 "$label" "the twin did not run; see $TMP/twin.err" ;;
  esac
done
rm -f "$repo/tools/weigh-twin.sh"

# The twin crediting wall-clock, the unit gh-ocannl-1503 replaced: the
# fixture's durations are three times its CPU, so every unit reads 3x.
path=$(mutant weigh-wall-clock "/fields.get\\('cpu_ns', 0\\)/ { sub(/fields.get\\('cpu_ns', 0\\)/, \"fields.get('ns', 0)\") } { print }")
if [ -z "$path" ] || cmp -s "$path" "$SRC"; then
  report 1 "negative control: weigh crediting wall-clock" "could not build the mutant"
else
  cp "$path" "$repo/tools/weigh-wall.sh"
  bash "$repo/tools/weigh-wall.sh" weigh "$TMP/trace" >"$TMP/wall.out" 2>"$TMP/wall.err"
  rm -f "$repo/tools/weigh-wall.sh"
  ! diff -q "$TMP/weigh.expected" "$TMP/wall.out" >/dev/null && grep -qx 'sweep_components 27' "$TMP/wall.out"
  report $? "negative control: weigh crediting wall-clock" "see $TMP/wall.out $TMP/wall.err"
fi

# The twin keyed by executable, the scheme gh-ocannl-1503 replaced: the same
# trace and oracle must reject it (online_softmax.exe's three tests in one
# unit, the driver and the inline-test runner under their own names).
path=$(mutant weigh-by-executable \
  '/cost\[unit_of\(fields, prog\)\]/ { sub(/unit_of\(fields, prog\)/, "(\"@rest\" if in_context(fields.get(\"dir\") or \"\") != split_dir else prog.removesuffix(\".exe\"))") } { print }')
if [ -z "$path" ] || cmp -s "$path" "$SRC"; then
  report 1 "negative control: weigh keyed by executable" "could not build the mutant"
else
  # Beside the credential deny-list the subject sources, as the canary twins run.
  cp "$path" "$repo/tools/weigh-by-exe.sh"
  bash "$repo/tools/weigh-by-exe.sh" weigh "$TMP/trace" >"$TMP/by-exe.out" 2>"$TMP/by-exe.err"
  rm -f "$repo/tools/weigh-by-exe.sh"
  ! diff -q "$TMP/weigh.expected" "$TMP/by-exe.out" >/dev/null &&
    grep -qx 'online_softmax 15' "$TMP/by-exe.out"
  report $? "negative control: weigh keyed by executable" "see $TMP/by-exe.out $TMP/by-exe.err"
fi

# merge-weights: CI runs' per-shard weigh lines -> the WEIGHTS block. The
# lines sit wherever `gh run view --log` puts them (job and step columns, a
# timestamp); a unit in several shards -- a shared dependency, which moving its
# alias does not move -- is left out of the table, in every run once any run
# saw it shared; @rest is the largest shard's beyond the others' mean (their
# @rest is the overhead every shard pays); runs are averaged per unit over
# the runs that measured it; units under 2 s are left to the default, counted
# on stderr.
{
  printf 'Build (ubuntu, main, 1/2)\tWeigh\t2026-10-21T10:00:00.1234567Z ci-shard-weight 1/3 heavy 100\n'
  printf '2026-10-21T10:00:00.2Z ci-shard-weight 1/3 @rest 30\n'
  printf 'ci-shard-weight 2/3 @rest 330\n'
  printf 'ci-shard-weight 2/3 light 50\n'
  printf 'ci-shard-weight 3/3 @rest 10\n'
  printf 'ci-shard-weight 3/3 tiny 1\n'
  printf 'ci-shard-weight 3/3 driver 3\n'
  printf 'ci-shard-weight 1/3 driver 4.4\n'
  printf 'unrelated ci-shard-weight-ish text\n'
} >"$TMP/run.log"
subject merge-weights "$TMP/run.log" >"$TMP/merge.out" 2>"$TMP/merge.err"
merge_rc=$?
ok=0
[ "$merge_rc" = 0 ] || ok=1
[ "$(cat "$TMP/merge.out")" = '@rest 310
heavy 100
light 50' ] || ok=1
grep -q 'driver ran in 2 shards, a shared dependency: fixed overhead of up to 4s in each, left out of the table' \
  "$TMP/merge.err" || ok=1
grep -q '3 shards; @rest overhead per shard 20s$' "$TMP/merge.err" || ok=1
grep -q '1 run(s); 1 units under 2s, mean 1.0s' "$TMP/merge.err" || ok=1
report $ok "merge-weights leaves a shared dependency out and nets @rest of the per-shard overhead" \
  "exit $merge_rc; $(cat "$TMP/merge.out" "$TMP/merge.err")"
[ "$(subject merge-weights <"$TMP/run.log" 2>/dev/null)" = "$(cat "$TMP/merge.out")" ]
report $? "merge-weights reads stdin without a LOG"
# A second run, of another deal and N: heavy and light measured again, a
# newcomer measured only here, and driver alone in one shard -- still left
# out, since the first run saw it shared.
{
  printf 'ci-shard-weight 1/2 heavy 140\n'
  printf 'ci-shard-weight 1/2 @rest 50\n'
  printf 'ci-shard-weight 2/2 light 30\n'
  printf 'ci-shard-weight 2/2 newcomer 12\n'
  printf 'ci-shard-weight 2/2 driver 9\n'
  printf 'ci-shard-weight 2/2 @rest 10\n'
} >"$TMP/run2.log"
subject merge-weights "$TMP/run.log" - <"$TMP/run2.log" >"$TMP/merge2.out" 2>"$TMP/merge2.err"
merge_rc=$?
ok=0
[ "$merge_rc" = 0 ] || ok=1
[ "$(cat "$TMP/merge2.out")" = '@rest 175
heavy 120
light 40
newcomer 12' ] || ok=1
grep -q 'run 1: 3 shards; @rest overhead per shard 20s$' "$TMP/merge2.err" || ok=1
grep -q 'run 2: 2 shards; @rest overhead per shard 10s$' "$TMP/merge2.err" || ok=1
grep -q 'run 1: driver ran in 2 shards, a shared dependency' "$TMP/merge2.err" || ok=1
grep -q '2 run(s); 1 units under 2s' "$TMP/merge2.err" || ok=1
report $ok "merge-weights averages runs per unit, over the runs that measured it" \
  "exit $merge_rc; $(cat "$TMP/merge2.out" "$TMP/merge2.err")"
refused "merge-weights: stdin named twice" 'stdin named twice' subject merge-weights - - <"$TMP/run.log"
grep -v '^ci-shard-weight 2/2' "$TMP/run2.log" >"$TMP/run2-missing.log"
refused "merge-weights: a later run missing a shard" 'run 2: no lines for shard\(s\) 2/2 of 2' \
  subject merge-weights "$TMP/run.log" "$TMP/run2-missing.log"
grep -v '^ci-shard-weight 3/3' "$TMP/run.log" >"$TMP/run-missing.log"
refused "merge-weights: a shard missing from the run" 'no lines for shard\(s\) 3/3 of 3' \
  subject merge-weights "$TMP/run-missing.log"
{ cat "$TMP/run.log"; printf 'ci-shard-weight 1/2 heavy 9\n'; } >"$TMP/run-mixed.log"
refused "merge-weights: shards of two Ns" 'mix shard counts' subject merge-weights "$TMP/run-mixed.log"
{ cat "$TMP/run.log"; printf 'ci-shard-weight 2/3 light 49\n'; } >"$TMP/run-twice.log"
refused "merge-weights: a unit twice in one shard" 'names light twice' subject merge-weights "$TMP/run-twice.log"
printf 'no weights here\n' >"$TMP/run-none.log"
refused "merge-weights: a log with no weigh lines" 'no ci-shard-weight lines' subject merge-weights "$TMP/run-none.log"
refused "merge-weights: a missing log" 'usage' subject merge-weights "$TMP/no-such-log"
refused "merge-weights: a missing second log" 'usage' subject merge-weights "$TMP/run.log" "$TMP/no-such-log"
# merge-weights --wall reads the ci-shard-wall lines, and only those.
{ printf 'ci-shard-wall 1/2 heavy 300\n'; printf 'ci-shard-wall 2/2 light 80\n'; cat "$TMP/run.log"; } >"$TMP/wall.log"
[ "$(subject merge-weights --wall "$TMP/wall.log" 2>/dev/null)" = 'heavy 300
light 80' ]
report $? "merge-weights --wall merges the ci-shard-wall lines"
refused "merge-weights --wall: a log without wall lines" 'no ci-shard-wall lines' subject merge-weights --wall "$TMP/run.log"
# A rule output named after no unit (as backend-0-0.log.actual is for
# runtest-micrograd_demo_logging) goes to the one unit whose stanza in the
# split directory's dune file names it; a key with no unit and no owner is
# reported loudly and kept.
printf '%s\n' '(rule (target "out-0.log.actual") (action (with-stdout-to %{target} (run ./a.exe))))' \
  '(rule (alias runtest-a) (action (diff a.expected "out-0.log.actual")))' >"$repo/$ops/dune"
printf 'ci-shard-weight 1/1 out-0 7\nci-shard-weight 1/1 a 2\nci-shard-weight 1/1 phantom 5\n' >"$TMP/fold.log"
subject merge-weights "$TMP/fold.log" >"$TMP/fold.out" 2>"$TMP/fold.err"
[ "$(cat "$TMP/fold.out")" = 'a 9
phantom 5' ] && grep -q 'out-0 is a target of a; its weight goes there' "$TMP/fold.err" &&
  grep -q 'WEIGHT NAMES NO UNIT: phantom 5' "$TMP/fold.err"
report $? "merge-weights credits a rule output to the unit that owns it, and flags a key with no unit" \
  "$(cat "$TMP/fold.out" "$TMP/fold.err")"
: >"$repo/$ops/dune"

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
  '/if d == split_dir and a in \(.runtest., .default.\):/ { match($0, /^ */); print substr($0, 1, RLENGTH) "if False:"; next } { print }' \
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
  "/share = 'all' if a == 'default' else a/ { match(\$0, /^ */); print substr(\$0, 1, RLENGTH) \"share = a\"; next } { print }" \
  "@@default"

# The `train` suite. Its units are the aggregates' members, read from the
# dune files: the fake dune is never asked (pointed at a missing command
# below, so a train run that called it would fail).
train_expected="@@sub2/train-sub
@@test/training/train-b
@@test/training/train-c
@@test/training/train-gate
@@test/training/train-heavy"
train_partition_holds() { # LOG
  local n k all
  : >"$1"
  for n in 1 2 3 4 5; do
    all=
    for ((k = 1; k <= n; k++)); do
      out=$(OCANNL_TOOL_CI_SHARD_DUNE="$TMP/no-such-dune" subject targets "$k/$n" train 2>>"$1") ||
        { echo "targets $k/$n train failed" >>"$1"; return 1; }
      [ -n "$out" ] || { echo "train shard $k/$n is empty" >>"$1"; return 1; }
      all="$all$out"$'\n'
    done
    if [ "$(printf '%s' "$all" | sort)" != "$(printf '%s\n' "$train_expected" | sort)" ]; then
      { echo "N=$n train targets differ from the members:"; printf '%s' "$all" | sort; } >>"$1"
      return 1
    fi
  done
}
train_partition_holds "$TMP/train-partition.log"
report $? "train: for N=1..5 the shards partition the aggregates' members exactly, without dune" \
  "see $TMP/train-partition.log"
train1=$(subject targets 1/2 train 2>"$TMP/train-deal.err")
train2=$(subject targets 2/2 train 2>/dev/null)
[ "$train1" = "@@test/training/train-heavy" ]
report $? "train: the heaviest member is dealt first, alone on shard 1/2" "got: $train1"
[ "$(printf '%s\n' "$train2" | tr '\n' ' ')" = "@@sub2/train-sub @@test/training/train-b @@test/training/train-c @@test/training/train-gate " ]
report $? "train: the rest go to the lighter shard, equal weights by name" "got: $train2"
grep -q 'weight for test/training/train-ghost, which is no unit any more' "$TMP/train-deal.err"
report $? "train: a weight naming no member warns on stderr" "see $TMP/train-deal.err"
subject plan 2 train 2>/dev/null | grep -q '^shard 2/2: 4 units, load 4 '
report $? "train: plan reports each shard's members and load"
[ "$(subject targets 1/1 2>/dev/null)" = "$(subject targets 1/1 main 2>/dev/null)" ]
report $? "the suite defaults to main"

train_refused() { # LABEL PATTERN DUNE-TEXT [FILE]
  printf '%s' "$3" >"${4:-$repo/test/training/dune}"
  refused "train: $1" "$2" subject targets 1/2 train
  reset_train_dune
}
train_refused "an aggregate dep that is no alias" 'not an \(alias train-<name>\)' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (deps (alias train-a) data.txt))'
train_refused "an aggregate dep on an alias_rec" 'not an \(alias train-<name>\)' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (deps (alias_rec train-a)))'
train_refused "an aggregate member nothing defines" 'names train-missing, which no stanza there defines' \
  '(rule (alias train-a) (deps (alias train-missing)) (action (progn)))
(alias (name train) (deps (alias train-a) (alias train-missing)))'
train_refused "an aggregate with an action" 'has action' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (deps (alias train-a)) (action (progn)))'
train_refused "an aggregate with enabled_if" 'has enabled_if' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (enabled_if false) (deps (alias train-a)))'
train_refused "a rule attached to train itself" 'attaches a rule stanza to the train alias itself' \
  '(rule (alias train-a) (action (progn)))
(rule (aliases train runtest) (action (progn)))
(alias (name train) (deps (alias train-a)))'
train_refused "two train aggregates in one directory" 'defines the train alias 2 times' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (deps (alias train-a)))
(alias (name train) (deps (alias train-a)))'
train_refused "a member named twice" 'names train-a twice' \
  '(rule (alias train-a) (action (progn)))
(alias (name train) (deps (alias train-a) (alias train-a)))'
train_refused "an include it cannot read" 'include stanza' \
  '(include train.inc)'
train_refused "a block comment it does not skip" 'comment' \
  '#| (alias (name train) (deps (alias train-a))) |#'
root_dune=$(cat "$repo/dune")
: >"$repo/dune"
train_refused "no aggregate anywhere" 'no tracked dune file defines a train aggregate' ''
printf '%s\n' "$root_dune" >"$repo/dune"
refused "an unknown suite" "unknown suite 'slow'" subject targets 1/2 slow
refused "plan with an unknown suite" "unknown suite 'slow'" subject plan 2 slow
train_partition_holds "$TMP/train-partition.log"
report $? "train: the fixture is restored after the refusals" "see $TMP/train-partition.log"

# The twin for the train reader: skip `subdir` stanzas instead of descending
# into them, and the root's aggregate member disappears from every shard --
# which the partition oracle must report as a member mismatch.
path=$(mutant skip-subdir '/train_stanzas\(stanza\[2:\]/ { match($0, /^ */); print substr($0, 1, RLENGTH) "pass"; next } { print }') &&
  install_subject "$path" && ! grep -q 'train_stanzas(stanza\[2:\]' "$repo/tools/ci-shard.sh"
twin_rc=$?
if [ "$twin_rc" != 0 ]; then
  report 1 "negative control: train subdir stanzas skipped" "could not build the mutant"
elif train_partition_holds "$TMP/train-twin.log"; then
  report 1 "negative control: train subdir stanzas skipped" "the train partition oracle accepted the mutant"
else
  grep -q 'train targets differ from the members' "$TMP/train-twin.log" &&
    ! grep -qxF '@@sub2/train-sub' "$TMP/train-twin.log"
  report $? "negative control: train subdir stanzas skipped" "rejected for an unrelated reason; see $TMP/train-twin.log"
fi
install_subject "$SRC"

finish

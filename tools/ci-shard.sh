#!/usr/bin/env bash
# Which dune targets does one shard of CI's sharded ubuntu `main` suite build?
#
# The suite is `dune build @default @runtest @bin-smoke` (SUITE_ALIASES below,
# the same three ci.yml's unsharded legs name). On the 4-vCPU ubuntu runner it
# ran 13-26min in that one step, three quarters of it executing
# test/operations: that one directory holds ~320 of the repository's ~420
# per-test aliases, so a split by directory cannot balance. The shards are cut
# INSIDE it instead, from units dune itself lists:
#
#   - every `runtest-<name>` alias of test/operations, one unit each, built
#     non-recursively (`@@`), so a same-named alias in a subdirectory is not
#     dragged along. Dune generates one per `(test)` stanza and per inline-test
#     library; a rule-based test carries a hand-written one that the
#     directory's `runtest` aggregates (env_var_deps enforces both halves), so
#     together they are that directory's `runtest`. Its share of `default`
#     (its `all`, below) is left out -- it would run every test there -- and
#     needs no unit of its own: every test output and the executable behind
#     it is already a dependency of some test's alias. What is not is an
#     executable no `runtest-<name>` is named after (the `@slow` tests' and
#     a few helpers): `@rest` links those by name, from `dune show targets`.
#   - one `@rest` unit: each suite alias, non-recursively, in every OTHER
#     directory holding a tracked dune file -- whatever `dune show aliases`
#     says that directory defines. A new directory joins it the day it lands.
#     `default` is the exception: no dune file here defines it, and dune's
#     implicit one is `(alias_rec all)` -- RECURSIVE even as `@@dir/default`,
#     so `@@./default` alone would build the whole tree's `all`, which runs
#     every `(test)` (its stdout capture is a file target). Its per-directory
#     share is `@@dir/all`. A dune file that defines `default` itself breaks
#     that equation, so the script refuses one.
#
# The units are dealt to N shards longest-first onto the least-loaded shard
# (LPT), by the WEIGHTS table below. Weights are needed, not a nicety: the
# costs are heavy-tailed (three tests carried a third of test/operations'
# CPU), and an unweighted round-robin by name measured 63/37 between two
# shards where LPT measures 50/50. A unit the table does not list weighs
# DEFAULT_WEIGHT. Stale entries degrade only the balance, never the
# coverage, so they warn rather than fail; refresh the table with `weigh`.
#
# Usage:
#   tools/ci-shard.sh targets K/N     # shard K's dune targets, one per line
#   tools/ci-shard.sh plan N          # each shard's units and estimated load
#   tools/ci-shard.sh aliases         # the suite aliases being sharded
#   tools/ci-shard.sh weigh TRACE     # per-unit CPU seconds from a dune trace
#
# `targets` and `plan` run `dune show aliases` and `dune show targets` (as
# $OCANNL_TOOL_CI_SHARD_DUNE when set, a hook for tools/test-ci-shard.sh), from
# the repository root. Both print on STDERR and exit 0 even for a directory
# they do not know,
# so the listing is validated instead of trusted: an `Error` line, a missing
# directory block or a test/operations without its per-test aliases refuses
# (exit 2) rather than printing a shard that silently covers less -- or more:
# a shard that reaches a recursive alias runs the other shards' tests too.
#
# `weigh` reads the trace of a full suite run --
#   dune build @default @runtest @bin-smoke --trace-file=TRACE
# -- (dune >= 3.24's canonical-S-expression trace; a fresh build directory, or
# cached rules record no process) and prints a replacement WEIGHTS block:
# CPU seconds per test/operations executable, keyed by the alias of the same
# name, plus `@rest` for every non-compiler process elsewhere. An executable
# run with `--ocannl_cc_backend_fast_math=true` is credited to its
# `<name>_fast_math` twin. An executable whose alias is named differently
# (sweep_harness_driver runs for runtest-sweep_harness) keeps its own name: map
# it by hand, or `plan` warns about it as a stale entry. Keep the units above a
# few seconds; the rest weigh DEFAULT_WEIGHT anyway.
# CPU, not wall-clock: the ubuntu runner is CPU-bound, while a
# workstation's wall-clock is dominated by whatever else it is doing (macOS
# XProtect stalls on fresh executables, other sessions' suites).

set -euo pipefail

SUITE_ALIASES="default runtest bin-smoke"
SPLIT_DIR=test/operations
DEFAULT_WEIGHT=1

# CPU seconds, measured 2026-10-01 on an M-series Mac (DUNE_CACHE=disabled,
# fresh build directory) with `weigh`. Units only need to be consistent with
# each other and with DEFAULT_WEIGHT; the long tail of ~300 aliases averaged
# ~0.6s each.
WEIGHTS='
autotune_candidate_release 118
placement_store 114
autotune_measured_refusal 103
@rest 50
sweep_harness 50
flip_abandonment 40
autotune_smoke 40
online_softmax_block_fast_math 39
online_softmax_block 38
schedule_conv_gemm 37
schedule_cache_numerics 32
online_softmax_fast_math 26
online_softmax 25
gpu_fission_mapping 17
schedule_epilogue_fusion 17
cost_model_selection 16
autotune_serial_baseline 15
autotune_split_reduce 13
reduction_forms 13
autotune_callback_release 13
schedule_strided_1x1 11
operations_tutorials 10
accum_width 8
online_softmax_block_mma 8
cc_march_census 8
env_var_deps 7
autotune_bound_pruning 6
dead_export_scan 6
schedule_contraction_nest 6
'

die() {
  echo "ci-shard.sh: $*" >&2
  exit 2
}

usage() {
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

root=$(cd "$(dirname "$0")/.." && pwd)
# The credential deny-list tools/test-run.sh applies (gh-ocannl-1280): dune records every spawned
# process's environment in `_build/trace.csexp`, so the `dune show` calls below run in a subshell
# that scrubs first, and fail rather than run when the scrub cannot complete.
credential_env=$(dirname "$0")/credential-env.sh
[ -r "$credential_env" ] || die "cannot read $credential_env"
# shellcheck source=credential-env.sh
. "$credential_env"
scrub_credentials() {
  eval "$(credential_env_scrub_text)" || {
    echo "ci-shard: cannot remove credential variables:$credential_env_left" >&2
    return 1
  }
}

# Every directory holding a tracked dune file, `.` for the root. Tracked, not
# found: an untracked checkout under .claude/worktrees/ is not this tree.
dune_dirs() {
  git -C "$root" ls-files -- dune '*/dune' | while IFS= read -r f; do
    d=$(dirname "$f")
    printf '%s\n' "$d"
  done | sort -u
}

# `dune show aliases DIR...` for every dune directory and `dune show targets`
# for the split one, stderr included (where dune prints both listings).
alias_listing() {
  local dune=${OCANNL_TOOL_CI_SHARD_DUNE:-dune}
  local dirs
  dirs=$(dune_dirs)
  [ -n "$dirs" ] || die "no tracked dune files under $root"
  # shellcheck disable=SC2086 # one argument per directory; none holds a space
  (cd "$root" && scrub_credentials && $dune show aliases $dirs 2>&1) || die "dune show aliases failed"
  printf '\n--targets--\n'
  (cd "$root" && scrub_credentials && $dune show targets "$SPLIT_DIR" 2>&1) || die "dune show targets failed"
  printf '\n--dirs--\n%s\n' "$dirs"
}

# The sharding itself: MODE (targets|plan), K, N, then alias_listing's output.
# (The listing is an argument because the heredoc below is python's stdin.)
shard() {
  python3 - "$@" "$SUITE_ALIASES" "$SPLIT_DIR" "$DEFAULT_WEIGHT" "$WEIGHTS" "$root" <<'PY'
import re
import sys

mode, k, n, text, suite, split_dir, default_weight, table, root = sys.argv[1:10]
k, n, default_weight = int(k), int(n), float(default_weight)
suite = suite.split()
listing, _, dirs = text.partition('\n--dirs--\n')
listing, _, targets = listing.partition('\n--targets--\n')
dirs = dirs.split()


def die(message):
    print('ci-shard.sh: ' + message, file=sys.stderr)
    sys.exit(2)


for line in listing.splitlines():
    if line.startswith('Error'):
        die('dune show aliases refused:\n' + listing)
for line in targets.splitlines():
    if line.startswith('Error'):
        die('dune show targets refused:\n' + targets)

# One block per directory: a `DIR:` header (dune omits it when asked about a
# single directory, which this never does), then one alias per line. Anything
# before the first header is dune's own chatter (a warning shares the stream):
# passed through, not parsed.
blocks, current = {}, None
for line in listing.splitlines():
    if not line.strip():
        continue
    if line.endswith(':') and line[:-1] in dirs:
        current = blocks.setdefault(line[:-1], set())
    elif current is None:
        print(line, file=sys.stderr)
    else:
        current.add(line.strip())
for d in dirs:
    # `default` is defined in every directory dune sees: its absence means
    # dune did not report on the directory at all.
    if 'default' not in blocks.get(d, ()):
        die('dune show aliases reported nothing for %s:\n%s' % (d, listing))
if split_dir not in blocks:
    die('%s holds no tracked dune file' % split_dir)
split_aliases = blocks[split_dir]
tests = sorted(a[len('runtest-'):] for a in split_aliases if a.startswith('runtest-'))
if 'runtest' not in split_aliases or not tests:
    die('%s lists no runtest-<name> aliases; refusing to shard it' % split_dir)

# A dune file defining `default` itself (the alias stanza's name, an alias or
# aliases field): comments stripped first, so this prose's own kind is safe.
explicit = re.compile(r'\(\s*name\s+default\s*\)|\(\s*alias(?:es)?\b[^()]*\bdefault\b')
for d in dirs:
    path = root + '/' + ('' if d == '.' else d + '/') + 'dune'
    code = re.sub(r';[^\n]*', '', open(path).read())
    if explicit.search(code):
        die('%s defines the default alias; @@dir/all no longer stands for its '
            'share of the suite, so teach this script what does' % path)

rest = []
for d in sorted(blocks):
    prefix = '' if d == '.' else d + '/'
    for a in suite:
        # The split directory's runtest and default ARE its per-test aliases.
        if d == split_dir and a in ('runtest', 'default'):
            continue
        # Implicit default is (alias_rec all): its non-recursive share is all.
        share = 'all' if a == 'default' else a
        if a in blocks[d]:
            if share not in blocks[d]:
                die('dune lists no %s alias in %s' % (share, d))
            rest.append('@@' + prefix + share)
executables = sorted(t[:-len('.exe')] for t in targets.split() if t.endswith('.exe'))
if not executables:
    die('dune show targets lists no executables in %s:\n%s' % (split_dir, targets))
rest += ['%s/%s.exe' % (split_dir, e) for e in executables if e not in tests]

weights = {}
for line in table.splitlines():
    if line.strip():
        name, value = line.split()
        weights[name] = float(value)
units = {'@rest': rest}
units.update({t: ['@@%s/runtest-%s' % (split_dir, t)] for t in tests})
for name in sorted(set(weights) - set(units)):
    print('ci-shard.sh: warning: weight for %s, which is no unit any more; '
          'refresh WEIGHTS with `weigh`' % name, file=sys.stderr)

def weight(u):
    return weights.get(u, default_weight)

loads = [0.0] * n
members = [[] for _ in range(n)]
for u in sorted(units, key=lambda u: (-weight(u), u)):
    i = loads.index(min(loads))
    loads[i] += weight(u)
    members[i].append(u)
assert sorted(u for m in members for u in m) == sorted(units), 'units lost or duplicated'

if mode == 'targets':
    for u in members[k - 1]:
        for target in units[u]:
            print(target)
else:
    total = sum(loads)
    for i, m in enumerate(members):
        print('shard %d/%d: %d units, load %.0f (%.0f%%)' % (i + 1, n, len(m), loads[i], 100 * loads[i] / total))
        for u in m:
            print('  %-40s %g' % (u, weight(u)))
PY
}

weigh() {
  python3 - "$1" "$SPLIT_DIR" <<'PY'
import collections
import os
import re
import sys

path, split_dir = sys.argv[1:3]
data = open(path, 'rb').read()


def parse(b, i):
    # Canonical S-expressions: `(`, `)`, and length-prefixed atoms `N:bytes`.
    out = []
    while i < len(b):
        c = b[i:i + 1]
        if c == b'(':
            sub, i = parse(b, i + 1)
            out.append(sub)
        elif c == b')':
            return out, i + 1
        else:
            j = b.index(b':', i)
            size = int(b[i:j])
            out.append(b[j + 1:j + 1 + size].decode('utf8', 'replace'))
            i = j + 1 + size
    return out, i


try:
    events, _ = parse(data, 0)
except (ValueError, IndexError):
    sys.exit('ci-shard.sh: %s is not a canonical-S-expression dune trace (dune >= 3.24)' % path)
toolchain = {'ppx.exe', 'menhir', 'cc', 'gcc', 'clang', 'ar', 'as', 'ld'}
cost = collections.Counter()
seen = 0
for e in events:
    if not (isinstance(e, list) and e[:2] == ['process', 'finish']):
        continue
    seen += 1
    fields = {x[0]: x[1] for x in e[3:] if isinstance(x, list) and len(x) == 2 and isinstance(x[0], str)}
    prog = os.path.basename(fields.get('prog', ''))
    if prog.startswith('ocaml') or prog in toolchain:
        continue
    usage = dict(fields.get('rusage', []))
    cpu = (int(usage.get('user_cpu_time', 0)) + int(usage.get('system_cpu_time', 0))) / 1e9
    # The build context's root is the last `default` component: `_build/default`,
    # a sandbox's `.sandbox/<hash>/default`, or a --build-dir's `<dir>/default`.
    directory = re.sub(r'^.*/default(?:/|$)', '', fields.get('dir', '')) or '.'
    if directory != split_dir:
        cost['@rest'] += cpu
        continue
    name = prog.removesuffix('.exe')
    if '--ocannl_cc_backend_fast_math=true' in fields.get('process_args', []):
        name += '_fast_math'
    cost[name] += cpu
if not seen:
    sys.exit('ci-shard.sh: no finished processes in %s (cached build?)' % path)
for name, cpu in cost.most_common():
    print('%s %.0f' % (name, cpu))
PY
}

case ${1-} in
  targets)
    [ $# -eq 2 ] || die "usage: tools/ci-shard.sh targets K/N"
    [[ $2 =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]] || die "shard must be K/N, got '$2'"
    k=${BASH_REMATCH[1]} n=${BASH_REMATCH[2]}
    [ "$k" -le "$n" ] || die "shard $k of only $n"
    listing=$(alias_listing)
    shard targets "$k" "$n" "$listing"
    ;;
  plan)
    [ $# -eq 2 ] && [[ $2 =~ ^[1-9][0-9]*$ ]] || die "usage: tools/ci-shard.sh plan N"
    listing=$(alias_listing)
    shard plan 1 "$2" "$listing"
    ;;
  aliases)
    printf '%s\n' $SUITE_ALIASES
    ;;
  weigh)
    [ $# -eq 2 ] && [ -f "$2" ] || die "usage: tools/ci-shard.sh weigh TRACE"
    weigh "$2"
    ;;
  -h | --help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

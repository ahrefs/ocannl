#!/usr/bin/env bash
# Which dune targets does one shard of one of CI's sharded ubuntu suites build?
# Two suites are sharded: `main` (the default) and `train`.
#
# `main` is `dune build @default @runtest @bin-smoke` (SUITE_ALIASES below,
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
# `train` is `dune build @train`, the training-integration tier
# (test/training/dune says what it is). Its units are the members of every
# directory's `(alias (name train) (deps ...))` aggregate, one `@@dir/train-<name>`
# each -- read from the tracked dune files themselves, as S-expressions, since
# the aggregate is exactly what `@train` builds there and needs no dune to
# read: ci.yml and test/operations/ci_matrix.sh both run it where no toolchain
# is installed. The members serialize on the `ocannl_training_test` lock, so a
# shard's run time is the SUM of its members' and the deal below is what
# balances it. The reader refuses (exit 2) whatever would make its member list
# differ from what `@train` builds: an aggregate dep that is not `(alias
# train-<name>)` of an alias the same file defines, an aggregate with a field
# beyond name/deps/package, a rule attached to `train` directly, a
# comment form or an include it does not read.
#
# The units are dealt to N shards longest-first onto the least-loaded shard
# (LPT), by the suite's weight table below. Weights are needed, not a nicety: the
# costs are heavy-tailed (three tests carried a third of test/operations'
# CPU), and an unweighted round-robin by name measured 63/37 between two
# shards where LPT measures 50/50. A unit the table does not list weighs
# DEFAULT_WEIGHT. Stale entries degrade only the balance, never the
# coverage, so they warn rather than fail; refresh WEIGHTS with `weigh`, and
# TRAIN_WEIGHTS as its comment says.
#
# Usage:
#   tools/ci-shard.sh targets K/N [SUITE]  # shard K's dune targets, one per line
#   tools/ci-shard.sh plan N [SUITE]       # each shard's units and estimated load
#   tools/ci-shard.sh aliases              # the aliases the `main` suite shards
#   tools/ci-shard.sh weigh TRACE          # per-unit CPU seconds from a dune trace
# SUITE is `main` (the default) or `train`.
#
# For `main`, `targets` and `plan` run `dune show aliases` and `dune show targets` (as
# $OCANNL_TOOL_CI_SHARD_DUNE when set, a hook for tools/test-ci-shard.sh), from
# the repository root. Both print on STDERR and exit 0 even for a directory
# they do not know,
# so the listing is validated instead of trusted: an `Error` line, a missing
# directory block or a test/operations without its per-test aliases refuses
# (exit 2) rather than printing a shard that silently covers less -- or more:
# a shard that reaches a recursive alias runs the other shards' tests too.
# `train` runs no dune at all.
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
# It decodes only the fields it needs (SECURITY, above `weigh` below).
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

# Wall-clock seconds of each `train` member, keyed by its unit (the target
# without `@@`). Wall-clock, not CPU as above: the members serialize on the
# ocannl_training_test lock and each saturates every core through the cc
# backend's OpenMP pool, so a shard's run time is the sum of its members' wall
# times. Measured 2026-10-19 on minix-amd-linux (32 threads, cc) with
# `tools/test-run.sh fresh @test/training/train-<name>... -j 4`, which prints
# each member's wall time from the run's dune trace; rerun it on a Linux box
# to refresh, and check the deal against the two CI shards' dune-step times
# (`tools/ci-times.sh`). Sub-second members weigh DEFAULT_WEIGHT.
TRAIN_WEIGHTS='
test/training/train-fsm_transformer 218
test/training/train-transformer_names 120
test/training/train-circles_conv 88
test/training/train-mlp_bn_names 33
test/training/train-bigram 20
test/training/train-mlp_names 6
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
# The sharding itself: MODE (targets|plan), K, N, SUITE (main|train), then
# alias_listing's output for `main` or the directory list for `train`. (The
# listing is an argument because the heredoc below is python's stdin.)
shard() {
  python3 - "$@" "$SUITE_ALIASES" "$SPLIT_DIR" "$DEFAULT_WEIGHT" "$WEIGHTS" "$TRAIN_WEIGHTS" "$root" <<'PY'
import re
import sys

mode, k, n, kind, text, suite, split_dir, default_weight, table, train_table, root = sys.argv[1:12]
k, n, default_weight = int(k), int(n), float(default_weight)
suite = suite.split()


def die(message):
    print('ci-shard.sh: ' + message, file=sys.stderr)
    sys.exit(2)


def dune_file(d):
    return root + '/' + ('' if d == '.' else d + '/') + 'dune'


def main_units():
    listing, _, dirs = text.partition('\n--dirs--\n')
    listing, _, targets = listing.partition('\n--targets--\n')
    dirs = dirs.split()
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
        path = dune_file(d)
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
    units = {'@rest': rest}
    units.update({t: ['@@%s/runtest-%s' % (split_dir, t)] for t in tests})
    return units


def sexps(path):
    # A dune file as nested lists of strings: atoms and quoted strings alike
    # (dune reads `"train"` as `train`), `;` comments dropped.
    code, i, stack = open(path).read(), 0, [[]]
    atom = re.compile(r'[^\s()";]+')
    while i < len(code):
        c = code[i]
        if c.isspace():
            i += 1
        elif c == ';':
            j = code.find('\n', i)
            i = len(code) if j < 0 else j
        elif code.startswith('#|', i) or code.startswith('#;', i):
            die('%s uses a %s comment, which this reader does not skip; teach it' % (path, code[i:i + 2]))
        elif c == '(':
            stack.append([])
            i += 1
        elif c == ')':
            if len(stack) == 1:
                die('%s: unbalanced )' % path)
            done = stack.pop()
            stack[-1].append(done)
            i += 1
        elif c == '"':
            j, value = i + 1, []
            while j < len(code) and code[j] != '"':
                if code[j] == '\\':
                    j += 1
                value.append(code[j:j + 1])
                j += 1
            if j >= len(code):
                die('%s: unterminated string' % path)
            stack[-1].append(''.join(value))
            i = j + 1
        else:
            m = atom.match(code, i)
            stack[-1].append(m[0])
            i = m.end()
    if len(stack) != 1:
        die('%s: unbalanced (' % path)
    return stack[0]


def train_stanzas(stanzas, d, path, seen):
    # Record, for directory D, the train-<name> aliases its stanzas define and
    # its train aggregate's deps; a `subdir` stanza's stanzas belong to D/NAME.
    defined, aggregates = seen.setdefault(d, (set(), []))
    for stanza in stanzas:
        if not (isinstance(stanza, list) and stanza and isinstance(stanza[0], str)):
            die('%s: a form that is not a stanza' % path)
        kind_ = stanza[0]
        fields = [f for f in stanza[1:] if isinstance(f, list) and f and isinstance(f[0], str)]
        if kind_ == 'subdir':
            if len(stanza) < 2 or not isinstance(stanza[1], str):
                die('%s: a subdir stanza without a directory name' % path)
            sub = stanza[1] if d == '.' else d + '/' + stanza[1]
            train_stanzas(stanza[2:], sub.removeprefix('./'), path, seen)
            continue
        if kind_ in ('include', 'dynamic_include'):
            die('%s has an %s stanza, whose aliases this reader cannot see; teach it' % (path, kind_))
        if kind_ == 'alias':
            names = [f[1:] for f in fields if f[0] == 'name']
            name = names[0][0] if len(names) == 1 and len(names[0]) == 1 else None
            if name == 'train':
                extra = sorted({f[0] for f in fields} - {'name', 'deps', 'package'})
                if extra:
                    die('%s: the train aggregate has %s, which may change what @train '
                        'builds; teach this script' % (path, ', '.join(extra)))
                aggregates.append([x for f in fields if f[0] == 'deps' for x in f[1:]])
            elif name is not None and name.startswith('train-'):
                defined.add(name)
            continue
        # A definition is a stanza's OWN alias/aliases field; the same form
        # inside its deps is a dependency.
        for f in fields:
            if f[0] in ('alias', 'aliases'):
                for a in f[1:]:
                    if a == 'train':
                        die('%s attaches a %s stanza to the train alias itself; give it a '
                            'train-<name> alias in the aggregate' % (path, kind_))
                    if isinstance(a, str) and a.startswith('train-'):
                        defined.add(a)


def train_units():
    seen = {}
    for d in text.split():
        path = dune_file(d)
        train_stanzas(sexps(path), d, path, seen)
    units = {}
    for d in sorted(seen):
        defined, aggregates = seen[d]
        if not aggregates:
            continue
        if len(aggregates) > 1:
            die('%s defines the train alias %d times' % (d, len(aggregates)))
        if not aggregates[0]:
            die('%s: the train aggregate has no members' % d)
        prefix = '' if d == '.' else d + '/'
        for dep in aggregates[0]:
            if not (isinstance(dep, list) and len(dep) == 2 and dep[0] == 'alias'
                    and isinstance(dep[1], str) and dep[1].startswith('train-')):
                die('%s: the train aggregate depends on %r, not an (alias train-<name>); '
                    'teach this script' % (d, dep))
            if dep[1] not in defined:
                die('%s: the train aggregate names %s, which no stanza there defines' % (d, dep[1]))
            unit = prefix + dep[1]
            if unit in units:
                die('%s: the train aggregate names %s twice' % (d, dep[1]))
            units[unit] = ['@@' + unit]
    if not units:
        die('no tracked dune file defines a train aggregate')
    return units


units = train_units() if kind == 'train' else main_units()
weights = {}
for line in (train_table if kind == 'train' else table).splitlines():
    if line.strip():
        name, value = line.split()
        weights[name] = float(value)
for name in sorted(set(weights) - set(units)):
    print('ci-shard.sh: warning: weight for %s, which is no unit any more; '
          'refresh the weights' % name, file=sys.stderr)

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

# SECURITY (gh-ocannl-1287): a trace also records dune's environment and argv
# and each process's arguments and captured stdout/stderr, any of which can
# hold a credential. This reader opens only `process finish` events, and of
# those decodes `prog` and `dir` (TEXT below) and the CPU-time pairs of
# `rusage`, each only when all decimal digits; it tests `process_args` for the
# one fast-math flag by comparing bytes, decoding none. Everything else -- every
# other event kind, the time slot, every other field -- is skipped by its
# length prefixes, and a refusal quotes byte offsets, never bytes. The boundary
# is tools/action-durations.sh's, and so is its limit: it assumes
# producer-framed csexp, so a tampered length prefix can swallow a neighbouring
# record into a decoded field; protection against that is producer-side
# scrubbing (gh-ocannl-1280). tools/test-ci-shard.sh plants a canary in every
# other field and kind of its inventory, and a twin per entry that decodes and
# prints it, which the canary check alone must reject.
weigh() {
  python3 - "$1" "$SPLIT_DIR" <<'PY'
import collections
import os
import re
import sys

path, split_dir = sys.argv[1:3]
data = open(path, 'rb').read()
n = len(data)
OPEN, CLOSE, COLON = 40, 41, 58
FINISH = (b'process', b'finish')
# The fields decoded as text, per event kind; no other kind is opened.
TEXT = {FINISH: {b'prog', b'dir'}}
FAST_MATH = b'--ocannl_cc_backend_fast_math=true'
CPU_TIMES = {b'user_cpu_time', b'system_cpu_time'}


class Bad(Exception):
    """Not a complete canonical-S-expression trace; carries an offset, never trace bytes."""


def atom(i):
    # `N:bytes` -> (start, end) of the bytes, which is also the next offset.
    j = i
    while j < n and 48 <= data[j] <= 57:
        j += 1
    if j == i or j >= n or data[j] != COLON:
        raise Bad(i)
    start = j + 1
    end = start + int(data[i:j])
    if end > n:
        raise Bad(i)
    return start, end


def skip(i):
    # Step over one element, atom or list, without decoding any of it.
    depth = 0
    while True:
        if i >= n:
            raise Bad(i)
        if data[i] == OPEN:
            depth += 1
            i += 1
        elif data[i] == CLOSE:
            if depth == 0:
                raise Bad(i)
            depth -= 1
            i += 1
        else:
            i = atom(i)[1]
        if depth == 0:
            return i


def skip_to_close(i):
    while True:
        if i >= n:
            raise Bad(i)
        if data[i] == CLOSE:
            return i + 1
        i = skip(i)


def head(i):
    # The atom at I as bytes, or None at a list or a close; and the offset after it.
    if i >= n:
        raise Bad(i)
    if data[i] in (OPEN, CLOSE):
        return None, i
    s, e = atom(i)
    return data[s:e], e


def value(i):
    # Decode one element: an atom as str, a list as a list.
    if i >= n:
        raise Bad(i)
    if data[i] != OPEN:
        s, e = atom(i)
        return data[s:e].decode('utf8', 'replace'), e
    out, i = [], i + 1
    while True:
        if i >= n:
            raise Bad(i)
        if data[i] == CLOSE:
            return out, i + 1
        v, i = value(i)
        out.append(v)


def has_fast_math(i):
    # Whether the element at I (process_args: a list of atoms) holds the
    # fast-math flag, compared as bytes; nothing is decoded.
    if i >= n:
        raise Bad(i)
    if data[i] != OPEN:
        s, e = atom(i)
        return data[s:e] == FAST_MATH, e
    found, j = False, i + 1
    while True:
        if j >= n:
            raise Bad(j)
        if data[j] == CLOSE:
            return found, j + 1
        if data[j] == OPEN:
            j = skip(j)
            continue
        s, e = atom(j)
        found = found or data[s:e] == FAST_MATH
        j = e


def cpu_ns(i):
    # The element at I is rusage's list of `(key value)` pairs: the sum of the
    # CPU-time values, each decoded only when all decimal digits (else a
    # refusal at its offset); every other pair is skipped.
    if i >= n:
        raise Bad(i)
    if data[i] != OPEN:
        return 0, skip(i)
    total, j = 0, i + 1
    while True:
        if j >= n:
            raise Bad(j)
        if data[j] == CLOSE:
            return total, j + 1
        if data[j] != OPEN:
            j = skip(j)
            continue
        key, k = head(j + 1)
        if key in CPU_TIMES:
            if k >= n or data[k] in (OPEN, CLOSE):
                raise Bad(k)
            s, e = atom(k)
            if not data[s:e].isdigit():
                raise Bad(k)
            total += int(data[s:e])
            k = e
        j = skip_to_close(k)


def event(i):
    # One top-level event -> (kind, fields) for a kind in TEXT, else None.
    if data[i] != OPEN:
        raise Bad(i)
    cat, i = head(i + 1)
    name, i = head(i) if cat is not None else (None, i)
    kind = (cat, name)
    text = TEXT.get(kind)
    if text is None:
        return None, skip_to_close(i)
    if data[i] == CLOSE:
        return (kind, {}), i + 1
    i = skip(i)  # the time slot
    fields = {}
    while True:
        if i >= n:
            raise Bad(i)
        if data[i] == CLOSE:
            return (kind, fields), i + 1
        if data[i] != OPEN:
            i = skip(i)
            continue
        key, j = head(i + 1)
        if key in text:
            fields[key.decode()], j = value(j)
        elif kind == FINISH and key == b'process_args':
            fields['fast_math'], j = has_fast_math(j)
        elif kind == FINISH and key == b'rusage':
            fields['cpu_ns'], j = cpu_ns(j)
        i = skip_to_close(j)


events, i = [], 0
try:
    while i < n:
        if data[i] in b' \t\r\n':
            i += 1
            continue
        ev, i = event(i)
        if ev is not None:
            events.append(ev)
except Bad as e:
    sys.exit('ci-shard.sh: %s is not a canonical-S-expression dune trace (dune >= 3.24): '
             'malformed or cut short at byte %d' % (path, e.args[0]))
toolchain = {'ppx.exe', 'menhir', 'cc', 'gcc', 'clang', 'ar', 'as', 'ld'}
cost = collections.Counter()
seen = 0
for kind, fields in events:
    if kind != FINISH:
        continue
    seen += 1
    prog = fields.get('prog')
    prog = os.path.basename(prog) if isinstance(prog, str) else ''
    if prog.startswith('ocaml') or prog in toolchain:
        continue
    cpu = fields.get('cpu_ns', 0) / 1e9
    # The build context's root is the last `default` component: `_build/default`,
    # a sandbox's `.sandbox/<hash>/default`, or a --build-dir's `<dir>/default`.
    directory = fields.get('dir')
    directory = re.sub(r'^.*/default(?:/|$)', '', directory if isinstance(directory, str) else '') or '.'
    if directory != split_dir:
        cost['@rest'] += cpu
        continue
    name = prog.removesuffix('.exe')
    if fields.get('fast_math'):
        name += '_fast_math'
    cost[name] += cpu
if not seen:
    sys.exit('ci-shard.sh: no finished processes in %s (cached build?)' % path)
for name, cpu in cost.most_common():
    print('%s %.0f' % (name, cpu))
PY
}

# The suite argument: `main` when absent, and nothing but the two suites.
suite_of() {
  case ${1:-main} in
    main | train) printf '%s' "${1:-main}" ;;
    *) die "unknown suite '$1' (main or train)" ;;
  esac
}

# What the sharding reads: dune's listing for `main`, the dune directories
# for `train` (whose files it reads itself).
units_input() {
  if [ "$1" = train ]; then
    dune_dirs
  else
    alias_listing
  fi
}

case ${1-} in
  targets)
    [ $# -eq 2 ] || [ $# -eq 3 ] || die "usage: tools/ci-shard.sh targets K/N [SUITE]"
    [[ $2 =~ ^([1-9][0-9]*)/([1-9][0-9]*)$ ]] || die "shard must be K/N, got '$2'"
    k=${BASH_REMATCH[1]} n=${BASH_REMATCH[2]}
    [ "$k" -le "$n" ] || die "shard $k of only $n"
    suite=$(suite_of "${3-}")
    input=$(units_input "$suite")
    shard targets "$k" "$n" "$suite" "$input"
    ;;
  plan)
    { [ $# -eq 2 ] || [ $# -eq 3 ]; } && [[ $2 =~ ^[1-9][0-9]*$ ]] || die "usage: tools/ci-shard.sh plan N [SUITE]"
    suite=$(suite_of "${3-}")
    input=$(units_input "$suite")
    shard plan 1 "$2" "$suite" "$input"
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

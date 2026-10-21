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
# coverage, so they warn rather than fail; refresh WEIGHTS from a CI run with
# `merge-weights` (below), and TRAIN_WEIGHTS as its comment says.
#
# Usage:
#   tools/ci-shard.sh targets K/N [SUITE]  # shard K's dune targets, one per line
#   tools/ci-shard.sh plan N [SUITE]       # each shard's units and estimated load
#   tools/ci-shard.sh aliases              # the aliases the `main` suite shards
#   tools/ci-shard.sh weigh TRACE          # per-unit seconds from a dune trace
#   tools/ci-shard.sh merge-weights [LOG]  # a WEIGHTS block from one CI run's log
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
# wall-clock seconds per test/operations executable, keyed by the alias of the same
# name, plus `@rest` for every non-compiler process elsewhere. An executable
# run with `--ocannl_cc_backend_fast_math=true` is credited to its
# `<name>_fast_math` twin. An executable whose alias is named differently
# (sweep_harness_driver runs for runtest-sweep_harness) keeps its own name: map
# it by hand, or `plan` warns about it as a stale entry. Keep the units above a
# few seconds; the rest weigh DEFAULT_WEIGHT anyway.
# It decodes only the fields it needs (SECURITY, above `weigh` below).
# Wall-clock -- dune's own duration of each process -- because what a unit
# costs a shard is the dune job slot it holds: on the 4-vCPU runner at `-j 4`
# cc_march_census held one for 806s on 474 CPU-s, and sweep_harness's driver
# for 343s on 135 (gh-ocannl-1464), so CPU weights dealt the slowest shard the
# longest poles. Wall-clock is only meaningful measured where the shards run,
# under their own load (a workstation's is dominated by whatever else it is
# doing: XProtect stalls, other sessions' suites), so measure it there.
#
# CI does it: each `main` shard
# job weighs its own build's trace (ci.yml's "Weigh the shard's units" step)
# and prints the table as `ci-shard-weight K/N UNIT SECONDS` lines, and
#   gh run view RUN --repo lukstafi/ocannl-staging --log | tools/ci-shard.sh merge-weights
# merges one run's shards into the replacement block (`merge_weights` says
# how); `gh run view --log` answers only once the whole run has completed, and
# before that each shard job's log is
#   gh api --allow-escape-sequences repos/lukstafi/ocannl-staging/actions/jobs/JOB/logs
# (its compiler warnings carry terminal escapes, which gh otherwise withholds).
# Then map by hand what `plan` warns is no unit (sweep_harness_driver ->
# sweep_harness; an inline-test runner or an executable several aliases run
# has no one unit, and is dropped), check the deal with `plan 2`, and the next
# run's shard dune-step times with `tools/ci-times.sh`.

set -euo pipefail

SUITE_ALIASES="default runtest bin-smoke"
SPLIT_DIR=test/operations
DEFAULT_WEIGHT=1

# Seconds per unit on the ubuntu runner (gh-ocannl-1464), from staging#1303's
# first CI run (39599006506, 2026-10-21): the larger of a unit's wall-clock and
# CPU time where its job's wall-clock table listed it, CPU time below that
# table's cut (under ~9s, where the two differ by seconds); the next refresh
# reads wall-clock throughout, from `weigh`. Units under 2s weigh
# DEFAULT_WEIGHT; the run's 250 of them averaged 0.3s.
WEIGHTS='
cc_march_census 806
autotune_staffed_singles 780
autotune_round_expansion 625
flip_twins 423
autotune_composite_playoff 413
sweep_harness 347
autotune_sketch_placements 336
autotune_device_memory 261
padded_conv_cache 256
flip_abandonment 237
schedule_conv_gemm 204
autotune_measured_refusal 163
autotune_split_reduce 149
placement_store 144
autotune_candidate_release 131
autotune_smoke 122
online_softmax 97
env_var_deps 94
@rest 91
autotune_budget 89
gpu_fission_mapping 83
bridge_arm_coverage 57
online_softmax_block_fast_math 56
online_softmax_fast_math 56
reduction_forms 53
schedule_cache_numerics 52
autotune_labels 51
online_softmax_block 48
fission_replay_segmentation 45
accum_width 34
autotune_concurrent_refusal 32
autotune_callback_release 29
batch_norm_running 29
online_softmax_block_mma 27
shipped_kernels_alone 27
codegen_text_inventory 23
autotune_replay_refusal 22
broadcast_constant_init 21
schedule_epilogue_fusion 21
schedule_strided_1x1 19
limits_field_classes 18
autotune_serial_baseline 18
autotune_seam_domains 17
verdict_ratchet 17
autotune_scope_menu 16
dead_export_scan 15
footprint_materialization 14
cost_model_selection 14
privatize_census 14
symbolic_extent_launch 13
gpu_packed_prec 13
bench_checksum_discrimination 13
buffer_aliasing 13
test_block_tensor 12
schedule_mma_matmul 12
gpu_lane_reduction 11
bandwidth_calibration 11
bench_self_test 11
test_concat_graph 10
virtual_chain_fanin 10
hardware_warp_shuffle 10
zero_elision 10
gpu_serial_lanes 9
slot_kind_live 9
gpu_parallel_prefix 9
narrow_storage_compute 9
sgd_variants 9
test_execution_deps 8
schedule_contraction_nest 8
flip_bound_pruning 8
epilogue_fusion_mma_seeds 8
affine_surjectivity 7
privatize_recheck 7
agent_notes_structure 7
autotune_progress 7
inline_flip_tune 7
qkv_zero_init 7
schedule_cache_device 7
schedule_coalesce 7
matmul_interchange 7
autotune_routine_name 6
rope_test 6
test_config_consistency 6
memory_budget_planner 5
provenance_tag_inventory 5
split_reduce_pair 5
test_uniform_odd_sizes 5
test_uniform_virtual_lane 5
tile_mma_declines 5
virtual_rejection_boundary 5
zero2hero_1of7_exec 5
autotune_bound_pruning 4
backend_golden_family_scan 4
config_usage_scan 4
digest_completeness 4
nonoverlap_pool_grads 4
schedule_pack_mma_matmul 4
schedule_split_reduce 4
virtual_shared_loop 4
atomic_file_rename_scan 3
bf16_builtins 3
cache_dir_ignores 3
cache_key_agreement 3
cc_dlopen_cause 3
cc_kept_library 3
context_handle_refusals 3
cost_model_template 3
cpu_parallel 3
cuda_like_builtins_emulation 3
dynamic_axis_coverage 3
integer_power_domain 3
ll_test_ratchet 3
model_load_tiebreak 3
reduction_accumulator_residency 3
reserved_routine_names 3
resnet_block 3
schedule_batch_grid 3
seeding_census 3
set_values_pool_coalescing 3
shell_scripts_parse 3
splice_semantics 3
symbolic_extent_backward 3
tensorized_census_ratchet 3
test_einsum3 3
tile_mma_geometry 3
transformer_test 3
api_drift_history 2
cmple_ops 2
conv_bias_shapes 2
cpu_scratch_matmul_grid 2
cpu_simd_reduction 2
decoder_only_test 2
digest_identity_flips 2
fission_schedule 2
gpu_small_leading_axis 2
launch_predicate_parity 2
ll_test_scan_cases 2
narrow_rng_nesting 2
operand_key_ratchet 2
overlapping_window_grads 2
rev_sides_width1_store 2
scan_loop 2
schedule_batched_mma 2
schedule_ops 2
schedule_partition 2
scope_over_materialized 2
seeded_default 2
seeded_split_reduce 2
simplify_if_narrowing 2
sketch_family_tree 2
stored_template_elision 2
symbolic_extent_contract 2
test_default_init_std 2
test_one_hot_embedding_backward 2
test_one_hot_embedding_lookup 2
test_random_histograms 2
tile_mma_narrow 2
virtual_affine 2
virtual_diagonal 2
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
# those decodes `prog` and `dir` (TEXT below) and the duration in the time slot
# `(start duration)`, only when both are all decimal digits; it tests
# `process_args` for the one fast-math flag by comparing bytes, decoding none.
# Everything else -- every other event kind, every other field, rusage included
# -- is skipped by its length prefixes, and a refusal quotes byte offsets, never
# bytes. The boundary
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


def duration_ns(i):
    # The time slot at I, `(start duration)` in nanoseconds: the duration, its
    # atoms checked as bytes and decoded only when all decimal digits; any other
    # shape is a refusal at its offset (a finish event dune always times).
    if i >= n or data[i] != OPEN:
        raise Bad(i)
    spans, j = [], i + 1
    while j < n and data[j] not in (OPEN, CLOSE):
        s, e = atom(j)
        spans.append((s, e))
        j = e
    if j >= n or data[j] != CLOSE or len(spans) != 2 or not all(data[s:e].isdigit() for s, e in spans):
        raise Bad(i)
    s, e = spans[1]
    return int(data[s:e]), j + 1


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
    fields = {}
    if kind == FINISH:
        fields['ns'], i = duration_ns(i)
    else:
        i = skip(i)  # the time slot
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
    seconds = fields.get('ns', 0) / 1e9
    # The build context's root is the last `default` component: `_build/default`,
    # a sandbox's `.sandbox/<hash>/default`, or a --build-dir's `<dir>/default`.
    directory = fields.get('dir')
    directory = re.sub(r'^.*/default(?:/|$)', '', directory if isinstance(directory, str) else '') or '.'
    if directory != split_dir:
        cost['@rest'] += seconds
        continue
    name = prog.removesuffix('.exe')
    if fields.get('fast_math'):
        name += '_fast_math'
    cost[name] += seconds
if not seen:
    sys.exit('ci-shard.sh: no finished processes in %s (cached build?)' % path)
for name, seconds in cost.most_common():
    print('%s %.0f' % (name, seconds))
PY
}

# One CI run's per-shard `weigh` tables -> a WEIGHTS block. Each `main` shard
# job prints its own trace's table as `ci-shard-weight K/N UNIT SECONDS` lines
# (ci.yml's "Weigh the shard's units" step), read here from LOG or stdin
# wherever on a line they sit: `gh run view RUN --log` prefixes each with its
# job and a timestamp. A unit's weight is its seconds summed over the shards,
# except `@rest`: every shard runs some processes outside test/operations
# (whatever its tests depend on there), so the shards without the `@rest` unit
# measure that overhead, and `@rest` weighs what the shard holding it -- the
# largest -- spent beyond their mean. Units under MIN_WEIGHT seconds are left
# to DEFAULT_WEIGHT; their count and mean go to stderr, to check that default
# against. Refuses (exit 2) a log missing a shard of its N, mixing Ns, or
# naming one unit twice for one shard (two runs, or a rerun's attempts).
MIN_WEIGHT=2
merge_weights() { # LOG|-
  local lines
  # Only the marked lines reach python, as an argument: the heredoc is its stdin.
  lines=$(grep -aoE 'ci-shard-weight [0-9]+/[0-9]+ [^ ]+ [0-9]+(\.[0-9]+)?' -- "$1") || die "no ci-shard-weight lines in ${1/#-/stdin}"
  python3 - "$lines" "$MIN_WEIGHT" <<'PY'
import collections
import sys

text, min_weight = sys.argv[1], float(sys.argv[2])


def die(message):
    print('ci-shard.sh: merge-weights: ' + message, file=sys.stderr)
    sys.exit(2)


rows = [line.split()[1:] for line in text.splitlines()]
ns = {int(shard.split('/')[1]) for shard, _, _ in rows}
if len(ns) != 1:
    die('the lines mix shard counts %s' % sorted(ns))
n = ns.pop()
shards = collections.defaultdict(dict)
for shard, unit, seconds in rows:
    k = int(shard.split('/')[0])
    if unit in shards[k]:
        die('shard %s names %s twice: one run, one attempt, at a time' % (shard, unit))
    shards[k][unit] = float(seconds)
missing = [k for k in range(1, n + 1) if k not in shards]
if missing or len(shards) != n:
    die('no lines for shard(s) %s of %d: a job that failed before weighing, or another run'
        % (', '.join('%d/%d' % (k, n) for k in missing) or '?', n))
rest = sorted((shards[k].get('@rest', 0.0), k) for k in shards)
holder = rest[-1][1]
others = [s for s, k in rest[:-1]]
overhead = sum(others) / len(others) if others else 0.0
weights = collections.Counter()
for k, table in shards.items():
    for unit, seconds in table.items():
        if unit != '@rest':
            weights[unit] += seconds
weights['@rest'] = max(0.0, shards[holder].get('@rest', 0.0) - overhead)
shared = sorted(u for u in weights if u != '@rest' and sum(u in t for t in shards.values()) > 1)
for u in shared:
    print('ci-shard.sh: merge-weights: warning: %s ran in more than one shard; its weight sums them' % u,
          file=sys.stderr)
kept = sorted(((w, u) for u, w in weights.items() if w >= min_weight), key=lambda x: (-x[0], x[1]))
small = [w for u, w in weights.items() if w < min_weight]
for w, u in kept:
    print('%s %.0f' % (u, w))
print('ci-shard.sh: merge-weights: %d shards; @rest overhead per shard %.0fs; %d units under %gs, mean %.1fs'
      % (n, overhead, len(small), min_weight, sum(small) / len(small) if small else 0.0), file=sys.stderr)
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
  merge-weights)
    { [ $# -eq 1 ] || { [ $# -eq 2 ] && { [ "$2" = - ] || [ -f "$2" ]; }; }; } ||
      die "usage: tools/ci-shard.sh merge-weights [LOG]"
    merge_weights "${2:--}"
    ;;
  -h | --help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

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
#   tools/ci-shard.sh weigh [--wall] TRACE # per-unit CPU (wall-clock) seconds from a dune trace
#   tools/ci-shard.sh merge-weights [--wall] [LOG...]
#                                          # a WEIGHTS (WALL_WEIGHTS) block from CI runs' logs, one per run
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
# CPU seconds per unit, plus `@rest` for every non-compiler process
# elsewhere. A process is credited by what dune ran it FOR -- its target, not
# its executable (gh-ocannl-1503): an alias action's target names its alias
# (`runtest-<name>` -> unit `<name>`), a rule's target file names its test
# (`<name>.exe.output`, `<name>.actual`, ... -> `<name>`). Keyed by executable,
# one program several aliases run was credited to one of them or to none
# (online_softmax.exe ran online_softmax_fp_licences for 143 s that its
# namesake was dealt; inline-test-runner and the cc_temp_ownership-<variant>
# family were dropped; sweep_harness_driver needed a hand mapping), so the
# deal balanced weights that were not the shards' work. A process the trace
# records no target for is placed by its directory and keeps its program's
# name. A rule output named after no unit (`backend-0-0.log.actual`, which
# runtest-micrograd_demo_logging diffs) keeps its stem here; `merge-weights`
# and `plan` credit it to the unit whose stanza in the split directory's dune
# file names it, and report loudly a key they cannot place. Keep the units
# above a few seconds; the rest weigh DEFAULT_WEIGHT anyway.
# It decodes only the fields it needs (SECURITY, above `weigh` below).
# CPU seconds -- the user and system time dune's rusage records for each
# process, its waited-for children (the C compiler a test runs) included --
# because they ADD UP: a shard runs at `-j 4` on a 4-vCPU runner whose cores
# its tests keep busy, so its dune step lasts about its units' CPU over 4.
# Wall-clock does not add up. A test's wall-clock is its CPU stretched by
# whatever shares the runner with it -- most between tests running OpenMP
# kernels, whose libgomp barrier spin is sized for their own threads, not the
# runner's (which is why ci.yml's build jobs now run with
# OMP_WAIT_POLICY=passive: gh-ocannl-1544 measured the spin at a third of
# the suite's CPU and up to 40% of a main shard's dune step). Dealt by
# wall-clock weights
# measured under one deal (gh-ocannl-1464's choice), the next deal predicted
# 60.5/60.5 min of slot time and ran 80.6 min on one shard while the other
# outlived the job's 45-minute timeout -- padded_conv_cache read 160 s by its
# weight and 394 s beside its new neighbours (gh-ocannl-1503). CPU moves too
# (spinning OpenMP threads burned it), but less, and within a run it adds up:
# master's deal measured 89.0 and 44.4 CPU-minutes over dune steps of 27.5 and
# 14.8 min. Check each new deal's run against `plan` all the same, and refresh
# from it: averaging runs of different deals averages out the neighbours.
# The one thing CPU hides is a single unit whose own wall-clock outlasts the
# rest of its shard, so `weigh --wall` measures wall-clock too, CI prints it
# as `ci-shard-wall` lines, and `plan` names each shard's longest unit by the
# WALL_WEIGHTS table they refresh. CPU is only meaningful measured on the
# runner's own processor, so measure it in CI.
#
# CI does it: each `main` shard
# job weighs its own build's trace (ci.yml's "Weigh the shard's units" step)
# and prints the table as `ci-shard-weight K/N UNIT SECONDS` lines, and
#   gh run view RUN --repo lukstafi/ocannl-staging --log >RUN.log
#   tools/ci-shard.sh merge-weights RUN1.log RUN2.log ...
# merges each run's shards, then averages the runs, into the replacement block
# (`merge_weights` says how). Average two runs or more when you have them:
# even CPU moves somewhat with the runner's load. `gh run view --log` answers only once
# the whole run has completed, and before that each shard job's log is
#   gh api --allow-escape-sequences repos/lukstafi/ocannl-staging/actions/jobs/JOB/logs
# (its compiler warnings carry terminal escapes, which gh otherwise withholds);
# concatenate one run's shard jobs into one LOG. Check the deal with `plan 2`
# (it warns about a weight that names no unit), and the next run's shard
# dune-step times with `tools/ci-times.sh`. Weigh only runs of the CURRENT
# code, and only runs whose `weigh` printed CPU seconds (before gh-ocannl-1503
# it printed wall-clock): a test's speed-up leaves its old weight dealing a
# phantom load (the autotune tests ran 2-3x faster after gh-ocannl-1340, and
# the table measured before it dealt one shard 2.3x the other's work while
# predicting them equal).

set -euo pipefail

SUITE_ALIASES="default runtest bin-smoke"
SPLIT_DIR=test/operations
DEFAULT_WEIGHT=1

# CPU seconds per unit on the ubuntu runner (gh-ocannl-1503), under
# OMP_WAIT_POLICY=passive (gh-ocannl-1544): `merge-weights` over the passive
# arm of dispatch runs 39932047808, 39932050243 and 39932052982, which built
# both main shards with and without the setting side by side. Against those
# runs' default arm, passive took the suite from 125 to 79 CPU-minutes, the
# autotune family's units to a third or less (autotune_composite_playoff 931
# -> 262, autotune_sketch_placements 453 -> 49), so the table measured under
# libgomp's default spin dealt one shard 37 CPU-minutes and the other 44.
# Shared dependencies are left out. Units under 2s weigh DEFAULT_WEIGHT; the
# 272 of them averaged 0.4s.
WEIGHTS='
autotune_staffed_singles 312
flip_abandonment 269
autotune_composite_playoff 262
padded_conv_cache 202
sweep_harness 186
autotune_round_expansion 165
@rest 161
flip_twins 133
autotune_device_memory 125
slot_kind_live 124
autotune_candidate_release 121
bridge_arm_coverage 92
autotune_smoke 87
schedule_conv_gemm 82
online_softmax_block_fast_math 81
online_softmax 74
placement_store 74
online_softmax_fp_licences 73
online_softmax_block 72
autotune_measured_refusal 67
online_softmax_fast_math 65
shipped_kernels_alone 65
autotune_split_reduce 50
autotune_sketch_placements 49
gpu_fission_mapping 46
tune_call_budget 42
reduction_forms 41
autotune_labels 37
accum_width 34
schedule_cache_numerics 34
schedule_strided_1x1 32
autotune_budget 30
cc_march_census-emission 30
operations_tutorials 30
autotune_serial_baseline 29
env_var_deps 28
fission_replay_segmentation 27
autotune_callback_release 26
autotune_concurrent_refusal 26
codegen_text_inventory 26
cost_model_selection 26
batch_norm_running 25
online_softmax_block_mma 23
cc_march_census-w16_native_foreign 22
cc_march_census-w64_native_foreign 22
gpu_kernel_cache 22
cc_march_census-w64_native_host 21
limits_field_classes 21
dead_export_scan 20
demo_models_smoke 19
verdict_ratchet 19
autotune_replay_refusal 17
autotune_seam_domains 17
broadcast_constant_init 17
schedule_epilogue_fusion 17
schedule_mma_matmul 17
autotune_scope_menu 15
footprint_materialization 15
cc_march_census-w16_wide_foreign 14
hardware_warp_shuffle 14
privatize_census 14
schedule_coalesce 14
test_block_tensor 14
deferred_hit_compiles 13
mobile_cnn_width 13
symbolic_extent_launch 13
cc_march_census-w16_native_host 12
cc_march_census-w32_native_foreign 12
gpu_lane_reduction 12
gpu_packed_prec 12
primitive_ops 11
virtual_chain_fanin 11
autotune_progress 10
bench_checksum_discrimination 10
hello_world_op 10
test_concat_graph 10
test_execution_deps 10
test_uniform_odd_sizes 10
agent_notes_structure 9
bench_self_test 9
cc_march_census-w32_native_host 9
cc_toolchain_attribution-codegen 9
gpu_parallel_prefix 9
gpu_serial_lanes 9
provenance_tag_inventory 9
rope_test 9
schedule_contraction_nest 9
sgd_variants 9
test_uniform_virtual_lane 9
virtual_rejection_boundary 9
bandwidth_calibration 8
narrow_storage_compute 8
zero2hero_1of7 8
zero2hero_1of7_exec 8
zero_elision 8
cache_key_agreement 7
cc_march_census-w16_wide_host 7
cc_march_census-w64_wide_foreign 7
cc_march_census-w64_wide_host 7
cuda_like_builtins_emulation 7
epilogue_fusion_mma_seeds 7
matmul_interchange 7
memory_budget_planner 7
test_config_consistency 7
tile_mma_declines 7
tile_mma_geometry 7
affine_surjectivity 6
autotune_report_drops 6
bf16_builtins 6
cc_march_census-w32_wide_foreign 6
cost_model_template 6
dynamic_axis_coverage 6
nonoverlap_pool_grads 6
privatize_recheck 6
qkv_zero_init 6
qkv_zero_init-approximate 6
schedule_split_reduce 6
split_reduce_pair 6
stored_template_elision 6
virtual_diagonal 6
virtual_shared_loop 6
autotune_bound_pruning 5
buffer_aliasing 5
capture_per_call 5
cc_march_census-w32_wide_host 5
cc_temp_ownership-compiler_debug 5
context_handle_refusals 5
integer_power_domain 5
reduction_accumulator_residency 5
resnet_block 5
schedule_pack_mma_matmul 5
tensorized_census_ratchet 5
test_einsum3 5
backend_golden_family_scan 4
cc_temp_ownership-compiler 4
cc_temp_ownership-compiler_dll 4
config_usage_scan 4
cpu_parallel 4
decoder_only_test 4
digest_completeness 4
digest_identity_flips 4
flip_bound_pruning 4
inline_flip_tune 4
ll_test_ratchet 4
overlapping_window_grads 4
reserved_routine_names 4
rev_sides_width1_store 4
schedule_batch_grid 4
schedule_partition 4
seeded_default 4
seeding_census 4
set_values_pool_coalescing 4
sketch_family_tree 4
sweep_components 4
tile_mma_narrow 4
transformer_test 4
atomic_file_rename_scan 3
autotune_routine_name 3
cache_dir_ignores 3
clamped_window_pool 3
cmple_ops 3
conv_family_tree 3
cpu_scratch_matmul_grid 3
env_var_deps_control 3
fission_schedule 3
gpu_small_leading_axis 3
launch_predicate_parity 3
ll_test_scan_cases 3
model_load_tiebreak 3
narrow_rng_nesting 3
narrow_uniform_bound 3
observable_grads 3
operand_key_ratchet 3
scan_loop 3
schedule_batched_mma 3
schedule_ops 3
scope_over_materialized 3
shell_scripts_parse 3
simplify_if_narrowing 3
splice_semantics 3
symbolic_extent_backward 3
symbolic_extent_contract 3
test_default_init_std 3
test_one_hot_embedding_backward 3
test_one_hot_embedding_lookup 3
test_random_histograms 3
virtual_affine 3
virtual_uncaptured_loop 3
api_drift_history 2
autotune_budget_unfloored 2
autotune_seeded_seed 2
cc_kept_library 2
closed_shape_write_guard 2
consume_code_replay 2
context_copy 2
conv_bias_shapes 2
cpu_simd_reduction 2
exit_stream_teardown 2
fission_equivalence 2
fp_licence_values 2
fresh_build_kernel_text 2
lowered_nest_order 2
one_element_reshape 2
padding_lifecycle 2
read_before_write_flip 2
runtime_integer_power 2
schedule_cache_device 2
schedule_packed_constants 2
seeded_split_reduce 2
stretch_resolution 2
test_fp8_codec_parity 2
test_uniform1 2
uint4x32_fixed_index_parity 2
virtual_guarded_setter 2
workgroup_memory_accounting 2
'

# Wall-clock seconds of the units whose own run is long, on the ubuntu
# runner: not dealt by, only `plan`'s critical-path check (the longest unit
# per shard, which a CPU load hides). From `merge-weights --wall` over the CI
# shards' `ci-shard-wall` lines -- the same three passive-arm runs as WEIGHTS
# (gh-ocannl-1544), under the deal the table before them made -- with units
# under 30s left out. Wall-clock moves with a unit's neighbours
# (gh-ocannl-1503), so read the check as a rough one.
WALL_WEIGHTS='
sweep_harness 411
autotune_staffed_singles 285
autotune_composite_playoff 266
flip_abandonment 265
padded_conv_cache 210
@rest 202
autotune_round_expansion 165
flip_twins 136
autotune_device_memory 129
slot_kind_live 126
autotune_candidate_release 124
autotune_smoke 89
schedule_conv_gemm 84
online_softmax_block_fast_math 82
placement_store 77
online_softmax 76
online_softmax_fp_licences 74
online_softmax_block 73
autotune_measured_refusal 68
online_softmax_fast_math 66
shipped_kernels_alone 66
bridge_arm_coverage 51
autotune_sketch_placements 50
autotune_split_reduce 48
gpu_fission_mapping 46
autotune_labels 42
tune_call_budget 42
reduction_forms 41
schedule_cache_numerics 35
accum_width 34
schedule_strided_1x1 33
autotune_budget 31
cc_march_census-emission 30
operations_tutorials 30
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
  python3 - "$@" "$SUITE_ALIASES" "$SPLIT_DIR" "$DEFAULT_WEIGHT" "$WEIGHTS" "$TRAIN_WEIGHTS" "$root" \
    "$WALL_WEIGHTS" "$MIN_WEIGHT" <<'PY'
import os
import re
import sys

mode, k, n, kind, text, suite, split_dir, default_weight, table, train_table, root, wall_table, min_weight \
    = sys.argv[1:14]
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


def parse(block):
    out = {}
    for line in block.splitlines():
        if line.strip():
            name, value = line.split()
            out[name] = float(value)
    return out


def own_fields(stanza, names):
    return [f for f in stanza[1:] if isinstance(f, list) and f and f[0] in names]


def strings(x):
    if isinstance(x, str):
        yield x
    else:
        for y in x:
            yield from strings(y)


def own_aliases(stanza):
    # The runtest-<name> units a stanza of the split directory defines: its
    # own alias/aliases field, an alias stanza's name, a test's implicit alias.
    out = {a[len('runtest-'):] for f in own_fields(stanza, ('alias', 'aliases')) for a in f[1:]
           if isinstance(a, str) and a.startswith('runtest-')}
    if stanza[0] == 'alias':
        out |= {a[len('runtest-'):] for f in own_fields(stanza, ('name',)) for a in f[1:]
                if isinstance(a, str) and a.startswith('runtest-')}
    if stanza[0] in ('test', 'tests'):
        out |= {a for f in own_fields(stanza, ('name', 'names')) for a in f[1:] if isinstance(a, str)}
    if stanza[0] == 'library' and own_fields(stanza, ('inline_tests',)):
        out |= {a for f in own_fields(stanza, ('name',)) for a in f[1:] if isinstance(a, str)}
    return out


def split_stanzas():
    path = dune_file(split_dir)
    if not os.path.exists(path):
        return []
    return [s for s in sexps(path) if isinstance(s, list) and s and isinstance(s[0], str)]


def fold(weights, units, label):
    # `weigh` keys a file target by its stem; a rule whose output is named
    # after no unit (`backend-0-0.log.actual`, which runtest-micrograd_demo_logging
    # diffs) is credited to the one unit whose stanza in the split directory's
    # dune file names that target. A key that names no unit and no single
    # owner is reported loudly and kept, so the mismatch is seen, not dealt.
    stanzas = split_stanzas()
    targets = {os.path.basename(a) for s in stanzas for f in own_fields(s, ('target', 'targets'))
               for a in strings(f[1:])}
    owners = {}
    for s in stanzas:
        own = own_aliases(s)
        for a in strings(s[1:]) if own else ():
            b = os.path.basename(a)
            if b in targets and b.split('.', 1)[0] not in units:
                owners.setdefault(b.split('.', 1)[0], set()).update(own)
    out = {}
    for name, value in weights.items():
        if name in units:
            out[name] = out.get(name, 0.0) + value
        elif len(owners.get(name, ())) == 1:
            unit, = owners[name]
            print('ci-shard.sh: %s: %s is a target of %s; its weight goes there' % (label, name, unit),
                  file=sys.stderr)
            out[unit] = out.get(unit, 0.0) + value
        else:
            print('ci-shard.sh: %s: WEIGHT NAMES NO UNIT: %s %g -- a renamed or removed test, or a rule '
                  'output this script cannot trace to its alias; fix the key' % (label, name, value),
                  file=sys.stderr)
            out[name] = value
    return out


if mode == 'resolve':
    # merge-weights' table, folded over the units the split directory's dune
    # file defines (read from the file: merge-weights runs where dune need not).
    units = set().union(*(own_aliases(s) for s in split_stanzas())) | {'@rest'}
    if len(units) == 1:
        print('ci-shard.sh: merge-weights: no %s to check the keys against' % dune_file(split_dir),
              file=sys.stderr)
        units = set(parse(text))
    folded = fold(parse(text), units, 'merge-weights')
    for name, value in sorted(folded.items(), key=lambda x: (-x[1], x[0])):
        if value >= float(min_weight):
            print('%s %.0f' % (name, value))
    sys.exit(0)

units = train_units() if kind == 'train' else main_units()
if kind == 'train':
    weights = parse(train_table)
    for name in sorted(set(weights) - set(units)):
        print('ci-shard.sh: warning: weight for %s, which is no unit any more; '
              'refresh the weights' % name, file=sys.stderr)
else:
    weights = fold(parse(table), set(units), 'warning')
walls = {} if kind == 'train' else parse(wall_table)

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
        # The critical-path check: the unit whose own wall-clock is longest
        # (WALL_WEIGHTS), which no CPU load shows. The train suite's weights
        # are wall-clock already.
        clock = walls if walls else {u: weight(u) for u in m}
        timed = [u for u in m if u in clock]
        longest = max(timed, key=lambda u: (clock[u], u)) if timed else None
        print('shard %d/%d: %d units, load %.0f (%.0f%%), longest by wall-clock %s'
              % (i + 1, n, len(m), loads[i], 100 * loads[i] / total,
                 '%s %gs' % (longest, clock[longest]) if longest else 'unknown'))
        for u in m:
            print('  %-40s %g' % (u, weight(u)))
PY
}

# SECURITY (gh-ocannl-1287): a trace also records dune's environment and argv
# and each process's arguments and captured stdout/stderr, any of which can
# hold a credential. This reader opens only `process finish` events, and of
# those decodes `prog`, `dir` and `target_files` (TEXT below) -- paths dune
# derives from the dune files -- the duration in the time slot `(start
# duration)`, only when both are all decimal digits, and of `rusage` only its
# `user_cpu_time` and `system_cpu_time`, each only when all decimal digits.
# Everything else -- every other event kind, every other field, rusage included
# -- is skipped by its length prefixes, and a refusal quotes byte offsets, never
# bytes. The boundary
# is tools/action-durations.sh's, and so is its limit: it assumes
# producer-framed csexp, so a tampered length prefix can swallow a neighbouring
# record into a decoded field; protection against that is producer-side
# scrubbing (gh-ocannl-1280). tools/test-ci-shard.sh plants a canary in every
# other field and kind of its inventory, and a twin per entry that decodes and
# prints it, which the canary check alone must reject.
weigh() { # TRACE [cpu|wall]
  python3 - "$1" "$SPLIT_DIR" "${2:-cpu}" <<'PY'
import collections
import os
import re
import sys

path, split_dir, measure = sys.argv[1:4]
data = open(path, 'rb').read()
n = len(data)
OPEN, CLOSE, COLON = 40, 41, 58
FINISH = (b'process', b'finish')
# The fields decoded as text, per event kind; no other kind is opened.
TEXT = {FINISH: {b'prog', b'dir', b'target_files'}}


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


CPU = (b'user_cpu_time', b'system_cpu_time')


def cpu_ns(i):
    # The rusage at I, `((KEY VALUE) ...)`: the sum of its user and system CPU
    # times in nanoseconds, each checked as bytes and decoded only when all
    # decimal digits; every other entry (maxrss, faults, ...) is skipped by its
    # length prefixes, never decoded.
    if i >= n or data[i] != OPEN:
        raise Bad(i)
    total, j = 0, i + 1
    while True:
        if j >= n:
            raise Bad(j)
        if data[j] == CLOSE:
            return total, j + 1
        if data[j] != OPEN:
            raise Bad(j)
        key, k = head(j + 1)
        if key in CPU:
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
# An alias action's target: `<build>/.actions/<context>/<dir>/<alias>-<digest>`.
ACTION = re.compile(r'(?:^|/)\.actions/[^/]+/(?:(.*)/)?([^/]+)-[0-9a-f]{32}$')


def in_context(path):
    # A path relative to the build context, whose root is the last `default`
    # component: `_build/default`, a sandbox's `.sandbox/<hash>/default`, or a
    # --build-dir's `<dir>/default`.
    return re.sub(r'^(?:.*/)?default(?:/|$)', '', path) or '.'


def unit_of(fields, prog):
    # The unit a process ran for, from its first target: an alias action of the
    # split directory's `runtest-<name>` is unit <name>, any other of its
    # aliases (bin-smoke) belongs to @rest; a file target <name>.<ext...> there
    # is unit <name>; anything elsewhere is @rest. A process without a target
    # is placed by its directory and named after its program.
    targets = fields.get('target_files')
    for t in targets if isinstance(targets, list) else []:
        if not isinstance(t, str):
            continue
        m = ACTION.search(t)
        if m:
            if (m.group(1) or '.') != split_dir or not m.group(2).startswith('runtest-'):
                return '@rest'
            return m.group(2)[len('runtest-'):]
        directory, base = os.path.split(in_context(t))
        return base.split('.', 1)[0] if (directory or '.') == split_dir else '@rest'
    directory = fields.get('dir')
    if in_context(directory if isinstance(directory, str) else '') != split_dir:
        return '@rest'
    return prog.removesuffix('.exe')


cost = collections.Counter()
seen = untimed = 0
for kind, fields in events:
    if kind != FINISH:
        continue
    seen += 1
    prog = fields.get('prog')
    prog = os.path.basename(prog) if isinstance(prog, str) else ''
    if prog.startswith('ocaml') or prog in toolchain:
        continue
    if measure == 'wall':
        cost[unit_of(fields, prog)] += fields.get('ns', 0) / 1e9
        continue
    if 'cpu_ns' not in fields:
        untimed += 1
    cost[unit_of(fields, prog)] += fields.get('cpu_ns', 0) / 1e9
if not seen:
    sys.exit('ci-shard.sh: no finished processes in %s (cached build?)' % path)
if untimed:
    print('ci-shard.sh: warning: %d finished processes carry no rusage; each weighs 0' % untimed,
          file=sys.stderr)
for name, seconds in cost.most_common():
    print('%s %.0f' % (name, seconds))
PY
}

# CI runs' per-shard `weigh` tables -> a WEIGHTS block. Each `main` shard job
# prints its own trace's table as `ci-shard-weight K/N UNIT SECONDS` lines
# (ci.yml's "Weigh the shard's units" step), read here from each LOG -- one
# run's shard jobs, `-` for stdin, stdin when none is named -- wherever on a
# line they sit: `gh run view RUN --log` prefixes each with its job and a
# timestamp. Within a run, a unit weighs what its shard spent on it. A unit
# that ran in more than one shard is a shared dependency: every shard holding
# one of its consumers builds it, wherever the deal puts its own alias, so its
# cost is fixed overhead of those shards that moving the unit does not move.
# It is left out of the table (dealt at DEFAULT_WEIGHT, so not counted as
# movable load) -- in every run, if any run saw it shared -- and its longest
# copy goes to stderr as overhead. (A dependency whose consumers all sit in
# its own shard looks movable; it is, together with them.) `@rest` is the exception: every
# shard runs some processes outside test/operations (whatever its tests depend
# on there), so the shards without the `@rest` unit measure that overhead, and
# `@rest` weighs what the shard holding it -- the largest -- spent beyond their
# mean. Across runs, a unit weighs its mean over the runs that measured it.
# Units under MIN_WEIGHT seconds are left to DEFAULT_WEIGHT; their count and
# mean go to stderr, to check that default against. Refuses (exit 2) a run
# missing a shard of its N, mixing Ns, or naming one unit twice for one shard
# (two runs in one LOG, or a rerun's attempts).
MIN_WEIGHT=2
merge_weights() { # cpu|wall LOG...|-
  local runs=() log stdin=0 one marker=ci-shard-weight table
  [ "$1" = wall ] && marker=ci-shard-wall
  shift
  [ $# -gt 0 ] || set -- -
  # Only the marked lines reach python, as arguments: the heredoc is its stdin.
  for log in "$@"; do
    if [ "$log" = - ]; then
      [ "$stdin" = 0 ] || die "merge-weights: stdin named twice"
      stdin=1
    fi
    one=$(grep -aoE "$marker [0-9]+/[0-9]+ [^ ]+ [0-9]+(\\.[0-9]+)?" -- "$log") ||
      die "no $marker lines in ${log/#-/stdin}"
    runs+=("$one")
  done
  table=$(python3 - "$MIN_WEIGHT" "${runs[@]}" <<'PY'
import collections
import sys

min_weight, texts = float(sys.argv[1]), sys.argv[2:]


def die(message):
    print('ci-shard.sh: merge-weights: ' + message, file=sys.stderr)
    sys.exit(2)


def merge_run(r, text):
    # One run's shards -> {unit: seconds}.
    where = 'run %d: ' % r if len(texts) > 1 else ''
    rows = [line.split()[1:] for line in text.splitlines()]
    ns = {int(shard.split('/')[1]) for shard, _, _ in rows}
    if len(ns) != 1:
        die('%sthe lines mix shard counts %s' % (where, sorted(ns)))
    n = ns.pop()
    shards = collections.defaultdict(dict)
    for shard, unit, seconds in rows:
        k = int(shard.split('/')[0])
        if unit in shards[k]:
            die('%sshard %s names %s twice: one run, one attempt, at a time' % (where, shard, unit))
        shards[k][unit] = float(seconds)
    missing = [k for k in range(1, n + 1) if k not in shards]
    if missing or len(shards) != n:
        die('%sno lines for shard(s) %s of %d: a job that failed before weighing, or another run'
            % (where, ', '.join('%d/%d' % (k, n) for k in missing) or '?', n))
    rest = sorted((shards[k].get('@rest', 0.0), k) for k in shards)
    holder = rest[-1][1]
    others = [s for s, k in rest[:-1]]
    overhead = sum(others) / len(others) if others else 0.0
    weights, shared = {}, set()
    for table in shards.values():
        for unit, seconds in table.items():
            if unit == '@rest':
                continue
            if unit in weights:
                shared.add(unit)
            weights[unit] = max(weights.get(unit, 0.0), seconds)
    weights['@rest'] = max(0.0, shards[holder].get('@rest', 0.0) - overhead)
    for u in sorted(shared):
        print('ci-shard.sh: merge-weights: %s%s ran in %d shards, a shared dependency: fixed overhead '
              'of up to %.0fs in each, left out of the table'
              % (where, u, sum(u in t for t in shards.values()), weights[u]), file=sys.stderr)
    print('ci-shard.sh: merge-weights: %s%d shards; @rest overhead per shard %.0fs' % (where, n, overhead),
          file=sys.stderr)
    return weights, shared


merged = [merge_run(r, text) for r, text in enumerate(texts, 1)]
runs = [w for w, _ in merged]
shared = set().union(*(s for _, s in merged))
weights = {}
for u in set().union(*runs) - shared:
    measured = [w[u] for w in runs if u in w]
    weights[u] = sum(measured) / len(measured)
kept = sorted(((w, u) for u, w in weights.items() if w >= min_weight), key=lambda x: (-x[0], x[1]))
small = [w for u, w in weights.items() if w < min_weight]
for w, u in kept:
    print('%s %.0f' % (u, w))
print('ci-shard.sh: merge-weights: %d run(s); %d units under %gs, mean %.1fs'
      % (len(runs), len(small), min_weight, sum(small) / len(small) if small else 0.0), file=sys.stderr)
PY
) || exit $?
  # Each key credited to the unit that owns it, read from the dune file (see
  # `fold` in the sharding): a key naming no unit is loud, never dropped.
  shard resolve 1 1 main "$table"
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
    if [ "${2-}" = --wall ]; then
      [ $# -eq 3 ] && [ -f "$3" ] || die "usage: tools/ci-shard.sh weigh [--wall] TRACE"
      weigh "$3" wall
    else
      [ $# -eq 2 ] && [ -f "$2" ] || die "usage: tools/ci-shard.sh weigh [--wall] TRACE"
      weigh "$2"
    fi
    ;;
  merge-weights)
    shift
    measure=cpu
    if [ "${1-}" = --wall ]; then
      measure=wall
      shift
    fi
    for log in "$@"; do
      [ "$log" = - ] || [ -f "$log" ] || die "usage: tools/ci-shard.sh merge-weights [--wall] [LOG...]"
    done
    merge_weights "$measure" "$@"
    ;;
  -h | --help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

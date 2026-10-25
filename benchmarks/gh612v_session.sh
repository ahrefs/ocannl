#!/bin/bash
# gh-ocannl-612 VERIFICATION session -- the cell sequence behind benchmarks/report-gh612-hip-verified.md.
#
# Every cell here ships the DEFAULT-PLACEMENT arm, via gh-ocannl-638's `tune_ship_arm=a`. The
# original session (report-gh612-hip.md) profiled arm A in every cell while the search shipped
# whichever arm was faster, so in three of four cells the profiled routine was never executed
# against anything -- the limitation that report states in its verdict. Forcing the arm makes the
# routine whose kernels are profiled the same routine whose losses the run reports, so the parity
# gate covers it and the pass-2 step p50 is a timing OF it.
#
# Why a script rather than a Reproduction block of hand-typed commands: a cell's treatment is a set
# of flags, and EVERY subcommand of that cell -- search, snap, replay, profile -- must repeat it
# exactly. A `replay` that omits `--ocannl_tune_ship_arm=a` still replays from the cache, still
# emits a step_ms record and still passes the driver's two-cache-hit gate, while shipping arm B: the
# pass-2 p50 would be an arm B timing under an arm A label, which is the exact confusion this
# session exists to remove. So the treatment is declared once, per cell, and every subcommand
# derives from it.
#
# Usage:  gh612v_session.sh [--box BOX] [--skip-box-check] <block>...
#   --box / --skip-box-check are handed to every gh612_cells.sh cell, whose fixture check runs
#   benchmarks/check-box.sh before the cell measures (benchmarks/box-gate.sh).
#   trees      resolve and print the three trees with their commits (measures nothing)
#   gh574      the gh-574 pair, 3 reps, arm order alternated: base574A vs feat574A
#   gh573      the gh-573 pair, 3 reps, alternated:           cap8A vs capoffA
#   caps       the cap-default pair, reps 4-6, alternated:    cap8A vs cap4A
#   replays    pass-2 replay (the protocol's step timings) of every cell of every block above
#   structure  snap + profile + finger on rep 1 of the four structural cells, then the diffs
#   capprofile Part 5's per-kernel profile of cap 4, plus the cap-8-vs-cap-4 structural diff
#   gate       LAST: provenance, the arm-A premise, parity over exactly these cells, artifacts.
#              It requires the structural and cap-4 profiles, so run it after structure/capprofile.
#
# Environment: BASE FEAT MASTER are the three built trees (see the report's Provenance for the
# commits); OUT_ROOT defaults to /tmp/gh612v. Blocks are independent and resumable -- a rerun of a
# block re-searches its cells from cold, which is the point (each rep is an independent search).
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
D=$HERE/gh612_cells.sh
. "$HERE/box-gate.sh"
box_gate_args "$@"; shift "$box_gate_shift"
# What every cell is handed: the box its gate checks, or the decision to skip that check.
box_gate_fwd=()
[ -z "$box_gate_box" ] || box_gate_fwd+=(--box "$box_gate_box")
[ "$box_gate_skip" = 0 ] || box_gate_fwd+=(--skip-box-check)
export OUT_ROOT=${OUT_ROOT:-/tmp/gh612v}

# WHAT certification means is pinned here; only WHERE to run it is inherited. OUT_ROOT and the three
# tree paths say where; everything below says what the report claims, and each was separately
# overridable until review found them one at a time: the workload (FIXTURE, and FIXTURE_MD5, which
# gh612_cells.sh treats as "skip the check" when empty), the correctness threshold (PARITY_MAX_ULP)
# and the sample count (EXPECT_STEPS). An exported value could weaken any of them while every
# provenance, arm and replay check stayed green -- `PARITY_MAX_ULP=1e12` certifying real divergence
# being the sharpest case. They are exported, not defaulted, so an inherited value is REPLACED.
export FIXTURE=fixtures/gpt2_mini.safetensors
export FIXTURE_MD5=5b3dfff860fc8c54af2a7d440f4cf202
export PARITY_MAX_ULP=64
export EXPECT_STEPS=8
# The execution discipline is part of what was measured, not a preference: every cell of this
# session ran under `taskset -c 0-15` (the CPU/iGPU contention on this APU is what report-gh612-hip.md
# had to discard a whole batch over), and gh612_cells.sh would otherwise take an inherited PIN.
export PIN="taskset -c 0-15"
export ARCH=gfx1151
export PROFILE_RUNS=3
# EXPECT_RUNS/EXPECT_CELLS/FIRST_REP are set per invocation below, so they are cleared rather than
# pinned: an inherited value must not leak into a block that does not set one.
unset EXPECT_RUNS EXPECT_CELLS FIRST_REP
# The wrapper's own memo state, which is caller-controllable otherwise: an exported TREES_VALIDATED
# naming a role and path skips that tree's entire validation, and the manifests are then written
# from an unvalidated checkout. Internal state gets initialized, not defaulted.
TREES_VALIDATED=""
ROLES_CHECKED=""
# Every variable gh612_cells.sh reads is now either pinned above (what the report claims) or set per
# invocation. That enumeration is the point: three review rounds each found "one more inherited
# value", so the list is closed against the driver's own reads rather than extended one name at a
# time -- `grep -o '\${[A-Z_]*[:-]' gh612_cells.sh` is how it was checked, and OUT_ROOT plus the
# three tree paths are the deliberate remainder, since they say where to run rather than what is
# being certified.
# Defaults relative to THIS checkout, matching where the report's Reproduction block creates them
# (`../wt-gh612v-*` from the repo root) -- author-absolute defaults made the published recipe fail on
# its first command for anyone whose checkout is not at /home/lukstafi/ocannl-staging. Overridable,
# since where the trees live is a property of the machine, not of the measurement.
REPO_ROOT=$(cd "$HERE/.." && pwd -P)
BASE=${BASE:-$(dirname "$REPO_ROOT")/wt-gh612v-base}
FEAT=${FEAT:-$(dirname "$REPO_ROOT")/wt-gh612v-feat}
MASTER=${MASTER:-$(dirname "$REPO_ROOT")/wt-gh612v-master}

# The cell table, keyed by ROLE rather than by path. A label that is not listed is an error rather
# than a default: with `MASTER` as a fallback, a typo would measure the wrong tree under a name the
# report trusts.
cell_role() {
  case $1 in
    base574A) printf 'BASE\n' ;;
    feat574A) printf 'FEAT\n' ;;
    cap8A|capoffA|cap4A) printf 'MASTER\n' ;;
    *) echo "gh612v_session: unknown cell '$1'" >&2; return 2 ;;
  esac
}
role_tree() {
  case $1 in
    BASE) printf '%s\n' "$BASE" ;;
    FEAT) printf '%s\n' "$FEAT" ;;
    MASTER) printf '%s\n' "$MASTER" ;;
    *) echo "gh612v_session: unknown role '$1'" >&2; return 2 ;;
  esac
}
cell_tree() { role_tree "$(cell_role "$1")" ; }

# The three roles are three DIFFERENT commits by construction, so two of them resolving to one
# checkout is never a configuration worth honoring -- and it is not a harmless one: with BASE=FEAT
# the gh-574 comparison becomes a tree against itself, and the arm, parity and replay gates all pass
# on it, under two trusted labels. Checked on physical paths (a symlinked alias resolves to the same
# checkout) and once per run.
assert_distinct_roles() {
  [ -z "${ROLES_CHECKED:-}" ] || return 0
  local r p seen="" phys
  for r in BASE FEAT MASTER; do
    p=$(role_tree "$r") || return 2
    phys=$(cd "$p" 2>/dev/null && pwd -P) || phys=""
    [ -n "$phys" ] || { echo "gh612v_session: role $r ($p) is not a usable directory" >&2; return 1; }
    case " $seen " in
      *" $phys "*)
        echo "gh612v_session: two roles resolve to the same checkout ($phys)" >&2
        echo "  BASE, FEAT and MASTER are three different commits; aliasing them would compare a" >&2
        echo "  tree with itself while every gate passed under two labels." >&2; return 1 ;;
    esac
    seen="$seen $phys"
  done
  ROLES_CHECKED=1
}
# Arm A is forced in EVERY cell, including the two that shipped it anyway in the original session:
# the point is that the treatment is uniform, so a cross-cell comparison is not conditioned on which
# arm each search happened to prefer.
cell_flags() {
  case $1 in
    base574A|feat574A|cap8A) printf '%s\n' "--ocannl_tune_ship_arm=a" ;;
    capoffA) printf '%s\n' "--ocannl_tune_ship_arm=a --ocannl_virtualize_max_inline_fanin=-1" ;;
    cap4A)   printf '%s\n' "--ocannl_tune_ship_arm=a --ocannl_virtualize_max_inline_fanin=4" ;;
    *) echo "gh612v_session: unknown cell '$1'" >&2; return 2 ;;
  esac
}
# The exact bytes of the backported files, as measured. All three trees carry the same five copied
# files, and `lib/train.ml` here is the selector as it landed in ffc428d2 -- BEFORE the two
# review-round fixes on the PR branch, which touch failure paths (a raising `on_ship`; which arm's
# failure propagates when both arms fail) that no cell in this session took: every cell shipped a
# successful arm A. Recorded so the claim "verbatim backport" is checkable rather than asserted.
# NOT overridable from the environment, unlike OUT_ROOT or the tree paths. These digests and the
# commits below do not configure the session -- they IDENTIFY the measurement the report publishes,
# and a run that supplies its own is measuring something else while inheriting the labels, the
# manifests and the gate's certification. Changing them is a source edit, which is reviewable;
# exporting a variable is not.
SHA_TRAIN=5d98ac818c547662608d7ef379c8ea6ecb0e5bc91be663a26c1b818c52195fcd
SHA_GPT=de3cb8b4d56cac22885f014f6532a170a4b69c5ab1561f7bda7061289b548a1e
SHA_HARNESS=f51ef550c714642594e7eb633745a60ee4bd1e6c3c94d716702aa666aaa7d2c1
SHA_CONV=77dea492448dd31781ab28e34a2a29c5266af07548fb523a3d8d015253dfe0f3
SHA_MLP=328c1dcc51a016eef36ad6142c9003315e8e62111c502d77da68b534eb2df44b
# utils.ml is patched rather than copied, so its digest is per ROLE: BASE and FEAT share one (the
# config machinery is identical at those two commits) and MASTER has its own. Pinned exactly like
# the copied files -- a structural "the key appears twice" check accepts any file that merely
# mentions it, including one that also changed config parsing, the profiles, or an optimizer default.
SHA_UTILS_BASE=6c0aace0ea29fd3dea19737cded0c2e7308ed6d8b8f75a9123eae3137a0446ef
SHA_UTILS_FEAT=6c0aace0ea29fd3dea19737cded0c2e7308ed6d8b8f75a9123eae3137a0446ef
SHA_UTILS_MASTER=e13a7fd401f867e835aa2e8e3af9abd6a2ea8bc6f5ab7f9a77414ffeb53666bd
utils_pin() {
  case $1 in
    BASE) printf '%s\n' "$SHA_UTILS_BASE" ;;
    FEAT) printf '%s\n' "$SHA_UTILS_FEAT" ;;
    MASTER) printf '%s\n' "$SHA_UTILS_MASTER" ;;
    *) return 1 ;;
  esac
}

# The commit each tree is pinned to in the report's Provenance. A tree carries that commit plus ONE
# commit, the gh-ocannl-638 backport -- so the pin is checked as an ancestor with a bounded distance,
# not as HEAD.
# Keyed by ROLE, not by path: a path-keyed lookup returns the FIRST role whose path matches, so two
# roles sharing a checkout would silently be pinned to one commit.
tree_pin() {
  case $1 in
    BASE) printf '%s\n' 6d14f401 ;;
    FEAT) printf '%s\n' 76f50dcd ;;
    MASTER) printf '%s\n' 5d0c86d8 ;;
    *) return 1 ;;
  esac
}

# What `gh612_cells.sh` checks per cell is the local configuration and the fixture -- the inputs. It
# does NOT check the tree, and it cannot: it takes the checkout it is handed. So a tree at the wrong
# commit, with edited sources, or with a stale binary built before the backport, is measured under
# the trusted `base574A` / `feat574A` / cap label and every gate downstream passes -- parity compares
# loss vectors, replays checks timings and cache hits, and even the arm-A premise check reads what
# the binary REPORTED. A pre-backport binary ignores the flag, ships arm B and reports "B", which
# fails loudly; a HALF-backported one (sources patched, binary stale) is the quiet case, and the
# mtime check below is what catches it.
#
# Validated here rather than in the optional `trees` block, because the measurement blocks are
# independently runnable and a printed DIRTY protects nothing. Memoized: once per tree per run.
validate_tree() {  # <role> -- the tree comes from the role, so the pin cannot be picked by path
  local role=$1 t pin base extra src
  assert_distinct_roles || return 1
  t=$(role_tree "$role") || return 1
  case " ${TREES_VALIDATED:-} " in *" $role:$t "*) return 0 ;; esac
  [ -d "$t/.git" ] || [ -f "$t/.git" ] || { echo "gh612v_session: not a git checkout: $t" >&2; return 1; }
  pin=$(tree_pin "$role") || { echo "gh612v_session: unknown role '$role'" >&2; return 1; }
  base=$(git -C "$t" rev-parse --verify --quiet "$pin^{commit}") || {
    echo "gh612v_session: $t does not contain the pinned commit $pin" >&2; return 1; }
  git -C "$t" merge-base --is-ancestor "$base" HEAD 2>/dev/null || {
    echo "gh612v_session: $t is not descended from its pinned commit $pin" >&2; return 1; }
  extra=$(git -C "$t" rev-list --count "$base..HEAD" 2>/dev/null) || extra=99
  [ "${extra:-99}" -le 1 ] || {
    echo "gh612v_session: $t carries $extra commits on top of $pin; expected at most the backport" >&2
    echo "  -- an unaccounted commit changes what the label 'the report's tree' means." >&2; return 1; }
  [ -z "$(git -C "$t" status --porcelain)" ] || {
    echo "gh612v_session: $t has uncommitted changes -- refusing to measure it" >&2; return 1; }
  # EXHAUSTIVE, then exact. The digests below pin the files the backport COPIES, but a commit that
  # also touched `arrayjit/lib/schedule.ml` would still satisfy them -- so first bound the whole
  # changed-path set to the backport's own, and only then check contents within it. Enumerated
  # rather than sampled: the previous two rounds of this check each answered "what about file X?"
  # with another named file, and a set equality answers every X at once.
  local changed expected
  expected=$(printf '%s\n' arrayjit/lib/utils.ml benchmarks/runners/ocannl/bench_conv.ml \
    benchmarks/runners/ocannl/bench_gpt.ml benchmarks/runners/ocannl/bench_harness.ml \
    benchmarks/runners/ocannl/bench_mlp.ml lib/train.ml ocannl_config.reference | sort)
  changed=$(git -C "$t" diff --name-only "$base" HEAD 2>/dev/null | sort)
  [ "$changed" = "$expected" ] || {
    echo "gh612v_session: $t's commit on top of $pin touches files the backport does not:" >&2
    diff <(printf '%s\n' "$expected") <(printf '%s\n' "$changed") | sed 's/^/    /' >&2
    echo "  Anything outside this set can change lowering, scheduling or the runners, and the cells" >&2
    echo "  would still be labelled as the report's tree." >&2; return 1; }

  # CONTENT, not shape. "One clean commit on top of the pinned base, and train.ml mentions
  # tune_ship_arm" accepts a commit that also changes lowering, scheduling or the runners -- and the
  # measurements would still be certified under the report's label. What the report actually claims
  # is that five files were copied VERBATIM, so that is what is checked: their exact digests, which
  # are identical in all three trees and equal to the selector as it landed in ffc428d2. A tree
  # carrying a different revision of those files is not the tree these numbers were measured on, and
  # says so rather than being silently measured. (The tree HEADs themselves -- ca4db3bf, e6b7b415,
  # 4d1ebb11 -- are local commits that no reproduction can match, which is why the pin is content.)
  local want got
  for src in "$SHA_TRAIN lib/train.ml" "$SHA_GPT benchmarks/runners/ocannl/bench_gpt.ml" \
             "$SHA_HARNESS benchmarks/runners/ocannl/bench_harness.ml" \
             "$SHA_CONV benchmarks/runners/ocannl/bench_conv.ml" \
             "$SHA_MLP benchmarks/runners/ocannl/bench_mlp.ml"; do
    want=${src%% *}; src=${src#* }
    got=$(sha256sum "$t/$src" 2>/dev/null | cut -d' ' -f1)
    [ "$got" = "$want" ] || {
      echo "gh612v_session: $t/$src is not the backported revision this session measured" >&2
      echo "    got  ${got:-<unreadable>}" >&2; echo "    want $want" >&2; return 1; }
  done
  # utils.ml is the ONE backported file whose base content legitimately differs per tree (the config
  # machinery moved between these commits), so its digest is pinned PER ROLE rather than shared. Both
  # checks run: the digest says it is the measured file, and the structural pair says what makes it
  # the backport -- registered AND classified -- so a future re-pin cannot quietly drop either.
  # `ocannl_config.reference` is documentation that nothing in a cell reads, so the path-set
  # membership above is all it needs.
  want=$(utils_pin "$role") || return 1
  got=$(sha256sum "$t/arrayjit/lib/utils.ml" 2>/dev/null | cut -d' ' -f1)
  [ "$got" = "$want" ] || {
    echo "gh612v_session: $t/arrayjit/lib/utils.ml is not the backported revision for role $role" >&2
    echo "    got  ${got:-<unreadable>}" >&2; echo "    want $want" >&2; return 1; }
  grep -q '"tune_ship_arm"' "$t/arrayjit/lib/utils.ml" 2>/dev/null || {
    echo "gh612v_session: $t/arrayjit/lib/utils.ml does not register tune_ship_arm" >&2; return 1; }
  [ "$(grep -c '"tune_ship_arm"' "$t/arrayjit/lib/utils.ml")" -ge 2 ] || {
    echo "gh612v_session: $t/arrayjit/lib/utils.ml registers tune_ship_arm but does not classify it" >&2
    return 1; }
  # BUILD it, rather than reasoning about its timestamp. An mtime ordering says the binary is not
  # older than the sources; it does not say it was built FROM them -- a `_build` copied or restored
  # from another checkout, or produced by a dune invocation that resolved to a different root, is
  # newer and unrelated. Building here makes the provenance true instead of plausible, and costs
  # nothing when the tree is already current.
  #
  # `--root .` is not decoration: dune resolves its root by walking UP, so an invocation inside a
  # checkout nested under another dune project builds the PARENT (the trap AGENTS.md documents for
  # .claude/worktrees). Pinning the root makes "built from this tree" mean this tree.
  local exe="$t/_build/default/benchmarks/runners/ocannl/bench_gpt.exe"
  ( cd "$t" && dune build --root . benchmarks/runners/ocannl/bench_gpt.exe ) >/dev/null 2>&1 || {
    echo "gh612v_session: $t failed to build benchmarks/runners/ocannl/bench_gpt.exe" >&2
    echo "  (run 'cd $t && dune build --root . @check bin/ benchmarks/' to see the errors)" >&2
    return 1; }
  [ -x "$exe" ] || { echo "gh612v_session: $t built no bench_gpt.exe at $exe" >&2; return 1; }
  TREES_VALIDATED="${TREES_VALIDATED:-} $role:$t"
  echo "tree ok: $role = $t at $(git -C "$t" rev-parse --short=8 HEAD) (pinned base $pin, only the backport's paths, digests match, clean, rebuilt from this tree)"
}

# Unquoted on purpose: the flag string is word-split into separate arguments. It is built here, not
# taken from the caller.
# The provenance a cell carries with it. `gate` runs independently (that is the documented way to
# use it), and an artifact directory does not otherwise record WHERE it came from: a stale OUT_ROOT
# populated straight from `gh612_cells.sh` against some other checkout passes the arm-A, parity and
# replay gates under the trusted labels, because all three read the artifacts and none of them knows
# what produced them. So a search writes down the validated role, the tree, its HEAD and the backport
# digest set, and the gate requires that manifest to match the tree it validates now.
tree_fingerprint() {  # <role> -- role, path, HEAD, and the digests validate_tree just checked
  local role=$1 t; t=$(role_tree "$role") || return 1
  # The fixture is part of what a cell measured, and it is gitignored -- no commit establishes it --
  # so its digest belongs in the provenance beside the sources'.
  printf 'role=%s\ntree=%s\nhead=%s\npin=%s\nfixture_md5=%s\naffinity=%s\nsha_train=%s\nsha_utils=%s\nsha_gpt=%s\nsha_harness=%s\n' \
    "$role" "$(cd "$t" && pwd -P)" "$(git -C "$t" rev-parse HEAD)" "$(tree_pin "$role")" \
    "$(md5sum "$(readlink -f "$t/benchmarks/$FIXTURE")" | cut -d' ' -f1)" "$PIN" \
    "$(sha256sum "$t/lib/train.ml" | cut -d' ' -f1)" \
    "$(sha256sum "$t/arrayjit/lib/utils.ml" | cut -d' ' -f1)" \
    "$(sha256sum "$t/benchmarks/runners/ocannl/bench_gpt.ml" | cut -d' ' -f1)" \
    "$(sha256sum "$t/benchmarks/runners/ocannl/bench_harness.ml" | cut -d' ' -f1)"
}

run_cell() {  # <subcommand> <label> <rep> [extra driver args before the flags]
  local sub=$1 label=$2 rep=$3; shift 3
  local role tree flags rc
  role=$(cell_role "$label") || return 2
  tree=$(role_tree "$role") || return 2
  flags=$(cell_flags "$label") || return 2
  validate_tree "$role" || return 1
  # Retract the manifest this subcommand is about to replace BEFORE dispatching: a failed or
  # interrupted regeneration must not leave the previous run's provenance standing over artifacts
  # that are now half-rewritten. The driver retracts the artifacts themselves the same way.
  case $sub in
    snap|profile|replay) rm -f "$OUT_ROOT/$label/r$rep/$sub.manifest" ;;
    *) ;;
  esac
  echo "--- $sub $label r$rep [$flags]"
  "$D" ${box_gate_fwd[@]+"${box_gate_fwd[@]}"} "$sub" "$tree" "$label" "$rep" "$@" $flags
  rc=$?
  # Only a SEARCH creates the cell, and only after it publishes (the driver stages and moves on
  # success), so the manifest is written here rather than by the driver -- and only on success, so a
  # discarded cell leaves no provenance to be trusted later.
  if [ $rc -eq 0 ] && [ -d "$OUT_ROOT/$label/r$rep" ]; then
    case $sub in
      search) tree_fingerprint "$role" > "$OUT_ROOT/$label/r$rep/tree.manifest" || return 1 ;;
      # snap and profile regenerate claim-bearing artifacts (the emitted arm A source, the per-kernel
      # CSVs) LONG after the search, so they carry their own provenance: the gate would otherwise
      # accept a profile rebuilt from a different tree beside a search manifest that still matched.
      snap) tree_fingerprint "$role" > "$OUT_ROOT/$label/r$rep/snap.manifest" || return 1 ;;
      profile) tree_fingerprint "$role" > "$OUT_ROOT/$label/r$rep/profile.manifest" || return 1 ;;
      # The pass-2 replay is where every quoted step p50 and every gated loss vector comes from, and
      # it is regenerable long after the search from any checkout -- the search manifest would still
      # match, because it attests the search.
      replay) tree_fingerprint "$role" > "$OUT_ROOT/$label/r$rep/replay.manifest" || return 1 ;;
      *) ;;
    esac
  fi
  return $rc
}

CELLS_GH574="base574A feat574A"
CELLS_GH573="cap8A capoffA"
CELLS_CAPS="cap8A cap4A"

# Alternate the order across reps (gh-481): without it one arm sits permanently earlier in the
# session than the other and thermal/driver drift is indistinguishable from the treatment.
balanced_block() {  # <first-rep> <reps> <label>...
  local first=$1 reps=$2; shift 2
  local labels=("$@") r i ordered
  for r in $(seq "$first" $((first + reps - 1))); do
    if [ $(( (r - first) % 2 )) -eq 0 ]; then ordered=("${labels[@]}")
    else ordered=(); for ((i=${#labels[@]}-1; i>=0; i--)); do ordered+=("${labels[$i]}"); done; fi
    echo "=== rep $r, order: ${ordered[*]} ==="
    for l in "${ordered[@]}"; do
      # Abort the block rather than report a half-balanced series: a missing cell is not a smaller
      # experiment, it is an unbalanced one.
      run_cell search "$l" "$r" || { echo "gh612v_session: $l r$r failed; aborting this block" >&2; return 1; }
    done
  done
}

block_trees() {
  local r rc=0
  for r in BASE FEAT MASTER; do
    # The same validation every cell runs, so `trees` is a dry run of the gate rather than a
    # cosmetic listing that a measurement block could then contradict.
    validate_tree "$r" || rc=1
  done
  return $rc
}

block_replays() {  # every searched cell of every block, pass 2
  local l r rc=0
  for l in base574A feat574A cap8A capoffA; do
    for r in 1 2 3; do run_cell replay "$l" "$r" || rc=1; done
  done
  for l in cap8A cap4A; do
    for r in 4 5 6; do run_cell replay "$l" "$r" || rc=1; done
  done
  return $rc
}

block_structure() {  # rep 1 of the four structural cells: emitted source, per-kernel profile, fingerprints
  local l
  for l in base574A feat574A cap8A capoffA; do
    run_cell snap "$l" 1 && run_cell profile "$l" 1 3 && "$D" finger "$l" 1 \
      || { echo "gh612v_session: structure failed for $l" >&2; return 1; }
  done
  "$D" profiles base574A/r1 feat574A/r1 cap8A/r1 capoffA/r1 || return 1
  # The mechanism diffs, on artifacts that are now output-verified on both sides.
  "$D" diff base574A 1 feat574A 1 || return 1   # gh-574: the fused lm_head+row-max kernel disappears
  "$D" diff capoffA 1 cap8A 1     || return 1   # gh-573: the ffn_b2 triangle disappears
  "$D" diff feat574A 1 capoffA 1  || return 1   # negative control: the guard is silent without a cap
}

# The premise of the whole session, and the one thing none of the driver's gates can see: `parity`
# compares loss vectors, `replays` checks timings and cache hits, and BOTH are satisfied by a
# perfectly good arm B session. A stale pre-backport binary in one worktree, a dropped flag in one
# subcommand, or a future driver that forgets to forward it, would then produce a green gate over
# exactly the arm-B artifacts this session exists to replace -- the original limitation, recreated
# and certified. So assert it directly, over BOTH passes: the searches say which arm was kept, and
# the pass-2 replays say which arm was kept AGAIN in the process whose timings are quoted.
assert_shipped_arm_A() {  # <cell>...
  local bad=0 n=0 cell kind f arm
  for cell in "$@"; do
    for kind in search.out replay2.out; do
      f="$OUT_ROOT/${cell%/*}/${cell#*/}/$kind"
      [ -s "$f" ] || { echo "gh612v_session: missing record $f" >&2; bad=1; continue; }
      # The LAST JSON line, matching what `parity` reads: a cell's stdout can carry earlier records
      # (bench_output.result_record, which strict_json.py's field command reads it through).
      arm=$(python3 "$HERE/strict_json.py" field "$f" tune.shipped 2>/dev/null)
      n=$((n + 1))
      [ "${arm:-?}" = "A" ] || {
        echo "gh612v_session: $cell $kind shipped arm ${arm:-?}, not A -- the profiled arm is NOT the" >&2
        echo "  executed one, which is the exact gap this session closes. Refusing to certify it." >&2
        bad=1; }
    done
  done
  [ "$n" -gt 0 ] || { echo "gh612v_session: no records to check" >&2; return 1; }
  [ "$bad" -eq 0 ] || return 1
  echo "arm-A premise: $n records (search + pass-2) all shipped arm A"
}

# Every expected cell must carry the manifest its search wrote, and it must match the tree this run
# validates for that cell's role. Without it the gate certifies artifacts, not measurements: the
# labels are just directory names.
assert_cell_provenance() {  # <cell>...
  local bad=0 n=0 back=0 cell role f want
  for cell in "$@"; do
    role=$(cell_role "${cell%/*}") || return 1
    validate_tree "$role" || return 1
    f="$OUT_ROOT/${cell%/*}/${cell#*/}/tree.manifest"
    [ -s "$f" ] || {
      echo "gh612v_session: $cell has no tree.manifest -- it was not produced by this session's" >&2
      echo "  driver, so nothing records which checkout measured it." >&2; bad=1; continue; }
    want=$(tree_fingerprint "$role") || return 1
    [ "$(cat "$f")" = "$want" ] || {
      echo "gh612v_session: $cell was measured on a different tree than role $role resolves to now:" >&2
      diff <(printf '%s\n' "$want") "$f" | sed 's/^/    /' >&2; bad=1; continue; }
    n=$((n + 1))
    # A manifest written after the fact attests the binding rather than recording it, so it is
    # counted and reported separately rather than being indistinguishable from one a search wrote.
    [ -e "$f.backfilled" ] && back=$((back + 1))
    # The derived artifacts, each against its own manifest: an emitted source (snap) and a
    # per-kernel profile can be regenerated from another checkout while the search manifest stays
    # true, and Part 5's per-kernel claim rests on exactly such a regeneration.
    local d="$OUT_ROOT/${cell%/*}/${cell#*/}" k
    # The four structural cells and cap4A/r4 MUST carry the derived artifacts: Parts 3-5 quote their
    # per-kernel numbers and their diffs. Elsewhere the artifacts are optional (only the searched
    # cells' replays are universal), but where a claim rests on one, absent is a failure, not a skip
    # -- otherwise `gate` certifies a session whose profiles were never produced.
    local required=""
    case $cell in
      base574A/r1|feat574A/r1|cap8A/r1|capoffA/r1|cap4A/r4) required="snap profile" ;;
    esac
    for k in snap:armA.path profile:kernels-1.csv replay:replay2.out; do
      if [ ! -e "$d/${k#*:}" ]; then
        case " $required " in
          *" ${k%%:*} "*)
            echo "gh612v_session: $cell is missing ${k#*:}, which a claim in the report rests on --" >&2
            echo "  run the 'structure' and 'capprofile' blocks before the gate." >&2; bad=1 ;;
        esac
        continue
      fi
      [ -s "$d/${k%%:*}.manifest" ] || {
        echo "gh612v_session: $cell has ${k#*:} but no ${k%%:*}.manifest -- that artifact was not" >&2
        echo "  produced through this session's wrapper, so its tree is unrecorded." >&2; bad=1; continue; }
      [ "$(cat "$d/${k%%:*}.manifest")" = "$want" ] || {
        echo "gh612v_session: $cell's ${k%%:*} artifacts come from a different tree than role $role" >&2
        bad=1; }
    done
  done
  # Three harness runs, not one: every profile total the report quotes is a three-run median, and a
  # profile regenerated with a smaller count leaves the earlier CSVs' manifest matching while the
  # readers (finger, diff) silently median over whatever they discover. The driver's own gate counts
  # them, so use it rather than re-deriving the rule here.
  [ "$bad" -eq 0 ] && { "$D" profiles base574A/r1 feat574A/r1 cap8A/r1 capoffA/r1 cap4A/r4 \
    > /dev/null || { echo "gh612v_session: the claim-bearing profiles are incomplete (three harness" >&2
      echo "  runs each are required; run 'structure' and 'capprofile')" >&2; bad=1; }; }
  [ "$bad" -eq 0 ] || return 1
  if [ "${back:-0}" -gt 0 ]; then
    echo "provenance: $n cells carry a manifest matching their validated tree ($back backfilled --"
    echo "  see the .backfilled note beside each; those cells predate this check)"
  else
    echo "provenance: $n cells carry a manifest matching their validated tree"
  fi
}

# Part 5's per-kernel instrument for cap 4, which `structure` does not cover (it profiles the four
# cells the ratios are built from). Routed through `run_cell` like everything else: the earlier
# revision of the report told the reader to call `gh612_cells.sh` directly here, which skipped tree
# validation and left the profile without provenance -- while the gate stayed green, because it only
# knew about searches.
block_capprofile() {
  run_cell snap cap4A 4 && run_cell profile cap4A 4 3 && "$D" finger cap4A 4 || return 1
  "$D" diff cap8A 1 cap4A 4 || return 1
}

block_gate() {
  local want=""
  local l r
  for l in base574A feat574A cap8A capoffA; do for r in 1 2 3; do want="$want $l/r$r"; done; done
  for l in cap8A cap4A; do for r in 4 5 6; do want="$want $l/r$r"; done; done
  # FIRST of all: the artifacts must come from the trees this run validates. `gate` is documented as
  # independently runnable, so without this it reads whatever is in OUT_ROOT and certifies it.
  assert_cell_provenance $(echo "$want") || return 1
  # Then the premise the other two gates presuppose and cannot test: a green parity gate over arm B
  # artifacts is a correct answer to the wrong question.
  assert_shipped_arm_A $(echo "$want") || return 1
  # EXPECT_CELLS pins the exact set, so a stale OUT_ROOT cannot substitute one cell for another, and
  # it requires a pass-2 loss vector per cell -- a timed artifact that was never output-verified is
  # exactly what this session exists to eliminate.
  EXPECT_CELLS="${want# }" "$D" parity || return 1
  "$D" replays $(echo "$want") || return 1
}

[ $# -ge 1 ] || { sed -n '/^# Usage:/,/^#   gate/p' "$0"; exit 2; }
rc=0
for block in "$@"; do
  echo "########## block: $block ($(date -u +%H:%M:%SZ))"
  case $block in
    trees)     block_trees || rc=1 ;;
    gh574)     balanced_block 1 3 $CELLS_GH574 || rc=1 ;;
    gh573)     balanced_block 1 3 $CELLS_GH573 || rc=1 ;;
    caps)      balanced_block 4 3 $CELLS_CAPS  || rc=1 ;;
    replays)   block_replays   || rc=1 ;;
    structure) block_structure || rc=1 ;;
    capprofile) block_capprofile || rc=1 ;;
    gate)      block_gate      || rc=1 ;;
    *) echo "gh612v_session: unknown block '$block'" >&2; rc=2 ;;
  esac
  [ $rc -eq 0 ] || { echo "gh612v_session: block '$block' failed; stopping" >&2; exit $rc; }
done
exit $rc

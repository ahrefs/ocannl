#!/bin/bash
# gh-612 measurement cells: the gh-573 / gh-574 HIP legs against one re-established denominator.
# Driver for benchmarks/report-gh612-hip.md. Every number in that report comes from these
# subcommands, so the report quotes invocations rather than restating commands — a transcribed
# command drifts from the one that ran, and four of this session's review findings were exactly
# that drift.
#
# Usage:
#   gh612_cells.sh search  <tree> <label> <rep> [extra ocannl flags...]
#   gh612_cells.sh snap    <tree> <label> <rep> [extra ocannl flags...]
#   gh612_cells.sh replay  <tree> <label> <rep> [extra ocannl flags...]   # PASS 2: step timings
#   gh612_cells.sh profile <tree> <label> <rep> [n-harness-runs]
#   gh612_cells.sh finger  <label> <rep>
#   gh612_cells.sh parity                       # the correctness gate over every cell
#   gh612_cells.sh replays <label/rep>...       # assert every cell has an accepted pass-2 replay
#   gh612_cells.sh profiles <label/rep>...      # assert every cell has a complete profile
#   gh612_cells.sh diff    <labelA> <repA> <labelB> <repB>
#   gh612_cells.sh sweep   <tree> <reps> <cap>...      # balanced cap order, see below
#   gh612_cells.sh roofline <tree>
#
# <tree> is a built checkout (dune build @check bin/ benchmarks/) whose
# benchmarks/fixtures/gpt2_mini.safetensors is the SAME file in every tree under comparison —
# symlink one fixture rather than regenerating per tree, so the input cannot differ.
#
# Results land under $OUT_ROOT/<label>/r<rep> (default /tmp/gh612).
set -u
# Hermetic against ambient configuration, categorically -- the same treatment gh514_cells.sh applies,
# and for the same reason: every cell pins its treatment on the command line (which out-ranks all
# other config sources), so an exported OCANNL_* or BENCH_* would otherwise reshape the graph or the
# precision while every cell was contaminated CONSISTENTLY -- which the parity gate cannot see, since
# it compares cells against each other. `BENCH_MATERIALIZE=1` or `BENCH_PRECISION=f16` in the
# environment would silently make "default-placement f32" a false label on every number here.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')

# Every subcommand `cd`s into a checkout, so any caller-supplied path that has to survive that cd
# must be absolutized HERE, before the first cd — a relative OUT_ROOT (or <tree>) would otherwise be
# created next to the invocation directory and then re-resolved inside the checkout, and the first
# redirection into it fails. FIXTURE is the deliberate exception: it is relative to <tree>/benchmarks,
# which is exactly where the runner is invoked from.
#
# And no path may be produced by a command substitution that yields an EMPTY string on failure:
# `rm -rf "$OUT_ROOT/$label/r$rep"` with an empty OUT_ROOT is `rm -rf /<label>/r<rep>`. `set -u` does
# not catch that (the variable is set, just empty), so resolution ABORTS instead of falling through.
resolve_dir() {                      # absolute PHYSICAL path to an existing usable directory, or abort
  local d=$1 abs
  mkdir -p "$d" 2>/dev/null || { echo "gh612_cells: cannot create directory: $d" >&2; exit 2; }
  abs=$(cd "$d" 2>/dev/null && pwd -P) || abs=""
  [ -n "$abs" ] && [ -d "$abs" ] || { echo "gh612_cells: not a usable directory: $d" >&2; exit 2; }
  printf '%s\n' "$abs"
}
# PHYSICAL, not logical: a symlinked <tree> would otherwise make check_config walk the alias's
# ancestors while the benchmark process runs with the physical directory as its kernel cwd -- so a
# local ocannl_config beside or above the real checkout would be consumed without ever being
# inspected. Validation has to follow the same ancestor chain configuration lookup does.
require_dir() {                      # like resolve_dir but must already exist (never creates)
  local d=$1 abs
  abs=$(cd "$d" 2>/dev/null && pwd -P) || abs=""
  [ -n "$abs" ] && [ -d "$abs" ] || { echo "gh612_cells: no such directory: $d" >&2; exit 2; }
  printf '%s\n' "$abs"
}
# NOTE: an `exit` inside $( ) exits only the substitution's subshell, so the abort has to be
# re-asserted at the call site. Every use of resolve_dir/require_dir is followed by this check.
OUT_ROOT=$(resolve_dir "${OUT_ROOT:-/tmp/gh612}")
OUT_ROOT_P=$(cd "$OUT_ROOT" 2>/dev/null && pwd -P) || OUT_ROOT_P=""
case ${OUT_ROOT:-} in /?*) ;; *) echo "gh612_cells: OUT_ROOT unusable, refusing to run" >&2; exit 2;; esac
PIN=${PIN:-taskset -c 0-15}
FIXTURE=${FIXTURE:-fixtures/gpt2_mini.safetensors}
# The fixture is GITIGNORED, so no checkout establishes its bytes. A stale or differently generated
# one is used consistently by every cell, which the parity gate cannot see -- it compares cells with
# each other, not with the workload the report names. Pin the digest of the artifact the published
# numbers were measured on; FIXTURE_MD5= (empty) opts out deliberately.
FIXTURE_MD5=${FIXTURE_MD5-5b3dfff860fc8c54af2a7d440f4cf202}
check_fixture() {
  local tree=$1 f="$1/benchmarks/$FIXTURE" got
  [ -e "$f" ] || { echo "gh612_cells: fixture missing: $f" >&2; return 2; }
  [ -n "${FIXTURE_MD5:-}" ] || return 0
  got=$(md5sum "$(readlink -f "$f")" 2>/dev/null | cut -d" " -f1)
  [ "$got" = "$FIXTURE_MD5" ] || {
    echo "gh612_cells: fixture digest mismatch -- refusing to measure a different workload" >&2
    echo "    $f" >&2; echo "    got  $got" >&2; echo "    want $FIXTURE_MD5" >&2; return 2; }
  return 0
}
ARCH=${ARCH:-gfx1151}
HERE=$(cd "$(dirname "$0")" && pwd)   # the driver's own directory: cmd_finger does not cd into the
                                      # tree, so its helper-script paths must be absolute
EXE=../_build/default/benchmarks/runners/ocannl/bench_gpt.exe
ROUTINE=cross_entropy_loss_fwd

# The autotune cache dir is per label AND per rep, never shared: a warm cache makes an A/B vacuous by
# replaying the other arm's crowned schedule (report-gh481-cuda.md measured compile_s 1.76 s warm
# against 29 s cold, with the "winner" being whatever the other arm found), and a rep that replays is
# not an independent search. NOT because of cross-Numerics aliasing -- gh-ocannl-568 fixed that and
# `numerics` is a Schedule_cache key component, so naming it here would reverse its own fix.
# Third path-safety finding in a row (relative paths, then empty paths, now traversal), so the gate
# lives HERE -- the single place every subcommand's paths are built -- rather than at each `rm -rf`.
# A label is caller input and feeds `rm -rf`: `../../../home/me/data` would escape OUT_ROOT entirely.
# Two independent checks, because either alone is bypassable: a character whitelist, and a
# containment test on the resolved parent.
# Configuration is searched from the invocation directory UPWARDS (Utils.config_file_args), and the
# root `ocannl_config` is gitignored -- so a reused worktree can carry one that `git status
# --porcelain` never reports, and it silently overrides defaults. A stray
# `virtualize_max_inline_fanin=4` would make every nominal cap-8 cell measure cap 4 while the report
# labels it by the verified commit. Refuse rather than measure through it.
check_config() {
  local tree=$1 d="$1" found=""
  while :; do
    [ -f "$d/ocannl_config" ] && found="$found $d/ocannl_config"
    [ "$d" = "/" ] && break
    d=$(dirname "$d")
  done
  [ -z "$found" ] || {
    echo "gh612_cells: local ocannl_config found -- it overrides defaults silently and is gitignored:" >&2
    for f in $found; do echo "    $f" >&2; done
    echo "  Remove it (or measure elsewhere) before running cells." >&2; return 2; }
  # the tracked benchmarks/ocannl_config must also match HEAD: it is the config the cells actually
  # run under, and an edit to it is invisible to the worktree-cleanliness check if it is staged away.
  git -C "$tree" diff --quiet HEAD -- benchmarks/ocannl_config 2>/dev/null || {
    echo "gh612_cells: $tree/benchmarks/ocannl_config differs from HEAD -- refusing to measure" >&2; return 2; }
  return 0
}

cell_dir() {
  local label=$1 rep=$2
  case $label in
    ""|*/*|*..*) echo "gh612_cells: bad label '$label' (no '/', no '..')" >&2; exit 2;;
    *[!A-Za-z0-9._-]*) echo "gh612_cells: bad label '$label' (allowed: A-Za-z0-9._-)" >&2; exit 2;;
    # RESERVE the dot-prefixed namespace. Staging lives at `.staging-<label>-r<rep>`, and the
    # whitelist above would otherwise accept that very string as a LABEL -- so a published
    # `.staging-foo-r1/r9` cell would be destroyed by `search ... foo 1`, whose staging path is the
    # same directory. A leading dot also keeps parity's `.staging` skip from hiding a real cell.
    .*) echo "gh612_cells: labels may not start with '.' (reserved for staging)" >&2; exit 2;;
  esac
  case $rep in ""|*[!0-9]*) echo "gh612_cells: bad rep '$rep' (digits only)" >&2; exit 2;; esac
  # Belt and braces: resolve the parent that will actually be operated on and require it to sit
  # beneath OUT_ROOT, so a whitelist gap cannot turn into a deletion outside the results tree.
  # A pre-existing SYMLINK at $OUT_ROOT/<label> passes a logical-path check -- `cd`+`pwd` reports the
  # path beneath OUT_ROOT -- while `rm -rf` follows it and deletes inside the link's target. Resolve
  # physically (`pwd -P`) and require the physical parent beneath the physical root.
  # BOTH components must be checked, not just the label: `$OUT_ROOT/<label>/r<rep>` can itself be a
  # symlink, and every caller then deletes through it (armA.path, armsnap, the profile artifacts).
  # So resolve the FULL cell directory physically and require the result under the physical root.
  [ -L "$OUT_ROOT/$label" ] && { echo "gh612_cells: refusing symlinked cell parent: $OUT_ROOT/$label" >&2; exit 2; }
  [ -L "$OUT_ROOT/$label/r$rep" ] && { echo "gh612_cells: refusing symlinked cell dir: $OUT_ROOT/$label/r$rep" >&2; exit 2; }
  [ -n "${OUT_ROOT_P:-}" ] || { echo "gh612_cells: OUT_ROOT has no physical path" >&2; exit 2; }
  local cell
  cell=$(mkdir -p "$OUT_ROOT/$label/r$rep" 2>/dev/null && cd "$OUT_ROOT/$label/r$rep" && pwd -P) || cell=""
  case ${cell:-} in
    "$OUT_ROOT_P"/*) ;;
    *) echo "gh612_cells: cell path escaped OUT_ROOT physically: '$OUT_ROOT/$label/r$rep' -> '$cell'" >&2; exit 2;;
  esac
  printf '%s\n' "$cell"
}

cmd_search() { (
  local tree=$1 label=$2 rep=$3; shift 3
  # Same rule as snap/replay/profile, and the FIFTH place it was needed: resolve the cell and clear
  # it before ANY other validation. A retry against a moved checkout would otherwise exit at the
  # tree check with the old search.out, cache, replay and profile artifacts still published -- and
  # the parity/replays/profiles gates would then certify the rejected retry from them.
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
  # STAGE, then publish atomically on success. Cleaning up after a nonzero child is not enough: a
  # signal to the wrapper (SIGTERM to the process group after the benchmark has emitted its JSON)
  # skips the cleanup branch entirely and leaves search.out and a populated cache in their accepted
  # locations, where a later replay uses the cache and the artifact gates certify the cell. Writing
  # into a staging dir means an interrupted run publishes nothing at all.
  # Staging lives OUTSIDE the `<label>/r<rep>` namespace: as a sibling `<cell>.staging` its
  # search.out sat at the same depth as a real cell's and matched parity's `*/r*/search.out`
  # discovery, so a leftover staging dir counted as an accepted search record -- and a parity run
  # could pass on staging directories alone. One level under OUT_ROOT cannot match that glob.
  local stage="$OUT_ROOT/.staging-$label-r$rep"
  rm -rf "$out" "$stage"; mkdir -p "$stage" || exit 2
  # STAGE is referenced by the handler at trap time, NOT interpolated into it: an OUT_ROOT
  # containing whitespace would otherwise be word-split by `rm -rf` and delete a sibling path.
  STAGE=$stage
  trap 'rm -rf "$STAGE"' EXIT INT TERM
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  check_config "$tree" || exit 2
  check_fixture "$tree" || exit 2
  cd "$tree/benchmarks" || exit 1
  local t0=$SECONDS
  BENCH_FIXTURE=$FIXTURE BENCH_TUNE=1 $PIN "$EXE" --ocannl_backend=hip \
    --ocannl_autotune_cache_dir="$stage/cache" \
    --ocannl_autotune_log=true --ocannl_schedule_log_declines=true "$@" \
    > "$stage/search.out" 2> "$stage/search.err"
  local st=$?
  # A nonzero exit can still leave a complete search.out and a populated cache, and the parity /
  # replays / profiles gates validate ARTIFACTS rather than this status -- so a failed search would
  # be certified as a completed cell. Publish only on success.
  if [ "$st" -ne 0 ]; then
    echo "$label r$rep: exit $st, $((SECONDS - t0))s -- discarding the cell" >&2
    rm -rf "$stage"; trap - EXIT INT TERM; return "$st"
  fi
  rm -rf "$out" && mv "$stage" "$out" || { echo "gh612_cells: could not publish $label r$rep" >&2; return 1; }
  trap - EXIT INT TERM
  echo "$label r$rep: exit $st, $((SECONDS - t0))s"
  grep -h '^{' "$out/search.out" | tail -1
  # The three claim-bearing lines: the deterministic untuned baseline, the crowned artifact per
  # arm, and which arm shipped.
  grep -E 'untuned-default pipeline|winner replay ok|tune_placements: winner' "$out/search.err"
  # Each fissioned winner's segmentation: entries persist every segment with its kind
  # (gh-ocannl-1164), so the count tells a fine winner from a coarse one.
  local entry kinds
  for entry in "$out"/cache/*.sexp; do
    [ -e "$entry" ] || continue
    kinds=$(grep -o '(seg_kind [A-Za-z]*)' "$entry" | sed 's/(seg_kind \(.*\))/\1/' | sort | uniq -c \
      | awk '{ printf "%s%s %s", sep, $1, $2; sep = ", " }' || true)
    [ -n "$kinds" ] && echo "fissioned winner $(basename "$entry"): $kinds"
  done
  # The crowned arm A candidate's calibration line: analytic FLOPs and bytes at the emitted kernel
  # count. The report's constant-FLOPs check and its 528 -> 472 MB traffic figure are read from here.
  local lbl; lbl=$(sed -n '1,/arm A (default placements) best/p' "$out/search.err" \
    | grep -o 'best: [0-9.]* ms ([A-Za-z_]*\[[^]]*\]' | sed 's/.*(//')
  [ -n "${lbl:-}" ] && grep -F "calibration: $lbl" "$out/search.err" | head -1
  # Return the BENCHMARK's status, not the last grep's. Printing evidence from a failed run and then
  # exiting 0 is how cmd_sweep came to accept a failed cell and still call the series balanced.
  return "$st"
) }

# Warm replay of the cell's cached winner, capturing the emitted source and the launch geometry.
# Arm A compiles first and arm B overwrites the same path, so snapshot by polling on content.
cmd_snap() { (
  local tree=$1 label=$2 rep=$3; shift 3
  # ORDER IS THE RULE, and it has now been got wrong three times (after the cache check, after the
  # tree check, in two functions): resolve the cell and RETRACT its published artifacts before ANY
  # other validation. Every exit from that point is a rejection, and a stale armA.path would let
  # `diff`/`finger`/`profile` keep consuming the previous accepted source as this cell's evidence --
  # and the `replays`/`profiles` gates would count it as coverage. Publish-on-success only.
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
  # ...including any provenance manifest describing them (gh612v_session.sh writes one per
  # producing subcommand). Retracted HERE rather than only in that wrapper: this entry point stays
  # public, and a direct invocation that rewrites the artifacts while a matching manifest survives
  # is exactly a foreign artifact wearing the previous run's provenance.
  rm -f "$out/armA.path" "$out/snap.manifest"
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  check_config "$tree" || exit 2
  check_fixture "$tree" || exit 2
  # A cell with no populated cache cannot be replayed: with BENCH_TUNE=1 and an empty cache dir this
  # would run a NEW cold search and crown a different artifact, so a structural diff would compare
  # something the report never measured. Refuse instead.
  [ -d "$out/cache" ] && [ -n "$(ls -A "$out/cache" 2>/dev/null)" ] || {
    echo "gh612_cells: $label r$rep has no populated autotune cache -- run \`search\` first;" >&2
    echo "  refusing to replay, because that would silently become a fresh cold search" >&2; exit 2; }
  local snap="$out/armsnap"; rm -rf "$snap"; mkdir -p "$snap"
  cd "$tree/benchmarks" || exit 1
  rm -rf build_files
  local f=build_files/bench_gpt/${ROUTINE}__seg.hip
  ( while :; do
      if [ -f "$f" ]; then h=$(md5sum "$f" 2>/dev/null | cut -d' ' -f1)
        [ -n "$h" ] && [ ! -f "$snap/$h.hip" ] && cp "$f" "$snap/$h.hip"; fi
      sleep 0.02
    done ) & local w=$!
  # An interrupted `snap` would otherwise orphan this poller, and a later retry that recreates
  # armsnap would have the stray copying sources from the earlier run into it -- contaminating the
  # candidate set pick_armA chooses from. Kill it on any exit or signal, not only the normal path.
  trap 'kill '"$w"' 2>/dev/null' EXIT INT TERM
  # Same hole `replay` had: a populated cache DIRECTORY is not a hit, so a differing tree or
  # cache-key flag would let this search afresh and `pick_armA` would snapshot a NEWLY crowned
  # artifact instead of the measured one -- silently corrupting every structural diff built on it.
  BENCH_FIXTURE=$FIXTURE BENCH_TUNE=1 $PIN "$EXE" --ocannl_backend=hip \
    --ocannl_autotune_cache_dir="$out/cache" --ocannl_autotune_search=false \
    --ocannl_autotune_log=true \
    --ocannl_output_debug_files_in_build_directory=true \
    --ocannl_schedule_log_launches=true "$@" \
    > "$out/replay.out" 2> "$out/launches.err"
  local st=$?
  kill $w 2>/dev/null; wait $w 2>/dev/null; trap - EXIT INT TERM
  echo "$label r$rep replay: exit $st"
  local hits; hits=$(grep -c 'cache hit:' "$out/launches.err" 2>/dev/null || echo 0)
  [ "${hits:-0}" -ge 2 ] || {
    echo "gh612_cells: $label r$rep snapshot is NOT a replay -- $hits cache hits (want 2, one per" >&2
    echo "  arm). With autotune_search=false a miss ships the untuned default, so the emitted source" >&2
    echo "  would not be the measured cell's. Re-run \`search\` for this cell." >&2; exit 2; }
  # A failed replay can still leave a complete source and launch log behind (it may have died after
  # compiling), so `pick_armA` succeeding proves nothing about the run. Refuse the artifact.
  [ "$st" -eq 0 ] || { echo "gh612_cells: replay failed (exit $st); refusing the snapshot" >&2; exit "$st"; }
  pick_armA "$out"
) }

# Arm A is the FIRST fissioned compile of the routine in the launch log; the `seg i/N` totals are
# the real fission widths. Do NOT read the count off an `F_saved[fine N segs]` label -- that N is
# the number of saved per-segment PLACEMENT ENTRIES, and on this workload it reads 77 where the
# emitted arm holds 136 kernels.
#
# Then validate before selecting. A content-polling watcher can copy a partially written file, and
# a truncated capture can already carry every `__global__` line while its last function body is
# incomplete -- so the kernel count alone does not identify a usable snapshot, and glob order is
# hash order, i.e. arbitrary. Require balanced braces AND a clean compile, which is the check
# report-gh569-hip.md used and the only one that rules out a torn tail.
pick_armA() {
  local out=$1
  local n; n=$(grep -o "$ROUTINE seg 0/[0-9]*" "$out/launches.err" | awk -F/ '$2>1{print $2; exit}')
  [ -n "$n" ] || { echo "no fissioned compile in $out/launches.err" >&2; return 1; }
  local f
  for f in "$out"/armsnap/*.hip; do
    [ "$(grep -c '__global__' "$f")" = "$n" ] || continue
    python3 - "$f" <<'EOF' || continue
import sys
s=open(sys.argv[1]).read()
sys.exit(0 if s.count('{') == s.count('}') and s.count('{') > 0 else 1)
EOF
    hipcc --offload-arch="$ARCH" -O2 -include hip/hip_runtime.h -c -o /dev/null "$f" \
      2>/dev/null || { echo "  $(basename "$f"): $n kernels but INCOMPLETE (no compile)" >&2; continue; }
    # Canonical ABSOLUTE path: `finger` and `diff` open this from the caller's directory, not from
    # the tree, so a relative entry would fail there -- or silently match an unrelated file. It is
    # absolute today because $out is, but the invariant is asserted rather than inherited.
    f=$(cd "$(dirname "$f")" && pwd)/$(basename "$f")
    case $f in /?*) ;; *) echo "gh612_cells: refusing non-absolute armA path: $f" >&2; return 2;; esac
    echo "$f" > "$out/armA.path"
    echo "  arm A = $f ($n kernels, braces balanced, compiles)"
    return 0
  done
  echo "no complete snapshot with $n kernels in $out/armsnap" >&2; return 1
}

cmd_profile() { (
  local tree=$1 label=$2 rep=$3 n=${4:-3}
  # Same rule as `snap` and `replay`, and the fourth place it was needed: resolve the cell and
  # retract the published profile artifacts before ANY other validation, tree check included.
  # Otherwise a rejected retry leaves the old CSVs and buckets, and the `profiles` gate certifies
  # the rejected run using stale measurements.
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
  rm -f "$out"/kernels-*.csv "$out"/kernels-*.err "$out"/bucket-*.txt "$out/profile.manifest"
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  local src; src=$(cat "$out/armA.path")
  # Clear this subcommand's OWN artifact set before regenerating it. Every producing subcommand here
  # does that (`search` rm -rf's the cell, `snap` the snapshot dir); `profile` writing a
  # caller-chosen NUMBER of files is the case where forgetting it is silent rather than obvious --
  # a later run with a smaller count leaves the earlier run's CSVs behind and every consumer
  # medians over a mixture of two profiles.
  cd "$tree/benchmarks" || exit 1
  python3 gpt2_kernel_harness.py --source "$src" --launches "$out/launches.err" \
          --out "$out/harness.hip" || exit 1
  hipcc --offload-arch="$ARCH" -O2 -o "$out/harness" "$out/harness.hip" || exit 1
  # Stage the evidence files. Shell redirection creates bucket-$i.txt the instant the command
  # starts, so an interruption partway through the LAST bucket run leaves 3 CSVs and 3 bucket files
  # in place -- and `profiles`, which counts filenames, accepts a truncated bucket table that Part 1
  # consumes as stability evidence. Publishing only after every run completes removes that state.
  local pstage="$OUT_ROOT/.staging-$label-r$rep-profile"
  rm -rf "$pstage"; mkdir -p "$pstage" || exit 2
  PSTAGE=$pstage
  trap 'rm -rf "$PSTAGE"' EXIT INT TERM
  local i
  for i in $(seq "$n"); do
    if ! $PIN "$out/harness" > "$pstage/kernels-$i.csv" 2> "$pstage/kernels-$i.err"; then
      # Clear the ENTIRE set, not just this run: `finger` accepts however many CSVs it discovers, so
      # leaving runs 1..i-1 behind turns a failed 3-run profile into plausible 1- or 2-run medians.
      echo "gh612_cells: harness run $i failed; discarding the whole profile for this cell" >&2
      rm -rf "$pstage"; trap - EXIT INT TERM; return 1
    fi
    # stderr's last line is the sum-vs-step validation the report quotes. Pair it against THIS
    # cell's own step p50 (search.out / replay.out): another rep is a different crowned artifact,
    # so pairing across reps compares a profile to a step it is not a profile of.
    tail -1 "$pstage/kernels-$i.err"
    if ! python3 gpt2_bucket.py --source "$src" --stats "$pstage/kernels-$i.csv" --steps 1 \
            > "$pstage/bucket-$i.txt" 2>&1; then
      echo "gh612_cells: gpt2_bucket.py failed on run $i; discarding the whole profile" >&2
      sed -n '$p' "$pstage/bucket-$i.txt" >&2
      rm -rf "$pstage"; trap - EXIT INT TERM; return 1
    fi
  done
  # every run completed: publish atomically
  rm -f "$out"/kernels-*.csv "$out"/kernels-*.err "$out"/bucket-*.txt
  mv "$pstage"/* "$out"/ || { echo "gh612_cells: could not publish the profile" >&2; return 1; }
  rmdir "$pstage"; trap - EXIT INT TERM
  # `profile` always measures the saved ARM A source, but the step p50 belongs to whichever arm the
  # search SHIPPED. Pairing them is only valid where arm A shipped; elsewhere the sum and the step
  # describe different routines and printing them together would invite exactly the wrong inference.
  local shipped; shipped=$(grep -ho '"shipped":"[AB]"' "$out/search.out" 2>/dev/null | head -1 | grep -o '[AB]')
  if [ "${shipped:-}" = "A" ]; then
    # ONLY the pass-2 replay. search.out's p50 is a pass-1 timing carrying search-process residue,
    # which the protocol rejects -- printing it here would re-offer it as valid paired evidence.
    if [ -s "$out/replay2.out" ]; then
      echo "  paired step p50 (pass-2 replay2.out; this cell shipped arm A, so the pairing is valid):"
      grep -ho '"p50":[0-9.]*' "$out/replay2.out" 2>/dev/null
    else
      echo "  NO paired step p50 yet: run \`replay\` (pass 2) for this cell. The pass-1 p50 in"
      echo "  search.out is not a valid comparand -- it carries the search process's own overhead."
    fi
  else
    echo "  NO paired step p50: this cell shipped arm ${shipped:-?}, and the profile above is arm A."
    echo "  An arm-B step time is not a validation of an arm-A kernel sum; they are different routines."
  fi
  # Every run, not just the first: Part 1 quotes run-1/2/3 shares as its stability evidence, so a
  # transcript that shows only run 1 cannot expose it.
  local bf
  for bf in "$out"/bucket-*.txt; do
    echo "--- $(basename "$bf") ---"
    sed -n '/| bucket/,/directly seeded/p' "$bf"
  done
) }

# The acceptance fingerprints, read off the emitted source. Kernel parameter lists span multiple
# lines, so this cannot be done with a line-oriented grep -- `[^)]*` hits the newline before it
# ever sees the closing paren and every pattern silently matches nothing. Parse with re.S, as
# gpt2_kernel_harness.py does.
cmd_finger() {
  local out; out=$(cell_dir "$1" "$2")
  case ${out:-} in /?*) ;; *) return 2;; esac
  # `|| return` on the COMMAND line: absent fingerprints must not look like passing ones. Without
  # it the geometry dump below still succeeds and `finger` exits 0 having printed no evidence.
  python3 - "$(cat "$out/armA.path")" "$out" "$HERE" <<'EOF' || return 1
import re,sys,csv,statistics,collections
src=open(sys.argv[1]).read(); out=sys.argv[2]
SIG=re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)', re.S)
sigs={int(i):[" ".join(p.split()).split()[-1].lstrip("*") for p in ps.split(",") if p.strip()]
      for _,i,ps in SIG.findall(src)}
# DISCOVER the CSV set rather than assuming how many runs `profile` was asked for: it takes a
# caller-chosen count, so a hard-coded 1..3 silently drops runs above 3 and, before `profile`
# started clearing its own outputs, could median across two different profiles.
import glob, os
t=collections.defaultdict(list)
csvs=sorted(glob.glob(os.path.join(out,"kernels-*.csv")))
if not csvs: sys.exit(f"no kernels-*.csv in {out} -- run `profile` first")
for f in csvs:
    for r in csv.DictReader(open(f)):
        t[int(r["Name"].rsplit("__seg",1)[1])].append(float(r["TotalDurationNs"])/1e6)
ms={i:statistics.median(v) for i,v in t.items()}
# A partial CSV must not be silently completed with zeros: `ms.get(i, 0)` below would then produce
# plausible totals over kernels that were never timed.
missing=[i for i in sigs if i not in ms]
if missing:
    sys.exit(f"{len(missing)} of {len(sigs)} emitted kernels have no timing (seg {missing[:5]}...) -- "
             "the CSV set is incomplete; re-run `profile`")
print(f"medians over {len(csvs)} harness run(s): {', '.join(os.path.basename(f) for f in csvs)}")
def show(title, pred):
    hits=[i for i in sorted(sigs) if pred(sigs[i])]
    tot=sum(ms.get(i,0) for i in hits)
    print(f"{title}: n={len(hits)} {tot:.3f} ms")
    for i in hits:
        pre=[n for n in sigs[i] if re.fullmatch(r'l\d+_ffn_b2', n)]
        print(f"  seg{i:<4} {ms.get(i,0):7.4f} ms  params={len(sigs[i]):2} ffn_b2_prefix={len(pre)}"
              f"  {', '.join(sigs[i])}")
# gh-573: the LayerNorm prefix must be BOUNDED and must RESET, not ramp with depth.
show("LayerNorm sites", lambda ns: any(n.startswith(("gamma_","beta_")) for n in ns))
# gh-574: the lm_head GEMM alone, the row-max its own kernel -- not one fused segment.
show("lm_head / CE tail", lambda ns: any(n in ("wte","logits","max_logits") for n in ns))
show("QK^T sites", lambda ns: any(n.endswith("_q") for n in ns) and any(n.endswith("_k") for n in ns))
# Part 1's five-kernel table and Part 5's attention line items.
g1=lambda ns: any(re.fullmatch(r'l\d+_ffn_w1', n) for n in ns)
show("FFN GEMM1 + gelu", g1)
show("q/k/v projections", lambda ns: any(re.fullmatch(r'w_[qkv]_l\d+', n) for n in ns))
show("out projections", lambda ns: any(re.fullmatch(r'w_o_l\d+', n) for n in ns))
# WHOLE-CHAIN totals. A chain must be summed over every fragment, not over the one that keeps its
# name: post-fission the QK^T site's mask, row-max and softmax work runs in separate downstream
# kernels, and comparing a standalone QK^T against a fused QK^T+mask+row-max is meaningless.
qk_chain=lambda ns: ((any(n.endswith("_q") for n in ns) and any(n.endswith("_k") for n in ns))
                     or "mask" in ns or any(n.endswith("_max_vals") for n in ns))
# NOT a bare `wte`: the input token-embedding gather reads wte too (bench_gpt builds the embedding as
# wte * onehot_x), and it belongs to the start of the model rather than the CE head. The lm_head
# kernel carries `logits` as well, so anchoring there selects the head without the gather.
ce_chain=lambda ns: any(n in ("logits","max_logits","log_probs","neg_nll","n810_log") for n in ns)
show("QK^T WHOLE CHAIN (qk + mask + row-max + softmax)", qk_chain)
show("lm_head / CE WHOLE CHAIN", ce_chain)
# Bucket totals on the SAME basis as everything above -- per-kernel medians grouped by
# gpt2_bucket.py's own assignment. The per-run bucket-N.txt tables median each RUN's bucket total
# instead, and median-of-sums != sum-of-medians: on base574 the two bases differ by 0.048 ms. Any
# reconciliation against the per-kernel total has to use this one or it will not close.
import subprocess
r=subprocess.run(["python3",os.path.join(sys.argv[3],"gpt2_bucket.py"),"--source",sys.argv[1],
                  "--stats",os.path.join(out,"kernels-1.csv"),"--steps","1","--dump"],
                 capture_output=True,text=True)
if r.returncode!=0: sys.exit(f"gpt2_bucket.py failed: {r.stderr.strip().splitlines()[-1:]}")
dump=r.stdout
assign={}
for ln in dump.splitlines():
    # the dump prints "| <routine>__segN | <bucket> | ..." -- no word boundary before "seg",
    # since "_" is a word character, so anchor on the "__seg" prefix instead
    mm=re.search(r"__seg(\d+)\s*\|\s*(ffn|attention|emb_logits|layernorm|other)\b", ln)
    if mm: assign[int(mm.group(1))]=mm.group(2)
if not assign: sys.exit("could not parse gpt2_bucket.py --dump; bucket basis unavailable")
if assign:
    agg=collections.defaultdict(float)
    for i,t in ms.items(): agg[assign.get(i,"UNASSIGNED")]+=t
    print("bucket totals (per-kernel-median basis; sums exactly to the total above):")
    for b in sorted(agg): print(f"  {b:12} {agg[b]:7.3f} ms")
    print(f"  {'TOTAL':12} {sum(agg.values()):7.3f} ms")
lm=lambda ns: "wte" in ns and any(n.endswith("_layer_norm") for n in ns)
five=[i for i in sigs if g1(sigs[i]) or lm(sigs[i])]
tot=sum(ms.values())
print(f"the five kernels (4x FFN GEMM1 + lm_head): {sum(ms.get(i,0) for i in five):.3f} ms "
      f"= {100*sum(ms.get(i,0) for i in five)/tot:.1f}% of {tot:.3f} ms")
# Part 1's launch-geometry census, aggregated by RESIDENT BLOCK COUNT -- the quantity that made the
# gh-569 story legible and the one Part 5 turns on.
geo={}
for m in re.finditer(r'seg (\d+)/(\d+) grid=\[(\d+);(\d+);(\d+)\] block=\[(\d+);(\d+);(\d+)\]',
                     open(os.path.join(out,"launches.err")).read()):
    if int(m.group(2))!=len(sigs): continue
    g=[int(m.group(i)) for i in (3,4,5)]; b=[int(m.group(i)) for i in (6,7,8)]
    geo[int(m.group(1))]=(g[0]*g[1]*g[2], b[0]*b[1]*b[2])
agg=collections.defaultdict(lambda:[0,0.0])
for i in sigs:
    if i in geo: agg[geo[i][0]][0]+=1; agg[geo[i][0]][1]+=ms.get(i,0.0)
print("blocks | kernels | ms | share")
for blk,(n,t) in sorted(agg.items()):
    print(f"  {blk:5} | {n:3} | {t:6.2f} | {100*t/tot:4.1f}%")
EOF
  echo "--- launch geometry (arm A) ---"
  local n; n=$(grep -o "$ROUTINE seg 0/[0-9]*" "$out/launches.err" | awk -F/ '$2>1{print $2; exit}')
  grep -E "$ROUTINE seg [0-9]+/$n " "$out/launches.err" \
    | sed -E 's/.*(seg [0-9]+).*grid=(\[[^]]*\]).*block=(\[[^]]*\]).*/\1 grid=\2 block=\3/'
}

# Cap sweep with BALANCED order: within a block the list runs FORWARD on the first rep and reversed
# on the next, alternating -- keyed on the offset from FIRST_REP, not on the absolute rep number, so
# a block starting at r4 still begins with the forward order. Getting that wrong silently produces
# the mirror image of a claimed sequence. Without the alternation, one cap is permanently earlier in
# the session than the other and thermal or driver drift is indistinguishable from the cap's effect
# -- the confound report-gh481-cuda.md's arm-order rule exists for, worth ~1.4pp here.
# <reps> is a COUNT. FIRST_REP (env, default 1) offsets the numbering so a second block can EXTEND an
# earlier one instead of overwriting it -- the report's cap-4 n=6 is two blocks of three, r1-r3 then
# FIRST_REP=4 r4-r6. It is an environment variable rather than a positional argument because a rep
# number and a cap value are indistinguishable in that position.
cmd_sweep() {
  local tree=$1 reps=$2 first=${FIRST_REP:-1}; shift 2
  local caps=("$@") r cap ordered i        # DECLARE before validating: under set -u a reference to an
                                          # undeclared array aborts, which broke every valid sweep.
  tree=$(require_dir "$tree")        # BEFORE the loop: each cell must see the same absolute tree
  case ${tree:-} in /?*) ;; *) return 2;; esac
  # An unrun sweep must not look like a successful one: `seq three` and `seq 0` both yield no
  # iterations, and the concluding notes would otherwise print over an empty experiment.
  case $reps in ""|*[!0-9]*) echo "gh612_cells: reps must be a positive integer, got '$reps'" >&2; return 2;; esac
  case $first in ""|*[!0-9]*) echo "gh612_cells: first-rep must be a positive integer, got '$first'" >&2; return 2;; esac
  [ "$reps" -ge 1 ] || { echo "gh612_cells: reps must be >= 1, got '$reps'" >&2; return 2; }
  [ "$first" -ge 1 ] || { echo "gh612_cells: first-rep must be >= 1, got '$first'" >&2; return 2; }
  [ ${#caps[@]} -ge 1 ] || { echo "gh612_cells: no caps given" >&2; return 2; }
  for r in $(seq "$first" $((first + reps - 1))); do
    if [ $(( (r - first) % 2 )) -eq 0 ]; then ordered=("${caps[@]}")
    else ordered=(); for ((i=${#caps[@]}-1; i>=0; i--)); do ordered+=("${caps[$i]}"); done; fi
    echo "=== sweep rep $r, order: ${ordered[*]} ==="
    for cap in "${ordered[@]}"; do
      cmd_search "$tree" "sweep-cap$cap" "$r" --ocannl_virtualize_max_inline_fanin="$cap" \
        || { echo "gh612_cells: cap $cap rep $r failed; aborting the sweep rather than reporting a" \
                  "half-balanced series" >&2; return 1; }
    done
  done
  echo "NOTE: an equal kernel count does NOT mean the guard was silent -- on gpt2_mini cap 16 and"
  echo "      cap -1 both emit 135 kernels while their placement differs (n792 is newly materialized)."
  echo "      Silence requires \`diff\` to report ZERO exclusive signatures on BOTH sides AND zero"
  echo "      differing multiplicities. Read the untuned column for the timing trade separately."
}

# Cross-cell signature-set diff: groups kernels by their SIGNATURE (the sorted parameter-name tuple,
# which is the identity that survives renumbering between two different fissions) and reports what
# exists only in one cell. This is what pins a mechanism to named kernels rather than to a bucket
# total -- Part 2's 14-vs-32 signatures and Part 3's 16-vs-17 are this subcommand's output.
cmd_diff() {
  local a; a=$(cell_dir "$1" "$2"); case ${a:-} in /?*) ;; *) return 2;; esac
  local b; b=$(cell_dir "$3" "$4"); case ${b:-} in /?*) ;; *) return 2;; esac
  python3 - "$1/r$2" "$a" "$3/r$4" "$b" <<'EOF' || return 1
import re,sys,csv,glob,os,statistics,collections
LOADED={}
import hashlib
SIG=re.compile(r'extern\s+"C"\s+__global__\s+void\s+(\w+__seg(\d+))\s*\(([^)]*)\)\s*\{', re.S)
def load(name, out):
    try: src=open(open(os.path.join(out,"armA.path")).read().strip()).read()
    except OSError as e: sys.exit(f"{name}: {e} -- run `snap` first")
    sigs={}; bodies={}; params=set()
    for m in SIG.finditer(src):
        i=m.end(); d=1; j=i
        while d:
            if src[j]=="{": d+=1
            elif src[j]=="}": d-=1
            j+=1
        # Canonicalized body: whitespace collapsed, then each generated loop symbol ALPHA-RENAMED to
        # a stable token in first-appearance order. Collapsing them all to one token instead would
        # erase the relationships between indices -- a[i1][i2] and a[i2][i1] would hash alike -- so
        # the renaming has to be injective. Compile-order numbering is still normalized away.
        raw=re.sub(r"\s+"," ",src[i:j-1]).strip()
        seen={}
        def _rn(mo):
            k=mo.group(0)
            if k not in seen: seen[k]=f"v{len(seen)}"
            return seen[k]
        can=re.sub(r"\bi\d+\b",_rn,raw)
        sig=tuple(sorted(" ".join(q.split()).split()[-1].lstrip("*")
                         for q in m.group(3).split(",") if q.strip()))
        sigs[int(m.group(2))]=sig
        bodies[int(m.group(2))]=hashlib.md5(can.encode()).hexdigest()[:10]
        for q in m.group(3).split(","):
            q=" ".join(q.split())
            if q and "*" in q: params.add(q.split()[-1].lstrip("*"))
    LOADED[name]=(sigs,bodies,params)
    t=collections.defaultdict(list)
    # Timings are OPTIONAL here: "did the guard fire" is a question about the emitted kernel set, so
    # a structural diff must work off `snap` alone, without `profile`.
    csvs=sorted(glob.glob(os.path.join(out,"kernels-*.csv")))
    for f in csvs:
        for r in csv.DictReader(open(f)):
            t[int(r["Name"].rsplit("__seg",1)[1])].append(float(r["TotalDurationNs"])/1e6)
    ms={i:statistics.median(v) for i,v in t.items()}
    # A CSV SET IS NOT A PROFILE unless every emitted segment is timed in every run: an interrupted
    # profile can leave one complete CSV beside a partial one, and `ms.get(i, 0.0)` below would then
    # fill the gaps with zeros while the file count still reads as N successful runs. Treat an
    # incomplete set as unprofiled rather than silently comparing inconsistent sample counts.
    complete = bool(csvs) and all(len(t.get(i,[])) == len(csvs) for i in sigs)
    if csvs and not complete:
        print(f"  {name}: WARNING incomplete profile ({len(csvs)} CSVs but not every segment timed in"
              " each) -- treating as unprofiled", file=sys.stderr)
    per=collections.defaultdict(float); n=collections.defaultdict(int)
    for i,sg in sigs.items(): per[sg]+=ms.get(i,0.0); n[sg]+=1
    return name, sigs, per, n, (len(csvs) if complete else 0)
(na,sa,pa,ca,ra)=load(sys.argv[1],sys.argv[2]); (nb,sb,pb,cb,rb)=load(sys.argv[3],sys.argv[4])
BODY_A=LOADED[na]; BODY_B=LOADED[nb]
def hdr(n,per,sg,r):
    t=f"{sum(per.values()):.3f} ms" if r else "(no timings: run `profile` for ms)"
    print(f"{n}: {t} / {len(sg)} kernels ({r} harness runs)")
hdr(na,pa,sa,ra); hdr(nb,pb,sb,rb)
onlya=[k for k in pa if k not in pb]; onlyb=[k for k in pb if k not in pa]
sh_a=sum(pa[k] for k in pa if k in pb); sh_b=sum(pb[k] for k in pb if k in pa)
# Timing fields ONLY when BOTH sides are profiled. An unprofiled kernel contributes 0 ms, so a
# profiled-vs-unprofiled comparison would print real times against zeros and read as a huge
# regression -- plausible, meaningless, and contradicting the "ms omitted without profile" note.
TIMED = ra > 0 and rb > 0
def msf(v): return f", {v:.3f} ms" if TIMED else ""
print(f"\nsignatures only in {na}: {sum(ca[k] for k in onlya)} kernels{msf(sum(pa[k] for k in onlya))}")
for k in sorted(onlya, key=lambda k:-pa[k])[:8]:
    print(f"  {pa[k]:7.3f} ms  {', '.join(k)[:120]}" if TIMED else f"  {', '.join(k)[:128]}")
print(f"signatures only in {nb}: {sum(cb[k] for k in onlyb)} kernels{msf(sum(pb[k] for k in onlyb))}")
for k in sorted(onlyb, key=lambda k:-pb[k])[:8]:
    print(f"  {pb[k]:7.3f} ms  {', '.join(k)[:120]}" if TIMED else f"  {', '.join(k)[:128]}")
if not TIMED:
    print(f"  (timings omitted: {na} has {ra} profile run(s), {nb} has {rb} -- run `profile` on both"
          "\n   for ms columns; the structural verdict below does not need them)")
# MULTISET, not set: a signature present in both cells but a different NUMBER of times is not
# "shared". Without this, `only in` counts of 0/0 could still describe differing kernel multisets and
# the negative-control reading would be wrong while looking right.
remult=[(k,ca[k],cb[k]) for k in sorted(set(ca)&set(cb)) if ca[k]!=cb[k]]
if remult:
    print(f"\nshared signatures with DIFFERING multiplicity: {len(remult)} "
          f"({sum(x for _,x,_ in remult)} occurrences in {na} vs {sum(y for _,_,y in remult)} in {nb})")
    for k,x,y in sorted(remult, key=lambda t:-abs(t[1]-t[2]))[:8]:
        print(f"  x{x} -> x{y}  {', '.join(k)[:110]}")
elif onlya or onlyb:
    # Do NOT announce multiset agreement here: exclusive signatures on either side already mean the
    # kernel sets differ, and saying "the multisets agree" would contradict the counts just printed.
    print("\nshared signatures with differing multiplicity: 0 (the DIFFERENCE is entirely the "
          "exclusive signatures above)")
else:
    print("\nIDENTICAL kernel sets: no exclusive signatures on either side and no differing "
          "multiplicity\n-- the two cells emit the same kernel multiset. This is the negative-control "
          "reading.")
if TIMED:
    print(f"\nshared signatures: {sh_a:.3f} -> {sh_b:.3f} ms ({sh_b-sh_a:+.3f}); "
          f"NET {sum(pb.values())-sum(pa.values()):+.3f} ms")
# What a signature multiset does and does not settle. A kernel's POINTER PARAMETERS are exactly the
# materialized nodes it touches, so signature-multiset equality is the right invariant for the
# PLACEMENT question and is deliberately insensitive to the crowned tile. Bodies are not: they move
# with the tile too, so a body diff cannot be read as a placement change. Both are printed, and the
# newly-materialized NODE set is printed as well, because that -- not the signature count -- is the
# quantity that corresponds to guard firings (one materialized node can change several consumers'
# signatures, and several can be absorbed into one).
_,ba,pna=BODY_A; _,bb,pnb=BODY_B
mba=collections.Counter(ba.values()); mbb=collections.Counter(bb.values())
print(f"\ncanonicalized kernel BODIES differing: {sum((mba-mbb).values())} in {na}, "
      f"{sum((mbb-mba).values())} in {nb}")
print("  (bodies move with the crowned tile as well as with placement, so a body diff is NOT by "
      "itself\n   evidence that a placement decision changed -- check the crowned sketch labels too)")
newb=sorted(pnb-pna); newa=sorted(pna-pnb)
print(f"\nmaterialized NODES: {len(pna)} in {na}, {len(pnb)} in {nb}")
print(f"  only in {nb} (+{len(newb)}): {', '.join(newb) if newb else '-'}")
print(f"  only in {na} (+{len(newa)}): {', '.join(newa) if newa else '-'}")
print("  This node count is the closest available proxy for guard FIRINGS; exact counts would need a"
      "\n  provenance-41 placement log, which does not exist today.")
print("\nThe negative-control reading requires BOTH `only in` signature sides at 0 AND zero differing"
      "\nmultiplicities. It is a claim about PARAMETER-SIGNATURE multisets, i.e. about placement -- not"
      "\nabout byte-identical kernels.")
EOF
}

# PASS 2 of the protocol benchmarks/README.md requires for tuned cells: a FRESH process replays the
# cached winner and provides the step timings. Pass 1 (`search`) leaves its own process measurably
# slower -- accumulated modules and buffers add per-launch overhead -- so its step p50 penalizes the
# tuned artifact for the search it already paid for in compile_s. `search`'s JSON is the search cost;
# the step numbers a report quotes must come from here.
cmd_replay() { (
  local tree=$1 label=$2 rep=$3; shift 3
  # Same rule as `snap`: resolve the cell and retract before ANY other validation, tree check
  # included. `profile` accepts any nonempty replay2.out as an accepted timing and the `replays`
  # gate counts it as coverage, so a stale one survives a rejected replay in both.
  local out; out=$(cell_dir "$label" "$rep")
  case ${out:-} in /?*) ;; *) exit 2;; esac
  rm -f "$out/replay2.out" "$out/replay.manifest"
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  check_config "$tree" || exit 2
  check_fixture "$tree" || exit 2
  [ -d "$out/cache" ] && [ -n "$(ls -A "$out/cache" 2>/dev/null)" ] || {
    echo "gh612_cells: $label r$rep has no populated cache -- run \`search\` first" >&2; exit 2; }
  cd "$tree/benchmarks" || exit 1
  # A populated cache DIRECTORY is not a cache hit: the key depends on the tree and the config, so a
  # miss here (wrong cap flag, a search that cached only one arm) would leave BENCH_TUNE=1 free to run
  # a fresh search -- which still emits step_ms and would be accepted as a "replay" while carrying
  # exactly the search-process residue this pass exists to exclude. Disable the search so a miss
  # cannot become one, and log the autotune decisions so the hit can be VERIFIED below.
  BENCH_FIXTURE=$FIXTURE BENCH_TUNE=1 $PIN "$EXE" --ocannl_backend=hip \
    --ocannl_autotune_cache_dir="$out/cache" --ocannl_autotune_search=false \
    --ocannl_autotune_log=true "$@" \
    > "$out/replay2.tmp" 2> "$out/replay2.err"
  local st=$?
  echo -n "$label r$rep pass-2 replay: exit $st  "
  local timing; timing=$(grep -h '^{' "$out/replay2.tmp" 2>/dev/null | tail -1 | grep -o '"step_ms":{[^}]*}')
  echo "${timing:-<no step_ms record>}"
  # Producing the timing IS this subcommand's purpose: a zero exit with no record would let the
  # pass-2 loop look successful while yielding nothing for the cell.
  [ "$st" -eq 0 ] || return "$st"
  [ -n "${timing:-}" ] || { echo "gh612_cells: $label r$rep produced no step_ms record" >&2; return 1; }
  # With the search disabled a miss does not fail -- it silently ships the UNTUNED default compile
  # (gh-ocannl-559's no_search_report), whose step time would be ~3x and would look like a result.
  # So require a cache hit for BOTH arms before accepting the timing.
  local hits; hits=$(grep -c 'cache hit:' "$out/replay2.err" 2>/dev/null || echo 0)
  [ "${hits:-0}" -ge 2 ] || {
    echo "gh612_cells: $label r$rep is NOT a replay -- $hits cache hits (want 2, one per arm)." >&2
    echo "  With autotune_search=false a miss ships the untuned default, so this timing is not a" >&2
    echo "  pass-2 replay of the crowned artifact. Re-run \`search\` for this cell." >&2; return 1; }
  mv "$out/replay2.tmp" "$out/replay2.out"
  return 0
) }

# The correctness gate, computed rather than eyeballed. This exists because the claim it replaces
# ("bit-identical losses") was produced by an ad-hoc script that rounded to 4 decimals before
# comparing: at full serialized precision the runs are NOT identical, they agree to a few f32 ulp --
# which is the right result for reassociation across different kernel schedules, and a different
# claim. Compares every cell's loss vector at the precision bench_gpt actually serializes.
# PARITY_MAX_ULP (default 64) is the failure threshold: comfortably above the few-ulp reassociation
# noise that different schedules produce, far below anything a real correctness regression shows.
# EXPECT_RUNS, if set, additionally pins how many cells the gate must have covered.
cmd_parity() { (
  python3 - "$OUT_ROOT" "${PARITY_MAX_ULP:-64}" "${EXPECT_RUNS:-0}" "${EXPECT_CELLS:-}" \
           "${EXPECT_STEPS:-8}" <<'EOF'
import json,glob,os,sys,math,collections
root=sys.argv[1]
seqs=collections.defaultdict(list)
# BOTH passes. Every quoted tuned p50 comes from a pass-2 process, and `replay` validates only its
# exit status, a timing fragment and the cache-hit count -- so a replay that reconstructed the
# computation incorrectly while the cold search was fine would ship an unverified timing, and
# scanning search.out alone would never see it.
for kind in ("search.out","replay2.out"):
  for f in sorted(glob.glob(os.path.join(root,"*","r*",kind))):
    # belt and braces alongside the staging-dir naming: never count a staging artifact as a cell
    if os.sep + ".staging" in f: continue
    t=[l for l in open(f).read().splitlines() if l.startswith("{")]
    if not t: continue
    d=json.loads(t[-1])
    if "losses" not in d: continue
    tag="/".join(f.split(os.sep)[-3:-1]) + ("" if kind=="search.out" else " [pass2]")
    seqs[tuple(d["losses"])].append(tag)
tot=sum(len(v) for v in seqs.values())
if not tot: sys.exit(f"no search.out with losses under {root}")
p2=sum(1 for ks in seqs.values() for c in ks if c.endswith("[pass2]"))
print(f"{tot} records ({tot-p2} search + {p2} pass-2 replay), {len(seqs)} distinct loss sequences")
for L,ks in sorted(seqs.items(), key=lambda kv:-len(kv[1])):
    print(f"  n={len(ks):2}  {L[0]!r} ...  e.g. {ks[0]}")
def ulp32(x):
    return 2.0**((math.frexp(x)[1]-1)-23)
# NaN/inf must be rejected BEFORE any span arithmetic: `max(0, nan)` is 0 in Python, so a run whose
# losses are all NaN would leave `worst` at zero and print PASSED -- a correctness gate certifying
# the precise outcome it exists to catch. A diverged loss now arrives as JSON `null` rather than as
# a line that failed to parse at all (gh-ocannl-676), so the test is None-aware and the exit is
# immediate: the span arithmetic below would raise a TypeError on it, which is a traceback where a
# finding belongs.
def isfin(v):
    return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
nonfinite=sorted({c for L,ks in seqs.items() for c in ks if not all(isfin(v) for v in L)})
if nonfinite:
    sys.exit("FAILED: non-finite losses (null/NaN/inf) in: " + " ".join(nonfinite))
worst=0.0
print("per-step agreement across ALL runs:")
for i in range(len(next(iter(seqs)))):
    vals=sorted({L[i] for L in seqs})
    span=max(vals)-min(vals); u=span/ulp32(vals[0]); worst=max(worst,u)
    print(f"  step {i}: {len(vals)} distinct, span {span:.3e} = {u:5.1f} f32 ulp (rel {span/vals[0]:.2e})")
maxulp=float(sys.argv[2]); expect=int(sys.argv[3])
print(f"WORST: {worst:.0f} f32 ulp (threshold {maxulp:.0f}). NOT bit-identity -- state it as"
      "\n       agreement to within that many ulp.")
bad=[]
if worst > maxulp: bad.append(f"loss divergence {worst:.0f} ulp exceeds PARITY_MAX_ULP={maxulp:.0f}")
# EXPECT_RUNS counts SEARCH cells; pass-2 records are extra coverage, not extra cells.
searched=sum(1 for ks in seqs.values() for c in ks if not c.endswith("[pass2]"))
if expect and searched != expect: bad.append(f"covered {searched} search runs, expected EXPECT_RUNS={expect}")
# A count is not a set: with a reusable OUT_ROOT a stale or unrelated cell can stand in for a missing
# required one and still total EXPECT_RUNS. EXPECT_CELLS pins the exact label/rep set.
want=set(filter(None, (sys.argv[4] if len(sys.argv)>4 else "").split()))
if want:
    have={c for ks in seqs.values() for c in ks if not c.endswith("[pass2]")}
    missing=sorted(want-have); extra=sorted(have-want)
    if missing: bad.append(f"missing required cells: {' '.join(missing)}")
    if extra: bad.append(f"unexpected cells present (stale OUT_ROOT?): {' '.join(extra)}")
    # The PASS-2 side needs the same treatment, and skipping it silently is the failure mode:
    # `replays` accepts a replay2.out on step_ms alone, and a record without `losses` is simply not
    # seen by the loop above -- so the gate could pass with a timed artifact never output-verified.
    havep2={c[:-len(" [pass2]")] for ks in seqs.values() for c in ks if c.endswith("[pass2]")}
    m2=sorted(want-havep2); x2=sorted(havep2-want)
    if m2: bad.append(f"pass-2 replays missing loss vectors (timed but unverified): {' '.join(m2)}")
    if x2: bad.append(f"unexpected pass-2 replays (stale OUT_ROOT?): {' '.join(x2)}")
lens={len(L) for L in seqs}
if len(lens) != 1: bad.append(f"loss vectors have differing lengths {sorted(lens)}")
# Consistent lengths are not enough: 26 records of `"losses":[]` share a length, skip the comparison
# loop entirely, leave worst at 0 and would print PASSED having checked no executed value at all.
# Require the expected sample count so a truncated or empty vector fails instead of certifying.
steps=int(sys.argv[5]) if len(sys.argv)>5 and sys.argv[5] else 8
got=lens.pop() if len(lens)==1 else None
if got is not None and steps and got != steps:
    bad.append(f"loss vectors have {got} samples, expected EXPECT_STEPS={steps}")
if got == 0: bad.append("loss vectors are EMPTY: the gate would compare nothing")
# `tot` counts records, and a search plus ITS OWN pass-2 replay is two records of one cached winner --
# a codegen error shared by both is then compared only with itself. Require two independent searches.
if searched < 2: bad.append(f"only {searched} independent search cell(s): nothing to compare against")
if bad:
    for b in bad: print("PARITY GATE FAILED: "+b, file=sys.stderr)
    sys.exit(1)
print("parity gate PASSED")
EOF
) }

# Every quoted tuned p50 comes from an accepted replay2.out, but the reproduction runs its replays as
# separate commands whose individual failures do not stop the block -- so a missing pass-2 timing
# would otherwise go unnoticed while the workflow reported success. This asserts the exact set.
cmd_replays() { (
  local missing=0 c
  for c in "$@"; do
    local f="$OUT_ROOT/$c/replay2.out"
    if [ ! -s "$f" ]; then echo "MISSING accepted pass-2 replay: $c" >&2; missing=$((missing+1))
    elif ! grep -q '"step_ms"' "$f" 2>/dev/null; then
      echo "replay2.out without step_ms: $c" >&2; missing=$((missing+1))
    fi
  done
  [ "$missing" -eq 0 ] || { echo "gh612_cells: $missing cell(s) lack an accepted pass-2 replay" >&2; exit 1; }
  echo "all $# cells have an accepted pass-2 replay"
) }

# Mirrors `replays` for the profile artifacts: the reproduction's profile block is separate commands
# whose individual failures do not stop it, and `diff` deliberately succeeds without CSVs by omitting
# milliseconds -- so a cell could silently never be profiled while the block reported success.
# PROFILE_RUNS (default 3) is the count each cell must have: accepting "at least one" would let a
# profile interrupted after its first iteration satisfy the gate, and finger/diff would then compute
# the report's stated three-run medians from a single run.
cmd_profiles() { (
  local want=${PROFILE_RUNS:-3} missing=0 c
  for c in "$@"; do
    local d="$OUT_ROOT/$c"
    if [ ! -s "$d/armA.path" ]; then echo "MISSING armA.path: $c" >&2; missing=$((missing+1)); continue; fi
    local k; k=$(ls "$d"/kernels-*.csv 2>/dev/null | wc -l)
    [ "$k" -eq "$want" ] || { echo "WRONG run count: $c has $k kernels-*.csv, want $want" >&2; missing=$((missing+1)); continue; }
    local b; b=$(ls "$d"/bucket-*.txt 2>/dev/null | wc -l)
    [ "$b" -eq "$k" ] || { echo "profile artifact mismatch: $c has $k CSVs but $b bucket files" >&2; missing=$((missing+1)); }
  done
  [ "$missing" -eq 0 ] || { echo "gh612_cells: $missing cell(s) lack a complete profile" >&2; exit 1; }
  echo "all $# cells have a complete profile"
) }

cmd_roofline() { (
  local tree=$1
  tree=$(require_dir "$tree")
  case ${tree:-} in /?*) ;; *) exit 2;; esac
  cd "$tree" || exit 1                      # roofline_hip.cpp path is relative to the TREE ROOT
  hipcc --offload-arch="$ARCH" -O3 -o "$OUT_ROOT/roofline" benchmarks/roofline_hip.cpp \
        -I/opt/rocm/include -L/opt/rocm/lib -lrocblas || exit 1
  # CPU quiet: the bandwidth leg shares the LPDDR5X controller with it on this APU.
  $PIN "$OUT_ROOT/roofline"
) }

sub=${1:-}; shift || true
case $sub in
  search)   cmd_search   "$@" ;;
  snap)     cmd_snap     "$@" ;;
  profile)  cmd_profile  "$@" ;;
  finger)   cmd_finger   "$@" ;;
  parity)   cmd_parity   "$@" ;;
  replays)  cmd_replays  "$@" ;;
  profiles) cmd_profiles "$@" ;;
  replay)   cmd_replay   "$@" ;;
  diff)     cmd_diff     "$@" ;;
  sweep)    cmd_sweep    "$@" ;;
  roofline) cmd_roofline "$@" ;;
  *) sed -n '2,20p' "$0"; exit 1 ;;
esac

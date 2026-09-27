#!/bin/bash
# gh-ocannl-833 / gh-ocannl-834 measurement driver: the same bounded steps on every box, so the
# issue comments quote invocations rather than restating commands.
#
# Usage (from anywhere; the checkout is the one this script lives in):
#   benchmarks/gh834_cells.sh BACKEND OUT FIXTURE CAP STEP...
#
#   BACKEND  cc | hip | cuda | metal, pinned on every command line (out-ranks every other source).
#   OUT      results directory, created, and refused unless empty. OUT/driver.log records
#            everything this script prints (provenance, the device state around each step, each
#            step's exit and wall); each measurement step also writes OUT/<step>.out and .err.
#   FIXTURE  absolute path of a gpt2_mini.safetensors (only the session steps read it).
#   CAP      wall cap in seconds for the build and for each measurement step; a capped step's
#            whole process group is terminated.
#   STEP     build | provenance | crown-fwd | crown-rev | session-isolated | session-queued
#            A measurement step needs `build` and `provenance` earlier in the same invocation:
#            _build/ is ignored, so the clean-tree check cannot vouch for binaries an earlier checkout
#            left there, and a measurement is only evidence beside the identity it was taken on.
# Exit: 0 all steps complete; 1 a step failed or lacked its evidence; 124 no failure but a SESSION
# hit CAP during its search (its record is a lower bound); 125 a process outlived its step;
# 130 interrupted; 2 usage. A crown or build that hits CAP, or a session capped before its first
# candidate attempt or after the runner's "search done" marker, is a failure: none of them leaves a
# partial search measurement.
# The environment is cleared of OCANNL_*, BENCH_* and the OpenMP controls (OMP_*, GOMP_*, KMP_*);
# device-selection variables (CUDA_*, HIP_*, ROCR_*, HSA_*, ...) are kept and recorded.
#
# crown-{fwd,rev}: gh-ocannl-833's instrument, bin/projection_shape_bench.exe 200 8 d <order>
#   seeds -- the out-projection sites (group d), every seeded candidate ranked by the batched
#   median, by Autotune.time_routine ~timing:Isolated and by ~timing:Queued; the closing
#   "gh-ocannl-755" table says whether the crown moves.
# session-<mode>: gh-ocannl-834's instrument, one gpt2_mini tuned search (BENCH_TUNE=1, both
#   placement arms) under autotune_timing=<mode> on a COLD schedule cache (OUT/cache-<mode>, wiped
#   first), with BENCH_TIMING_TRACE=1 splitting the wall between candidate timing and the rest.
#   compile_s in the result line is the whole search's wall.
#   Cold also means the backend's own compiled-code caches, which outlive the process: in
#   gh-ocannl-834's CUDA pair the isolated session ran first and the driver's PTX ComputeCache
#   served the queued one warm (406 s vs 64 s of compile and bookkeeping outside timing). On cuda
#   each session therefore gets its own empty CUDA_CACHE_PATH=OUT/nvcache-<mode>, recorded in
#   driver.log (with its final size) and removed after the step -- on every exit path -- so OUT
#   archives only results; it overrides any the caller exported. Other backends keep caches this
#   script cannot redirect (HIP's comgr cache, macOS's Metal shader cache), and a fresh OUT does not
#   reset them: the first session ever run warms every later one, the next invocation's included.
#   There, bring every measured session to the same cache state -- a discarded warm-up session first
#   (or the cache cleared by hand before each session) -- then run the modes ABBA (session-isolated
#   session-queued in one invocation, the reverse in a second on a fresh OUT) and compare each
#   mode's pair, so order effects cancel.
set -u
# Hermetic against ambient configuration, as gh612_cells.sh is: every treatment is pinned on the
# command line, so an exported OCANNL_* or BENCH_* could only contaminate every cell consistently.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(OCANNL_[A-Z0-9_]*\)=.*/\1/p')
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(BENCH_[A-Z0-9_]*\)=.*/\1/p')
# The CPU backend's OpenMP controls are treatment too (OMP_NUM_THREADS=1 turns the cc machine
# serial, and cc_backend keys the schedule cache on them), so the cells run the backend's defaults.
while read -r v; do unset "$v"; done < <(env | sed -n 's/^\(\(OMP\|GOMP\|KMP\)_[A-Z0-9_]*\)=.*/\1/p')

[ $# -ge 5 ] || { sed -n '/^# Usage/,/^# crown-/p' "$0" | sed '$d' >&2; exit 2; }
backend=$1 out=$2 fixture=$3 cap=$4
shift 4
case $backend in cc | hip | cuda | metal) ;; *) echo "gh834: unknown backend $backend" >&2; exit 2 ;; esac
case $cap in '' | *[!0-9]* | 0*) echo "gh834: CAP must be a positive decimal integer, got '$cap'" >&2; exit 2 ;; esac
# A fresh OUT per invocation: driver.log appends and old step files would otherwise sit beside new
# ones, so provenance from an earlier checkout could be read as this run's.
if [ -e "$out" ] && [ -n "$(ls -A "$out" 2>/dev/null)" ]; then
  echo "gh834: OUT $out is not empty; give each invocation a fresh directory" >&2
  exit 2
fi
mkdir -p "$out" || exit 2
out=$(cd "$out" && pwd -P) && case $out in /?*) ;; *) false ;; esac ||
  { echo "gh834: OUT did not resolve to an absolute directory" >&2; exit 2; }
# Everything printed from here on is written to OUT/driver.log synchronously -- an asynchronous tee
# could still be writing when the caller archives OUT, and its failures would go unobserved -- and
# the whole log is replayed to the caller's stdout on exit. Follow a live run with tail -f.
exec 3>&1
exec >>"$out/driver.log" 2>&1 || exit 2
# A session's CUDA driver cache (below) is scratch: whatever path ends the run -- an interrupt, a
# survivor of the cap, a usage error mid-list -- leaves none of it in OUT.
trap 'rm -rf "$out"/nvcache-*; cat "$out/driver.log" >&3' EXIT
root=$(cd "$(dirname "$0")/.." && pwd -P)
# The runners read the nearest ocannl_config; benchmarks/ has the suite's own.
cd "$root/benchmarks" || exit 2
# Every artifact is labelled by the commit, so the tracked tree must BE that commit: a locally edited
# benchmarks/ocannl_config (autotune_repeats, a profile, a placement key) would change every cell
# while the label stayed the same. The effective config file is recorded as well as checked.
dirty=$(git -C "$root" status --porcelain --untracked-files=no)
if [ -n "$dirty" ]; then
  echo "gh834: refusing to measure from a tree with tracked changes:" >&2
  echo "$dirty" >&2
  exit 2
fi
# Device selection is left to the caller, and recorded, so the archive says which device was visible.
env | grep -E '^(CUDA|HIP|ROCR|HSA|GPU_DEVICE|MTL|METAL)[A-Z0-9_]*=' | sed 's/^/env /'
echo "config benchmarks/ocannl_config sha256 $(shasum -a 256 ocannl_config | cut -d' ' -f1):" \
  "$(grep -v '^#' ocannl_config | grep -v '^$' | tr '\n' ' ')"

# A portable wall cap over the step's whole PROCESS GROUP (macOS has no timeout(1) or setsid(1)):
# a cc candidate's compiler is a child of the runner, and a compile outliving its capped runner
# would load every later cell. The runner leads its own group; at the cap the group gets SIGTERM,
# then SIGKILL after 10 s, and the step waits until no member is left before the next one starts
# (exit 124; 125 if a member survives even SIGKILL, and the driver stops rather than measure beside
# it). A member left behind by a runner that exited on its own is reaped the same way. An INT,
# TERM or HUP reaching the supervisor (the driver below forwards its own) takes the group down the
# same way, and the supervisor then exits 128+signal. SIGTERM ends an OCaml runner without at_exit
# -- which is why the trace carries running totals.
# Always started as a job (`capped ... &`): the exec makes the job's pid the supervisor itself, so
# a signal the driver forwards to that pid reaches the process that owns the group -- a function
# run as a job is otherwise a subshell, whose death would orphan the supervisor and its group.
capped() {
  exec perl -e '
    use POSIX ();
    my $cap = shift;
    my $pid = fork // die "gh834: fork: $!\n";
    if ($pid == 0) { setpgrp(0, 0); exec @ARGV or POSIX::_exit(127) }
    setpgrp($pid, $pid);
    my $timed_out = 0;
    # TERM at the cap or on an external signal, KILL 10 s later if the leader is still not reaped.
    my ($stage, $external) = (0, 0);
    my $escalate = sub {
      if ($stage++) { kill "KILL", -$pid } else { kill "TERM", -$pid; alarm 10 }
    };
    $SIG{ALRM} = sub { $timed_out = 1 unless $external; $escalate->() };
    my %num = (INT => 2, HUP => 1, TERM => 15);
    for my $sig (keys %num) {
      $SIG{$sig} = sub { $external ||= $num{$sig}; $escalate->() if $stage == 0 };
    }
    alarm $cap;
    my ($got, $st);
    do { $got = waitpid($pid, 0); $st = $? } until $got == $pid || ($got == -1 && !$!{EINTR});
    alarm 0;
    # A member counts only if it can still run: a zombie (an orphan a slow PID 1 has not reaped)
    # answers kill 0 but consumes nothing and cannot be killed, and must not read as a survivor.
    # Unreadable ps output counts as alive, the conservative side.
    my $alive = sub {
      waitpid($pid, POSIX::WNOHANG());
      return 0 unless kill 0, -$pid;
      open(my $ps, "-|", "ps", "-A", "-o", "pgid=,stat=") or return 1;
      my ($live, $rows) = (0, 0);
      while (<$ps>) {
        my ($g, $st) = split;
        next unless defined $st;
        $rows++;
        $live = 1 if $g == $pid && $st !~ /^Z/;
      }
      close $ps;
      return $rows ? $live : 1;
    };
    # The signals go to any REACHABLE group (kill 0), whatever the census says: ps is a snapshot, and
    # a member forked during the scan must not escape cleanup. The census only ends the grace waits
    # early and decides whether what is left is a survivor or zombies.
    if (kill 0, -$pid) {
      kill "TERM", -$pid;
      for (1 .. 10) { last unless $alive->(); sleep 1 }
      kill "KILL", -$pid if kill 0, -$pid;
      for (1 .. 10) { last unless $alive->(); sleep 1 }
      if ($alive->()) { print STDERR "gh834: process group $pid survived SIGKILL\n"; exit 125 }
    }
    exit 128 + $external if $external;
    exit 124 if $timed_out;
    exit($st & 127 ? 128 + ($st & 127) : $st >> 8);
  ' "$cap" "$@"
}

device_state() {
  echo "-- $(date -u +%FT%TZ) $(uptime)"
  case $backend in
  hip) for f in /sys/class/drm/card*/device/gpu_busy_percent; do [ -r "$f" ] && echo "   gpu_busy_percent $(cat "$f")"; done ;;
  cuda) nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | sed 's/^/   nvidia-smi /' ;;
  metal) ps -Ao %cpu= | awk '{s+=$1} END {print "   host cpu% sum " s}' ;;
  esac
}

step() {
  local name=$1
  shift
  echo "== step $name: $*"
  device_state
  local t0 t1 rc
  t0=$(date +%s)
  # The supervisor runs as a job the shell waits on, so a signal to the driver is forwarded to it
  # (and through it to the group) instead of ending the shell with the step still running.
  capped "$@" >"$out/$name.out" 2>"$out/$name.err" &
  local sup=$! interrupted=
  trap 'interrupted=1; kill -TERM "$sup" 2>/dev/null' INT TERM HUP
  while kill -0 "$sup" 2>/dev/null; do wait "$sup"; done
  wait "$sup"
  rc=$?
  trap - INT TERM HUP
  [ -n "$interrupted" ] && { echo "== step $name: interrupted (exit $rc); stopping"; exit 130; }
  [ "$rc" -eq 125 ] && { echo "== step $name: a process survived the cap; stopping"; exit 125; }
  t1=$(date +%s)
  echo "== step $name: exit $rc, wall $((t1 - t0)) s"
  device_state
  return "$rc"
}

status=0 capped_any=
# A step that exited 0 without a line its conclusion rests on fails the run rather than publishing
# an incomplete record (a regressed hook, a config source that did not take).
require() {
  grep -q -- "$3" "$out/$1.$2" && return 0
  echo "== step $1: MISSING EVIDENCE in $1.$2: no line matching $3"
  status=1
}

# Every session must be a COMPLETE two-arm experiment on an uncontended box, or its wall and totals
# are not comparable across modes: a placement arm that dies can still ship its sibling
# (Train.tune_placements), and "searched":true covers a died search too.
require_complete_session() {
  local line arm
  line=$(grep '^{' "$out/$1.out" | tail -1)
  for arm in A B; do
    case $line in
    *"\"arm\":\"$arm\",\"state\":\"searched\""*) ;;
    *) echo "== step $1: INCOMPLETE SESSION: arm $arm did not complete a search"; status=1 ;;
    esac
    # A search can complete having timed nothing (every candidate declined; the GPU serial baseline
    # is never dispatched): such an arm holds no evidence about timing cost.
    case $(printf '%s' "$line" | sed -n "s/.*\"arm\":\"$arm\"[^}]*\"best_ms\":\([^,}]*\).*/\1/p") in
    '' | null) echo "== step $1: INCOMPLETE SESSION: arm $arm timed no candidate"; status=1 ;;
    esac
  done
  if ! grep -q '^timing-trace: summary: [0-9.]*s wall, [0-9]* candidate attempts, [1-9][0-9]* timing calls' \
    "$out/$1.err"; then
    echo "== step $1: INCOMPLETE SESSION: the trace recorded no timing call"
    status=1
  fi
  # Whatever shape a terminal failure takes, it is not null; and every contention count is zero.
  if [ "$(printf '%s' "$line" | grep -o '"terminal_failure":' | wc -l)" -ne \
    "$(printf '%s' "$line" | grep -o '"terminal_failure":null' | wc -l)" ]; then
    echo "== step $1: INCOMPLETE SESSION: an arm carries a terminal failure"
    status=1
  fi
  # A timing call that raised is in none of the trace's totals, so its cost split is not the session's.
  if grep -q '^timing-trace: summary: .*INCOMPLETE' "$out/$1.err"; then
    echo "== step $1: INCOMPLETE SESSION: the trace summary is missing raised timing calls"
    status=1
  fi
  # Per arm, [timings_contended] counts every refused window and [timings_unbatched] the ones refused
  # because queued calibration measured no batch within its target (gh-ocannl-1098). Either leaves
  # the session incomplete. They are named apart because the readings behind a no-batch refusal
  # cannot tell a queue threshold from a stall on every probe: one that repeats on an idle rerun is
  # the threshold. The two fields are adjacent on each arm.
  refusals=$(printf '%s' "$line" | grep -o '"timings_contended":[0-9]*,"timings_unbatched":[0-9]*')
  if printf '%s\n' "$refusals" | awk -F'[:,]' 'NF && $2 > $4 { f = 1 } END { exit !f }'; then
    echo "== step $1: CONTENDED SESSION: timing windows were refused for host contention"
    status=1
  fi
  if printf '%s\n' "$refusals" | awk -F'[:,]' 'NF && $4 > 0 { f = 1 } END { exit !f }'; then
    echo "== step $1: UNBATCHED SESSION: queued calibration measured no batch within the target" \
      "(a queue threshold if it repeats on an idle rerun, else a stall)"
    status=1
  fi
}

built= identified=
need_build() {
  [ -n "$built" ] && [ -n "$identified" ] && return 0
  echo "gh834: step $1 needs the build and provenance steps earlier in this invocation" >&2
  exit 2
}

for s in "$@"; do
  case $s in
  build)
    step build sh -c "cd '$root' && dune build bin/projection_shape_bench.exe \
      benchmarks/runners/ocannl/bench_gpt.exe" || { cat "$out/build.err"; exit 1; }
    built=1
    ;;
  provenance)
    echo "host $(hostname) sha $(git -C "$root" rev-parse HEAD) backend $backend cap ${cap}s"
    # The device identity and the fixture digest are what the results are labelled by: a probe
    # that yields nothing fails the run instead of archiving an unidentified measurement.
    case $backend in
    hip) ident=$(rocminfo 2>/dev/null | grep -E "Marketing|gfx" | sort -u | head -4) ;;
    cuda) ident=$(nvidia-smi -L 2>/dev/null) ;;
    metal) ident=$(system_profiler SPDisplaysDataType 2>/dev/null | grep -E "Chipset|Cores") ;;
    cc) ident=$(lscpu 2>/dev/null | grep -E "Model name|^CPU\(s\)" || sysctl -n machdep.cpu.brand_string 2>/dev/null) ;;
    esac
    if [ -n "$ident" ]; then echo "$ident"; else
      echo "== provenance: MISSING EVIDENCE: no device identity for $backend"
      status=1
    fi
    if [ -r "$fixture" ]; then
      echo "fixture $fixture sha256 $(shasum -a 256 "$fixture" | cut -d' ' -f1)"
    else
      echo "== provenance: MISSING EVIDENCE: fixture $fixture is not readable"
      status=1
    fi
    # A measurement needs a provenance that identified the device and the fixture.
    [ "$status" -eq 0 ] && identified=1
    ;;
  crown-fwd | crown-rev)
    need_build "$s"
    if step "$s" ../_build/default/bin/projection_shape_bench.exe 200 8 d "${s#crown-}" seeds \
      --ocannl_backend="$backend" --ocannl_log_config_sourcing=true; then
      require "$s" out '^== gh-ocannl-755: candidate ranking'
      require "$s" out '^   crown: '
      require "$s" err "^Found $backend, commandline --ocannl_backend=$backend\$"
    else status=1; fi
    # The ranking table and the crown verdicts, which are the gh-ocannl-833 deliverable.
    sed -n '/== gh-ocannl-755/,$p' "$out/$s.out"
    grep 'Found .*--ocannl_backend=' "$out/$s.err" | sort -u
    ;;
  session-isolated | session-queued)
    need_build "$s"
    mode=${s#session-}
    rm -rf "$out/cache-$mode"
    # A cold driver-side code cache per session (see the header): only CUDA's is redirectable.
    # The ${a[@]+...} form keeps an empty array legal under set -u on macOS's bash 3.2.
    nvcache=()
    if [ "$backend" = cuda ]; then
      rm -rf "$out/nvcache-$mode" && mkdir "$out/nvcache-$mode" || exit 2
      nvcache=(CUDA_CACHE_PATH="$out/nvcache-$mode")
      echo "env CUDA_CACHE_PATH=$out/nvcache-$mode (empty, for $s)"
    fi
    if step "$s" env ${nvcache[@]+"${nvcache[@]}"} BENCH_FIXTURE="$fixture" BENCH_TUNE=1 \
      BENCH_TIMING_TRACE=1 \
      ../_build/default/benchmarks/runners/ocannl/bench_gpt.exe --ocannl_backend="$backend" \
      --ocannl_autotune_timing="$mode" --ocannl_autotune_cache_dir="$out/cache-$mode" \
      --ocannl_autotune_log=false --ocannl_log_config_sourcing=true; then
      # A completed session is evidence only with its treatment and its cost record in it.
      require "$s" out '"compile_s":'
      require "$s" out '"searched":true'
      require "$s" err '^timing-trace: summary: '
      require "$s" err "^Found $backend, commandline --ocannl_backend=$backend\$"
      require "$s" err "^Found $mode, commandline --ocannl_autotune_timing=$mode\$"
      require_complete_session "$s"
    else
      rc=$?
      # A capped session is a lower bound only if the search had begun under the pinned treatment;
      # a cap spent loading the fixture or building the graph measured no search at all.
      # ... and only if the search was still going: a cap reached after the runner's
      # "search done" marker cut off the post-search steps, not the search.
      if [ "$rc" -eq 124 ] && grep -q '^timing-trace: attempt ' "$out/$s.err" &&
        ! grep -q '^timing-trace: search done: ' "$out/$s.err" &&
        grep -q "^Found $backend, commandline --ocannl_backend=$backend\$" "$out/$s.err" &&
        grep -q "^Found $mode, commandline --ocannl_autotune_timing=$mode\$" "$out/$s.err"; then
        capped_any=1
      else
        [ "$rc" -eq 124 ] &&
          echo "== step $s: capped outside its search (before it began or after it ended); not a lower bound"
        status=1
      fi
    fi
    # The session's driver cache is scratch, not evidence: record its size, then drop it.
    if [ "$backend" = cuda ]; then
      echo "nvcache-$mode $(du -sk "$out/nvcache-$mode" | cut -f1) KiB after $s; removed"
      rm -rf "$out/nvcache-$mode"
    fi
    # The result line (compile_s is the search wall) and the trace's last word: the summary on a
    # completed run, the running totals of the last timing call on a capped one.
    cat "$out/$s.out"
    grep -c 'timing-trace: attempt' "$out/$s.err" | sed 's/^/candidate attempts /'
    grep 'timing-trace: \(call\|summary\)' "$out/$s.err" | tail -2
    grep 'Found .*--ocannl_\(backend\|autotune_timing\)=' "$out/$s.err" | sort -u
    ;;
  *) echo "gh834: unknown step $s" >&2; exit 2 ;;
  esac
done
[ "$status" -eq 0 ] && [ -n "$capped_any" ] && exit 124
exit "$status"

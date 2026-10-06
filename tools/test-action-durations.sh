#!/usr/bin/env bash
# Hermetic legs for tools/action-durations.sh over synthetic dune traces.
#
# The fixtures are canonical S-expressions written here, never a real trace:
# a real one records dune's environment (gh-ocannl-1280). Every field the
# subject must not decode -- the environment and argv of `config init`, a
# non-digit time slot, process arguments, captured stdout, rusage, a log
# message -- carries the CANARY string, and every leg fails on any output that
# holds it, refusals included. Two fault-injected twins (a refusal quoting the
# bytes before the bad offset; actions named by their process arguments, as a
# command line would name them) must be rejected by that canary check alone.
#
#   tools/test-action-durations.sh          # run every leg
#   tools/test-action-durations.sh --keep   # keep the scratch directory
#
# On no dune alias, like its siblings in CI's shell-harness job: the subject
# runs `python3` under that exact name, which a host without one skips.
set -u
. "$(cd "$(dirname "$0")/../scripts" && pwd)/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch test-action-durations
HERE=$(cd "$(dirname "$0")" && pwd)
SUBJECT=$HERE/action-durations.sh
CANARY=CANARY-0c3f-not-for-output
mkdir -p "$TMP/runs"

python3 - "$TMP" "$CANARY" <<'PY'
import sys
from pathlib import Path

root, canary = Path(sys.argv[1]), sys.argv[2]
S = 1_000_000_000
T0 = 1_791_291_000 * S


def atom(s):
    s = str(s).encode()
    return str(len(s)).encode() + b':' + s


def sexp(x):
    return b'(' + b''.join(sexp(e) if isinstance(e, list) else atom(e) for e in x) + b')'


def rest(pid, prog, directory, targets, exit_=None):
    out = [['process_args', ['--ocannl_backend=cc', '--token=' + canary]], ['pid', str(pid)],
           ['categories', []], ['prog', prog], ['dir', directory]]
    if exit_ is not None:
        out.append(['exit', exit_])
    if targets:
        out.append(['target_files', targets])
    if exit_ is not None:
        out += [['stdout', canary],
                ['rusage', [['user_cpu_time', '1'], ['system_cpu_time', canary]]]]
    return out


def proc(pid, start_s, dur_s, prog, directory, targets, exit_='0'):
    start = T0 + int(start_s * S)
    events = [sexp(['process', 'start', str(start)] + rest(pid, prog, directory, targets)
                   + [['queued', '1000']])]
    if dur_s is not None:
        events.append(sexp(['process', 'finish', [str(start), str(int(round(dur_s * S)))]]
                           + rest(pid, prog, directory, targets, exit_)))
    return events


b = '_build/default/'
head = [
    sexp(['config', 'init', canary, ['version', '3.24.2'], ['argv', ['dune', 'build', '--x=' + canary]],
          ['env', ['FAKE_API_TOKEN=' + canary, 'HOME=/home/' + canary]], ['pid', '1']]),
    sexp(['log', 'info', str(T0), ['message', canary]]),
    sexp(['build', 'build-start', str(T0), ['run_id', '1']]),
]
body = (
    proc(11, 1, 21.5, '/opt/bin/fsm_transformer.exe', b + 'test/training',
         [b + 'test/training/fsm_transformer.actual'])
    + proc(12, 2, 6.72, 'env_var_deps.exe', '_build/.sandbox/53d3fec81e775d978ce99f0295ebd390/default/test/operations',
           [b + 'test/operations/env_var_deps.actual', b + 'test/operations/env_var_deps.filelist'])
    + proc(13, 3, 1.96, 'env_var_deps.exe', b + 'test/operations',
           [b + 'test/operations/env_var_deps_control.actual'])
    + proc(14, 4, 3.25, '/opt/bin/ocamlopt.opt', b + 'arrayjit/lib', [])
    + proc(15, 5, 5.43, '/bin/sh', b + 'test/operations',
           ['_build/.actions/default/test/operations/runtest-slot_kind_cases-4050ca3943e5638ece3bbcdcaf49ae6e'], '1')
    + proc(16, 10, None, 'hung_probe.exe', b + 'test/operations', [b + 'test/operations/hung_probe.exe.output'])
)
tail = [sexp(['action', 'write-file', [str(T0 + 50 * S), '1000'], ['file', canary], ['size', '3']])]
(root / 'trace').write_bytes(b''.join(head + body + tail))
# Killed mid-event: the finish record of pid 21 is cut short, so pid 21 is open
# and its lower bound runs to the last complete event, 7 s after its start.
cut = proc(21, 1, 30, 'slow.exe', b + 'test/operations', [b + 'test/operations/slow.output'])
mark = sexp(['action', 'write-file', [str(T0 + 8 * S), '1000'], ['file', canary], ['size', '3']])
(root / 'truncated').write_bytes(b''.join(head + [cut[0], mark]) + cut[1][:len(cut[1]) // 2])
# Malformed: a stray byte right after an atom that holds the canary.
(root / 'malformed').write_bytes(head[0] + b'(3:log4:info' + atom(canary) + b'!)')
(root / 'no-process').write_bytes(b''.join(head))
PY

cat >"$TMP/table.expected" <<'EOF'
processes=6  open=1
  seconds  exit  prog                         action
    40.00  open  hung_probe.exe               test/operations/hung_probe.exe.output
    21.50     0  fsm_transformer.exe          test/training/fsm_transformer.actual
     6.72     0  env_var_deps.exe             test/operations/env_var_deps.actual +1
     5.43     1  sh                           @test/operations/runtest-slot_kind_cases
     3.25     0  ocamlopt.opt                 (in arrayjit/lib)
     1.96     0  env_var_deps.exe             test/operations/env_var_deps_control.actual
EOF
cat >"$TMP/prog.expected" <<'EOF'
processes=6  open=1
    max_s   total_s     n  prog
    40.00     40.00     1  hung_probe.exe
    21.50     21.50     1  fsm_transformer.exe
     6.72      8.68     2  env_var_deps.exe
     5.43      5.43     1  sh
     3.25      3.25     1  ocamlopt.opt
EOF
cat >"$TMP/dir.expected" <<'EOF'
processes=6  open=1
    max_s   total_s     n  dir
    40.00     54.11     4  test/operations
    21.50     21.50     1  test/training
     3.25      3.25     1  arrayjit/lib
EOF
cat >"$TMP/filtered.expected" <<'EOF'
processes=4  open=1
  seconds  exit  prog                         action
    40.00  open  hung_probe.exe               test/operations/hung_probe.exe.output
    21.50     0  fsm_transformer.exe          test/training/fsm_transformer.actual
EOF
cat >"$TMP/truncated.expected" <<'EOF'
processes=1  open=1
  seconds  exit  prog                         action
     7.00  open  slow.exe                     test/operations/slow.output
EOF

# run_subject SUBJECT LABEL ARG... -> $TMP/runs/LABEL/{stdout,stderr,rc}
run_subject() {
  local subject=$1 label=$2 rc=0
  shift 2
  mkdir -p "$TMP/runs/$label"
  bash "$subject" "$@" >"$TMP/runs/$label/stdout" 2>"$TMP/runs/$label/stderr" || rc=$?
  printf '%s\n' "$rc" >"$TMP/runs/$label/rc"
}
leaked() { # LABEL: did either stream carry the canary?
  grep -qF -- "$CANARY" "$TMP/runs/$1/stdout" "$TMP/runs/$1/stderr"
}
# exact SUBJECT LABEL EXPECTED ARG...: exit 0, the exact table, an empty stderr, no canary.
exact() {
  local subject=$1 label=$2 expected=$3
  shift 3
  run_subject "$subject" "$label" "$@"
  if leaked "$label"; then return 1; fi
  if [ "$(cat "$TMP/runs/$label/rc")" != 0 ]; then return 1; fi
  if [ -s "$TMP/runs/$label/stderr" ]; then return 1; fi
  cmp -s "$expected" "$TMP/runs/$label/stdout"
}
# refused SUBJECT LABEL RC PATTERN ARG...: that exit, PATTERN on stderr, an empty stdout, no canary.
refused() {
  local subject=$1 label=$2 want=$3 pattern=$4
  shift 4
  run_subject "$subject" "$label" "$@"
  if leaked "$label"; then return 1; fi
  if [ "$(cat "$TMP/runs/$label/rc")" != "$want" ]; then return 1; fi
  if [ -s "$TMP/runs/$label/stdout" ]; then return 1; fi
  grep -qE -- "$pattern" "$TMP/runs/$label/stderr"
}
leg() { # LABEL COMMAND...: report the command's verdict, pointing at the run on failure
  local label=$1 rc=0
  shift
  "$@" || rc=$?
  report "$rc" "$label" "see $TMP/runs"
}

leg "slowest actions, open process as a lower bound, canary-free" \
  exact "$SUBJECT" table "$TMP/table.expected" "$TMP/trace"
leg "--group prog sums repeated executables" \
  exact "$SUBJECT" prog "$TMP/prog.expected" --group prog "$TMP/trace"
leg "--group dir groups by the action's directory" \
  exact "$SUBJECT" dir "$TMP/dir.expected" --group dir "$TMP/trace"
leg "--prog filters by program basename and -n limits rows" \
  exact "$SUBJECT" filtered "$TMP/filtered.expected" --prog '\.exe$' -n 2 "$TMP/trace"

mkdir -p "$TMP/repo/tools" "$TMP/repo/_build"
cp "$SUBJECT" "$TMP/repo/tools/"
cp "$TMP/trace" "$TMP/repo/_build/trace.csexp"
leg "the default trace is _build/trace.csexp at the repository root" \
  exact "$TMP/repo/tools/action-durations.sh" default "$TMP/table.expected"

truncated_leg() {
  run_subject "$SUBJECT" truncated "$TMP/truncated"
  if leaked truncated; then return 1; fi
  if [ "$(cat "$TMP/runs/truncated/rc")" != 0 ]; then return 1; fi
  if ! cmp -s "$TMP/truncated.expected" "$TMP/runs/truncated/stdout"; then return 1; fi
  grep -qE 'warning: the trace ends inside an event at byte [0-9]+ of [0-9]+' "$TMP/runs/truncated/stderr"
}
leg "a trace cut mid-event is read up to there, with a warning" truncated_leg

leg "a malformed trace is refused by offset, quoting no bytes" \
  refused "$SUBJECT" malformed 1 'not a canonical-S-expression dune trace .*malformed at byte [0-9]+$' "$TMP/malformed"
leg "a trace with no process is refused" \
  refused "$SUBJECT" no-process 1 'no process in' "$TMP/no-process"
leg "--prog matching nothing is refused" \
  refused "$SUBJECT" no-match 1 'matches --prog' --prog '^nothing$' "$TMP/trace"
leg "a missing trace is a usage error" \
  refused "$SUBJECT" missing 2 'no trace at' "$TMP/no-such-trace"
leg "an unknown --group is a usage error" \
  refused "$SUBJECT" bad-group 2 'wants prog or dir' --group alias "$TMP/trace"
leg "a non-numeric -n is a usage error" \
  refused "$SUBJECT" bad-n 2 'wants a count' -n x "$TMP/trace"
leg "an invalid --prog regex is a usage error" \
  refused "$SUBJECT" bad-regex 2 -- '--prog: ' --prog '[' "$TMP/trace"

# Fault-injected twins. Each must change the subject (else the control is
# vacuous) and be rejected by the canary check, not by some other difference.
twin() { # NAME OLD NEW [OLD NEW]...: each OLD must occur exactly once
  local name=$1
  shift
  python3 - "$SUBJECT" "$TMP/$name.sh" "$@" <<'PY'
import sys
src, dst, pairs = sys.argv[1], sys.argv[2], sys.argv[3:]
text = open(src).read()
for old, new in zip(pairs[::2], pairs[1::2]):
    if text.count(old) != 1:
        sys.exit('twin: expected one occurrence of %r' % old)
    text = text.replace(old, new)
open(dst, 'w').write(text)
PY
}
twin_leg() { # NAME LABEL ORACLE...
  local name=$1 label=$2
  shift 2
  if [ ! -s "$TMP/$name.sh" ]; then
    report 1 "negative control: $label" "the twin could not be built"
  elif "$@"; then
    report 1 "negative control: $label" "the shipping oracle accepted the twin"
  elif leaked "$name"; then
    report 0 "negative control: $label"
  else
    report 1 "negative control: $label" "rejected without the canary; see $TMP/runs/$name"
  fi
}
twin quoting \
  "'malformed at byte %d' % (path, e.args[0]))" \
  "'malformed at byte %d near %r' % (path, e.args[0], data[max(0, e.args[0] - 40):e.args[0]]))"
twin_leg quoting "a refusal quoting the bytes before the bad offset leaks the canary" \
  refused "$TMP/quoting.sh" quoting 1 'malformed at byte' "$TMP/malformed"
twin by-args \
  "FIELDS = {b'prog', b'dir', b'target_files', b'exit', b'pid'}" \
  "FIELDS = {b'prog', b'dir', b'target_files', b'exit', b'pid', b'process_args'}" \
  "% (s, ex, prog, action(f)[1]))" \
  "% (s, ex, prog, action(f)[1] + ' ' + ' '.join(f.get('process_args', []))))"
twin_leg by-args "actions named by their process arguments leak the canary" \
  exact "$TMP/by-args.sh" by-args "$TMP/table.expected" "$TMP/trace"
finish

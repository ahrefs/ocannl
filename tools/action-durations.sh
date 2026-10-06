#!/usr/bin/env bash
# Which dune actions took longest? Reads a dune trace -- `_build/trace.csexp`,
# which every dune >= 3.24 invocation rewrites, or a `--trace-file` -- and
# prints the slowest processes dune started, longest first, so a per-action
# bound (test-run.sh's `--test-cap`, a slot's cap) is set from data rather
# than from memory. Each process dune starts is one action: a test executable,
# a compiler call, a formatter run. Durations are the trace's wall-clock
# nanoseconds from start to exit; under a loaded box they include the load.
#
# Usage: tools/action-durations.sh [options] [TRACE]
#   TRACE              the trace to read (default: _build/trace.csexp at the
#                      repository root, a bare dune's; `tools/test-run.sh run`
#                      hands its batch a trace file in the run directory and
#                      keeps this tool's five slowest rows as `slowest`)
#   -n N               print the N slowest rows (default: 20; 0 prints all)
#   --prog REGEX       keep processes whose program basename matches (Python
#                      regular expression search), e.g. '\.exe$' for tests
#   --group prog|dir   summarize instead: per program basename (one test
#                      executable is one unit) or per directory of the
#                      action, as max/total seconds and count, by max
#   -h, --help         this header
#
# The trace records no alias membership, so a row names its action by what it
# produces: `@dir/alias` for an alias action, else its first target relative
# to the build context (`+k` more), else the directory it ran in. A process
# the trace saw start but never finish (a run killed hard, or a trace read
# while dune runs) prints `open` for its exit, and its seconds are a lower
# bound: up to the trace's last timestamp. A trace that ends mid-event is read
# up to there, with a warning. Exit 0 on a table, 1 when the trace is
# malformed or holds no (matching) process, 2 on a usage error.
#
# Paths are shown relative to the build context: the build directory is the
# trace's own `build_dir` (so `--build-dir`/DUNE_BUILD_DIR traces read alike),
# `_build` when the trace lacks one.
#
# SECURITY (gh-ocannl-1280): a trace also records dune's environment, argv,
# and each process's arguments and captured stdout/stderr, any of which can
# hold a credential. This reader decodes only event kinds, the time slot (kept
# only when it is all decimal digits), `build_dir` of `config init`, and five
# fields of process events -- prog, dir, target_files, exit, pid (WANTED
# below). Everything else is skipped by its length prefixes without being
# decoded, and no diagnostic quotes trace bytes (only offsets). Keep it that
# way: tools/test-action-durations.sh plants a canary in every other field of
# the schema it inventories and fails on any output that carries it.
# The boundary assumes producer-framed csexp, as dune writes it: the length
# prefixes are what separate a field from the next record, so a corrupted or
# tampered prefix can swallow a neighbouring record (the environment
# included) into a whitelisted field, and that prints with exit 0. Nesting
# checks cannot catch it, since the result is still well-formed csexp.
# Protection against tampered traces is producer-side scrubbing
# (ahrefs/ocannl#1280), not this reader; the harness pins the boundary.

set -euo pipefail

root=$(cd "$(dirname "$0")/.." && pwd)
rows=20
prog_re=
group=
trace=

die() {
  echo "action-durations.sh: $*" >&2
  exit 2
}

while [ $# -gt 0 ]; do
  case "$1" in
  -n)
    [ $# -ge 2 ] || die "-n needs a value"
    rows=$2
    shift 2
    ;;
  --prog)
    [ $# -ge 2 ] || die "--prog needs a value"
    prog_re=$2
    shift 2
    ;;
  --group)
    [ $# -ge 2 ] || die "--group needs a value"
    group=$2
    shift 2
    ;;
  -h | --help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  -*) die "unknown option: $1 (try --help)" ;;
  *)
    [ -z "$trace" ] || die "one trace at a time (got '$trace' and '$1')"
    trace=$1
    shift
    ;;
  esac
done
case "$rows" in '' | *[!0-9]*) die "-n wants a count, got: $rows" ;; esac
case "$group" in '' | prog | dir) ;; *) die "--group wants prog or dir, got: $group" ;; esac
trace=${trace:-$root/_build/trace.csexp}
[ -f "$trace" ] || die "no trace at $trace (build something first, or name one)"

python3 - "$trace" "$rows" "$prog_re" "$group" <<'PY'
import os
import re
import sys

path, rows, prog_re, group = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
try:
    prog_pat = re.compile(prog_re) if prog_re else None
except re.error as e:
    print('action-durations.sh: --prog: %s' % e, file=sys.stderr)
    sys.exit(2)
data = open(path, 'rb').read()
n = len(data)
OPEN, CLOSE, COLON = 40, 41, 58
PROCESS = {b'prog', b'dir', b'target_files', b'exit', b'pid'}
WANTED = {(b'process', b'start'): PROCESS, (b'process', b'finish'): PROCESS,
          (b'config', b'init'): {b'build_dir'}}


class Short(Exception):
    """The trace ends inside an event."""


class Bad(Exception):
    """Not canonical S-expressions; carries an offset, never trace bytes."""


def atom(i):
    # `N:bytes`; returns the (start, end) of the bytes and the next offset.
    j = i
    while j < n and 48 <= data[j] <= 57:
        j += 1
    if j >= n:
        raise Short()
    if j == i or data[j] != COLON:
        raise Bad(i)
    start = j + 1
    end = start + int(data[i:j])
    if end > n:
        raise Short()
    return start, end, end


def skip(i):
    # Step over one element, atom or list, without decoding any of it.
    depth = 0
    while True:
        if i >= n:
            raise Short()
        if data[i] == OPEN:
            depth += 1
            i += 1
        elif data[i] == CLOSE:
            if depth == 0:
                raise Bad(i)
            depth -= 1
            i += 1
        else:
            i = atom(i)[2]
        if depth == 0:
            return i


def value(i):
    # Decode one element: an atom as str, a list as a list.
    if i >= n:
        raise Short()
    if data[i] != OPEN:
        s, e, i = atom(i)
        return data[s:e].decode('utf8', 'replace'), i
    out, i = [], i + 1
    while True:
        if i >= n:
            raise Short()
        if data[i] == CLOSE:
            return out, i + 1
        v, i = value(i)
        out.append(v)


def skip_to_close(i):
    while True:
        if i >= n:
            raise Short()
        if data[i] == CLOSE:
            return i + 1
        i = skip(i)


def head_atom(i):
    if i >= n:
        raise Short()
    if data[i] == OPEN or data[i] == CLOSE:
        return None, i
    s, e, i = atom(i)
    return data[s:e], i


def time_slot(i):
    # The third element is the event's time: `ts` or `(ts duration)`, in ns.
    # Its atoms are checked as bytes and decoded only when all decimal digits.
    if i >= n:
        raise Short()
    if data[i] != OPEN:
        s, e, j = atom(i)
        spans = [(s, e)]
    else:
        spans, j = [], i + 1
        while True:
            if j >= n:
                raise Short()
            if data[j] == CLOSE:
                j += 1
                break
            if data[j] == OPEN:
                return None, skip(i)
            s, e, j = atom(j)
            spans.append((s, e))
    if spans and all(data[s:e].isdigit() for s, e in spans):
        return [int(data[s:e]) for s, e in spans], j
    return None, j


def event(i):
    # One top-level event -> (kind, time, fields), with only whitelisted parts decoded.
    if data[i] != OPEN:
        raise Bad(i)
    cat, i = head_atom(i + 1)
    name, i = head_atom(i) if cat is not None else (None, i)
    if i >= n:
        raise Short()
    if data[i] == CLOSE:
        return None, i + 1
    t, i = time_slot(i)
    wanted = WANTED.get((cat, name))
    if wanted is None:
        return ((cat, name), t, None), skip_to_close(i)
    fields = {}
    while True:
        if i >= n:
            raise Short()
        if data[i] == CLOSE:
            return ((cat, name), t, fields), i + 1
        if data[i] != OPEN:
            i = skip(i)
            continue
        key, j = head_atom(i + 1)
        if key in wanted:
            v, j = value(j)
            fields[key.decode()] = v
        i = skip_to_close(j)


events, i, truncated = [], 0, None
try:
    while i < n:
        if data[i] in b' \t\r\n':
            i += 1
            continue
        start = i
        ev, i = event(i)
        if ev is not None:
            events.append(ev)
except Short:
    truncated = start
except Bad as e:
    sys.exit('action-durations.sh: %s is not a canonical-S-expression dune trace (dune >= 3.24): '
             'malformed at byte %d' % (path, e.args[0]))
if truncated is not None:
    print('action-durations.sh: warning: the trace ends inside an event at byte %d of %d '
          '(dune still running, or killed); read up to there' % (truncated, n), file=sys.stderr)

last = 0
for _, t, _ in events:
    if t:
        last = max(last, sum(t) if len(t) == 2 else t[0])


config = next((f for kind, _, f in events if kind == (b'config', b'init')), {})
build_dir = config.get('build_dir') if isinstance(config.get('build_dir'), str) else '_build'


def in_context(path):
    # (alias or None, path relative to the build context) for a path under the build directory:
    # `<build>/<ctx>/...`, a sandbox's `<build>/.sandbox/<hash>/<ctx>/...`, or an alias action's
    # `<build>/.actions/<ctx>/<dir>/<alias>-<hash>`. Anything else is returned unchanged.
    prefix = build_dir.rstrip('/') + '/'
    if not path.startswith(prefix):
        return None, path
    rel = path[len(prefix):]
    m = re.match(r'\.actions/[^/]+/(?:(.*)/)?([^/]+)-[0-9a-f]{32}$', rel)
    if m:
        return m.group(2), m.group(1) or '.'
    m = re.match(r'(?:\.sandbox/[^/]+/)?[^/]+(?:/(.*))?$', rel)
    return None, (m.group(1) or '.') if m else rel


def action(f):
    # (directory, text) naming what the process produced, relative to the build context.
    targets = f.get('target_files')
    targets = [t for t in targets if isinstance(t, str)] if isinstance(targets, list) else []
    if not targets:
        d = in_context(f['dir'])[1] if isinstance(f.get('dir'), str) else '?'
        return d, '(in %s)' % d
    alias, rel = in_context(targets[0])
    if alias is not None:
        d, text = rel, '@%s%s' % ('' if rel == '.' else rel + '/', alias)
    else:
        d, text = os.path.dirname(rel) or '.', rel
    return d, text + (' +%d' % (len(targets) - 1) if len(targets) > 1 else '')


def exit_text(f):
    e = f.get('exit', '?')
    return e if isinstance(e, str) else ' '.join(x if isinstance(x, str) else '?' for x in e)


started, done = {}, []
for kind, t, f in events:
    if f is None or not t or kind[0] != b'process':
        continue
    key = (f.get('pid'), t[0])
    if kind[1] == b'start':
        started[key] = f
    elif len(t) == 2:
        started.pop(key, None)
        done.append((t[1], exit_text(f), f))
done += [(max(0, last - key[1]), 'open', f) for key, f in started.items()]
procs = []
for ns, ex, f in done:
    prog = os.path.basename(f.get('prog', '')) if isinstance(f.get('prog'), str) else '?'
    if prog_pat is None or prog_pat.search(prog):
        procs.append((ns / 1e9, ex, prog, f))
if not procs:
    sys.exit('action-durations.sh: no process in %s%s (a cached build starts none)'
             % (path, ' matches --prog' if prog_pat else ''))

print('processes=%d  open=%d' % (len(procs), sum(1 for p in procs if p[1] == 'open')))
if group:
    agg = {}
    for s, ex, prog, f in procs:
        k = prog if group == 'prog' else action(f)[0]
        mx, tot, cnt = agg.get(k, (0.0, 0.0, 0))
        agg[k] = (max(mx, s), tot + s, cnt + 1)
    ranked = sorted(agg.items(), key=lambda kv: (-kv[1][0], kv[0]))
    print('%9s %9s %5s  %s' % ('max_s', 'total_s', 'n', group))
    for k, (mx, tot, cnt) in ranked[:rows or None]:
        print('%9.2f %9.2f %5d  %s' % (mx, tot, cnt, k))
else:
    procs.sort(key=lambda p: (-p[0], p[2], action(p[3])[1]))
    print('%9s %5s  %-28s %s' % ('seconds', 'exit', 'prog', 'action'))
    for s, ex, prog, f in procs[:rows or None]:
        print('%9.2f %5s  %-28s %s' % (s, ex, prog, action(f)[1]))
PY

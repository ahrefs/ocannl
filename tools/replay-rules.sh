#!/usr/bin/env bash
# The replay half of `tools/test-run.sh repeat --direct` (gh-ocannl-1377): re-run
# the rules a test's own action is made of, in the build directory dune has
# already built, without asking dune to rebuild anything. dune 3.24 re-runs a
# target-producing rule (a `(test)` stanza's run into `NAME.exe.output`, a
# `(rule (target NAME.actual) ...)`) only when its dependencies change, and
# trusts its database over the file system, so neither `--force` nor deleting
# the target makes it run again; and a clean build directory per iteration is a
# full rebuild (21-23 s of a 22 s `runtest-hello_world_op` iteration on
# mac-studio). So a direct repeat builds once, and then each iteration replays
# those rules here and lets dune's `--force` re-run the alias actions (the
# golden diffs, a run-and-diff action, an inline-test runner) itself.
#
# Not for callers: tools/test-run.sh drives both subcommands.
#
#   replay-rules.sh plan --dune DUNE --build-dir B --rows ROWS --aliases ALIASES --out PLAN
#   replay-rules.sh run --plan PLAN --out DIR [--parents FILE] [--snapshot]
#
# `plan` reads ROWS -- the build phase's processes test-run.sh attributed to
# the named aliases, `ALIAS<TAB>DIR<TAB>SECONDS<TAB>EXIT<TAB>PROG<TAB>ACTION
# [<TAB>TARGET]...`, DIR the process's directory relative to the build
# directory (tools/action-durations.sh --tsv --dirs) -- and asks dune for the
# rule of every file target among them (`dune rules`), dune's own expanded
# form of the action: its directory, programs, arguments, redirections and
# variables, nothing restated here. A row whose only target is the alias
# action needs no plan: dune's `--force` re-runs it. It refuses, before any
# iteration, what a replay cannot reproduce faithfully:
#   - exit 4, NOT RUN: a named alias none of whose processes the build phase
#     ran (a compile error, or an alias with no test of its own);
#   - exit 3, SANDBOXED: a rule dune ran in a sandbox (`(sandbox always)`):
#     its directory and its view of the build tree are dune's to make;
#   - exit 5, UNSUPPORTED: an action form outside the subset below, or a
#     directory target.
# The subset is the one every target rule under test/ and arrayjit/test used
# when this was written: chdir, setenv, run, progn, with-stdout-to,
# with-stderr-to, with-outputs-to, with-stdin-from, ignore-stdout,
# ignore-stderr, ignore-outputs, with-accepted-exit-codes, echo, copy,
# write-file. Rules are ordered so a rule reading another's target runs after it.
#
# `run` replays the plan into DIR. Each run starts in the environment dune
# gives an action -- read afresh from `dune exec --no-build` (the build
# context's PATH, OCAMLPATH, INSIDE_DUNE, DUNE_SOURCEROOT, ... over the
# caller's environment), plus what dune adds per action and `exec` does not:
# a TMPDIR of its own (removed afterwards) and the build context in
# BUILD_PATH_PREFIX_MAP; dune's per-action DUNE_ACTION_TRACE_DIR is not set.
# Standard input is /dev/null unless the rule redirects it, as under dune. It
# writes:
#   outputs        every replayed target's bytes, `== REL (N bytes)` before each
#   statuses       `REL: ok` or `REL: exit N` per rule (REL its first target)
#   replay-stdout  what a rule printed to a stdout it did not redirect
#   replay-stderr  the same for stderr
# and exits 0, or 1 when a rule failed (a status its action does not accept),
# as dune would; 2 on usage, 126 when the plan or the environment cannot be
# read. Files a test writes beside its targets (build_files/, logs) stay in
# the build tree between iterations, as they would between two dune runs.
# `--snapshot` executes nothing: it writes `outputs` for the targets as they
# stand -- dune's own run of the same rules, right after the build phase,
# which the repeat compares with the first replay (the replay's fidelity
# witness). `--parents FILE` records this process's pid there, so the
# per-test cap's watcher (tools/per-test-cap.sh) times the processes it
# starts as it times dune's.
#
# Boundary: a rule that only copies or concatenates a replayed output
# without starting a process is in no row, so it is not replayed; the
# `outputs` record still holds the replayed bytes it would copy.

set -u
command -v python3 >/dev/null 2>&1 || { echo "replay-rules: python3 is required" >&2; exit 126; }
# PEP 538: a python started under the C locale coerces it and exports LC_CTYPE to
# every child -- a variable dune never gives an action. Disabled for this process,
# and the switch taken back out of its environment before anything is spawned.
[ -n "${PYTHONCOERCECLOCALE+set}" ] || export PYTHONCOERCECLOCALE=0 REPLAY_RULES_COERCE_UNSET=1
exec python3 - "$@" <<'PY'
import json
import os
import shutil
import subprocess
import sys
import tempfile

if os.environ.pop('REPLAY_RULES_COERCE_UNSET', None):
    os.environ.pop('PYTHONCOERCECLOCALE', None)


def die(code, msg):
    sys.stderr.write('replay-rules: %s\n' % msg)
    sys.exit(code)


def dune_cmd(dune):
    # tools/dune-quiet.sh stands in for dune on Windows: a script, run by bash.
    return ['bash', dune] if dune.endswith('.sh') else [dune]


def options(argv, names, flags=()):
    out, i = {}, 0
    while i < len(argv):
        a = argv[i]
        if a in flags:
            out[a[2:]] = True
            i += 1
        elif a in names and i + 1 < len(argv):
            out[a[2:]] = argv[i + 1]
            i += 2
        else:
            die(2, 'unexpected argument: %s' % a)
    return out


# --- dune's printed S-expressions -------------------------------------------

def unquote(s):
    out, i = [], 1
    simple = {'n': '\n', 't': '\t', 'r': '\r', 'b': '\b', '\\': '\\', '"': '"', "'": "'", ' ': ' '}
    while i < len(s) - 1:
        c = s[i]
        if c != '\\':
            out.append(c)
            i += 1
            continue
        n = s[i + 1]
        if n in simple:
            out.append(simple[n])
            i += 2
        elif n.isdigit():
            out.append(chr(int(s[i + 1:i + 4])))
            i += 4
        elif n == 'x':
            out.append(chr(int(s[i + 2:i + 4], 16)))
            i += 4
        elif n == '\n':
            i += 2
            while i < len(s) - 1 and s[i] in ' \t':
                i += 1
        else:
            out.append(c)
            i += 1
    return ''.join(out)


def parse_all(text):
    stack, top, i, n = [], [], 0, len(text)
    cur = top
    while i < n:
        c = text[i]
        if c in ' \t\r\n':
            i += 1
        elif c == '(':
            stack.append(cur)
            new = []
            cur.append(new)
            cur = new
            i += 1
        elif c == ')':
            if not stack:
                die(126, 'unbalanced dune rules output')
            cur = stack.pop()
            i += 1
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == '\\' else 1
            cur.append(unquote(text[i:j + 1]))
            i = j + 1
        else:
            j = i
            while j < n and text[j] not in ' \t\r\n()"':
                j += 1
            cur.append(text[i:j])
            i = j
    if stack:
        die(126, 'unterminated dune rules output')
    return top


def fields(rule):
    return {x[0]: x[1:] for x in rule if isinstance(x, list) and x and isinstance(x[0], str)}


SUPPORTED = {'chdir', 'setenv', 'run', 'progn', 'with-stdout-to', 'with-stderr-to', 'with-outputs-to',
             'with-stdin-from', 'ignore-stdout', 'ignore-stderr', 'ignore-outputs',
             'with-accepted-exit-codes', 'echo', 'copy', 'write-file'}
NESTED = {'chdir': 2, 'setenv': 3, 'with-stdout-to': 2, 'with-stderr-to': 2, 'with-outputs-to': 2,
          'with-stdin-from': 2, 'ignore-stdout': 1, 'ignore-stderr': 1, 'ignore-outputs': 1,
          'with-accepted-exit-codes': 2}


def unsupported(a):
    # The first form of the action outside the subset, or None.
    if not isinstance(a, list) or not a or not isinstance(a[0], str):
        return repr(a)[:80]
    h = a[0]
    if h == 'ext' and len(a) > 1 and isinstance(a[1], str):
        # dune prints an action it implements outside its core language as `(ext "(NAME ...)")`
        # -- `system`, `bash`, ... -- so name the form inside.
        return a[1].lstrip('(').split(' ', 1)[0] or h
    if h not in SUPPORTED:
        return h
    if h == 'progn':
        for x in a[1:]:
            u = unsupported(x)
            if u:
                return u
        return None
    if h in NESTED:
        if len(a) != NESTED[h] + 1:
            return '%s with %d operands' % (h, len(a) - 1)
        for x in a[1:-1]:
            if not isinstance(x, str) and h != 'with-accepted-exit-codes':
                return '%s with a non-atom operand' % h
        return unsupported(a[-1])
    if any(not isinstance(x, str) for x in a[1:]):
        return '%s with a non-atom operand' % h
    if h in ('copy', 'write-file') and len(a) != 3:
        return '%s with %d operands' % (h, len(a) - 1)
    if h == 'run' and len(a) < 2:
        return 'run without a program'
    return None


def plan(argv):
    o = options(argv, {'--dune', '--build-dir', '--rows', '--aliases', '--out'})
    for k in ('dune', 'build-dir', 'rows', 'aliases', 'out'):
        if k not in o:
            die(2, 'plan needs --%s' % k)
    build = os.path.realpath(o['build-dir'])
    aliases = [l.rstrip('\n') for l in open(o['aliases']) if l.strip()]
    rows = []
    for line in open(o['rows']):
        f = line.rstrip('\n').split('\t')
        if len(f) < 6:
            die(126, 'malformed attributed row: %r' % line[:200])
        rows.append({'alias': f[0], 'dir': f[1], 'targets': f[6:] or [f[5]]})
    seen = {a: [] for a in aliases}
    for r in rows:
        seen.setdefault(r['alias'], []).append(r)
    missing = [a for a in aliases if not seen.get(a)]
    if missing:
        die(4, 'NOT RUN -- no process of the build phase is the action of %s, so there is '
               'nothing to replay' % ', '.join(missing))
    wanted, owners = [], {}
    for r in rows:
        files = [t for t in r['targets'] if not t.startswith('@') and not t.startswith('(')]
        if not files:
            continue
        if r['dir'].startswith('.sandbox/'):
            die(3, '%s runs %s in a dune sandbox, which a replay cannot reproduce'
                   % (r['alias'], files[0]))
        for t in files:
            if t not in owners:
                wanted.append(t)
                owners[t] = []
            if r['alias'] not in owners[t]:
                owners[t].append(r['alias'])
    rules = []
    if wanted:
        trace = o['out'] + '.trace'
        found = []
        for k in range(0, len(wanted), 200):
            cmd = dune_cmd(o['dune']) + ['rules', '--build-dir=' + build, '--trace-file=' + trace] + wanted[k:k + 200]
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
            try:
                os.unlink(trace)
            except OSError:
                pass
            if p.returncode != 0:
                sys.stderr.write(p.stderr.decode('utf8', 'replace')[-2000:])
                die(126, '`dune rules` exited %d' % p.returncode)
            found += parse_all(p.stdout.decode('utf8', 'surrogateescape'))
        by_target = {}
        for rule in found:
            f = fields(rule)
            files, dirs = [], []
            for x in (f.get('targets') or [[]])[0]:
                if isinstance(x, list) and x and x[0] == 'files':
                    files = x[1] if len(x) > 1 else []
                if isinstance(x, list) and x and x[0] == 'directories':
                    dirs = x[1] if len(x) > 1 else []
            ctx = (f.get('context') or ['default'])[0]
            entry = {'targets': files, 'directories': dirs, 'context': os.path.join(build, ctx),
                     'action': (f.get('action') or [None])[0], 'deps': []}
            for d in (f.get('deps') or [[]])[0]:
                if isinstance(d, list) and len(d) == 2 and d[0] == 'File' and isinstance(d[1], list) \
                        and len(d[1]) == 2 and d[1][0] == 'In_build_dir':
                    entry['deps'].append(d[1][1])
            for t in files:
                by_target[t] = entry
        for t in wanted:
            hits = [e for p, e in by_target.items()
                    if os.path.relpath(os.path.realpath(p), os.path.realpath(e['context'])) == t]
            if len(hits) != 1:
                die(126, '`dune rules` gave %d rules for %s' % (len(hits), t))
            e = hits[0]
            if any(e is r for r in rules):
                continue
            if e['directories']:
                die(5, 'the rule for %s has a directory target, which a replay does not make' % t)
            u = unsupported(e['action'])
            if u:
                die(5, 'the rule for %s uses `%s`, outside the forms a replay interprets'
                       % (t, u))
            e['aliases'] = owners[t]
            e['rel'] = [os.path.relpath(os.path.realpath(x), os.path.realpath(e['context']))
                        for x in e['targets']]
            rules.append(e)
        # A rule reading another's target runs after it (stable otherwise).
        made = {}
        for i, e in enumerate(rules):
            for t in e['targets']:
                made[t] = i
        order, state = [], {}

        def visit(i):
            if state.get(i) == 2:
                return
            if state.get(i) == 1:
                die(126, 'the replayed rules depend on each other in a cycle')
            state[i] = 1
            for d in rules[i]['deps']:
                if d in made and made[d] != i:
                    visit(made[d])
            state[i] = 2
            order.append(i)
        for i in range(len(rules)):
            visit(i)
        rules = [rules[i] for i in order]
    out = {'dune': o['dune'], 'build_dir': build, 'root': os.getcwd(), 'rules': [
        {'targets': e['targets'], 'rel': e['rel'], 'context': e['context'], 'action': e['action'],
         'aliases': e['aliases']} for e in rules]}
    with open(o['out'] + '.tmp', 'w') as fh:
        json.dump(out, fh)
    os.replace(o['out'] + '.tmp', o['out'])
    for e in rules:
        print('replay: %s <- %s' % (' '.join(e['rel']), ', '.join(e['aliases'])))
    if not rules:
        print('replay: no file rule to replay; dune --force re-runs every named alias action')


# --- the replay ---------------------------------------------------------------

class Failed(Exception):
    pass


def accepts(pred, code):
    if pred is None:
        return code == 0
    if isinstance(pred, str):
        return pred.lstrip('-').isdigit() and int(pred) == code
    h, rest = pred[0], pred[1:]
    if h == 'or':
        return any(accepts(p, code) for p in rest)
    if h == 'and':
        return all(accepts(p, code) for p in rest)
    if h == 'not' and len(rest) == 1:
        return not accepts(rest[0], code)
    raise Failed('an exit-code predicate a replay cannot read: %r' % (pred,))


def remove(path):
    try:
        os.chmod(path, 0o644)
    except OSError:
        pass
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def interp(a, cwd, env, stdin, out, err, pred):
    # Returns the failing status, or 0.
    h = a[0]
    if h == 'progn':
        for x in a[1:]:
            st = interp(x, cwd, env, stdin, out, err, pred)
            if st:
                return st
        return 0
    if h == 'chdir':
        return interp(a[2], os.path.normpath(os.path.join(cwd, a[1])), env, stdin, out, err, pred)
    if h == 'setenv':
        env = dict(env)
        env[a[1]] = a[2]
        return interp(a[3], cwd, env, stdin, out, err, pred)
    if h in ('with-stdout-to', 'with-stderr-to', 'with-outputs-to'):
        path = os.path.join(cwd, a[1])
        remove(path)
        with open(path, 'wb', buffering=0) as fh:
            if h == 'with-stdout-to':
                return interp(a[2], cwd, env, stdin, fh, err, pred)
            if h == 'with-stderr-to':
                return interp(a[2], cwd, env, stdin, out, fh, pred)
            return interp(a[2], cwd, env, stdin, fh, fh, pred)
    if h == 'with-stdin-from':
        with open(os.path.join(cwd, a[1]), 'rb') as fh:
            return interp(a[2], cwd, env, fh, out, err, pred)
    if h in ('ignore-stdout', 'ignore-stderr', 'ignore-outputs'):
        with open(os.devnull, 'wb') as null:
            return interp(a[1], cwd, env, stdin,
                          out if h == 'ignore-stderr' else null,
                          err if h == 'ignore-stdout' else null, pred)
    if h == 'with-accepted-exit-codes':
        return interp(a[2], cwd, env, stdin, out, err, a[1])
    if h == 'echo':
        out.write(' '.join(a[1:]).encode('utf8', 'surrogateescape'))
        return 0
    if h == 'write-file':
        path = os.path.join(cwd, a[1])
        remove(path)
        with open(path, 'wb') as fh:
            fh.write(a[2].encode('utf8', 'surrogateescape'))
        return 0
    if h == 'copy':
        dst = os.path.join(cwd, a[2])
        remove(dst)
        shutil.copyfile(os.path.join(cwd, a[1]), dst)
        return 0
    if h == 'run':
        prog = a[1] if os.path.isabs(a[1]) else os.path.join(cwd, a[1])
        try:
            p = subprocess.run([prog] + a[2:], cwd=cwd, env=env, stdin=stdin, stdout=out, stderr=err)
            code = p.returncode if p.returncode >= 0 else 128 - p.returncode
        except OSError as e:
            err.write(('replay-rules: cannot run %s: %s\n' % (prog, e)).encode())
            code = 127
        if accepts(pred, code):
            return 0
        return code if code else 1
    raise Failed('unsupported form %s' % h)


# Read by perl, which leaves the environment it was given as it was: a python
# would add its own locale coercion (LC_CTYPE) to what it reports.
DUMP = 'print map { "$_=$ENV{$_}\\0" } keys %ENV'


def record_outputs(fh, rule):
    for path, rel in zip(rule['targets'], rule['rel']):
        try:
            data = open(path, 'rb').read()
        except OSError:
            fh.write(('== %s (missing)\n' % rel).encode())
            continue
        fh.write(('== %s (%d bytes)\n' % (rel, len(data))).encode())
        fh.write(data)
        if data and not data.endswith(b'\n'):
            fh.write(b'\n')


def run(argv):
    o = options(argv, {'--plan', '--out', '--parents'}, {'--snapshot'})
    if 'plan' not in o or 'out' not in o:
        die(2, 'run needs --plan and --out')
    try:
        p = json.load(open(o['plan']))
    except (OSError, ValueError) as e:
        die(126, 'cannot read the plan %s: %s' % (o['plan'], e))
    d = o['out']
    if o.get('snapshot'):
        with open(os.path.join(d, 'outputs'), 'wb') as fh:
            for rule in p['rules']:
                record_outputs(fh, rule)
        return 0
    if 'parents' in o:
        with open(o['parents'], 'a') as fh:
            fh.write('%d\n' % os.getpid())
    trace = os.path.join(d, 'exec.trace')
    q = subprocess.run(dune_cmd(p['dune']) + ['exec', '--no-build', '--build-dir=' + p['build_dir'],
                        '--trace-file=' + trace, '--', 'perl', '-e', DUMP],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL,
                       cwd=p['root'])
    try:
        os.unlink(trace)
    except OSError:
        pass
    base = {}
    for kv in q.stdout.decode('utf8', 'surrogateescape').split('\0'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            base[k] = v
    if q.returncode != 0 or 'PATH' not in base:
        sys.stderr.write(q.stderr.decode('utf8', 'replace')[-2000:])
        die(126, '`dune exec --no-build` did not give the action environment (exit %d)' % q.returncode)
    base.pop('DUNE_ACTION_TRACE_DIR', None)
    # The shell that starts dune names the program it runs in `_`, and dune passes that on to
    # every action; the dune `exec` above was started from python, which sets none.
    if '_' not in base:
        found = shutil.which(dune_cmd(p['dune'])[0], path=base.get('PATH'))
        if found:
            base['_'] = found
    systmp = base.get('TMPDIR') or tempfile.gettempdir()
    failed = 0
    with open(os.path.join(d, 'outputs'), 'wb') as outputs, \
            open(os.path.join(d, 'statuses'), 'w') as statuses, \
            open(os.path.join(d, 'replay-stdout'), 'wb') as rout, \
            open(os.path.join(d, 'replay-stderr'), 'wb') as rerr, \
            open(os.devnull, 'rb') as null:
        for rule in p['rules']:
            for t in rule['targets']:
                remove(t)
            env = dict(base)
            tmp = tempfile.mkdtemp(prefix='build_', suffix='_dune', dir=systmp)
            env['TMPDIR'] = tmp
            prev = base.get('BUILD_PATH_PREFIX_MAP')
            env['BUILD_PATH_PREFIX_MAP'] = (prev + ':' if prev else '') + '/workspace_root=' + rule['context']
            so = tempfile.TemporaryFile(buffering=0)
            se = tempfile.TemporaryFile(buffering=0)
            try:
                st = interp(rule['action'], p['root'], env, null, so, se, None)
            except (Failed, OSError) as e:
                se.write(('replay-rules: %s\n' % e).encode())
                st = 1
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            statuses.write('%s: %s\n' % (rule['rel'][0], 'ok' if not st else 'exit %d' % st))
            failed = failed or st
            for fh, sink in ((so, rout), (se, rerr)):
                fh.seek(0)
                data = fh.read()
                fh.close()
                if data:
                    sink.write(('== %s\n' % rule['rel'][0]).encode())
                    sink.write(data)
                    if not data.endswith(b'\n'):
                        sink.write(b'\n')
            record_outputs(outputs, rule)
    return 1 if failed else 0


if len(sys.argv) < 2:
    die(2, 'usage: replay-rules.sh plan ... | run ... (driven by tools/test-run.sh repeat --direct)')
sub, rest = sys.argv[1], sys.argv[2:]
if sub == 'plan':
    plan(rest)
    sys.exit(0)
if sub == 'run':
    sys.exit(run(rest))
die(2, 'unknown subcommand: %s' % sub)
PY

#!/usr/bin/env bash
# Evaluate ci.yml's actual matrix expressions and execute its commit guard.
# Requires only python3 and bash; run on the Ubuntu main leg before opam setup.
# ocannl-harness: standalone
# Usage: test/operations/ci_matrix.sh [--keep|--help]
set -eu
root=$(cd "$(dirname "$0")/../.." && pwd)
. "$root/scripts/harness-support.sh"
harness_args "$@"
harness_require python3
harness_scratch ci-matrix
rc=0
python3 - "$root/.github/workflows/ci.yml" "$TMP" "$("$root/tools/ci-shard.sh" aliases)" <<'PY' || rc=$?
import ast
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

source = Path(sys.argv[1]).read_text()


def field(text, name):
    match = re.search(r'^( +)' + re.escape(name) + r': >-\n', text, re.M)
    assert match, name
    indent = len(match[1])
    lines = text[match.end():].splitlines()
    content = list(itertools.takewhile(lambda line: len(line) - len(line.lstrip()) > indent, lines))
    return ' '.join(line.strip() for line in content)


def expression(value, event, windows=False):
    # This is the workflow's small expression vocabulary, not a second matrix.
    # Parse and whitelist the translated AST so a new operator fails loudly.
    value = value.removeprefix('${{').removesuffix('}}').strip()
    tokens = re.findall(r"'(?:[^']*)'|github\.event_name|inputs\.[a-z_]+|fromJSON|&&|\|\||!=|==|!|[(),]|\s+", value)
    assert ''.join(tokens) == value, value
    mapping = {'github.event_name': 'event', 'inputs.windows_only': 'windows',
               '&&': ' and ', '||': ' or ', '!': ' not '}
    translated = ''.join(mapping.get(token, token) for token in tokens).strip()
    tree = ast.parse(translated, mode='eval')
    permitted = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                 ast.Compare, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call)
    assert all(isinstance(node, permitted) for node in ast.walk(tree)), translated
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == 'fromJSON'
    return eval(compile(tree, '<workflow expression>', 'eval'), {'__builtins__': {}},
                dict(event=event, windows=windows, fromJSON=json.loads))


def documentation_triggers(text):
    # These are compiler inputs now. Ask the actual workflow's event blocks,
    # rather than maintain a second list of selected documentation paths.
    trigger = text.split('\non:\n', 1)[1].split('\nconcurrency:', 1)[0]
    for event in ('push', 'pull_request'):
        blocks = re.findall(r'^  ' + event + r': *\n(.*?)(?=^  [a-z_]+:|\Z)',
                            trigger, re.M | re.S)
        assert len(blocks) == 1, event + ' trigger absent, repeated or unsupported'
        # Fail on unsupported fields/quoting rather than infer that an unread
        # paths filter is absent. The current trigger dialect is deliberately small.
        branches = False
        for line in blocks[0].splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if event == 'push' and line == '    branches:':
                assert not branches, 'repeated push branch selection'
                branches = True
            elif event == 'push' and branches and re.fullmatch(r'      - .+', line):
                continue
            else:
                raise AssertionError(event + ' trigger has an unsupported field: ' + line)



def job_keys(text):
    # Every line at the job-key indentation is a job ID GitHub accepts or a
    # refusal: a key this reader skipped would be a job it never compared.
    section = text.split('\njobs:\n', 1)[1]
    jobs = []
    for line in section.splitlines():
        if not line.strip() or line.lstrip().startswith('#') or line.startswith('   '):
            continue
        assert line.startswith('  '), 'a top-level key after jobs: is unsupported here: ' + line
        job = re.fullmatch(r'  ([A-Za-z_][A-Za-z0-9_-]*):', line)
        assert job, 'unsupported job key syntax: ' + line
        jobs.append(job[1])
    return jobs


def matrix(text, event, windows=False):
    systems = expression(field(text, 'os'), event, windows)
    includes = expression(field(text, 'include'), event, windows)
    axes = []
    for name in ('ocaml-compiler', 'suite'):
        match = re.search(r'^        ' + name + r':\n((?:          - .+\n)+)', text, re.M)
        assert match, name
        axes.append([line.strip().removeprefix('- ') for line in match[1].splitlines()])
    jobs = [job + ('',) for job in itertools.product(systems, *axes)]
    jobs += [(entry['os'], entry['ocaml-compiler'], entry['suite'], entry.get('shard', ''))
             for entry in includes]
    # Every job but the matrix itself and the triage notifier (which runs
    # after all of them) is a side job the Windows fallback must skip: read
    # them from the workflow, so a new job that lacks the guard is refused.
    side = [job for job in job_keys(text) if job not in ('run', 'notify-triage-routine')]
    assert side, 'no side jobs found'
    selected = {}
    for job in side:
        guard = re.search(r'^  ' + re.escape(job) + r':\n    if: (.*)$', text, re.M)
        assert guard, job + ' selection missing'
        selected[job] = bool(expression(guard[1], event, windows))
    assert len(set(selected.values())) == 1, 'side jobs select differently: %s' % selected
    return unshard(sorted(jobs)), selected[side[0]]


def unshard(jobs):
    # ubuntu's 5.5 main suite as N shard jobs stands for ONE suite, but only if
    # its shards are exactly 1/N..N/N: a missing K is coverage lost without a red,
    # a duplicate is a job run twice. Any other job must not carry a shard.
    shards = [job[3] for job in jobs if job[:3] == ('ubuntu-latest', '5.5.x', 'main')]
    rest = [job for job in jobs if job[:3] != ('ubuntu-latest', '5.5.x', 'main')]
    assert all(job[3] == '' for job in rest), 'shard on an unsharded suite'
    if not shards:
        return [job[:3] for job in rest]
    counts = {shard.partition('/')[2] for shard in shards}
    assert len(counts) == 1 and counts.pop().isdigit(), 'shards disagree on N: %s' % shards
    n = int(shards[0].partition('/')[2])
    assert sorted(shards) == sorted('%d/%d' % (k, n) for k in range(1, n + 1)), \
        'shards are not exactly 1/%d..%d/%d: %s' % (n, n, n, shards)
    return sorted([job[:3] for job in rest] + [('ubuntu-latest', '5.5.x', 'main')])


normal = sorted([('ubuntu-latest', '5.5.x', 'main'),
                 ('macos-latest', '5.5.x', 'main'), ('macos-latest', '5.5.x', 'train')])
full = sorted(normal + [('windows-latest', '5.5.x', 'main'),
                        ('windows-latest', '5.5.x', 'train'), ('ubuntu-latest', '5.3.x', 'main')])
windows = [('windows-latest', '5.5.x', 'main'), ('windows-latest', '5.5.x', 'train')]


def suite_step(text):
    # The unsharded legs name the suite's aliases in ci.yml; the shards get
    # theirs from tools/ci-shard.sh. Drift between the two would let the shards
    # build a different suite than the job they replace, green either way.
    step = re.search(r"^      run: opam exec -- dune build \$\{\{ matrix.suite == 'train' "
                     r"&& '\"@train\"' \|\| '([^']*)' \}\}$", text, re.M)
    assert step, 'unsharded suite step missing'
    named = [token.strip('"').removeprefix('@') for token in step[1].split()]
    assert named == sys.argv[3].split(), 'ci.yml names %s, ci-shard.sh %s' % (named, sys.argv[3].split())


def controls(text):
    documentation_triggers(text)
    # Opt-in stays false in the actual dispatch schema, not just this evaluator.
    option = text.split('      windows_only:\n', 1)[1].split('      expected_sha:', 1)[0]
    assert 'default: false' in option, 'fallback must be opt-in'
    for option in (False, True):
        for event in ('pull_request', 'push'):
            assert matrix(text, event, option) == (normal, True), event
        assert matrix(text, 'schedule', option) == (full, True), 'scheduled full coverage'
    assert matrix(text, 'workflow_dispatch', False) == (normal, True), 'ordinary manual run'
    assert matrix(text, 'workflow_dispatch', True) == (windows, False), 'explicit Windows fallback'
    suite_step(text)


def extra_job(name, guard=''):
    block = ('  %s:\n%s    runs-on: ubuntu-latest\n    timeout-minutes: 5\n    steps:\n'
             '    - run: true\n\n' % (name, guard))
    return source.replace('  notify-triage-routine:\n', block + '  notify-triage-routine:\n', 1)


controls(source)
# The side-job list is read, not restated: a new job carrying the fallback
# guard joins the selection unremarked (its unguarded twin is a mutant below).
fallback_guard = re.search(r'^  fmt:\n(    if: .*\n)', source, re.M)[1]
controls(extra_job('new-side-job', fallback_guard))
print('PASS documentation compiler inputs are unfiltered on push and PR')
print('PASS normal, scheduled and explicit Windows fallback matrix selections')
print('PASS side jobs are read from ci.yml: a new guarded job joins the selection')
print('PASS ubuntu main shards are exactly 1/N..N/N, over the aliases ci-shard.sh shards')
second = '{"os": "ubuntu-latest", "ocaml-compiler": "5.5.x", "suite": "main", "shard": "2/2"},'
for label, mutant in (
    ('fallback enabled by default', source.replace('default: false', 'default: true')),
    ('push skips documentation inputs', source.replace('  push:\n', '  push:\n    paths-ignore:\n      - "docs/**"\n', 1)),
    ('PR skips documentation inputs', source.replace('  pull_request:\n', '  pull_request:\n    paths-ignore:\n      - "docs/**"\n', 1)),
    ('quoted PR documentation filter', source.replace('  pull_request:\n', '  pull_request:\n    "paths-ignore": ["docs/**"]\n', 1)),
    ('repeated PR trigger hides documentation filter', source.replace('  pull_request:\n', '  pull_request:\n  pull_request:\n    paths-ignore: ["docs/**"]\n', 1)),
    ('automatic Windows jobs', source.replace("github.event_name == 'schedule'", "github.event_name != 'workflow_dispatch'")),
    ('schedule loses coverage', source.replace("github.event_name == 'schedule'", "github.event_name == 'never'")),
    ('schedule narrowed by dispatch input', source.replace("github.event_name == 'workflow_dispatch' && inputs.windows_only", 'inputs.windows_only')),
    ('duplicate formatting job', source.replace("github.event_name != 'workflow_dispatch' || !inputs.windows_only", "github.event_name != 'never'", 1)),
    ('harnesses in the Windows fallback', source.replace("  harnesses:\n    if: github.event_name != 'workflow_dispatch' || !inputs.windows_only", "  harnesses:\n    if: github.event_name != 'never'")),
    ('Dune-floor promotion harnesses in the Windows fallback', source.replace("  promotion-floor:\n    if: github.event_name != 'workflow_dispatch' || !inputs.windows_only", "  promotion-floor:\n    if: github.event_name != 'never'")),
    ('CPU-torch benchmark tests in the Windows fallback', source.replace("  torch-runner:\n    if: github.event_name != 'workflow_dispatch' || !inputs.windows_only", "  torch-runner:\n    if: github.event_name != 'never'")),
    ('a new side job without the Windows-fallback guard', extra_job('new-side-job')),
    ('per-PR shard dropped', source[:source.rindex(second)] + source[source.rindex(second) + len(second):]),
    ('scheduled shard dropped', source.replace(second, '', 1)),
    ('shard renumbered', source.replace('"shard": "2/2"', '"shard": "2/3"', 1)),
    ('floor job sharded', source.replace('"ocaml-compiler": "5.3.x", "suite": "main"}', '"ocaml-compiler": "5.3.x", "suite": "main", "shard": "1/1"}')),
    ('suite alias drift', source.replace('"@default" "@runtest" "@bin-smoke"', '"@default" "@runtest"')),
):
    assert mutant != source, label
    try:
        controls(mutant)
    except AssertionError:
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)


# Run the exact bash guard from the workflow with git reporting a fixture HEAD
# and gh answering the compare API for the one direction the guard must ask:
# is the intended commit behind the dispatched head?
step = source.split('    - name: Verify dispatch commit\n', 1)[1].split('    - uses: ', 1)[0]
assert "if: github.event_name == 'workflow_dispatch'" in step
script = step.split('      run: |\n', 1)[1]
script = '\n'.join(line[8:] for line in script.splitlines())
sha, other = 'a' * 40, 'b' * 40
master, branch = 'refs/heads/master', 'refs/heads/topic'
scratch = sys.argv[2]
git = Path(scratch) / 'git'
git.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$CHECKOUT_SHA"\n')
git.chmod(0o755)
gh = Path(scratch) / 'gh'
gh.write_text('#!/usr/bin/env bash\n'
              '[ "$1 $2" = "api repos/$GITHUB_REPOSITORY/compare/$EXPECTED_SHA...$GITHUB_SHA" ] || exit 1\n'
              '[ -n "$RELATION" ] || exit 1\n'
              'printf "%s\\n" "$RELATION"\n')
gh.chmod(0o755)

def guard(code, expected, run, checkout, only=True, ref=master, relation='ahead'):
    env = dict(os.environ, PATH=scratch + os.pathsep + os.environ['PATH'],
               EXPECTED_SHA=expected, GITHUB_SHA=run, CHECKOUT_SHA=checkout,
               WINDOWS_ONLY=str(only).lower(), GITHUB_REF=ref,
               GITHUB_REPOSITORY='owner/repo', RELATION=relation)
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', code], env=env,
                            capture_output=True, text=True)
    return result.returncode

assert guard(script, sha, sha, sha) == 0
assert guard(script, sha, sha, sha, ref=branch) == 0  # a branch's own head
assert guard(script, '', sha, sha, False) == 0  # ordinary manual dispatch
assert guard(script, sha, other, sha, False) == 0  # bisection probe: an ancestor of master's head
print('PASS accepts the head, an ordinary dispatch, and an ancestor of master')
for label, expected, run, checkout, ref, relation in (
    ('missing intended SHA', '', sha, sha, master, 'ahead'),
    ('malformed SHA', 'abc', sha, sha, master, 'ahead'),
    ('wrong checkout', sha, sha, other, master, 'ahead'),
    ('obsolete branch head', sha, other, sha, branch, 'ahead'),
    ('commit not behind master', sha, other, sha, master, 'diverged'),
    ('commit ahead of master', sha, other, sha, master, 'behind'),
    ('unread ancestry', sha, other, sha, master, ''),
):
    assert guard(script, expected, run, checkout, ref=ref, relation=relation) == 1, label
    print('PASS rejects', label)
for label, check, args in (
    ('master-only ancestry', '[ "$GITHUB_REF" = refs/heads/master ] &&', dict(ref=branch)),
    ('ancestry', '[ "$relation" = ahead ]', dict(relation='diverged')),
):
    assert check in script, label
    mutant = script.replace(check, 'true &&' if check.endswith('&&') else 'true')
    assert guard(mutant, sha, other, sha, **args) == 0, label  # proves the refusal owns this case
    print('PASS negative control exposes removed', label)
check = '[ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]'
assert check in script
assert guard(script.replace(check, 'true'), sha, sha, other) == 0, 'checkout identity'
print('PASS negative control exposes removed checkout identity')

# A bisection probe judges the commit it names only if EVERY job builds that
# commit: a job checking out the event's commit would test master's head and
# report it under the probe's name. And a probe's red must not fire triage.
checkouts = re.findall(r'^    - uses: actions/checkout@v7\n(      with:\n        ref: \$\{\{ inputs\.expected_sha \}\}\n)?',
                       source, re.M)
assert len(checkouts) >= 3 and all(checkouts), 'a job checks out the event commit, not the dispatched one'
notify = re.search(r'^  notify-triage-routine:\n(?:    .*\n)*?    if: >-\n((?:      .*\n)+)', source, re.M)
assert notify and "!(github.event_name == 'workflow_dispatch' && inputs.expected_sha)" in notify[1], \
    'a pinned dispatch fires the triage routine'
print('PASS every job builds the dispatched commit, and a pinned dispatch fires no triage')
# A job missing from the triage job's `needs` can go red on master without
# firing it: `failure()` there reads only the jobs it waits for. Derive the
# job list from the workflow rather than keep a second copy of it here.
def triage_waits_for_every_job(text):
    jobs = job_keys(text)
    needs = re.search(r'^  notify-triage-routine:\n(?:    .*\n)*?    needs: \[([^]]*)\]\n', text, re.M)
    assert needs, 'triage job has no inline needs list'
    others = sorted(job for job in jobs if job != 'notify-triage-routine')
    assert len(others) >= 2, 'found too few jobs: %s' % jobs
    assert sorted(name.strip() for name in needs[1].split(',')) == others, \
        'triage needs [%s], the other jobs are %s' % (needs[1], others)


triage_waits_for_every_job(source)
last = re.search(r'^    needs: \[.*(, [A-Za-z0-9_-]+)\]$', source, re.M)
assert last, 'triage needs list unreadable'
for label, mutant, needle in (
    ('a job dropped from the triage needs', source.replace(last[0], last[0].replace(last[1], ''), 1),
     last[1].lstrip(', ')),
    ('an underscore-named job the triage omits', extra_job('extra_job'), 'extra_job'),
    ('a quoted job key', extra_job('"extra job"'), 'unsupported job key syntax'),
):
    assert mutant != source, label
    try:
        triage_waits_for_every_job(mutant)
    except AssertionError as e:
        assert needle in str(e), '%s: rejected for another reason: %s' % (label, e)
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)
print('PASS the triage job waits for every other job')
PY
report "$rc" "ci matrix controls"
finish

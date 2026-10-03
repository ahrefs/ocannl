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
    selected = []
    for job in ('fmt', 'harnesses'):
        guard = re.search(r'^  ' + job + r':\n    if: (.*)$', text, re.M)
        assert guard, job + ' selection missing'
        selected.append(bool(expression(guard[1], event, windows)))
    assert selected[0] == selected[1], 'formatting and harnesses select differently'
    return unshard(sorted(jobs)), selected[0]


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


controls(source)
print('PASS normal, scheduled and explicit Windows fallback matrix selections')
print('PASS ubuntu main shards are exactly 1/N..N/N, over the aliases ci-shard.sh shards')
second = '{"os": "ubuntu-latest", "ocaml-compiler": "5.5.x", "suite": "main", "shard": "2/2"},'
for label, mutant in (
    ('fallback enabled by default', source.replace('default: false', 'default: true')),
    ('automatic Windows jobs', source.replace("github.event_name == 'schedule'", "github.event_name != 'workflow_dispatch'")),
    ('schedule loses coverage', source.replace("github.event_name == 'schedule'", "github.event_name == 'never'")),
    ('schedule narrowed by dispatch input', source.replace("github.event_name == 'workflow_dispatch' && inputs.windows_only", 'inputs.windows_only')),
    ('duplicate formatting job', source.replace("github.event_name != 'workflow_dispatch' || !inputs.windows_only", "github.event_name != 'never'", 1)),
    ('harnesses in the Windows fallback', source.replace("  harnesses:\n    if: github.event_name != 'workflow_dispatch' || !inputs.windows_only", "  harnesses:\n    if: github.event_name != 'never'")),
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
PY
report "$rc" "ci matrix controls"
finish

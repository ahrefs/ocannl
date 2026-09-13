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
python3 - "$root/.github/workflows/ci.yml" "$TMP" <<'PY' || rc=$?
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


def expression(value, event, windows=False, extended=False):
    # This is the workflow's small expression vocabulary, not a second matrix.
    # Parse and whitelist the translated AST so a new operator fails loudly.
    value = value.removeprefix('${{').removesuffix('}}').strip()
    tokens = re.findall(r"'(?:[^']*)'|github\.event_name|inputs\.[a-z_]+|fromJSON|&&|\|\||!=|==|!|[(),]|\s+", value)
    assert ''.join(tokens) == value, value
    mapping = {'github.event_name': 'event', 'inputs.windows_only': 'windows',
               'inputs.extended': 'extended', '&&': ' and ', '||': ' or ', '!': ' not '}
    translated = ''.join(mapping.get(token, token) for token in tokens).strip()
    tree = ast.parse(translated, mode='eval')
    permitted = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                 ast.Compare, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call)
    assert all(isinstance(node, permitted) for node in ast.walk(tree)), translated
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id == 'fromJSON'
    return eval(compile(tree, '<workflow expression>', 'eval'), {'__builtins__': {}},
                dict(event=event, windows=windows, extended=extended, fromJSON=json.loads))


def matrix(text, event, windows=False, extended=False):
    systems = expression(field(text, 'os'), event, windows, extended)
    includes = expression(field(text, 'include'), event, windows, extended)
    axes = []
    for name in ('ocaml-compiler', 'suite'):
        match = re.search(r'^        ' + name + r':\n((?:          - .+\n)+)', text, re.M)
        assert match, name
        axes.append([line.strip().removeprefix('- ') for line in match[1].splitlines()])
    jobs = list(itertools.product(systems, *axes))
    jobs += [(entry['os'], entry['ocaml-compiler'], entry['suite']) for entry in includes]
    fmt = re.search(r'^  fmt:\n    if: (.*)$', text, re.M)
    assert fmt, 'Formatting selection missing'
    return sorted(jobs), bool(expression(fmt[1], event, windows, extended))


normal = sorted([('ubuntu-latest', '5.5.x', 'main'),
                 ('macos-latest', '5.5.x', 'main'), ('macos-latest', '5.5.x', 'train')])
full = sorted(normal + [('windows-latest', '5.5.x', 'main'),
                        ('windows-latest', '5.5.x', 'train'), ('ubuntu-latest', '5.3.x', 'main')])
windows = [('windows-latest', '5.5.x', 'main'), ('windows-latest', '5.5.x', 'train')]


def controls(text):
    for event in ('pull_request', 'push'):
        assert matrix(text, event) == (normal, True), event
    for option in (False, True):
        assert matrix(text, 'schedule', option) == (full, True), 'scheduled full coverage'
        assert matrix(text, 'workflow_dispatch', True, option) == (windows, False), 'Windows only'
    assert matrix(text, 'workflow_dispatch', False, False) == (normal, True)
    assert matrix(text, 'workflow_dispatch', False, True) == (full, True)


controls(source)
print('PASS normal, scheduled, extended and Windows-only matrix selections')
for label, mutant in (
    ('duplicate platform jobs', source.replace("github.event_name == 'workflow_dispatch' && inputs.windows_only", 'inputs.extended')),
    ('schedule loses extended coverage', source.replace("github.event_name == 'schedule' || inputs.extended", 'inputs.extended')),
    ('schedule narrowed by dispatch input', source.replace("github.event_name == 'workflow_dispatch' && inputs.windows_only", 'inputs.windows_only')),
    ('duplicate formatting job', source.replace("github.event_name != 'workflow_dispatch' || !inputs.windows_only", "github.event_name != 'never'")),
):
    try:
        controls(mutant)
    except AssertionError:
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)

# Run the exact bash guard from the workflow with git reporting a fixture HEAD.
step = source.split('    - name: Verify dispatch commit\n', 1)[1].split('    # Hermetic', 1)[0]
assert "if: github.event_name == 'workflow_dispatch'" in step
script = step.split('      run: |\n', 1)[1]
script = '\n'.join(line[8:] for line in script.splitlines())
sha, other = 'a' * 40, 'b' * 40
scratch = sys.argv[2]
git = Path(scratch) / 'git'
git.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$CHECKOUT_SHA"\n')
git.chmod(0o755)

def guard(code, expected, run, checkout, only=True):
    env = dict(os.environ, PATH=scratch + os.pathsep + os.environ['PATH'],
               EXPECTED_SHA=expected, GITHUB_SHA=run, CHECKOUT_SHA=checkout,
               WINDOWS_ONLY=str(only).lower())
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', code], env=env,
                            capture_output=True, text=True)
    return result.returncode

assert guard(script, sha, sha, sha) == 0
assert guard(script, '', sha, sha, False) == 0  # existing extended dispatch
for label, expected, run, checkout in (
    ('missing intended SHA', '', sha, sha), ('malformed SHA', 'abc', sha, sha),
    ('obsolete run head', sha, other, other), ('wrong checkout', sha, sha, other),
):
    assert guard(script, expected, run, checkout) == 1, label
    print('PASS rejects', label)
for label, check, args in (
    ('run identity', '[ "$GITHUB_SHA" = "$EXPECTED_SHA" ] &&', (sha, other, sha)),
    ('checkout identity', '[ "$(git rev-parse HEAD)" = "$EXPECTED_SHA" ]', (sha, sha, other)),
):
    assert check in script
    mutant = script.replace(check, 'true &&' if check.endswith('&&') else 'true')
    assert guard(mutant, *args) == 0, label  # proves the refusal owns this case
    print('PASS negative control exposes removed', label)
PY
report "$rc" "ci matrix controls"
finish

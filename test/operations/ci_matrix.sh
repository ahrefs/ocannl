#!/usr/bin/env bash
# Evaluate ci.yml's actual matrix expressions and execute its commit guard, and
# read both workflows' triggers: the full matrix per PR and never on a master
# push, ci-env.yml on both (gh-ocannl-1816).
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
import time

source = Path(sys.argv[1]).read_text()


def field(text, name):
    match = re.search(r'^( +)' + re.escape(name) + r': >-\n', text, re.M)
    assert match, name
    indent = len(match[1])
    lines = text[match.end():].splitlines()
    content = list(itertools.takewhile(lambda line: len(line) - len(line.lstrip()) > indent, lines))
    return ' '.join(line.strip() for line in content)


def gh_format(template, *args):
    # GitHub's format(): {N} is the Nth argument, {{ and }} are literal braces.
    return re.sub(r'\{\{|\}\}|\{(\d+)\}',
                  lambda m: m[0][0] if m[1] is None else str(args[int(m[1])]), template)


def expression(value, event, windows=False, compiler=''):
    # This is the workflow's small expression vocabulary, not a second matrix.
    # Parse and whitelist the translated AST so a new operator fails loudly.
    value = value.removeprefix('${{').removesuffix('}}').strip()
    tokens = re.findall(r"'(?:[^']*)'|github\.event_name|inputs\.[a-z_]+|fromJSON|format|&&|\|\||!=|==|!|[(),]|\s+", value)
    assert ''.join(tokens) == value, value
    mapping = {'github.event_name': 'event', 'inputs.windows_only': 'windows',
               'inputs.ocaml_compiler': 'compiler',
               '&&': ' and ', '||': ' or ', '!': ' not '}
    translated = ''.join(mapping.get(token, token) for token in tokens).strip()
    tree = ast.parse(translated, mode='eval')
    permitted = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not,
                 ast.Compare, ast.Eq, ast.NotEq, ast.Name, ast.Load, ast.Constant, ast.Call)
    assert all(isinstance(node, permitted) for node in ast.walk(tree)), translated
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            assert isinstance(node.func, ast.Name) and node.func.id in ('fromJSON', 'format')
    return eval(compile(tree, '<workflow expression>', 'eval'), {'__builtins__': {}},
                dict(event=event, windows=windows, compiler=compiler,
                     fromJSON=json.loads, format=gh_format))


def unfiltered_triggers(text, events, absent=()):
    # Documentation is compiler input now. Ask the actual workflow's event
    # blocks, rather than maintain a second list of selected documentation paths.
    trigger = text.split('\non:\n', 1)[1].split('\nconcurrency:', 1)[0]
    for event in absent:
        assert not re.search(r'^  ' + event + r':', trigger, re.M), event + ' trigger present'
    for event in events:
        blocks = re.findall(r'^  ' + event + r': *\n(.*?)(?=^  [a-z_]+:|\Z)',
                            trigger, re.M | re.S)
        assert len(blocks) == 1, event + ' trigger absent, repeated or unsupported'
        # Fail on unsupported fields/quoting rather than infer that an unread
        # paths filter is absent. The current trigger dialect is deliberately small.
        branches = None
        for line in blocks[0].splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            if event == 'push' and line == '    branches:':
                assert branches is None, 'repeated push branch selection'
                branches = []
            elif event == 'push' and branches is not None and re.fullmatch(r'      - .+', line):
                branches.append(line.strip().removeprefix('- '))
            else:
                raise AssertionError(event + ' trigger has an unsupported field: ' + line)
        assert event != 'push' or branches == ['master'], 'push selects %s, not master' % branches


def documentation_triggers(text):
    # The full matrix gates PRs and never runs on a master push: a merge's
    # verdict is ci-env.yml's (gh-ocannl-1816).
    unfiltered_triggers(text, ('pull_request',), absent=('push',))



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


def raw_jobs(text, event, windows=False, compiler=''):
    systems = expression(field(text, 'os'), event, windows, compiler)
    compilers = expression(field(text, 'ocaml-compiler'), event, windows, compiler)
    includes = expression(field(text, 'include'), event, windows, compiler)
    match = re.search(r'^        suite:\n((?:          - .+\n)+)', text, re.M)
    assert match, 'suite'
    suites = [line.strip().removeprefix('- ') for line in match[1].splitlines()]
    jobs = [job + ('',) for job in itertools.product(systems, compilers, suites)]
    jobs += [(entry['os'], entry['ocaml-compiler'], entry['suite'], entry.get('shard', ''))
             for entry in includes]
    return jobs


def matrix(text, event, windows=False, compiler=''):
    jobs = raw_jobs(text, event, windows, compiler)
    # Every job but the matrix itself and the triage notifier (which runs
    # after all of them) is a side job the Windows fallback must skip: read
    # them from the workflow, so a new job that lacks the guard is refused.
    side = [job for job in job_keys(text) if job not in ('run', 'notify-triage-routine')]
    assert side, 'no side jobs found'
    selected, scheduled = {}, {}
    for job in side:
        guard = re.search(r'^  ' + re.escape(job) + r':\n    if: (.*)$', text, re.M)
        assert guard, job + ' selection missing'
        # A guard opening with the schedule names a SCHEDULED side job, too
        # costly per PR (gh-ocannl-1367): it must select exactly where the
        # ordinary side jobs do AND the event is the schedule or a dispatch.
        if guard[1].startswith(SCHEDULED_GUARD):
            scheduled[job] = bool(expression(guard[1], event, windows, compiler))
        else:
            selected[job] = bool(expression(guard[1], event, windows, compiler))
    assert selected, 'no ordinary side jobs found'
    assert len(set(selected.values())) == 1, 'side jobs select differently: %s' % selected
    ordinary = next(iter(selected.values()))
    want = ordinary and event in ('schedule', 'workflow_dispatch')
    assert all(value == want for value in scheduled.values()), \
        'scheduled side jobs select differently from the ordinary ones on %s: %s, ordinary %s' % (
            event, scheduled, ordinary)
    return unshard(sorted(jobs))[0], ordinary


SCHEDULED_GUARD = "github.event_name == 'schedule' || "


# The suites ubuntu's 5.5 jobs may shard: tools/ci-shard.sh knows how to cut
# exactly these, and a shard of anything else would be a job it cannot name.
SHARDABLE = {('ubuntu-latest', '5.5.x', 'main'), ('ubuntu-latest', '5.5.x', 'train')}


def unshard(jobs):
    # A suite as N shard jobs stands for ONE job, but only if its shards are
    # exactly 1/N..N/N: a missing K is coverage lost without a red, a duplicate
    # is a job run twice, and an unsharded twin beside them is the suite twice.
    # Returns the collapsed jobs and each sharded suite's N.
    groups = {}
    for job in jobs:
        groups.setdefault(job[:3], []).append(job[3])
    collapsed, counts = [], {}
    for group, shards in groups.items():
        collapsed.append(group)
        if shards == ['']:
            continue
        assert group in SHARDABLE, 'shard on a suite ci-shard.sh does not cut: %s' % (group,)
        assert '' not in shards, 'unsharded job beside the shards of %s' % (group,)
        ns = {shard.partition('/')[2] for shard in shards}
        assert len(ns) == 1 and ns.pop().isdigit(), 'shards disagree on N: %s' % shards
        n = int(shards[0].partition('/')[2])
        assert sorted(shards) == sorted('%d/%d' % (k, n) for k in range(1, n + 1)), \
            'shards are not exactly 1/%d..%d/%d: %s' % (n, n, n, shards)
        counts[group] = n
    return sorted(collapsed), counts


# Per PR: ubuntu carries both suites, macOS only `main` -- ONE
# job on the scarce runners (gh-ocannl-1422). The schedule adds macOS `train`.
normal = sorted([('ubuntu-latest', '5.5.x', 'main'), ('ubuntu-latest', '5.5.x', 'train'),
                 ('macos-latest', '5.5.x', 'main')])
full = sorted(normal + [('macos-latest', '5.5.x', 'train'), ('windows-latest', '5.5.x', 'main'),
                        ('windows-latest', '5.5.x', 'train'), ('ubuntu-latest', '5.3.x', 'main')])
windows = [('windows-latest', '5.5.x', 'main'), ('windows-latest', '5.5.x', 'train')]


def build_steps(text):
    # The steps of the `run` job, each as its text from `    - ` on.
    job = text.split('\n  run:\n', 1)[1].split('\n  fmt:\n', 1)[0]
    return ['    - ' + step for step in job.split('\n    - ')[1:]]


def suite_step(text):
    # The unsharded legs name the suite's aliases in ci.yml; the shards get
    # theirs from tools/ci-shard.sh. Drift between the two would let the shards
    # build a different suite than the job they replace, green either way.
    step = re.search(r"^      SUITE_TARGETS: \$\{\{ matrix.suite == 'train' "
                     r"&& '@train' \|\| '([^']*)' \}\}$", text, re.M)
    assert step, 'unsharded suite targets missing'
    named = [token.removeprefix('@') for token in step[1].split()]
    assert named == sys.argv[3].split(), 'ci.yml names %s, ci-shard.sh %s' % (named, sys.argv[3].split())
    steps = build_steps(text)
    # Both unsharded steps build those targets: bash word-splits them, and
    # PowerShell must split them into an array (gh-ocannl-1304's PowerShell
    # step), since a literal unquoted `@word` there is an empty splat.
    assert any('opam exec -- dune build $SUITE_TARGETS 2>&1 | tee -a "$RUNNER_TEMP/dune-build.log"' in step
               for step in steps), 'bash unsharded step does not build SUITE_TARGETS'
    windows = [step for step in steps if "@($env:SUITE_TARGETS -split ' ')" in step]
    assert len(windows) == 1, 'PowerShell unsharded step does not build SUITE_TARGETS'
    # Its pipe into Tee-Object ends in the cmdlet's status, not dune's.
    assert re.search(r'^        exit \$LASTEXITCODE$', windows[0], re.M), \
        "PowerShell build step does not exit with dune's status"
    # A bash step piping a build into tee reports tee's status unless the
    # shell runs with pipefail, which only an explicit `shell: bash` gives
    # (the implicit default is `bash -e`): a red build would read green.
    for step in steps:
        if re.search(r'\| tee\b', step):
            assert '\n      shell: bash\n' in step, 'a step pipes into tee without pipefail: ' + step.splitlines()[0]
    # The sharded step hands the script the job's suite: without it every
    # `train` shard would build a share of `main` instead, green either way.
    shard = text.split("    - if: ${{ matrix.shard }}\n", 1)
    assert len(shard) == 2, 'sharded suite step missing'
    shard = shard[1].split('\n    - ', 1)[0]
    assert '        SUITE: ${{ matrix.suite }}\n' in shard, 'sharded step does not read matrix.suite'
    assert re.search(r'^        targets=\$\(opam exec -- tools/ci-shard\.sh targets "\$SHARD" "\$SUITE"\)$',
                     shard, re.M), 'sharded step does not pass its suite to ci-shard.sh'


def compiler_input(text):
    # The dispatch schema's `ocaml_compiler` choice, read from ci.yml: its
    # options and its default. Only the shape written there is accepted.
    block = re.search(r'^      ocaml_compiler:\n((?:        .*\n)+)', text, re.M)
    assert block, 'ocaml_compiler dispatch input missing'
    body = block[1]
    assert re.search(r'^        type: choice$', body, re.M), 'ocaml_compiler must be a choice'
    options = re.search(r'^        options:\n((?:          - .+\n)+)', body, re.M)
    default = re.findall(r'^        default: (.+)$', body, re.M)
    assert options and len(default) == 1, 'ocaml_compiler options/default unreadable'
    options = [line.strip().removeprefix('- ') for line in options[1].splitlines()]
    assert len(set(options)) == len(options) and default[0] in options, options
    return options, default[0]


def controls(text):
    documentation_triggers(text)
    # Opt-in stays false in the actual dispatch schema, not just this evaluator.
    option = text.split('      windows_only:\n', 1)[1].split('      expected_sha:', 1)[0]
    assert 'default: false' in option, 'fallback must be opt-in'
    compilers, ordinary = compiler_input(text)
    # Inputs are empty on every other event; no value of either may narrow them.
    for option in (False, True):
        for compiler in [''] + compilers:
            assert matrix(text, 'pull_request', option, compiler) == (normal, True), 'pull_request'
            assert matrix(text, 'schedule', option, compiler) == (full, True), 'scheduled full coverage'
    assert matrix(text, 'workflow_dispatch', False, ordinary) == (normal, True), 'ordinary manual run'
    # The input's default IS the compiler of the ordinary matrix, read from it.
    assert {job[1] for job in normal} == {ordinary}, 'default compiler is not the ordinary matrix'
    for compiler in compilers:
        assert matrix(text, 'workflow_dispatch', True, compiler) == (windows, False), \
            'explicit Windows fallback'
        if compiler != ordinary:
            assert matrix(text, 'workflow_dispatch', False, compiler) == \
                ([('ubuntu-latest', compiler, 'main')], False), 'compiler dispatch ' + compiler
    # Every compiler the schedule alone builds can be dispatched, and its
    # dispatch runs exactly the scheduled job (gh-ocannl-1267).
    floors = sorted({job[1] for job in full} - {ordinary})
    assert floors, 'the schedule builds no compiler beside the ordinary one'
    for compiler in floors:
        assert compiler in compilers, 'scheduled compiler %s cannot be dispatched' % compiler
        assert matrix(text, 'workflow_dispatch', False, compiler)[0] == \
            [job for job in full if job[1] == compiler], 'compiler dispatch is not the scheduled job'
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
# A scheduled side job: the schedule, or a dispatch the fallback guard admits.
scheduled_guard = re.search(r'^  harness-legs-alone:\n(    if: .*\n)', source, re.M)[1]
assert scheduled_guard.startswith('    if: ' + SCHEDULED_GUARD), 'scheduled side-job guard moved'
controls(extra_job('new-scheduled-job', scheduled_guard))
print('PASS ci.yml runs per PR, unfiltered, and has no push trigger')
print('PASS normal, scheduled and explicit Windows fallback matrix selections')
print('PASS every ocaml_compiler option: the default is the ordinary matrix, another is one ubuntu job')
print('PASS every scheduled-only compiler dispatches as exactly its scheduled job')
print('PASS side jobs are read from ci.yml: a new guarded job joins the selection')
print('PASS a scheduled side job runs on the schedule and an ordinary dispatch, never per PR or in the fallback')
print('PASS ubuntu main and train shards are exactly 1/N..N/N, over the aliases ci-shard.sh shards')
print('PASS per PR macOS runs one job, main; the schedule adds its train job')
print('PASS every build step piping into tee has pipefail; the PowerShell one exits with dune\'s status')
second = '{"os": "ubuntu-latest", "ocaml-compiler": "5.5.x", "suite": "main", "shard": "2/2"},'
train_second = '{"os": "ubuntu-latest", "ocaml-compiler": "5.5.x", "suite": "train", "shard": "2/2"},'
train_last = train_second.rstrip(',') + "]'"
assert source.count(train_second) == 1 and source.count(',\n               ' + train_last) == 1, \
    'train 2/2 entries moved; update the mutants below'
side_guard = "github.event_name != 'workflow_dispatch' || !inputs.windows_only && inputs.ocaml_compiler == '5.5.x'"
assert source.count(side_guard) == 5, 'side-job guard text moved; update the mutants below'
compiler_axis = "&& format('[\"{0}\"]', inputs.ocaml_compiler)"
assert compiler_axis in source, 'compiler axis text moved; update the mutants below'
for label, mutant in (
    ('fallback enabled by default', source.replace('default: false', 'default: true')),
    ('full matrix back on master push', source.replace('\non:\n', '\non:\n  push:\n    branches:\n      - master\n', 1)),
    ('PR skips documentation inputs', source.replace('  pull_request:\n', '  pull_request:\n    paths-ignore:\n      - "docs/**"\n', 1)),
    ('quoted PR documentation filter', source.replace('  pull_request:\n', '  pull_request:\n    "paths-ignore": ["docs/**"]\n', 1)),
    ('repeated PR trigger hides documentation filter', source.replace('  pull_request:\n', '  pull_request:\n  pull_request:\n    paths-ignore: ["docs/**"]\n', 1)),
    ('automatic Windows jobs', source.replace("github.event_name == 'schedule'", "github.event_name != 'workflow_dispatch'")),
    ('schedule loses coverage', source.replace("github.event_name == 'schedule'", "github.event_name == 'never'")),
    ('schedule narrowed by dispatch input', source.replace("github.event_name == 'workflow_dispatch' && inputs.windows_only", 'inputs.windows_only')),
    ('schedule narrowed by compiler input', source.replace("github.event_name == 'workflow_dispatch' && inputs.ocaml_compiler", 'inputs.ocaml_compiler')),
    ('duplicate formatting job', source.replace(side_guard, "github.event_name != 'never'", 1)),
    ('harnesses in the Windows fallback', source.replace("  harnesses:\n    if: " + side_guard, "  harnesses:\n    if: github.event_name != 'never'")),
    ('real-dune shell harnesses in the Windows fallback', source.replace("  harness-real-dune:\n    if: " + side_guard, "  harness-real-dune:\n    if: github.event_name != 'never'")),
    ('Dune-floor promotion harnesses in the Windows fallback', source.replace("  promotion-floor:\n    if: " + side_guard, "  promotion-floor:\n    if: github.event_name != 'never'")),
    ('CPU-torch benchmark tests in the Windows fallback', source.replace("  torch-runner:\n    if: " + side_guard, "  torch-runner:\n    if: github.event_name != 'never'")),
    ('formatting in the compiler dispatch', source.replace(side_guard, "github.event_name != 'workflow_dispatch' || !inputs.windows_only", 1)),
    ('compiler dispatch builds the ordinary compiler', source.replace(compiler_axis, "&& '[\"5.5.x\"]'")),
    ('compiler dispatch keeps macOS', source.replace("&& '[\"ubuntu-latest\"]'", "&& '[\"ubuntu-latest\", \"macos-latest\"]'")),
    ('compiler dispatch keeps the 5.5 shards and train', source.replace("inputs.ocaml_compiler != '5.5.x'\n          && '[]'", "inputs.ocaml_compiler == 'never'\n          && '[]'")),
    ('compiler dispatch combined with the Windows fallback', source.replace("&& !inputs.windows_only\n", "\n", 1)),
    ('scheduled floor not dispatchable', source.replace('          - 5.5.x\n          - 5.3.x\n', '          - 5.5.x\n', 1)),
    ('compiler input defaults to the floor', source.replace('          - 5.3.x\n        default: 5.5.x\n', '          - 5.3.x\n        default: 5.3.x\n', 1)),
    ('compiler input as free text', source.replace('        type: choice\n', '        type: string\n', 1)),
    ('a new side job without the Windows-fallback guard', extra_job('new-side-job')),
    ('a scheduled side job in the Windows fallback',
     extra_job('new-scheduled-job', "    if: github.event_name == 'schedule' || github.event_name == 'workflow_dispatch'\n")),
    ('a scheduled side job per PR',
     extra_job('new-scheduled-job', "    if: github.event_name == 'schedule' || github.event_name != 'never'\n")),
    ('a scheduled side job off the schedule',
     extra_job('new-scheduled-job', "    if: github.event_name == 'schedule' || github.event_name == 'never'\n")),
    ('per-PR shard dropped', source[:source.rindex(second)] + source[source.rindex(second) + len(second):]),
    ('scheduled shard dropped', source.replace(second, '', 1)),
    ('shard renumbered', source.replace('"shard": "2/2"', '"shard": "2/3"', 1)),
    ('floor job sharded', source.replace('"ocaml-compiler": "5.3.x", "suite": "main"}', '"ocaml-compiler": "5.3.x", "suite": "main", "shard": "1/1"}')),
    ('suite alias drift', source.replace("'@default @runtest @bin-smoke'", "'@default @runtest'")),
    ('unsharded bash build without pipefail',
     source.replace("    - if: ${{ !matrix.shard && runner.os != 'Windows' }}\n      shell: bash\n",
                    "    - if: ${{ !matrix.shard && runner.os != 'Windows' }}\n", 1)),
    ('sharded build without pipefail',
     source.replace('    - if: ${{ matrix.shard }}\n      shell: bash\n', '    - if: ${{ matrix.shard }}\n', 1)),
    ('PowerShell build exits with the pipe status', source.replace('        exit $LASTEXITCODE\n', '', 1)),
    ('PowerShell build names its aliases as literals',
     source.replace("@($env:SUITE_TARGETS -split ' ')", '@default @runtest @bin-smoke', 1)),
    ('per-PR train shard dropped', source.replace(',\n               ' + train_last, "]'", 1)),
    ('scheduled train shard dropped', source.replace(train_second, '', 1)),
    ('train back on macOS per PR', source[:source.rindex(train_last)] + train_second
     + ' {"os": "macos-latest", "ocaml-compiler": "5.5.x", "suite": "train"}]\'' + source[source.rindex(train_last) + len(train_last):]),
    ('scheduled macOS train dropped', source.replace('               {"os": "macos-latest", "ocaml-compiler": "5.5.x", "suite": "train"},\n', '', 1)),
    ('an unsharded ubuntu train beside its shards', source.replace(train_second, train_second + ' {"os": "ubuntu-latest", "ocaml-compiler": "5.5.x", "suite": "train"},', 1)),
    ('a shard of a suite ci-shard.sh does not cut', source.replace('"suite": "train"}]\'', '"suite": "train", "shard": "1/1"}]\'', 1)),
    ('sharded step drops the suite', source.replace('targets "$SHARD" "$SUITE")', 'targets "$SHARD")')),
    ('sharded step reads no suite', source.replace('        SUITE: ${{ matrix.suite }}\n', '')),
):
    assert mutant != source, label
    try:
        controls(mutant)
    except AssertionError:
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)


# ci-env.yml, master's per-merge verdict (gh-ocannl-1816): on every master push
# and every PR, unfiltered, never cancelling a push run.
env_source = (Path(sys.argv[1]).parent / 'ci-env.yml').read_text()


def env_controls(text):
    unfiltered_triggers(text, ('push', 'pull_request'))
    concurrency = text.split('\nconcurrency:\n', 1)[1].split('\njobs:', 1)[0] + '\n'
    assert "  cancel-in-progress: ${{ github.event_name == 'pull_request' }}\n" in concurrency, \
        'ci-env.yml cancels more than superseded PR runs'


env_controls(env_source)
print('PASS ci-env.yml runs on master push and per PR, unfiltered, cancelling only superseded PR runs')
for label, mutant in (
    ('ci-env.yml without push', env_source.replace('  push:\n    branches:\n      - master\n', '', 1)),
    ('ci-env.yml push off master', env_source.replace('      - master\n', '      - main\n', 1)),
    ('ci-env.yml PR skips documentation', env_source.replace('  pull_request:\n', '  pull_request:\n    paths-ignore:\n      - "docs/**"\n', 1)),
    ('ci-env.yml cancels push runs', env_source.replace("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", 'cancel-in-progress: true', 1)),
):
    assert mutant != env_source, label
    try:
        env_controls(mutant)
    except AssertionError:
        print('PASS rejected mutant:', label)
    else:
        raise AssertionError('accepted mutant: ' + label)


# The `train` shards together build exactly the tier `@train` builds. The tier
# is read here on its own terms -- each tracked dune file's top-level `train`
# aggregate, its `(alias train-<name>)` deps -- a reader deliberately simpler
# than ci-shard.sh's, so a disagreement between the two is a red, not a shared
# blind spot. ci-shard.sh's train mode runs no dune, so this needs none.
root = Path(sys.argv[1]).resolve().parents[2]
listed = subprocess.run(['git', '-C', str(root), 'ls-files', '--', 'dune', '*/dune'],
                        capture_output=True, text=True, check=True).stdout.split()
tier = set()
for path in listed:
    prefix = '' if '/' not in path else path.rsplit('/', 1)[0] + '/'
    code = re.sub(r';[^\n]*', '', (root / path).read_text())
    for stanza in re.split(r'\n(?=\()', '\n' + code):
        if re.match(r'\(alias\s+\(name\s+train\)', stanza):
            tier |= {'@@' + prefix + name for name in re.findall(r'\(alias\s+(train-[^\s()]+)\s*\)', stanza)}
assert len(tier) >= 2, 'no train tier found: %s' % sorted(tier)


def train_shards(n):
    out = []
    for k in range(1, n + 1):
        run = subprocess.run([str(root / 'tools/ci-shard.sh'), 'targets', '%d/%d' % (k, n), 'train'],
                             capture_output=True, text=True)
        assert run.returncode == 0, 'ci-shard.sh targets %d/%d train: %s' % (k, n, run.stderr)
        out.append(run.stdout.split())
    return out


def partitions(shards, whole):
    # Every shard nonempty, none sharing a target, together exactly WHOLE.
    flat = [t for shard in shards for t in shard]
    return all(shards) and len(flat) == len(set(flat)) and set(flat) == whole


group = ('ubuntu-latest', '5.5.x', 'train')
for event in ('pull_request', 'schedule'):
    counts = unshard(sorted(raw_jobs(source, event)))[1]
    assert group in counts, '%s runs no ubuntu train shards' % event
    shards = train_shards(counts[group])
    assert partitions(shards, tier), '%s: the %d train shards %s are not a partition of %s' % (
        event, counts[group], shards, sorted(tier))
    # The oracle discriminates: a lost shard, a member it does not carry.
    assert not partitions(shards[:-1], tier) and not partitions(shards, tier | {'@@x/train-ghost'})
print('PASS the train shards partition the tier the train aggregates define, every shard nonempty')

# The main shards' weigh step prints what `tools/ci-shard.sh merge-weights` reads back from a run's
# log (gh-ocannl-1464): run the step's own script for each shard over a fixture trace, beside the
# real tools, and merge the two jobs' output into the WEIGHTS block it stands for.
weigh_step = source.split("    - name: Weigh the shard's units\n", 1)
assert len(weigh_step) == 2, 'the main shards have no weigh step'
weigh_step = weigh_step[1].split('\n    - ', 1)[0]
assert "      if: ${{ matrix.shard && matrix.suite == 'main' && !cancelled() }}\n" in weigh_step, \
    'the weigh step does not run on every main shard that built'
assert '      shell: bash\n' in weigh_step and '        SHARD: ${{ matrix.shard }}\n' in weigh_step
# The block scalar ends at the first line indented less than its body (the next step's comments).
weigh_script = '\n'.join(line[8:] for line in itertools.takewhile(
    lambda line: line.startswith('        ') or not line.strip(),
    weigh_step.split('      run: |\n', 1)[1].splitlines()))


def csexp(x):
    if isinstance(x, list):
        return b'(' + b''.join(csexp(e) for e in x) + b')'
    x = str(x).encode()
    return str(len(x)).encode() + b':' + x


def weigh_job(script, shard, units):
    # The step's stdout for SHARD, its trace holding UNITS: (directory, program, CPU seconds).
    work = Path(sys.argv[2]) / ('weigh-' + shard.replace('/', '-'))
    (work / '_build').mkdir(parents=True, exist_ok=True)
    if not (work / 'tools').exists():
        (work / 'tools').symlink_to(root / 'tools')
    (work / '_build' / 'trace.csexp').write_bytes(b''.join(
        csexp(['process', 'finish', ['1', str(int(s * 1e9))], ['prog', prog], ['dir', '/w/_build/default/' + d],
               ['rusage', [['user_cpu_time', str(int(s * 1e9))], ['system_cpu_time', '0']]]])
        for d, prog, s in units))
    run = subprocess.run(['bash', '-eo', 'pipefail', '-c', script], cwd=work, capture_output=True,
                         text=True, env=dict(os.environ, SHARD=shard))
    assert run.returncode == 0, 'weigh step for %s: %s' % (shard, run.stderr)
    return run.stdout


def merged(script):
    log = (weigh_job(script, '1/2', [('test/operations', 'heavy.exe', 100), ('lib', 'gen.exe', 300)])
           + weigh_job(script, '2/2', [('test/operations', 'light.exe', 50), ('lib', 'gen.exe', 20)]))
    run = subprocess.run([str(root / 'tools/ci-shard.sh'), 'merge-weights'], input=log,
                         capture_output=True, text=True)
    return run.returncode, run.stdout


assert merged(weigh_script) == (0, '@rest 280\nheavy 100\nlight 50\n'), merged(weigh_script)
marker = 'sed "s|^|ci-shard-weight $SHARD |"'
assert weigh_script.count(marker) == 1, 'the weigh step marks its lines another way'
for label, mutant in (('unmarked lines', weigh_script.replace(marker, 'cat')),
                      ('lines without their shard', weigh_script.replace('$SHARD |', '|'))):
    assert merged(mutant)[0] == 2, 'accepted mutant: ' + label
    print('PASS rejected mutant: weigh step with', label)
print("PASS the weigh step's output, merged over one run's main shards, is the WEIGHTS block")

# Run the exact bash guard from the workflow with git reporting a fixture HEAD
# and gh answering the compare API for the one direction the guard must ask:
# is the intended commit behind the dispatched head?
# The step ends at the next line indented as a step or a comment between steps.
step = re.split(r'\n    (?=\S)', source.split('    - name: Verify dispatch commit\n', 1)[1], maxsplit=1)[0]
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

compilers, ordinary = compiler_input(source)
floor = [compiler for compiler in compilers if compiler != ordinary][0]
# The guard's literal for the ordinary compiler is the schema's default.
assert '[ "$OCAML_COMPILER" != %s ]' % ordinary in script, 'guard names another ordinary compiler'

def guard(code, expected, run, checkout, only=True, ref=master, relation='ahead', compiler=ordinary):
    env = dict(os.environ, PATH=scratch + os.pathsep + os.environ['PATH'],
               EXPECTED_SHA=expected, GITHUB_SHA=run, CHECKOUT_SHA=checkout,
               WINDOWS_ONLY=str(only).lower(), GITHUB_REF=ref,
               GITHUB_REPOSITORY='owner/repo', RELATION=relation, OCAML_COMPILER=compiler)
    result = subprocess.run(['bash', '-eo', 'pipefail', '-c', code], env=env,
                            capture_output=True, text=True)
    return result.returncode

assert guard(script, sha, sha, sha) == 0
assert guard(script, sha, sha, sha, ref=branch) == 0  # a branch's own head
assert guard(script, '', sha, sha, False) == 0  # ordinary manual dispatch
assert guard(script, sha, other, sha, False) == 0  # bisection probe: an ancestor of master's head
assert guard(script, sha, sha, sha, False, ref=branch, compiler=floor) == 0  # floor at a PR head
assert guard(script, sha, other, sha, False, compiler=floor) == 0  # floor probe inside a master span
print('PASS accepts the head, an ordinary dispatch, an ancestor of master, and a floor dispatch')
for label, only, expected in (
    ('floor dispatch without an intended SHA', False, ''),
    ('floor dispatch combined with the Windows fallback', True, sha),
):
    assert guard(script, expected, sha, sha, only, ref=branch, compiler=floor) == 1, label
    print('PASS rejects', label)
check = '[ "$OCAML_COMPILER" != %s ]' % ordinary
assert guard(script.replace(check, 'false'), '', sha, sha, False, ref=branch, compiler=floor) == 0
assert guard(script.replace(check, 'false'), sha, sha, sha, True, ref=branch, compiler=floor) == 0
print('PASS negative control exposes removed floor-dispatch refusals')
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


# The disk watchdog (gh-ocannl-1771): run the start step's and the report step's own scripts with
# stub df/sudo/pkill and the polls shortened, the free space moved under the watchdog's feet.
def step_script(name):
    step = source.split('    - name: %s\n' % name, 1)
    assert len(step) == 2, 'step missing: ' + name
    step = step[1].split('\n    - ', 1)[0]
    assert "runner.os == 'Linux'" in step.split('      run: |\n', 1)[0], name + ' is not Linux-only'
    return '\n'.join(line[8:] for line in itertools.takewhile(
        lambda line: line.startswith('        ') or not line.strip(),
        step.split('      run: |\n', 1)[1].splitlines()))


watch_start = step_script('Free disk space and start the disk watchdog (Linux)')
watch_report = step_script('Report the disk headroom (Linux)')
assert 'always()' in source.split('    - name: Report the disk headroom (Linux)\n', 1)[1].split('\n', 1)[0], \
    'the headroom report does not run on a cancelled or red job'
state_dir = '/dev/shm/ocannl-disk-watch'
timing = 'floor_mb=1024 poll=5 grace=15 min= tripped='
assert watch_start.count(state_dir) == 1 and watch_report.count(state_dir) == 1, 'watchdog state moved'
assert watch_start.count(timing) == 1, 'watchdog timings moved'
watch_root = Path(sys.argv[2]) / 'disk-watch'
watch_bin = watch_root / 'bin'
watch_bin.mkdir(parents=True)
for name, body in (
        # df: one line per path argument, every filesystem with $DF_AVAIL MB free; each call logged.
        ('df', 'echo "$*" >> "$WATCH_LOG/df"\n'
               'echo "Filesystem 1M-blocks Used Available Capacity Mounted on"\n'
               'for a in "$@"; do case $a in -*) ;; *) echo "stub 99999 1 $(cat "$DF_AVAIL") 1% $a" ;; esac; done\n'),
        # sudo: the cleanup, held until the test releases it, then freeing space (atomically, as below).
        ('sudo', 'echo "$*" >> "$WATCH_LOG/sudo"\n'
                 'while [ ! -e "$WATCH_LOG/release" ]; do sleep 0.02; done\n'
                 'echo 50000 > "$DF_AVAIL.new" && mv "$DF_AVAIL.new" "$DF_AVAIL"\n'),
        ('pkill', 'echo "$*" >> "$WATCH_LOG/pkill"\n')):
    (watch_bin / name).write_text('#!/usr/bin/env bash\n' + body)
    (watch_bin / name).chmod(0o755)


def watch_run(start, report):
    # (tripped before the floor, report exit and output while healthy, report exit and output after
    # the trip, pkill's calls, sudo's calls); the watchdog is killed however this ends.
    run = watch_root / ('run%d' % len(list(watch_root.glob('run*'))))
    state, log = run / 'state', run / 'log'
    log.mkdir(parents=True)
    avail = run / 'avail'
    avail.write_text('20000\n')
    env = dict(os.environ, PATH=str(watch_bin) + os.pathsep + os.environ['PATH'], DF_AVAIL=str(avail),
               WATCH_LOG=str(log), GITHUB_WORKSPACE=str(run))
    start = start.replace(state_dir, str(state)).replace(timing, 'floor_mb=1024 poll=0.05 grace=0 min= tripped=')
    report = report.replace(state_dir, str(state))

    def wait_for(path):
        for _ in range(100):
            if path.exists() and path.read_text().strip():
                return True
            time.sleep(0.05)
        return False

    def report_run():
        r = subprocess.run(['bash', '-e', '-c', report], env=env, capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout

    # A new session: its group is the watchdog's too, killed at the end. The step must return at
    # once, its output closed to the watchdog (on GitHub a held pipe keeps the step running).
    proc = subprocess.Popen(['bash', '-e', '-c', start], env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=5)
        assert proc.returncode == 0, 'watchdog start step: ' + out
        # Polls while the cleanup still runs (the step's own two df calls, then the watchdog's), and
        # only then the cleanup ends: a low counted before its end reads 20000.
        for _ in range(100):
            if (log / 'df').exists() and len((log / 'df').read_text().splitlines()) >= 4:
                break
            time.sleep(0.05)
        (log / 'release').touch()
        wait_for(state / 'min')
        healthy = report_run()
        early = (state / 'tripped').exists()
        # Atomically: a df reading the file mid-truncation would report 1 MB free.
        (run / 'avail.new').write_text('600\n')
        os.replace(run / 'avail.new', avail)
        wait_for(state / 'tripped')
        time.sleep(0.3)
        after = report_run()
        read = lambda path: path.read_text() if path.exists() else ''
        return early, healthy, after, read(log / 'pkill'), read(log / 'sudo'), read(state / 'min')
    finally:
        try:
            os.killpg(proc.pid, 9)
        except ProcessLookupError:
            pass


def watch_ok(start, report):
    early, healthy, after, pkill, sudo, low = watch_run(start, report)
    return (not early and healthy[0] == 0 and 'watchdog saw after the cleanup: 50000 MB at ' in healthy[1]
            and '; before the cleanup: 20000 MB, after it: 50000 MB\n' in healthy[1]
            and '::error' not in healthy[1] and after[0] == 1
            and '::error title=disk space::Free disk space fell to 600 MB' in after[1]
            and pkill.startswith('-TERM -x dune\n-KILL -x dune\n-TERM -x dune\n') and low.startswith('600 MB at ')
            and sudo.startswith('rm -rf /usr/local/lib/android '))


assert watch_ok(watch_start, watch_report), watch_run(watch_start, watch_report)
print('PASS the disk watchdog stops dune under its floor and the report names disk space, red')
for label, start, report in (
        ('a floor never reached', watch_start.replace('-lt "$floor_mb"', '-lt 0'), watch_report),
        ('a low counted before the cleanup ends', watch_start.replace('[ -n "$cleaned" ] && ', ''), watch_report),
        ('a watchdog that stops dune only once',
         watch_start.replace('pkill -KILL -x dune || true\n', 'pkill -KILL -x dune || true\n  return 0\n'),
         watch_report),
        ('a watchdog that only asks dune to stop', watch_start.replace('pkill -KILL -x dune', 'true'), watch_report),
        ('a watchdog holding the step\'s output', watch_start.replace('disk_watch </dev/null >/dev/null 2>&1 &', 'disk_watch &'),
         watch_report),
        ('a report that stays green on a trip', watch_start, watch_report.replace('  exit 1\n', '', 1)),
        ('no cleanup', watch_start.replace('sudo rm -rf', 'true'), watch_report)):
    assert (start, report) != (watch_start, watch_report), label
    try:
        ok = watch_ok(start, report)
    except (AssertionError, subprocess.TimeoutExpired):
        ok = False
    assert not ok, 'accepted mutant: ' + label
    print('PASS rejected mutant: disk watchdog with', label)
PY
report "$rc" "ci matrix controls"
finish

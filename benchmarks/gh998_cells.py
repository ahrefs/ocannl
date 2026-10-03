#!/usr/bin/env python3
"""Ablate simplify_llc float algebra on byte-identical benchmark fixtures (#998).

prepare --out DIR: bind the revision, executable bytes and fixture digests, run the existing
Torch CPU exact oracle. Build bench_mlp.exe and the selected fixture runners beforehand, outside a timing hold.
dry --out DIR --backends metal,cc: arms-on/off self-test twice per backend (discard timings).
The self-test uses the same compilation, execution and emission path with a short protocol.
run --out DIR --backends metal,cc: four reverse-paired rounds of all, none and each family off.
summarize --out DIR: print the per-workload envelopes against arms-on and the Torch oracle.
--workloads selects a comma-separated subset (use the same selection for prepare/dry/run).
--rounds selects the repeat count for run (default 4, even); summarize reads the recorded matrix.
Each pair contributes the geometric mean of its two off/on p50 ratios; the table
reports the median and range of these pair means. An independent all-control treatment
must emit byte-identical sources; its raw ratio spread is reported beside each workload.
The preflight records the host and CPU; cc measurements on different CPUs are separate rows.

Each cell uses the suite's unchanged f32 protocol, untuned default schedule and fixed compiler
math settings. A family's removal may expose another simplifier arm or be undone by backend
compilation; ratios measure its marginal end-to-end effect under those settings, not a strict
IEEE promise. A null/non-finite loss or incomplete trajectory refuses an envelope. Full stdout,
stderr, flags and emitted sources are retained per cell. No builds/searches/setup run in `run`.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import shutil
from pathlib import Path
import statistics
import subprocess
import time

import fixture_digest
import cell_group
import bench_venv

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / 'benchmarks'
FAMILIES = ['contract', 'constants', 'sub', 'mul_div', 'pow', 'identities']
WORKLOADS = ['lenet', 'gpt2_mini', 'gpt2_mini_train']
TREATMENTS = {'all': 'all', 'all-control': 'all', 'none': 'none', **{
    'no-' + arm: ','.join(a for a in FAMILIES if a != arm) for arm in FAMILIES}}


CELL_TIMEOUT_S = 1800
DEADLINE = None
_cancellation = cell_group.CancellationDeferral('gh998')


def run_child(argv, **kwargs):
    # The matrix deadline is a wait timeout, so it cannot interrupt spawn or cleanup.
    # The shared deferral handles operator cancellation in those ownership windows.
    with _cancellation.deferring():
        proc = cell_group.spawn(argv, cwd=HERE, **kwargs)
        try:
            timeout = CELL_TIMEOUT_S if DEADLINE is None else min(
                CELL_TIMEOUT_S, max(0., DEADLINE - time.monotonic()))
            with _cancellation.cancellable():
                status = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            status = 'timeout'
        finally:
            cleanup = cell_group.terminate(proc, grace=1.)
            if cleanup.observation is not cell_group.GONE or not cleanup.reaped:
                raise cell_group.CleanupFailed(
                    f'gh998: child group {proc.pid} cleanup {cleanup.observation.value}; '
                    'clear survivors before retrying the matrix')
    return status


def visit_cells(backends, workloads, repeat):
    treatments = list(TREATMENTS)
    offset = (repeat // 2) % len(treatments)
    treatments = treatments[offset:] + treatments[:offset]
    blocks = [(b, w) for b in backends for w in workloads]
    if repeat % 2:
        treatments.reverse()
        blocks.reverse()
    return [(b, w, t) for b, w in blocks for t in treatments]


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def executable(workload):
    return ROOT / '_build/default/benchmarks/runners/ocannl' / (
        'bench_conv.exe' if workload == 'lenet' else 'bench_gpt.exe')


def host_identity():
    if platform.system() == 'Darwin':
        cpu = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    else:
        cpu = next((line.split(':', 1)[1].strip() for line in
                    Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), 'unknown')
    return dict(host=platform.node(), system=platform.platform(), machine=platform.machine(), cpu=cpu)


def identity(workloads):
    entries = fixture_digest.read_digests(HERE / 'fixtures/DIGESTS.txt')
    fixtures = {}
    for workload in workloads:
        status, sha, size, origins = fixture_digest.status(
            HERE / 'fixtures' / (workload + '.safetensors'), entries)
        if status != 'MATCH':
            raise RuntimeError(f'{workload}: fixture {status}')
        fixtures[workload] = dict(sha256=sha, size=size, origins=origins)
    return dict(revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        diff_sha256=hashlib.sha256(subprocess.check_output(
            ['git', 'diff', 'HEAD'], cwd=ROOT)).hexdigest(),
        host=host_identity(), fixtures=fixtures, executables={w: digest(executable(w)) for w in workloads},
        smoke_executable=digest(ROOT / '_build/default/benchmarks/runners/ocannl/bench_mlp.exe'))


def clean_env():
    return {k: v for k, v in os.environ.items() if not k.startswith(
        ('OCANNL_', 'BENCH_', 'OMP_', 'GOMP_', 'KMP_'))}


def result(path):
    rows = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and 'losses' in row:
            rows.append(row)
    if len(rows) != 1:
        raise RuntimeError(f'{path}: expected one result line, got {len(rows)}')
    return rows[0]


def cell(out, name, argv, env):
    base = out / name
    base.mkdir()  # never overwrite evidence
    (base / 'command.json').write_text(json.dumps(argv, indent=2) + '\n')
    print(f'cell {name}', flush=True)
    with (base / 'stdout').open('w') as stdout, (base / 'stderr').open('w') as stderr:
        status = run_child(argv, env=env, stdout=stdout, stderr=stderr)
    (base / 'exit.json').write_text(json.dumps(status) + '\n')
    if status != 0:
        raise RuntimeError(f'{name}: exit {status}; see {base}')
    row = result(base / 'stdout')
    (base / 'result.json').write_text(json.dumps(row, indent=2, allow_nan=False) + '\n')
    return row


def ocannl(out, backend, workload, treatment, repeat, dry=False):
    name = f'{"dry-" if dry else ""}{backend}-{workload}-{treatment}-{repeat}'
    env = clean_env()
    env.update(BENCH_TUNE='0', BENCH_MATERIALIZE='0', BENCH_DOMINANT_KERNEL='0')
    if not dry:
        env['BENCH_FIXTURE'] = str(HERE / 'fixtures' / (workload + '.safetensors'))
    prefix = 'gh998-' + hashlib.sha256(str(out).encode()).hexdigest()[:12] + '-' + name
    artifacts = HERE / 'build_files' / prefix
    if artifacts.exists():
        raise RuntimeError(f'stale artifact directory: {artifacts}')
    argv = [str(executable(workload)), f'--ocannl_backend={backend}',
            '--ocannl_default_prec=single', '--ocannl_schedule_fission=true',
            '--ocannl_automatic_gpu_schedule=true', '--ocannl_autotune_search=false',
            '--ocannl_autotune_cache_dir=', '--ocannl_debug_log_from_routines=false',
            '--ocannl_cc_backend_fast_math=false', '--ocannl_cc_backend_fp_contract=auto',
            '--ocannl_online_softmax=false', '--ocannl_online_softmax_backward=false',
            '--ocannl_output_debug_files_in_build_directory=true',
            '--ocannl_clean_up_build_files_on_startup=false',
            f'--ocannl_build_files_prefix={prefix}',
            f'--ocannl_simplify_fp_algebra={TREATMENTS[treatment]}']
    if dry:
        argv[0] = str(ROOT / '_build/default/benchmarks/runners/ocannl/bench_mlp.exe')
        argv.append('--self-test')
    try:
        row = cell(out, name, argv, env)
    finally:
        if artifacts.exists():
            shutil.move(str(artifacts), str(out / name / 'artifacts'))
    (out / name / 'sources.json').write_text(json.dumps(source_manifest(out / name), indent=2) + '\n')
    verify_control(out, backend, workload, repeat, dry=dry)
    expected_workload = 'selftest-tiny' if dry else workload
    if row['backend'] != backend or row['workload'] != expected_workload or row['searched']:
        raise RuntimeError(f'{name}: wrong backend/workload or a searching process')
    if row.get('simplify_fp_algebra') != dict(value=TREATMENTS[treatment], source='commandline'):
        raise RuntimeError(f'{name}: result does not confirm the selected float algebra')
    if dry:
        envelope(row['losses'], row['losses'])
    else:
        oracle = result(out / f'torch-{workload}' / 'stdout')['losses']
        envelope(row['losses'], oracle)
    return row



def source_manifest(base):
    artifacts = base / 'artifacts'
    sources = {str(p.relative_to(artifacts)): digest(p)
               for p in sorted(artifacts.rglob('*')) if p.suffix in {'.c', '.cu', '.metal'}}
    if not sources:
        raise RuntimeError(f'{base}: no emitted backend source to verify')
    return sources


def verify_control(out, backend, workload, repeat, dry=False):
    stem = ('dry-' if dry else '') + f'{backend}-{workload}-'
    paths = [out / f'{stem}{t}-{repeat}' / 'sources.json' for t in ['all', 'all-control']]
    if all(p.exists() for p in paths):
        if json.loads(paths[0].read_text()) != json.loads(paths[1].read_text()):
            raise RuntimeError(f'{stem}{repeat}: all-control sources differ from all')


def envelope(got, ref):
    if len(got) != len(ref) or not got:
        raise RuntimeError('empty or incomplete parity trajectory')
    if not all(isinstance(v, (float, int)) and math.isfinite(v) for v in got + ref):
        raise RuntimeError('non-finite trajectory: no finite envelope')
    return (max(abs(a-b) for a, b in zip(got, ref)),
            max(abs(a-b)/max(abs(b), 1e-12) for a, b in zip(got, ref)),
            sum(a != b for a, b in zip(got, ref)))


def summarize(out):
    matrix = json.loads((out / 'matrix.json').read_text())
    preflight = json.loads((out / 'preflight.json').read_text())
    if matrix['rounds'] < 2 or matrix['rounds'] % 2:
        raise RuntimeError('summary requires complete forward/reverse round pairs')
    print(f"Host: {preflight['host']['host']}; CPU: {preflight['host']['cpu']}; revision: {preflight['revision']}; rounds: {matrix['rounds']}; reverse pairs: {matrix['rounds'] // 2}.")
    for workload in matrix['workloads']:
        fixture = preflight['fixtures'][workload]
        print(f"Fixture {workload}: SHA-256 {fixture['sha256']}; {fixture['size']} bytes; recorded origins: {fixture['origins']}.")
    print()
    for backend in matrix['backends']:
        for workload in matrix['workloads']:
            controls = []
            for repeat in range(matrix['rounds']):
                verify_control(out, backend, workload, repeat)
                control = out / f'{backend}-{workload}-all-control-{repeat}'
                baseline = out / f'{backend}-{workload}-all-{repeat}'
                # Re-read source bytes, so the drift claim cannot be satisfied by stale manifests.
                if source_manifest(control) != source_manifest(baseline):
                    raise RuntimeError(f'{control}: drift control is not byte-identical')
                controls.append(json.loads((control / 'result.json').read_text())['step_ms']['p50'] /
                                json.loads((baseline / 'result.json').read_text())['step_ms']['p50'])
            print(f'Drift control {backend}/{workload}: verified byte-identical sources; '
                  f'raw off/on p50 range {min(controls):.4f}–{max(controls):.4f}.')
    print()
    print('| backend | workload | arm | source matches on | off/on paired p50 ratio median (range) | max abs / rel vs on | changed losses | max abs / rel vs Torch |')
    print('|---|---|---|---|---|---|---|---|')
    for backend in matrix['backends']:
        for workload in matrix['workloads']:
            oracle = result(out / f'torch-{workload}' / 'stdout')['losses']
            for treatment in TREATMENTS:
                sources_equal = True
                ratios, vs_on, vs_torch = [], [], []
                for repeat in range(matrix['rounds']):
                    row = json.loads((out / f'{backend}-{workload}-{treatment}-{repeat}' / 'result.json').read_text())
                    on = json.loads((out / f'{backend}-{workload}-all-{repeat}' / 'result.json').read_text())
                    sources_equal &= source_manifest(out / f'{backend}-{workload}-{treatment}-{repeat}') == source_manifest(out / f'{backend}-{workload}-all-{repeat}')
                    ratios.append(row['step_ms']['p50'] / on['step_ms']['p50'])
                    vs_on.append(envelope(row['losses'], on['losses']))
                    vs_torch.append(envelope(row['losses'], oracle))
                paired = [math.sqrt(a * b) for a, b in zip(ratios[::2], ratios[1::2])]
                print(f'| {backend} | {workload} | {treatment} | {sources_equal} | {statistics.median(paired):.4f} ({min(paired):.4f}–{max(paired):.4f}) | '
                      f'{max(v[0] for v in vs_on):.3g} / {max(v[1] for v in vs_on):.3g} | '
                      f'{max(v[2] for v in vs_on)} | {max(v[0] for v in vs_torch):.3g} / {max(v[1] for v in vs_torch):.3g} |')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('phase', choices=['prepare', 'dry', 'run', 'summarize'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--backends', default='metal,cc')
    ap.add_argument('--workloads', default=','.join(WORKLOADS))
    ap.add_argument('--rounds', type=int, default=4)
    ap.add_argument('--deadline-seconds', type=int, default=7200)
    args = ap.parse_args()
    if args.rounds < 2 or args.rounds % 2:
        ap.error('rounds must be even and at least 2')
    if args.deadline_seconds <= 0:
        ap.error('deadline must be positive')
    global DEADLINE
    DEADLINE = time.monotonic() + args.deadline_seconds
    _cancellation.install()
    out = args.out.resolve()
    workloads = args.workloads.split(',')
    if not workloads or len(set(workloads)) != len(workloads) or any(w not in WORKLOADS for w in workloads):
        ap.error('workloads must be a nonempty distinct list from ' + ','.join(WORKLOADS))
    backends = args.backends.split(',')
    if len(set(backends)) != len(backends) or any(b not in ['metal', 'cc', 'cuda'] for b in backends):
        ap.error('backends must be metal,cc,cuda')
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == 'summarize':
        summarize(out)
        return
    current = identity(workloads)
    preflight = out / 'preflight.json'
    if args.phase == 'prepare':
        if preflight.exists():
            raise RuntimeError('prepare requires a fresh output directory')
        preflight.write_text(json.dumps(current, indent=2) + '\n')
        python = str(bench_venv.venv_python(HERE))
        for workload in workloads:
            cell(out, f'torch-{workload}', [python, str(HERE / 'runners/pytorch/run.py'),
                 '--fixture', str(HERE / 'fixtures' / (workload + '.safetensors')),
                 '--device', 'cpu', '--regime', 'exact'], clean_env())
        return
    if json.loads(preflight.read_text()) != current:
        raise RuntimeError('revision, diff, binary or fixture changed since prepare')
    if args.phase == 'dry':
        for backend in backends:
            for treatment in ['all', 'all-control', 'none']:
                for repeat in range(2):
                    ocannl(out, backend, 'selftest', treatment, repeat, dry=True)
        (out / 'dry-ok.json').write_text(json.dumps(backends) + '\n')
        return
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, text=True).strip():
        raise RuntimeError('timing requires a clean tree')
    if not set(backends) <= set(json.loads((out / 'dry-ok.json').read_text())):
        raise RuntimeError('backend has not passed dry run')
    matrix = out / 'matrix.json'
    if matrix.exists():
        raise RuntimeError('timing requires a fresh matrix; existing evidence is never overwritten')
    visits = [visit_cells(backends, workloads, r) for r in range(args.rounds)]
    matrix.write_text(json.dumps(dict(backends=backends, workloads=workloads,
                                     rounds=args.rounds, visit_order=visits), indent=2) + '\n')
    for repeat, cells in enumerate(visits):
        for backend, workload, treatment in cells:
            ocannl(out, backend, workload, treatment, repeat)


if __name__ == '__main__':
    main()

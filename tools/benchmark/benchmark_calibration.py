#!/usr/bin/env python3
"""Run calibration commands and collect paired runtime and reconstruction metrics."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def field(value, key):
    for part in key.split('.'):
        value = value[part]
    return value


def verify_source_lock(spec):
    """Reject a changed package before launching a frozen experiment command."""
    lock = spec.get('source_lock')
    if lock is None:
        return
    path = Path(lock['path'])
    if digest(path) != lock['sha256']:
        raise ValueError(f'Source lock changed: {path}')
    payload = json.loads(path.read_text())
    root = Path(payload['root'])
    actual = {str(p.relative_to(root)): digest(p) for p in (root / 'track4d_calib').rglob('*.py')}
    if actual != payload['files']:
        raise ValueError(f'Calibration package differs from source lock: {root}')


def run_step(spec, directory, name):
    """Measure the whole subprocess, or import an explicitly hashed measurement."""
    record = Path(spec['result'])
    verify_source_lock(spec)
    if 'command' in spec:
        if record.exists():
            raise ValueError(f'Refusing to overwrite existing result: {record}')
        command = spec['command']
        if not isinstance(command, list) or not command or not all(isinstance(x, str) for x in command):
            raise ValueError('Commands must be nonempty argument lists')
        started = time.perf_counter()
        with (directory / f'{name}.log').open('x') as log:
            process = subprocess.run(command, cwd=spec.get('cwd'),
                                     env=dict(os.environ, **spec.get('env', {})),
                                     stdout=log, stderr=subprocess.STDOUT)
        elapsed = time.perf_counter() - started
        execution = {'command': command, 'seconds': elapsed, 'returncode': process.returncode}
        write_json(directory / f'{name}_execution.json', execution)
        if process.returncode:
            raise RuntimeError(f'{name} exited with {process.returncode}; see {directory}')
        provenance = 'measured'
    else:
        if digest(record) != spec['sha256']:
            raise ValueError(f'Saved evidence changed: {record}')
        elapsed = None
        provenance = 'saved'
    result = json.loads(record.read_text())
    verify_source_lock(spec)
    if spec.get('expected_engine'):
        protocol = json.loads((record.parent / 'protocol.json').read_text())
        executed = Path(protocol['command'][1])
        relative = executed.resolve().relative_to((record.parent / 'code').resolve()).as_posix()
        if relative != spec['expected_engine']:
            raise ValueError(f'Executed engine {relative} differs from {spec["expected_engine"]}')
        if spec.get('source_lock'):
            lock = json.loads(Path(spec['source_lock']['path']).read_text())
            prefix = 'track4d_calib/_engine/'
            expected = {name[len(prefix):]: value for name, value in lock['files'].items()
                        if name.startswith(prefix)}
            if protocol['code_sha256'] != expected:
                raise ValueError('Executed code snapshot differs from the frozen engine')
    for key, expected in spec.get('expect', {}).items():
        if field(result, key) != expected:
            raise ValueError(f'{record}: {key} did not equal {expected!r}')
    if provenance == 'saved' and 'seconds_field' in spec:
        elapsed = float(field(result, spec['seconds_field']))
    return {'result': str(record), 'sha256': digest(record), 'seconds': elapsed,
            'provenance': provenance}, result


def validate_plan(plan):
    required = {'iters', 'resolution', 'train_views', 'test_views', 'frame_ids',
                'seed', 'preprocess', 'checkpoint_init'}
    if required - plan['protocol'].keys():
        raise ValueError(f'Missing protocol fields: {sorted(required - plan["protocol"].keys())}')
    names = [case['name'] for case in plan['cases']]
    if len(set(names)) != len(names) or not all(re.fullmatch(r'[a-zA-Z0-9_-]+', n) for n in names):
        raise ValueError('Case names must be unique path-safe identifiers')
    if plan['baseline'] not in names:
        raise ValueError('Baseline must name a declared case')
    for case in plan['cases']:
        for name in ('calibration', 'evaluation'):
            spec = case.get(name)
            if spec is None and name == 'evaluation':
                continue
            if not isinstance(spec, dict) or 'result' not in spec:
                raise ValueError(f'{case["name"]}: missing {name} result contract')
            if 'command' not in spec and 'sha256' not in spec:
                raise ValueError('Reused records require a SHA-256 digest')
        if 'command' not in case['calibration'] and 'seconds_field' not in case['calibration']:
            raise ValueError('Imported calibration requires seconds_field')


def summarize(output, plan):
    rows = []
    for case in plan['cases']:
        path = output / case['name'] / 'result.json'
        rows.append(json.loads(path.read_text()) if path.exists() else
                    {'name': case['name'], 'label': case.get('label', case['name']), 'status': 'pending'})
    baseline = next(r for r in rows if r['name'] == plan['baseline'])
    for row in rows:
        if row.get('calibration_seconds') and baseline.get('calibration_seconds'):
            row['speedup'] = baseline['calibration_seconds'] / row['calibration_seconds']
    summary = {'protocol': plan['protocol'], 'baseline': plan['baseline'], 'rows': rows,
               'complete': all(r['status'] == 'complete' for r in rows)}
    write_json(output / 'summary.json', summary)
    columns = ['name', 'label', 'status', 'calibration_seconds', 'attempt_seconds',
               'speedup', 'psnr', 'ssim', 'lpips']
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    return summary


def print_table(summary):
    width = max(18, *(len(row['label']) for row in summary['rows']))
    print(f"{'Method':<{width}}  {'Seconds':>10}  {'Speedup':>8}  {'PSNR':>8}  Status")
    for row in summary['rows']:
        seconds = f"{row['calibration_seconds']:.2f}" if 'calibration_seconds' in row else '—'
        speedup = f"{row['speedup']:.2f}x" if 'speedup' in row else '—'
        psnr = f"{row['psnr']:.2f}" if 'psnr' in row else '—'
        print(f"{row['label']:<{width}}  {seconds:>10}  {speedup:>8}  {psnr:>8}  {row['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--summarize-only', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    validate_plan(plan)
    output = args.output.absolute()
    if args.summarize_only:
        if json.loads((output / 'plan.json').read_text()) != plan:
            raise ValueError('Plan differs from the executed snapshot')
        print_table(summarize(output, plan))
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'plan.json', plan)
    write_json(output / 'runner.json', {'sha256': digest(__file__),
                                      'timing': 'Complete calibration subprocess wall time; NVS excluded'})

    def calibrate(case):
        directory = output / case['name']
        directory.mkdir()
        row = {'name': case['name'], 'label': case.get('label', case['name']),
               'settings': case.get('settings', {}), 'status': 'calibrating'}
        write_json(directory / 'result.json', row)
        try:
            execution, _ = run_step(case['calibration'], directory, 'calibration')
            seconds = execution['seconds']
            if seconds is None or not math.isfinite(seconds) or seconds <= 0:
                raise ValueError('Calibration time must be finite and positive')
            row.update(status='calibrated', calibration_seconds=seconds, calibration=execution)
        except Exception as error:
            row.update(status='failed', error=str(error))
            execution_path = directory / 'calibration_execution.json'
            record = Path(case['calibration']['result'])
            if execution_path.exists():
                row['attempt_seconds'] = json.loads(execution_path.read_text())['seconds']
            if execution_path.exists() and record.exists():
                outcome = json.loads(record.read_text())
                if outcome.get('status') == 'partial':
                    execution = json.loads(execution_path.read_text())
                    row.update(calibration_seconds=execution['seconds'],
                               missing_dynamic_poses=outcome.get('missing_dynamic_poses'),
                               attempted_frames=outcome.get('attempted_frames'))
        write_json(directory / 'result.json', row)
        return row

    def evaluate(case, row):
        directory = output / case['name']
        if row['status'] == 'failed':
            return
        try:
            if case.get('evaluation'):
                execution, result = run_step(case['evaluation'], directory, 'evaluation')
                row['evaluation'] = execution
                for metric, key in [('psnr', 'mean_psnr'), ('ssim', 'mean_ssim'), ('lpips', 'mean_lpips_alex')]:
                    value = float(result['overall'][key])
                    if not math.isfinite(value):
                        raise ValueError(f'Nonfinite {metric}')
                    row[metric] = value
            row['status'] = 'complete'
        except Exception as error:
            row.update(status='failed', error=str(error))
        write_json(directory / 'result.json', row)
        print(json.dumps(row), flush=True)

    if plan.get('overlap_evaluation', False):
        with ThreadPoolExecutor(max_workers=1) as pool:
            futures = [pool.submit(calibrate, case) for case in plan['cases']]
            for case, future in zip(plan['cases'], futures):
                evaluate(case, future.result())
    else:
        for case in plan['cases']:
            evaluate(case, calibrate(case))
    summary = summarize(output, plan)
    print_table(summary)
    if not summary['complete']:
        raise SystemExit('Some cases failed; all outcomes are retained in summary.json')


if __name__ == '__main__':
    main()

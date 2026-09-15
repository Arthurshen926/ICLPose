"""Run all accepted v414 ancestors, both seeds, and every official query."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenes', nargs='+', required=True)
    p.add_argument('--device', required=True)
    p.add_argument('--batch-workers', type=int, default=2)
    p.add_argument('--root', type=Path, default=Path('output/cambridge_full_v418'))
    a = p.parse_args()
    env = dict(os.environ, PYTHONPATH='.', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    for scene in a.scenes:
        r = a.root / scene
        base = r / 'runtime_base'
        start = time.monotonic()
        while not (base / 'RUNTIME_ASSETS_COMPLETE.json').is_file():
            for suffix in ['', '_atlas', '_runtime']:
                log = a.root / (scene + suffix + '.log')
                if log.is_file() and 'Traceback (most recent call last):' in log.read_text():
                    raise RuntimeError('Upstream failed; inspect ' + str(log))
            if time.monotonic() - start > 86400:
                raise TimeoutError('Runtime assets unavailable: ' + scene)
            time.sleep(5)
        def run(module, args, log):
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open('a') as f:
                subprocess.run([sys.executable, '-u', '-m', 'feature_extract.tools.vfm.' + module,
                                *map(str, args)], env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
        if not (base / 'READY.json').exists():
            run('prepare_cambridge_complete_queries', ['--scene', scene, '--device', a.device,
                '--root', a.root], r / 'query_preparation.log')
        ready = json.loads((base / 'READY.json').read_text())
        def infer_split(split):
            target = r / 'full_test' / split
            front, back = target / 'frontend', target / 'backend'
            if not (front / 'COMPLETE.json').exists():
                run('replay_complete_frontend', ['--base', base, '--output', front, '--split', split,
                    '--device', a.device, '--new-query-inference'], target / 'frontend.log')
            if not (back / 'COMPLETE.json').exists():
                run('replay_complete_backend', ['--base', base, '--frontend', front, '--output', back,
                    '--split', split, '--device', a.device, '--new-query-inference'], target / 'backend.log')
            print(scene, split, 'complete v414 both seeds finished', flush=True)
        with ThreadPoolExecutor(max_workers=a.batch_workers) as pool:
            list(pool.map(infer_split, ready['splits']))
        if not (r / 'full_test_evaluation/metrics.json').exists():
            run('evaluate_cambridge_complete_scene', ['--scene', scene, '--root', a.root], r / 'evaluation.log')
        print(scene, 'complete official test evaluated', flush=True)


if __name__ == '__main__':
    main()

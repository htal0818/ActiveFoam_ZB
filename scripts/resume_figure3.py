"""Run the remainder of a compatible Figure 3 manifest with recorded solver provenance."""
import argparse
import fcntl
import hashlib
import json
import os
import platform
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from time import sleep

import reproduce_paper
import numpy as np
from numba import __version__ as numba_version
from figure3_checkpoint import Figure3Checkpoint
from reproduce_figure3 import figure3_dynamics
from reproduce_requested import ROOT, source_hash, task


def run(job, directory, digest, checkpoint_steps):
    p = job['parameters']
    print(f'Started {job["id"]} W={p["w"]:g} activity={p["mu"]:g} seed={p["seed"]} pid={os.getpid()}', flush=True)
    checkpoint = Figure3Checkpoint(Path(directory) / '.progress' / f'{job["id"]}.npz',
                                   dict(job=job, solver=digest, machine=platform.machine(),
                                        numpy=np.__version__, numba=numba_version),
                                   checkpoint_steps)

    def measure(**params):
        params.update(record_snapshot=False, example=None)
        return dict(figure3_dynamics(**params, checkpoint=checkpoint), solver_source_hash=digest)

    result = task(job, directory, {'dynamics': measure})
    if result[1] == 'ok':
        checkpoint.path.unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--manifest', type=Path,
                        help='Use a local copy of the original manifest')
    parser.add_argument('--seeds', nargs='+', type=int, choices=range(1, 11),
                        help='Process only these seed IDs from the original manifest')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--checkpoint-steps', type=int, default=12000,
                        help='Save partial state at this interval in integration steps')
    parser.add_argument('--handoff-after', nargs='+', default=[],
                        help='Require these earlier jobs to have finished before starting')
    parser.add_argument('--pause-file', type=Path,
                        help='Stop accepting jobs while this file exists')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    if args.checkpoint_steps < 100 or args.checkpoint_steps % 100:
        parser.error('checkpoint interval must be a positive multiple of 100')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.run.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = args.manifest or out / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get('profile') != 'full' or len(manifest['jobs']) != 780
                or manifest.get('protocol', {}).get('msd_center') != 'polygon area centroid'):
            raise ValueError('Expected the 780-run Figure 3 full-model manifest')
        local_manifest = out / 'manifest.json'
        if not local_manifest.is_file():
            temporary = out / '.manifest.tmp'
            temporary.write_bytes(manifest_path.read_bytes())
            temporary.replace(local_manifest)
        elif manifest_path.resolve() != local_manifest:
            info = local_manifest.stat()
            if not (getattr(info, 'st_flags', 0) & 0x40000000):
                if json.loads(local_manifest.read_text()) != manifest:
                    raise ValueError('The output directory has a different manifest')
        digest = hashlib.sha256(source_hash().encode() + Path(reproduce_paper.__file__).read_bytes()
                                + (ROOT/'scripts/reproduce_figure3.py').read_bytes()
                                + (ROOT/'scripts/figure3_checkpoint.py').read_bytes()
                                + Path(__file__).read_bytes()).hexdigest()
        jobs = [job for job in manifest['jobs']
                if args.seeds is None or job['parameters']['seed'] in args.seeds]
        if not jobs:
            raise ValueError('No matching jobs in the original manifest')
        completed, pending = [], []
        for job in jobs:
            if not (out / (job['id'] + '.npz')).is_file():
                pending.append(job)
            else:
                completed.append(job['id'])
        if args.handoff_after and not set(args.handoff_after).issubset(completed):
            raise ValueError('The outgoing worker still has unfinished jobs')
        progress = out / '.progress'
        progress.mkdir(exist_ok=True)
        for job_id in completed:
            (progress / f'{job_id}.npz').unlink(missing_ok=True)
            (progress / f'{job_id}.tmp.npz').unlink(missing_ok=True)
        record = dict(
            manifest_hash=manifest['source_hash'], solver_source_hash=digest,
            prior_completed=completed, remaining=len(pending),
            acceleration='compiled boundary validation and T1 candidate checks',
            omitted_optional_observables=['3b trajectories', '3d example geometry'],
            checkpoint_steps=args.checkpoint_steps,
            seeds=sorted({job['parameters']['seed'] for job in jobs}),
            parity='identical complete states in 600-step replays at W=0 and W=1',
            source_files={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [*sorted((ROOT/'activefoam').glob('*.py')),
                                    *[ROOT/'scripts'/name for name in
                                      ('reproduce_figure3.py', 'reproduce_paper.py', 'reproduce_requested.py',
                                       'resume_figure3.py', 'figure3_checkpoint.py')]]},
        )
        temp = out / '.solver_update.tmp'
        temp.write_text(json.dumps(record, indent=2))
        temp.replace(out / 'solver_update.json')
        print(f'Full Figure 3 sweep. {len(completed)} retained and {len(pending)} pending across {args.workers} workers.',
              flush=True)
        queue = iter(pending)
        completed_count = 0
        active = set()
        exhausted = False
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            while active or not exhausted:
                paused = args.pause_file is not None and args.pause_file.exists()
                if not paused:
                    for _ in range(args.workers - len(active)):
                        job = next(queue, None)
                        if job is None:
                            exhausted = True
                            break
                        active.add(pool.submit(run, job, str(out), digest, args.checkpoint_steps))
                if not active:
                    if paused:
                        sleep(10)
                    continue
                done, active = wait(active, timeout=10., return_when=FIRST_COMPLETED)
                for future in done:
                    ident, status, elapsed = future.result()
                    if status == 'ok':
                        (out / f'{ident}.error.json').unlink(missing_ok=True)
                    completed_count += 1
                    print(f'[{completed_count}/{len(pending)}] {ident} {status} {elapsed:.1f}s', flush=True)
        failed = [j['id'] for j in jobs if not (out / (j['id'] + '.npz')).is_file()]
        temp = out / '.status.tmp'
        temp.write_text(json.dumps(dict(total=len(jobs), completed=len(jobs)-len(failed),
                                        failed=failed), indent=2))
        temp.replace(out / 'status.json')
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

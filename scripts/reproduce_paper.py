"""Checkpointed simulations for Kim et al., Nature Physics (2021)."""
import argparse
import fcntl
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from reproduce_requested import ROOT, source_hash, task
from activefoam.paper_measure import dynamics_paper, hysteresis_paper, shear_paper, static_paper


HANDLERS = dict(equilibrium=static_paper, dynamics=dynamics_paper,
                shear=shear_paper, hysteresis=hysteresis_paper)
PANELS = ('1f', '2a', '2b', '2c', '2d', '2e', '2f', '2g', '2h', '2i', '2j', '2k',
          '3a', '3b', '3c', '3d', '4a', '4b', '4d', '4e', '4f', '4g', '4h',
          'ED1', 'ED2', 'S2', 'S3', 'S4', 'S5')
ACTIVITIES = (.25, .5, .75, 1., 1.25, 1.5)
RHOS = tuple(round((.90 + i * .02)**2, 4) for i in range(16))
STATIC_W = tuple(i / 50 for i in range(41))
YIELD_W = tuple(i / 10 for i in range(20))
ACTIVE_W = YIELD_W[:13]
SHAPE_W = (1.5, 1.55, 1.6, 1.65, 1.7, 1.75, 1.8, 1.85, 1.9, 1.95, 1.97, 1.99, 2.)
SNAPSHOTS = tuple((w, rho) for rho in (1.21, 1., .81) for w in (0., .2, .4))
UNRESOLVED = {
    '1f': 'Dimensionless junction example. The published workbook gives only the experimental length trace.',
    '3d': 'The event images show local reruns. Published example frames are unavailable.',
    '4c': 'Illustration of the shear and tension-fluctuation mechanisms.',
    'S1': 'Analytical force-balance illustration.',
}


def build_plan(panels, profile, seeds, n_side, geometry_policy, version, extra_relaxation_tauT=0.,
               seed_start=1, pressure_normals='reference', record_shear_diagnostics=False):
    selected = set(panels)
    jobs = {}
    pilot = profile == 'pilot'

    def add(kind, uses, repetitions=10, **params):
        uses = selected.intersection(uses.split())
        if not uses:
            return
        for offset in range(seeds if seeds is not None else repetitions):
            seed = seed_start + offset
            params = dict(params, seed=seed, n_side=n_side, geometry_policy=geometry_policy)
            if pressure_normals != 'reference':
                params['pressure_normals'] = pressure_normals
            if kind == 'shear' and record_shear_diagnostics:
                params['record_diagnostics'] = True
            key = kind, tuple(sorted(params.items()))
            if key not in jobs:
                jobs[key] = dict(kind=kind, parameters=params, panels=set())
            if kind == 'equilibrium' and seeds is None and offset >= 10:
                jobs[key]['panels'].update(uses - {'2e'})
            else:
                jobs[key]['panels'].update(uses)

    if profile == 'full':
        ws, rhos, ys, active_ws, mus, shape_ws = STATIC_W, RHOS, YIELD_W, ACTIVE_W, ACTIVITIES, SHAPE_W
    elif profile in ('pilot', 'spot'):
        ws, rhos, ys = (0., .2, .4, .5, .8), (.81, 1., 1.44), (0., .2, .5, 1., 1.5, 1.9)
        active_ws, mus, shape_ws = (0., .5, 1., 1.2), (.5, 1.5), (1.5, 1.99, 2.)
    else:
        ws, rhos, ys = STATIC_W[::5], RHOS[::3], YIELD_W[::3]
        active_ws, mus, shape_ws = ACTIVE_W[::2], ACTIVITIES, SHAPE_W[::3]
    static = {}
    for rho in rhos:
        for w in ws:
            static.setdefault((w, rho), set()).update(('2b', '2c'))
    for rho in (.81, 1., 1.44):
        for w in ws:
            if rho != .81 or w >= .1:
                static.setdefault((w, rho), set()).add('2d')
            static.setdefault((w, rho), set()).add('2e')
    for rho in rhos:
        static.setdefault((0., rho), set()).add('2f')
    for w in shape_ws:
        static.setdefault((w, 1.), set()).add('2g')
    for coord in SNAPSHOTS:
        static.setdefault(coord, set()).add('2a')
    static.setdefault((0., 1.), set()).add('2f')
    for w, rho in static:
        uses = selected & static[w, rho]
        quantitative = uses & {'2b', '2c', '2d', '2f', '2g'}
        add('equilibrium', ' '.join(uses), 20 if quantitative else 10 if '2e' in uses else 1,
            w=w, rho=rho, record_space='2d' in uses, record_snapshot=bool(uses & {'2a', '2f', '2g'}),
            extra_relaxation_tauT=extra_relaxation_tauT)
    for rho in (.81, 1., 1.44):
        for direction in ('up', 'down'):
            add('hysteresis', '2e', rho=rho, direction=direction, pilot=pilot)

    for mu in mus:
        for w in sorted({*active_ws, 1.}):
            uses = '3c 3d 4f 4g ED2 S2 S4'
            if w >= .8:
                uses += ' ED1'
            if w == 1.:
                uses += ' 3a'
                if mu in (.5, 1.5):
                    uses += ' 3b'
                if mu in (.5, 1., 1.5):
                    uses += ' 4h'
            if mu >= .5:
                uses += ' 4d 4e'
            if w == .5 and mu in (.5, 1., 1.5):
                uses += ' 4a'
            if mu == .5 and w == 1.:
                uses += ' 1f'
            example = None
            if '3d' in selected and mu == 1.5:
                example = {0.: 'gain', 1.: 'swap'}.get(w)
            add('dynamics', uses, w=w, rho=1., mu=mu,
                duration_tauT=2. if pilot else 120.,
                record_msd=bool(selected & set(uses.split()) & {'3a', '3b', '3c', 'ED1', 'ED2'}),
                record_snapshot=bool(selected & set(uses.split()) & {'3b', '4h'}),
                record_junction='1f' in selected and '1f' in uses.split(), example=example)
    for rho in (1.1025, 1.21):
        ws_s3 = ACTIVE_W[:7 if rho == 1.1025 else 4]
        if profile in ('pilot', 'spot'):
            ws_s3 = ws_s3[::len(ws_s3) - 1]
        elif profile == 'coarse':
            ws_s3 = ws_s3[::2]
        for w in ws_s3:
            for mu in (.5, 1., 1.5):
                add('dynamics', 'S3', w=w, rho=rho, mu=mu,
                    duration_tauT=2. if pilot else 120., record_msd=False)

    passive = {}
    for rho in rhos:
        for w in ys:
            passive.setdefault((w, rho), set()).update(('2k', 'S5'))
    for w in ys:
        passive.setdefault((w, 1.), set()).add('2j')
    for w in (0., .2, .5, 1., 1.5):
        passive.setdefault((w, 1.), set()).add('2i')
    passive.setdefault((.5, 1.), set()).add('2h')
    for (w, rho), uses in passive.items():
        add('shear', ' '.join(uses), w=w, rho=rho, mu=0., duration=5. if pilot else 280.,
            pilot=pilot, record_snapshot='2h' in selected & uses)

    active_shear = {}
    for w in active_ws:
        for mu in mus:
            if mu >= .5:
                active_shear[w, mu] = {'4d', '4e', '4g', 'ED2'}
                if w == 1. and mu in (.5, 1., 1.5):
                    active_shear[w, mu].add('4h')
    for mu in (0., .5, 1., 1.5):
        active_shear.setdefault((.5, mu), set()).update(('4a', '4b'))
    for (w, mu), uses in active_shear.items():
        if mu == 0. and selected & passive[.5, 1.]:
            # The zero-activity curve and passive yield use the same checkpoint.
            add('shear', '4a 4b', w=w, rho=1., mu=mu, duration=5. if pilot else 280.,
                pilot=pilot, record_snapshot='2h' in selected)
        else:
            duration = 1000. if selected & uses & {'4d', '4e', '4g', '4h', 'ED2'} else 580.
            add('shear', ' '.join(uses), w=w, rho=1., mu=mu,
                duration=5. if pilot else 280. if mu == 0. else duration,
                pilot=pilot, record_snapshot=False)

    result = []
    for job in jobs.values():
        job['panels'] = sorted(job['panels'])
        job['source_hash'] = version
        job['profile'] = profile
        key = json.dumps({k: v for k, v in job.items() if k != 'panels'}, sort_keys=True)
        job['id'] = job['kind'] + '_' + hashlib.sha256(key.encode()).hexdigest()[:16]
        result.append(job)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('pilot', 'spot', 'coarse', 'full'), default='pilot',
                        help='Spot uses the pilot grid at publication-length observation windows')
    parser.add_argument('--panels', nargs='+', choices=PANELS, default=PANELS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--n-side', type=int)
    parser.add_argument('--seeds', type=int)
    parser.add_argument('--seed-start', type=int, default=1, help='First seed in the requested ensemble')
    parser.add_argument('--adhesion', type=float, nargs='+', help='Run selected W/T0 values from the grid')
    parser.add_argument('--activity', type=float, nargs='+', help='Run selected tension fluctuations from the grid')
    parser.add_argument('--density', type=float, nargs='+', help='Run selected densities from the grid')
    parser.add_argument('--kind', nargs='+', choices=tuple(HANDLERS), help='Run selected protocols')
    parser.add_argument('--extra-relaxation-tauT', type=float, default=0.,
                        help='Additional passive relaxation for equilibrium sensitivity runs')
    parser.add_argument('--geometry-policy', choices=('strict', 'reference'), default='reference')
    parser.add_argument('--pressure-normals', choices=('reference', 'boundary'), default='reference',
                        help='Choose pressure normals from the MATLAB centroid test or signed cell boundaries')
    parser.add_argument('--shear-diagnostics', action='store_true',
                        help='Record crossing checks over time and retain the trace if a shear run fails')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Write the job manifest and report counts')
    args = parser.parse_args()
    n_side = args.n_side if args.n_side is not None else (4 if args.profile in ('pilot', 'spot') else 10)
    seeds = args.seeds if args.seeds is not None else (None if args.profile == 'full' else 1)
    if n_side < 3 or min(args.workers, args.seed_start) < 1 or (seeds is not None and seeds < 1):
        parser.error('n-side must be at least 3, and workers and seeds must be positive')
    if not math.isfinite(args.extra_relaxation_tauT) or args.extra_relaxation_tauT < 0.:
        parser.error('extra-relaxation-tauT must be finite and nonnegative')
    runner = Path(__file__).read_bytes()
    version = hashlib.sha256(source_hash().encode() + runner).hexdigest()
    jobs = build_plan(args.panels, args.profile, seeds, n_side, args.geometry_policy,
                      version, extra_relaxation_tauT=args.extra_relaxation_tauT, seed_start=args.seed_start,
                      pressure_normals=args.pressure_normals, record_shear_diagnostics=args.shear_diagnostics)
    for parameter, values in (('w', args.adhesion), ('mu', args.activity), ('rho', args.density)):
        if values is not None:
            jobs = [job for job in jobs if job['parameters'].get(parameter) in values]
    if args.kind is not None:
        jobs = [job for job in jobs if job['kind'] in args.kind]
    if not jobs:
        parser.error('no jobs match the selected panels and parameter values')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'another runner is writing to {args.out}')
        manifest = dict(paper='10.1038/s41567-021-01215-1', model='FoamTissue (Numba)',
                        profile=args.profile, source_hash=version, n_side=n_side,
                        geometry_policy=args.geometry_policy, pressure_normals=args.pressure_normals,
                        panels=list(args.panels),
                        limitations=UNRESOLVED, jobs=jobs)
        path = args.out / 'manifest.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(manifest, indent=2))
        temp.replace(path)
        counts = {kind: sum(job['kind'] == kind for job in jobs) for kind in HANDLERS}
        print(f'{args.profile}, {n_side**2} cells, {len(jobs)} jobs, {counts}', flush=True)
        if args.plan:
            return
        pending = [job for job in jobs if not (args.resume and (args.out / (job['id'] + '.npz')).exists())]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(task, job, str(args.out), HANDLERS) for job in pending]
            for index, future in enumerate(as_completed(futures), 1):
                name, status, elapsed = future.result()
                print(f'[{index}/{len(pending)}] {name} {status} {elapsed:.1f}s', flush=True)
        failed = [job['id'] for job in jobs if not (args.out / (job['id'] + '.npz')).exists()]
        (args.out / 'status.json').write_text(json.dumps(dict(total=len(jobs), completed=len(jobs) - len(failed),
                                                            failed=failed), indent=2))
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

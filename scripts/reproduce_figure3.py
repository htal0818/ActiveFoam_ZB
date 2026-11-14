"""Run Figure 3 ensembles with the paper solver and store the plotted measurements."""
import argparse
import fcntl
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import reproduce_paper
from reproduce_requested import source_hash, task
from activefoam._kernels import polygon_centroids
from activefoam.foam_measure import evolve, initial_voronoi, initialize, quality
from activefoam.paper_measure import ContactEvents, snapshot
import numpy as np


PANELS = ('3a', '3b', '3c', '3d')


def figure3_dynamics(w, rho, mu, seed, n_side, geometry_policy, duration_tauT,
                     record_msd, record_snapshot, record_junction, example, checkpoint=None):
    if not record_msd or record_junction:
        raise ValueError('Figure 3 requires MSD and uses its own contact measurements')
    if checkpoint is not None and (record_snapshot or example is not None):
        raise ValueError('Checkpointed runs support MSD and neighbor exchange measurements')
    continuing = checkpoint is not None and checkpoint.path.is_file()
    if continuing:
        foam = initial_voronoi(n_side, seed, True, w=w, rho=rho)
        foam.geometry_policy = geometry_policy
    else:
        foam = initialize(w, rho, seed, n_side, anneal_mu=mu, geometry_policy=geometry_policy)
    duration = duration_tauT * foam.tauT
    stride, phase = 10, 3
    sample_dt = stride * foam.dt
    total_steps = round(duration / foam.dt)
    frames = (total_steps - phase) // stride + 1
    progress_every = round(100. / sample_dt)
    cells = foam.nFa
    trajectory = np.empty((frames, cells, 2))
    previous_pairs = np.zeros(cells * cells, dtype=bool)
    current_pairs = np.zeros_like(previous_pairs)
    previous_center = None
    final_snapshot = None
    step = frame = gains = losses = 0
    events = ContactEvents(foam, example=example) if example else None
    if continuing:
        step, frame, previous_center, previous_pairs, gains, losses = checkpoint.load(
            foam, trajectory, stride, phase, total_steps)
        print(f'Resuming W={w:g} activity={mu:g} seed={seed} at step {step}/{total_steps}', flush=True)

    def observe(_time, tissue):
        nonlocal step, frame, gains, losses, previous_center, previous_pairs, current_pairs
        nonlocal final_snapshot
        step += 1
        if checkpoint is not None and step % checkpoint.every == 0 and step < total_steps:
            checkpoint.save(tissue, step, frame, trajectory, previous_center, previous_pairs, gains, losses)
        if step % stride != phase:
            return
        centers = polygon_centroids(*tissue._fast_geometry_inputs(), tissue.bs, tissue.sstn)
        if frame:
            delta = centers - previous_center
            delta -= tissue.bs * np.rint(delta / tissue.bs)
            trajectory[frame] = trajectory[frame - 1] + delta
        else:
            trajectory[frame] = centers
        previous_center = centers

        # Count distinct cell pairs, including changes made by adjacent T4.
        a, b = tissue.e_f[tissue.e_f[:, 0] >= 0].T
        current_pairs.fill(False)
        valid = (b >= 0) & (a != b)
        lo = np.minimum(a[valid], b[valid])
        hi = np.maximum(a[valid], b[valid])
        current_pairs[lo * cells + hi] = True
        if frame:
            gains += np.count_nonzero(current_pairs & ~previous_pairs)
            losses += np.count_nonzero(previous_pairs & ~current_pairs)
        previous_pairs, current_pairs = current_pairs, previous_pairs
        frame += 1
        if frame % progress_every == 0:
            print(f'Progress W={w:g} activity={mu:g} seed={seed} '
                  f'{frame * sample_dt:g}/{duration:g} tau_R', flush=True)
        if record_snapshot and frame == frames:
            final_snapshot = snapshot(tissue)

    def capture(iteration, transition, tissue):
        if events.example is not None:
            events(iteration, transition, tissue)
            if events.example is None:
                del capture.before_transition

    if events is not None:
        capture.before_transition = events.before_transition
    evolve(foam, (total_steps - step)*foam.dt, mu, start=step, sample_every=1, observer=observe,
           contact_observer=capture if events is not None else None)
    if frame != frames or not np.isfinite(trajectory).all():
        raise ValueError(f'Invalid centroid trajectory with {frame}/{frames} frames')

    last_lag = min(frames - 1, round(100 * foam.tauT / sample_dt))
    lags = np.unique(np.rint(np.geomspace(1, last_lag, 136)).astype(int))
    msd = np.empty(lags.size)
    msd_com = np.empty_like(msd)
    centers_of_mass = trajectory.mean(axis=1)
    for i, lag in enumerate(lags):
        delta = trajectory[lag:] - trajectory[:-lag]
        msd[i] = np.mean(np.sum(delta**2, axis=2)) / foam.A0
        drift = centers_of_mass[lag:] - centers_of_mass[:-lag]
        msd_com[i] = max(0., msd[i] - np.mean(np.sum(drift**2, axis=1)) / foam.A0)
    result = dict(
        tlag=lags * sample_dt / foam.tauT, msd=msd, msd_com=msd_com,
        msd_center='area', sample_dt=sample_dt, sample_frames=frames,
        sampled_neighbor_change_count=2 * (gains + losses),
        sampled_contact_gains=gains, sampled_contact_losses=losses,
        sampled_neighbor_change_duration=(frames - 1) * sample_dt, **quality(foam),
    )
    if record_snapshot:
        indices = np.linspace(0, cells - 1, min(cells, 7), dtype=int)
        result.update(final_snapshot, trajectory=trajectory[-last_lag - 1:, indices],
                      trajectory_cells=indices)
    if events is not None:
        result.update(events.frames)
    return result


def run_job(job, out):
    p = job['parameters']
    print(f'Started {job["id"]} W={p["w"]:g} activity={p["mu"]:g} seed={p["seed"]}', flush=True)
    return task(job, out, {'dynamics': figure3_dynamics})


def job_order(job):
    p = job['parameters']
    adhesion = p['w']
    activity = p['mu']
    first = ((1., .5), (1., 1.5), (0., 1.5))
    if (adhesion, activity) in first:
        return p['seed'], -1, first.index((adhesion, activity)), 0
    anchor = (1., 0., .3, .7, 1.2)
    priority = anchor.index(adhesion) if adhesion in anchor else len(anchor)
    return p['seed'], priority, adhesion, (.5, 1.5, .25, .75, 1., 1.25).index(activity)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('full', 'pilot'), default='full')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')

    version = hashlib.sha256(source_hash().encode()
                             + Path(reproduce_paper.__file__).read_bytes()
                             + Path(__file__).read_bytes()).hexdigest()
    n_side = 10 if args.profile == 'full' else 4
    seeds = None if args.profile == 'full' else 1
    jobs = reproduce_paper.build_plan(PANELS, args.profile, seeds, n_side, 'reference', version)
    jobs.sort(key=job_order)
    manifest = dict(paper='10.1038/s41567-021-01215-1', model='FoamTissue (Numba)',
                    profile=args.profile, source_hash=version, n_side=n_side,
                    geometry_policy='reference', pressure_normals='reference',
                    panels=list(PANELS), jobs=jobs, limitations=reproduce_paper.UNRESOLVED,
                    protocol=dict(density=1., dt_tauR=.005, tauT_over_tauR=10.,
                                  sampling_steps=10, sample_phase=3,
                                  msd_center='polygon area centroid', msd_frame='lab',
                                  neighbor_rate='two cell incidences per gained or lost pair per cell per tau_R',
                                  cell_count_basis='100-cell reproduction choice for the full ensemble'),
                    saved_measurements='MSD curves, Figure 3b tracks, and sampled contact gains and losses')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / '.run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'another runner is writing to {args.out}')
        temporary = args.out / 'manifest.tmp'
        temporary.write_text(json.dumps(manifest, indent=2))
        temporary.replace(args.out / 'manifest.json')
        print(f'{args.profile}, {n_side**2} cells, {len(jobs)} dynamics jobs', flush=True)
        if args.plan:
            return
        pending = [job for job in jobs
                   if not (args.resume and (args.out / (job['id'] + '.npz')).is_file())]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run_job, job, str(args.out)) for job in pending]
            for index, future in enumerate(as_completed(futures), 1):
                name, status, elapsed = future.result()
                print(f'[{index}/{len(pending)}] {name} {status} {elapsed:.1f}s', flush=True)
        failed = [job['id'] for job in jobs if not (args.out / (job['id'] + '.npz')).is_file()]
        (args.out / 'status.json').write_text(json.dumps(
            dict(total=len(jobs), completed=len(jobs) - len(failed), failed=failed), indent=2))
        if failed:
            raise SystemExit(1)


if __name__ == '__main__':
    main()

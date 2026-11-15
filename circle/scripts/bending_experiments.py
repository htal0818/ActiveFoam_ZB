"""Deterministic, fixed-topology cortical bending pilot; no imposed motion."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import pickle
import time

import numpy as np

from circle.bending import (BendingFoam, measure_frame, smooth_patch, edge_roles,
                            t1_candidates, validate_closed_ring, T1_LENGTH)
from circle.invagination import TopologyLimit
from circle.force_probe import isolated_copy
from circle.scripts.invagination_size import inner_radius_for_size, save_compressed
from circle.collective import save_result
from circle import state

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'circle/data/bending_48_20261115'
PREPARED = None


def prepare():
    cache = Path('/tmp/activefoam_bending48_prepared_v1.pkl')
    if cache.exists() and (DATA/'preparation.json.gz').exists():
        with cache.open('rb') as stream:
            f = pickle.load(stream)
        validate_closed_ring(f)
        return f
    f = BendingFoam.ring(n=48, inner_radius=inner_radius_for_size(48))
    f.activation_weights = f.active_cells.astype(float); f.activation_ramp = 0.
    history = []
    for k in range(1, round(120/f.dt)+1):
        f.advance_scheduled(k, 0.)
        if k % round(.5/f.dt) == 0:
            validate_closed_ring(f)
            history.append(measure_frame(f.frame()))
        if k % round(20/f.dt) == 0:
            print('preparation', round(f.morph_time, 1), flush=True)
    save_compressed(DATA/'preparation.json.gz', dict(frame=f.frame(), history=history))
    for key in state.KEYED_CACHES:
        f.__dict__.pop(key, None)
    f.e_mid = [np.asarray(p).copy() for p in f.e_mid]
    with cache.open('wb') as stream:
        pickle.dump(f, stream)
    return f


def cases():
    result = []
    def add(name, a=2., bulk=5., lateral=0., patch=8, ramp=5., delay=20.):
        result.append(dict(name=name, apical=a, bulk=bulk, lateral=lateral,
                           patch=patch, ramp=ramp, delay=delay, duration=100., dt=.0025,
                           spacing=.075, collapse_fraction=.005))
    for bulk in (.5, 5., 50., 500.):
        add(f'control_B{bulk:g}', a=0., bulk=bulk)
    add('sharp_A2', ramp=0.)
    add('sharp_ramped_A2', ramp=5.)
    add('smooth_equal_total', a=2*8/float(smooth_patch(48, 8).sum()))
    for a in (.5, 1., 2., 4.):
        add(f'apical_A{a:g}', a=a)
    for bulk in (.5, 50., 500.):
        add(f'lumen_B{bulk:g}', bulk=bulk)
    for lateral in (.5, 1.):
        add(f'lateral_L{lateral:g}', lateral=lateral)
    add('weak_combined', a=.5, lateral=.5)
    for patch in (4, 12):
        add(f'patch_P{patch}', patch=patch)
    for label, spacing in [('dt', .075), ('mesh', .0375)]:
        control_name = f'control_B5_{label}_half'
        for name, amplitude in [(control_name, 0.), (f'refine_{label}_A2', 2.)]:
            add(name, a=amplitude)
            result[-1].update(dt=.00125, spacing=spacing, duration=50., control_name=control_name)
    return result


def run_case(config):
    path = DATA/f"{config['name']}.json.gz"
    if path.exists():
        with gzip.open(path, 'rt') as stream:
            previous = json.load(stream)
        if previous['parameters'] != config:
            raise ValueError('saved experiment parameters differ')
        return previous['summary']
    f = isolated_copy(PREPARED)
    f.dt = config['dt']; f.edpc = config['spacing']
    if f.edpc < PREPARED.edpc:
        before_areas = f.f_area.copy()
        refined = []
        for points in f.e_mid:
            points = np.asarray(points)
            new = np.empty((2*len(points)-1, 2))
            new[::2] = points; new[1::2] = (points[:-1]+points[1:])/2
            refined.append(new)
        f.e_mid = refined; f.e_rfn = np.array([len(p)-2 for p in refined])
        f._update_faces()
        np.testing.assert_allclose(f.f_area, before_areas, atol=1e-10, rtol=0)
    f.activation_start = f.morph_time
    f.activation_ramp = config['ramp']; f.lateral_delay = config['delay']
    f.apical_amplitude = config['apical']; f.lateral_amplitude = config['lateral']
    f.activation_weights = (f.active_cells.astype(float) if config['name'].startswith('sharp')
                           else smooth_patch(f.nFa, config['patch']))
    if 'activation_weights' in config:
        weights = np.asarray(config['activation_weights'], dtype=float)
        if weights.shape != (f.nFa,) or not np.all(np.isfinite(weights)) or np.any((weights < 0) | (weights > 1)):
            raise ValueError('activation weights must have one finite value in [0, 1] per cell')
        f.activation_weights = weights.copy()
    f.active_cells = f.activation_weights > 0
    # Match both geometry AND lumen pressure when varying the compression modulus.
    initial_area = f.interface_data()[2]; initial_pressure = f.lumen_pressure(initial_area)
    f.lumen_reference_area = initial_area; f.lumen_initial_pressure = initial_pressure
    f.lumen_bulk_modulus = config['bulk']
    start = f.morph_time; roles = edge_roles(f); original_ends = f.e_v.copy()
    original_faces = f.e_f.copy(); first_frame = f.frame(); first = measure_frame(first_frame)
    collapse = config['collapse_fraction']*float(np.mean(first['height']))
    history = [dict(first, elapsed=0.)]; frames = [first_frame]; proposals = {}
    status = 'complete'; reason = None; event = None; last_valid = 0.
    sample = max(1, round(.5/f.dt)); store = max(1, round(5/f.dt))
    wall_start = time.monotonic(); start_step = round(start/f.dt)
    for k in range(1, round(config['duration']/f.dt)+1):
        iteration = start_step+k
        f.log.iteration = iteration
        before = f.edge_lengths() if iteration % 3 == 0 else None
        try:
            f.step(0.)             # explicitly fixed topology; no transition operations
            lengths = f.edge_lengths()
            chords = np.linalg.norm(f.vpos[f.e_v[:, 1]]-f.vpos[f.e_v[:, 0]], axis=1)
            if before is not None:
                for e in t1_candidates(f, before, lengths):
                    key = str(e)
                    record = proposals.setdefault(key, dict(kind='T1', edge=e, role=str(roles[e]),
                        cells=f.e_f[e].tolist(), first_elapsed=k*f.dt, last_elapsed=k*f.dt,
                        eligible_steps=0, min_length=float(lengths[e]), applied=False))
                    record['last_elapsed'] = k*f.dt; record['eligible_steps'] += 1
                    record['min_length'] = min(record['min_length'], float(lengths[e]))
            quality = validate_closed_ring(f)   # every step, including local microfolds
            minimum = np.minimum(lengths, chords)
            if not np.all(np.isfinite(minimum)):
                raise ValueError('nonfinite interface geometry')
            e = int(np.argmin(minimum))
            if minimum[e] < collapse:
                status = 'collapse_guard'
                event = dict(edge=e, role=str(roles[e]), cells=f.e_f[e].tolist(),
                             length=float(lengths[e]), chord=float(chords[e]),
                             threshold=collapse, elapsed=k*f.dt)
                reason = 'interface reached numerical collapse guard; no topology rule applied'
                break
            if k % sample == 0:
                frame = f.frame(); measured = measure_frame(frame)
                measured.update(elapsed=k*f.dt, quality=quality,
                    min_length_by_role={role:float(lengths[roles == role].min())
                                        for role in ('apical', 'basal', 'lateral')},
                    actual_tensions={role:float(f.e_t[roles == role].max())
                                     for role in ('apical', 'basal', 'lateral')},
                    target_tensions={role:float(f.interface_data()[5][roles == role].max())
                                     for role in ('apical', 'basal', 'lateral')})
                history.append(measured); last_valid = k*f.dt
                if k % store == 0:
                    frames.append(frame)
            if k % round(25/f.dt) == 0:
                print(config['name'], 'elapsed', k*f.dt, flush=True)
        except ValueError as exc:
            status = 'geometry_failure'; reason = str(exc)
            event = dict(elapsed=k*f.dt)
            break
        except (FloatingPointError, ZeroDivisionError, OverflowError) as exc:
            status = 'numerical_failure'; reason = f'{type(exc).__name__}: {exc}'
            event = dict(elapsed=k*f.dt)
            break
    if history[-1]['elapsed'] > 0 and frames[-1]['time'] < frame['time']:
        frames.append(frame)
    if not np.array_equal(f.e_v, original_ends) or not np.array_equal(f.e_f, original_faces):
        raise RuntimeError('fixed topology changed')
    summary = dict(name=config['name'], status=status, reason=reason,
                   last_valid_elapsed=last_valid, event=event,
                   proposal_roles=sorted({p['role'] for p in proposals.values()}),
                   proposal_count=len(proposals), wall_seconds=time.monotonic()-wall_start)
    result = dict(parameters=config, summary=summary, history=history, frames=frames,
                  proposals=list(proposals.values()), initial_lumen_pressure=initial_pressure,
                  initial_lumen_area=initial_area, activation_weights=f.activation_weights,
                  fixed_topology=True, checks='every timestep; zero tolerated proper crossings',
                  collapse_threshold=collapse)
    save_compressed(path, result)
    return summary


def audit_original_stop():
    """Identify the original scheduled proposal before changing its treatment."""
    class AuditFoam(BendingFoam):
        def _guard(self, kind, ident, site, operation, accepted_of, rejected_value):
            if kind == 'T1':
                e = int(ident)
                self.audit_result = dict(kind=kind, edge=e, role=str(edge_roles(self)[e]),
                    length=float(self.edge_lengths()[e]), cells=self.e_f[e].tolist(),
                    elapsed=self.morph_time-self.activation_start)
            else:
                self.audit_result = dict(kind=kind, ident=str(ident),
                                        elapsed=self.morph_time-self.activation_start)
            raise TopologyLimit('audited original stop')
    f = isolated_copy(PREPARED); f.__class__ = AuditFoam
    f.activation_start = f.morph_time; f.apical_amplitude = 2.; f.activation_ramp = 0.
    for k in range(1, round(20/f.dt)+1):
        try:
            f.advance_scheduled(round(f.activation_start/f.dt)+k, 0.)
        except TopologyLimit:
            save_result(DATA/'original_proposal.json', f.audit_result)
            print('Original proposal', f.audit_result, flush=True)
            return
    raise RuntimeError('original stop was not reproduced')


def main():
    global PREPARED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--only', nargs='*')
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    PREPARED = prepare()
    if not (DATA/'original_proposal.json').exists():
        audit_original_stop()
    configs = cases()
    if args.only:
        configs = [c for c in configs if c['name'] in args.only]
    summaries = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context('fork')) as pool:
        futures = {pool.submit(run_case, config): config for config in configs}
        for future in as_completed(futures):
            summary = future.result(); summaries.append(summary)
            print(json.dumps(summary), flush=True)
    all_summaries = []
    for path in sorted(DATA.glob('*.json.gz')):
        with gzip.open(path, 'rt') as stream:
            result = json.load(stream)
        if 'summary' in result:
            all_summaries.append(result['summary'])
    save_result(DATA/'run_summaries.json', all_summaries)


if __name__ == '__main__':
    main()

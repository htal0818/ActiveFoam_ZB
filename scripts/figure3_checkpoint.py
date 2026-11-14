"""Store and restore the state required to continue one Figure 3 seed."""
import json
from pathlib import Path

import numpy as np

from activefoam._kernels import List
from activefoam.foam_measure import quality


def pack(rows):
    lengths = np.fromiter((len(row) for row in rows), dtype=np.int64)
    return np.concatenate(rows), np.cumsum(lengths)[:-1]


class Figure3Checkpoint:
    def __init__(self, path, identity, every):
        if every < 100 or every % 100:
            raise ValueError('Checkpoint interval must be a positive multiple of 100 steps')
        self.path = Path(path)
        self.identity = identity
        self.every = every

    def save(self, foam, step, frame, trajectory, center, pairs, gains, losses):
        mids, film_ends = pack(list(foam.e_mid))
        vertex_loops, vertex_ends = pack([np.asarray(row, dtype=np.int64) for row in foam.f_v])
        edge_loops, edge_ends = pack([np.asarray(row, dtype=np.int64) for row in foam.f_e])
        pressure_area, pressure_center = foam._pressure_geometry
        temporary = self.path.with_suffix('.tmp.npz')
        np.savez(temporary, identity=json.dumps(self.identity, sort_keys=True),
                 step=step, frame=frame, rng=json.dumps(foam.rng.bit_generator.state),
                 mids=mids, film_ends=film_ends, vertex_loops=vertex_loops, vertex_ends=vertex_ends,
                 edge_loops=edge_loops, edge_ends=edge_ends, pressure_area=pressure_area,
                 pressure_center=pressure_center, trajectory=trajectory[:frame], center=center,
                 pairs=pairs, gains=gains, losses=losses, **quality(foam),
                 **{key: getattr(foam, key) for key in
                    ('vpos', 'v_e', 'v_f', 'e_v', 'e_f', 'e_t', 'e_rfn', 'f_area', 'f_center')})
        temporary.replace(self.path)

    def load(self, foam, trajectory, sample_stride, sample_phase, total_steps):
        with np.load(self.path, allow_pickle=False) as state:
            if json.loads(state['identity'].item()) != self.identity:
                raise ValueError(f'Checkpoint protocol or solver mismatch in {self.path}')
            step, frame = int(state['step']), int(state['frame'])
            if (step <= 0 or step >= total_steps or step % 100
                    or frame != (step - sample_phase) // sample_stride + 1
                    or state['trajectory'].shape != trajectory[:frame].shape
                    or state['center'].shape != trajectory.shape[1:]
                    or state['pairs'].shape != (foam.nFa**2,)):
                raise ValueError(f'Invalid checkpoint position in {self.path}')
            for key in ('vpos', 'v_e', 'v_f', 'e_v', 'e_f', 'e_t', 'e_rfn', 'f_area', 'f_center'):
                setattr(foam, key, state[key])
            for key in quality(foam):
                setattr(foam, '_' + key, state[key].item())
            foam.e_mid = List([np.ascontiguousarray(row) for row in np.split(state['mids'], state['film_ends'])])
            foam.f_v = [row.tolist() for row in np.split(state['vertex_loops'], state['vertex_ends'])]
            foam.f_e = [row.tolist() for row in np.split(state['edge_loops'], state['edge_ends'])]
            foam._pressure_geometry = (state['pressure_area'], state['pressure_center'])
            foam.rng.bit_generator.state = json.loads(state['rng'].item())
            trajectory[:frame] = state['trajectory']
            return step, frame, state['center'], state['pairs'], int(state['gains']), int(state['losses'])

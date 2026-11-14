"""Measurements for the disc model.

Wall quantities come from ``DiscFoam.wall`` (a ``WallRecord``): nodal reaction
forces on the node identities before remeshing, in the code's force units
(displacement = psi*dt*force).  A junction is one node, however many films end
on it, so every sum below counts each physical node once.
"""
import numpy as np

from activefoam.foam_full import SPACE


def wall_load(record):
    """Integrated wall reaction of one step.

    total      sum of nodal reaction magnitudes (integrated traction magnitude)
    resultant  vector sum of the nodal reactions
    torque     torque of the reactions about the disc centre (zero for a
               frictionless circular wall)
    direct     sum of the continuous-constraint values max(0, f.n) over nodes
               that were on the wall at the start of the step
    contacts   number of nodes on the wall at the start of the step
    arrivals   number of nodes that reached the wall during the step
    """
    reaction = np.vstack([record.junction_reaction, record.node_reaction])
    position = np.vstack([record.junction_pos, record.node_pos])
    flags = np.concatenate([record.junction_flag, record.node_flag])
    magnitude = np.linalg.norm(reaction, axis=1)
    arm = position - record.center
    return dict(total=float(magnitude.sum()),
                resultant=reaction.sum(axis=0),
                torque=float(np.sum(arm[:, 0] * reaction[:, 1] - arm[:, 1] * reaction[:, 0])),
                direct=float(record.junction_direct.sum() + record.node_direct.sum()),
                arrivals=int(np.sum(flags == 2)),
                contacts=int(np.sum(flags == 1)))


def wall_traction(foam):
    """Per-node traction (force per unit wall length) of the last step.

    Returns angles, traction and tributary lengths for nodes with a reaction.
    Traction is nodal force divided by tributary length; it is a per-length
    estimate only and is never multiplied by the length again.
    """
    record = foam.wall
    reaction = np.vstack([record.junction_reaction, record.node_reaction])
    position = np.vstack([record.junction_pos, record.node_pos])
    tributary = np.concatenate([record.junction_tributary, record.node_tributary])
    magnitude = np.linalg.norm(reaction, axis=1)
    keep = (magnitude > 0) & (tributary > 0)
    angle = np.arctan2(position[keep, 1] - foam.center[1], position[keep, 0] - foam.center[0])
    return angle, magnitude[keep] / tributary[keep], tributary[keep]


def traction_profile(foam, bins=36):
    """Wall reaction per unit wall arclength in angular bins.

    ``profile[i] * R_w * dtheta`` summed over bins equals the integrated
    reaction ``wall_load(...)['total']``.
    """
    record = foam.wall
    reaction = np.vstack([record.junction_reaction, record.node_reaction])
    position = np.vstack([record.junction_pos, record.node_pos])
    magnitude = np.linalg.norm(reaction, axis=1)
    angle = np.arctan2(position[:, 1] - foam.center[1], position[:, 0] - foam.center[0])
    edges = np.linspace(-np.pi, np.pi, bins + 1)
    force, _ = np.histogram(angle, bins=edges, weights=magnitude)
    arclength = foam.radius * np.diff(edges)
    return 0.5 * (edges[1:] + edges[:-1]), force / arclength, arclength


def packing_fraction(foam):
    """Cell area over disc area: a necessary consistency check, not an overlap test."""
    return foam.volume_fraction()


def cell_centroids(foam):
    """Area centroids of the cells' curved polygons."""
    out = np.empty((foam.nFa, 2))
    for cell in range(foam.nFa):
        points = foam._cell_polyline(cell)
        x, y = points.T
        cross = x * np.roll(y, -1) - np.roll(x, -1) * y
        area = 0.5 * cross.sum()
        out[cell, 0] = np.sum((x + np.roll(x, -1)) * cross) / (6 * area)
        out[cell, 1] = np.sum((y + np.roll(y, -1)) * cross) / (6 * area)
    return out


def touching_wall(foam, tolerance=None):
    """Boolean per cell: one of its outer-film nodes lies on the wall."""
    tolerance = 1e-6 * foam.radius if tolerance is None else tolerance
    touching = np.zeros(foam.nFa, dtype=bool)
    for e in np.flatnonzero((foam.e_f[:, 0] == SPACE) & (foam.e_f[:, 1] >= 0)):
        radius = np.hypot(*(np.asarray(foam.e_mid[e]) - foam.center).T)
        if radius.max() >= foam.radius - tolerance:
            touching[foam.e_f[e, 1]] = True
    return touching


def wall_layer(foam, depth=1.0):
    """Boolean per cell: centroid within ``depth`` of the wall (default one cell size)."""
    centers = cell_centroids(foam)
    return np.hypot(*(centers - foam.center).T) > foam.radius - depth


def neighbor_pairs(foam):
    """Set of distinct neighbouring cell pairs."""
    faces = foam.e_f[(foam.e_f[:, 0] != SPACE) & (foam.e_f[:, 1] != SPACE)]
    return {(int(min(a, b)), int(max(a, b))) for a, b in faces if a != b}


def zone_statistics(foam, depth=1.0):
    """Contact number, shape factor and area, split into wall layer and interior."""
    layer = wall_layer(foam, depth)
    contacts = foam.neighbor_number()
    perimeter = np.array([foam._perimeter(c) for c in range(foam.nFa)])
    shape = perimeter / np.sqrt(foam.f_area)
    out = dict(packing=packing_fraction(foam), n_wall_layer=int(layer.sum()),
               n_interior=int((~layer).sum()), n_touching=int(touching_wall(foam).sum()))
    for name, mask in (("wall", layer), ("interior", ~layer)):
        if mask.any():
            out[f"z_{name}"] = float(contacts[mask].mean())
            out[f"shape_{name}"] = float(shape[mask].mean())
            out[f"area_{name}"] = float(foam.f_area[mask].mean())
        else:
            out[f"z_{name}"] = out[f"shape_{name}"] = out[f"area_{name}"] = np.nan
    return out


class DynamicsRecorder:
    """Motion and neighbour exchange, split into wall layer and interior.

    Use as the ``observer`` of ``protocol.evolve``.  Zones are assigned from the
    centroids at the first sample.  Displacements are in the frame of the fixed
    wall.  A neighbour change is a cell pair that is in contact in one sample
    and not in the next (or the reverse); each change counts once for each of
    the two cells.
    """

    def __init__(self, foam, depth=1.0):
        self.zone_wall = wall_layer(foam, depth)
        self.A0 = foam.A0
        self.times = [0.0]
        self.centroids = [cell_centroids(foam)]
        self.pairs = neighbor_pairs(foam)
        self.changes = np.zeros(foam.nFa)
        self.loads = []

    def __call__(self, time, foam):
        self.times.append(float(time))
        self.centroids.append(cell_centroids(foam))
        pairs = neighbor_pairs(foam)
        for a, b in pairs ^ self.pairs:
            self.changes[a] += 1
            self.changes[b] += 1
        self.pairs = pairs
        if foam.wall is not None:
            self.loads.append(wall_load(foam.wall)["total"])

    def results(self, n_lags=12):
        times = np.array(self.times)
        track = np.array(self.centroids)
        duration = times[-1] - times[0]
        lags = np.unique(np.round(np.geomspace(1, max(1, len(times) - 1), n_lags)).astype(int))
        out = dict(duration=float(duration), lag_time=(times[1] - times[0]) * lags,
                   wall_load_mean=float(np.mean(self.loads)) if self.loads else np.nan)
        for name, mask in (("wall", self.zone_wall), ("interior", ~self.zone_wall)):
            out[f"n_{name}"] = int(mask.sum())
            if not mask.any():
                out[f"msd_{name}"] = np.full(len(lags), np.nan)
                out[f"exchange_rate_{name}"] = np.nan
                continue
            out[f"msd_{name}"] = np.array([
                np.mean(np.sum((track[lag:, mask] - track[:-lag, mask]) ** 2, axis=2)) / self.A0
                for lag in lags])
            out[f"exchange_rate_{name}"] = float(self.changes[mask].mean() / duration)
        return out

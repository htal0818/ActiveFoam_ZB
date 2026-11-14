"""State classification, snapshot and rollback for ``DiscFoam``.

Every instance attribute belongs to exactly one class:

* PARAMETERS   constants of the run;
* PHYSICAL     the simulation state, restored exactly on rollback (the random
               generator is physical state too, handled through its bit-generator);
* CACHES       values derived from the physical state.  ``STATE_CACHES`` are
               returned to their pre-attempt values, ``KEYED_CACHES`` are cleared
               and rebuilt on demand;
* DIAGNOSTICS  counters, wall records and logs.  Never restored, so a rejected
               attempt stays visible.

Snapshot and restore are generated from these tables rather than from a
hand-written field list.  ``unclassified`` reports any attribute the tables do
not know about, and the tests require it to be empty.
"""
import copy

import numpy as np
from numba.typed import List

from activefoam._kernels import copy_polylines

PARAMETERS = frozenset({
    "accelerated", "reference_pressure_lag", "pressure_normals", "bs", "sstn", "rho", "psi",
    "lsc", "A0", "P0", "w", "beta", "gam", "tauT", "dt", "edpc", "nFa", "radius", "center",
    "wall_tol", "delta_max", "audit", "remesh", "geometry_policy", "_measure_vertex_force",
})
PHYSICAL = ("vpos", "v_e", "v_f", "e_v", "e_f", "e_t", "e_rfn", "e_mid", "f_v", "f_e",
            "f_area", "f_center", "_pressure_geometry")
RNG = "rng"
STATE_CACHES = ("_t4_boxes", "_prev_lens", "_adjacent_revision")
KEYED_CACHES = ("_fast_faces", "_fast_face_key", "_validation_vertices", "_validation_vertex_key")
DIAGNOSTICS = frozenset({
    "wall", "log", "audit_calls", "_quality_checks", "_microfold_checks",
    "_resolved_crossing_checks", "_max_crossing_depth", "_max_vertex_force",
})
_ABSENT = object()


def unclassified(foam):
    """Instance attributes that no table accounts for."""
    known = (PARAMETERS | set(PHYSICAL) | {RNG} | set(STATE_CACHES) | set(KEYED_CACHES)
             | DIAGNOSTICS)
    return sorted(set(vars(foam)) - known)


def _copy(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, List):
        return copy_polylines(value)
    if isinstance(value, list):
        return [_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy(item) for item in value)
    if isinstance(value, dict):
        return {key: _copy(item) for key, item in value.items()}
    return copy.copy(value)


def _same(a, b):
    if a is _ABSENT or b is _ABSENT:
        return a is b
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                and a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b))
    if isinstance(a, (List, list, tuple)) and isinstance(b, (List, list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
        return bool(a == b)
    return type(a) is type(b) and a == b


class Snapshot:
    """Physical state, generator state and state-derived caches at one instant."""
    __slots__ = ("physical", "rng", "caches")

    def __init__(self, foam):
        self.physical = {name: _copy(getattr(foam, name)) for name in PHYSICAL}
        self.rng = copy.deepcopy(foam.rng.bit_generator.state)
        self.caches = {name: _copy(getattr(foam, name, _ABSENT)) for name in STATE_CACHES}


def snapshot(foam):
    return Snapshot(foam)


def restore(foam, snap):
    """Return ``foam`` to ``snap``.  The snapshot stays valid for reuse."""
    for name, value in snap.physical.items():
        setattr(foam, name, _copy(value))
    foam.rng.bit_generator.state = copy.deepcopy(snap.rng)
    for name, value in snap.caches.items():
        if value is _ABSENT:
            if hasattr(foam, name):
                delattr(foam, name)
        else:
            setattr(foam, name, _copy(value))
    for name in KEYED_CACHES:
        if hasattr(foam, name):
            delattr(foam, name)


def differences(foam, snap):
    """Names of physical fields, generator state or state caches that differ from ``snap``."""
    changed = [name for name in PHYSICAL if not _same(getattr(foam, name), snap.physical[name])]
    if foam.rng.bit_generator.state != snap.rng:
        changed.append(RNG)
    changed += [name for name in STATE_CACHES
                if not _same(getattr(foam, name, _ABSENT), snap.caches[name])]
    return changed

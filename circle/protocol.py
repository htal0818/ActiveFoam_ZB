"""Preparation, evolution and validation for the disc and for matched periodic runs.

The sequence mirrors ``activefoam.foam_measure`` (confluent relaxation, space
seeding at every junction, anneal, quench) so that a disc run and a periodic
run differ only in the domain.  Both use boundary-orientation pressure normals
and current-step areas; the planar defaults are left alone, which is why the
periodic runs are built here and not through ``foam_measure.initialize``.
"""
import numpy as np

from activefoam import foam_measure
from activefoam.foam_full import FoamTissue
from activefoam.topology import build_periodic_voronoi

from . import state
from .disc import DiscFoam
from .kernels import contained_cells, film_crossings

SHORT_EDGE = 0.001 * 2 * np.sqrt(np.pi)
FOLD_DEPTH = 0.01 * 2 * np.sqrt(np.pi)


# --------------------------------------------------------------------------- #
#  validation
# --------------------------------------------------------------------------- #
def validate(foam, strict=True):
    """Geometry checks for a ``DiscFoam``; returns a dictionary of diagnostics.

    The packing fraction is a necessary check only: overlapping cells can sum to
    less than the disc area.  Overlap is tested explicitly through per-cell
    self-crossings (planar report), crossings between films that share no
    junction, and cells nested inside other cells.
    """
    foam.geometry_policy = "strict" if strict else "reference"
    foam_measure.validate(foam)                 # finite, positive areas, self-crossings
    radius = foam.max_radius()
    if not radius <= foam.radius * (1 + foam.REJECT_TOL):
        raise ValueError(f"node outside the wall by {radius - foam.radius:.3g}")
    packing = foam.volume_fraction()
    if packing > 1 + 1e-9:
        raise ValueError(f"cell area exceeds the disc area: packing fraction {packing:.6g}")
    mids, loops = foam._fast_geometry_inputs()
    crossings = int(film_crossings(mids, foam.e_v))
    nested = int(contained_cells(mids, loops, foam.e_f))
    if strict and nested:
        raise ValueError(f"{nested} cell(s) nested inside another cell")
    return dict(packing=packing, max_radius=radius, film_crossings=crossings, nested=nested,
                **foam_measure.quality(foam))


def evolve(foam, duration, mu, start=0, sample_every=0, observer=None, check_every=100,
           strict=True):
    """Advance by ``duration`` (units of tau_R) on the planar transition cadence."""
    count = int(round(duration / foam.dt))
    check = validate if isinstance(foam, DiscFoam) else (lambda f, strict=True: foam_measure.validate(f))
    k = 0
    try:
        for k in range(1, count + 1):
            foam.advance_scheduled(start + k, mu)
            if check_every and k % check_every == 0:
                check(foam, strict)
            if observer is not None and sample_every and k % sample_every == 0:
                observer(k * foam.dt, foam)
        check(foam, strict)
    except ValueError as exc:
        raise ValueError(f"{exc} at elapsed time {k * foam.dt:g} tau_R") from exc
    return start + count


# --------------------------------------------------------------------------- #
#  preparation
# --------------------------------------------------------------------------- #
def relax_confluent(foam, steps=1500):
    """Confluent relaxation with T1 on short, shrinking contacts (cf. ``relaxed_tiling``)."""
    previous = foam.edge_lengths()
    scale = 10
    for iteration in range(1, steps + 1):
        foam.step(0.0)
        if iteration % 5:
            continue
        lengths = foam.edge_lengths()
        short = np.flatnonzero(lengths < SHORT_EDGE * scale)
        if not short.size:
            scale = min(scale + 1, 90)
        for e in short:
            if e >= len(foam.e_f) or e >= len(previous):
                continue
            a, b = foam.e_f[e]
            if a < 0 or b < 0:
                continue                       # outer films of the disc are not contacts
            if min(len(foam.f_v[a]), len(foam.f_v[b])) > 3 and lengths[e] < previous[e]:
                foam.t1_foam(int(e))
        previous = foam.edge_lengths()
    return foam


def _seed_spaces(foam):
    for v in range(len(foam.vpos)):
        foam.t2_reverse(v)


def initialize(n_cells, w, rho, seed, anneal_mu=0.5, anneal_tauT=1.0, relax_steps=1500,
               strict=True, **kwargs):
    """Disc prepared as in the planar protocol: relax, seed spaces, anneal."""
    foam = DiscFoam.random(n_cells, seed=seed, w=w, rho=rho, **kwargs)
    relax_confluent(foam, relax_steps)
    foam.e_t = np.array([foam.fixed_tension(e) for e in range(len(foam.e_v))])
    _seed_spaces(foam)
    evolve(foam, anneal_tauT * foam.tauT, anneal_mu, strict=strict)
    return foam


def equilibrated(n_cells, w, rho, seed, quench_tauT=3.0, strict=True, **kwargs):
    """Anneal, reset tensions to their fixed points, then relax at zero activity."""
    foam = initialize(n_cells, w, rho, seed, strict=strict, **kwargs)
    foam.e_t = np.array([foam.fixed_tension(e) for e in range(len(foam.e_v))])
    evolve(foam, quench_tauT * foam.tauT, 0.0, start=round(foam.tauT / foam.dt), strict=strict)
    return foam


def periodic_foam(n_side, w, rho, seed, **kwargs):
    """Periodic foam with the disc's pressure convention (for matched comparisons)."""
    tissue = build_periodic_voronoi(n_side, np.random.default_rng(100 + seed))
    return FoamTissue(tissue, w=w, rho=rho, seed=seed, accelerated=True,
                      reference_pressure_lag=False, pressure_normals="boundary", **kwargs)


def periodic_initialize(n_side, w, rho, seed, anneal_mu=0.5, anneal_tauT=1.0,
                        relax_steps=1500, strict=True, **kwargs):
    foam = periodic_foam(n_side, w, rho, seed, **kwargs)
    foam.geometry_policy = "strict" if strict else "reference"
    relax_confluent(foam, relax_steps)
    foam.e_t = np.array([foam.fixed_tension(e) for e in range(len(foam.e_v))])
    _seed_spaces(foam)
    evolve(foam, anneal_tauT * foam.tauT, anneal_mu, strict=strict)
    return foam


def periodic_equilibrated(n_side, w, rho, seed, quench_tauT=3.0, strict=True, **kwargs):
    foam = periodic_initialize(n_side, w, rho, seed, strict=strict, **kwargs)
    foam.e_t = np.array([foam.fixed_tension(e) for e in range(len(foam.e_v))])
    evolve(foam, quench_tauT * foam.tauT, 0.0, start=round(foam.tauT / foam.dt), strict=strict)
    return foam


__all__ = ["validate", "evolve", "relax_confluent", "initialize", "equilibrated",
           "periodic_foam", "periodic_initialize", "periodic_equilibrated", "state"]

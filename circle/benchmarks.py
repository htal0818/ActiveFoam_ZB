"""Wall benchmarks with their numerical parameters stated explicitly.

Each function returns a dictionary holding the time step, film spacing, run
duration, the equilibrium criterion and whether it was met, and the measured
errors.  The tests assert on these dictionaries and
``scripts/wall_benchmarks.py`` writes them to ``BENCHMARKS.md``.

Equilibrium criterion for single cells (``relax``): total cell area and
integrated wall reaction each change by less than a relative tolerance between
two checks ``CHECK_TIME`` apart.  The residual nodal force is reported as well
but is not the criterion: film nodes on a frictionless wall keep sliding along
it under forces of third order in the segment angle, long after the shape and
the traction have converged.

All runs are at zero activity (``mu = 0``) and therefore deterministic, except
``resampling_effect_active``.
"""
import numpy as np
from scipy.optimize import brentq

from . import state
from .disc import DiscFoam, circle_polyline
from .measure import wall_load, wall_traction

DT = 0.005
SPACINGS = (0.15, 0.075, 0.0375)
CHECK_TIME = 5.0          # tau_R between equilibrium checks
AREA_TOL = 1e-9           # relative change of total cell area per check
LOAD_TOL = 1e-7           # relative change of integrated wall reaction per check
CRITERION = (f"area and wall reaction change by less than {AREA_TOL:g} and {LOAD_TOL:g} "
             f"(relative) over {CHECK_TIME:g} tau_R")


def free_area(rho, pressure=10.0, tension=1.0):
    """Area of an isolated cell: p_n = T_eff/r with p_n = P0 (rho/A - 1)/sqrt(rho)."""
    radius = brentq(lambda r: pressure * (rho / (np.pi * r * r) - 1) / np.sqrt(rho) - tension / r,
                    1e-2, 5.0)
    return np.pi * radius ** 2


def relax(foam, max_time=400.0):
    """Zero-activity steps until the equilibrium criterion ``CRITERION`` is met.

    Returns elapsed time, whether the criterion was met, the final residual
    nodal force, the number of film resampling events, the number of steps in
    which a node arrived at the wall, and the lowest instantaneous wall reaction
    relative to its final value after first contact.
    """
    steps = int(round(max_time / foam.dt))
    every = int(round(CHECK_TIME / foam.dt))
    resampled = arrival_steps = 0
    previous = None
    converged = False
    k = 0
    for k in range(1, steps + 1):
        foam.step(0.0)
        wall = foam.wall
        resampled += wall.resampled
        load = wall_load(wall)
        arrival_steps += load["arrivals"] > 0
        if k % every == 0:
            current = (float(foam.f_area.sum()), load["total"])
            if previous is not None:
                area_change = abs(current[0] - previous[0]) / current[0]
                scale = max(current[1], previous[1])
                load_change = abs(current[1] - previous[1]) / scale if scale > 0 else 0.0
                if area_change < AREA_TOL and load_change < LOAD_TOL:
                    converged = True
                    break
            previous = current
    return dict(time=k * foam.dt, steps=k, converged=converged,
                residual=float(foam.wall.residual), resample_events=int(resampled),
                arrival_steps=int(arrival_steps))


def _parameters(foam, criterion=CRITERION):
    return dict(dt=foam.dt, spacing=foam.edpc, remesh=foam.remesh, criterion=criterion)


def free_cell(spacing=0.15, dt=DT, remesh=True):
    """Isolated cell far from the wall: planar free-cell state and no wall reaction."""
    center = np.array([12.0, 12.0])
    ring = circle_polyline(center + [0.4, -0.3], 0.6, spacing)
    foam = DiscFoam.from_closed_cells(3.0, [ring], center=center, dt=dt, edpc=spacing,
                                      remesh=remesh)
    run = relax(foam)
    reference = free_area(foam.rho)
    return dict(name="free cell", **_parameters(foam), **run,
                wall_reaction=wall_load(foam.wall)["total"], area=float(foam.f_area[0]),
                reference_area=reference, area_error=float(foam.f_area[0] / reference - 1),
                nodes=len(foam.e_mid[0]) - 1, max_radius=foam.max_radius() / foam.radius)


def squeezed_cell(spacing=0.15, dt=DT, remesh=False, start="on_wall", rho=1.5):
    """One cell larger than the disc: integrated wall traction against the reference.

    Reference (code units): ``2 pi R_w (p_n - T_eff/R_w)`` with
    ``p_n = P0 (rho/A - 1)/sqrt(rho)``, ``T_eff = max(T, 0)`` and ``A`` the cell's
    polygon area.  ``start="on_wall"`` begins with every node on the wall;
    ``start="grow"`` begins with a smaller off-centre cell that expands, reaches
    the wall on one side first and slides to fill the disc.
    """
    radius = np.sqrt(1.0 / np.pi)
    center = np.array([4 * radius, 4 * radius])
    if start == "on_wall":
        ring = circle_polyline(center, radius, spacing)
    elif start == "grow":
        ring = circle_polyline(center + [0.2 * radius, 0.1 * radius], 0.7 * radius, spacing)
    else:
        raise ValueError(start)
    foam = DiscFoam.from_closed_cells(radius, [ring], rho=rho, dt=dt, edpc=spacing,
                                      remesh=remesh)
    run = relax(foam)
    load = wall_load(foam.wall)
    p_n = foam.P0 * (foam.rho / foam.f_area[0] - 1) / np.sqrt(foam.rho)
    t_eff = max(float(foam.e_t[0]), 0.0)
    reference = 2 * np.pi * foam.radius * (p_n - t_eff / foam.radius)
    angle, traction, tributary = wall_traction(foam)
    lengths = np.hypot(*np.diff(np.asarray(foam.e_mid[0]), axis=0).T)
    return dict(name="squeezed cell", start=start, **_parameters(foam), **run,
                p_n=float(p_n), t_eff=t_eff, reference=float(reference),
                total=load["total"], traction_error=float(load["total"] / reference - 1),
                direct=load["direct"],
                projection_vs_constraint=float(abs(load["total"] - load["direct"])
                                               / load["direct"]),
                resultant=float(np.hypot(*load["resultant"]) / load["total"]),
                torque=float(abs(load["torque"]) / (load["total"] * foam.radius)),
                traction_mean=float(np.average(traction, weights=tributary)),
                traction_reference=float(p_n - t_eff / foam.radius),
                traction_spread=float(traction.std() / traction.mean()),
                tributary_sum=float(tributary.sum()), nodes=len(lengths),
                spacing_ratio=float(lengths.max() / lengths.min()),
                packing=foam.volume_fraction(),
                chord_coefficient=float((1 - foam.volume_fraction())
                                        / (lengths.mean() / foam.radius) ** 2))


def detaching_cell(spacing=0.1, dt=DT, remesh=True, rho=0.5):
    """A cell that starts on the wall but prefers a smaller area leaves it."""
    radius = np.sqrt(1.0 / np.pi)
    foam = DiscFoam.from_closed_cells(
        radius, [circle_polyline((4 * radius, 4 * radius), radius, spacing)],
        rho=rho, dt=dt, edpc=spacing, remesh=remesh)
    foam.step(0.0)
    first = wall_load(foam.wall)
    run = relax(foam)
    reference = free_area(rho)
    return dict(name="detaching cell", **_parameters(foam), **run,
                first_step_reaction=first["total"], first_step_contacts=first["contacts"],
                final_contacts=wall_load(foam.wall)["contacts"],
                max_radius=foam.max_radius() / foam.radius, area=float(foam.f_area[0]),
                reference_area=reference, area_error=float(foam.f_area[0] / reference - 1))


def projection_vs_constraint(dts=(0.005, 0.0025, 0.00125), n_cells=16, seed=1, w=1.0,
                             spacing=0.15, settle_steps=400):
    """Fixed mesh, one step from one state: projection estimate against max(0, f.n)."""
    foam = DiscFoam.random(n_cells, seed=seed, w=w, edpc=spacing)
    for _ in range(settle_steps):
        foam.step(0.0)
    foam.remesh = False
    for _ in range(5):                      # every wall node becomes a start-of-step contact
        foam.step(0.0)
    start = state.snapshot(foam)
    rows, direct = [], None
    for dt in dts:
        state.restore(foam, start)
        foam.dt = dt
        foam.step(0.0)
        wall = foam.wall
        contact = wall.node_flag == 1
        estimate = np.linalg.norm(wall.node_reaction[contact], axis=1)
        constraint = wall.node_direct[contact]
        same = direct is None or np.array_equal(constraint, direct)
        direct = constraint if direct is None else direct
        rows.append(dict(dt=dt, contacts=int(contact.sum()),
                         max_error=float(np.abs(estimate - constraint).max()),
                         relative_error=float(np.abs(estimate - constraint).sum()
                                              / constraint.sum()),
                         constraint_unchanged=bool(same)))
    return dict(name="projection vs constraint", spacing=spacing, remesh=False,
                criterion="single step from a common state", rows=rows)


def three_cell_refinement(scales=(1, 2, 4), duration=150.0, w=1.0, seed=0,
                          wall_resample=False):
    """Three-cell disc, remeshing on, joint refinement of dt and spacing.

    The cells touch the wall over short arcs, so this tests partial contact.
    Traction is averaged over the last fifth of the run.  Equilibrium criterion:
    the means of the two halves of that window differ by less than 2e-3 of the
    mean.  ``intermittency`` is the standard deviation over the window divided
    by the mean.  ``transient_dip`` is the lowest instantaneous reaction over
    its running mean during the interval 15-30 tau_R, when films are still
    being resampled: contact lost by freshly interpolated nodes shows up there.
    """
    rows = []
    for scale in scales:
        foam = DiscFoam.random(3, seed=seed, w=w, dt=DT / scale, edpc=0.15 / scale,
                               wall_resample=wall_resample)
        steps = int(round(duration / foam.dt))
        window = steps // 5
        early = (int(round(15.0 / foam.dt)), int(round(30.0 / foam.dt)))
        loads, transient, resampled, arrival_steps = [], [], 0, 0
        for k in range(steps):
            foam.step(0.0)
            resampled += foam.wall.resampled
            load = wall_load(foam.wall)
            arrival_steps += load["arrivals"] > 0
            if early[0] <= k < early[1]:
                transient.append(load["total"])
            if k >= steps - window:
                loads.append(load["total"])
        loads, transient = np.array(loads), np.array(transient)
        half = len(loads) // 2
        drift = abs(loads[:half].mean() - loads[half:].mean()) / loads.mean()
        rows.append(dict(dt=foam.dt, spacing=foam.edpc, duration=duration,
                         window=window * foam.dt, mean=float(loads.mean()),
                         intermittency=float(loads.std() / loads.mean()), drift=float(drift),
                         converged=bool(drift < 2e-3), residual=float(foam.wall.residual),
                         transient_dip=float(transient.min() / transient.mean()),
                         resample_events=int(resampled), arrival_steps=int(arrival_steps),
                         contacts=wall_load(foam.wall)["contacts"],
                         packing=foam.volume_fraction()))
    for coarse, fine in zip(rows, rows[1:]):
        fine["change_from_coarser"] = abs(fine["mean"] - coarse["mean"]) / fine["mean"]
    return dict(name="three cells, remeshing on", remesh=True, wall_resample=wall_resample,
                criterion="window halves agree to 2e-3", rows=rows)


def resampling_effect_active(n_cells=16, w=1.0, rho=1.3, mu=1.0, duration=150.0,
                             seeds=(1, 2, 3)):
    """Time-averaged wall reaction of an active tissue, plain vs wall-aware resampling."""
    rows = []
    for wall_resample in (False, True):
        means, arrivals = [], []
        for seed in seeds:
            foam = DiscFoam.random(n_cells, seed=seed, w=w, rho=rho,
                                   wall_resample=wall_resample)
            steps = int(round(duration / foam.dt))
            loads, arrived = [], 0
            for k in range(1, steps + 1):
                foam.advance_scheduled(k, mu)
                if k > steps // 5:
                    load = wall_load(foam.wall)
                    loads.append(load["total"])
                    arrived += load["arrivals"]
            means.append(float(np.mean(loads)))
            arrivals.append(arrived / len(loads))
        rows.append(dict(wall_resample=wall_resample, dt=DT, spacing=0.15, duration=duration,
                         seeds=len(seeds), mean=float(np.mean(means)),
                         standard_error=float(np.std(means, ddof=1) / np.sqrt(len(means))),
                         arrivals_per_step=float(np.mean(arrivals))))
    return dict(name="active tissue, time-averaged reaction", n_cells=n_cells, w=w, rho=rho,
                mu=mu, criterion="first fifth of the run discarded", rows=rows)


def chord_deficit(spacings=(0.2, 0.1, 0.05), n_cells=36, seed=2):
    """Area left between the polyline outer films and the wall at full coverage."""
    rows = []
    for spacing in spacings:
        foam = DiscFoam.random(n_cells, seed=seed, edpc=spacing)
        deficit = 1.0 - foam.volume_fraction()
        rows.append(dict(spacing=spacing, deficit=float(deficit),
                         coefficient=float(deficit / (spacing / foam.radius) ** 2)))
    return dict(name="chord-area deficit", leading_order_coefficient=1 / 6, rows=rows)

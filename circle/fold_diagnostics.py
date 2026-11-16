"""Mechanical diagnostics of epithelial-ring runs, computed from saved trajectories.

Read-only analysis of the files written by ``scripts/bending_experiments.py``
(``history`` every 0.5 tau_R, ``frames`` every 5 tau_R).  Nothing here touches
the dynamics.  Conventions:

* cells are indexed counter-clockwise; the vegetal pole is at polar angle -90
  degrees and the activated domain is centred on it;
* the apical surface is the outer free boundary, the basal surface faces the
  lumen;
* the midline is traversed counter-clockwise, so the undeformed ring has
  positive signed curvature and an inward (invaginated) bend is negative;
* ``inward`` normals point to the left of the counter-clockwise tangent, i.e.
  toward the lumen.
"""
import numpy as np
from scipy.spatial import cKDTree

from .bending import boundary_path, compare_midlines, curvature, path_midpoint, polygon_geometry, smooth_ramp

P0, A0, TAU_T = 10.0, 1.0, 10.0          # the ring runs use rho = 1, so psi = lsc = 1


# --------------------------------------------------------------------------- #
#  indexing
# --------------------------------------------------------------------------- #
def vegetal_order(n=48, cells=24):
    """Indices of the ``cells`` profiles around the vegetal pole, ordered across the domain."""
    theta = 2 * np.pi * (np.arange(n) + .5) / n
    delta = np.angle(np.exp(1j * (theta + np.pi / 2)))
    selected = np.flatnonzero(np.abs(delta) < np.pi * cells / n)
    return selected[np.argsort(delta[selected])]


def at_time(history, elapsed):
    """History row at a given elapsed time (exact match on the saved sampling)."""
    row = min(history, key=lambda m: abs(m['elapsed'] - elapsed))
    if abs(row['elapsed'] - elapsed) > 1e-6:
        raise KeyError(f'no sample at elapsed time {elapsed}')
    return row


# --------------------------------------------------------------------------- #
#  curvature of the midline
# --------------------------------------------------------------------------- #
def turning_angles(points):
    """Signed exterior angle at each midline point; they sum to 2 pi for a simple closed curve."""
    p = np.asarray(points, float)
    a = p - np.roll(p, 1, axis=0)
    b = np.roll(p, -1, axis=0) - p
    return np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0], np.sum(a * b, axis=1))


def arc_weights(points):
    """Midline length attributed to each point (half of the two adjacent segments)."""
    p = np.asarray(points, float)
    forward = np.linalg.norm(np.roll(p, -1, axis=0) - p, axis=1)
    return .5 * (forward + np.roll(forward, 1))


def curvature_summary(row, control, domain, spans=(1, 2, 3, 4)):
    """Curvature of the midline at several smoothing spans, against the matched control.

    ``span = k`` uses the circumcircle through points k cells apart, so larger
    spans average over polyline-scale noise.  A feature is treated as
    persistent only if it appears at every span.
    """
    points, reference = np.asarray(row['midline']), np.asarray(control['midline'])
    weights = arc_weights(points)
    turn, turn0 = turning_angles(points), turning_angles(reference)
    out = dict(total_turning=float(turn.sum()),
               domain_turning=float(turn[domain].sum()),
               control_domain_turning=float(turn0[domain].sum()),
               domain_turning_change=float(turn[domain].sum() - turn0[domain].sum()),
               spans={})
    negative_everywhere = np.ones(len(points), bool)
    for span in spans:
        k, k0 = curvature(points, span), curvature(reference, span)
        negative = k < 0
        negative_everywhere &= negative
        out['spans'][span] = dict(
            minimum=float(k.min()), maximum=float(k.max()),
            argmin=int(np.argmin(k)), argmax=int(np.argmax(k)),
            negative_profiles=int(negative.sum()),
            negative_arc_length=float(weights[negative].sum()),
            sign_changes_from_control=int(np.sum((k < 0) & (k0 > 0))),
            domain_mean=float(np.average(k[domain], weights=weights[domain])),
            control_domain_mean=float(np.average(k0[domain], weights=arc_weights(reference)[domain])))
    out['negative_at_every_span'] = np.flatnonzero(negative_everywhere).tolist()
    run = longest = 0
    for flag in np.r_[negative_everywhere, negative_everywhere]:     # allow wrap-around
        run = run + 1 if flag else 0
        longest = max(longest, run)
    out['longest_negative_run'] = int(min(longest, len(points)))
    return out


# --------------------------------------------------------------------------- #
#  fold depth and radial response
# --------------------------------------------------------------------------- #
def sagitta(points, lumen_center, cells):
    """Distance of the centre of ``cells`` beyond the chord joining its two end cells.

    Positive when the centre bulges away from the lumen, as on the undeformed
    ring; it decreases, and can change sign, as the stretch folds inward.
    """
    p = np.asarray(points)[cells]
    start, end = p[0], p[-1]
    middle = .5 * (p[len(p) // 2 - 1] + p[len(p) // 2]) if len(p) % 2 == 0 else p[len(p) // 2]
    chord = end - start
    normal = np.array([chord[1], -chord[0]]) / np.linalg.norm(chord)
    if np.dot(normal, .5 * (start + end) - np.asarray(lumen_center)) < 0:
        normal = -normal                                      # point away from the lumen
    return float(np.dot(middle - start, normal)), float(np.linalg.norm(chord))


def fold_depth(row, control, domain, patch=8):
    """Inward depth relative to the shoulders, with the control's sagitta removed.

    ``domain_depth`` uses the two end cells of the activated domain as
    shoulders; ``patch_depth`` uses the ends of the central ``patch`` cells and
    so isolates bending localised at the vegetal centre.  Positive = inward.
    """
    centre = len(domain) // 2
    local = domain[centre - patch // 2:centre + patch // 2]
    out = {}
    for name, cells in (('domain', domain), ('patch', local)):
        s, chord = sagitta(row['midline'], row['lumen_center'], cells)
        s0, chord0 = sagitta(control['midline'], control['lumen_center'], cells)
        out[f'{name}_depth'] = s0 - s
        out[f'{name}_sagitta'] = s
        out[f'{name}_shoulder_separation'] = chord
        out[f'{name}_shoulder_separation_change'] = chord - chord0
    return out


def radial_response(row, control, domain_cells=24):
    """Radial displacement of the midline against the control, split into uniform and local parts."""
    result = compare_midlines(row, control, domain_cells)
    if not result['radial_profile_valid']:
        return dict(valid=False)
    theta, residual = np.asarray(result['theta']), np.asarray(result['radial_residual'])
    vegetal = np.cos(theta + np.pi / 2) > 0                    # the half centred on -90 degrees
    spectrum = np.abs(np.fft.rfft(residual)) * 2 / len(residual)
    return dict(valid=True, theta=theta, total=np.asarray(result['radial_difference']),
                residual=residual, uniform=result['uniform_radial_difference'],
                residual_rms=float(np.sqrt(np.mean(residual ** 2))),
                vegetal_mean=float(residual[vegetal].mean()),
                animal_mean=float(residual[~vegetal].mean()),
                pole_value=float(np.interp(-np.pi / 2 % (2 * np.pi), theta, residual,
                                           period=2 * np.pi)),
                mode_amplitudes=spectrum[1:7].tolist())


# --------------------------------------------------------------------------- #
#  cell shapes
# --------------------------------------------------------------------------- #
def cell_changes(row, control):
    """Relative change of each cell's apical and basal length, height, area and wedging."""
    rel = lambda key: np.asarray(row[key]) / np.asarray(control[key]) - 1
    return dict(apical=rel('apical'), basal=rel('basal'), height=rel('height'), area=rel('area'),
                wedge=np.asarray(row['wedge_ratio']),
                wedge_control=np.asarray(control['wedge_ratio']),
                wedge_change=np.asarray(row['wedge_ratio']) - np.asarray(control['wedge_ratio']))


def classify_cells(changes, tolerance=0.02):
    """Sort each cell's response by which of its two free surfaces changed length.

    Apical shortening alone is not wedging, and wedging alone is not a fold;
    this only says what each cell did.
    """
    apical, basal = changes['apical'], changes['basal']
    labels = np.full(len(apical), 'little change', dtype='<U40')
    short_a, short_b, long_b = apical < -tolerance, basal < -tolerance, basal > tolerance
    labels[short_a & ~short_b & ~long_b] = 'apical shorter, basal unchanged'
    labels[short_a & short_b] = 'apical and basal shorter'
    labels[short_a & long_b] = 'apical shorter, basal longer'
    labels[~short_a & short_b] = 'basal shorter only'
    labels[~short_a & long_b] = 'basal longer only'
    return labels


def frame_cells(frame):
    """Per-cell geometry that needs the full polygons: axis tilt, anisotropy, apical curvature."""
    apical_tree, basal_tree = cKDTree(frame['outer']), cKDTree(frame['inner'])
    lumen = polygon_geometry(frame['inner'])[1]
    out = dict(tilt=[], anisotropy=[], apical_paths=[], basal_paths=[], apical_mid=[], basal_mid=[],
               area=[], apical_turning=[])
    for polygon in frame['polygons']:
        polygon = np.asarray(polygon)
        apical, basal = boundary_path(polygon, apical_tree), boundary_path(polygon, basal_tree)
        ca, cb = path_midpoint(apical)[1], path_midpoint(basal)[1]
        axis = (ca - cb) / np.linalg.norm(ca - cb)
        radial = .5 * (ca + cb) - lumen
        radial /= np.linalg.norm(radial)
        out['tilt'].append(float(np.arctan2(radial[0] * axis[1] - radial[1] * axis[0],
                                            np.dot(radial, axis))))
        area, centroid = polygon_geometry(polygon)
        q = polygon - centroid
        nxt = np.roll(q, -1, axis=0)
        cross = q[:, 0] * nxt[:, 1] - q[:, 1] * nxt[:, 0]
        ixx = np.sum((q[:, 1] ** 2 + q[:, 1] * nxt[:, 1] + nxt[:, 1] ** 2) * cross) / 12
        iyy = np.sum((q[:, 0] ** 2 + q[:, 0] * nxt[:, 0] + nxt[:, 0] ** 2) * cross) / 12
        ixy = np.sum((q[:, 0] * nxt[:, 1] + 2 * q[:, 0] * q[:, 1] + 2 * nxt[:, 0] * nxt[:, 1]
                      + nxt[:, 0] * q[:, 1]) * cross) / 24
        eig = np.linalg.eigvalsh(np.array([[iyy, ixy], [ixy, ixx]]) * np.sign(area))
        out['anisotropy'].append(float(np.sqrt(eig[1] / eig[0])))
        segments = np.diff(apical, axis=0)
        angles = np.arctan2(segments[:, 1], segments[:, 0])
        out['apical_turning'].append(float(np.sum(np.angle(np.exp(1j * np.diff(angles))))))
        out['apical_paths'].append(apical); out['basal_paths'].append(basal)
        out['apical_mid'].append(ca); out['basal_mid'].append(cb); out['area'].append(area)
    for key in ('tilt', 'anisotropy', 'apical_mid', 'basal_mid', 'area', 'apical_turning'):
        out[key] = np.asarray(out[key])
    out['lumen_center'] = lumen
    return out


# --------------------------------------------------------------------------- #
#  lumen, pressures, wall
# --------------------------------------------------------------------------- #
def cell_pressure(area):
    """The model's osmotic cell pressure for these runs: P0 (A0/A - 1), with psi = 1."""
    return P0 * (A0 / np.asarray(area) - 1)


def wall_clearance(frame):
    """Smallest gap between the tissue's outer surface and the rigid wall, and contacts."""
    radius = np.linalg.norm(np.asarray(frame['outer']) - np.asarray(frame['center']), axis=1)
    gap = frame['wall_radius'] - radius
    return dict(min_gap=float(gap.min()), contact_nodes=int(np.sum(gap < 1e-9 * frame['wall_radius'])))


def lumen_accounting(row, control, first):
    """Where the deformation went: lumen area, cell area, and the mean radial shift."""
    return dict(lumen_area=row['lumen_area'], lumen_pressure=row['lumen_pressure'],
                lumen_area_change=row['lumen_area'] / first['lumen_area'] - 1,
                lumen_area_vs_control=row['lumen_area'] / control['lumen_area'] - 1,
                lumen_pressure_vs_control=row['lumen_pressure'] - control['lumen_pressure'],
                cell_area_change=float(np.sum(row['area']) / np.sum(first['area']) - 1),
                cell_area_vs_control=float(np.sum(row['area']) / np.sum(control['area']) - 1),
                mean_cell_pressure=float(cell_pressure(row['area']).mean()),
                cell_pressure_vs_control=cell_pressure(row['area']) - cell_pressure(control['area']))


# --------------------------------------------------------------------------- #
#  tensions: reconstructed, because they do not depend on geometry
# --------------------------------------------------------------------------- #
def apical_tension(parameters, weights, elapsed, base=1.0):
    """Actual and target apical tension of every cell at the given elapsed times.

    With zero noise the OU update is ``T <- T - dt/tau_T (T - target)``, and the
    target depends only on time and on the cell's activation weight.  Replaying
    that recursion reproduces the simulated tensions without the geometry.
    """
    dt, weights = parameters['dt'], np.asarray(weights, float)
    steps = np.rint(np.asarray(elapsed, float) / dt).astype(int)
    tension = np.full(len(weights), base)
    actual, target = [], []
    wanted = set(steps.tolist())
    for k in range(int(steps.max()) + 1):
        goal = base + parameters['apical'] * smooth_ramp(k * dt, parameters['ramp']) * weights
        if k in wanted:
            actual.append(tension.copy()); target.append(goal)
        tension = tension - dt / TAU_T * (tension - goal)
    index = {k: i for i, k in enumerate(sorted(wanted))}
    order = [index[k] for k in steps]
    return np.array(actual)[order], np.array(target)[order]


# --------------------------------------------------------------------------- #
#  incremental force of the activation (Experiment II)
# --------------------------------------------------------------------------- #
def incremental_force(frame, increment):
    """Force added by the extra apical tension, evaluated on the saved geometry.

    Film forces are linear in tension, so the difference between the activated
    and the control force field on the *same* geometry is the film force of the
    tension increment alone.  A tensed film exerts no net force on itself: it
    pulls on its two end junctions and presses along its curvature.  Each cell
    is therefore credited with the force on its interior apical nodes plus half
    of the force on each of its two apical junctions (the other half belongs to
    the neighbour sharing that junction).

    Returns, per cell, the force resolved along the midline tangent
    (counter-clockwise) and the inward normal, and its moment about the cell's
    midline point (positive = counter-clockwise).
    """
    cells = frame_cells(frame)
    n = len(cells['apical_paths'])
    midline = .5 * (cells['apical_mid'] + cells['basal_mid'])
    tangent = np.roll(midline, -1, axis=0) - np.roll(midline, 1, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    inward = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    interior, moment_interior = np.zeros((n, 2)), np.zeros(n)
    start_force, end_force = np.zeros((n, 2)), np.zeros((n, 2))
    starts, ends = np.zeros((n, 2)), np.zeros((n, 2))
    for i, path in enumerate(cells['apical_paths']):
        unit = np.diff(path, axis=0)
        unit /= np.linalg.norm(unit, axis=1)[:, None]
        node_force = increment[i] * (unit[1:] - unit[:-1])              # interior nodes
        interior[i] = node_force.sum(axis=0)
        arm = path[1:-1] - midline[i]
        moment_interior[i] = np.sum(arm[:, 0] * node_force[:, 1] - arm[:, 1] * node_force[:, 0])
        start_force[i], end_force[i] = increment[i] * unit[0], -increment[i] * unit[-1]
        starts[i], ends[i] = path[0], path[-1]
    # Which neighbour shares each end?  Match junction positions instead of assuming an order.
    tree = cKDTree(starts)
    junction_force, junction_moment = np.zeros((n, 2)), np.zeros(n)
    for i in range(n):
        distance, j = tree.query(ends[i])
        if distance > 1e-8:
            raise ValueError('apical films do not form a closed chain')
        total = end_force[i] + start_force[j]                           # force on that junction
        for cell in (i, j):
            junction_force[cell] += .5 * total
            arm = ends[i] - midline[cell]
            junction_moment[cell] += .5 * (arm[0] * total[1] - arm[1] * total[0])
    force = interior + junction_force
    return dict(force=force, tangential=np.sum(force * tangent, axis=1),
                normal_inward=np.sum(force * inward, axis=1),
                moment=moment_interior + junction_moment,
                net_force=force.sum(axis=0), midline=midline, tangent=tangent, inward=inward,
                height=np.linalg.norm(cells['apical_mid'] - cells['basal_mid'], axis=1),
                apical_turning=cells['apical_turning'])


# --------------------------------------------------------------------------- #
#  outcome classes of the plan
# --------------------------------------------------------------------------- #
def classify_run(run, control, domain, patch=8, sustain=25.0):
    """Class 0-4 of the invagination plan, from independently measured geometry.

    0 no significant bending, 1 broad flattening, 2 localised invagination
    initiated, 3 sustained invagination, 4 numerical or topological failure.

    Thresholds (stated here because the plan leaves them open):
    * displacement tolerance: 2% of the mean control cell height, the
      convention of ``compare_midlines``;
    * flattening: the activated domain has lost more than 5% of its turning, or
      its centre has moved inward relative to its shoulders by more than the
      tolerance;
    * localised: at least two adjacent profiles have negative midline curvature
      at every smoothing span, and the central ``patch`` is deeper than the
      tolerance relative to its own shoulders;
    * sustained: localised, the central depth is at least one mean cell height,
      and it was within 5% of that value, or still growing, ``sustain`` tau_R
      earlier.
    A class describes one run at one resolution; it is not a converged result.
    """
    if run['summary']['status'] != 'complete':
        return dict(cls=4, label='numerical or topological failure',
                    reason=run['summary']['reason'])
    history = run['history']
    final = history[-1]
    reference = at_time(control['history'], final['elapsed'])
    height = float(np.mean(reference['height']))
    tolerance = .02 * height
    depth = fold_depth(final, reference, domain, patch)
    curve = curvature_summary(final, reference, domain)
    localized = curve['longest_negative_run'] >= 2 and depth['patch_depth'] > tolerance
    flattened = curve['domain_turning_change'] < -0.05 * curve['control_domain_turning']
    if localized:
        earlier = at_time(history, final['elapsed'] - sustain)
        before = fold_depth(earlier, at_time(control['history'], earlier['elapsed']), domain, patch)
        held = (depth['patch_depth'] >= height and before['patch_depth'] > tolerance
                and depth['patch_depth'] >= .95 * before['patch_depth'])
        cls, label = (3, 'sustained invagination') if held else (2, 'localised invagination initiated')
    elif flattened or depth['domain_depth'] > tolerance:
        cls, label = 1, 'broad flattening'
    else:
        cls, label = 0, 'no significant bending'
    return dict(cls=cls, label=label, tolerance=tolerance, mean_height=height,
                patch_depth=depth['patch_depth'], domain_depth=depth['domain_depth'],
                patch_depth_in_heights=depth['patch_depth'] / height,
                longest_negative_run=curve['longest_negative_run'],
                domain_turning_change=curve['domain_turning_change'])

"""Fixed-topology epithelial bending experiments and geometric diagnostics.

The existing force, pressure and OU update are reused. Only prescribed cortical
mean tensions and the closed-lumen compliance parameters change. No bending
force, displacement constraint or energy functional is introduced.
"""
import numpy as np
from scipy.spatial import cKDTree

from .invagination import PolarFoam, boundary_loops
from . import protocol

T1_LENGTH = .01 * 2 * np.sqrt(np.pi)


def polygon_geometry(points):
    p = np.asarray(points, float)
    origin = p.mean(axis=0)
    q = p-origin
    nxt = np.roll(q, -1, axis=0)
    cross = q[:, 0]*nxt[:, 1]-q[:, 1]*nxt[:, 0]
    area = cross.sum()/2
    return area, origin+np.sum((q+nxt)*cross[:, None], axis=0)/(6*area)


def path_midpoint(points):
    p = np.asarray(points, float)
    lengths = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0., np.cumsum(lengths)]
    if s[-1] <= 0:
        raise ValueError('zero-length interface')
    half = s[-1]/2
    j = min(np.searchsorted(s, half, side='right')-1, len(p)-2)
    return float(s[-1]), p[j]+(half-s[j])/lengths[j]*(p[j+1]-p[j])


def boundary_path(polygon, tree):
    """Recover one free film from a saved cell polygon and boundary node set."""
    p = np.asarray(polygon)
    on = tree.query(p)[0] < 1e-8
    segments = on & np.roll(on, -1)
    starts = np.flatnonzero(segments & ~np.roll(segments, 1))
    if len(starts) != 1:
        raise ValueError('cell must have exactly one contiguous film on this boundary')
    start = int(starts[0]); indices = [start]
    while segments[indices[-1]]:
        indices.append((indices[-1]+1) % len(p))
        if len(indices) > len(p):
            raise ValueError('boundary path did not end')
    return p[indices]


def curvature(points, span=1):
    """Signed circumcircle curvature of ordered CCW midline points.

    span=1 uses adjacent cells; span=2 is a wider spatial diagnostic. Both are
    exactly 1/R for points on a circle; no critical shape index is involved.
    """
    p = np.asarray(points, float)
    a = p-np.roll(p, span, axis=0)
    b = np.roll(p, -span, axis=0)-p
    denominator = np.linalg.norm(a, axis=1)*np.linalg.norm(b, axis=1)*np.linalg.norm(a+b, axis=1)
    return 2*(a[:, 0]*b[:, 1]-a[:, 1]*b[:, 0])/denominator


def measure_frame(frame):
    apical_tree, basal_tree = cKDTree(frame['outer']), cKDTree(frame['inner'])
    names = ('apical', 'basal', 'apical_chord', 'basal_chord', 'height', 'area')
    values = {key: [] for key in names}
    midline = []
    for polygon in frame['polygons']:
        apical = boundary_path(polygon, apical_tree)
        basal = boundary_path(polygon, basal_tree)
        la, ca = path_midpoint(apical); lb, cb = path_midpoint(basal)
        values['apical'].append(la); values['basal'].append(lb)
        values['apical_chord'].append(np.linalg.norm(apical[-1]-apical[0]))
        values['basal_chord'].append(np.linalg.norm(basal[-1]-basal[0]))
        values['height'].append(np.linalg.norm(ca-cb))
        values['area'].append(polygon_geometry(polygon)[0])
        midline.append((ca+cb)/2)
    values = {key: np.asarray(value) for key, value in values.items()}
    midline = np.asarray(midline)
    values.update(time=frame['time'], midline=midline,
                  lumen_center=polygon_geometry(frame['inner'])[1],
                  curvature=curvature(midline), curvature_wide=curvature(midline, 2),
                  wedge_ratio=values['apical']/values['basal'],
                  wedge_chord_ratio=values['apical_chord']/values['basal_chord'],
                  lumen_area=frame['lumen_area'], lumen_pressure=frame['lumen_pressure'])
    return values


def compare_midlines(active, control, patch_cells=8):
    """Center/rotation-aligned radial response with uniform contraction removed.

    Radii are interpolated at equal geometric polar angles, not averaged by
    material cell index. A non-star-shaped contour is reported, not converted
    into a misleading single-valued radial profile.
    """
    a = np.asarray(active['midline'])-active['lumen_center']
    c = np.asarray(control['midline'])-control['lumen_center']
    za, zc = a[:, 0]+1j*a[:, 1], c[:, 0]+1j*c[:, 1]
    phase = np.angle(np.sum(zc*np.conj(za)))
    za *= np.exp(1j*phase)
    angles_a, angles_c = np.unwrap(np.angle(za)), np.unwrap(np.angle(zc))
    star = (np.all(np.diff(np.r_[angles_a, angles_a[0]+2*np.pi]) > 0) and
            np.all(np.diff(np.r_[angles_c, angles_c[0]+2*np.pi]) > 0))
    if not star:
        return dict(radial_profile_valid=False, classification='non_star_shaped', rotation=phase)
    theta = np.linspace(0, 2*np.pi, 384, endpoint=False)
    radial = np.interp(theta, angles_a, np.abs(za), period=2*np.pi)-np.interp(
        theta, angles_c, np.abs(zc), period=2*np.pi)
    uniform = radial.mean(); residual = radial-uniform
    n = len(a); half = np.pi*patch_cells/n
    delta = np.angle(np.exp(1j*(theta+np.pi/2)))
    center = np.abs(delta) <= half/2
    shoulders = (np.abs(delta) >= half) & (np.abs(delta) <= half+2*np.pi/n)
    inward = float(-residual[center].mean())
    depth = float(residual[shoulders].mean()-residual[center].mean())
    cell_delta = np.angle(np.exp(1j*(angles_c+np.pi/2)))
    patch = np.abs(cell_delta) < half
    raw = np.asarray(active['curvature']); wide = np.asarray(active['curvature_wide'])
    control_k = np.asarray(control['curvature'])
    reverse = patch & (raw < 0) & (wide < 0) & (control_k > 0)
    adjacent = bool(np.any(reverse & np.roll(reverse, 1)))
    tolerance = .02*float(np.mean(control['height']))
    classification = 'no_fold'
    if inward > tolerance and depth > tolerance:
        classification = 'curvature_inversion' if adjacent else 'shallow_dimple'
    return dict(radial_profile_valid=True, theta=theta, radial_difference=radial,
                uniform_radial_difference=float(uniform), radial_residual=residual,
                inward_displacement=inward, shoulder_depth=depth,
                min_patch_curvature=float(raw[patch].min()),
                min_patch_curvature_wide=float(wide[patch].min()),
                curvature_difference=raw-control_k,
                adjacent_curvature_reversal=adjacent, displacement_tolerance=tolerance,
                classification=classification, rotation=float(phase))


def smooth_patch(n, cells):
    theta = 2*np.pi*(np.arange(n)+.5)/n
    delta = np.angle(np.exp(1j*(theta+np.pi/2)))
    half = np.pi*cells/n
    weights = np.where(np.abs(delta) < half, np.cos(np.pi*delta/(2*half))**2, 0.)
    return weights/weights.max()


def smooth_ramp(elapsed, duration):
    x = np.clip(elapsed/duration, 0., 1.) if duration > 0 else float(elapsed >= 0)
    return x*x*(3-2*x)


class BendingFoam(PolarFoam):
    """Prescribed smooth cortical means; dynamics are inherited unchanged."""
    STATE_PARAMETERS = PolarFoam.STATE_PARAMETERS + ('activation_weights', 'activation_ramp')

    def interface_data(self):
        outer, inner, area, pressure, pn, _, count = super().interface_data()
        fixed = count-self.gam*np.all(self.e_f >= 0, axis=1)
        weights = getattr(self, 'activation_weights', self.active_cells.astype(float))
        ramp = getattr(self, 'activation_ramp', 0.)
        apical = self.apical_amplitude*smooth_ramp(self.morph_time-self.activation_start, ramp)
        lateral = self.lateral_amplitude*smooth_ramp(
            self.morph_time-self.activation_start-self.lateral_delay, ramp)
        for e in outer['edges']:
            fixed[e] += apical*weights[self.e_f[e, 1]]
        contact = np.flatnonzero(np.all(self.e_f >= 0, axis=1))
        a, b = self.e_f[contact].T
        fixed[contact] += lateral*(weights[a]+weights[b])/2
        return outer, inner, area, pressure, pn, fixed, count


def edge_roles(foam):
    outer, inner = boundary_loops(foam)
    roles = np.full(len(foam.e_v), 'lateral', dtype='<U8')
    roles[outer['edges']] = 'apical'; roles[inner['edges']] = 'basal'
    return roles


def t1_candidates(foam, previous, current):
    """Eligible shrinking films on an unchanged topology; no RNG is consumed."""
    candidates = []
    for e in np.flatnonzero((current < T1_LENGTH) & (current < previous)):
        if np.any(foam.e_f[e] < 0):
            adjacent = {int(k) for v in foam.e_v[e] for k in foam.v_e[v] if k >= 0}
            far = set(foam.e_v[list(adjacent)].ravel())-set(foam.e_v[e])
            if len(far) != 4:
                continue
        candidates.append(int(e))
    return candidates


def validate_closed_ring(foam):
    quality = protocol.validate(foam)
    # The disc validator tolerates junction-local microfolds. This study does not.
    if quality['film_crossings']:
        raise ValueError(f"{quality['film_crossings']} film crossing(s), including local microfolds")
    boundary_loops(foam)
    return quality

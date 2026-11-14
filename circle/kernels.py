"""Numba kernels for the disc model (strict IEEE, no fastmath).

``advance_disc`` is the disc-specific integration path.  The planar kernel
``activefoam._kernels.advance`` integrates and remeshes in one call, so a wall
projection placed after it would act on resampled nodes.  Here the order is

    forces -> trial positions -> wall projection -> records -> remesh

and the force arithmetic is copied term for term from the planar kernel with
boundary-orientation normals, so that with the wall at infinity the two agree.
"""
import numpy as np
from numba import njit
from numba.typed import List

from activefoam._kernels import resample, _intersection_parameter, edge_bounds


@njit(cache=True)
def resample_on_wall(mid, spacing, cx, cy, radius, tol):
    """``activefoam._kernels.resample`` with wall-following segments kept on the wall.

    A new node interpolated between two old nodes that both lie on the wall is
    moved radially onto the wall.  Contact is read from the positions at this
    instant; nothing is stored.  Without this, such a node starts on the chord,
    slightly inside the wall, and carries no reaction until pressure returns it.
    """
    lengths = np.sqrt(np.sum((mid[1:]-mid[:-1])**2, axis=1))
    total = lengths.sum()
    n = int(np.floor(total/spacing))
    out = np.empty((n+2, 2))
    out[0], out[-1] = mid[0], mid[-1]
    if n:
        on = np.empty(len(mid), np.bool_)
        for i in range(len(mid)):
            on[i] = np.hypot(mid[i, 0]-cx, mid[i, 1]-cy) >= radius-tol
        cum = np.empty(len(mid)); cum[0] = 0.
        for i in range(len(lengths)): cum[i+1] = cum[i]+lengths[i]/total
        for i in range(n):
            t = (i+1)/(n+1)
            k = min(max(np.searchsorted(cum, t, side='right')-1, 0), len(lengths)-1)
            f = (t-cum[k])/(cum[k+1]-cum[k]+1e-30)
            out[i+1] = mid[k]+f*(mid[k+1]-mid[k])
            if on[k] and on[k+1]:
                x = out[i+1, 0]-cx; y = out[i+1, 1]-cy; r = np.hypot(x, y)
                if r > 0.:
                    out[i+1, 0] = cx+x*(radius/r); out[i+1, 1] = cy+y*(radius/r)
    return out, n


@njit(cache=True)
def advance_disc(vpos, ends, faces, tension, mids, areas, psi, dt, pressure, lsc,
                 spacing, outward, cx, cy, radius, tol, remesh, wall_resample=False):
    """One force step with projection onto a rigid circular wall.

    Nodal forces are in the planar code's units: displacement = psi*dt*force.
    ``residual`` is the largest constrained nodal force (net displacement of a
    node over ``psi*dt``, after projection, before remeshing): it vanishes at a
    mechanical equilibrium.

    Returns the new geometry and, on pre-remesh node identities, the nodal wall
    reaction estimate ``(x_projected - x_trial)/(psi*dt)``, the continuous
    constraint value ``max(0, force.n)`` for nodes on the wall at the start of
    the step, a flag (1 = on the wall at step start, 2 = arrived this step) and
    the tributary length along outer films.
    """
    nv = len(vpos)
    step = psi*dt
    force = np.zeros_like(vpos)
    new = List()
    nodal = List()
    interior_total = 0
    for e in range(len(mids)):
        mid = mids[e]; f = np.zeros_like(mid)
        f0, f1 = faces[e]
        pn = 0.
        if f0 >= 0: pn -= pressure*(lsc*lsc/areas[f0]-1.)/lsc
        if f1 >= 0: pn += pressure*(lsc*lsc/areas[f1]-1.)/lsc
        teff = max(0., tension[e])
        for j in range(len(mid)-1):
            dx = mid[j+1, 0] - mid[j, 0]
            dy = mid[j+1, 1] - mid[j, 1]
            length = np.hypot(dx, dy)
            if length < 1e-14: continue
            tx, ty = dx/length, dy/length
            nx, ny = -ty, tx
            nx, ny = nx*outward[e], ny*outward[e]
            normal = pn*length/2.
            f[j, 0] += teff*tx+normal*nx
            f[j, 1] += teff*ty+normal*ny
            f[j+1, 0] += -teff*tx+normal*nx
            f[j+1, 1] += -teff*ty+normal*ny
        for axis in range(2):
            force[ends[e, 0], axis] += f[0, axis]
            force[ends[e, 1], axis] += f[-1, axis]
        new.append(mid+psi*dt*f)
        nodal.append(f)
        interior_total += len(mid)-2
    vertices = vpos+psi*dt*force

    # ---- junctions: constraint value at step start, then projection ----------
    excursion = -np.inf
    residual = 0.
    v_react = np.zeros((nv, 2)); v_direct = np.zeros(nv)
    v_flag = np.zeros(nv, np.int8); v_trib = np.zeros(nv)
    for v in range(nv):
        x0 = vpos[v, 0]-cx; y0 = vpos[v, 1]-cy; r0 = np.hypot(x0, y0)
        if r0 >= radius-tol:
            v_flag[v] = 1
            fn = (force[v, 0]*x0+force[v, 1]*y0)/r0
            if fn > 0.: v_direct[v] = fn
        x = vertices[v, 0]-cx; y = vertices[v, 1]-cy; r = np.hypot(x, y)
        if r-radius > excursion: excursion = r-radius
        if r > radius:
            px = cx+x*(radius/r); py = cy+y*(radius/r)
            v_react[v, 0] = (px-vertices[v, 0])/step
            v_react[v, 1] = (py-vertices[v, 1])/step
            vertices[v, 0] = px; vertices[v, 1] = py
            if v_flag[v] == 0: v_flag[v] = 2
        residual = max(residual, np.hypot(vertices[v, 0]-vpos[v, 0],
                                          vertices[v, 1]-vpos[v, 1])/step)

    # ---- film nodes: shared endpoints, projection, records, remesh -----------
    n_edge = np.empty(interior_total, np.int64); n_index = np.empty(interior_total, np.int64)
    n_pos = np.empty((interior_total, 2)); n_react = np.zeros((interior_total, 2))
    n_direct = np.zeros(interior_total); n_flag = np.zeros(interior_total, np.int8)
    n_trib = np.zeros(interior_total)
    k = 0
    resampled = 0
    counts = np.empty(len(mids), np.int64)
    for e in range(len(new)):
        mid = new[e]; old = mids[e]; f = nodal[e]
        for axis in range(2):
            mid[0, axis] = vertices[ends[e, 0], axis]
            mid[-1, axis] = vertices[ends[e, 1], axis]
        first = k
        for j in range(1, len(mid)-1):
            flag = 0; direct = 0.; rx = 0.; ry = 0.
            x0 = old[j, 0]-cx; y0 = old[j, 1]-cy; r0 = np.hypot(x0, y0)
            if r0 >= radius-tol:
                flag = 1
                fn = (f[j, 0]*x0+f[j, 1]*y0)/r0
                if fn > 0.: direct = fn
            x = mid[j, 0]-cx; y = mid[j, 1]-cy; r = np.hypot(x, y)
            if r-radius > excursion: excursion = r-radius
            if r > radius:
                px = cx+x*(radius/r); py = cy+y*(radius/r)
                rx = (px-mid[j, 0])/step; ry = (py-mid[j, 1])/step
                mid[j, 0] = px; mid[j, 1] = py
                if flag == 0: flag = 2
            residual = max(residual, np.hypot(mid[j, 0]-old[j, 0], mid[j, 1]-old[j, 1])/step)
            if flag != 0:
                n_edge[k] = e; n_index[k] = j
                n_react[k, 0] = rx; n_react[k, 1] = ry
                n_direct[k] = direct; n_flag[k] = flag
                k += 1
        for i in range(first, k):                      # after every node is projected
            j = n_index[i]
            n_pos[i, 0] = mid[j, 0]; n_pos[i, 1] = mid[j, 1]
            n_trib[i] = .5*(np.hypot(mid[j, 0]-mid[j-1, 0], mid[j, 1]-mid[j-1, 1])
                            + np.hypot(mid[j+1, 0]-mid[j, 0], mid[j+1, 1]-mid[j, 1]))
        if faces[e, 0] < 0 and faces[e, 1] >= 0:       # outer film: junction tributary
            last = len(mid)-1
            v_trib[ends[e, 0]] += .5*np.hypot(mid[1, 0]-mid[0, 0], mid[1, 1]-mid[0, 1])
            v_trib[ends[e, 1]] += .5*np.hypot(mid[last, 0]-mid[last-1, 0],
                                              mid[last, 1]-mid[last-1, 1])
        total = 0.
        minimum = np.inf
        maximum = -np.inf
        for j in range(len(mid)-1):
            dx = mid[j+1, 0] - mid[j, 0]
            dy = mid[j+1, 1] - mid[j, 1]
            length = np.sqrt(dx*dx + dy*dy)
            total += length
            minimum = np.minimum(minimum, length)
            maximum = np.maximum(maximum, length)
        n = len(mid)-2
        if remesh and (total < spacing*(n-1/3) or total > spacing*(n+4/3)
                       or maximum/max(minimum, 1e-12) > 2):
            if wall_resample:
                mid, n = resample_on_wall(mid, spacing, cx, cy, radius, tol)
            else:
                mid, n = resample(mid, spacing)
            resampled += 1
        new[e] = mid; counts[e] = n
    return (vertices, new, counts, excursion, residual, resampled, v_react, v_direct, v_flag, v_trib,
            n_edge[:k], n_index[:k], n_pos[:k], n_react[:k], n_direct[:k], n_flag[:k],
            n_trib[:k])


@njit(cache=True)
def max_radius(vpos, mids, cx, cy):
    """Largest distance of any junction or film node from the disc centre."""
    largest = 0.
    for v in range(len(vpos)):
        largest = max(largest, np.hypot(vpos[v, 0]-cx, vpos[v, 1]-cy))
    for mid in mids:
        for j in range(len(mid)):
            largest = max(largest, np.hypot(mid[j, 0]-cx, mid[j, 1]-cy))
    return largest


@njit(cache=True)
def films_finite(vpos, mids):
    """True if every junction and every film node has finite coordinates."""
    for v in range(len(vpos)):
        if not (np.isfinite(vpos[v, 0]) and np.isfinite(vpos[v, 1])):
            return False
    for mid in mids:
        for j in range(len(mid)):
            if not (np.isfinite(mid[j, 0]) and np.isfinite(mid[j, 1])):
                return False
    return True


@njit(cache=True)
def film_crossings(mids, ends):
    """Number of proper crossings between film segments that share no node.

    Every pair of segments is tested, within one film and between films,
    including films that share a junction: such films can cross again away from
    it.  Only segment pairs that legitimately meet are left out, namely
    consecutive segments of one film and the two segments incident on a shared
    junction.
    """
    bounds = edge_bounds(mids)
    count = 0
    for a in range(len(mids)):
        ax0, ax1, ay0, ay1 = bounds[a]
        p = mids[a]; na = len(p)-1
        closed = ends[a, 0] == ends[a, 1]
        for i in range(na):                               # within film a
            dx = p[i+1, 0]-p[i, 0]; dy = p[i+1, 1]-p[i, 1]
            for j in range(i+2, na):
                if closed and i == 0 and j == na-1:
                    continue
                if _intersection_parameter(p, p, i, j, dx, dy) > 0.:
                    count += 1
        for b in range(a+1, len(mids)):
            bx0, bx1, by0, by1 = bounds[b]
            if bx1 < ax0 or bx0 > ax1 or by1 < ay0 or by0 > ay1:
                continue
            q = mids[b]; nb = len(q)-1
            for i in range(na):
                dx = p[i+1, 0]-p[i, 0]; dy = p[i+1, 1]-p[i, 1]
                for j in range(nb):
                    # segments incident on a junction the two films share
                    if ((i == 0 and j == 0 and ends[a, 0] == ends[b, 0]) or
                            (i == 0 and j == nb-1 and ends[a, 0] == ends[b, 1]) or
                            (i == na-1 and j == 0 and ends[a, 1] == ends[b, 0]) or
                            (i == na-1 and j == nb-1 and ends[a, 1] == ends[b, 1])):
                        continue
                    if _intersection_parameter(p, q, i, j, dx, dy) > 0.:
                        count += 1
    return count


@njit(cache=True)
def _inside(point, polygon):
    """Even-odd test of a point against a closed polygon."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i+1) % n]
        if (y0 > y) != (y1 > y) and x < x0+(y-y0)*(x1-x0)/(y1-y0):
            inside = not inside
    return inside


@njit(cache=True)
def contained_cells(mids, loops, faces):
    """Number of ordered cell pairs (A, B) with A nested inside B.

    A is nested in B when every node of A on a film that does not border B lies
    inside B.  Requiring all of them separates nesting from a sub-threshold
    fold at a junction, which puts only a few nodes on the wrong side and is
    reported by the per-cell crossing report instead.  Together with
    ``film_crossings`` this covers overlap: two simple polygons whose boundaries
    do not cross are either disjoint or nested.
    """
    polygons = List()
    boxes = np.empty((len(loops), 4))
    for cell in range(len(loops)):
        size = 0
        for signed in loops[cell]:
            size += len(mids[signed if signed >= 0 else -signed-1])-1
        points = np.empty((size, 2)); index = 0
        for signed in loops[cell]:
            line = mids[signed if signed >= 0 else -signed-1]
            for j in range(len(line)-1):
                points[index] = line[j if signed >= 0 else len(line)-1-j]
                index += 1
        polygons.append(points)
        boxes[cell, 0] = points[:, 0].min(); boxes[cell, 1] = points[:, 0].max()
        boxes[cell, 2] = points[:, 1].min(); boxes[cell, 3] = points[:, 1].max()
    count = 0
    for a in range(len(loops)):
        for b in range(len(loops)):
            if a == b: continue
            if (boxes[a, 0] < boxes[b, 0] or boxes[a, 1] > boxes[b, 1] or
                    boxes[a, 2] < boxes[b, 2] or boxes[a, 3] > boxes[b, 3]):
                continue                                # A's box must lie within B's
            tested = 0; inside = 0
            for signed in loops[a]:
                e = signed if signed >= 0 else -signed-1
                if faces[e, 0] == b or faces[e, 1] == b:
                    continue
                line = mids[e]
                for j in range(len(line)):
                    tested += 1
                    if _inside(line[j], polygons[b]):
                        inside += 1
                    else:
                        break
                if inside < tested:
                    break
            if tested > 0 and inside == tested:
                count += 1
    return count

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
def advance_disc(vpos, ends, faces, tension, mids, areas, psi, dt, pressure, lsc,
                 spacing, outward, cx, cy, radius, tol, remesh):
    """One force step with projection onto a rigid circular wall.

    Nodal forces are in the planar code's units: displacement = psi*dt*force.
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

    # ---- film nodes: shared endpoints, projection, records, remesh -----------
    n_edge = np.empty(interior_total, np.int64); n_index = np.empty(interior_total, np.int64)
    n_pos = np.empty((interior_total, 2)); n_react = np.zeros((interior_total, 2))
    n_direct = np.zeros(interior_total); n_flag = np.zeros(interior_total, np.int8)
    n_trib = np.zeros(interior_total)
    k = 0
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
            mid, n = resample(mid, spacing)
        new[e] = mid; counts[e] = n
    return (vertices, new, counts, excursion, v_react, v_direct, v_flag, v_trib,
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
def film_crossings(mids, ends):
    """Number of proper crossings between films that share no junction."""
    bounds = edge_bounds(mids)
    count = 0
    for a in range(len(mids)):
        ax0, ax1, ay0, ay1 = bounds[a]
        p = mids[a]
        for b in range(a+1, len(mids)):
            if (ends[a, 0] == ends[b, 0] or ends[a, 0] == ends[b, 1] or
                    ends[a, 1] == ends[b, 0] or ends[a, 1] == ends[b, 1]):
                continue
            bx0, bx1, by0, by1 = bounds[b]
            if bx1 < ax0 or bx0 > ax1 or by1 < ay0 or by0 > ay1:
                continue
            q = mids[b]
            for i in range(len(p)-1):
                dx = p[i+1, 0]-p[i, 0]; dy = p[i+1, 1]-p[i, 1]
                for j in range(len(q)-1):
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
    """Number of ordered cell pairs (A, B) with a boundary node of A inside B.

    The test node lies on a film of A that does not border B, so cells that
    merely share a contact are not counted.  Together with ``film_crossings``
    this detects overlap: two simple polygons whose boundaries do not cross are
    either disjoint or nested.
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
            if (boxes[a, 1] < boxes[b, 0] or boxes[a, 0] > boxes[b, 1] or
                    boxes[a, 3] < boxes[b, 2] or boxes[a, 2] > boxes[b, 3]):
                continue
            for signed in loops[a]:
                e = signed if signed >= 0 else -signed-1
                if faces[e, 0] == b or faces[e, 1] == b:
                    continue
                line = mids[e]
                m = len(line)//2
                point = line[m] if len(line) > 2 else .5*(line[0]+line[1])
                if _inside(point, polygons[b]):
                    count += 1
                break
    return count

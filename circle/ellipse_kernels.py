"""Equal-area ellipse kernel. Force/remeshing arithmetic copied from advance_disc;
only the obstacle projection and boundary normal differ. radius is sqrt(area/pi).
The circle case delegates to the existing kernel exactly.
"""
import numpy as np
from numba import njit
from numba.typed import List
from activefoam._kernels import resample
from .kernels import advance_disc, resample_on_wall


@njit(cache=True)
def ellipse_normal(x, y, a, b):
    nx=x/(a*a); ny=y/(b*b)
    norm=np.hypot(nx,ny)
    return nx/norm,ny/norm


@njit(cache=True)
def project_ellipse(x,y,a,b):
    """Euclidean projection of an exterior point onto the filled ellipse."""
    if (x/a)**2+(y/b)**2 <= 1.:
        return x,y
    if a==b:
        r=np.hypot(x,y)
        return x*(a/r),y*(a/r)
    lo=0.; hi=max(a*a,b*b)
    while (a*x/(hi+a*a))**2+(b*y/(hi+b*b))**2 > 1.:
        hi*=2.
    for _ in range(70):
        mid=(lo+hi)/2.
        if (a*x/(mid+a*a))**2+(b*y/(mid+b*b))**2 > 1.:lo=mid
        else:hi=mid
    return a*a*x/(hi+a*a),b*b*y/(hi+b*b)


@njit(cache=True)
def advance_ellipse(vpos, ends, faces, tension, mids, areas, psi, dt, pressure, lsc,
                 spacing, outward, cx, cy, radius, tol, remesh, wall_resample=False, aspect=1.0):
    """One force step with closest-point projection onto a rigid elliptical wall.

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
    if aspect == 1.0:
        return advance_disc(vpos, ends, faces, tension, mids, areas, psi, dt, pressure,
                            lsc, spacing, outward, cx, cy, radius, tol, remesh, wall_resample)
    if wall_resample:
        raise ValueError("wall-aware circular resampling is unavailable for an ellipse")
    a = radius*np.sqrt(aspect)
    b = radius/np.sqrt(aspect)
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
        x0 = vpos[v, 0]-cx; y0 = vpos[v, 1]-cy; r0 = radius*np.sqrt((x0/a)**2+(y0/b)**2)
        if r0 >= radius-tol:
            v_flag[v] = 1
            nx, ny = ellipse_normal(x0, y0, a, b)
            fn = force[v, 0]*nx+force[v, 1]*ny
            if fn > 0.: v_direct[v] = fn
        x = vertices[v, 0]-cx; y = vertices[v, 1]-cy; r = radius*np.sqrt((x/a)**2+(y/b)**2)
        if r-radius > excursion: excursion = r-radius
        if r > radius:
            px, py = project_ellipse(x, y, a, b)
            px += cx; py += cy
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
            x0 = old[j, 0]-cx; y0 = old[j, 1]-cy; r0 = radius*np.sqrt((x0/a)**2+(y0/b)**2)
            if r0 >= radius-tol:
                flag = 1
                nx, ny = ellipse_normal(x0, y0, a, b)
                fn = f[j, 0]*nx+f[j, 1]*ny
                if fn > 0.: direct = fn
            x = mid[j, 0]-cx; y = mid[j, 1]-cy; r = radius*np.sqrt((x/a)**2+(y/b)**2)
            if r-radius > excursion: excursion = r-radius
            if r > radius:
                px, py = project_ellipse(x, y, a, b)
                px += cx; py += cy
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


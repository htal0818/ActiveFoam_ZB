"""Rigid equal-area ellipse extension; a/b=aspect and pi*a*b=pi*radius**2."""
import copy
import numpy as np
from activefoam._kernels import boundary_normal_signs
from activefoam.foam_full import SPACE
from .disc import DiscFoam,WallRecord
from .ellipse_kernels import advance_ellipse
from .kernels import films_finite


class EllipseFoam(DiscFoam):
    def __init__(self,tiling,aspect=1.5,**kwargs):
        if not 1. <= aspect <= 2.:raise ValueError('validated aspect range is 1 to 2')
        self.aspect=float(aspect)
        tiling=copy.deepcopy(tiling)
        if aspect!=1.:
            scale=np.array([np.sqrt(aspect),1/np.sqrt(aspect)])
            def transform(points):return tiling.center+(np.asarray(points)-tiling.center)*scale
            if hasattr(tiling,'pos'):
                tiling.pos=transform(tiling.pos)
                tiling.arcs={k:transform(v) for k,v in tiling.arcs.items()}
                if tiling.seeds is not None:tiling.seeds=transform(tiling.seeds)
            else:tiling.polylines=[transform(p) for p in tiling.polylines]
        if kwargs.get('box') is None:kwargs['box']=12*tiling.radius
        super().__init__(tiling,**kwargs)

    @property
    def axes(self):return np.array([self.radius*np.sqrt(self.aspect),self.radius/np.sqrt(self.aspect)])

    def max_radius(self):
        """Equivalent-area radius times max ellipse level, for confinement checks."""
        arrays=[self.vpos]+list(self.e_mid)
        return self.radius*max(float(np.linalg.norm((p-self.center)/self.axes,axis=1).max()) for p in arrays)

    def _invalid_reason(self):
        mids,_=self._fast_geometry_inputs()
        if not films_finite(self.vpos,mids):return 'nonfinite junction or film node'
        if not np.all(np.isfinite(self.f_area)) or np.any(self.f_area<=0):return 'nonpositive or nonfinite area'
        if self.max_radius()>self.radius*(1+self.REJECT_TOL):return 'node outside elliptical wall'
        return None

    def step(self, mu=0.0):
        """One Euler-Maruyama step with projection onto the wall."""
        mids, loops = self._fast_geometry_inputs()
        outward = boundary_normal_signs(self.e_f, loops)
        before = self.vpos
        (vertices, new, counts, excursion, residual, resampled, v_react, v_direct, v_flag,
         v_trib,
         n_edge, n_index, n_pos, n_react, n_direct, n_flag, n_trib) = advance_ellipse(
            self.vpos, self.e_v, self.e_f, self.e_t, mids, self.f_area, self.psi, self.dt,
            self.P0, self.lsc, self.edpc, outward, self.center[0], self.center[1],
            self.radius, self.wall_tol, self.remesh, self.wall_resample, self.aspect)
        if not excursion <= self.delta_max:
            raise ValueError(f"trial position {excursion:.3g} beyond the wall; "
                             "the Euclidean guarantee of the auxiliary box no longer holds")
        if getattr(self, "_measure_vertex_force", False):
            self._max_vertex_force = (np.linalg.norm(vertices - before - self.psi * self.dt * v_react,
                                                     axis=1).max() / (self.psi * self.dt))
        self.vpos, self.e_mid, self.e_rfn = vertices, new, counts
        self.wall = WallRecord(vertices.copy(), v_react, v_direct, v_flag, v_trib,
                               n_edge, n_index, n_pos, n_react, n_direct, n_flag, n_trib,
                               float(excursion), float(residual), int(resampled),
                               self.center.copy(),
                               self.radius)
        count = np.sum(self.e_f != SPACE, axis=1).astype(float)
        fixed = count - self.gam * np.all(self.e_f != SPACE, axis=1)
        xi = self.rng.standard_normal(len(count))
        self.e_t = (self.e_t - self.dt / self.tauT * (self.e_t - fixed)
                    + mu / self.tauT * np.sqrt(self.dt) / 2 * count * xi)
        self._wrap_vertices()
        self._update_faces()


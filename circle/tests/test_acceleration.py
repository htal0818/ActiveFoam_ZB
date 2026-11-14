"""Step 6: the compiled disc kernel against its pure-Python execution."""
import numpy as np
import pytest
from numba.typed import List

from activefoam._kernels import boundary_normal_signs, copy_polylines
from circle.disc import DiscFoam
from circle.kernels import advance_disc


def kernel_arguments(foam, remesh, wall_resample):
    mids, loops = foam._fast_geometry_inputs()
    outward = boundary_normal_signs(foam.e_f, loops)
    return (foam.vpos.copy(), foam.e_v, foam.e_f, foam.e_t, copy_polylines(mids), foam.f_area,
            foam.psi, foam.dt, foam.P0, foam.lsc, foam.edpc, outward, foam.center[0],
            foam.center[1], foam.radius, foam.wall_tol, remesh, wall_resample)


@pytest.mark.parametrize("remesh,wall_resample", [(True, False), (False, False), (True, True)])
def test_compiled_kernel_matches_pure_python(remesh, wall_resample):
    """Same inputs through the njit kernel and through its Python source."""
    foam = DiscFoam.random(16, seed=2, w=0.5, rho=1.2)
    for k in range(1, 401):
        foam.advance_scheduled(k, 1.0)
    for _ in range(3):
        compiled = advance_disc(*kernel_arguments(foam, remesh, wall_resample))
        python = advance_disc.py_func(*kernel_arguments(foam, remesh, wall_resample))
        assert len(compiled) == len(python)
        worst = 0.0
        for a, b in zip(compiled, python):
            if isinstance(a, List):
                assert len(a) == len(b)
                for x, y in zip(a, b):
                    assert x.shape == y.shape
                    worst = max(worst, np.abs(x - y).max())
            else:
                a, b = np.asarray(a), np.asarray(b)
                assert a.shape == b.shape and a.dtype == b.dtype
                if a.size:
                    worst = max(worst, np.abs(a.astype(float) - b.astype(float)).max())
        print(f"remesh={remesh} wall_resample={wall_resample}: largest difference {worst:.3g}")
        assert worst <= 1e-12
        assert len(compiled[8]) and np.abs(compiled[10]).sum() > 0   # the wall was acting
        foam.step(1.0)

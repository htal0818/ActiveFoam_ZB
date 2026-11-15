import numpy as np
import pytest
from circle.invagination import PolarFoam,boundary_loops,TopologyLimit
from circle.force_probe import isolated_copy
from circle import protocol,state


def test_annulus_has_two_oriented_boundaries_and_trivalent_junctions():
    f=PolarFoam.ring(n=16);outer,inner=boundary_loops(f)
    assert outer['area']>0 and inner['area']<0
    assert len(outer['edges'])==len(inner['edges'])==16
    assert np.all(np.sum(f.v_e>=0,axis=1)==3)
    assert sum(f.f_area)==pytest.approx(outer['area']+inner['area'],rel=1e-12)
    protocol.validate(f);assert not state.unclassified(f)


def test_closed_lumen_pressure_rises_when_area_falls():
    f=PolarFoam.ring()
    assert f.lumen_pressure(.9*f.lumen_reference_area)>f.lumen_pressure(f.lumen_reference_area)
    assert f.lumen_pressure(f.lumen_reference_area)==pytest.approx(f.lumen_initial_pressure)


def test_lumen_pressure_pushes_inner_boundary_outward_once():
    f=PolarFoam.ring(n=16,dt=1e-6);g=isolated_copy(f);g.lumen_initial_pressure+=.3
    before=f.vpos.copy();f.step(0.);g.step(0.)
    radial=before[:16]-f.center
    change=g.vpos[:16]-f.vpos[:16]
    assert np.all(np.sum(change*radial,axis=1)>0)
    # No direct extra pressure force is added to outer boundary junctions.
    np.testing.assert_allclose(g.vpos[16:],f.vpos[16:],atol=1e-13)


def test_apical_activation_uses_outer_connectivity_not_radial_threshold():
    f=PolarFoam.ring();f.apical_amplitude=2.
    outer,inner,_,_,_,fixed,_=f.interface_data()
    assert all(fixed[e]==1 for e in inner['edges'])
    for e in outer['edges']:assert fixed[e]==(3 if f.active_cells[f.e_f[e,1]] else 1)
    assert sum(fixed[e]>1 for e in outer['edges'])==sum(f.active_cells)


def test_prototype_stops_instead_of_silently_suppressing_topology():
    f=PolarFoam.ring();before=f.vpos.copy()
    with pytest.raises(TopologyLimit):f.t1_foam(0)
    np.testing.assert_array_equal(f.vpos,before)

"""The experimental apparatus must not modify the supplied simulation."""
import numpy as np
import pytest
from circle.disc import DiscFoam
from circle import protocol
from circle import force_probe as fp


def representation(value):
    if isinstance(value,np.ndarray): return ('array',value.dtype.str,value.shape,value.tobytes())
    if isinstance(value,np.random.Generator): return representation(value.bit_generator.state)
    if isinstance(value,dict): return tuple((k,representation(v)) for k,v in sorted(value.items()))
    if isinstance(value,(list,tuple,fp.List)): return tuple(representation(v) for v in value)
    if hasattr(value,'__dict__'): return (type(value).__name__,representation(vars(value)))
    return repr(value)


@pytest.fixture(params=['disc','periodic'])
def foam(request):
    f=(DiscFoam.random(16,seed=1,w=1.,rho=1.3,edpc=.15) if request.param=='disc' else
       protocol.periodic_foam(4,w=1.,rho=1.3,seed=1,edpc=.15))
    f._fast_geometry_inputs();fp.validate(f)
    return f


def test_input_unchanged_including_cache_identity_on_preemption(foam):
    edge=next(e for e in range(len(foam.e_f)) if min(foam.e_f[e])>=0 and fp.separation(foam,foam.e_v[e])[0]>.1)
    before=representation(vars(foam));identities={k:id(v) for k,v in vars(foam).items()}
    result=fp.probe(foam,edge,fp.ProbeConfig(increments=2,min_hold=.02,max_hold=.02,window=.01))
    assert result['status']=='preempted'
    assert representation(vars(foam))==before
    assert {k:id(v) for k,v in vars(foam).items()}==identities


def test_input_unchanged_on_unexpected_exception(foam,monkeypatch):
    edge=next(e for e in range(len(foam.e_f)) if min(foam.e_f[e])>=0 and fp.separation(foam,foam.e_v[e])[0]>.1)
    before=representation(vars(foam))
    def fail(work,*args):
        work.vpos[:]=123;work.rng.standard_normal(5)
        raise RuntimeError('injected failure after mutation')
    monkeypatch.setattr(fp,'clamp_step',fail)
    with pytest.raises(RuntimeError,match='injected failure'):fp.probe(foam,edge)
    assert representation(vars(foam))==before


def test_external_move_preserves_shared_endpoints_and_center(foam):
    edge=next(e for e in range(len(foam.e_f)) if min(foam.e_f[e])>=0 and fp.separation(foam,foam.e_v[e])[0]>.1)
    vertices=foam.e_v[edge];local=foam._local(foam.vpos[vertices]);q,n=fp.separation(foam,vertices)
    fp.move_junctions(foam,vertices,np.array([n,-n])*.001)
    assert fp.separation(foam,vertices)[0]==pytest.approx(q-.002,abs=1e-12)
    updated=foam._local(np.vstack([local[0],foam.vpos[vertices]]))[1:]
    np.testing.assert_allclose(updated.mean(axis=0),local.mean(axis=0),atol=1e-12)
    for e in np.unique(foam.v_e[vertices]):
        if e<0:continue
        for j in [0,1]:
            endpoint=foam.e_mid[e][0 if j==0 else -1]
            d=foam._local(np.array([endpoint,foam.vpos[foam.e_v[e,j]]]))
            np.testing.assert_allclose(d[0],d[1],atol=1e-12)


def test_work_is_pair_force_times_shortening_without_extra_factor_two():
    result=fp.curve_metrics([dict(separation=q,force=2.) for q in [1.,.75,.5]])
    assert result['work']==pytest.approx(1.)
    assert result['excess_work']==pytest.approx(0.)
    assert result['work_per_shortening']==pytest.approx(2.)

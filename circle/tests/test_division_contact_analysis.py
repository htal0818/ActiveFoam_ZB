"""Counting reversible contact flicker must not imply irreversible rearrangement."""
import pytest
from circle.scripts.division_contact_analysis import persistent_changes
from circle.scripts.division_dynamics import nonaffine_msd
import numpy as np
from types import SimpleNamespace
from circle.scripts.division_large import vectorized_uid_pairs
from circle.biology_experiments import uid_pairs


def test_short_loss_and_same_time_reversal_are_not_persistent():
    events=[dict(time=1.,lost=[(1,2)],gained=[]),
            dict(time=1.05,lost=[],gained=[(1,2)]),
            dict(time=2.,lost=[(1,2)],gained=[]),
            dict(time=2.,lost=[],gained=[(1,2)])]
    assert persistent_changes([(1,2)],events,0.,3.,.2)==[]


def test_long_loss_and_recovery_count_separately_with_censoring():
    events=[dict(time=1.,lost=[(1,2)],gained=[]),
            dict(time=2.,lost=[],gained=[(1,2)]),
            dict(time=2.9,lost=[(1,2)],gained=[])]
    result=persistent_changes([(1,2)],events,0.,3.,.2)
    assert [e['time'] for e in result]==pytest.approx([1.2,2.2])
    assert [e['present'] for e in result]==[False,True]


def test_centroid_metric_removes_rigid_motion_and_uniform_contraction():
    x=np.array([[0.,0.],[1.,0.],[.5,1.],[0.,2.]])
    q=np.array([[0.,-1.],[1.,0.]])
    y=.6*x@q+[3.,4.]
    assert nonaffine_msd(x,y)<1e-28
    y[-1,0]+=.1
    assert nonaffine_msd(x,y)>1e-5


@pytest.mark.parametrize('n',[1,8,32,256])
def test_vectorized_observer_matches_original_pairs(n):
    rng=np.random.default_rng(n)
    faces=rng.integers(-1,n,size=(5*n,2))
    foam=SimpleNamespace(e_f=faces,cell_uid=rng.permutation(n)+31)
    assert vectorized_uid_pairs(foam)==uid_pairs(foam)

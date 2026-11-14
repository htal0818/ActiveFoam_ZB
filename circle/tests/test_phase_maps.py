"""Prevent activity maps from accidentally plotting passive preparation values."""
import pytest
from circle.scripts.phase_maps import normalize


def active_record():
    return dict(task=dict(domain='disc',mode='dynamics'),accepted=True,stage='complete',
                structure=dict(z_all=99.,shape_all=99.,packing=99.),
                dynamics=dict(neighbor_mean=dict(all=4.2,bulk=5.2),shape_mean=3.9,
                              shape_series=dict(bulk=[3.7,3.8,4.]),zone_counts=dict(bulk=1),
                              packing_mean=.91,contact_change_rate=dict(all=.04),tauT=10.,t1_rate=.02))


def test_active_maps_use_measurement_window_not_preparation():
    result=normalize([active_record()])[0]['values']
    assert result==pytest.approx(dict(z=4.2,z_bulk=5.2,q=3.9,q_bulk=3.9,phi=.91,exchange=.4,t1=.2))


def test_old_active_record_requires_additional_measurement():
    record=active_record();del record['dynamics']['neighbor_mean']
    with pytest.raises(ValueError,match='during dynamics'):
        normalize([record])

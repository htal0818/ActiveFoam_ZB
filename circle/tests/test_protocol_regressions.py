"""Validation, continuation and paired-comparison regressions."""
from types import SimpleNamespace
import numpy as np
from circle import protocol, benchmarks
from circle.disc import DiscFoam


def test_chunked_disc_evolution_preserves_transition_cadence():
    whole = DiscFoam.random(8, seed=3, w=.5, dt=.001)
    split = DiscFoam.random(8, seed=3, w=.5, dt=.001)
    protocol.evolve(whole, .012, .2)
    protocol.evolve(split, .005, .2)
    protocol.evolve(split, .007, .2)
    assert whole.log.iteration == split.log.iteration == 12
    assert whole.log.accepted == split.log.accepted
    np.testing.assert_array_equal(whole.vpos, split.vpos)
    np.testing.assert_array_equal(whole.e_t, split.e_t)


def test_paired_benchmark_excludes_both_members_when_one_fails(monkeypatch):
    def initialize(n, w, rho, seed, wall_resample=False):
        if seed == 2 and wall_resample:
            raise ValueError('deliberate invalid geometry')
        return SimpleNamespace(dt=.005, wall=(seed,wall_resample))
    def evolve(foam, duration, mu, **kwargs):
        kwargs['observer'](duration,foam)
    monkeypatch.setattr(protocol, 'initialize', initialize)
    monkeypatch.setattr(protocol, 'evolve', evolve)
    monkeypatch.setattr(protocol, 'validate', lambda f: {'local_film_microfolds':0})
    monkeypatch.setattr(benchmarks, 'wall_load', lambda wall: {'total':10*wall[0]+int(wall[1]), 'arrivals':0})
    result=benchmarks.resampling_effect_active(duration=.05, seeds=(1,2,3))
    assert [r['seeds'] for r in result['rows']] == [2,2]
    assert [r['mean'] for r in result['rows']] == [20.,21.]
    assert result['paired_mean'] == 1.
    assert result['paired_standard_error'] == 0.
    assert [r['seed'] for r in result['paired_differences']] == [1,3]
    failed=[r for r in result['runs'] if not r['valid']]
    assert len(failed)==1 and failed[0]['seed']==2 and 'geometry' in failed[0]['error']

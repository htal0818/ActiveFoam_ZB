"""Step 1: the disc-clipped network."""
import numpy as np
import pytest

from activefoam.foam_full import SPACE
from circle.disc import DiscFoam
from circle.tiling import clipped_voronoi


def polygon_area(points):
    x, y = points.T
    return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)


@pytest.mark.parametrize("n_cells", [3, 5, 16, 36, 100])
def test_network_is_trivalent_with_single_outer_films(n_cells):
    foam = DiscFoam.random(n_cells, seed=n_cells)
    degree = np.zeros(len(foam.vpos), dtype=int)
    for a, b in foam.e_v:
        degree[a] += 1
        degree[b] += 1
    assert np.all(degree == 3)
    assert np.all(np.sum(foam.v_e >= 0, axis=1) == 3)

    tiling = clipped_voronoi(n_cells, np.random.default_rng(100 + n_cells),
                             center=foam.center, spacing=foam.edpc)
    # Junctions are the interior Voronoi vertices plus the wall crossings: the
    # arc samples are film nodes only.
    assert len(foam.vpos) == tiling.n_interior + tiling.n_crossing
    outer = np.flatnonzero(foam.e_f[:, 0] == SPACE)
    assert len(outer) == tiling.n_outer_films == tiling.n_crossing
    for e in outer:
        radius = np.hypot(*(foam.e_mid[e] - foam.center).T)
        assert np.allclose(radius, foam.radius, rtol=0, atol=1e-12)
        assert len(foam.e_mid[e]) == foam.e_rfn[e] + 2
        # each outer film borders one cell and joins two distinct wall junctions
        assert foam.e_f[e, 1] >= 0 and foam.e_v[e, 0] != foam.e_v[e, 1]
    on_wall = np.isclose(np.hypot(*(foam.vpos - foam.center).T), foam.radius, atol=1e-12)
    assert on_wall.sum() == tiling.n_crossing


@pytest.mark.parametrize("n_cells", [5, 16, 36, 100])
def test_initial_areas_are_those_of_the_inscribed_polygon(n_cells):
    foam = DiscFoam.random(n_cells, seed=3)
    assert np.all(foam.f_area > 0)
    # Order the outer-film nodes around the wall: the cells tile that polygon.
    # (shared junctions appear twice; a repeated point adds no area)
    nodes = np.vstack([foam.e_mid[e] for e in np.flatnonzero(foam.e_f[:, 0] == SPACE)])
    angle = np.arctan2(*(nodes - foam.center).T[::-1])
    inscribed = polygon_area(nodes[np.argsort(angle)])
    assert foam.f_area.sum() == pytest.approx(inscribed, rel=1e-12)
    assert foam.f_area.sum() < foam.disc_area
    # Areas come from the curved outer films, not from chords between junctions.
    offset = foam.vpos - foam.center
    on_wall = np.isclose(np.hypot(*offset.T), foam.radius, atol=1e-12)
    wall_junctions = foam.vpos[on_wall][np.argsort(np.arctan2(offset[on_wall, 1],
                                                              offset[on_wall, 0]))]
    assert foam.f_area.sum() > polygon_area(wall_junctions)


@pytest.mark.parametrize("n_cells", [3, 16, 36, 100, 400])
def test_mean_contact_number_of_the_confluent_tiling(n_cells):
    """z = 6 - (6 + 2B)/N holds for this connected, confluent, trivalent tiling only."""
    foam = DiscFoam.random(n_cells, seed=7)
    outer_films = int(np.sum(foam.e_f[:, 0] == SPACE))
    contacts = int(np.sum(foam.e_f[:, 0] != SPACE))
    assert 2 * contacts / n_cells == pytest.approx(6 - (6 + 2 * outer_films) / n_cells)
    assert foam.neighbor_number().mean() <= 2 * contacts / n_cells + 1e-12


def test_chord_area_deficit_scales_with_spacing_squared():
    deficits, spacings = [], [0.2, 0.1, 0.05]
    for spacing in spacings:
        foam = DiscFoam.random(36, seed=2, edpc=spacing)
        deficits.append(1.0 - foam.volume_fraction())
    for coarse, fine in zip(deficits, deficits[1:]):
        assert 3.2 < coarse / fine < 4.8
    # Leading-order reference for uniform sampling; the coefficient is not exact
    # because each film rounds its own node count.
    reference = (np.array(spacings) / foam.radius) ** 2 / 6
    assert np.all(np.array(deficits) < 1.05 * reference)
    assert np.all(np.array(deficits) > 0.5 * reference)


def test_static_geometry_is_independent_of_the_auxiliary_box():
    """Same absolute coordinates, three box sizes: identical areas, centres, lengths."""
    reference = DiscFoam.random(36, seed=5)
    for factor in (16.0, 1000.0):
        other = DiscFoam.random(36, seed=5, box=factor * reference.radius)
        assert np.array_equal(other.vpos, reference.vpos)
        assert np.array_equal(other.f_area, reference.f_area)
        assert np.array_equal(other.f_center, reference.f_center)
        assert np.array_equal(other.edge_lengths(), reference.edge_lengths())
        assert other.volume_fraction() == reference.volume_fraction()


def test_disc_must_fit_the_box_with_margin():
    tiling = clipped_voronoi(16, np.random.default_rng(0), center=(1.0, 1.0))
    with pytest.raises(ValueError):
        DiscFoam(tiling)
    tiling = clipped_voronoi(16, np.random.default_rng(0), center=(9.0, 9.0))
    with pytest.raises(ValueError):
        DiscFoam(tiling, box=4.0 * tiling.radius)

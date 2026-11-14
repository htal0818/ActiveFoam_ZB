# Eight full-size cells to 256, without growth

This starts with eight founding cells, each with preferred area K/P0=1, and
divides all cells synchronously at t=2, 6, 10, 14 and 18 τR. The run continues
to t=22. Thus the sequence is 8→16→32→64→128→256, with four τR of mechanical
relaxation after the final round. No division is staggered between mothers.

Total preferred area remains 8; final daughter preferred areas are 1/32. These
are the same final preferred sizes as in the earlier 1→32 example. This is a
new eight-founder cluster, not a continuation of the eight tiny descendants
in that earlier movie, whose total preferred area was only 1.

The rigid circular wall has radius 0.7√8 ≈ 1.980 throughout this run. Its initial
scale is larger than the earlier movie to accommodate the eight founders; it
does not grow, contract or deform during the simulation. All frames use a
fixed spatial scale. Colors identify the eight founding lineages.

Parameters: W/T0=1, P0=10, τT=10, dt=0.001, film spacing=0.025, seed=1.
Growth, tension noise and delayed feedback are all zero. Founding-cell units
remain fixed as daughters get smaller. Area is conserved at each instantaneous
cut; total actual occupied area can subsequently change through mechanical
relaxation. Total K is conserved throughout.

Preparation uses an eight-seed clipped Voronoi network in the disc, 1.5 τR of
confluent relaxation at W=0 with spacing 0.2, adoption at W=1 and spacing 0.025,
space seeding, five τR of zero-noise annealing, tension reset to the fixed points,
and ten τR of zero-noise relaxation. These preparation times precede movie t=0.

The standard foam topology schedule remains enabled. Contact-change counting
excludes septum insertion and mother-to-daughter ID replacement. A new pair
state must persist continuously for 0.2 τR before it is counted. Each pair loss
or gain counts separately, so these counts are not numbers of four-cell T1
exchanges. Short reversible toggles are retained in the raw data. The observation
window after the last division is four τR; this alone cannot determine sustained
fluidity or long-time arrest. There is no matched skip-division control in this run.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/foam-mpl XDG_CACHE_HOME=/tmp/foam-cache \
  .venv/bin/python circle/scripts/division_large.py
```

Use `--render-only` to regenerate the figures from the stored simulation.
Data are in `circle/data/division_8_to_256_20261114/simulation.json.gz`.
No model force law, transition thresholds or planar defaults are changed.

## Observed outcome

All five rounds completed. Total actual cell area falls from 7.029 at movie
t=0 to 5.189 at t=22, while total preferred area remains 8. The minimum final
cell area is 0.01551; every final cell polygon has at least 22 sampled vertices.
This sampling check is not a numerical convergence study.

| Round | Pair changes persisting ≥0.2 τR in the next 4 τR | Changes / cell / τR |
|---|---:|---:|
| 8→16 | 0 | 0 |
| 16→32 | 0 | 0 |
| 32→64 | 4 | 0.015625 |
| 64→128 | 42 | 0.08203125 |
| 128→256 | 184 | 0.1796875 |

After the last round, 85.1% of the 705 initial contacts never undergo a persistent
loss during the four-τR window. The endpoint graph has lost 76 initial pairs and
gained 72 others. Mean shape index falls from 4.110 immediately after the last
division to 3.821. Residual centroid MSD, with translation, rotation and uniform
contraction removed, reaches 0.06974 times the initial mean daughter area and is
still increasing at the observation endpoint. These are transient measurements;
they do not establish sustained fluidity. Increasing generation also decreases
cell size and changes actual occupied area, so cell count alone does not explain
the trend.

There are 227 geometry/conservation checks, with no resolved crossing, nested
cell or wall violation. Small local junction microfolds remain: the peak is 96
simultaneous local crossings, the final frame has 13, and the maximum reported
depth is 0.000965. This illustrative run retains the existing junction tolerance;
convergence of its contact-change rates is untested. Every sampled contact graph
reconstructs exactly from the recorded event stream. All 19 targeted division
and measurement tests pass.

The contact observer uses a vectorized implementation checked against the
original pair sets for 1, 8, 32 and 256 cells. It changes measurement cost only.
Use `--reference-contacts` to use the slower original observer. The stored model
source hash is `acf54bc462e7a5fae11c6dd99c8d1273a869a1529a53886de2a5938e8eb90503`.

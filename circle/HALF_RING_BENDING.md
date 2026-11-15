# Activating half of a 48-profile epithelial ring

The requested half-ring intervention is implemented as a prescribed change in
the cortical mean tensions of 24 contiguous cell profiles. The observed response
is mostly flattening with a shallow local bend: these experiments do **not**
produce a validated inward fold of all 24 profiles.

## Results

All five cases use the same prepared geometry, closed compliant lumen and matched
unactivated control. Four complete 100 tauR. The strongest case stops on a film
crossing at 63.2825 tauR; only its last valid sampled state at 63 tauR is shown.
It is a geometric failure, not evidence of physical fluidization or a successful
invagination. No branch executes topology changes.

| Spatial profile | Peak apical mean increment | Delayed lateral increment | Last valid sampled time | Selected profiles moving inward | Selected profiles reversing curvature | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cosine | 2 | 1 | 100 | 10 | 0 | Complete |
| Broad plateau | 2 | 1 | 100 | 18 | 2 | Complete |
| Cosine | 4 | 1 | 100 | 8 | 0 | Complete |
| Cosine | 8 | 1 | 63 | 6 | 2 | Film crossing at 63.2825 |
| Cosine | 4 | 0 | 100 | 6 | 4 | Complete |

The plateau distributes inward displacement over more profiles, while cosine
activation with apical +4 and unchanged lateral means produces four adjacent
profiles with reversed midline curvature. Its central-patch mean inward
displacement is 0.13369 and its depression relative to the shoulders is 0.28048
in simulation length units. This is a shallow fold, not hemispheric inversion.
The plateau's two reversed profiles are isolated and do not satisfy the existing
adjacency criterion for a developed local inversion. Every run and its diagnostic
curves are retained; the before/after figure and movie select the completed run
with the most profiles reversing curvature, then greatest shoulder depth.

## Intervention and controls

The initial ring, 120-tauR preparation, one-sided fixed wall, pressure convention,
OU equation, tension clipping, mobility and remeshing are the same as in
[`BENDING_STUDY.md`](BENDING_STUDY.md). Noise, growth and division are zero.
No inward force or displacement is imposed. No energy functional is introduced.
Basal mean tensions stay at one. The apical increments refer to mean tension
targets, not to the OU noise amplitude.

The cosine profile has nonzero weights on exactly the lower 24 of 48 material
cells. The alternative plateau has 16 cells at full weight and a four-cell
cubic taper at each end, also selecting exactly 24. Both profiles have the same
peak but different integrated activation: their target increments are not a
comparison at fixed total stimulation. All increments ramp over 5 tauR; lateral
activation, when present, starts 20 tauR later and averages the two adjacent
cell weights at each contact. Actual tensions approach these means with the
unchanged OU relaxation time of 10 tauR.

The lumen uses the existing anchored law
`p_l = p_prepared + 5*(A_prepared/A - 1)`. The old `control_B5` trajectory is reused:
the script verifies exact agreement of initial cell polygons, lumen area and
lumen pressure. It compares measurements at equal elapsed times. Timestep is
0.0025 and film spacing is 0.075. Connectivity remains fixed and the existing
strict crossing, closed-loop and interface-collapse checks run every timestep.

## What the counts mean

The selected-cell count is an input; the two response counts are outputs.
Inward displacement is evaluated against the control after lumen-centroid and
rotation alignment and subtraction of uniform radial contraction. A selected
profile counts as moving inward if the radial difference at its control polar
angle is below minus 0.02 times the mean control cell height (about 0.0209).
This is a geometric radial-response diagnostic, not a Lagrangian displacement
tracking individual material points. Curvature reversal requires negative
signed midline curvature in both the adjacent-cell and wider two-cell estimates.

These are deterministic exploratory runs, not an ensemble or a biological
calibration. Quantitative spacing convergence remains unresolved in the prior
bending study, and no new spatial-convergence claim is made here. The domain is
flat; the measured ring curvature is not Gaussian surface curvature.

## Reproduction and validation

```bash
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl \
  .venv/bin/python -m circle.scripts.half_ring_bending
```

The runner reuses matching saved results; conflicting parameter dictionaries
raise an error. Portable raw trajectories and analyses are in
`data/half_ring_bending_20261115/`; figures and the animation are in
`../figures/circle/half_ring_bending_20261115/`. The prepared state and control
are from the preceding `bending_48_20261115` study.

All 11 existing bending and invagination tests pass. A direct target check for
each of the five configurations verifies exactly 24 elevated apical targets and
unchanged basal means. The simulation itself verifies fixed connectivity and
validates geometry at every step. No files in `activefoam/` were edited.

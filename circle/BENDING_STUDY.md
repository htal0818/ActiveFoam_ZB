# Fixed-topology cortical bending pilot

This follows the user's requested diagnostic-first experiment on the 48-profile
planar epithelial ring. It is a two-dimensional cross-section, with outward
apical surfaces, lumen-facing basal surfaces, and lateral cell-cell contacts.
It is neither axisymmetric nor a full spherical cellular shell. Cell geometry
and lumen compliance remain exploratory, not calibrated to microscopy.

## Results and limits

All 23 branches finished: 18 completed their requested holds, and five stopped
at the apical-film collapse guard. No branch stopped for a detected geometric
crossing. This count includes controls and resolution checks. Standard holds
are 100 tauR; resolution checks are 50 tauR.

The smooth protocol produces local curvature reversal with no imposed inward
motion. At the standard spacing 0.075, the following matched-control results
are measured at 100 tauR. Inward displacement is the central-patch average
after center/rotation alignment and removal of uniform radial contraction.

| Apical mean increment | Delayed lateral mean increment | Local inward displacement | Final morphology |
| ---: | ---: | ---: | --- |
| 0.5 | 0 | 0.110 | shallow dimple |
| 1 | 0 | 0.198 | curvature inversion |
| 2 | 0 | 0.216 | curvature inversion |
| 2 | 0.5 | 0.309 | curvature inversion |
| 2 | 1 | 0.376 | curvature inversion |
| 0.5 | 0.5 | 0.096 | shallow dimple |

For apical +2, adding lateral +1 lowers the mean height of the two central cells
from 1.602 to 1.252, increases localized inward displacement by about 74%, and
keeps the shortest apical film at 0.364 rather than 0.0109. Thus lateral
shortening helps in this regime, but it does not universally enhance bending:
the weak combined protocol has less displacement than weak apical activation
alone. The apical +1 case also reverses curvature with no near-collapse film
(shortest apical film 0.508).

Across lumen moduli 0.5 to 500 at apical +2, localized displacement changes only
from 0.214 to 0.216, while lumen-area reduction relative to control changes from
0.9565% to 0.0032%. Global deflation is therefore not required for the local bend
in this tested regime. A four-profile patch reaches the collapse guard at
14.3175 tauR; an eight-profile patch and a twelve-profile patch complete 100 tauR.
Those width comparisons also change total activation, as documented below.

Sharp instantaneous activation stops at 9.7625 tauR; the sharp spatial profile
with a 5-tauR temporal ramp stops at 12.315 tauR. A smooth profile with the same
total target increment and temporal ramp stops at 19.35 tauR. Smoothing delays
the failure at matched total activation, but does not eliminate it. Simply
increasing apical activation to +4 also causes collapse (18.4125 tauR).

**Quantitative spatial convergence is not established.** For smooth apical +2,
halving the timestep changes displacement by less than 0.02% at time 50.
Halving film spacing as well retains the curvature reversal but moves its first
sampled onset from 23 to 32.5 tauR and reduces displacement at time 50 from
0.16709 to 0.12009 (28.1%). The wider-scale minimum curvature changes from
-0.40789 to -0.15054. The finer run's shortest apical film is still 0.326,
well above its collapse guard. This supports a qualitative bending mechanism
without requiring a collapsed interface, but does not validate precise timing,
depth, or a quantitative phase boundary. The combined protocol has not itself
undergone the spacing comparison. No numerical mobility rescaling was used to
force agreement, and no noisy or biological ensemble is claimed.

See `outcomes.csv`, `resolution_comparison.json`, `analysis.json`, and the raw
per-run archives for continuous values and separate run-status labels. Figures
include the original cell diagnostics, the four controlled sweeps, the selected
two-stage trajectory and its animation, and the resolution comparison.

## What the original run actually did

The original scheduled run was replayed. Its first proposed event occurs at
9.6375 tauR after activation: T1 on edge 97, an **apical free-boundary film**
of cell 32, of length 0.0348648. It is not the collapse of a lateral contact
bridging the apical and basal boundaries.

At the last sampled time, 9.5 tauR, the two edge cells of the sharp eight-cell
activation patch have apical arclengths 0.0668, down from 1.0425. Their basal
arclengths increase from 0.9109 to 1.1544. The interior cells respond much less.
The apical/basal ratios of those edge cells are about 0.058. Negative curvature
appears at isolated cell-scale locations, but not in the wider spatial estimate.
These measurements are in `original_diagnostics.json` and the corresponding
eight-panel figure; symmetry makes pairs of cell traces overlap.

## Preserved mechanics and prescribed interventions

`BendingFoam` inherits the existing polarized ring's force integration, pressure
law, clipping, OU update, mobility, remeshing and wall constraint. It changes only
the prescribed mean-tension profile. The planar code and original `PolarFoam`
defaults are untouched. No inward force, imposed displacement, energy functional,
preferred perimeter, growth, division or ECM is added. Noise is zero.

Material cell weights are a cosine-squared bump about the vegetal pole, zero
outside the prescribed 4-, 8- or 12-profile patch and normalized to a sampled
peak of one. They are fixed to the material cells rather than reselected as the
geometry moves. A cubic smoothstep raises each prescribed increment over 5 tauR.
The OU drift then controls the approach of actual tensions to their targets.
The prescribed apical/lateral mean increments are distinct from the OU noise
amplitude; the latter stays zero in every run here.

Only apical means receive the first increment. Basal means remain unchanged.
In the two-stage runs, lateral activation starts 20 tauR later, with the same
5-tauR ramp; a contact receives the average of its two cells' weights.

All branches clone the same prepared state at 120 tauR. Lumen pressure is
anchored to its value in this state when varying the compression modulus:

```
p_l(A) = p_prepared + B*(A_prepared/A - 1)
```

Thus all branches start with identical geometry and pressure, without a pressure
jump. `B` here is the area-normalized modulus at the prepared state. The old
parameter `B=5` was referenced to the constructed lumen area, so its slope at
the prepared state differs slightly. The original-event replay retains the old
reference exactly; the controlled sweep consistently uses the anchored law.

Pilot values: apical peak increments 0.5, 1, 2, 4; lumen moduli 0.5, 5, 50, 500;
delayed lateral increments 0.5 and 1; patch widths 4, 8, 12. There is also a weak
combined protocol (apical 0.5, lateral 0.5). Standard runs request 100 tauR.
Each modulus has its own matched control.

Smooth and sharp profiles at equal peak have different total increments. An
additional smooth profile matches the sharp eight-cell profile's total increment
of 16, and a sharp profile with the same 5-tauR temporal ramp is included. Patch
width scans hold peak amplitude fixed, so they change the total increment too;
they are not pure fixed-dose width comparisons.

## Fixed topology and stopping conditions

The user explicitly requested fixed-topology evolution for this experiment.
The runner calls the ordinary mechanical step without applying topology events.
It audits the original shrinking-edge T1 eligibility condition every third step
and logs **every eligible film by interface type**, aggregating repeated proposals
for each edge. No event-order random draws are needed because events are not
applied. The ordinary OU draw convention is retained, with amplitude zero.

Strict geometry validation runs every timestep. Unlike the general disc
validator, this experiment allows no detected proper crossings, even local
microfolds. The two boundary loops must remain intact. Invalid geometry stops
the branch. A film stops the run if its arclength or endpoint separation falls
below 0.005 times the initial mean cell height (about 0.005225 here). This is a
**numerical collapse guard**, not proof of a physical singularity. Apical, basal
and lateral films are all monitored. The last displayed sampled frame precedes
the stop; it is never an invalid trial configuration.

T4-like crossing proposals therefore terminate as geometry failures. The intact
ring has no all-cell junctions for space creation and no space-space films to
merge. This protocol does not validate an epithelial topology rule or measure
neighbor exchange rates. Accepted connectivity is checked to remain unchanged.

## Measurements and descriptive categories

Per cell: apical and basal arclengths, their chord lengths, area, apicobasal
height (distance between arclength midpoints), and both width ratios. Film
arclengths can differ from widths because of scalloping. Height and width-ratio
curvature estimates are imperfect for strongly sheared cells.

The epithelial midline joins the averages of each cell's apical and basal
arclength midpoints in material order. Signed circumcircle curvature is computed
using adjacent cells and again using points two cells apart. Both definitions
give positive `1/R` for a circular CCW midline. The wider estimate tests whether
a sign reversal extends beyond an isolated cell-scale feature.

Active and control midlines are aligned by lumen area centroid and a rigid
rotation. Radius is interpolated at equal geometric polar angles. The angular
mean radial difference is removed before measuring local displacement. A
non-star-shaped midline is reported as having an unavailable single-valued
radial diagnostic; this is not automatically a simulation failure.

Descriptive categories use central-patch inward displacement and depression
relative to the two shoulders. Both must exceed 0.02 of the initial cell height
to call a shallow dimple. Curvature inversion additionally requires at least two
adjacent patch profiles to have negative curvature in both spatial estimates,
with positive control curvature. These are declared diagnostic conventions,
not a rigidity threshold or an inferred phase boundary. Raw continuous values,
first inversion time and duration of final continuous inversion are retained.
Run status (completed, collapse guard, geometry/numerical failure) is always
reported separately from the last valid morphological category.

## Resolution checks and reproduction

The main spacing is 0.075 and timestep 0.0025. The apical +2 case and its control
also run to 50 tauR at timestep 0.00125, and jointly at spacing 0.0375 and
timestep 0.00125. The latter starts from the same prepared polygonal geometry,
with collinear midpoint subdivision; initial cell-area preservation is checked.
Mobility is not rescaled with spacing. These comparisons test numerical
sensitivity, not biological variation in cell count.

```bash
OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. \
  .venv/bin/python -m circle.scripts.bending_experiments --workers 5
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl \
  .venv/bin/python -m circle.scripts.plot_bending
```

Data: `data/bending_48_20261115/`. Figures: `../figures/circle/bending_48_20261115/`.
The prepared-state pickle is only a local `/tmp` cache; the reproducible input
geometry, preparation history, parameters, individual trajectories and event
logs are retained in portable JSON/gzip files.

Validation: 23 targeted tests and 42 unchanged planar regression tests pass.
Tests include analytic circular curvature, rigid-motion and uniform-contraction
removal, an imposed synthetic *diagnostic* fold, recovery of actual polygon
areas, exact agreement with the old sharp-target OU dynamics, delayed lateral
targets with unchanged basal means, and proposal auditing without RNG changes.
The synthetic fold is a measurement test only; it is never used in a simulation.

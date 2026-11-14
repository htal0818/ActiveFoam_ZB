# Division triggers rearrangements; this run does not show sustained fluidity

The original division-only movie was replayed with committed contact-changing
topology transactions recorded. Its final polygons at t=22 matched exactly.
The 32-cell cluster was then followed to t=42, or 24 τR after the final division,
without further division, growth, feedback or tension noise.

Before each of the five rounds, an identical copy of the complete simulation
state was saved. These controls omit that round and all subsequent divisions,
then evolve for the same observation interval. This is a counterfactual test of
performing the division at that state. It is not a comparison at matched cell
number, cell size, packing fraction or mechanical equilibrium.

## Counting contacts

Direct sister-contact creation, inheritance of mother contacts, and replacement
of mother IDs by daughter IDs are excluded. Each observation window starts with
the post-division graph. Cell IDs remain fixed throughout that window.

The raw event stream contains thousands of rapid losses and recoveries of the
same contacts. Those counts cannot be interpreted as thousands of independent
neighbor exchanges or as evidence of fluidity. The main figure uses a dwell
filter: a new contact state must last continuously for at least 0.2 τR before
the filtered graph changes. The change is timestamped at confirmation, not
backdated. Intervals too near the observation endpoint are right-censored.
One lost pair and one gained pair count as two pair changes, not one T1 exchange.
The filter is descriptive; it does not fix or validate the numerical origin of
rapid topology toggling. Even a filtered change can later reverse.

First 4 τR after each round:

| Division | Persistent pair changes | Skip-division control | Rate per daughter per τR | Control rate per mother per τR |
|---|---:|---:|---:|---:|
| 1→2 | 0 | 0 | 0 | 0 |
| 2→4 | 1 | 0 | 0.0625 | 0 |
| 4→8 | 5 | 1 | 0.15625 | 0.0625 |
| 8→16 | 5 | 1 | 0.078125 | 0.03125 |
| 16→32 | 25 | 1 | 0.1953125 | 0.015625 |

The last-round counts are 25 versus 2 at dwell 0.1 τR, and 16 versus 0 at dwell
0.5 τR. The qualitative increase is robust to these choices; the count itself
is threshold dependent. Full sensitivity results are in `persistent_summary.json`.

## After the last round

- There are 77 distinct contacts immediately after 16→32. At lag 24 τR,
  seven have been lost and seven different pairs gained: 70/77, or 90.9%, of
  the initial contacts are present. This endpoint comparison does not prove
  that the seven replacements are seven individually paired T1 events.
- Fourteen initial contacts underwent a persistent loss at some point;
  some recovered. The fraction never persistently lost is 63/77 = 81.8%.
- The last raw contact-changing transaction is at lag 10.707 τR. There are
  no raw or filtered contact changes in the final ten τR of observation.
- Mean shape index decreases from 4.115 immediately after division to 3.772.
  No vertex-model rigidity threshold is assumed to apply to this force model.
- Centroid motion, after fitting out translation, proper rotation and uniform
  contraction, slows toward a plateau. Residual MSD / initial mean daughter
  area is 0.121 at lag 14 and 0.129 at lag 24, versus 0.020 at the control endpoint.
  This finite-time residual metric is not a diffusion coefficient or viscosity.
- The skip-division control retains 32/33 original contacts at the endpoint.

Adding septa changes the force balance and drives shape relaxation, with some
lasting changes in neighbors. Continued macroscopic flow is not established.
The finite cluster never contacts the circular wall, all daughters get smaller
without growth, and actual occupied area changes. Therefore this run cannot
locate a bulk fluidity transition or isolate division from all size/density effects.
A rheological response test, more realizations and resolution checks would be
needed before interpreting the response as a material fluidity change.

## Validation and reproduction

All sampled contact graphs reconstruct exactly from the recorded transaction
streams. There are 1,640 geometry checks across the movie continuation and
controls, with no resolved crossings, nesting or wall violations. Transient
junction microfolds remain within the existing tolerance (maximum reported
depth 0.000883 in the dividing run). Three analysis tests check rapid reversible
toggles, dwell-time confirmation and censoring, and removal of rigid motion and
uniform contraction. Numerical convergence of these rearrangement rates has
not been established.

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/foam-mpl XDG_CACHE_HOME=/tmp/foam-cache \
  .venv/bin/python circle/scripts/division_dynamics.py
MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python circle/scripts/division_contact_analysis.py
.venv/bin/python -m pytest circle/tests/test_division_contact_analysis.py -q
```

`persistent_division_dynamics.png` is the main figure. `division_dynamics.png`
shows the unfiltered counts for diagnostic comparison only. Raw simulation data
are in `circle/data/division_dynamics_20261114/results.json.gz`; no model source
or original movie data were modified. The source hash is included in the data.

# Strictly synchronous cleavage with random orientations

Every current mother divides at the same prescribed time. There is no stochastic
cell-cycle scheduler, no partial-wave commitment and no growth. The optional
independent orientation generator proposes uniform cut-normal angles on [0, pi).
All plans use the unchanged predivision geometry. Invalid cuts and conflicts on
shared interfaces are redrawn before any split is committed. Accepted angles are
therefore conditioned on geometric validity. The whole wave rolls back on failure;
orientation attempts consume draws, but the simulation's OU generator does not.

`BiologyFoam.divide_synchronously(orientation_rng)` enables this protocol. Omitting
the argument preserves the previous shape-dependent division rule. Daughters
inherit K/P0 in proportion to area, preserving pressure and total osmotic content.
New contacts start at 2T0-W. No mechanical integration occurs between splits.

The stress diagnostic now uses each daughter's inherited preferred area instead
of the founder's A0. This corrects reported isotropic stress only; the pressure
forces already used the inherited value. All original force laws remain intact.

Completed preliminary runs in `data/synchronous_cleavage_20261115/`:

- One random-orientation wave for 8 and 16 prepared mothers, with relaxation and
  skipped-wave controls.
- One-cell to 32-cell passive reference, five waves four relaxation times apart.
  Instantaneous area and pressure conservation pass. Subsequent total cell area
  decreases to about 0.540 from about 0.769 despite conserved K/P0=1. This is
  compressible relaxation, not added growth or a pressure jump at cleavage.

The planned larger interval/noise/refinement batch has **not** been run; work
shifted to the separately requested epithelial-ring geometry. Do not interpret
this initial reference as an ensemble result or assume mild compression persists
through all generations. Transition thresholds remain in founder units.

Reproduce the completed cases:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m circle.scripts.synchronous_cleavage --batch gates
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m circle.scripts.synchronous_cleavage --batch pilot
```

Measurements include pressure variance, corrected stress, cell anisotropy, wall
reaction, persistent contact changes and junction nonaffine speed measured before
topology can rename vertices. Division-created contacts are excluded. The full
suite passed 207 tests before adding the separate epithelial-ring prototype.

# Dynamics experiments after the circular-wall response maps

The first implementation and batch are in `experiments.py`,
`scripts/run_experiments.py`, and `data/experiments_20261114/`.
The force law, tension normalization, circular-wall integration, transition rules
and all files in `activefoam/` remain unchanged. Elliptical containers are an
explicit extension of the wall geometry.

## Experiment matrix

| Family | Controls | Runs | Warmup / observation in τR |
|---|---|---:|---|
| Numerical refinement | N=36, W=1, ρ=1.3, μ=1.5; dt=0.005/0.0025, spacing=0.15/0.075 | 8 | 10 / 50 |
| Active adhesion | N=36/100, W=0.5/1/1.5, μ=0.25/0.5/1, ρ=1.2 | 54 | 50 / 100 |
| Opening spaces | N=36, W=0.25/0.35, ρ=0.9/1, μ=0.5 | 12 | 50 / 100 |
| Persistence, fixed amplitude | τT=2/5/10/20, μ=0.5 | 12 | 5τT / 10τT |
| Persistence, fixed raw variance | τT=2/5/10/20, μ scaled below | 12 | 5τT / 10τT |
| Long trajectories | N=36, W=1/1.5, μ=0.5, ρ=1.2 | 6 | 200 / 500 |
| Pulses and shams | N=36/100; sham, uniform, wall layer, angular sector | 24 | 50 / 70 |
| Equal-area geometry | N=36, a/b=1/1.5/2, W=1, μ=0.5, ρ=1.2 | 9 | 50 / 100 |

Total: 137 runs. Three seeds per scientific point, two per refinement point.
Default dt=0.005, spacing=0.15, τT=10. Preparation relaxes a confluent network at
W=0, ρ=1 for 7.5τR, adopts target parameters, opens spaces, anneals for 10τR at
μ=0.5, resets tensions, and quenches for 30τR. Persistence variants share τT=10
preparation; their target τT starts at warmup. Seed labels across dt/spacing
refinements do not imply coupled Brownian paths or identical meshes.

## What is recorded

- Every committed outer topology transaction: time, type, location, changed pairs
  and affected cells. Nested operations that a parent rejects are not events.
  Suboperations within one successful outer transaction are coalesced.
- Pair gains/losses immediately after those transactions and after each scheduled
  timestep. Initial-neighbor overlap includes contacts that re-form; continuous
  survival excludes them. Contact lifetimes retain left/right-censor flags.
- Cell centroids, contacts, areas, shape index, polygon area-moment shape tensor,
  wall contact, free-film status, wall distance/curvature, and length-weighted
  incident tension every 0.5τR. Cell IDs remain fixed; film IDs do not.
- Mean/cage-relative MSD, overlap, 2-D non-Gaussian displacement parameter,
  finite-window χ4, displacement distributions, event waiting times, four
  clustering-cutoff choices with spatially shuffled catalogs, tension–shape lags,
  boundary profiles, and four stationarity blocks.
- Final polygons, source hash, complete parameters and failures. Failed trajectories
  are never substituted with zero-valued observables.

The plotted event rate counts **contact-changing transactions per cell per τR**.
It is not the old sampled pair-change rate or a count of distinct lasting exchanges.
The transition-type CSV distinguishes T1 from T3/T4 operations. Repeated reversals
can inflate both transaction counts and apparent event clusters.

## Controlled extensions

Fixed-amplitude persistence uses μ=0.5. The second protocol uses
`μ(τT)=0.5*sqrt((2τT−dt)/(20−dt))`, holding stationary raw Euler–Maruyama contact
variance constant. Rectification, free-boundary switching and finite observation
can still change force statistics. The τT=10 controls are duplicate trajectories,
not six independent seeds.

All activity protocols retain the original one-draw-per-film OU update and the
half-amplitude free-boundary convention. Pulses increase μ from 0.25 to 1 during
measurement t∈[10,20)τR, globally, within one code length of the wall, or within
±π/4 of the positive x axis. The wall stays fixed. Each intervention is compared
with its same-seed uniform-baseline sham; topology differences eventually change
how random draws are assigned to films. These protocols have equal peak noise
amplitude, but are not matched for total noise input or number of activated films.

`EllipseFoam` uses axes `a=R√aspect`, `b=R/√aspect`, preserving area πR²=N.
An initially circular clipped tiling is deformed with this area-preserving affine
map and then relaxed. Force arithmetic and polyline remeshing are inherited;
exterior trial nodes are projected to the **nearest Euclidean point** on the ellipse,
whose normal is proportional to `(x/a²,y/b²)`. This differs from radial projection.
The integration kernel agrees with the circle kernel away from the obstacle, and
aspect=1 delegates to that kernel. The auxiliary box is 12R. Tests cover exact
confinement, normal projection, rollback, the circle limit and compiled/Python
agreement. The geometry extension currently supports 1≤aspect≤2 and plain remeshing.
Local ellipse wall distance/curvature measurements use a 1024-point boundary sample;
the dynamical projection itself uses a solved closest-point condition. Circle-only
arclength traction profiles are not used for ellipses. Unlike a circular wall, an
elliptical frictionless wall can exert torque about the center.

## Acceptance and interpretation

Geometry checks occur every 0.5τR in preparation and ordinary experiment stages.
The refinement gate checks every timestep after preparation. A finer preparation
check interval exposed transient resolved crossings between scheduled repairs;
we retained the established preparation protocol and did not loosen the geometric
threshold. Passing sampled checks therefore does not certify a crossing-free path.
A separate six-run validation-cadence audit repeats selected baseline parameters
with checks at every timestep after preparation, without changing their dynamics.

Tolerated local microfolds are still reported. Numerical failures do not define a
physical phase. Complete-case averages may be biased when some seeds fail.
Bands/error bars use seed SD, not cell-frame pseudoreplication. Three seeds are a
pilot; time blocks and correlated time origins cannot replace independent seeds.
Equal-area ellipses remove the variation of N and total area, but global shape,
orientation and local curvature still covary. A correlation with local curvature
alone does not establish causality.

Stable contact-lineage tracking for tension–length correlations, correlated noise,
osmotic pulses, delayed feedback and growth/division are deferred. The current
cell-averaged tension–shape lag does not substitute for a contact-specific mechanism.

## Reproduction

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests circle/tests -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/run_experiments.py --workers 4
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/audit_experiments.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/summarize_cadence_audit.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/verify_experiments.py
MPLCONFIGDIR=/tmp/foam-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/plot_experiments.py
MPLCONFIGDIR=/tmp/foam-mpl OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/summarize_experiments.py
```

The runner resumes only when the task list and core-source hash match its manifest.
Use a new output directory after changing core sources. Figures, CSVs, methods and
run counts are in [the gallery](../figures/circle/experiments_20261114/index.html).

## Completed first batch

The main batch completed 96 of 137 runs; all 41 failures were geometric. All six
every-timestep baseline audits completed and exactly matched their ordinary
counterparts. The full regression suite passes 158 tests; `activefoam/` is untouched.
All 24 pulse/sham runs and all 24 persistence runs completed. See
[the findings](../figures/circle/experiments_20261114/FINDINGS.md) for measured effects
and limitations. Thirteen PNG/PDF figure sets and all raw records are retained.

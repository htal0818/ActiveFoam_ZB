# Delayed tension feedback, growth and synchronous cell division

These are explicit model extensions. `activefoam/` and the existing circular
force kernel are unchanged. The rigid circular container never deforms or expands.
The extensions live in `biology.py`, `division.py` and `biology_experiments.py`.

## Delayed contact feedback

For a contact between persistent cell identities i and j, define its total film
length Lij (summing multiple films between that pair) and its length Lij,b when the
contact first appears. The strain sensor is

```
epsilon_ij(t) = Lij(t)/Lij,b - 1

tauT dTij = [-(Tij - Tfixed,ij)
             + g tanh(epsilon_ij(t-delay)/epsilon_scale)] dt + mu dWij
```

The experiments set epsilon_scale=0.1. Positive delayed extension raises the
contact's target tension; shortening lowers it. Saturation bounds the feedback
contribution by ±g. The original force uses max(Tij,0). Fixed-point baseline tensions
remain 2T0−W at contacts and T0 at free films. Free films have zero feedback and
retain half the contact noise amplitude. This is a phenomenological mechanosensing
rule; it is not part of the original paper model and does not require an energy.

The contact reference remains fixed while the cell pair stays in contact. A lost
and re-formed pair receives a new reference and zero prehistory. New daughter pairs
also start fresh. Persistent cell IDs avoid confusing edge-array reindexing with
contact identity. Histories are sampled at a fixed interval (0.05τR by default),
with linear interpolation of the delayed value. Gain, delay and history sampling
are configured when creating the extension, not changed during a run.

Each feedback pilot uses the same deterministic initial contact-tension perturbation
`0.1*cos(1.7*sum((face_indices+1)**2))`. There is **no ongoing stochastic forcing**
(mu=0). Zero gain and zero delay are separate controls. The purpose is to distinguish
sustained feedback-driven fluctuations from a damped response to an initial kick.
The original seeded random ordering of eligible topology operations is retained;
mu=0 switches off stochastic tension forcing, not that algorithmic ordering.

## Cell growth and pressure

Each cell now has its own preferred area Ai,preferred=Ki/P0, while keeping the
osmotic law Pi=Ki/Ai. In the founding-cell code units:

```
p_i = (P0/lsc) * (preferred_i / A_i - 1)
d preferred_i / dt = growth_rate * preferred_i
```

Mobility, lsc, P0 and all coordinate/time units stay fixed to the founding cells.
We do not renormalize units when N changes. `foam.rho` remains the founding-cell
parameter; use `foam.nominal_density = sum(preferred_i)/wall_area` for the current
total preferred-area density. Measured packing remains `sum(A_i)/wall_area`.
The existing force kernel receives equivalent pressure denominators A_i*A0/preferred_i;
its segment-force arithmetic is unchanged. The uniform no-growth case bypasses
this transformation and matches the original implementation exactly.

## Synchronous division

`foam.divide_synchronously()` divides **every current mother at one clock time**.
There is no mechanical timestep, growth update or random draw between individual
splits. Newly created daughters are not divided again in the same round.

1. Plan all cleavages using the pre-round geometry. Each cleavage bisects the
   mother's polygon area. Its normal follows the major axis of the mother's area
   moment tensor (daughters separate along the long axis).
2. Use deterministic angular adjustments if a cut would hit an existing junction,
   coincide with a neighboring planned cut, or intersect a nonconvex boundary more
   than twice. Isotropic cells use a deterministic orientation based on persistent
   ID. The chosen angles and adjustments are recorded. This geometric tie-breaking
   can depend on mother order; it does not introduce different biological times.
3. Split the crossed films without displacing their old nodes, insert two new triple
   junctions and a straight daughter contact, and update neighboring cell loops.
   The new contact starts at 2T0−W. Existing film pieces inherit their tensions.
4. Partition the mother's K in proportion to the equal daughter areas. Both total K
   and total geometric cell area are conserved at cleavage. Daughter Ki/Ai initially
   equals the mother's, avoiding an artificial osmotic-pressure jump.
5. Retire the mother ID, assign two new daughter IDs, and record their common birth
   time and parent. Validate the full round. If any split or validation fails, restore
   **the entire round**, including N, lineage, K, feedback memories, geometry,
   connectivity, RNG and caches. Rejection logs remain as diagnostics.

Thus the algorithm constructs cuts sequentially in memory but commits one atomic
synchronous event. It does not implement mitotic rounding, a resolved contractile
ring, stochastic cell-cycle times or asynchronous divisions. The operation can be
called at multiple scheduled times; tests verify a 3→6→12 sequence. The first
scientific pilot uses one 16→32 round.

## Pilot design

- Feedback: N=16, rho=1.2, W/T0=1, tauT=10τR, gains 0.5/1/2/4, delays 0/2/10τR,
  plus zero-gain control; three seeds, 150τR observation. 39 runs.
- Division controls: neither growth nor division, division only, growth only,
  and growth plus division; three seeds. Start N=16, rho=0.6. With growth enabled,
  preferred areas double exponentially over 0–20τR and growth then stops. With
  division enabled, all mothers split at 20τR. Observe to 60τR with mu=0.05. 12 runs.
- Resolution checks: seed 1, zero gain and gain 0.5 at delays 2/10; run to 300τR.
  Jointly refine dt 0.005→0.0025, film spacing 0.1→0.05 and history sampling
  0.05→0.025. Preparation is repeated at each resolution, so trajectories need not
  match pointwise. These six runs are sensitivity checks, not a convergence order.
- One additional every-timestep geometry audit repeats the gain=0.5, delay=2 pilot.
  Selection of the refinement cases is based on the initial pilot, not an independent
  validation ensemble.

The main pilot completed 36/51 runs, including all 12 growth/division controls.
All seven longer/refined/audited runs completed. A follow-up combined pilot adds
gain=0.5, delay=2τR to the same growth/division protocol and noise level; all three
seeds completed. This combined case has not received its own refinement study.
Geometric failures remain recorded and excluded from physical statistics.
Ordinary evolution checks geometry every 0.1τR; preparation retains the established
0.5τR cadence and local microfold tolerance. The stronger-feedback failures are
numerical limitations, not labeled biological phases.

Measurements include individual shapes and centroids, contact tensions, measured
packing, preferred-area density, wall reaction, lineage, division times, and
committed contact-changing T1/T3/T4 operations. Wall reaction records refer to the
most recent integration step; instantaneous cleavage does not compute a new wall
reaction until the next mechanical step. Direct cleavage changes are excluded
from spontaneous neighbor-exchange counts. Neighbor overlap restarts at each round;
cell-displacement or contact-memory statistics must not compare unrelated mother
and daughter IDs as if the population were fixed. Spectral peaks are reported as
diagnostics and checked against late-time amplitude and numerical refinement.

## Tests and use

The combined suite passes 170 tests, including exact original-model parity with
extensions disabled, causality of delayed feedback, fresh memory after contact
reformation, pressure/area/K conservation, trivalence, lineage, synchronous birth
times, two rounds, single-cell cleavage, and complete rollback after a later mother
fails or an unexpected exception occurs.

```python
from circle.biology import BiologyFoam
from circle.protocol import equilibrated

base = equilibrated(16, w=1, rho=0.6, seed=1, edpc=0.1)
foam = BiologyFoam.from_disc(base, gain=0.5, delay=2.0)
# Advance on the existing schedule; growth_rate may be prescribed separately.
for k in range(1, 4001):
    foam.advance_scheduled(k, mu=0.0)
assert foam.divide_synchronously()  # every current mother, same clock time
```

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest tests circle/tests -q
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/run_biology.py --workers 4
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/refine_biology.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/combine_biology.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python circle/scripts/verify_biology.py
MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python circle/scripts/render_biology.py division
MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python circle/scripts/render_biology.py feedback
MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python circle/scripts/plot_biology.py
MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python circle/scripts/summarize_biology.py
```

[Figures and animations](../figures/circle/biology_20261114/index.html).
[Measured findings](../figures/circle/biology_20261114/FINDINGS.md).
Raw records: `data/biology_20261114/` and `data/biology_refinement_20261114/`.
Historical experiment source hashes refer to their recorded code versions; the
previous experiment suite is reproducible at commit `93044c5`.

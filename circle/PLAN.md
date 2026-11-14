# Shared plan: active foam inside a rigid circular wall

Implementation continued from `e306dbe` on `codex/circular-wall-validation`. All disc code lives in `circle/`; nothing in
`activefoam/` is edited. If a planar edit proves necessary it is raised first.

## Model

The planar force-balance model of `activefoam/foam_full.py` (junctional tension, cell
pressure, curved polyline films, stochastic tensions, T1–T4), confined inside a rigid,
frictionless, nonadhesive circular wall of fixed radius `R_w = sqrt(N/pi)`. Free cells and
clusters are validation fixtures only. Production ensembles are deferred until the
confinement and wall-adjacent transition checks pass.

Conventions fixed for the disc and for matched periodic comparison runs:

- Pressure normals from the oriented cell boundary, areas of the current step
  (`pressure_normals="boundary"`, `reference_pressure_lag=False`). Planar defaults unchanged.
- Units as in the planar code: a node moves by `psi*dt*f`, so mobility `M = psi = sqrt(rho)`;
  nodal force `f = T_eff*t + p_n*(l/2)*n` per segment end, `T_eff = max(T, 0)`,
  `p_n = P0*(rho/A - 1)/sqrt(rho)`.
- Continuous wall law `F_wall = -max(0, F.n) n`. Numerically: trial Euler step, radial
  projection, then remeshing. The nodal reaction estimate is
  `(x_projected - x_trial)/(M*dt)`, recorded before remeshing.
- Films are polylines. No persistent contact labels and no claim of exact circular coverage.
- Auxiliary box: disc centre at fixed absolute coordinates `(4R_w, 4R_w)`, box side
  `>= 8R_w`, shear zero. `_local` and `_wrap_vertices` are Euclidean in `DiscFoam`; audit
  mode asserts the periodic versions would have returned the same values.
- A transition attempt, accepted or rejected, consumes no random draws. Rejected attempts
  are rolled back from a snapshot generated from the attribute tables in `state.py`
  (parameters, physical state, caches, diagnostics); the log of attempts is not restored.

## Steps, gates and status

Each step's tests and the 42 planar tests must pass before the next step.

| # | Step | Tests | Owner | Status |
|---|---|---|---|---|
| 1 | Clipped network (`tiling.py`) | `tests/test_tiling.py` | Claude | done |
| 2 | Disc integrator and wall (`kernels.py`, `disc.py`) | `tests/test_wall.py` | Claude | done |
| 3 | Transition guards and rollback (`state.py`, `disc.py`) | `tests/test_transitions.py` | Claude | done |
| 4 | Active-run audit, box independence, overlap diagnostics | `tests/test_consistency.py` | Claude | done |
| 5 | Measurements (`measure.py`) | `tests/test_measure.py` | Claude | done |
| 6 | Compiled vs pure-Python kernel; then disc vs matched periodic | `tests/test_disc_acceleration.py`, `scripts/pilot_compare.py` | Claude | kernel check done; pilot comparison only (2 seeds), ensembles deferred |
| R | Independent review of each pushed step | — | ChatGPT | reviewed through `e306dbe`; validation and benchmark follow-up implemented by Codex |

Measured results: [`BENCHMARKS.md`](BENCHMARKS.md) and the summary in [`README.md`](README.md).

## Follow-up after independent review

- Strict validation now enforces inter-film crossings, with explicit topology and
  distance witnesses for tolerated local folds at collapsed spaces. Unrelated
  overlapping circles are rejected even when both cells are individually simple.
- Preparation remains at W=0, rho=1. A previously accepted high-activity preparation
  develops a resolved crossing; it is now a rejection regression rather than a
  successful measurement fixture.
- Active resampling comparisons use corrected preparation and checked trajectories;
  averages use only complete valid seed pairs, with failures and paired differences
  retained in JSON. Wall-aware resampling stays off by default.
- The finest-level comparison uses scale 8 on both sides. Duplicate test module
  names were removed, so `python -m pytest tests circle/tests -q` works.
- Disk evolution continues the transition cadence across successive calls.

## Open points

- **`rho = 1` is not a matched comparison point.** The periodic box holds the tissue in
  tension at cell area 1; a nonadhesive wall cannot, so the disc tissue shrinks to cell
  areas of 0.86–0.93 and leaves the wall. Ensembles should compare at matched interior
  cell area or pressure, or begin with compressed points such as `rho >= 1.2` in this pilot. Verify contact
  at each point; this is not a universal threshold.
- **Partial-contact traction converges slowly.** About 9% high at spacing 0.15 for three
  cells; 0.5% between the two finest levels. Wall reaction in ensembles needs a spacing
  study at the cell numbers used.
- **Remeshing and contact.** Instantaneous loads depend on sampling. Use the regenerated
  paired results in `BENCHMARKS.md`: two wall-aware seeds failed, leaving one
  complete pair. The old unvalidated
  raw-tiling averages have been withdrawn. The optional wall-aware mode remains off.
- **Interior shape factor.** Exploratory size comparisons are now available in
  `data/curvature_sweep/`; compare only accepted seed pairs and report numerical
  convergence before interpreting differences physically.
- **No natural out-of-wall rejection** has been observed; the guard is tested with
  fixtures only.

## Not in scope

Sphere and ring models, wall adhesion, exact circular-arc contact segments, shear rheology,
production ensembles.

Rerun outcome: 36/36 pilot runs accepted under the documented tolerance; 12 disk
endpoints clean and six with tolerated local microfolds. Scientific convergence
remains open. No wall-aware-default change is justified by the surviving pair.

## Follow-up: parameter, activity and wall-curvature sweeps

User-requested follow-up adds `sweep.py`, `scripts/parameter_sweep.py`,
`scripts/plot_sweep.py` and measurement tests, without changing `activefoam/`.
The grid uses N=36,100,196, three seeds, matched periodic controls, nine passive
adhesion/density points and four activity points per size and domain (234 tasks).
The fixed circle's curvature is 1/R with R=√(N/π); surface curvature remains zero,
and wall curvature remains coupled to finite size.

Snapshots, structure maps, MSD, accepted T1 counts, sampled neighbor-change rates,
and shape-index trajectories/changes are exported alongside all per-run records.
Dynamic wall/interior zones are assigned after warmup. Seed SEM, failed-run counts
and complete-pair disk-minus-periodic contrasts prevent cell-wise pseudoreplication
and silent substitution of zeros for failed runs. High-activity geometry failures
remain reported; no phase boundary or long-time diffusion is inferred from the
short 50-τR observation window. See `data/curvature_sweep/REPORT.md`.

The combined regression suite passes 145 tests. The existing force law, wall
constraint, resampling default, transition rules and planar defaults are unchanged.

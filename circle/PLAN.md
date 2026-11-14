# Shared plan: active foam inside a rigid circular wall

Branch: `claude/sharp-mendel-6bax2h`. All disc code lives in `circle/`; nothing in
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
| 6 | Compiled vs pure-Python kernel; then disc vs matched periodic | `tests/test_acceleration.py`, `scripts/pilot_compare.py` | Claude | kernel check done; pilot comparison only (2 seeds), ensembles deferred |
| R | Independent review of each pushed step | — | ChatGPT | steps 1–2 reviewed at `cb24540`; three findings fixed |

Measured results: [`BENCHMARKS.md`](BENCHMARKS.md) and the summary in [`README.md`](README.md).

## Working together

- Run the gates:

  ```bash
  python -m pytest tests -q
  ```

  ```bash
  python -m pytest circle/tests -q
  ```

- Hand changes back as patches against a named commit of this branch, with the test output.
- Disagreements about the model or a test criterion go in this file under "Open points",
  not into the code.

## Open points

- **`rho = 1` is not a matched comparison point.** The periodic box holds the tissue in
  tension at cell area 1; a nonadhesive wall cannot, so the disc tissue shrinks to cell
  areas of 0.86–0.93 and leaves the wall. Ensembles should compare at matched interior
  cell area or pressure, or stay at `rho >= 1.2`, where the tissue presses on the wall.
- **Partial-contact traction converges slowly.** About 9% high at spacing 0.15 for three
  cells; 0.5% between the two finest levels. Wall reaction in ensembles needs a spacing
  study at the cell numbers used.
- **Remeshing and contact.** Instantaneous reaction dips as low as 48% of its mean after
  resampling; no resolved bias in the time average (21.6 ± 0.1 vs 21.8 ± 0.1). Wall-aware
  resampling exists as an option and is off by default. Decide whether to turn it on.
- **Interior shape factor** is 0.03–0.05 above the periodic value at every pilot point.
  Size dependence untested.
- **No natural out-of-wall rejection** has been observed; the guard is tested with
  fixtures only.

## Not in scope

Sphere and ring models, wall adhesion, exact circular-arc contact segments, shear rheology,
production ensembles.

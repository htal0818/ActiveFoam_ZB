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
| 3 | Transition guards and rollback (`state.py`, `disc.py`) | `tests/test_transitions.py` | Claude | in progress |
| 4 | Active-run audit and box independence | `tests/test_consistency.py` | Claude | pending |
| 5 | Measurements (`measure.py`) | `tests/test_measure.py` | Claude | pending |
| 6 | Compiled vs pure-Python kernel; then disc vs matched periodic | `tests/test_acceleration.py`, `scripts/` | Claude | pending |
| R | Independent review of steps 1–2 against this plan | — | ChatGPT (proposed) | open |

Step 2 results so far (single-cell fixtures, zero activity):

- Free cell far from the wall: area within 0.06% of the planar free-cell root, zero reaction,
  identical trajectory to the planar kernel.
- Squeezed cell: integrated traction vs `2*pi*R_w*(p_n - T_eff/R_w)`: errors 1.8%, 0.45%,
  0.11% at spacings 0.15, 0.075, 0.0375 (second order).
- Chord-area deficit: `0.161–0.165 * (spacing/R_w)^2`, against the leading-order `1/6`.
- Projection estimate vs constraint value on a fixed mesh: first order in `dt`.

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

- A nonadhesive wall leaves extracellular space beside it even at `rho = 1`: boundary
  junctions retract from the wall and the films leave it tangentially. The interior
  therefore sits at a higher effective density than `rho`. This matters for the matched
  periodic comparison in step 6.
- Nodes created by remeshing a wall-following film start slightly inside the wall and are
  recorded as arrivals when pressure returns them. Fixed-mesh tests isolate the projection
  error; time-averaged traction is used once remeshing is on.

## Not in scope

Sphere and ring models, wall adhesion, exact circular-arc contact segments, shear rheology,
production ensembles.

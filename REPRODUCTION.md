# Full curved-edge reproduction work

The targets are **2b, 2c, 3a, 4a, and 4g** in Kim et al., *Embryonic tissues
as active foams*, Nature Physics 17, 859–866 (2021). The runner uses the
curved-edge `FoamTissue` for all five panels.

## Sources and numerical choices

- Baseline: branch `claude/paper-reproduction-python-6mhmye`, commit
  `896a7f62c574326e4655daff5403acb6e66143ec`.
- The supplied MATLAB files are preserved in `reference_matlab/`; its
  `provenance.json` records the archive checksum.
- Paper and Methods: https://pmc.ncbi.nlm.nih.gov/articles/PMC8336761/
- Euler–Maruyama dt = 0.005 tau_R, tau_T = 10 tau_R, P0 = 10,
  intermediate-point spacing = 0.15, beta = 0.
- The noise input is MATLAB's `mu`: the interior diffusion coefficient is
  `mu/tau_T`, and a free film receives half that coefficient. It is **not** the
  stationary standard deviation of the OU process.
- `advance_scheduled` checks shrinking short edges every 3 steps, non-adjacent
  T4 at step mod 5 = 1, adjacent T4 at mod 5 = 2, and space creation at mod 5 = 3.
  The T1 sweep visits cell contacts before free boundaries, shuffles each
  class, and rechecks candidate length after accepted transitions. The
  Python sweep batches space-space edge removal after the T1 candidates.
- Equilibrium: initialize spaces, anneal for tau_T at mu=0.5, reset tensions to
  their fixed points, then quench for 3 tau_T. Both phi and z include all cells,
  including cells with zero neighbours. Neighbour number counts distinct cells,
not disjoint contact segments; final edge-face tables are saved for auditing.
- Dynamics: initialize, discard 20 tau_T, measure for 120 tau_T at W=1 (MSD
  through lag 100 tau_T), and 25 tau_T at the other shape-factor points.
- Stress: initialize, pre-equilibrate for 4 tau_T, measure active stress over
  1000 tau_R, apply strain 1.5, and measure relaxation for 580 tau_R. The
  full topology schedule remains enabled after shear; the supplied MATLAB
  step driver instead repeats Type1 in two post-shear schedule slots.
- MSD stores both lab-frame and COM-subtracted results using polygon area
  centroids. The force integrator still uses the boundary vertex mean from
  MATLAB's `ts_faceArea`. The plotted MSD averages lab-frame displacements
  over cells and equally spaced time origins and normalizes by A0. Shape
  factors use the complete curved boundary.

## Topology changes

Closed-film T4 cases now reuse existing physical vertices and free edges,
including contact formation between two isolated cells. Face-edge reconstruction
supports one-junction closed cells and two-junction curved cells with parallel
edges. Small-cell T1 and closed-boundary T3 allow those contacts to separate
again. Ordinary T1 preserves outer-film curvature, retains the pre-transition edge
length (rather than expanding it to the discretization spacing), rotates toward
the correct retained cell sides, and updates all moved endpoints. The transition tests cover formation/separation, positive areas,
edge incidence, endpoint agreement, bridge tension, and sheared periodic images.

Cut positions use nearest finite-segment distance rather than the MATLAB slope
heuristic, which is singular for vertical segments. Periodic unwrapping handles
multiple lattice translations. Non-adjacent T4 sweeps freeze the initial edge
count, as in MATLAB. Geometry guards inspect curved cell boundaries. During measured runs, crossings
within the existing T1 length threshold of a physical junction are counted as
sub-threshold folds (tiny triangular spaces can invert at finite dt); crossings
farther from a junction abort the run. These diagnostics are retained in every
NPZ and `geometry_quality.json`, and flagged on the plots. This criterion is a
resolution diagnostic, not proof that the tiny folds are harmless; timestep and
threshold convergence are still required for quantitative conclusions.

These are numerical implementation changes, **not a claim of bit-for-bit MATLAB
reproduction**. The original Python implementation also contains geometric
acceptance guards and transition acceptance rules. Agreement with published
curves must be assessed from the measured outputs; passing topology tests alone
does not establish quantitative agreement or long-time convergence.

## Acceleration

`FoamTissue(..., accelerated=True)` enables optional Numba kernels for segment
forces, resampling, face geometry, and crossing searches. Random numbers still
come from the same NumPy generator. No fastmath, larger timesteps, or surrogate
model is used. Bounding boxes are reused until a successful T4 changes geometry;
face incidence replaces repeated whole-network edge lookups. Rejected adjacent
repairs no longer trigger an unchanged global crossing search, and sweeps with
no lens candidates skip redundant Python bounding-box construction. Independent runs
use separate worker processes, with BLAS/Numba thread counts capped at one.

Run `scripts/benchmark_foam.py` for a compilation-excluded benchmark. Results
are saved in `data/benchmark.json`; actual wall time depends on topology and
machine load. Force-step acceleration does not imply the same speedup for a
complete parameter sweep. Tests compare accelerated trajectories and transition
searches against the Python path.

## Running and outputs

Use the native ARM `.venv-native` on this Apple Silicon machine (dependencies
pinned in `requirements-lock.txt`). `.venv` is the Intel/Rosetta comparison
environment. The current run is under `data/full_foam_native`:

```bash
.venv-native/bin/python -m pytest tests -q
.venv-native/bin/python scripts/reproduce_requested.py --coarse-grid --seeds 1 --workers 4 --resume --out data/full_foam_native
.venv-native/bin/python scripts/plot_requested.py data/full_foam_native
```

The initial full-duration diagnostic pass used a coarse grid and one seed.
The published ensembles use 20 independent runs per point for 2b/2c and 10
for 3a/4a/4g. Select each group with `--figures` and set `--seeds` separately.
The default map grid uses the publisher workbook's 16 densities
(0.90², 0.92², …, 1.20²) and 41 adhesion values (0, 0.02, …, 0.80).
For the 100-cell map protocol, use `--figures 2b 2c --n-side 10 --seeds 20`.
`--coarse-grid` selects a subset of the published points. Supply custom axes
with `--map-grid` using a JSON object with increasing arrays `w` and `rho`
and an optional `source` field. `--pilot` shortens the dynamics and stress
durations and labels the plots as pipeline validation.

Each run writes an independent compressed NPZ with its parameters, source hash,
seed, raw measurements, and wall time. The manifest lists every requested run.
Failed runs have `.error.json` diagnostics and remain missing; the renderer does
not fill gaps or substitute older/confluent results. Resume keys include the
source hash, so changed simulations cannot silently reuse stale data.

New plots are written as PNG and PDF under `figures/full_foam/`. Maps 2b and 2c
show the adhesion–density plane. Map summary NPZ files include means, SD and
valid-seed counts; `analysis.json` records fitted MSD exponents and stress-fit
parameters. Stress fits use t >= tau_R and do not extrapolate a phase label for
figure 4g. A single seed supplies no between-run uncertainty estimate.
The renderer also writes `fig2bc_model.npz` with the two means and the
`WS` / `RHOS` axes expected by Claude's source-data comparison harness.

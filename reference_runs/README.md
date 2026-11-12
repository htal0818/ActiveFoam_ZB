# Author-code reference runs (Fig 2h-k passive shear)

`ref_stepzero_stress.m` is `ts_simStepZero.m` with the state dumps replaced by a
stress sample every 10 steps. Stage lengths, fluctuation amplitudes, tension
resets, the affine strain step and the transition cadence are copied from it.
`ref_stress.m` is NOT author code: it is the tension term of the paper's Eq. 4/5,
sum over edge segments of max(T,0) * lx * ly / |l|, divided by the box area,
unnormalised. The published archive contains no stress calculation.

The author scripts are not stored here; put the 44 `ts_*.m` files in `author/`.

## Two reference spots (MATLAB, single core)
```
matlab -batch "addpath('author'); addpath('fast_override'); ref_stepzero_stress(10,0.2,1,1.5,1,'trace_rho1_W0.2_seed1.csv')"
matlab -batch "addpath('author'); addpath('fast_override'); ref_stepzero_stress(10,0.5,1,1.5,1,'trace_rho1_W0.5_seed1.csv')"
```
Arguments: box side (10 -> 100 cells), W/T0, psi = sqrt(rho), strain, seed, output CSV
(columns: stage, t relative to the strain step in tau_R, sxy, phi, vertex count).
Strain 1.5 is an assumption: it is the only value in the archive (active-step template).
Octave: replace `matlab -batch` by `octave-cli --quiet --eval`.

## Same-checkpoint replay benchmark
```
matlab -batch "addpath('author'); replay_bench('state15_36cells_v7.mat',100)"
matlab -batch "addpath('author'); addpath('fast_override'); replay_bench('state15_36cells_v7.mat',100)"
```
Expected final line on either path: `sxy=0.833186 phi=0.90932`, V=160, E=240
(Octave 8.4, seed 1; MATLAB uses a different generator, and the replay has zero
fluctuation, so the values should still agree if the two engines agree).
Octave 8.4 on the cloud machine: 1.27 steps/s with the author range check,
3.14 steps/s with `fast_override/ts_edgeRangeCheck.m` (SHA-256 223bfb43...fe821).

## Verification of fast_override/ts_edgeRangeCheck.m (Octave 8.4, 36 cells)
| state | strain | edges | candidate lists identical | total candidates | T4 sweep output identical |
|---|---|---|---|---|---|
| state0  | 0   | 321 | yes | 0  | yes (no cut occurred) |
| state15 | 1.5 | 282 | yes | 24 | yes (no cut occurred) |
Neither state triggered an actual cut, so the sweep comparison is weaker than
Codex's 100-cell MATLAB check; the candidate comparison at strain 1.5 is the
informative one here.

## Shims
`shims/heaviside.m` (0.5 at zero, MATLAB Symbolic convention), `shims/normrnd.m`,
and `shims/rng.m` (no-op, so `ts_initialConf`'s `rng('shuffle')` cannot reseed).
In MATLAB with the Symbolic and Statistics toolboxes, only `rng.m` matters.

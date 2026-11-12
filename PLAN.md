# Shared plan: Figures 2b, 2c, 3a, 4a, 4g on the curved-edge model

Shared code base: branch `codex/reference-init` (commit `3de2965`).
Scoring: published Source Data on static-content.springer.com
(`41567_2021_1215_MOESM11/12/13_ESM.xlsx` = Figs 2/3/4).

## Stage 0 - one base (now)
- Codex rebases local work onto `3de2965`; further handoffs are patches against it.
- Claude keeps the scoring harness and runs sweeps on the cloud machine (4 cores).

## Stage 1 - decide whether 2b/2c is ready for 20 seeds
| # | Check | Owner | Pass criterion |
|---|---|---|---|
| 1a | One-seed full grid (656 pts, 100 cells), error map vs source | Claude (running) | no region outside the transition band with abs(dz) > 0.3 or abs(dphi) > 0.03 |
| 1b | Bistability: 10 seeds at (rho .81, W .5) and (rho 1.0, W .4), MATLAB vs Python, fraction confluent | Codex (MATLAB), Claude (Python) | confluent fractions agree within counting error |
| 1c | W=0 contacts: same seeds in MATLAB at rho .81 | Codex | MATLAB single runs also scatter around 1.2-1.5 |
| 1d | Time-step convergence at dt/2, (rho .81, W 0) and (rho 1.0, W .4) | Codex | phi, z shift < seed s.d. |
Gate: all four pass -> Stage 2. A failure in 1b/1c -> topology fix first; in 1d -> fold fix first.

## Stage 2 - 2b/2c production
- 20 seeds x 656 points x 100 cells: ~35 h on 4 cores here; split by seed with --resume
  (Claude seeds 1-10, Codex 11-20 if the Mac is free). Merge per-run NPZs; source hash must match.
- Deliverable: maps with z = 4, 5, 6 and phi = 1 contours, error map, per-point seed s.d.

## Stage 3 - 3a, 4a, 4g (in parallel with Stage 2, Codex leads)
| Panel | Protocol source | Score against | Known risk |
|---|---|---|---|
| 3a MSD vs activity | ts_simDynamics.m (36 cells, 120 tau_T) | MOESM12 | resolved crossings at high activity |
| 4a stress relaxation | ts_simStep.m (strain step, 580 tau_R) | MOESM13 | post-shear schedule differs from MATLAB |
| 4g shape factor vs W | ts_simDynamics.m | MOESM13 | short runs at non-W=1 points |
- First: 1 seed per curve at published durations; Claude scores against source data.
- High-activity 3a is not reported until its dt/2 check passes (same criterion as 1d).
- Then 10 seeds per point, as published.

## Stage 4 - wrap-up
- One merged branch, REPRODUCTION.md updated with final protocol and every remaining departure
  from the MATLAB (cut-position rule, geometry guards, T1 batching, fold criterion).
- Side-by-side figures and an error table per panel.

## Open questions
- Published caption/Methods could not be fetched from this machine (PMC captcha, Nature blocked);
  ensemble sizes (20 / 10) come from the bioRxiv caption and Codex's notes.
- Should 2b/2c use 100 cells throughout (run template) - yes unless the Methods say otherwise.

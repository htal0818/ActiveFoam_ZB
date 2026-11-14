# One cell to 32, division only

Five synchronous rounds occur at t = 2, 6, 10, 14 and 18 τR, followed by
mechanical relaxation to t = 22 τR. All cells present before a round divide
once at that time. The cuts bisect cell area, approximately perpendicular to
the mother's major area-moment axis, with small deterministic angle adjustments
where needed to avoid existing junctions. This is instantaneous septum insertion,
not a resolved model of cytokinetic furrow ingression.

- Growth rate, tension-noise amplitude and feedback gain are all zero.
- Each daughter inherits half its mother's K (to equal-area numerical tolerance).
  Total preferred area ΣK/P0 stays 1; each final daughter has preferred area 1/32.
- W/T0 = 1, P0 = 10 and τT = 10 in the original founding-cell units.
- The circular wall radius stays 0.7. The cluster does not contact the wall.
- Time step is 0.001 τR; film spacing is 0.025 in founding-cell length units.
  The usual foam topology-transition schedule remains enabled.
- The initial circular cell uses the analytic free-cell equilibrium radius.
  New interfaces then cause mechanical relaxation. Total actual area is not
  constrained: it falls from approximately 0.838 to 0.539. This contraction is
  a pressure/tension response, not growth or removal of K.

`division_1_to_32.gif` shows recorded simulation geometry every 0.1 τR, with
additional frames immediately before and after each division. Pauses at division
events are for display only. Teal and orange identify the two first-generation
lineages. `division_snapshots.png`/`.pdf` show the last recorded frame at each
cell count, with a fixed spatial scale. `division_conservation.png` distinguishes
conservation of K from changes in actual cell area.

Validation: all five rounds succeeded; area at the instant of division and total
K are conserved within the asserted numerical tolerances. There are 447 geometry
and conservation checks (every 0.05 τR plus round/end checks). No resolved film
crossing, nested cell, or wall violation was detected. Transient junction
microfolds occur: up to 20 simultaneous crossings within the existing local
junction tolerance; the reported maximum crossing depth is 0.000883. The final
frame has zero crossings. This is an illustrative trajectory, not a spatial or
temporal convergence study. The 12 biology regression tests pass.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/foam-mpl XDG_CACHE_HOME=/tmp/foam-cache \
  .venv/bin/python circle/scripts/division_lineage.py
```

Use `--render-only` to rebuild the figures from the saved geometry without
rerunning the simulation. Data, checks, parameters and division lineage are in
`circle/data/division_1_to_32_20261114/simulation.json.gz`. The simulation source
SHA-256 is `acf54bc462e7a5fae11c6dd99c8d1273a869a1529a53886de2a5938e8eb90503`.
No model source or planar defaults were changed for this demonstration.

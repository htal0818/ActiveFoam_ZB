# 512-cell hollow epithelial ring

This is an artificial size experiment, not an anatomical cross-section through
a 512-cell embryo. The revised biological interpretation and 48-profile
simulation are documented in [BLASTULA_CROSS_SECTION.md](BLASTULA_CROSS_SECTION.md).

This enlarges the 24-cell ring preview while retaining approximately unit
constructed cell area and the original epithelial thickness. It does not put
512 cells inside the old radius, and it is not a cleavage or growth simulation.
The outer wall is stationary throughout each trajectory.

With `h = sqrt(9 + 24/pi) - 3`, the constructed radii are

```
R_inner = (512/pi - h*h)/(2*h) = 74.97132412
R_outer = R_inner + h         = 76.05047076
R_wall  = 1.3*R_outer         = 98.86561198
```

There are 512 cells, 1024 trivalent junctions, 512 lateral contacts, and two
closed free-boundary loops. Circular boundaries are sampled as polylines.
The initial geometry has no detected film crossings or nested cells. Its
constructed cell areas are approximately 0.99999985 each.

The force equations and transition schedule are those of the 24-cell polarized
ring preview. `W=1`, `P0=10`, `rho=1`, `tauT=10`, `mu=0`, `dt=0.0025`, and
film spacing is 0.075. There is no division, ECM, preferred perimeter or energy
functional. The lumen has the same closed compliance law,

```
p_l(A) = p_l0 + 5*(A_l0/A - 1)
p_l0   = 1/R_outer + 1/R_inner
```

The larger reference lumen area and lower initial pressure are recorded in the
data. This is a geometrically enlarged system, not an isolated test of cell
number at fixed lumen compliance or fixed pressure. The surface remains planar.

The preparation lasts 30 tauR. The selected vegetal sector spans pi/3 and
contains 86 material cells. Their outer apical mean tensions then receive an
increment of 2; actual tensions follow the original OU drift. The requested
activation interval is 30 tauR. A no-activation control starts from exactly the
same prepared state and runs to the selected active snapshot time.

Strict geometry validation occurs every 0.5 tauR. Stored frames are normally
2.5 tauR apart; a stopped run also retains its last validated 0.5-tauR frame.
The existing prototype stops before its first proposed topology operation,
because polarized interface inheritance through rearrangements is unvalidated.
This restriction applies to preparation, activation, and control alike. It is
not evidence for stable invagination or for neighbor exchange.

## Outcome

Preparation completed. The last validated activated snapshot is at `t=39 tauR`,
9 tauR after activation, before the first proposed T1. The matched control
reaches the same time. All sampled geometry checks are clean, with no detected
crossings, tolerated microfolds or nesting. Local cell reshaping is visible near
the interface between activated and unactivated cells; the whole ring remains
approximately circular. Deep invagination has not been established.

| Quantity | Prepared | Matched control | Activated |
| --- | ---: | ---: | ---: |
| Time (tauR) | 30 | 39 | 39 |
| Lumen area | 17686.79374 | 17687.86411 | 17687.72070 |
| Lumen gauge pressure | — | 0.01803020 | 0.01807067 |

The lumen continues expanding slightly in both branches. Activation reduces
that expansion relative to the control and hence gives a slightly higher lumen
pressure. The preparation is a fixed-duration protocol, not a certified global
mechanical equilibrium. The sector-edge close-up rotates all three views by the
same 30 degrees solely for display.

Reproduce the run and figures:

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl \
  .venv/bin/python -m circle.scripts.invagination_size
```

Raw frames, validation diagnostics and parameters are in
`data/invagination_512_20261115/simulation.json.gz`. The rendered summary and
images are in `../figures/circle/invagination_512_20261115/`. The close-ups share
the same absolute spatial window and scale; the control is sampled at the same
time as the activated state. The initial 512-cell geometry and subsequent
snapshots are actual simulation output, not illustrations.

Regression check: `pytest circle/tests/test_invagination.py circle/tests/test_wall.py -q`
passes all 17 tests. No dynamical source code in `activefoam/` or `circle/` is
changed for this size experiment; the new code is a simulation and plotting script.

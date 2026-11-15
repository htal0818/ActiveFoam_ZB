# Hollow epithelial ring: initial geometry and simulation preview

This is a separate constitutive extension, not an unmodified isotropic foam or a
validated gastrulation model. It has 24 epithelial cells, an outward apical surface,
a lumen-facing basal surface, lateral cell-cell contacts, and a fixed frictionless
nonadhesive circular container. No ECM, division, growth or energy functional is
included. The lumen is a hole, not an extra cell. Boundary roles follow oriented
connectivity, so they do not switch merely because a surface moves inward.

The original cell pressure, segment-end tension forces, tension clipping, mobility
and OU form are retained. The user-selected closed, compliant lumen is represented by

    p_l(A) = p_l0 + B_l (A_l0/A - 1),  B_l = 5.

This pressure is a gauge relative to the external bath. It enters only the basal
pressure difference, once per segment end. The reference area is the constructed
lumen's polygon area; p_l0=1/R_outer+1/R_inner is a stated initialization scale,
not a measured embryonic pressure. The continuum closure has effective content
B_l*A_l0 and a constant pressure offset. Fluid exchange is not modeled.

Geometry: inner radius 3, outer radius sqrt(9+24/pi), wall radius 1.3 times the
outer radius. W=1, P0=10, tauT=10, rho=1, dt=0.0025, film spacing=0.075, mu=0.
The constructed annulus relaxes passively for 30 tauR. Four material cells in a
vegetal sector of width pi/3 then receive an apical mean-tension increment of 2.
The actual tension approaches this new mean through the unchanged OU drift;
it is not instantaneously overwritten. Basal and lateral target tensions remain
unchanged in the displayed apical experiment. A matched unactivated control is
run from the same prepared state.

## What the snapshot shows

The displayed active snapshot is 9.5 tauR after activation. It shows local
flattening and reshaping, not deep invagination. Lumen area changes from
30.21535 to 30.11545; gauge pressure rises from 0.25684 to 0.27236. The dashed wall
stays fixed and does no displacement work.

**Topology limitation:** polarized tension inheritance through T1–T4 events has
not been implemented or validated. The preview therefore ends at the first
proposed topology operation; it does not suppress that operation and continue.
The displayed snapshot is the last validated sampled state before this limit.
The apical and combined arms stop at the first proposed T1, after the displayed
9.5-tauR state. The combined arm stops before its delayed lateral activation at
10 tauR, so it is not evidence for a completed combined mechanism. The lateral
arm reaches an 8-tauR snapshot; the unperturbed control completes 30 tauR.
No neighbor-exchange or stable-invagination claim is made from this preview.

The supplied kernel's new optional precomputed pressure-difference argument
leaves the original default path intact. Five new geometry/pressure/polarity/stop
checks and twelve existing wall checks pass. This is an initial implementation
check, not timestep/spacing convergence or validation for Patiria miniata.

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl .venv/bin/python -m circle.scripts.invagination_preview
```

Raw frames: `data/invagination_preview_20261115/simulation.json`.
Images: `../figures/circle/invagination_preview_20261115/`.

# A 48-profile planar cross-section of a sealed blastula

The biological reference count of approximately 512 cells refers to cells on a
three-dimensional embryo surface. It is not the number of polygons required in
a planar section. Uniform surface packing gives the rough estimate
`N_section ~ sqrt(pi*N_surface)`, or 40.1 profiles for 512 surface cells. We use
48 profiles as a provisional initialization within the requested 40–48 range.
This is neither a spherical cellular shell nor an axisymmetric model. Its
dimensions have not been calibrated to microscopy, and the simulation does not
assign a developmental onset time to invagination.

The older 512-polygon ring in `INVAGINATION_512.md` remains an artificial size
experiment. It is not an anatomical section through a 512-cell embryo.

## Geometry and protocol

This run starts with a sealed epithelial annulus. It tests the response of an
already closed tissue to localized contractility. Closure of an initially open,
vented tissue is a different experiment and is not simulated here.

The initial inner/basal radius is 6.53957332, outer/apical radius 7.61871995,
and epithelial thickness 1.07914663 in model length units. The fixed,
frictionless, nonadhesive wall has radius 9.90433594, leaving clearance. Each
cell spans `2*pi/48` radians; its initial inner and outer widths follow from
these radii and angular spacing. Thickness is inherited from the earlier
prototype, not inferred from biological measurements. The constructed annulus
has area 48 before the small polyline discretization error.

The apical boundary faces the external bath and the basal boundary faces the
lumen. There is no division, growth or ECM. Baseline tensions are uniform.
The existing pressure, clipped tension forces, OU update, wall constraint and
transition schedule are unchanged. The compliant lumen uses the previously
specified area law `p_l(A)=p_l0+5*(A_l0/A-1)`, with
`p_l0=1/R_outer+1/R_inner` and reference area 134.35074720. This is an explicit
2D closure law; it is not a three-dimensional lumen-volume calculation.

Parameters: `W=1`, `P0=10`, `rho=1`, `tauT=10`, `mu=0`, `dt=0.0025`,
film-node spacing 0.075. Passive preparation lasts 120 tauR. Over its final
10 tauR, the lumen-area range divided by its mean is `1.5654e-5` (0.0015654%),
and the lumen-pressure range divided by `P0` is `7.5772e-6`. These indicate
nearly stationary global geometry; they do not certify vanishing nodal forces
or numerical convergence.

Eight material cell profiles in the vegetal sector then receive an apical
mean-tension increment of 2. Actual tensions approach that mean through the
existing OU drift. A no-activation control starts from the same prepared state.

## Actual outcome

The last validated active frame is 9.5 tauR after activation. The matched
control reaches the same time. The active branch stops before the first
proposed T1, since the prototype does not yet validate polarized tension
inheritance through rearrangements. It does not silently suppress the event
and continue.

| Quantity | Prepared | Control at +9.5 tauR | Activated at +9.5 tauR |
| --- | ---: | ---: | ---: |
| Lumen area | 138.78030 | 138.78168 | 138.63569 |
| Lumen gauge pressure | — | 0.124534 | 0.129631 |

The snapshots show local cell reshaping and a modest change in the vegetal
contour. Sustained invagination, tissue closure and cell-neighbor exchange
have not been demonstrated. Every sampled strict geometry check is clean.

Changing the ring cell count at fixed radii and thickness changes physical
cell widths and aspect ratios. The suggested 32/48/64-profile comparison is
therefore a separate coarse-graining study, not a completed mesh convergence
test. Film-node spacing and timestep refinement must be assessed separately.

## Reproduction and artifacts

```bash
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl \
  .venv/bin/python -m circle.scripts.invagination_size \
  --n 48 --prep 120 --tag blastula_crosssection_48_20261115
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. MPLCONFIGDIR=/tmp/foam-mpl \
  .venv/bin/python -m circle.scripts.blastula_snapshots
```

Raw data and validation records:
`data/invagination_blastula_crosssection_48_20261115/simulation.json.gz`.
Figures, numerical summaries, a time sequence, and a discrete-frame animation:
`../figures/circle/invagination_blastula_crosssection_48_20261115/`.
The animation uses actual saved frames without interpolation, and playback
time is not physical time.

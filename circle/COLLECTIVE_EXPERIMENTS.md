# Controlled collective-dynamics experiments

These experiments use the original force balance, clipped tensions, OU process,
osmotic pressure, transition rules and rigid one-sided wall. There is no growth,
feedback, energy functional, preferred perimeter or Gaussian surface curvature.
All new code is under `circle/`. The input branch was
`codex/circular-wall-validation` at `efb0f8f`; work continues on the isolated branch
`codex/collective-dynamics-experiments`.

Results: [`../figures/circle/collective_dynamics_20261114/`](../figures/circle/collective_dynamics_20261114/).
Raw results: [`data/collective_dynamics_20261114/`](data/collective_dynamics_20261114/).
The generated summary retains failed runs. A failure is never assigned zero work
or interpreted as fluidity. These are small controlled pilots, not an ensemble
phase diagram.

## Experiments

1. **Local rearrangement and propagation.** For each selected contact, perform
   a quasi-static contraction to the original T1 threshold. Fork its loaded
   state into an original-T1 branch and a release-without-T1 branch. A third
   branch evolves unforced, with connectivity frozen for the same elapsed
   loading time. All three then run the ordinary transition schedule for 10 τR.
   Compare persistent secondary pair changes, centroid displacements and shape.
   Pairs directly changed by the imposed T1 are excluded from secondary counts.
   Reversal of the forced contact is reported separately from a lasting exchange.
2. **Mechanical facilitation.** On one nearby and one distant contact common to
   the loaded, immediately exchanged and unforced states, measure the external
   work for a 10% shortening in four increments. Compare baseline-subtracted
   work per shortening. This is a local resistance measurement, not a complete
   T1 barrier. Incomplete holds and preempted paths are retained but excluded
   from quantitative differences. Each test uses an isolated copy.
3. **Synchronous division timing.** Identical eight-cell founder preparation,
   five rounds to 256 cells; intervals 2, 4 and 8 τR, first division at 2 τR.
   All final generations are observed for exactly four τR. Each run also forks
   a skipped-final-division control at the final division time (128 cells).
   Wall radius `0.7 sqrt(8)`, total preferred area `sum(K/P0)=8`, W=1, zero
   tension noise, zero growth and zero feedback. dt=0.001, spacing=0.025.
   Instantaneous area conservation is checked at division; subsequent actual
   area may relax. Comparisons across intervals use N=256; divide/skip controls
   differ in N, which per-cell normalization does not eliminate. One founder
   realization, so no ensemble error bars are inferred.
4. **Boundary response.** N=36, W=1, rho=1.3, seeds 1 and 2. Disc contacts are
   classified as wall layer if either adjacent cell centroid is within one
   simulation length of the wall, otherwise interior. Periodic controls have
   the same N, pressure convention and preparation sequence. Disc eligibility
   requires actual wall contact. Longest contacts are attempted first, with up
   to two candidates per state; this is a selected-contact pilot, not an
   unbiased contact distribution or a population stiffness estimate.
5. **Preparation dependence.** Repeat the mechanical comparisons with quenches
   of 1 and 5 τT from the same seeded preparation. Contact selection is repeated
   after quenching, so the recorded cell pairs must be checked before treating
   results as same-contact comparisons. The same initial Voronoi seeds do not
   imply identical final topology.
6. **Tension persistence.** N=36, W=1.5, rho=1.3, seeds 1 and 2, τT=2,10,20.
   Burn for 5 τT, then observe for 100 τR (50, 10 and 5 persistence times,
   respectively). dt=0.005, spacing=0.1. Reference noise amplitudes 3, 0.5 and
   0.25 at τT=10 are tested separately; failed stronger-noise runs motivated
   the weaker follow-up. A zero-noise control at τT=10 is also included.
   The actual Euler OU equation has stationary raw contact variance
   `mu²/(2*tauT-dt)`. Therefore set
   `mu=mu_ref*sqrt((2*tauT-dt)/(20-dt))`.
   Free-boundary noise remains half the contact amplitude. This fixes the
   theoretical variance of an unchanged OU contact, not the variance of
   clipped forces or the observed contact population after creation/deletion.
   The latter raw second moment about the fixed tension is recorded separately.

## Measurements and controls

Contact graphs are observed after every ordinary integration/transition step.
A change is counted only after continuously lasting 0.2 τR. Loss and gain are
separate pair changes: one ideal T1 normally contributes two. A loss and recovery
within one timestep is unresolved, rather than classified as a lasting exchange.
Division itself is excluded by resetting the reference graph after each round.
End-of-window changes without enough dwell time are right-censored.

Periodic centroids are unwrapped before computing displacement. Global translation
is removed from MSD. For the disc a separate residual MSD removes translation,
proper rotation and uniform contraction; it does not fit arbitrary shear. A
finite container ultimately bounds displacement, so an MSD plateau alone does
not diagnose rigidity. Spatial differences and repeated time points are not
independent replicas.

Disc validation includes confinement, finite positive areas, per-cell crossings,
inter-film crossings and nesting, with the existing explicitly reported local
junction-microfold tolerance. Periodic runs use the existing planar validation;
it does not provide the disc's independent global inter-film/nesting audit.
This limits any quantitative boundary claim from this pilot. Checks are made
periodically during trajectories and at endpoints, not at every intermediate
sub-operation. The original topology guards remain unchanged.

## Mechanical acceptance gates

`collective_gates.py` checks one complete reference path per domain (N=16,
W=1,rho=1.3,seed=1). Relative work changes under 24→48 shortening increments,
5→10 τR force averaging windows and dt/2 are at most 0.86% (disc) and 1.12%
(periodic). The independently forced weak pulse, after subtracting a zero-force
sham, agrees with the local curve to 0.19% and 0.36%. The last two 20-τR pulse
blocks also settle. All gates use the predeclared 5% tolerance, not a tolerance
chosen after observing the answer. Individual scientific paths must still
complete their own settling and geometry checks; the reference tests do not
establish mesh convergence for every tissue.

The independent legacy `probe.py` curve agrees within 0.40% with matched
resolution and longer holds. Earlier disagreement was not resolved by relaxing
the 3% cross-implementation threshold.

## Reproduce

From the repository root, using the environment's scientific Python:

```bash
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH=.
export MPLCONFIGDIR=/tmp/foam-mpl
.venv/bin/python -m circle.scripts.collective_gates
.venv/bin/python -m circle.scripts.collective_mechanics
.venv/bin/python -m circle.scripts.collective_division
.venv/bin/python -m circle.scripts.collective_activity --mu 3 .5 .25 0
.venv/bin/python -m circle.scripts.plot_collective
.venv/bin/python -m pytest tests circle/tests -q
```

Each runner skips existing result files, including failures. To intentionally
rerun a case, move its result to a separate archive first; do not mix duplicate
realizations into a statistical sample. Partial checkpoints are not plotted as
completed runs. The summary lists every attempted case and its status.

## Comparison with earlier division counts

The new observer samples the contact graph after the complete scheduled step.
The old event observer recorded multiple topology operations with exactly the
same timestamp. Counting these sequentially can reset the persistence timer on
a loss/recovery pair of **zero physical duration**. Thus the two counters need
not give the same finite-dwell count. The new convention counts continuity in
the physical step-end trajectory, without claiming resolution within a timestep.

`audit_collective_division.py` groups the historical events by timestep and then
applies the dwell criterion. It checks agreement with the new event stream at
dwell times 0.1, 0.2 and 0.5 τR, and compares saved polygons with the historical
interval-four run. Changes in counts must not be attributed to changed forces.

The division summary also reports dwell sensitivity at 0.1 and 0.5 τR and the
maximum tolerated microfold count. Passing the existing geometry policy is not
an independent resolution-convergence result for 256 small daughter cells;
the model's geometric thresholds remain in founder units. These four-time-unit
responses establish finite-time remodeling, not sustained fluidity after driving
ceases or a thermodynamic transition.

Selected failed activity points are also rerun using
`python -m circle.scripts.collective_resolution`: keep the same initial prepared
state, halve dt and spacing together, and adjust the noise coefficient to preserve
the coarse run's exact stationary variance. This is a numerical diagnostic with
new discrete noise increments, not a pathwise comparison or a new phase map.
Finish the report with `python -m circle.scripts.audit_collective_division` and
`python -m circle.scripts.report_collective`. The latter refuses an incomplete
batch and writes the code hashes, package versions, tables and plots.

## Topological meaning of the imposed T1

In the selected prepared foams, both endpoints of each probed contact adjoin an
extracellular space (`opposite = [-1, -1]`). The original foam T1 consequently
removes the contact into a space, without immediately creating a new cell-cell
pair. Subsequent reclosure is correctly classified as a reversal. These completed
paths measure **contact-opening work**, not an observed lasting four-cell swap.
The distinction is recorded explicitly; no transition rule is changed to force
the paper's confluent topology onto the foam.

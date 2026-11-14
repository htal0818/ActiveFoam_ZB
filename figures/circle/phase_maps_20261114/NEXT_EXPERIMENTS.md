# Next experiments for active foam in a rigid circular container

These are proposed follow-up experiments, not additional simulations already run.
The fixed, frictionless, nonadhesive wall and flat tissue surface remain the baseline.
The current phase maps are finite-time response maps; they do not classify a fluid/solid
transition. High-activity geometry failures must not be treated as a physical phase.

## Numerical gate before expanding activity

The current high-activity runs fail geometry checks. Before interpreting that regime
or extending its duration, compare a few matched seeds under timestep and film-spacing
refinement (for example dt=0.005 versus 0.0025 and spacing=0.15 versus 0.075).
Keep acceptance criteria fixed. Record where the first invalid geometry appears and
which transition or integration step precedes it. A failure threshold is not a phase
boundary; relaxing the guard to complete a run would not establish valid dynamics.

## First priority: expose rearrangements without changing the force law

1. **Joint adhesion and activity.** The current adhesion maps are passive and the
   activity maps fix W/T0=1. Sweep active adhesion points around W/T0=0.5,1,1.5,
   using low-to-moderate activity first. After numerical checks, add finer
   lower-adhesion points (for example 0.25,0.35,0.5) and densities around or
   below rho=1 to search for opening spaces and loss of connectedness. The
   current grid mostly samples contact-rich interiors; do not assume a known
   critical adhesion in the circular domain. Changing W changes the mean contact tension
   2T0-W, and hence the relative importance of a given tension fluctuation. Begin
   with 36 and 100 cells at rho=1.2 or 1.3, validate trajectories, then expand the grid.
2. **Longer observations at validated points.** A starting target is 20 tauT of
   burn-in and 50–100 tauT of observation, with checkpointed geometry checks and
   multiple seeds. Establish stationarity from time blocks rather than assuming
   a chosen burn-in is sufficient. These are targets, not evidence that the
   present integrator will remain valid for that duration at every point.
3. **Density near wall attachment.** Refine rho around the observed crossover from
   a detached cluster to substantial wall contact. Measure attachment fraction,
   packing, contact lifetimes, and rearrangements separately in wall and interior
   populations. Also distinguish cells next to the tissue’s own free surface from
   cells deep inside the tissue: distance from the container wall alone does not
   identify true tissue bulk when the cluster detaches. Avoid interpreting rho
   itself as the measured area fraction.

## Measurements that distinguish different dynamics

| Measurement | Physical question | Additional data needed |
|---|---|---|
| Persistent neighbor overlap and contact survival | Do cells permanently exchange neighbors, or repeatedly undo local changes? | Initial contact graph plus timestamped pair gains/losses; resolve transient reversals |
| T1 event waiting times and spatial clusters | Are rearrangements independent, intermittent, or organized into cascades? | Accepted event time, location, affected cells, and before/after contact pairs |
| Neighbor-relative MSD and self-overlap | Are cells escaping their cages, or moving together? | Unwrapped per-cell centroid trajectories and a defined initial neighbor set |
| Displacement distribution, non-Gaussian parameter and chi4 | Is motion heterogeneous even when the mean MSD is small? | Per-cell displacements over multiple lag times and independent seeds |
| Radial profiles and wall-contact residence times | Does the boundary promote or suppress rearrangement? | Time-resolved radius, contact state, shape, velocity and event locations |
| Tension–length–shape lag correlations | Do tension fluctuations precede contact shortening and rearrangement? | Film tension/length histories with stable IDs across topology events; cell shapes |
| Contact-oriented shape tensor and velocity correlations | Is there orientational order or collective motion? | Cell shape tensors and spatially resolved velocities |

Accepted T1 operations include boundary operations and reversals. An event catalog
should distinguish operations that change cell neighbors from those that do not.
A cascade analysis should report sensitivity to its spatial/time clustering cutoffs
and compare with shuffled event catalogs preserving the observation window and rate.
Cell points and time origins are correlated; uncertainty is estimated across seeds
or suitable time blocks, not by treating all cells as independent replicas.

## Controlled extensions that can generate new dynamics

| Extension | What to vary | What it can test |
|---|---|---|
| Fixed noncircular container | Equal-area ellipse or stadium, at fixed N and preferred cell area | Local curvature effects without changing the total cell count; the wall stays rigid |
| Tension persistence | tauT/tauR, for example 2,5,10,20 | Fast fluctuations versus sustained forcing and delayed rearrangements |
| Pulsed or patterned activity | Uniform pulse, wall-layer activation, or one angular sector | Relaxation, propagation of rearrangements and boundary-to-bulk coupling |
| Spatially correlated tension noise | Correlation length, keeping marginal noise variance fixed | Collective deformation and correlations between rearrangements |
| Preferred-area or osmotic pulse | Small transient change in A0 or pressure control, fixed wall | Rate-dependent compaction, relaxation and hysteresis |
| Delayed mechanosensitive tension feedback | Feedback strength and delay | Whether feedback can produce pulsations or instabilities; this changes the original model |

For a persistence sweep, use two clearly separated protocols. With the current OU
normalization, a fixed interior contact has stationary raw variance mu^2/(2*tauT).
Holding mu fixed therefore changes both persistence and variance. To hold the
continuum variance fixed relative to a reference run, use
`mu(tauT)=mu_ref*sqrt(tauT/tauT_ref)`. The exact Euler–Maruyama variance uses
`2*tauT-dt` in place of `2*tauT`, so its matched-variance scaling replaces that
square-root ratio accordingly. Keep the existing half-amplitude convention at free
boundaries. Distinguish these protocols in labels and metadata.

A fixed stadium or ellipse is especially useful for separating local wall geometry
from global size. A circular wall couples curvature, density and N through
`kappa*L0=sqrt(pi*rho/N)`. At fixed N, changing rho also changes curvature in
preferred-cell units. The maps report both `kappa*ell` and `kappa*L0`; equal-area
noncircular containers provide different local curvatures within one fixed-size
system. This extension requires the correct closest-point projection and wall
normal, rather than reusing the circle’s radial projection.

Local velocity vortices can be measured, but a frictionless circular wall contributes
no torque about its center. Persistent net rotation should not be assumed; introducing
an explicit chiral drive would be a separate extension of the force law.

Implement event and trajectory recording first, then joint active adhesion and longer
validated runs. Persistence and patterned-activity experiments are the next controlled
extensions; feedback, proliferation and other topology-changing biology come later.

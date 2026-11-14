# Force-based mechanical probes: specification and execution gates

**Implementation in progress; item 1 acceptance gates are not yet passed.**
The first random-tissue smoke checks did not converge within their 60-τR holds.
After extending the hold budget and testing drift of block-mean force rather
than requiring instantaneous force fluctuations to vanish, complete trial curves
were obtained on both 16-cell domains. These are implementation smoke checks,
not accepted work measurements: step-count, hold-duration, dt and independent
force-pulse agreement gates still need to be executed. Seven implementation
tests pass. Trial outputs and the preparation parameters are saved under
`data/force_probe_smoke_20261114/` for review.
No disc/periodic work comparison, adhesion scan, size series, division-response
scan or activity comparison is claimed for this new probe. Items 2–6 remain
blocked by the numerical gates below. Existing historical experiments are
separate from this protocol.

The dynamics, OU drift/noise, pressure, tension clipping, topology operations and
unilateral frictionless wall are unchanged. All new code lives in `circle/`.
No preferred perimeter, target shape index, energy functional or critical shape
index is introduced. The disc remains flat: comparisons concern boundary and
finite-size effects. The user's supplied summary of Thomas & Hopyan motivates
the measurements; direct access to its full text has not been verified here.

Implementation order: (1) quasi-static forced contact shortening and acceptance
gates; (2) compressed disc versus equally sized periodic samples with matching
preparation/pressure conventions, separating wall layer from interior; (3)
adhesion scan; (4) equal-N size series spanning 24–128; (5) divided versus
skipped-division branches at the same time; (6) matched tension-noise dynamics.
The optional no-detachment boundary needs separate user approval and is not
implemented. Force-Jacobian soft modes remain deferred.

## External control and work convention

`force_probe.py` operates unchanged on `DiscFoam`, periodic `FoamTissue` and
passive `BiologyFoam`. Every operation runs on an isolated copy of the complete
instance. The supplied object's physical state, RNG, parameters, diagnostics,
original arrays and identity-keyed caches are never mutated, including when a
probe fails unexpectedly. Cache invalidation is confined to the working copy.

At zero activity with topology frozen, call the ordinary `step(0)` and then move
the two selected junctions along their joining line by equal and opposite
external displacements. All shared film endpoints follow their physical junction.
With trial separation d and target q, each correction has magnitude (d−q)/2.
The compressive holding force on either junction is therefore

    F = (d − q) / (2 psi dt).

There is no additional factor of two in work: total pair work is F times the
reduction in separation. Pressure and tension drift continue to evolve during
holds exactly as in the original zero-noise integrator. Film remeshing continues;
connectivity stays fixed until the explicitly requested T1.

First hold q at its initial value to measure F0. Then reduce q in discrete steps.
At each target require the means of three consecutive five-τR force windows to
satisfy the absolute/relative drift tolerances. Instantaneous within-window SD
is retained as a diagnostic, not required to vanish: in the initial disc test
the late mean was approximately 0.020 with SD 0.028, consistent with a substantial
force ripple. A mean plateau alone is not sufficient acceptance; work must also
pass the independent dt, increment-count and hold-duration refinements below.
Nonconverged holds are reported and are not accepted as quasi-static data.
The finite instantaneous displacement at a target change is excluded from force
plateaus; its numerical impulse is not counted as a mechanical barrier.

Report the settled F(q), initial tangent stiffness versus shortening s=q0−q,
peak F, signed work W=integral F ds, peak cumulative work, and W/s. Also report
baseline-subtracted force and work, keeping both conventions explicit. Negative
work is possible and is retained; these quantities are external mechanical work,
not changes of a presumed state-function energy.

The existing T1 trigger is full polyline length 0.01*2*sqrt(pi), not junction
separation. Record both. By default the final junction target is 0.98 times that
length; if the relaxed film is still too long, report preemption rather than
call an out-of-threshold T1. A newly short competing film or invalid geometry
also preempts the probe. Existing short films are recorded separately. Wall
penetration by the external control is preempted rather than silently projected
into a different, non-central force. The wall condition itself is unchanged.

After a valid terminal hold, call the existing T1, release the external control
and resume the original topology schedule. Classify by the original and opposite
cell pairs as exchange, reversal, space, T1 declined or preempted, and retain the
immediate and final contact graphs. The observation duration is always stated.

## Acceptance gates before scientific comparisons

- Exact input-state preservation after success, expected failure and unexpected
  exception, including RNG and caches; no schema-based omissions.
- At least one complete reference path in each domain. Double shortening
  increments, lengthen holds, and halve dt. Pilot tolerance for total work is
  5% relative, with an absolute floor of 1e-4 force×length units. A failed gate
  blocks scientific work comparisons; it is not relaxed to obtain a pass.
- An independent weak constant-force pulse, subtracting its zero-force sham,
  agrees with the local stepwise force curve within 5% or 1e-4 length units.
  The pulse must settle; transient ramp work is not used to establish agreement.
- Record preemptions, convergence failures and geometry diagnostics. Never
  average incomplete paths as zero work or infer a physical phase from failures.

The model's existing junction-microfold tolerance is reported, not strengthened
by a passing probe test. Pressure convention is boundary normals with current
areas in both domains. All science comparisons must retain matching conventions
and compressed disc states with measured wall contact.

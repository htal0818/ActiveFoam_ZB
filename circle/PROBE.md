# Forced neighbour exchange: a rigidity probe taken from Thomas & Hopyan (2023)

Source: E. C. Thomas and S. Hopyan, *Shape-driven confluent rigidity transition in curved
biological tissues*, Biophys. J. 122, 4264–4273 (2023). The full text was read from a
copy supplied by the user. Code: [`probe.py`](probe.py), tests:
[`tests/test_probe.py`](tests/test_probe.py).

**Status:** this document describes the earlier independent prototype.
The gated implementation used for the new experiments is
[`force_probe.py`](force_probe.py), documented in
[`MECHANICAL_PROBES.md`](MECHANICAL_PROBES.md). New results and exact commands are
in [`COLLECTIVE_EXPERIMENTS.md`](COLLECTIVE_EXPERIMENTS.md). The legacy pilot
scripts below were not used for these results.

The old `probe.save/load` preserves selected simulation fields and resets some
caches; it does **not** satisfy the stronger requirement of preserving every
cache and diagnostic exactly. New probes run on isolated copies and leave the
supplied object, including its cache identities, untouched. Cross-validation of
this older routine is performed on such a copy.

## What the paper does

- **Model.** A vertex model with energy `sum_i [(a_i - 1)^2 + (p_i - p0)^2 / r]`, on a flat
  torus, on spheres (Gaussian curvature `4*pi/N`) and on bounded hyperbolic discs.
- **Three probes of rigidity.**
  1. *Local barriers.* Each edge is shortened to a point in small steps while the rest of
     the tissue relaxes; the change of energy is the barrier to that neighbour exchange.
  2. *Zero modes.* The fraction of zero eigenvalues of the Hessian at a relaxed state.
  3. *Finite temperature.* Mean-square displacement of self-propelled cells.
- **Curvature result.** The critical shape index falls with positive Gaussian curvature
  and rises with negative curvature.
- **Boundary result.** The hyperbolic discs need a hard boundary, so the authors repeat
  the *flat* calculation on a bounded disc. Their boundary vertices are pinned to the
  boundary and slide along it. Compared with the periodic sheet, the bounded flat disc
  has higher barriers, a higher critical index and a more gradual transition. They
  attribute this to boundary nodes moving in one dimension only, which suppresses
  long-range collective modes.
- **Finite size.** Smaller systems give lower apparent critical values.
- **Division.** Not in their model. They expect division to make tissue rearrange more
  easily and suggest it as a follow-up.

## What transfers to the disc foam, and what does not

| Paper | Disc foam | Status |
|---|---|---|
| Energy barrier to a forced T1 | mechanical work to shorten a contact to the T1 length | implemented, `probe.contract_contact` |
| Outcome of the forced T1 | apply the model's T1 operation, let go, classify | implemented, `probe.force_exchange` |
| Periodic sheet vs bounded flat disc | matched periodic `FoamTissue` vs `DiscFoam` | script written, not run |
| Finite-temperature MSD | tension-noise runs with motion measurements | already in `EXPERIMENTS.md` |
| Division expected to fluidise | probe a divided and a skipped-division branch | script written, not run |
| Zero modes of the Hessian | soft modes of the force Jacobian | deferred (see limits) |
| Gaussian-curvature phase diagram | does not apply: the tissue surface is flat | not imported |
| Target shape index `p0`, value 3.813 | no counterpart: the foam has no preferred perimeter | not imported |

Three differences matter when reading any comparison.

- **No energy.** The foam is defined by force balance with tension clipped at zero.
  The probe therefore measures force and work done by an external agent. No energy
  functional is introduced or required.
- **Curvature.** The paper's control parameter is Gaussian curvature of the surface.
  The disc's surface has none. Its wall has curvature `1/R_w`, coupled to system size
  through `R_w = sqrt(N/pi)`. A size dependence in the disc is a boundary and finite-size
  effect, not the paper's curvature effect.
- **Boundary type.** The paper's boundary vertices cannot leave the boundary. The
  disc's wall is one-sided: cells can detach. The paper's stiffening argument relies on
  the boundary removing freedom, so it need not carry over.

## Method

An external agent applies equal and opposite forces to the two junctions of one
cell–cell contact, along the line joining them. It enters by operator splitting: after
each ordinary step the two junctions are displaced by `M*dt*f` with `M = psi`, the
mobility of every other force. The force law, tension dynamics and transition rules are
unchanged, and the foam is restored exactly after each probe.

**Contraction (`contract_contact`).** Connectivity frozen, zero activity.

1. Hold the junction separation at its initial value `d0`; the holding force is the
   baseline of the prepared state (zero at a true equilibrium).
2. Shorten by 3% of `d0`, then in 8 equal steps down to the model's T1 length
   `0.01*2*sqrt(pi)`.
3. At each level hold the separation until the holding force, averaged over 1 τ_R
   windows, changes by less than `2e-3*(1+|f|)` between windows (at most 15 τ_R).
4. Holding force on each junction = the correction that restores the separation after a
   free step, over `2*M*dt`.

Reported: force at each level relative to the baseline, the stiffness (force over
shortening at the 3% level), the peak force, and the **work**, the trapezoidal integral
of force over shortening. Work divided by total shortening is the mean resisting force.

**Forced exchange (`force_exchange`).** After the contraction, the model's own guarded
T1 operation is applied to the contact, space–space films are removed as the schedule
would, and the foam then evolves freely with all transitions for 10 τ_R. Outcome:
`exchange` (pair apart, new contact formed), `reversal` (original contact back), `space`
(pair apart, nothing formed), `T1 declined`, or `preempted` (another contact reached the
T1 length first). Changes count only if they have lasted 0.2 τ_R.

**Weak pulse (`dipole_pulse`).** A constant force is held and released. Used for the
linear response and as a cross-check; a zero-force sham run removes the drift of the
prepared state.

## Validation

Tests in [`tests/test_probe.py`](tests/test_probe.py), 16 cells, `W/T0 = 1`, `rho = 1.3`,
`dt = 0.005`, spacing 0.15.

| Check | Result |
|---|---|
| Legacy probe restoration | selected state and generator; see cache/diagnostic limitation above |
| Work with 8 vs 16 levels | 0.6359 vs 0.6348 |
| Work with holds up to 15 vs 40 τ_R | 0.6359 vs 0.6299 |
| Work at `dt` vs `dt/2` | 0.6359 vs 0.6356 |
| Stiffness with holds up to 15 vs 40 τ_R | 1.62 vs 1.56 |
| Force control against separation control | a held force of 0.100 sits at 0.103 on the stepwise curve |
| Recovery after a weak pulse | 1.2% of the shortening unrecovered after 30 τ_R |

A continuous ramp was tried first and rejected: at shortening rates of 0.1, 0.05 and
0.025 per τ_R the work was 0.59, 0.53 and 0.49, still falling.

## Limits

- **Stiffness is the least converged quantity** (4% between hold criteria): the tissue
  creeps slowly under a held separation.
- **Prepared states drift.** A contact's separation changed by up to 0.006 over 60 τ_R
  without any forcing. The baseline hold and the sham run account for it; they do not
  remove it.
- **The transition schedule will not remove a contact that is held shut.** Its T1
  trigger asks that the contact shrink across a free step. `force_exchange` therefore
  applies the T1 operation directly.
- **Frozen connectivity during contraction.** Neighbouring contacts cannot rearrange
  while the probed one is shortened. A probe is marked `preempted` if a contact that
  started above 1.5 T1 lengths reaches the T1 length first. Prepared states contain
  contacts just above that length which cross it through drift alone; counting one such
  contact made 11 of 12 probes on a 100-cell disc `preempted` in the first, aborted pilot.
- **The T1 length is absolute.** In runs where cells shrink (division without growth)
  contacts approach it, and the work to reach it falls for that reason alone. Any
  before/after-division comparison has to normalise for this.
- **No zero-mode analysis.** Film nodes slide along films and along the wall at no
  cost, so a force Jacobian has many trivial soft modes that would have to be removed
  before counting.

## Plan

| # | Step | Status |
|---|---|---|
| 1 | Probe with stepwise holds, state restoration, tests | done |
| 2 | Disc vs matched periodic box, three adhesions, two sizes | script written, not run |
| 3 | Divided branch vs skipped-division branch | script written, not run |
| 4 | Size series at fixed adhesion to separate wall from finite-size effects | open |
| 5 | Soft-mode count after removing sliding modes | deferred |

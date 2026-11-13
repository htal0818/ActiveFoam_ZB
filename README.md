# Active Foam — Python reproduction of Kim *et al.*, *Nature Physics* (2021)

Python reimplementation of the dynamic vertex ("active foam") model from

> S. Kim, M. Pochitaloff, G. A. Stooke‑Vaughan & O. Campàs,
> **"Embryonic tissues as active foams"**, *Nature Physics* **17**, 859–866 (2021).
> https://doi.org/10.1038/s41567-021-01215-1

The reference implementation shipped with the paper is a set of MATLAB routines
(`ts_*.m`). This repository ports the model to a compact, vectorised NumPy code
and reproduces the paper's key quantitative figures, shown side‑by‑side with the
originals in [`figures/`](figures/).

---

## The model

Vertices (triple junctions) move by overdamped dynamics (paper Eq. 1)

```
η_R dR_α/dt = Σ_{i,j∈F(α)} ( T_ij Θ(T_ij) + N_ij )
```

* **Junctional tension** `T_ij` pulls each cell–cell contact; it is an
  Ornstein–Uhlenbeck process (Eq. 2) fluctuating with amplitude `ΔT` around a
  fixed point `T⁰_ij = 2T₀ − W` (interior) / `T₀` (free), with persistence time
  `τ_T`. `Θ` is a Heaviside preventing negative tensions.
* **Normal (osmotic pressure) force** `N_ij = (Δp_i − Δp_j)ℓ_ij/2` with
  `Δp_i = P₀(A₀/A_i − 1)` pushes cell boundaries outward.

### Parameter mapping (decoded from the MATLAB code)

| Paper symbol | Code arg | Value used | Meaning |
|---|---|---|---|
| `W/T₀`      | `w`    | swept       | relative cell–cell adhesion → `T⁰_int = 2 − w` |
| `ΔT/T₀`     | `dT`   | swept       | tension‑fluctuation magnitude |
| `ρ ≡ NL₀²/A_T` | `rho`  | 1 (`= ψ²`) | density; preferred area `A₀ = ρ` |
| `P₀L₀/T₀`   | `P0`   | 10          | relative pressure force |
| `τ_T/τ_R`   | `tauT` | 10          | tension persistence (τ_R ≡ 1) |
| `Δt`        | `dt`   | 0.005 τ_R   | Euler–Maruyama step |
| —           | `l_t1` | `0.01·2√π`  | short‑edge threshold for T1 (the code's `shEd`) |

A short derivation (in `activefoam/topology.py`) shows that with a **single
straight segment per contact**, the model's tension + pressure vertex forces are
**algebraically identical** to the reference code's per‑edge forces
(`ts_iteration.m`, `ts_edgeNormalForce.m`) — i.e. `Δp_i · ∂A_i/∂R_α` exactly
equals the code's distributed normal force. This reproduction therefore uses the
same force law, restricted to the confluent regime.

## What is reproduced

| Figure | Quantity | Script |
|---|---|---|
| **2a** | Curved‑cell configurations with extracellular spaces vs `W/T₀` | `scripts/make_fig2a_panel.py` |
| **2b/2c** | Volume fraction `φ` and contact number `z` vs `W/T₀`, `ρ` | `scripts/run_fig2b_foam_full.py`, `scripts/run_fig2c_map.py` |
| **2f** | Foam‑limit jamming: `z` vs `φ`, `φ_c≈0.83`, `z_c=4` | `scripts/run_fig2f_foam.py` |
| **2g/4g** | Cell shape factor `s̄` vs `W/T₀` (equilibrium & dynamic) | `scripts/run_shapefactor.py` |
| **2i** | Shear‑stress relaxation `σ_xy(t)/σ₀` vs `W/T₀` (paper Eq. 4/5) | `scripts/run_fig2i_stress_relax.py` |
| **2j** | Yield stress `σ_Y` vs `W/T₀` (paper Eq. 4/5) | `scripts/run_fig2j_yieldstress.py` |
| **3a** | MSD(t) vs `t/τ_T` for varying `ΔT/T₀`, and exponent `α(ΔT)` | `scripts/run_fig3_msd.py` |
| **3c/3d** | MSD and cellular NE rate vs `W/T₀` (minimum at the structural transition) | `scripts/run_fig3cd.py`, `scripts/run_fig3cd_confluent.py` |
| **4a** | Shear‑stress relaxation → stretched exponential | `scripts/run_fig4a_stress.py` |
| **4d/4e** | `τ_SR` map and fluid/solid phase diagram over `(W/T₀, ΔT/T₀)` | `scripts/run_fig4de.py` |

Each driver saves `data/*.npz`; `scripts/make_comparisons.py` renders the
side‑by‑side comparison PNGs into `figures/`.

### Stress, yield stress and stress relaxation (Figs 2i, 2j, 4a, 4d, 4e)

These use the paper's **Methods stress tensor** (Eq. 4/5),
`σ_mn = ρ [ −Σ_i Δp_i a_i δ_mn + Σ_ij t_ij ℓ_m ℓ_n / |ℓ| ]`, implemented on the
curved‑edge foam model (`FoamTissue.stress_tensor`), together with the affine
shear‑step‑and‑relax protocol (`apply_affine_shear`). In Fig 2i the elastic jump
falls with adhesion (0.95, 0.63, 0.32 at `W/T₀` = 0.5, 1.0, 1.5 for a strain of
1.0; the paper uses 1.5) and the stress is zero at `W/T₀=2`. The residual stress
after 30 τ_R is 0.41, 0.39, 0.20, against 0.37, ≈0.24, ≈0.11 in the paper at
long times. There is **no `W=0` curve**: both seeds at that adhesion tangled and
ran past the per‑run time limit. In Fig 2j the yield stress peaks at
`W/T₀≈0.4` (0.32) and vanishes at `W/T₀=2`; the `W=0` value (0.12 ± 0.09, two
of three seeds) is not reliable.

All foam‑model figures were regenerated after the space‑annihilation step was
removed from the model (see the transition inventory below).

**Fig 3c/3d.** The foam model covers all adhesions but only over a 20 τ_R
window (NE-rate floor ≈ 7×10⁻⁴, MSD lag 2 τ_T). The NE rate falls steeply from
the non-confluent side to a minimum near W/T₀ ≈ 0.6–0.9, but the rise on the
confluent side is weak: at ΔT/T₀ = 1.0 it goes 0.010 → 0.024 and at 1.5 it is
flat at ≈ 0.03–0.04 (paper ≈ 0.03 → 0.15). Lower
activities are floor-limited, so on the confluent branch (W/T₀ ≥ 0.6) the fast
confluent model is run for 2000 τ_R, giving MSD at the paper's 10² τ_T lag and
NE rates down to ~10⁻⁵. Its NE rate counts net contact losses between snapshots
one τ_T apart, while the foam model counts T1 events; at ΔT/T₀ ≤ 0.5 the two
differ by one to two orders of magnitude, so the sets are drawn separately
rather than joined.

For the **τ_SR map (4d)** and **phase diagram (4e)** the paper states that
long‑timescale stress relaxation is driven by *actively induced NE (T1)
transitions*, with the stress‑relaxation time set by the inverse cellular NE
rate. We therefore measure the steady‑state NE (topological‑event) rate `k_NE`
directly from the foam model across `(W/T₀, ΔT/T₀)` and set
`τ_SR = τ_T · (k_NE^max / k_NE)` (stress relaxes within one tension‑persistence
time at the fully‑fluid NE ceiling, diverging as the NE rate → 0). Applying the
paper's fluidity criterion `τ_SR/τ_T = 10²` gives only two solid points
(`ΔT/T₀ = 0.5`, `W/T₀` = 0.6 and 0.8), next to the structural transition; the
paper's solid region is much larger (up to `ΔT/T₀ = 1.0`, `W/T₀` ≈ 0.4–1.2).
The foam model's T1 rate on the confluent side is too high to reproduce it.
This is a reduced NE‑rate reconstruction, not a per‑point stretched‑exponential
fit of the full stress‑relaxation curve.

## Faithful vertex-network foam model (`foam_full.py`)

`foam_full.py` ports the reference model's **actual data structure** rather than a
confluent surrogate:

* **physical vertices** — triple junctions (3 incident edges / 3 faces each);
* **intermediate vertices** — extra points along every contact, re-sampled by
  `edge_mid_vrtx_average` (port of `ts_edgeMidVrtxAverage.m`), giving each contact
  **curvature** (paper Fig 1g, blue dots);
* **extracellular spaces** — encoded as face id `-1` (the MATLAB's face 0), with
  free-boundary tension `T₀` and cell–cell tension `2−W`;
* forces from a direct port of `ts_iteration.m` / `ts_edgeNormalForce.m` /
  `ts_edgeVector.m`;
* **T2-reverse** (`ts_t2ReverseTransition.m`) opens a triangular space at every
  vertex — the paper's space-initialisation protocol.

`figures/compare_fig2a_configs.png` shows the result: curved cells with
adhesion-dependent extracellular spaces, matching Fig 2a.

### Topological-transition inventory

| Transition | Meaning | Status |
|---|---|---|
| **T1** | neighbour exchange (edge flip) | ✅ confluent + foam model (`t1_foam`, handles adjacent spaces), validated |
| **T2-reverse** | create a triangular extracellular space at a vertex | ✅ `ts_t2ReverseTransition` port (`t2_reverse`) |
| **T2** (space annihilation) | remove a collapsed space | — not in the model: neither the reference code nor the paper's Methods removes spaces; a closed space persists as a tiny triangle |
| **T3-reverse** | merge two spaces across a space–space edge | ✅ `ts_t3ReverseTransition` port (`t3_reverse`) |
| **T4 (adjacent)** | uncross two curved edges *sharing a vertex* that folded through each other | ✅ `ts_t4AdjacentTransition` + `ts_edgeCrossAdjacent` port (`t4_adjacent`), with revert guard, validated |
| **T4 (non-adjacent)** | cut two *non-adjacent* space-bordering edges crossing at two points (a lens) and splice a bridge edge | ✅ `ts_t4Transition` + `ts_edgeCutPiece` + `ts_topTransitionType3` port (`t4_transition` / `top_transition_type3`), validated |

**The contact-breaking chain now works.** As a space grows, a cell–cell contact
collapses → **T1** flips it into a space–space edge → **T3-reverse** merges the
two spaces, so the two cells lose their contact. The unified foam model therefore
reproduces **both φ(W/T₀) and z(W/T₀)** on its own — z rises from the foam limit
toward 6 with adhesion, matching the paper for W/T₀ ≳ 0.5.

**W→0 over-fragmentation — reduced, not solved.** Before the non-adjacent T4
was ported, contacts could break (T1 → T3-reverse) but never re-form, so the
foam limit over-fragmented (z ≈ 2.8–3.2 at W=0, ρ=1). The non-adjacent T4 is the
reference code's contact-*formation* step; with it the same point gives
z ≈ 4.5–4.7, above the isostatic value 4; the published source data give
z = 4.09 (φ = 0.830) at that point.

What remains broken in the low-adhesion corner:

* **Tangling.** In the (W/T₀, ρ) map (`run_fig2c_map.py`) 8 of 64 points —
  all at W/T₀ ≤ 0.2 — produced no valid configuration (self-overlapping cells,
  φ > 1, or a run past the time limit). In the coarse sweep on the published
  grid (`run_fig2bc_coarse.py`) it is 5 of 54, again all at W/T₀ ≤ 0.2. Under
  activity, 5 of 72 runs in the Fig 3c/3d sweep were invalid, all at
  W/T₀ ≤ 0.15 (no valid seed at W=0, ΔT/T₀ = 1.5). These points are excluded
  and marked, not filled in.
* **Late confluence at low density.** Against the published Fig 2b/2c source
  data the coarse sweep has RMS differences of 0.067 in φ and 0.27 in z over
  its 49 valid points. The largest errors are at ρ = 0.81–0.92, where the
  model is still non-confluent (φ ≈ 0.78) at W/T₀ = 0.6–0.8 and the published
  data are confluent from W/T₀ ≈ 0.58. Cell–space edges are never flipped by
  T1 here (the reference code does flip them), so a space with four or more
  sides cannot shed sides and close.
* **Transient coarsening.** After spaces are opened they keep coarsening, so
  low-adhesion NE rates and MSDs are partly non-steady-state.
* The underlying corner force balance (a contact of tension 2−W against two
  free edges of tension 1 each; Supplementary Section 1) is still only
  approximately captured by the refined polyline edges.

**T4 — both modes ported.** Two curved edges *sharing a vertex* that fold
through each other are resolved by `t4_adjacent` (port of
`ts_t4AdjacentTransition` + `ts_edgeCrossAdjacent`): the shared vertex is moved
onto the edges' intersection point and both folded-over pieces are trimmed to
it, with the reference's revert guard (undo if an edge collapses below 5 % of
its length or an incident face area drops below `10⁻²`).

Two *non-adjacent* space-bordering edges (films) that cross at **two** points —
a lens — are resolved by `t4_transition` (port of `ts_t4Transition` +
`ts_edgeCutPiece`, driven over every space-bordering edge by
`top_transition_type3` = `ts_topTransitionType3`). Following the reference
general case, each edge is cut at both crossing points and re-spliced into a
first piece, a second piece and a shared **bridge** edge — adding exactly two
trivalent vertices and three edges, with the bridge tension set by the code's
fixed-point averaging rule. A validity guard reverts the cut if it would leave a
cell degenerate or self-intersecting. Both modes are exercised by the test suite
(`test_t4_adjacent_resolves_fold`, `test_t4_transition_cuts_lens`). They run in
the MATLAB order inside `do_transitions` (T4-adjacent → T1 → T3-reverse →
T4-non-adjacent); `advance` runs them on the `ts_simStep` cadence.

A cut is only made for a real lens: if the two crossing points coincide (one
crossing at a polyline node counted twice) the cut is refused, since it would
create a zero-length bridge edge and NaN forces (`test_t4_rejects_degenerate_lens`).

## Scope & honesty about "exact"

A stochastic simulation seeded with MATLAB's `rng('shuffle')`, `voronoin` and
`normrnd` cannot be reproduced bit‑for‑bit in Python. "Exact reproduction" here
means recovering the paper's **quantitative results** — the same curves,
scaling exponents, and transition values (`φ_c≈0.83`, `z_c=4`, `s_c≈3.81`,
caged→diffusive MSD with `α: 0→1`, stretched‑exponential stress relaxation).

The confluent dynamic vertex model (Figs 3, 4a, 2g, 4g) is a faithful port of the
paper's force law and dynamics. The foam limit (Fig 2f) is reproduced with the
standard O'Hern soft‑particle jamming model the paper cites (ref. 36). The full
**non‑confluent structural sweep** (Figs 2a–c) and the **paper's stress tensor**
(Figs 2i, 2j, 4d, 4e) use the curved‑edge foam model with the T2/T3‑reverse
extracellular‑space machinery ported from `ts_t*.m` — subject to the W→0
over‑fragmentation and full‑T4 limitations noted above.

## Movies

| File | What it shows | Script |
|---|---|---|
| `figures/fluid_solid_movie.gif` | **Fig 5j-style**: a fluid MPZ (high activity, thousands of T1s) beside a solid PSM (low activity, ≈0 T1s) — active fluidisation at fixed adhesion, cells coloured by shape factor with tracked trajectories | `scripts/make_fluid_solid_movie.py` |
| `figures/active_foam_full_movie.gif` | the **faithful foam model**: curved cells with extracellular spaces (red) opening under tension fluctuations and rearranging via the full T1–T4 machinery near the structural transition (spaces coarsen at long times — a model limit) | `scripts/make_foam_movie.py` |
| `figures/active_foam_movie.gif` | single active tissue (confluent) with T1 events and tracked cell trails | `scripts/make_movie.py` |

## Running

```bash
pip install numpy scipy matplotlib pillow
python tests/test_model.py            # correctness checks (incl. full T4)
python scripts/run_fig3_msd.py        # (each driver writes data/*.npz)
python scripts/make_comparisons.py    # writes figures/compare_*.png
python scripts/make_fig4h_panel.py    # Fig 4h shape-factor snapshots
python scripts/make_fluid_solid_movie.py   # Fig 5j fluid-vs-solid movie
python scripts/make_foam_movie.py     # curved-cell foam-with-spaces movie
```

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
| **2b/2c** | Volume fraction `φ` and contact number `z` vs `W/T₀`, `ρ` | `scripts/run_fig2b_unified.py` |
| **2f** | Foam‑limit jamming: `z` vs `φ`, `φ_c≈0.83`, `z_c=4` | `scripts/run_fig2f_foam.py` |
| **2g/4g** | Cell shape factor `s̄` vs `W/T₀` (equilibrium & dynamic) | `scripts/run_shapefactor.py` |
| **2i** | Shear‑stress relaxation `σ_xy(t)/σ₀` vs `W/T₀` (paper Eq. 4/5) | `scripts/run_fig2i_stress_relax.py` |
| **2j** | Yield stress `σ_Y` vs `W/T₀` (paper Eq. 4/5) | `scripts/run_fig2j_yieldstress.py` |
| **3a** | MSD(t) vs `t/τ_T` for varying `ΔT/T₀`, and exponent `α(ΔT)` | `scripts/run_fig3_msd.py` |
| **4a** | Shear‑stress relaxation → stretched exponential | `scripts/run_fig4a_stress.py` |
| **4d/4e** | `τ_SR` map and fluid/solid phase diagram over `(W/T₀, ΔT/T₀)` | `scripts/run_fig4de.py` |

Each driver saves `data/*.npz`; `scripts/make_comparisons.py` renders the
side‑by‑side comparison PNGs into `figures/`.

### Stress, yield stress and stress relaxation (Figs 2i, 2j, 4a, 4d, 4e)

These use the paper's **Methods stress tensor** (Eq. 4/5),
`σ_mn = ρ [ −Σ_i Δp_i a_i δ_mn + Σ_ij t_ij ℓ_m ℓ_n / |ℓ| ]`, implemented on the
curved‑edge foam model (`FoamTissue.stress_tensor`), together with the affine
shear‑step‑and‑relax protocol (`apply_affine_shear`). Fig 2i recovers the paper's
family exactly — largest elastic jump at `W=0`, highest residual (yield) stress at
the structural transition `W/T₀≈0.5`, and zero stress at `W/T₀=2`.

For the **τ_SR map (4d)** and **phase diagram (4e)** the paper states that
long‑timescale stress relaxation is driven by *actively induced NE (T1)
transitions*, with the stress‑relaxation time set by the inverse cellular NE
rate. We therefore measure the steady‑state NE (topological‑event) rate `k_NE`
directly from the foam model across `(W/T₀, ΔT/T₀)` and set
`τ_SR = τ_T · (k_NE^max / k_NE)` (stress relaxes within one tension‑persistence
time at the fully‑fluid NE ceiling, diverging as the NE rate → 0). Applying the
paper's fluidity criterion `τ_SR/τ_T = 10²` reproduces the topology of Fig 4e:
a solid band bordering the structural transition at low activity, fluid
everywhere for large enough `ΔT/T₀`. This is a reduced NE‑rate reconstruction,
not a per‑point stretched‑exponential fit of the full stress‑relaxation curve.

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
| **T2** | annihilate a collapsed space back to a triple junction | ✅ `t2_annihilate_spaces` (+ id remap `_remove_ve`) |
| **T3-reverse** | merge two spaces across a space–space edge | ✅ `ts_t3ReverseTransition` port (`t3_reverse`) |
| **T4** | resolve two curved edges that geometrically cross | ⚠️ not ported (needs `ts_edgeCut*` / `ts_t4Transition`) |

**The contact-breaking chain now works.** As a space grows, a cell–cell contact
collapses → **T1** flips it into a space–space edge → **T3-reverse** merges the
two spaces, so the two cells lose their contact. The unified foam model therefore
reproduces **both φ(W/T₀) and z(W/T₀)** on its own — z rises from the foam limit
toward 6 with adhesion, matching the paper for W/T₀ ≳ 0.5.

**Known limitation — W→0 over-fragmentation.** In the pure-foam W→0 limit the
model *over-fragments*: z falls to ≈2.8–3.7 versus the paper's ≈6 at ρ=1 (even
*below* the isostatic value 4). This is the delicate curved-edge force balance of
Supplementary Section 1: a corner where a cell–cell contact (tension 2−W) meets
two free edges (tension 1 each) cannot balance by tension alone at W=0
(2 > 1+1), so finite contacts are held open only by the free edges' Laplace
pressure/curvature — which the refined-but-nearly-straight edges under-capture.
The following were tested and did **not** fix it (confirming it is a
force-calibration issue, not a transition-logic or cascade bug): gentler/zero
annealing (z 3.4→3.7), finer edge refinement (worse), and limiting the T1
break-rate (worse). The contact-breaking chain itself is correct and gives the
right z(W/T₀) trend for W/T₀ ≳ 0.25.

**Known limitation — full T4.** The extreme foam corner can still tangle via
*inter-cell* edge crossing. A lightweight `_cell_self_intersects` guard
(port of `ts_lineCross`) rejects transitions that self-intersect a cell, but the
full lens-resolution (`ts_t4Transition` + `ts_edgeCutPiece`/`SinglePiece`) is not
ported; the sweep flags un-resolvable corners as NaN.

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

## Running

```bash
pip install numpy scipy matplotlib
python tests/test_model.py            # correctness checks
python scripts/run_fig3_msd.py        # (each driver writes data/*.npz)
python scripts/make_comparisons.py    # writes figures/compare_*.png
```

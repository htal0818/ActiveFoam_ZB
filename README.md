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
| **3a** | MSD(t) vs `t/τ_T` for varying `ΔT/T₀`, and exponent `α(ΔT)` | `scripts/run_fig3_msd.py` |
| **2f** | Foam‑limit jamming: `z` vs `φ`, `φ_c≈0.83`, `z_c=4` | `scripts/run_fig2f_foam.py` |
| **4a** | Shear‑stress relaxation → stretched exponential | `scripts/run_fig4a_stress.py` |
| **2g/4g** | Cell shape factor `s̄` vs `W/T₀` (equilibrium & dynamic) | `scripts/run_shapefactor.py` |

Each driver saves `data/*.npz`; `scripts/make_comparisons.py` renders the
side‑by‑side comparison PNGs into `figures/`.

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

**Known limitation:** in the pure-foam W→0 limit the model *over-fragments*
(z falls below the paper's ≈4–6) because small triangular spaces are not fully
stabilised — that is the delicate curved-edge force balance of the paper's
Supplementary Section 1. The **T4** edge-crossing machinery (`ts_edgeCutPiece`,
`ts_t4Transition`) is also still to be ported for geometric robustness under large
deformation. The `crit ≈ 0.05` size guard from `ts_t4AdjacentTransition` is the
analogue we approximate with the snapshot/revert validation.

## Scope & honesty about "exact"

A stochastic simulation seeded with MATLAB's `rng('shuffle')`, `voronoin` and
`normrnd` cannot be reproduced bit‑for‑bit in Python. "Exact reproduction" here
means recovering the paper's **quantitative results** — the same curves,
scaling exponents, and transition values (`φ_c≈0.83`, `z_c=4`, `s_c≈3.81`,
caged→diffusive MSD with `α: 0→1`, stretched‑exponential stress relaxation).

The confluent dynamic vertex model (Figs 3, 4, 2g) is a faithful port of the
paper's force law and dynamics. The foam limit (Fig 2f) is reproduced with the
standard O'Hern soft‑particle jamming model the paper cites (ref. 36). The full
**non‑confluent structural sweep** (Fig 2b–d) additionally requires the
extracellular‑space machinery (T2/T3/T4 transitions in `ts_t*.m`); that part is
described but not ported here.

## Running

```bash
pip install numpy scipy matplotlib
python tests/test_model.py            # correctness checks
python scripts/run_fig3_msd.py        # (each driver writes data/*.npz)
python scripts/make_comparisons.py    # writes figures/compare_*.png
```

# Findings: delayed feedback and synchronous division

**Synchronous division is implemented, and the moderate delayed-feedback extension produces sustained shape oscillations in these pilots.** All cells present at a scheduled division time split in one atomic round; there is no mechanical evolution between individual mother splits. The circular wall remains rigid.

[Watch synchronous division](synchronous_division.gif) · [Watch delayed feedback](delayed_feedback.gif) · [Open the figure gallery](index.html)

## Division and growth are separate controls

All 12 main growth/division control runs completed. At t=20τR, every mother divides in the division groups: N jumps from 16 to 32. Each mother’s K is partitioned between daughters in proportion to their equal areas, conserving total K and avoiding an instantaneous osmotic-pressure jump. Growth independently doubles K over 0–20τR, after which it stops. Thus density changes through growth, not through a hidden resize of the container or a mass jump at cleavage.

| Growth | Synchronous division | Final N | Final preferred-area density | Mean final packing | Spontaneous contact-changing events, seeds 1–3 |
|---|---|---:|---:|---:|---|
| False | False | 16 | 0.600 | 0.5330 | 0, 0, 0 |
| False | True | 32 | 0.600 | 0.5123 | 76, 45, 17 |
| True | False | 16 | 1.200 | 0.9936 | 1, 0, 3 |
| True | True | 32 | 1.200 | 0.9908 | 26, 0, 29 |

Direct topology changes made by cleavage are excluded from the spontaneous-event count. The new daughter contacts nevertheless trigger subsequent mechanical relaxation and rearrangements. Division-only cells have more total cortex per conserved K, so measured packing can fall slightly even though preferred-area density remains constant.

## Delayed feedback produces a qualitatively different response

These runs use g=0.5, rho=1.2, W/T0=1 and tauT=10τR. After the same initial deterministic tension perturbation, mu=0: there is no ongoing stochastic tension drive. The original seeded random ordering of topology operations is retained.

| Delay / τR | Mean late local shape RMS ± seed SD | Spontaneous events, seeds 1–3 | Completed |
|---:|---:|---|---:|
| 0 | 0.00010 ± 0.00002 | 0, 0, 0 | 3/3 |
| 2 | 0.01679 ± 0.00491 | 0, 4, 0 | 3/3 |
| 10 | 0.10559 ± 0.00876 | 114, 152, 126 | 3/3 |

With zero delay, the response mostly decays toward a small numerical fluctuation floor. Delays of 2 and 10τR maintain much larger local shape fluctuations across all three seeds. The short-delay seed-1 case oscillates without exchanging neighbors, separating an oscillatory deformation regime from fluidization. At delay 10, recurrent contact changes coexist with substantial retention of the initial neighbor graph: final overlaps in the three 150τR pilots are 0.971, 0.941 and 0.971. Many events are therefore reversible.

## Numerical sensitivity: the qualitative oscillation survives, but rates are unresolved

The selected seed-1 cases were extended to 300τR. Baseline dt/film-spacing/history-spacing = 0.005/0.1/0.05; refined = 0.0025/0.05/0.025. Preparation is repeated at each resolution.

| Gain | Delay | dt | Late local shape RMS | Dominant spectral-bin period / τR | Spontaneous events |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.005 | 0.00010 | 37.6 | 0 |
| 0 | 0 | 0.0025 | 0.00007 | 25.0 | 0 |
| 0.5 | 2 | 0.005 | 0.01134 | 15.0 | 0 |
| 0.5 | 2 | 0.0025 | 0.00909 | 15.0 | 0 |
| 0.5 | 10 | 0.005 | 0.07566 | 37.6 | 190 |
| 0.5 | 10 | 0.0025 | 0.07927 | 37.6 | 324 |

The delayed cases retain oscillations at both resolutions, with dominant local-shape periods near 15τR and 38τR. These are finite-window spectral-bin estimates, not high-precision period measurements. The no-feedback control also has a largest Fourier bin; its listed period does not indicate a physical oscillation.

**Amplitude and event rates are not quantitatively converged.** The short-delay amplitude changes by about 20% under joint refinement. At delay 10, the late RMS changes by about 5%, while the event count changes from 190 to 324. A larger ensemble and more refinement are needed before quoting a physical exchange rate or locating a dynamical phase boundary.

The selected delay=2, g=0.5 pilot also passes geometry checks at every timestep after preparation, with exactly the same stored trajectory and events as the ordinary run. That audit does not certify every other parameter point or the unmonitored preparation intervals.

## Combined feedback plus synchronous division

All three combined pilots completed. They match the growth-plus-division controls, including mu=0.05, while adding g=0.5 and delay=2τR. This combined case has not received its own numerical refinement study.

| Seed | Control events | Combined events | Control events after division | Combined events after division | Control final shape index | Combined final shape index |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26 | 149 | 26 | 149 | 3.8690 | 3.9734 |
| 2 | 0 | 94 | 0 | 94 | 3.8847 | 3.9672 |
| 3 | 29 | 222 | 28 | 221 | 3.8613 | 3.9965 |

The combined model has visibly greater deformation and more repeated contact-changing operations in this pilot. Those counts should not be equated with lasting mixing or a fluid phase.

## Validation and scope

- 61 experimental runs: 36/51 main runs completed, 7/7 refinement/audit runs completed, and 3/3 combined runs completed. The 15 failures were stronger-feedback geometries and remain explicitly recorded.
- 170 regression tests pass. They cover original-model parity with extensions disabled, delay causality, contact-memory resetting, daughter pressure continuity, K/area conservation, trivalence, persistent lineage IDs, synchronous birth times and full-round rollback.
- Both animations are replays verified to reproduce the stored samples, events, division logs, snapshots and summary exactly. Cell labels are persistent IDs; colors show shape index.
- Feedback is an explicit phenomenological addition to the original tension equation. Division is instantaneous geometric cleavage with no resolved mitotic rounding or contractile ring. Three-seed pilots and one-seed refinement checks are not a complete biological validation.

Methods: [REPORT.md](REPORT.md) and `circle/BIOLOGY.md`. Numerical summaries: [summary.csv](summary.csv), [refinement.csv](refinement.csv), and [combined_summary.csv](combined_summary.csv).

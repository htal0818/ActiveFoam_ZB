# Circular-wall parameter sweeps

208 of 234 attempted runs passed their domain-specific geometry checks.

The tissue surface is flat. κ=1/R is wall curvature. R=√(N/π), so nominal density stays fixed across sizes; curvature and finite size cannot be separated by this disk sweep alone.

3 independent seeds are attempted at each point. Error bars are seed SEM, not cell-wise uncertainty. Means exclude failed runs and are conditional on acceptance. Local microfolds remain tolerated and counted. Periodic runs retain their original per-cell geometry checks.

Structure: W={0.5,1,1.5}, ρ={1,1.2,1.4}, μ=0 after standard preparation.
Dynamics: W=1, ρ=1.3, μ={0,0.5,1,1.5}; 10 τR warmup and 50.0 τR measured (τT=10). Sampling every 0.5 τR; MSD uses lags ≤ one quarter of the measurement window. The one-τT warmup does not establish stationarity; shape changes can include residual relaxation. These short runs do not establish diffusion exponents, rigidity thresholds or a phase diagram.

T1 rate counts accepted T1 operations/N/time, including boundary operations and repeated reversals; it is not identical to the cell-neighbor exchange rate. Contact-change rate counts sampled contact gains and losses per cell/time; transient changes between samples can be missed. Shape index is P/√A. Wall/interior zones for dynamics are fixed after warmup. Cell-level scatter points are correlated and do not supply independent error bars.

| Domain | Mode | N | Accepted / attempted |
|---|---|---|---|
| disc | dynamics | 36 | 9 / 12 |
| disc | dynamics | 100 | 9 / 12 |
| disc | dynamics | 196 | 7 / 12 |
| disc | structure | 36 | 27 / 27 |
| disc | structure | 100 | 27 / 27 |
| disc | structure | 196 | 26 / 27 |
| periodic | dynamics | 36 | 8 / 12 |
| periodic | dynamics | 100 | 7 / 12 |
| periodic | dynamics | 196 | 7 / 12 |
| periodic | structure | 36 | 27 / 27 |
| periodic | structure | 100 | 27 / 27 |
| periodic | structure | 196 | 27 / 27 |

## Disk activity response

Means ± independent-seed SEM where at least two runs pass. Missing points are failures, not zero rates. T1 operations and sampled neighbor changes are distinct quantities.

| N | μ | Accepted | T1 / cell / τT | Neighbor changes / cell / τT | Mean P/√A | Final − initial P/√A |
|---|---|---|---|---|---|---|
| 36 | 0 | 3 / 3 | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 3.8471 ± 0.0034 | -0.0001 ± 0.0001 |
| 36 | 0.5 | 3 / 3 | 0.0111 ± 0.0064 | 0.0000 ± 0.0000 | 3.8619 ± 0.0037 | 0.0034 ± 0.0009 |
| 36 | 1 | 3 / 3 | 0.4074 ± 0.0744 | 0.0370 ± 0.0196 | 3.9113 ± 0.0058 | 0.0139 ± 0.0050 |
| 36 | 1.5 | 0 / 3 | — | — | — | — |
| 100 | 0 | 3 / 3 | 0.0053 ± 0.0053 | 0.0027 ± 0.0027 | 3.8593 ± 0.0041 | -0.0023 ± 0.0014 |
| 100 | 0.5 | 3 / 3 | 0.0600 ± 0.0335 | 0.0133 ± 0.0071 | 3.8723 ± 0.0024 | -0.0007 ± 0.0042 |
| 100 | 1 | 3 / 3 | 0.6287 ± 0.0948 | 0.0600 ± 0.0106 | 3.9234 ± 0.0016 | 0.0064 ± 0.0104 |
| 100 | 1.5 | 0 / 3 | — | — | — | — |
| 196 | 0 | 3 / 3 | 0.0044 ± 0.0039 | 0.0034 ± 0.0025 | 3.8614 ± 0.0112 | -0.0039 ± 0.0023 |
| 196 | 0.5 | 3 / 3 | 0.1405 ± 0.0998 | 0.0184 ± 0.0105 | 3.8740 ± 0.0104 | -0.0063 ± 0.0053 |
| 196 | 1 | 1 / 3 | 0.7929 | 0.0918 | 3.9380 | -0.0116 |
| 196 | 1.5 | 0 / 3 | — | — | — | — |

## Failures

Failures are retained in per-run JSON and outcomes.csv; no interpolation fills missing points. summary.csv holds separate-domain means; paired_differences.csv uses only seeds accepted in both domains for disk-minus-periodic contrasts.

- disc dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.046495
- disc dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0885491
- disc dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: 1 resolved film crossing(s); films 198, 199 at (15.0771, 12.3968); local-junction distance 0.0512653
- disc dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.221593
- disc dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.0637021
- disc dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.251507
- disc dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.0439866
- disc dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=1: warmup: 1 resolved film crossing(s); films 1185, 1186 at (29.0389, 38.2942); local-junction distance 0.0720261
- disc dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0354542
- disc structure, N=196, W=1.5, ρ=1.4, μ=0.0, seed=3: preparation: 3 resolved film crossing(s); films 473, 1611 at (37.1527, 28.9141); local-junction distance inf at elapsed time 7.5 tau_R
- disc dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.0461247
- disc dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.0587572
- periodic dynamics, N=36, W=1.0, ρ=1.3, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.0362829
- periodic dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.054841
- periodic dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0467344
- periodic dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.101073
- periodic dynamics, N=100, W=1.0, ρ=1.3, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.281394
- periodic dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.293285
- periodic dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.173835
- periodic dynamics, N=100, W=1.0, ρ=1.3, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.214465
- periodic dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.201646
- periodic dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.052371
- periodic dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=2: measurement: resolved boundary crossing: junction distance 0.0449
- periodic dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.039085
- periodic dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.248227
- periodic dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.0768919

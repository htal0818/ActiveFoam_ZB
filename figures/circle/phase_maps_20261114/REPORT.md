# Curvature, adhesion, activity and density maps

These are maps of continuous observables on a sampled grid, not inferred fluid/solid phase boundaries. No interpolation or smoothing fills failed points. Cell contacts means distinct neighbors per cell, not the rate of neighbor exchange. A confluent trivalent interior can remain near six neighbors while cells rearrange; these maps do not impose a universal shape-index threshold for fluidity.

Passive source: `/workspace/ActiveFoam_ZB/circle/data/curvature_sweep_rerun_20261114`. Active source: `/workspace/ActiveFoam_ZB/circle/data/phase_sweep_20261114`. Both use the same force law and geometry acceptance rules; the active measurement was extended to record contact number and packing during dynamics.

Wall curvature is κ=1/R. Axes use κℓ with ℓ=√(A_T/N), giving κℓ=√(π/N). The CSV also reports κL0=√(πρ/N): at fixed N, varying density changes curvature measured in preferred-cell units. Curvature and finite size remain coupled; the tissue surface is flat.

Adhesion maps: μ=0, W/T0={0.5,1,1.5}, ρ={1,1.2,1.4}. Activity maps: W/T0=1, ΔT/T0 (code μ)={0,0.5,1,1.5}, ρ={1,1.2,1.3,1.4}. Sizes N={36,100,196}, three seeds per point. The repeated ρ=1.3 trajectories are reproducibility checks and are not pooled as additional independent seeds. The activity label assumes unit white-noise covariance in code time units.

Passive measurements are preparation endpoints. Active quantities average 50 τR after 10 τR warmup, sampling every 0.5 τR, with τT=10 τR. These are short finite-time averages; stationarity, mesh convergence and long-time phases are not established. Interior means centroid farther than one simulation length unit from the container wall. Dynamic zones are fixed after warmup; passive zones use endpoint geometry. When the tissue detaches, this wall-distance definition can include cells on the tissue’s free surface and is not a pure tissue-bulk classification.

ρ=N A0/A_T is the control parameter. Area fraction φ=ΣA_i/A_T is measured; it is not independently prescribed. Curves against φ are conditional observations, not sweeps at controlled φ.

Means exclude failed runs and are conditional on acceptance. Labels show accepted/attempted seeds; SEM is over seed means and is unavailable for one surviving seed. Local junction microfolds remain tolerated and reported in raw data. Sampling geometry every 100 steps does not certify all intermediate states.

| Mode | N | Accepted / attempted |
|---|---|---|
| dynamics | 36 | 32 / 48 |
| dynamics | 100 | 31 / 48 |
| dynamics | 196 | 28 / 48 |
| structure | 36 | 27 / 27 |
| structure | 100 | 27 / 27 |
| structure | 196 | 26 / 27 |

## Example: active response at N=100, rho=1.3, W/T0=1

| Activity input | Accepted | Contacts z | Shape P/√A | Area fraction | Neighbor changes / cell / tauT |
|---|---|---|---|---|---|
| 0 | 3/3 | 5.3000 | 3.8593 | 0.9976 | 0.0027 |
| 0.5 | 3/3 | 5.3000 | 3.8723 | 0.9974 | 0.0133 |
| 1 | 3/3 | 5.3056 | 3.9234 | 0.9973 | 0.0600 |
| 1.5 | 0/3 | — | — | — | — |

These are conditional seed means. The CSV gives SEM and counts. Nearly constant mean contact number does not imply absent neighbor exchange.


## Figures

- [phase_diagrams_overview](phase_diagrams_overview.png) ([PDF](phase_diagrams_overview.pdf))
- [contacts_curvature_adhesion](contacts_curvature_adhesion.png) ([PDF](contacts_curvature_adhesion.pdf))
- [contacts_curvature_tension_fluctuations](contacts_curvature_tension_fluctuations.png) ([PDF](contacts_curvature_tension_fluctuations.pdf))
- [shape_index_curvature_adhesion](shape_index_curvature_adhesion.png) ([PDF](shape_index_curvature_adhesion.pdf))
- [shape_index_curvature_tension_fluctuations](shape_index_curvature_tension_fluctuations.png) ([PDF](shape_index_curvature_tension_fluctuations.pdf))
- [packing_fraction_curvature_adhesion](packing_fraction_curvature_adhesion.png) ([PDF](packing_fraction_curvature_adhesion.pdf))
- [packing_fraction_curvature_tension_fluctuations](packing_fraction_curvature_tension_fluctuations.png) ([PDF](packing_fraction_curvature_tension_fluctuations.pdf))
- [neighbor_exchange_curvature_tension_fluctuations](neighbor_exchange_curvature_tension_fluctuations.png) ([PDF](neighbor_exchange_curvature_tension_fluctuations.pdf))
- [density_maps_passive](density_maps_passive.png) ([PDF](density_maps_passive.pdf))
- [density_maps_active](density_maps_active.png) ([PDF](density_maps_active.pdf))
- [contacts_shape_vs_volume_fraction](contacts_shape_vs_volume_fraction.png) ([PDF](contacts_shape_vs_volume_fraction.pdf))

## Failures

- structure, N=196, W=1.5, ρ=1.4, μ=0.0, seed=3: preparation: 3 resolved film crossing(s); films 473, 1611 at (37.1527, 28.9141); local-junction distance inf at elapsed time 7.5 tau_R
- dynamics, N=36, W=1.0, ρ=1.0, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.243511
- dynamics, N=36, W=1.0, ρ=1.0, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.038827
- dynamics, N=36, W=1.0, ρ=1.0, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0406312
- dynamics, N=36, W=1.0, ρ=1.0, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.221681
- dynamics, N=36, W=1.0, ρ=1.2, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.254374
- dynamics, N=36, W=1.0, ρ=1.2, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.0396197
- dynamics, N=36, W=1.0, ρ=1.2, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.225881
- dynamics, N=36, W=1.0, ρ=1.2, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.309845
- dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.046495
- dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0885491
- dynamics, N=36, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: 1 resolved film crossing(s); films 198, 199 at (15.0771, 12.3968); local-junction distance 0.0512653
- dynamics, N=36, W=1.0, ρ=1.4, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.185699
- dynamics, N=36, W=1.0, ρ=1.4, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.0828754
- dynamics, N=36, W=1.0, ρ=1.4, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.0923832
- dynamics, N=36, W=1.0, ρ=1.4, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.058841
- dynamics, N=36, W=1.0, ρ=1.4, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.383036
- dynamics, N=100, W=1.0, ρ=1.0, μ=1.5, seed=1: measurement: 1 resolved film crossing(s); films 280, 281 at (23.6709, 19.222); local-junction distance 0.063587
- dynamics, N=100, W=1.0, ρ=1.0, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.292373
- dynamics, N=100, W=1.0, ρ=1.0, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.037947
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.199637
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.0, seed=2: measurement: resolved boundary crossing: junction distance 0.16045
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.191246
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.0511708
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.177021
- dynamics, N=100, W=1.0, ρ=1.2, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.0460146
- dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=1: measurement: resolved boundary crossing: junction distance 0.221593
- dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.0637021
- dynamics, N=100, W=1.0, ρ=1.3, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.251507
- dynamics, N=100, W=1.0, ρ=1.4, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.225264
- dynamics, N=100, W=1.0, ρ=1.4, μ=1.0, seed=2: measurement: resolved boundary crossing: junction distance 0.0441041
- dynamics, N=100, W=1.0, ρ=1.4, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.349409
- dynamics, N=100, W=1.0, ρ=1.4, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.190859
- dynamics, N=100, W=1.0, ρ=1.4, μ=1.5, seed=3: warmup: 1 resolved film crossing(s); films 52, 53 at (25.5635, 25.8303); local-junction distance 0.0356853
- dynamics, N=196, W=1.0, ρ=1.0, μ=1.0, seed=2: measurement: resolved boundary crossing: junction distance 0.0497092
- dynamics, N=196, W=1.0, ρ=1.0, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.154879
- dynamics, N=196, W=1.0, ρ=1.0, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0673679
- dynamics, N=196, W=1.0, ρ=1.0, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.041007
- dynamics, N=196, W=1.0, ρ=1.2, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.0371607
- dynamics, N=196, W=1.0, ρ=1.2, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.0451378
- dynamics, N=196, W=1.0, ρ=1.2, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.153042
- dynamics, N=196, W=1.0, ρ=1.2, μ=1.5, seed=2: warmup: resolved boundary crossing: junction distance 0.0373946
- dynamics, N=196, W=1.0, ρ=1.2, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.0443479
- dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.0439866
- dynamics, N=196, W=1.0, ρ=1.3, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.0461247
- dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=1: warmup: 1 resolved film crossing(s); films 1185, 1186 at (29.0389, 38.2942); local-junction distance 0.0720261
- dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0354542
- dynamics, N=196, W=1.0, ρ=1.3, μ=1.5, seed=3: measurement: resolved boundary crossing: junction distance 0.0587572
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.0, seed=1: measurement: resolved boundary crossing: junction distance 0.064472
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.0, seed=2: measurement: resolved boundary crossing: junction distance 0.183543
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.0, seed=3: measurement: resolved boundary crossing: junction distance 0.19999
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.5, seed=1: warmup: resolved boundary crossing: junction distance 0.244173
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.5, seed=2: measurement: resolved boundary crossing: junction distance 0.0456288
- dynamics, N=196, W=1.0, ρ=1.4, μ=1.5, seed=3: warmup: resolved boundary crossing: junction distance 0.215847

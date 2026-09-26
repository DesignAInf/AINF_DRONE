# Frozen V5 Results

Protocol: 20 held-out paired seeds, 6 conditions, 3 tasks, 360 missions. Seed offset: 1000.

## Primary comparison

| Task | Constitutive success | Diagnostic-only success | Paired final-gap difference | Paired architecture-score difference |
|---|---:|---:|---:|---:|
| Construct | 95% | 0% | -26.57 +/- 0.86 | +0.748 +/- 0.031 |
| Stabilize | 95% | 0% | -8.23 +/- 0.60 | +0.495 +/- 0.029 |
| Transform | 100% | 0% | 0.00 +/- 0.00 | +0.335 +/- 0.010 |

The zero transformation gap difference is predicted: both routings can remain valid blankets. Transformation is detected by the changed physical role assignment, higher boundary precision, and a paired viability advantage of `+0.090 +/- 0.005`.

## Interpretation

The primary result is not merely higher role-classification accuracy. The constitutive condition changes the causal matrix approximately three times in construction, once after a stabilization breach, and once in transformation. The diagnostic-only condition changes it zero times. The task-adjusted indirect association from condition to viability through architecture score is 0.508, with a paired-seed bootstrap 95% interval of [0.313, 0.760].

The result is strong inside the simulator but bounded. Random design performs well on shielding because repeated attempts can eventually close sparse leaks; it fails transformation because only a specific precision-sensitive swap is useful. This makes the transformation task the clearest test of joint pragmatic and structural policy selection.

Machine-readable results are in `results/mission_level.csv`, `results/summary.csv`, `results/paired_effects.csv`, and `results/mediation.json`.

## Collective benchmark

Protocol: 20 held-out paired seeds, 7 conditions, 3 tasks, 420 missions. Seed offset: 2000.

| Task | Collective constitutive | Individual constitutive | Communication only | Oracle |
|---|---:|---:|---:|---:|
| Assemble | 80% | 25% | 0% | 100% |
| Repair | 90% | 15% | 0% | 100% |
| Reconfigure | 100% | 40% | 0% | 100% |

Against equal-bandwidth communication, paired group-viability advantages are `+0.114 +/- 0.018`, `+0.052 +/- 0.004`, and `+0.111 +/- 0.004`. Against individual constitutive agents, paired architecture-score advantages are `+0.373 +/- 0.071`, `+0.414 +/- 0.056`, and `+0.224 +/- 0.043`.

Random bilateral action succeeds in 60% of assembly and 75% of repair missions because a small shielding space can be searched by chance. It succeeds in 0% of targeted reconfiguration missions. This limits the claim correctly: EFE is not necessary for every causal change, but it supplies selectivity when multiple structurally valid transformations are available.

Machine-readable collective results are in `collective_results/mission_level.csv`, `collective_results/summary.csv`, `collective_results/paired_effects.csv`, and `collective_results/run_metadata.json`.

# raw arep randomized generalization report

## What is measured
- clean success rate: mean success rate from demo_clean results.
- randomized success rate: success rate from demo_randomized results.
- abs drop: clean minus randomized. Larger means worse robustness.
- retention ratio: randomized divided by clean. Higher means better generalization.

## Summary table

| task | model | clean mean | randomized mean | abs drop | retention | clean runs | randomized runs |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| click_bell | DP | 0.620 | 0.000 | 0.620 | 0.0% | 1 | 1 |
| click_bell | raw_arep | 0.680 | 0.030 | 0.650 | 4.4% | 3 | 1 |
| dump_bin_bigbin | DP | 0.450 | 0.000 | 0.450 | 0.0% | 1 | 1 |
| dump_bin_bigbin | raw_arep | 0.580 | 0.000 | 0.580 | 0.0% | 3 | 1 |
| lift_pot | DP | 0.830 | - | - | - | 3 | - |
| place_fan | raw_arep | 0.064 | - | - | - | 10 | - |
| stack_bowls_three | raw_arep | 0.347 | - | - | - | 3 | - |
| stack_bowls_two | raw_arep | 0.585 | - | - | - | 6 | - |
| turn_switch | raw_arep | 0.363 | - | - | - | 6 | - |

## Readout
- raw arep average retention on the available randomized tasks: 2.2%.
- raw arep average absolute drop on the available randomized tasks: 0.615.
- Evidence coverage is narrow: only tasks with both clean and demo_randomized results are included.
- A zero or near-zero retention ratio means the model did not preserve useful visual robustness after domain randomization.
- randomized result sources in this report: manual_terminal_verified, result_file.
- manual_terminal_verified means the terminal confirmed the final success rate, but the evaluation script did not persist a standard _result.txt file.

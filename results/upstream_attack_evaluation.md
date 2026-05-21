# Upstream Attack Evaluation

This report is generated from the cleaned `results/upstream_attack_results.csv`.
The CSV was reduced from 126 appended result rows to 51 newest rows, keeping the last row for each `dataset/model/attack/epsilon` configuration.

| Dataset | Test samples |
|---|---:|
| Banknote | 275 |
| Diabetes | 154 |
| Wilt | 968 |

Metrics:

- `clean_acc`: model accuracy before attack.
- `adv_acc`: model accuracy on adversarial samples.
- `acc_drop`: `clean_acc - adv_acc`; higher means stronger attack.
- `ASR`: attack success rate over originally correct predictions.
- `l0_mean`, `l2_mean`, and `linf_mean`: average perturbation size.

## Best Attack Per Dataset And Model

| Dataset | Model | Best attack | Clean acc | Adv acc | Acc drop | ASR | L0 mean | L2 mean | Linf mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Banknote | lin | hsj | 0.956 | 0.040 | 0.916 | 1.000 | 4.000 | 0.169 | 0.121 |
| Banknote | nn | bim eps=0.20 | 1.000 | 0.000 | 1.000 | 1.000 | 3.989 | 0.386 | 0.200 |
| Banknote | svm | hsj | 1.000 | 0.000 | 1.000 | 1.000 | 4.000 | 0.142 | 0.102 |
| Banknote | xgb | hsj | 0.996 | 0.004 | 0.993 | 1.000 | 4.000 | 0.166 | 0.156 |
| Diabetes | lin | zoo | 0.734 | 0.045 | 0.688 | 0.938 | 3.838 | 0.238 | 0.136 |
| Diabetes | nn | bim eps=0.20 | 0.760 | 0.273 | 0.487 | 0.957 | 6.364 | 0.481 | 0.200 |
| Diabetes | svm | zoo | 0.734 | 0.175 | 0.558 | 0.761 | 3.429 | 0.195 | 0.117 |
| Diabetes | xgb | zoo | 0.740 | 0.123 | 0.617 | 0.833 | 2.377 | 0.192 | 0.130 |
| Wilt | lin | hsj | 0.943 | 0.943 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| Wilt | nn | bim eps=0.20 | 0.987 | 0.017 | 0.970 | 0.997 | 4.959 | 0.221 | 0.106 |
| Wilt | svm | zoo | 0.981 | 0.031 | 0.950 | 0.968 | 4.656 | 0.054 | 0.036 |
| Wilt | xgb | hsj | 0.983 | 0.017 | 0.967 | 1.000 | 4.951 | 0.018 | 0.017 |

## Banknote

| Model | Attack | Clean -> Adv acc | Acc drop | ASR | L2 mean | Linf mean |
|---|---|---:|---:|---:|---:|---:|
| lin | hsj | 0.956 -> 0.040 | 0.916 | 1.000 | 0.169 | 0.121 |
| lin | lpf | 0.956 -> 1.000 | -0.044 | 0.000 | 0.290 | 0.217 |
| lin | zoo | 0.956 -> 0.105 | 0.851 | 0.890 | 0.167 | 0.100 |
| nn | bim eps=0.00 | 1.000 -> 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | bim eps=0.10 | 1.000 -> 0.113 | 0.887 | 0.887 | 0.198 | 0.100 |
| nn | bim eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 | 0.386 | 0.200 |
| nn | fgm eps=0.00 | 1.000 -> 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | fgm eps=0.10 | 1.000 -> 0.204 | 0.796 | 0.796 | 0.198 | 0.100 |
| nn | fgm eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 | 0.387 | 0.200 |
| nn | pgd eps=0.00 | 1.000 -> 1.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | pgd eps=0.10 | 1.000 -> 0.113 | 0.887 | 0.887 | 0.198 | 0.100 |
| nn | pgd eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 | 0.387 | 0.200 |
| svm | hsj | 1.000 -> 0.000 | 1.000 | 1.000 | 0.142 | 0.102 |
| svm | lpf | 1.000 -> 1.000 | 0.000 | 0.000 | 0.335 | 0.257 |
| svm | zoo | 1.000 -> 0.138 | 0.862 | 0.862 | 0.142 | 0.088 |
| xgb | hsj | 0.996 -> 0.004 | 0.993 | 1.000 | 0.166 | 0.156 |
| xgb | zoo | 0.996 -> 0.160 | 0.836 | 0.839 | 0.169 | 0.147 |

Best attacks by accuracy drop: lin: hsj; nn: bim eps=0.20; svm: hsj; xgb: hsj.

## Diabetes

| Model | Attack | Clean -> Adv acc | Acc drop | ASR | L2 mean | Linf mean |
|---|---|---:|---:|---:|---:|---:|
| lin | hsj | 0.734 -> 0.266 | 0.468 | 1.000 | 0.217 | 0.169 |
| lin | lpf | 0.734 -> 1.000 | -0.266 | 0.000 | 0.375 | 0.280 |
| lin | zoo | 0.734 -> 0.045 | 0.688 | 0.938 | 0.238 | 0.136 |
| nn | bim eps=0.00 | 0.734 -> 0.734 | 0.000 | 0.000 | 0.001 | 0.001 |
| nn | bim eps=0.10 | 0.714 -> 0.487 | 0.227 | 0.700 | 0.241 | 0.100 |
| nn | bim eps=0.20 | 0.760 -> 0.273 | 0.487 | 0.957 | 0.481 | 0.200 |
| nn | fgm eps=0.00 | 0.740 -> 0.740 | 0.000 | 0.000 | 0.001 | 0.001 |
| nn | fgm eps=0.10 | 0.714 -> 0.578 | 0.136 | 0.573 | 0.270 | 0.100 |
| nn | fgm eps=0.20 | 0.740 -> 0.305 | 0.435 | 0.930 | 0.531 | 0.200 |
| nn | pgd eps=0.00 | 0.714 -> 0.714 | 0.000 | 0.000 | 0.001 | 0.001 |
| nn | pgd eps=0.10 | 0.747 -> 0.461 | 0.286 | 0.713 | 0.253 | 0.100 |
| nn | pgd eps=0.20 | 0.727 -> 0.305 | 0.422 | 0.955 | 0.471 | 0.200 |
| svm | hsj | 0.734 -> 0.266 | 0.468 | 1.000 | 0.192 | 0.144 |
| svm | lpf | 0.734 -> 1.000 | -0.266 | 0.000 | 0.420 | 0.292 |
| svm | zoo | 0.734 -> 0.175 | 0.558 | 0.761 | 0.195 | 0.117 |
| xgb | hsj | 0.740 -> 0.260 | 0.481 | 1.000 | 0.143 | 0.129 |
| xgb | zoo | 0.740 -> 0.123 | 0.617 | 0.833 | 0.192 | 0.130 |

Best attacks by accuracy drop: lin: zoo; nn: bim eps=0.20; svm: zoo; xgb: zoo.

## Wilt

| Model | Attack | Clean -> Adv acc | Acc drop | ASR | L2 mean | Linf mean |
|---|---|---:|---:|---:|---:|---:|
| lin | hsj | 0.943 -> 0.943 | 0.000 | 0.000 | 0.000 | 0.000 |
| lin | lpf | 0.943 -> 0.943 | 0.000 | 0.000 | 0.065 | 0.047 |
| lin | zoo | 0.943 -> 0.943 | 0.000 | 0.000 | 0.530 | 0.395 |
| nn | bim eps=0.00 | 0.986 -> 0.986 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | bim eps=0.10 | 0.983 -> 0.030 | 0.954 | 0.986 | 0.207 | 0.099 |
| nn | bim eps=0.20 | 0.987 -> 0.017 | 0.970 | 0.997 | 0.221 | 0.106 |
| nn | fgm eps=0.00 | 0.987 -> 0.987 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | fgm eps=0.10 | 0.983 -> 0.030 | 0.954 | 0.986 | 0.209 | 0.100 |
| nn | fgm eps=0.20 | 0.988 -> 0.026 | 0.962 | 0.986 | 0.391 | 0.199 |
| nn | pgd eps=0.00 | 0.987 -> 0.987 | 0.000 | 0.000 | 0.000 | 0.000 |
| nn | pgd eps=0.10 | 0.987 -> 0.165 | 0.821 | 0.843 | 0.154 | 0.093 |
| nn | pgd eps=0.20 | 0.987 -> 0.290 | 0.696 | 0.719 | 0.354 | 0.189 |
| svm | hsj | 0.981 -> 0.207 | 0.775 | 0.807 | 0.011 | 0.009 |
| svm | lpf | 0.981 -> 1.000 | -0.019 | 0.000 | 0.242 | 0.197 |
| svm | zoo | 0.981 -> 0.031 | 0.950 | 0.968 | 0.054 | 0.036 |
| xgb | hsj | 0.983 -> 0.017 | 0.967 | 1.000 | 0.018 | 0.017 |
| xgb | zoo | 0.983 -> 0.074 | 0.909 | 0.924 | 0.052 | 0.038 |

Best attacks by accuracy drop: lin: hsj; nn: bim eps=0.20; svm: zoo; xgb: hsj.

## Cross-Model Attack Pattern

| Attack family | Strongest observed behavior |
|---|---|
| bim | Largest drop on Banknote nn (bim eps=0.20, acc drop 1.000, ASR 1.000). |
| fgm | Largest drop on Banknote nn (fgm eps=0.20, acc drop 1.000, ASR 1.000). |
| hsj | Largest drop on Banknote svm (hsj, acc drop 1.000, ASR 1.000). |
| lpf | Largest drop on Banknote svm (lpf, acc drop 0.000, ASR 0.000). |
| pgd | Largest drop on Banknote nn (pgd eps=0.20, acc drop 1.000, ASR 1.000). |
| zoo | Largest drop on Wilt svm (zoo, acc drop 0.950, ASR 0.968). |

## Saved Upstream Models

| Model | Saved file |
|---|---|
| Linear | `models/<dataset>/upstream_lin.pkl` |
| SVM | `models/<dataset>/upstream_svm.pkl` |
| XGB-style GradientBoosting | `models/<dataset>/upstream_xgb.pkl` |
| Neural network | `models/<dataset>/upstream_nn.pt` |

## Caveats

- ZOO used `max_iter=20` for tractable full-sample computation.
- `xgb` follows the upstream naming but is implemented as `GradientBoostingClassifier`, not the external XGBoost package.
- LowProFool cannot run on the tree/gradient-boosting wrapper because ART requires loss gradients for that attack.
- Neural-network epsilon `0.00` rows are sanity checks and are excluded from the best-attack summary.

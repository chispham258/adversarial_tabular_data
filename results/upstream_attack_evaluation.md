# Upstream Attack Evaluation

This report evaluates the upstream-style attacks implemented in `src/upstream_attacks.py` using the full test split for each dataset:

| Dataset | Test samples |
|---|---:|
| Wilt | 968 |
| Banknote | 275 |
| Diabetes | 154 |

Metrics:

- `clean_acc`: model accuracy before attack.
- `adv_acc`: model accuracy on adversarial samples.
- `acc_drop`: `clean_acc - adv_acc`; higher means stronger attack.
- `ASR`: attack success rate over originally correct predictions.
- Perturbation columns (`l0_mean`, `l2_mean`, `linf_mean`) are averaged over generated adversarial samples.

The source CSV is append-only, so this report uses the latest full-sample rows from `results/upstream_attack_results.csv` and excludes earlier 25-sample exploratory runs.

## Best Attack Per Dataset And Model

| Dataset | Model | Best attack | Clean acc | Adv acc | Acc drop | ASR | L0 mean | L2 mean | Linf mean |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Banknote | lin | hsj | 0.956 | 0.040 | 0.916 | 1.000 | 4.000 | 0.169 | 0.121 |
| Banknote | nn | fgm eps=0.20 | 1.000 | 0.000 | 1.000 | 1.000 | 4.000 | 0.387 | 0.200 |
| Banknote | svm | hsj | 1.000 | 0.000 | 1.000 | 1.000 | 4.000 | 0.142 | 0.102 |
| Banknote | xgb | hsj | 0.996 | 0.004 | 0.993 | 1.000 | 3.996 | 0.164 | 0.155 |
| Diabetes | lin | zoo | 0.734 | 0.013 | 0.721 | 0.982 | 3.981 | 0.241 | 0.137 |
| Diabetes | nn | bim eps=0.20 | 0.760 | 0.273 | 0.487 | 0.957 | 6.364 | 0.481 | 0.200 |
| Diabetes | svm | zoo | 0.734 | 0.162 | 0.571 | 0.779 | 3.481 | 0.198 | 0.115 |
| Diabetes | xgb | zoo | 0.740 | 0.104 | 0.636 | 0.860 | 2.305 | 0.185 | 0.131 |
| Wilt | lin | zoo / hsj / lpf | 0.943 | 0.943 | 0.000 | 0.000 | 4.756 | 0.530 | 0.395 |
| Wilt | nn | bim eps=0.20 | 0.987 | 0.017 | 0.970 | 0.997 | 4.959 | 0.221 | 0.106 |
| Wilt | svm | zoo | 0.981 | 0.031 | 0.950 | 0.968 | 4.656 | 0.054 | 0.036 |
| Wilt | xgb | hsj | 0.983 | 0.020 | 0.964 | 0.997 | 4.933 | 0.017 | 0.016 |

## Banknote

Banknote is the easiest dataset to break. Most strong attacks reduce accuracy close to zero.

| Model | Attack | Clean -> Adv acc | Acc drop | ASR |
|---|---|---:|---:|---:|
| lin | hsj | 0.956 -> 0.040 | 0.916 | 1.000 |
| lin | zoo | 0.956 -> 0.105 | 0.851 | 0.890 |
| lin | lpf | 0.956 -> 1.000 | -0.044 | 0.000 |
| nn | fgm eps=0.10 | 1.000 -> 0.204 | 0.796 | 0.796 |
| nn | fgm eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 |
| nn | pgd eps=0.10 | 1.000 -> 0.113 | 0.887 | 0.887 |
| nn | pgd eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 |
| nn | bim eps=0.10 | 1.000 -> 0.113 | 0.887 | 0.887 |
| nn | bim eps=0.20 | 1.000 -> 0.000 | 1.000 | 1.000 |
| svm | hsj | 1.000 -> 0.000 | 1.000 | 1.000 |
| svm | zoo | 1.000 -> 0.138 | 0.862 | 0.862 |
| svm | lpf | 1.000 -> 1.000 | 0.000 | 0.000 |
| xgb | hsj | 0.996 -> 0.004 | 0.993 | 1.000 |
| xgb | zoo | 0.996 -> 0.160 | 0.836 | 0.839 |

Interpretation:

- Gradient attacks on the neural network are very strong once epsilon reaches `0.20`.
- HopSkipJump is strongest for linear, SVM, and XGB-style models.
- LowProFool does not help here; it improves or preserves accuracy instead of producing successful adversarial degradation.

## Diabetes

Diabetes is harder to attack than Banknote and Wilt. Clean accuracy is lower, and adversarial accuracy does not collapse as completely for neural-network attacks.

| Model | Attack | Clean -> Adv acc | Acc drop | ASR |
|---|---|---:|---:|---:|
| lin | zoo | 0.734 -> 0.013 | 0.721 | 0.982 |
| lin | hsj | 0.734 -> 0.266 | 0.468 | 1.000 |
| lin | lpf | 0.734 -> 1.000 | -0.266 | 0.000 |
| nn | fgm eps=0.10 | 0.714 -> 0.578 | 0.136 | 0.573 |
| nn | fgm eps=0.20 | 0.740 -> 0.305 | 0.435 | 0.930 |
| nn | pgd eps=0.10 | 0.747 -> 0.461 | 0.286 | 0.713 |
| nn | pgd eps=0.20 | 0.727 -> 0.305 | 0.422 | 0.955 |
| nn | bim eps=0.10 | 0.714 -> 0.487 | 0.227 | 0.700 |
| nn | bim eps=0.20 | 0.760 -> 0.273 | 0.487 | 0.957 |
| svm | zoo | 0.734 -> 0.162 | 0.571 | 0.779 |
| svm | hsj | 0.734 -> 0.266 | 0.468 | 1.000 |
| svm | lpf | 0.734 -> 1.000 | -0.266 | 0.000 |
| xgb | zoo | 0.740 -> 0.104 | 0.636 | 0.860 |
| xgb | hsj | 0.740 -> 0.260 | 0.481 | 1.000 |

Interpretation:

- ZOO is strongest for linear, SVM, and XGB on Diabetes by accuracy drop.
- HopSkipJump often reaches ASR `1.000`, but its adversarial accuracy is higher than ZOO because it does not necessarily push all samples into wrong final predictions under the same metric.
- BIM at epsilon `0.20` is the strongest neural-network attack.

## Wilt

Wilt shows a clear split: the linear model is unaffected, while NN, SVM, and XGB are highly vulnerable.

| Model | Attack | Clean -> Adv acc | Acc drop | ASR |
|---|---|---:|---:|---:|
| lin | zoo | 0.943 -> 0.943 | 0.000 | 0.000 |
| lin | hsj | 0.943 -> 0.943 | 0.000 | 0.000 |
| lin | lpf | 0.943 -> 0.943 | 0.000 | 0.000 |
| nn | fgm eps=0.10 | 0.983 -> 0.030 | 0.954 | 0.986 |
| nn | fgm eps=0.20 | 0.988 -> 0.026 | 0.962 | 0.986 |
| nn | pgd eps=0.10 | 0.987 -> 0.165 | 0.821 | 0.843 |
| nn | pgd eps=0.20 | 0.987 -> 0.290 | 0.696 | 0.719 |
| nn | bim eps=0.10 | 0.983 -> 0.030 | 0.954 | 0.986 |
| nn | bim eps=0.20 | 0.987 -> 0.017 | 0.970 | 0.997 |
| svm | zoo | 0.981 -> 0.031 | 0.950 | 0.968 |
| svm | hsj | 0.981 -> 0.219 | 0.762 | 0.796 |
| svm | lpf | 0.981 -> 1.000 | -0.019 | 0.000 |
| xgb | hsj | 0.983 -> 0.020 | 0.964 | 0.997 |
| xgb | zoo | 0.983 -> 0.074 | 0.909 | 0.924 |

Interpretation:

- The neural network is very vulnerable to FGM and BIM; BIM epsilon `0.20` is strongest overall.
- ZOO is strongest against SVM.
- HopSkipJump is strongest against XGB and uses a very small mean perturbation (`l2_mean=0.017`, `linf_mean=0.016`).
- The linear model appears robust in this setup, but this may also indicate that the attack configuration is not finding useful boundary-crossing perturbations for that model/dataset pair.

## Cross-Model Attack Pattern

| Attack family | Strongest target behavior |
|---|---|
| FGM | Strong against NN on Banknote and Wilt at epsilon `0.20`; weaker on Diabetes. |
| PGD | Strong against NN on Banknote; weaker than BIM/FGM on Wilt and weaker than BIM on Diabetes. |
| BIM | Best NN attack on Wilt and Diabetes; tied with FGM/PGD on Banknote at epsilon `0.20`. |
| ZOO | Best black-box attack for Diabetes linear/SVM/XGB and Wilt SVM. |
| HopSkipJump | Best black-box attack for Banknote linear/SVM/XGB and Wilt XGB. |
| LowProFool | Not effective in these runs; often preserves or improves accuracy. |

## Caveats

- ZOO used `max_iter=20` for tractable full-sample computation. The upstream notebook/code often uses larger settings, so exact upstream ZOO may be slower and may differ.
- `xgb` follows the upstream naming but is implemented as `GradientBoostingClassifier`, not the external XGBoost package.
- LowProFool cannot run on the tree/gradient-boosting wrapper because ART requires loss gradients for that attack.
- The neural-network rows include multiple epsilon settings. Epsilon `0.00` is a sanity check and should have no attack effect.

# Covertype scoring audit

The six winners from comparison `20261002_171633_4c55f3e3` were retrained
on the RTX 4060 with their exact archived chromosomes, training seeds,
split seed, subset sizes, and five-epoch schedule. All six validation
accuracies reproduced exactly. No test-set predictions were made.

For each replay, scikit-learn independently calculated accuracy from predictions,
and SciPy/NumPy calculated mean cross-entropy from float64 logits. These agreed
with the production evaluator (loss tolerance: 1e-6). The audit also verified
the earliest highest-scoring valid trial against every saved winner, the
20-candidate budgets, identical recorded subset hashes, and comparison means
and sample standard deviations. Detailed results and confusion matrices are
in `scoring-audit.json`.

| Method | Seed | Accuracy | Balanced accuracy |
|---|---:|---:|---:|
| GA | 42 | 74.19% | 43.80% |
| Aging | 42 | 74.78% | 45.94% |
| Random | 42 | 73.98% | 40.90% |
| GA | 43 | 70.97% | 34.69% |
| Aging | 43 | 73.12% | 44.75% |
| Random | 43 | 73.87% | 37.91% |

Accuracy is correct predictions divided by validation samples. Balanced accuracy
is mean recall over the seven classes. Their difference exposes poor minority
class performance on this imbalanced dataset; it is not an arithmetic bug.
The search objective remains overall validation accuracy. Before changing it,
declare the objective and rerun every method under the same protocol.

These are short proxy-training scores on 10,000 validation rows. They do not
establish fully trained performance or superiority with only two search seeds.
Repeated selection against validation can overfit it; reserve the test set for
the final, fixed protocol.

The validation regression test includes an uneven final batch and dropout,
checking sample-weighted metrics and evaluation mode. All 28 tests passed.

To repeat the audit with the local dataset cache and a CUDA environment:

```powershell
.\.venv\Scripts\python.exe audit_scoring.py experiments/tabular/comparisons/20261002_171633_4c55f3e3/comparison.json
```

The archived raw comparison files are ignored experiment outputs and must
still be available locally to replay this audit.

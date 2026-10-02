# Search space and algorithms

## Schema 3 chromosome

`ga/chromosome.py` defines the order below. A chromosome stores **indices** into
each row's choices. For example, width gene `2` selects 64, not a width of 2.
`decode` rejects a wrong length, noninteger indices (including booleans), and
out-of-range values. The schema version prevents historical CNN genes from being
interpreted as current MLP genes.

| Index | Gene | Choices |
| --- | --- | --- |
| 0 | `num_layers` | 1, 2, 3, 4 |
| 1 | `width_1` | 16, 32, 64, 128 |
| 2 | `width_2` | 16, 32, 64, 128 |
| 3 | `width_3` | 16, 32, 64, 128 |
| 4 | `width_4` | 16, 32, 64, 128 |
| 5 | `activation` | ReLU, leaky ReLU, ELU |
| 6 | `dropout` | 0, 0.1, 0.3 |
| 7 | `layer_norm` | false, true |
| 8 | `use_residual` | false, true |

There are 36,864 encoded combinations. This overcounts distinct effective models:
unused width genes do not create layers, and a residual flag has no effect where
no adjacent dimensions match. Search retains inactive genes in records so the
controller can reproduce its exact random sequence. It does not deduplicate
effective architectures or cache scores between repeated evaluations.

## Model construction

`models/mlp.py` constructs each hidden stage as linear → optional LayerNorm →
activation → dropout. If the residual flag is true and incoming/outgoing widths
match, the stage adds its input to its output. A final linear layer produces
class logits; there is no softmax before cross-entropy training.

The supported input widths are 30 and 54, which differ from all searched hidden
widths. Residual additions therefore apply only between matching hidden layers
for the current datasets. LayerNorm works with singleton batches. The parameter
estimate counts each linear layer's weights and bias, and two trainable vectors
for each LayerNorm; it is checked against actual model parameters in tests.

## Generational genetic algorithm

`ga/engine.py::run_nas` starts with a uniformly sampled population and evaluates
it in order. For each completed generation, it ranks the current observations,
keeps `n_elites` chromosomes, and chooses the remaining parents using tournament
selection. A tournament samples `k` indices **with replacement** and selects the
highest observed fitness; exact ties follow contestant order.

Pairs of parents undergo single-point crossover with probability 0.8 by default.
The split falls between genes, and no crossover returns copies of the parents.
An unpaired parent is copied. Per-gene mutation then draws a valid replacement
with probability 0.1. The new value may equal the old value; GA mutation does not
guarantee an expressed architecture change.

After three generations without a strictly better generation best, the mutation
probability doubles, capped at 0.3. A strict improvement resets the stagnation
counter. This rule is recorded in the search configuration for recovery.

The next generation includes elites and offspring. **Elites are retrained** with
the next trial seeds, consuming evaluation budget; their previous scores are not
reused during ordinary forward execution. Historical-best tracking still retains
the best individual observation, even if its architecture later scores lower.

`evaluation_budget` overrides population × generations. The last generation can
be partial; the implementation evaluates only the remaining budget and ends
before constructing another population. A 50-trial budget with population 10
means five evaluated generations. The default tournament size is 5, with 2 elites.

## Aging evolution

`ga/aging.py::run_aging_evolution` evaluates a uniformly sampled initial population
and stores `(chromosome, fitness)` pairs in a FIFO deque. Each later cycle chooses
a tournament winner from the current population, mutates one active choice,
evaluates the child, appends it, and removes the oldest population member.

`mutate_active` excludes widths beyond the current depth. It also excludes the
residual flag when no adjacent hidden widths match, because flipping it would
not change the expressed model. A selected gene changes to a different value.
Increasing depth can activate previously stored width genes.

Replacement is based on age, not on whether the child beats the parent or the
oldest member. The algorithm can remove a good model from its live population;
the shared historical archive preserves its observation as a possible winner.
With population 10 and budget 50, there are 10 initial evaluations and 40 child
evaluations. Existing population scores are reused in tournaments without retraining.

## Random search

`models/baselines/random_nas.py::run_random_search` samples chromosomes independently
from the same encoded space and evaluates them under the shared protocol. The
implementation creates candidate lists of up to 20 for deterministic replay,
but executes each candidate sequentially; this is not a GPU training batch.
Sampling ignores observed scores. The historical archive returns the best trial.

## Fair comparison and ties

Search methods use `random.Random(search_seed)` and the same sampling function.
For a given comparison seed, the initial population candidates match the first
random-search candidates. Trial `i` uses a training seed derived from SHA-256 of
`"search_seed:i"`, taking its first four bytes as a big-endian integer. This pairs
training seeds by trial position across methods, while later architectures diverge.

All methods use the same split seed, fixed subsets, proxy epochs, optimizer, and
candidate budget. Matching trial counts does not match wall-clock compute.
Repeated architectures remain separate trials; different initialization seeds
can produce different fitnesses. A tie is legitimate if methods retain the same
initial winner or produce the same discrete validation accuracy.

The search winner is the **earliest strictly highest valid observed accuracy**.
Loss, F1, size, and elapsed time do not break ties. GA population summaries and
aging live-population summaries are not necessarily historical best-so-far curves;
use the trial journal to reconstruct a historical progression.

## Failures and budgets

Successful `ok` trials can win. `parameter_limit`, `out_of_memory`, and `diverged`
outcomes receive fitness zero and consume candidate budget, but cannot become the
historical winner. Synthetic `smoke` trials can win only a smoke search. A fatal
exception is recorded as `error`, stops the search, and does not consume a
completed-trial budget. A search with no valid candidates cannot return a winner.

There is no automatic retry policy. Fix the cause and invoke recovery. The journal
records repeated fatal attempts separately, and their time contributes to
evaluation time. See [resuming-runs.md](resuming-runs.md) for replay details.

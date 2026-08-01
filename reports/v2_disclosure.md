# v2 Benchmark Disclosure — text for the methodology section

## As supplied by the supervisor (use verbatim if preferred)

> In the original synthetic benchmark (v1), the five signal features were
> generated as identically distributed noisy observations of a single latent
> variable ($sig_i = base + 0.3 \cdot \mathcal{N}(0,1)$). Consequently, the
> signal set contained mathematical redundancy, allowing classifiers to achieve
> maximal theoretical AUC using only 3 of the 5 features, which rendered the
> 5-feature recall metric unattainable. In the revised benchmark (v2), the
> signal generation has been updated to remove this redundancy, ensuring that
> all 5 features independently contribute to the target classification boundary.

## Suggested amendment (adds the measured evidence and one correction)

The paragraph above is accurate for L1–L3. **On L4 the situation is different
and stricter**, and the wording "3 of the 5" does not cover it. Recommended
replacement:

> In the original synthetic benchmark (v1), the five signal features were
> generated as identically distributed noisy observations of a single latent
> variable ($sig_i = base + 0.3 \cdot \mathcal{N}(0,1)$). The signal set
> therefore carried mathematical redundancy rather than five independent
> contributions. Measured directly on Level 2, the best three-of-five signal
> subset attains an inner-validation AUC of 1.0000, and adding the remaining two
> features changes it by $+0.0000$; the fitness function's optimum consequently
> sits at $k=3$. On Level 4 the constraint is stricter still: the signal set
> rotates at the switch, so features $s_0$–$s_2$ are predictive only before it
> and $s_3$–$s_4$ only after, leaving at most three live features in any window.
> Recall of all five signal features was therefore not merely difficult but
> unattainable by construction, and it stood in direct conflict with the
> compactness term of the fitness function, which rewards the smallest
> sufficient subset. In the revised benchmark (v2) the five signal features are
> generated from five independent latent processes whose weighted combination
> defines the target, so that each contributes independently to the decision
> boundary. Under v2 the same measurement yields $+0.0683$ AUC for the full
> signal set over the best three-of-five subset, and the fitness optimum moves
> to $k=5$.

## Supporting measurements (for a table or footnote)

| Quantity (Level 2, AR(1), fold 7) | v1 | v2 |
|---|---|---|
| AUC, best 3-of-5 signal subset | 1.0000 | 0.8885 |
| AUC, all 5 signal features | 1.0000 | 0.9567 |
| Marginal value of features 4–5 | **+0.0003** | **+0.0683** |
| Fitness argmax over signal subsets | $k=3$ | $k=5$ |

Two further v1 defects were corrected in v2 and should be disclosed alongside,
because they are the same class of problem:

**Signal and noise were separable without reference to the label.** In v1 the
45 noise features were i.i.d. at every level, while the signal features
inherited the temporal structure of `base`. On L2–L4 a selector could therefore
distinguish signal from noise by autocorrelation alone, never consulting $y$. In
v2 the noise features are drawn from the same process as the signal latents, so
the two are distinguishable only through the target.

**Level 3's drift did not stress the model.** With $y = \mathbb{1}\{base>0\}$ and
$sig \approx base$, the optimal decision boundary sits at $sig \approx 0$ and is
stationary in feature space — covariate shift with an invariant $P(y \mid x)$,
which tree ensembles handle by construction. Measured, the all-features baseline
AUC *rises* across folds on v1 L3 (0.987 → 0.997) while the label prior drifts
from a class balance of 0.63 to 0.87. In v2, L3 holds the latent processes
stationary and rotates the weight vector instead, giving concept drift with a
fixed marginal feature distribution; the baseline AUC then *falls* across folds
(0.844 → 0.634) as intended.

## Comparability warning

v1 and v2 results are not comparable and must never appear in the same table
without explicit labelling. `make_benchmark_v2.py` writes to `data/v2_*.pkl` and
leaves the v1 files untouched, so both remain reproducible. Output JSONs now
carry a `provenance` block recording the configuration and source hash each
number was produced under.

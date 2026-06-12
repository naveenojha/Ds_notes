# Chapter 3: Decision Trees
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

A decision tree recursively partitions feature space with axis-aligned splits, choosing at each node the feature-threshold pair that most reduces impurity (Gini/entropy for classification, variance/MSE for regression). Predictions are the majority class or mean of the leaf a point lands in. Single trees are interpretable but high-variance — small data changes produce different trees — which is exactly why they're rarely deployed alone and instead serve as the base learner of Random Forest and Gradient Boosting. Interviewers use trees to test whether you understand impurity math, the greedy algorithm, overfitting mechanics, and why ensembles exist.

---

## 2. Interview Articulation (3–4 Minute Answer)

"A decision tree is the model that works the way humans naturally make decisions — a sequence of if-then questions. Is order distance more than 5 km? Is it raining? Is it peak dinner hour? Each question carves the feature space into rectangles, and the prediction in each rectangle is just the average outcome of training points that fell there.

The interesting part is how the tree picks the questions. At every node, it does an exhaustive greedy search: for each feature, for each candidate threshold, it asks — if I split here, how much purer do the two children get compared to the parent? Purity is measured by Gini impurity or entropy for classification, variance for regression. The split with the biggest impurity reduction — the information gain — wins, and we recurse on each child.

Two words in that description matter for interviews: **greedy** and **exhaustive**. Greedy means it optimizes one split at a time with no lookahead — finding the globally optimal tree is NP-hard, so we accept locally optimal splits. That's why a tree can miss structure like XOR, where no single split helps but two together solve it. Exhaustive means training cost scales with features times split candidates times rows — which is exactly the cost that LightGBM's histogram trick later attacks.

We stop splitting based on pre-pruning rules — max depth, minimum samples per leaf, minimum impurity decrease — or we grow fully and post-prune with cost-complexity pruning, which penalizes leaf count with an alpha parameter, very much like L1 for trees.

Prediction is a root-to-leaf traversal — depth-many comparisons, so logarithmic-ish time, extremely fast.

The model's character: it makes essentially no assumptions — no linearity, no scaling needed because splits only care about ordering, handles mixed feature types, captures interactions automatically because a split on feature B underneath a split on feature A *is* an interaction. Its fatal flaw is variance: it's an unstable learner — perturb the training data slightly and the greedy search cascades into a completely different tree. It also can't extrapolate: outside the training range it predicts the nearest edge leaf's constant. And single deep trees memorize noise.

In industry, single trees survive in exactly two places: as interpretable rule-extraction tools — risk policies, eligibility rules, segmentation a business team can read — and as the base learner inside forests and boosting, which is where 95% of their real-world value lives. If an interviewer asks why ensembles dominate, the one-line answer is: trees are low-bias, high-variance learners, and that's the perfect raw material — bagging averages the variance away, boosting stacks the bias reduction.

When not to use: smooth linear relationships — a tree approximates a straight line with a staircase, wasteful and ugly; very small datasets; anything needing extrapolation, like forecasting a trending metric.

Traps interviewers set: thinking trees need feature scaling (they don't — splits are order-based); thinking Gini vs entropy matters much in practice (it almost never changes the tree); and the high-cardinality categorical trap — ID-like features get spuriously high gain because they can shatter the data, which is one of the classic leakage-adjacent mistakes."

---

## 3. Mathematical Foundation

**Impurity measures** (node with class proportions pₖ):
```
Gini:     G = 1 − Σₖ pₖ²            (expected misclassification rate of random labeling)
Entropy:  H = −Σₖ pₖ log₂ pₖ        (bits of uncertainty)
Regression: variance of node targets, Var = (1/n)Σ(yᵢ − ȳ)²
```
Both Gini and entropy are maximal at uniform class mix, zero at purity; entropy penalizes mixed nodes slightly more (log curvature). In practice they pick the same splits >95% of the time; Gini is the default because it's cheaper (no log).

**Split criterion (information gain / impurity decrease):**
```
ΔI = I(parent) − [ (n_L/n)·I(left) + (n_R/n)·I(right) ]
```
Choose (feature j, threshold t) maximizing ΔI. Weighted by child sizes — a pure-but-tiny child shouldn't dominate.

**Regression split:** minimize weighted child variances ⟺ maximize variance reduction. Leaf prediction = mean of leaf targets (the constant minimizing squared error in that region); for MAE criterion, the median.

**Why leaf-mean minimizes MSE:** argmin_c Σ(yᵢ−c)² ⟹ c = ȳ. One-line derivation interviewers like.

**Cost-complexity (post-)pruning:**
```
R_α(T) = R(T) + α·|leaves(T)|
```
R(T) = training error of tree T. As α grows, optimal subtree shrinks; pick α by cross-validation. This is the principled alternative to depth-capping.

**Complexity of finding a split:** for one numeric feature, sort values O(n log n), then scan thresholds maintaining running class counts O(n) — total per node O(d·n log n). Knowing the sort-then-scan trick (and that impurity updates are incremental) is a senior-level detail.

**Categorical splits:** for binary classification, the optimal subset split of a k-level categorical can be found in O(k log k) by sorting levels by their positive rate (Breiman's result) — otherwise it's 2^(k−1)−1 subsets. LightGBM uses this; sklearn historically just one-hots.

**Why greedy can fail — XOR:**
```
y = x₁ ⊕ x₂ : every single split gives ΔI = 0, yet depth-2 tree solves it perfectly.
```
Greedy with zero lookahead can stall on interaction-only signal. (Mitigations: random splits inject exploration; in practice boosting's residual fitting finds it.)

---

## 4. Step-by-Step Numerical Example

Classification: will a user order? 8 sessions.

| Rain | Hour=Dinner | Ordered |
|---|---|---|
| Y | Y | 1 |
| Y | Y | 1 |
| Y | N | 1 |
| Y | N | 0 |
| N | Y | 1 |
| N | Y | 0 |
| N | N | 0 |
| N | N | 0 |

Root: 4 positive, 4 negative.
```
Gini(root) = 1 − (0.5² + 0.5²) = 0.5
```

**Candidate split: Rain?**
```
Rain=Y: {1,1,1,0} → p=0.75 → Gini = 1 − (0.75² + 0.25²) = 0.375
Rain=N: {1,0,0,0} → p=0.25 → Gini = 0.375
ΔI = 0.5 − (4/8·0.375 + 4/8·0.375) = 0.5 − 0.375 = 0.125
```

**Candidate split: Dinner?**
```
Dinner=Y: {1,1,1,0} → Gini = 0.375
Dinner=N: {1,0,0,0} → Gini = 0.375
ΔI = 0.125  (tie)
```

Tie → pick Rain (first found / random). Recurse on Rain=Y node (3+, 1−, Gini 0.375):
```
Split by Dinner: Y → {1,1} pure (Gini 0); N → {1,0} Gini 0.5
ΔI = 0.375 − (2/4·0 + 2/4·0.5) = 0.125 → split accepted
```
Final tree (depth 2): Rain & Dinner → order (100%); Rain & not-Dinner → 50/50 leaf; no Rain & Dinner → 50/50; no Rain & not-Dinner → no order. Note the interaction captured: dinner hour matters *more* when raining — no linear model gets that without an explicit Rain×Dinner term. Practice computing exactly this on a whiteboard; it's a standard Amazon/Swiggy round exercise.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| max_depth | Caps tree depth | Variance ↑, overfit; interactions of higher order captured | Bias ↑, underfit | "Depth d captures up to d-way interactions" — quotable line |
| min_samples_split | Min rows to attempt a split | More conservative, smoother | Deeper, noisier trees | "Which prevents tiny noisy leaves better, this or min_samples_leaf?" → leaf |
| min_samples_leaf | Min rows per leaf | Regularizes; stabilizes leaf estimates | Memorization of single points | Most effective single regularizer on noisy data |
| min_impurity_decrease | Gain threshold to split | Prunes weak splits | Splits on noise | Pre-pruning vs post-pruning tradeoff question |
| ccp_alpha | Cost-complexity pruning | Smaller tree, bias ↑ | Larger tree | "How do you choose α?" → CV over the pruning path |
| max_features | Features considered per split | (toward all) stronger greedy splits, correlated trees in ensembles | More randomness — key RF lever | Belongs more to RF; know it lives at split level |
| criterion | Gini vs entropy vs log_loss | — | — | "Does it matter?" → rarely; Gini cheaper |
| class_weight | Reweights impurity by class | Minority recall ↑ | — | Imbalance handling inside the split math |

Trap: **pre-pruning can stop too early** (a weak split may enable a strong one below — the XOR effect); post-pruning grows fully then cuts, avoiding that horizon problem at extra compute cost.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | O(d · n log n) per level, roughly O(d · n log² n) total for balanced trees; exact greedy is the cost driver histogram methods attack |
| Inference | O(depth) comparisons — nanoseconds; branch-heavy but trivially cheap |
| Scalability | Single trees train fine to ~10⁷ rows; beyond that use histogram-based learners; trees parallelize per-feature at split search |
| Memory | Nodes ∝ leaves; deep unpruned trees on big data can be surprisingly large (millions of nodes) |
| Distributed training | Rare for single trees; the patterns (feature-parallel / data-parallel split finding) matter for the GBM chapter |
| Monitoring | Leaf population drift (rows per leaf shifting), input range drift (extrapolation = constant edge-leaf predictions), structural churn across retrains (expected — trees are unstable; alert stakeholders who screenshot "the tree" that it will change) |
| Serving form | Often compiled to nested if-else / SQL CASE statements — zero-dependency deploys; rule extraction for policy engines |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **Gini vs entropy — difference and which to use?** Both measure node impurity; entropy from information theory, Gini = expected misclassification under random labeling. Nearly identical splits in practice; Gini default (no log computation).
2. **Why don't trees need feature scaling?** Splits depend only on the *ordering* of values, not magnitudes; any monotone transform of a feature yields the identical tree.
3. **How does a tree handle non-linearity?** Piecewise-constant approximation: enough axis-aligned splits approximate any boundary — staircases for smooth functions (inefficient but consistent).
4. **How do trees capture interactions?** A split on B conditional on A's branch makes B's effect depend on A — depth-k trees express up-to-k-way interactions without manual crosses.
5. **What causes overfitting in trees and how do you control it?** Growth until leaves memorize noise; control via depth/leaf-size limits, impurity thresholds, cost-complexity pruning, or — the real answer — ensembling.
6. **Pre-pruning vs post-pruning?** Pre: stop early (cheap, risks myopia — the XOR horizon problem). Post: grow full, prune back with CV'd α (better trees, more compute).
7. **How are missing values handled?** Options: surrogate splits (CART — backup features mimicking the primary split), default-direction learning (XGBoost/LightGBM send missings to the gain-maximizing child), or imputation. Knowing default-direction is the modern answer.
8. **Why are trees unstable / high-variance?** Greedy split choice: a small data change flips the top split and the entire subtree cascades differently. This instability is the *precondition* for bagging to help.
9. **Can trees extrapolate?** No — predictions outside the training range are the nearest boundary leaf's constant. Critical for trending time-series targets; detrend or use a linear component.
10. **Regression tree: what does a leaf predict and why?** Mean of leaf targets — the constant minimizing squared error; median if MAE criterion.
11. **What is information gain ratio and why did C4.5 use it?** Gain divided by the split's own entropy (split info) — penalizes many-valued features whose raw gain is inflated by fragmentation.
12. **High-cardinality categoricals — what goes wrong?** Features like user_id can shatter data into near-pure tiny groups → huge spurious gain, zero generalization (memorization). Mitigate: target encoding with CV, frequency capping, exclusion.
13. **Decision tree vs logistic regression?** Tree: non-linear, interactions free, no scaling, poor calibration, no extrapolation, unstable. LR: linear log-odds, calibrated, stable, needs FE. Decide by data shape and downstream use of scores.
14. **How do you get probabilities from a tree, and are they good?** Leaf class fractions; typically poorly calibrated (small leaves → extreme 0/1 estimates). Calibrate (Platt/isotonic) or smooth (Laplace correction).
15. **Feature importance from a single tree — how, and a caveat?** Total impurity decrease attributed per feature. Caveats: biased toward high-cardinality/continuous features; unstable across retrains (single-tree importances are nearly anecdotal).

### Advanced (15)

1. **Prove leaf-mean optimality for MSE.** d/dc Σ(yᵢ−c)² = −2Σ(yᵢ−c) = 0 ⟹ c = ȳ. Then note the tree objective decomposes leaf-wise, so greedy leaf fitting is exact *given* the partition.
2. **Why is optimal tree construction NP-hard, and what's the practical consequence?** Exponential partition space; hence greedy local search, no global optimality guarantee, sensitivity to data perturbation, and the XOR-style failure mode.
3. **Walk through the O(n log n) split-search algorithm for a numeric feature.** Sort by feature; sweep thresholds between consecutive distinct values maintaining running class counts/sums; impurity updates in O(1) per step. Midpoints as candidate thresholds.
4. **Breiman's categorical trick — why does sorting by positive rate work?** For binary outcomes with Gini/entropy, the optimal subset split respects the ordering of category-level target rates, reducing 2^(k−1)−1 subsets to k−1 ordered cuts. (Holds for binary; multiclass is genuinely hard.)
5. **Derive the Gini gain for a candidate split symbolically and explain the size weighting.** ΔI as in §3; child impurities weighted by n_child/n so the criterion equals reduction in *expected* impurity of a random sample — prevents tiny pure leaves from dominating.
6. **Why does entropy's log make it slightly more sensitive to class probability changes near 0/1?** Derivative of −p log p diverges as p→0; entropy rewards purifying rare classes marginally more than Gini's polynomial curvature — hence occasional split differences on imbalanced nodes.
7. **Axis-aligned vs oblique trees?** Standard trees split on one feature; oblique trees split on linear combinations (hyperplanes) — more expressive, far costlier to search, rarely worth it because ensembles + feature engineering close the gap.
8. **How does class_weight change the split math?** Impurity computed on *weighted* class proportions; equivalent to oversampling in the criterion; shifts splits toward isolating the minority class.
9. **Connection between tree leaves and adaptive nearest neighbors?** A tree's leaf defines a data-adaptive neighborhood; prediction = average over neighbors *as defined by the tree metric*. (This view explains forests as adaptive-kernel methods — strong staff-level aside.)
10. **Why are single-tree probability estimates extreme, and what is Laplace smoothing here?** Small leaves yield 0/1 frequencies; add-one smoothing (k+1)/(n+2) pulls estimates toward 0.5, trading bias for calibration sanity.
11. **What's the variance-reduction interpretation of a regression split?** ΔVar = between-group variance created by the split (law of total variance): splitting moves variance from "within" to "between," and the tree greedily maximizes explained variance — regression trees are greedy ANOVA.
12. **How do surrogate splits work and when do they fail?** Backup features ranked by agreement with the primary split's partition; used when the primary feature is missing. Fail when missingness is informative (MNAR) — default-direction learning handles that better.
13. **Minimal cost-complexity pruning path — what's the algorithm?** Weakest-link pruning: repeatedly collapse the internal node with smallest per-leaf error increase g(t) = (R(t)−R(T_t))/(|T_t|−1); yields a nested subtree sequence indexed by α; CV picks the winner.
14. **Effect of duplicated features (perfect copies) on a tree vs on its feature importances?** Tree is unaffected (one copy gets picked arbitrarily); importance mass splits arbitrarily between copies across retrains — why impurity importance misleads with correlated features (permutation importance shares the same dilution issue; SHAP splits credit).
15. **When does a depth-1 tree (stump) outperform a deep tree?** Tiny/noisy data, single dominant threshold effect, or as the weak learner in AdaBoost — where bias is intentionally kept high and boosting composes the complexity.

### Staff-Level (10)

1. **A risk team deploys your decision tree as written policy rules. Six months later performance decays. Discuss the failure modes and your redesign.** Rules froze a high-variance snapshot; population drift + adversarial adaptation (fraud) erode rectangles; no probability outputs for graded action. Redesign: retrain cadence with stability constraints (restrict structure changes), monitor per-leaf volume/positive-rate drift, move policy to score+threshold from a calibrated ensemble while keeping a rule-extracted shadow for auditability.
2. **Stakeholders love that they can read the tree. You know it's unstable. How do you give interpretability without lying?** Present *stable* artifacts: permutation/SHAP importances aggregated over bootstraps, partial dependence, extracted rule sets with support/confidence — not "the" tree. If a literal tree is mandated, constrain depth and demonstrate stability across resamples before publishing it.
3. **Your tree's top split is on a feature that's 40% missing. Walk through your concerns.** Is missingness informative (MNAR) or pipeline breakage? Default-direction learned on training missingness may not transfer; train/serve missing-rate parity checks; consider missing-indicator features; quantify gain attribution to the missingness pattern itself vs the values.
4. **Design a system that converts a trained tree into production SQL for a no-ML-infra client. Risks?** CASE-statement compilation, fixed-point thresholds, type/null semantics parity, version pinning, golden-row regression tests between Python and SQL scorers; the real risk is silent transform skew upstream of the thresholds.
5. **An ID-like feature shows the highest information gain in your churn tree. What's your read?** Memorization/leakage red flag: high-cardinality shatter or target leakage through identity. Verify with grouped CV (split by entity), drop or encode with out-of-fold target stats, re-examine gain. The offline metric was likely fiction.
6. **Trees for uplift modeling — what changes?** Split criterion changes from outcome purity to *treatment-effect heterogeneity* (e.g., maximizing divergence between treatment/control outcome distributions across children — uplift trees, causal trees with honest splitting: separate samples for structure vs leaf estimates). Knowing "honesty" (Athey-Imbens) is the differentiator.
7. **When would you bound tree depth at 3–4 even with abundant data?** Interpretability/policy constraints, interaction-order control as a scientific prior, base learners for boosting (bias intentionally high), latency-bounded scoring in tight loops, and stability for regulated decisioning.
8. **Your regression tree must predict a trending target (weekly GMV). Architecture?** Trees can't extrapolate: model trend/seasonality with a linear/statistical component and let the tree fit detrended residuals (hybrid), or featurize the trend (time index handled linearly). Pure tree = flat forecasts at the data edge — say this proactively.
9. **How do trees interact with differential privacy or data minimization requirements?** Leaves are aggregates — small leaves can re-identify (k-anonymity violation); enforce min_samples_leaf ≥ k for privacy, noise leaf statistics for DP trees, audit rule outputs for quasi-identifier combinations.
10. **You must explain to a regulator why two near-identical customers got different decisions from a tree-based system.** Boundary discontinuity: piecewise-constant models step at thresholds. Mitigations: margin/score-based decisions with smooth calibration layers, threshold buffer zones with manual review, or monotonic constraints so direction of effects is guaranteed and explainable.

---

## 8. Comparison Section

| | Decision Tree | Random Forest | Gradient Boosting | Logistic Regression | KNN |
|---|---|---|---|---|---|
| Bias / Variance | Low bias / high variance | Low bias / reduced variance | Tunable (sequential bias reduction) | High bias / low variance | Low bias / high variance |
| Interpretability | Fully readable (if small) | Importances only | SHAP needed | Coefficients | None |
| Training | One greedy pass | Parallel trees | Sequential trees | Convex optimization | None (lazy) |
| Calibration | Poor | Mediocre | Poor → recalibrate | Good | Poor |
| Extrapolation | No | No | No | Yes (linear) | No |
| Scaling needed | No | No | No | Yes | Yes (critical) |

**Tree vs Forest in one line:** a forest is variance therapy for trees — same low-bias learner, averaged over bootstraps and feature subsets until the instability cancels.

**Tree vs GBM in one line:** forest averages independent strong trees; boosting adds dependent weak trees, each fixing the last one's residuals — variance reduction vs bias reduction.

**Stump vs deep tree:** stumps = additive main effects only (no interactions); deep trees = high-order interactions + memorization risk. Depth is the interaction-order dial.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Scaling features for trees (signals shallow understanding instantly).
- Unable to compute a Gini gain by hand.
- Saying "trees overfit" without naming the mechanism (greedy growth to pure leaves) or the controls.
- Missing the greedy/no-lookahead limitation and the XOR example.
- Treating single-tree feature importance as reliable.

**Production mistakes:**
- Shipping unpruned trees ("it worked offline") — memorized leaves decay fast.
- Freezing a tree into business rules without drift monitoring.
- Ignoring extrapolation on trending targets.
- High-cardinality ID features silently memorizing (grouped CV would have caught it).

**Modeling mistakes:**
- Random CV with entity repetition (user appears in train and test) — trees exploit identity leakage aggressively.
- Reading probabilities off raw leaf frequencies without calibration.
- Comparing tree importance across correlated features and drawing causal conclusions.

---

## 10. Real Industry Use Cases

- **Amazon** — single trees as rule extraction for seller/buyer abuse policies; CART-style segmentation in operations (why a shipment is late — readable splits for ops teams).
- **Google** — decision trees as the pedagogical/diagnostic layer; production value lives in their boosted/forest descendants; rule mining for spam heuristics historically.
- **Netflix** — segmentation trees for messaging eligibility rules; diagnostic trees explaining A/B heterogeneity (which cohorts moved).
- **Meta** — the famous GBDT-as-feature-transformer pipeline begins with understanding single-tree leaves as learned crossed features.
- **Uber** — eligibility/policy rules (promo abuse gates) compiled from trees; uplift trees for incentive targeting.
- **Swiggy/Zomato** — readable churn-driver trees for category teams; COD-risk rule policies; serviceability rules (rain × distance × prep-time interactions — exactly the worked example above). *Naveen: a depth-3 tree on delivery-delay drivers is a great "explainability artifact" story to pair with your GBM ranking work — the readable model that earned stakeholder trust for the complex one.*
- **Flipkart** — return-risk policy rules, seller-tiering segmentation trees.
- **Games24x7** — responsible-gaming intervention rules (readable, auditable by compliance — a domain where the *single tree's* interpretability is the requirement, not a limitation); fraud triage trees routing cases to manual review queues.

---

## 11. Coding From Scratch (NumPy only)

CART for classification with Gini, depth and leaf-size controls.

```python
import numpy as np

class Node:
    __slots__ = ("feature", "threshold", "left", "right", "value")
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature, self.threshold = feature, threshold   # split definition
        self.left, self.right = left, right                 # child nodes
        self.value = value                                  # class probs if leaf

class DecisionTreeScratch:
    def __init__(self, max_depth=5, min_samples_leaf=2, min_gain=1e-7):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_gain = min_gain
        self.root, self.n_classes = None, None

    @staticmethod
    def _gini(counts):
        # counts: class counts in a node. Gini = 1 - sum(p_k^2).
        n = counts.sum()
        if n == 0:
            return 0.0
        p = counts / n
        return 1.0 - (p @ p)

    def _best_split(self, X, y):
        n, d = X.shape
        parent_counts = np.bincount(y, minlength=self.n_classes)
        parent_gini = self._gini(parent_counts)
        best = (None, None, 0.0)                      # (feature, threshold, gain)

        for j in range(d):
            order = np.argsort(X[:, j])               # sort once per feature
            xs, ys = X[order, j], y[order]
            left = np.zeros(self.n_classes)           # running counts: the O(1)
            right = parent_counts.astype(float).copy()#   incremental-update trick

            for i in range(n - 1):                    # sweep candidate thresholds
                left[ys[i]] += 1
                right[ys[i]] -= 1
                if xs[i] == xs[i + 1]:                # can't split between equal values
                    continue
                nl, nr = i + 1, n - i - 1
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = parent_gini - (nl * self._gini(left) +
                                      nr * self._gini(right)) / n
                if gain > best[2]:
                    best = (j, (xs[i] + xs[i + 1]) / 2.0, gain)  # midpoint threshold
        return best

    def _grow(self, X, y, depth):
        counts = np.bincount(y, minlength=self.n_classes)
        # Stopping conditions → make a leaf storing class probabilities.
        if depth >= self.max_depth or len(np.unique(y)) == 1 \
           or len(y) < 2 * self.min_samples_leaf:
            return Node(value=counts / counts.sum())

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain < self.min_gain:      # pre-pruning on weak gain
            return Node(value=counts / counts.sum())

        mask = X[:, feat] <= thr
        return Node(feature=feat, threshold=thr,
                    left=self._grow(X[mask], y[mask], depth + 1),
                    right=self._grow(X[~mask], y[~mask], depth + 1))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self.n_classes = y.max() + 1
        self.root = self._grow(X, y, depth=0)
        return self

    def _traverse(self, x, node):
        while node.value is None:                     # until we hit a leaf
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X):
        return np.array([self._traverse(x, self.root) for x in np.asarray(X, float)])

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)
```

Narration points that earn senior credit:
- **Sort-once, sweep, O(1) incremental count updates** in `_best_split` — say "this is why split search is O(n log n) per feature, and what histogram methods later approximate."
- **Midpoint thresholds between distinct values** — and the `xs[i] == xs[i+1]` guard; forgetting it is the classic from-scratch bug.
- **Leaves store probability vectors**, not labels — enables predict_proba and a calibration discussion.
- **min_gain as pre-pruning** — connect to the lookahead limitation you mentioned in articulation.
- Recursion depth: mention you'd convert to an explicit stack for very deep trees in production code.

---

## 12. ML System Design Perspective

**Choose a single tree when:** the deliverable is *rules humans will read or audit* (policy, compliance, ops playbooks); you need a fast non-linear baseline to estimate interaction structure before ensembling; serving must compile to SQL/if-else with zero dependencies; teaching/diagnostic contexts (explaining heterogeneity in experiment results).

**Avoid when:** accuracy is the goal (use the ensemble), the target trends (no extrapolation), data is tiny and noisy (variance kills you), or smooth monotone relationships dominate (linear models are cheaper and better).

**Data requirements:** robust to mixed types, outliers (split ordering immune), unscaled features, moderate missingness (with default-direction handling); vulnerable to high-cardinality IDs and entity leakage — grouped CV is non-negotiable when entities repeat.

**Latency:** depth-many comparisons; effectively free. The latency conversation only becomes real with hundreds of boosted trees.

**Scale limits:** exact greedy split search is the bottleneck (O(d·n log n) per level); past ~10⁷ rows or 10³ features, move to histogram-based implementations — which is the engineered answer, not "trees don't scale."

---

## 13. Resume Discussion Angle

**Recommendation/ranking systems:** "Where do trees fit in your ranking story?" Strong answer: as the base learner — "my ranking work was LambdaMART, which is boosted regression trees optimizing a ranking objective, so single-tree mechanics — split gain, depth-as-interaction-order, leaf values — are the substrate I tuned." Then one concrete: depth choices in your ranker as deliberate interaction-order control. *This makes Chapter 3 the foundation of your LambdaMART depth, not a separate topic — frame it that way explicitly.*

**Fraud detection:** "Why Isolation Forest + XGBoost rather than a readable tree?" Strong answer: readable trees served as the *triage/policy layer* (audit requirement), ensembles as the scoring layer; describe how you kept rule extraction in the loop for compliance while the ensemble handled accuracy. Expect the follow-up "how did you stop the tree memorizing player IDs" → grouped CV by player, out-of-fold encodings.

**Personalization/marketing ML:** "How did you find which segments responded?" Strong answer: trees on experiment data to surface heterogeneous treatment effects (and name *honest* splitting if pushed) — but caveat exploratory vs confirmatory; segments found by trees were re-validated with targeted experiments before budget moved.

**Universal trap:** "Show me you understand why your GBM works." The expected substrate: trees are low-bias/high-variance greedy partitioners; boosting controls bias sequentially, regularization and shrinkage control variance; depth controls interaction order. If you can't discuss the single tree crisply, your GBM credibility drops — interviewers explicitly chain these.

---
*Previous: Logistic Regression ← | Next: Random Forest →*

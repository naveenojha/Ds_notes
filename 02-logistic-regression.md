# Chapter 2: Logistic Regression
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

Logistic regression models the probability of a binary outcome by passing a linear combination of features through a sigmoid. It's trained by maximizing likelihood — equivalently, minimizing log loss — via gradient-based optimization, since no closed form exists. It outputs *calibrated probabilities* (when trained on representative data), has a linear decision boundary, and remains the most deployed classifier in industry: CTR prediction, fraud scoring, churn, credit risk. At senior level, interviewers use it to test probability theory, MLE, calibration, and class-imbalance reasoning.

---

## 2. Interview Articulation (3–4 Minute Answer)

"Logistic regression answers a different question than linear regression: not 'how much' but 'how likely'. The naive idea — fit a line to 0/1 labels — fails because lines produce values outside [0,1] and squared loss doesn't match the geometry of probabilities. So we keep the linear score, w-transpose-x, but interpret it as the **log-odds** of the positive class, and squash it through the sigmoid to get a probability.

That log-odds framing is the heart of the model. The sigmoid isn't an arbitrary squashing function — it's exactly what you get when you say 'the log of p over 1-minus-p is linear in the features.' That gives coefficients a clean meaning: increase feature j by one unit, the odds multiply by e-to-the-beta-j. That's why credit risk and healthcare still run on it — every coefficient is a defensible statement.

Training is maximum likelihood. Each example contributes p if it's positive, 1−p if it's negative; take logs, flip sign, and you get binary cross-entropy. There's no closed form because the sigmoid makes the score equations nonlinear — but the loss is convex, so gradient descent or Newton's method finds the global optimum. The gradient turns out to be beautifully simple: for each example, (predicted probability minus label) times the features. Identical in form to linear regression's gradient — that's not a coincidence, both are GLMs with canonical links.

Inference is one dot product plus a sigmoid, microsecond-fast, and crucially you get a probability, not just a class. In real systems we almost never use 0.5 as the threshold — we pick it from the precision-recall tradeoff and the business cost matrix, or we don't threshold at all and consume the probability downstream, like expected-value ranking in ads: bid times pCTR.

Assumptions: linearity in the *log-odds*, independent observations, no severe multicollinearity. Notably no Gaussian assumptions anywhere. Weaknesses: linear boundary unless you engineer interactions, sensitivity to correlated features for interpretation, and complete separation — if a feature perfectly splits the classes, weights diverge to infinity unless you regularize.

In industry it's the default first classifier and often the last: CTR models at scale were logistic regression with massive sparse one-hot features for a decade — hashed features, FTRL online learning, billions of parameters. It's also the calibration layer: even when a GBM or neural net does the heavy lifting, a logistic (Platt) layer often sits on top to fix the probabilities.

When not to use it: heavy non-linear interaction structure with no appetite for feature engineering — GBMs win on raw tabular accuracy. And the traps: people say it's a 'linear model' and conclude it can't be — it's linear in log-odds with a non-linear probability response; people threshold at 0.5 by reflex; and people downsample negatives and forget the probabilities are no longer calibrated — you have to correct the intercept."

---

## 3. Mathematical Foundation

**Model:**
```
p(y=1|x) = σ(wᵀx) = 1 / (1 + e^(−wᵀx))

Equivalent log-odds (logit) form:
log[ p / (1−p) ] = wᵀx        ← linear in features
```

**Likelihood and loss:** with yᵢ ∈ {0,1}, pᵢ = σ(wᵀxᵢ):
```
L(w) = Πᵢ pᵢ^yᵢ (1−pᵢ)^(1−yᵢ)

NLL = −Σᵢ [ yᵢ log pᵢ + (1−yᵢ) log(1−pᵢ) ]     (binary cross-entropy)
```

**Gradient derivation** (the most-asked derivation in DS interviews):
```
Key fact: σ'(z) = σ(z)(1−σ(z))

∂NLL/∂w = Σᵢ (pᵢ − yᵢ) xᵢ        i.e.  Xᵀ(p − y)
```
Sketch: ∂/∂w[yᵢ log pᵢ] = yᵢ(1−pᵢ)xᵢ and ∂/∂w[(1−yᵢ)log(1−pᵢ)] = −(1−yᵢ)pᵢxᵢ; sum and simplify to (pᵢ−yᵢ)xᵢ. Memorize the cancellation — interviewers want it fluently.

**Why convex:** the Hessian is XᵀSX with S = diag(pᵢ(1−pᵢ)) ⪰ 0 ⟹ positive semi-definite ⟹ global optimum, no local minima.

**Optimization:**
- Gradient descent / SGD: w ← w − α Xᵀ(p−y)/n
- Newton / IRLS: w ← w − (XᵀSX)⁻¹Xᵀ(p−y) — quadratic convergence, cost O(d³) per step; this is what statsmodels/R use
- L-BFGS: the practical default (sklearn)
- FTRL-Proximal: online learning with per-coordinate rates + L1, the classic web-scale CTR setup

**Regularization:** L2 (default; also fixes separation divergence), L1 (sparse, for huge one-hot spaces). MAP view identical to ridge/lasso priors.

**Multiclass (softmax):**
```
p(y=k|x) = exp(wₖᵀx) / Σⱼ exp(wⱼᵀx);  loss = −Σ log p(y=true class)
```

**Calibration note:** trained on representative data with log loss, predicted probabilities match observed frequencies in aggregate (log loss is a *proper scoring rule*). If you downsample negatives at rate r, correct the intercept: β₀' = β₀ − log(1/r) (equivalently subtract log of the sampling odds).

---

## 4. Step-by-Step Numerical Example

One feature, w = 1.5, b = −3 (already trained). Predict churn from inactivity weeks.

```
x = 4:  z = 1.5(4) − 3 = 3.0     p = 1/(1+e⁻³)   ≈ 0.953
x = 2:  z = 1.5(2) − 3 = 0.0     p = 1/(1+e⁰)    = 0.500
x = 1:  z = 1.5(1) − 3 = −1.5    p = 1/(1+e^1.5) ≈ 0.182
```

**One gradient step by hand.** Data: (x=2, y=0) and (x=4, y=1); start w=0, b=0, lr=0.1.
```
Both predictions: σ(0) = 0.5
Errors (p − y):  (0.5 − 0) = 0.5   and  (0.5 − 1) = −0.5

∂L/∂w = mean[(p−y)·x] = (0.5·2 + (−0.5)·4)/2 = (1 − 2)/2 = −0.5
∂L/∂b = mean[(p−y)]   = (0.5 − 0.5)/2 = 0

w ← 0 − 0.1(−0.5) = 0.05 ;  b ← 0
```
The weight moved positive — toward predicting higher churn probability for higher inactivity. Verify the loss before the step: −[log(0.5) + log(0.5)] / 2 = 0.693 = log 2, the classic "untrained binary classifier" loss. Quoting that number fluently is a good signal.

**Odds-ratio interpretation:** w = 1.5 ⟹ each extra inactivity week multiplies churn odds by e^1.5 ≈ 4.48.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| C (sklearn) = 1/λ | Inverse regularization | Less regularization → variance ↑, separation risk | More shrinkage → bias ↑ | "C vs λ direction" trips many candidates |
| Penalty (L1/L2/EN) | Type of shrinkage | — | — | "Why L1 for CTR models?" → sparse weights over billions of hashed features |
| class_weight | Reweights loss per class | More weight on minority → recall ↑, precision ↓, probabilities distorted | — | "class_weight vs oversampling vs threshold moving?" |
| Solver | Optimizer | — | — | "When liblinear vs lbfgs vs saga?" → saga for L1 at scale; lbfgs default |
| max_iter / tol | Convergence | — | Early stop → underfit | Convergence warnings = unscaled features, usually |
| Decision threshold | Operating point (not a training hyperparameter!) | Precision ↑ recall ↓ | Opposite | "How do you pick it?" → cost matrix / PR curve, never default 0.5 |

Trap: **threshold is a business decision, not a model parameter** — separating those two cleanly is a senior-level signal.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | O(n·d) per epoch (sparse: O(nnz)); Newton O(d³)/step — avoid for large d |
| Inference | O(d) dense, O(active features) sparse — microseconds; SQL/edge deployable |
| Scalability | The canonical web-scale classifier: hashing trick + FTRL handled billions of sparse features at Google/Meta-era CTR systems; Spark/parameter-server friendly |
| Memory | One float per feature; L1 prunes most weights in sparse regimes |
| Online learning | First-class: per-example SGD/FTRL updates; easy warm-starts between retrains |
| Monitoring | **Calibration drift** (reliability curves, expected calibration error) — the #1 silent failure; prior drift (base rate shift moves the intercept); AUC/log loss on rolling windows; weight diffs across retrains |

Calibration monitoring matters because downstream systems often consume the probability itself (bids, expected-loss decisions) — AUC can stay flat while calibration quietly breaks.

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **Why not linear regression on 0/1 labels?** Unbounded predictions, heteroscedastic errors by construction, MSE is a poor match (non-convex pairing with sigmoid); log loss + sigmoid is the principled MLE setup.
2. **Why sigmoid specifically?** It's the inverse of the logit; follows from modeling log-odds as linear. Also the canonical link of the Bernoulli GLM (bonus: arises naturally from log-odds in Bayes' rule with exponential-family class conditionals).
3. **Interpret a coefficient.** One-unit increase in xⱼ multiplies the *odds* by e^βⱼ, holding others fixed. Not the probability — the relationship to probability is non-linear and depends on the operating point.
4. **What loss and why?** Binary cross-entropy = negative log likelihood of Bernoulli; convex; proper scoring rule ⟹ calibrated probabilities.
5. **Why no closed-form solution?** Score equations Xᵀ(p−y)=0 are nonlinear in w because p = σ(wᵀx); requires iterative optimization.
6. **What is complete separation?** A hyperplane perfectly splits classes ⟹ likelihood increases without bound, weights → ∞. Symptoms: huge coefficients/SEs, non-convergence. Fix: L2 regularization or Firth's penalized likelihood.
7. **Logistic regression vs decision trees?** LR: linear boundary, probabilities, extrapolates smoothly, needs FE for interactions. Trees: axis-aligned non-linearity, no scaling needed, poor calibration, no extrapolation.
8. **Handling class imbalance?** First question: do you need classification or ranking/probabilities? Options: class weights, resampling (then recalibrate), threshold tuning, focal-style losses; evaluate with PR-AUC, not accuracy.
9. **Why is accuracy bad for imbalanced data?** A 99%-negative dataset gives 99% accuracy to a useless constant predictor. Use PR-AUC, recall@precision, cost-weighted metrics.
10. **ROC-AUC vs PR-AUC — when each?** ROC-AUC insensitive to base rate (good for comparing rankers); PR-AUC reflects performance on the rare positive class — preferred for fraud/anomaly settings.
11. **Does logistic regression need feature scaling?** Not for correctness of the optimum, but yes in practice: faster convergence and required for fair regularization.
12. **One-hot vs target encoding here?** One-hot natural for LR (each level gets a weight); high-cardinality → hashing trick or target encoding with leakage protection.
13. **Multiclass extension?** Softmax (multinomial) trained jointly — preferred; or one-vs-rest. Softmax probabilities sum to 1; OvR's don't.
14. **What does the intercept mean?** Log-odds of the positive class when features are at zero/reference; encodes the base rate — which is why sampling changes require an intercept correction.
15. **Model predicts 0.9 — what does that mean operationally?** If calibrated: among similar predictions, ~90% are positive. Verify with a reliability diagram before letting downstream systems consume it as a probability.

### Advanced (15)

1. **Derive ∂NLL/∂w.** Show the σ'(z)=σ(1−σ) cancellation arriving at Xᵀ(p−y). Whiteboard fluency expected.
2. **Prove convexity.** Hessian XᵀSX, S = diag(p(1−p)) ≥ 0 ⟹ PSD. Implication: any local min is global; solvers differ only in speed.
3. **MLE vs MAP view of regularization.** L2 = Gaussian prior, L1 = Laplace prior on weights; λ ↔ prior precision.
4. **Why does log loss yield calibrated probabilities but hinge loss not?** Log loss is a strictly proper scoring rule — minimized in expectation only by the true conditional probability. Hinge loss targets the decision boundary margin; SVM scores aren't probabilities (hence Platt scaling — which is literally fitting a logistic regression on the scores).
5. **Downsampled negatives at 1% for CTR training — what breaks and how do you fix it?** Predicted probabilities inflate ~100×in odds terms; correct intercept by −log(100) (or recalibrate). AUC/ranking unaffected; calibration is what breaks.
6. **Newton/IRLS vs gradient descent tradeoffs.** Newton: quadratic convergence, O(d³)/step, gives standard errors via the Hessian; GD/L-BFGS: scalable. IRLS interpretation: weighted least squares with weights p(1−p) recomputed each step.
7. **Why do weights p(1−p) make intuitive sense in IRLS?** Examples near p=0.5 are most informative about the boundary and get the most weight; confident examples contribute little curvature.
8. **Connection between logistic regression and a single-neuron network?** Identical model; BCE + sigmoid = same gradient (p−y)x. Softmax regression = single dense layer + softmax. Good bridge answer into deep learning rounds.
9. **Naive Bayes vs logistic regression — the generative/discriminative story.** NB models p(x|y)p(y), LR models p(y|x) directly. Ng & Jordan: NB converges faster with tiny data (higher bias, lower variance); LR wins asymptotically. Gaussian NB with shared covariance implies a logistic posterior — same form, different fitting.
10. **How does multicollinearity affect LR?** Same as OLS: unstable coefficients, inflated SEs, sign flips; predictions can remain fine. VIF on the design matrix, ridge to stabilize.
11. **Pseudo-R² — why no real R²?** No residual variance decomposition; use McFadden's 1 − ll/ll_null, or better: log loss/AUC/calibration on holdout. Production answer: nobody uses pseudo-R².
12. **FTRL — why was it the standard for online CTR?** Per-coordinate adaptive learning rates + L1 sparsity in a streaming setting; produces small sparse models from billions of hashed features with one-pass training.
13. **How do you get confidence intervals on coefficients?** Asymptotic normality: SEs from inverse Hessian at the MLE; Wald/likelihood-ratio tests; bootstrap when assumptions are shaky or after selection.
14. **Effect of measurement noise in a feature?** Attenuation bias — coefficient biased toward zero; ranking degrades gracefully but coefficient-based decisions become understated.
15. **Why can adding an irrelevant but correlated feature hurt calibration after a retrain?** Effective regularization per informative feature changes; weight mass redistributes among correlated columns; intercept compensates differently. Monitor calibration, not just AUC, across retrains.

### Staff-Level (10)

1. **Your fraud model's PR-AUC is stable but losses are rising. Investigate.** Adversarial drift in feature space (fraudsters adapt within the same score distribution), label latency/maturation bias, threshold staleness as base rate shifts, calibration drift, segment-level decay masked by aggregates. Propose segment monitoring, faster label feedback, periodic threshold re-optimization against the cost matrix.
2. **GBM beats LR by 3% AUC offline for credit decisions. Recommendation?** Weigh regulatory explainability (adverse-action reasons), calibration quality, fairness auditability, monotonicity requirements; options: monotonic-constrained GBM + SHAP reason codes, or LR with engineered interactions; quantify the 3% in ₹ terms and decide with risk/compliance, not offline AUC alone.
3. **Design web-scale CTR prediction with logistic regression.** Hashed sparse one-hot + cross features; FTRL online updates; negative downsampling with intercept correction; calibration layer + reliability monitoring; cold-start priors by ad/advertiser hierarchy; position bias handling in training labels (position as a feature at train, fixed at serve — a clean segue into your LTR depth).
4. **The model's probabilities feed an expected-value bidder. What's your monitoring stack?** Calibration curves & ECE per traffic segment, base-rate drift vs intercept, score distribution shift (PSI), realized-vs-predicted ratio dashboards, canary retrains with champion/challenger on revenue-weighted metrics.
5. **When is logistic regression the right *final* layer on top of complex models?** Stacking/calibration: combining sub-model scores with a convex, auditable, online-tunable blend; Platt scaling for SVM/GBM outputs; distillation of complex models into deployable scorecards.
6. **How do you defend coefficient-based "drivers of churn" insights to leadership?** Caveat associational vs causal; check stability across resamples/time; partial dependence consistency; propose targeted experiments on the top actionable drivers before committing spend.
7. **Severe class imbalance (1:10⁵) with delayed labels (chargebacks). Training design?** Time-aware splits, label maturation windows, importance-weighted sampling with calibration correction, PU-learning or delayed-feedback modeling if labels censor, cost-sensitive thresholding tied to expected loss per decision.
8. **You inherit an LR with 2M hashed features and no documentation. De-risking plan?** Freeze + shadow a retrain; reconstruct feature lineage; measure hash-collision impact; distill to an interpretable feature set; calibration and segment audits before any change ships.
9. **Fairness ask: equalize false-positive rates across user segments. Implications?** Per-segment thresholds vs constrained training; tension with calibration (impossibility results: can't have calibration + equal FPR/FNR with different base rates — know this result); align with policy/legal on which criterion binds.
10. **Why might a perfectly good LR degrade after a feature store migration with no code changes?** Train/serve skew: imputation defaults, scaling stats, timezone/window semantics, or null-handling changed; weights are tuned to the old transforms. Detect via score distribution diffs and feature-level parity tests between old/new pipelines.

---

## 8. Comparison Section

**Logistic Regression vs XGBoost** (asked constantly):

| | Logistic Regression | XGBoost |
|---|---|---|
| Boundary | Linear in features | Non-linear, interactions automatic |
| Probabilities | Naturally calibrated (proper loss, representative data) | Often miscalibrated → recalibrate |
| Feature prep | Scaling, one-hot, manual interactions | Minimal; handles raw numerics, missing values |
| Interpretability | Odds ratios, regulatory-grade | SHAP, weaker for compliance |
| Latency | µs | ms (hundreds of tree traversals) |
| Online learning | Native (SGD/FTRL) | Retrain-based |
| When it wins | Sparse high-dim (text/CTR one-hots), tiny data, compliance, speed | Dense tabular with interactions |

**vs Naive Bayes:** discriminative vs generative; NB wins with very little data or when features genuinely near-independent (classic text spam); LR wins asymptotically and with correlated features.

**vs SVM:** hinge maximizes margin, no probabilities; LR gives probabilities and degrades gracefully with overlap; both linear-boundary by default. Platt scaling = LR bolted onto SVM scores.

**vs Neural nets:** LR = 0-hidden-layer NN; choose NN when representation learning is needed (embeddings, raw signals), LR when features are already informative and you want speed + audit.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Interpreting coefficients as probability changes instead of odds multipliers.
- Defaulting to 0.5 threshold / accuracy on imbalanced data.
- Unable to derive the gradient or quote log(2) untrained loss.
- Saying "logistic regression can't work for non-linear data" without mentioning feature engineering / its dominance in sparse high-dim regimes.
- Forgetting calibration entirely — at senior level this is *the* differentiating topic.

**Production mistakes:**
- Downsampling negatives without intercept correction.
- Monitoring AUC only while calibration drifts.
- Train/serve transform skew (scaling stats, null handling).
- Stale thresholds as base rates shift.

**Modeling mistakes:**
- Ignoring separation (silent in unregularized fits — huge weights).
- Leakage via post-outcome features (fraud labels especially).
- One-hot dummy trap with unregularized fits.
- Treating class_weight as free recall — it distorts probabilities; recalibrate.

---

## 10. Real Industry Use Cases

- **Google** — the canonical FTRL logistic-regression CTR system for search ads (the "Ad Click Prediction: a View from the Trenches" lineage); still the conceptual baseline every DLRM-style model is compared against.
- **Meta** — historical GBDT+LR stack: trees as feature transformers feeding a logistic layer for ads CTR.
- **Amazon** — buy-propensity and deal-ranking probability layers; fraud and abuse scoring with auditable scorecards.
- **Netflix** — propensity layers in messaging/notification optimization; calibration layers over ranker outputs.
- **Uber** — fraud (payment risk) scorecards, driver-churn early warning, ETA-reliability classification components.
- **Swiggy/Zomato** — conversion propensity (menu→cart→order funnel), churn/winback targeting, COD-risk scoring, restaurant-ad pCTR baselines. *Naveen: pCTR-for-ads with calibration ties directly to your Zomato ads revenue story and your calibration work — lead with that pairing.*
- **Flipkart** — CTR/CVR baselines in search ads and recommendation re-rank blends, return-abuse risk scoring.
- **Games24x7** — responsible-play risk flags, deposit-propensity models, churn scoring for engagement campaigns; LR favored where compliance teams need reason codes.

---

## 11. Coding From Scratch (NumPy only)

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, lr=0.1, n_iters=2000, l2=0.0, tol=1e-7):
        self.lr, self.n_iters, self.l2, self.tol = lr, n_iters, l2, tol
        self.w = None     # weights incl. bias at index 0

    @staticmethod
    def _sigmoid(z):
        # Numerically stable sigmoid: avoid exp overflow for large |z|.
        out = np.empty_like(z)
        pos = z >= 0
        out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
        ez = np.exp(z[~pos])
        out[~pos] = ez / (1.0 + ez)
        return out

    def _add_bias(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(np.asarray(X, float))
        y = np.asarray(y, float)
        n, d = Xb.shape
        self.w = np.zeros(d)
        prev_loss = np.inf

        for _ in range(self.n_iters):
            p = self._sigmoid(Xb @ self.w)               # predictions in (0,1)
            grad = Xb.T @ (p - y) / n                    # the (p - y) gradient
            grad[1:] += (self.l2 / n) * self.w[1:]       # L2, skip bias
            self.w -= self.lr * grad

            # Clipped log loss for monitoring/early stop (avoid log(0)).
            eps = 1e-12
            loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps)) \
                   + (self.l2/(2*n)) * (self.w[1:] @ self.w[1:])
            if abs(prev_loss - loss) < self.tol:
                break                                     # converged
            prev_loss = loss
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._add_bias(np.asarray(X, float)) @ self.w)

    def predict(self, X, threshold=0.5):
        # Threshold is a *business* parameter; 0.5 is just a default.
        return (self.predict_proba(X) >= threshold).astype(int)
```

Points to narrate while coding (these earn the senior signal):
- **Stable sigmoid**: naive `1/(1+exp(-z))` overflows for z ≪ 0; the branch trick keeps exp arguments negative. Interviewers at Google/Amazon specifically look for numerical-stability awareness.
- **Gradient is Xᵀ(p−y)/n** — same shape logic as linear regression; say "GLM with canonical link" once.
- **Clipping inside the log**, not the gradient — clipping probabilities used in the gradient would bias updates.
- **Bias excluded from L2** — same reasoning as ridge.
- Zero initialization is fine here (convex problem — unlike neural nets, no symmetry-breaking needed). Stating *why* zero-init is okay here but not in NNs is a great unprompted aside.

---

## 12. ML System Design Perspective

**Choose it when:** sparse high-dimensional features (text, one-hot ID crosses); calibrated probabilities are consumed downstream (bidding, expected-loss decisions); online/streaming updates needed; compliance demands reason codes; latency budget is microseconds; it's the calibration or blending layer over stronger models.

**Avoid when:** dense tabular data with strong interactions and no FE budget (GBM); raw perceptual inputs without embeddings; the problem is fundamentally ranking with listwise objectives (go to LTR — though pointwise LR pCTR remains a legitimate stage-1).

**Data requirements:** rule of thumb ≥ 10–20 *events per variable* (positives, in imbalanced settings — a subtle, senior-level detail); representative sampling or documented corrections.

**Latency:** the fastest probabilistic classifier available; sparse dot products scale with active features, not total features.

**Scale limits:** none meaningful for inference; training scales to billions of rows/features with hashing + SGD/FTRL. Constraint is expressiveness, not throughput.

---

## 13. Resume Discussion Angle

**Recommendation/ranking systems:** "You used pCTR in ads ranking — why LR / how calibrated?" Strong answer: pointwise pCTR as stage-1 or blend input; trained on logged impressions with position-bias handling; negative downsampling with intercept correction; reliability-diagram monitoring because the bid multiplies the probability — miscalibration directly misprices auctions. *This connects your Zomato ads (₹40L/month) story to your calibration work in one arc — rehearse that link.*

**Fraud detection:** Expect "why not just XGBoost?" Strong answer: LR as the auditable scorecard/challenger and calibration reference; cost-matrix-driven thresholds, not 0.5; delayed-label handling; segment-level monitoring. If your Rummy collusion project comes up: position LR as the explainable baseline that justified the Isolation Forest + XGBoost escalation.

**Personalization/churn (marketing ML):** "How did you pick who to target?" Strong answer separates propensity from *uplift* — high churn probability ≠ persuadable user; mention you'd evolve propensity targeting toward uplift modeling/holdout-validated incrementality. Interviewers at McKinsey QB in particular probe exactly this distinction.

**The universal trap question:** "Your model outputs 0.7 — what do you tell the business it means?" Never say "70% chance, period." Say: "If calibrated — which we verify with reliability curves — then among users scored ~0.7, about 70% churn. Then the action depends on the cost matrix and persuadability, not the score alone."

---
*Previous: Linear Regression ← | Next: Decision Trees →*

# Chapter 1: Linear Regression
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

Linear regression models a continuous target as a weighted sum of input features plus an intercept. It finds the weights that minimize squared prediction error, either through a closed-form solution (OLS) or gradient descent. It's the baseline for every regression problem, the foundation of GLMs and causal inference, and the most common "depth probe" in senior interviews — interviewers use it to test whether you actually understand optimization, assumptions, and statistical reasoning, not whether you can fit a line.

---

## 2. Interview Articulation (3–4 Minute Answer)

> Say this naturally. Pause where the line breaks suggest.

"Linear regression is the simplest possible statement about a relationship: I believe my target moves proportionally with my features. If I'm predicting delivery time, I'm saying — every extra kilometer adds roughly X minutes, every extra item in the order adds Y minutes, and there's some base time. That's the entire model: y equals a weighted sum of features plus a bias.

The question is how to find those weights. The standard answer is least squares — I pick the weights that minimize the sum of squared differences between my predictions and the actual values. Squared, not absolute, for two reasons: it's differentiable everywhere so optimization is clean, and statistically it corresponds to maximum likelihood if I assume Gaussian noise around the true relationship. That equivalence is worth knowing — minimizing MSE *is* MLE under Gaussian errors.

Training has two routes. If the data fits in memory, there's a closed-form solution — the normal equation, beta equals X-transpose-X inverse times X-transpose-y. Geometrically it's projecting the target vector onto the column space of the features. At scale, inverting a d-by-d matrix is cubic in dimensionality, so for high-dimensional or streaming data we use gradient descent instead — the gradient of MSE is just the residuals projected back through the features, so each update nudges weights in the direction that shrinks errors.

Inference is trivially fast — one dot product. That's actually one of its production superpowers: microsecond latency, tiny memory footprint, and you can implement scoring in SQL if you have to.

The assumptions matter and are where interviews go: linearity of the relationship, independence of errors, homoscedasticity — constant error variance — and normality of *errors*, not features, and only if you want valid confidence intervals. For pure prediction, you can violate normality and still predict fine. The silent killer is multicollinearity — correlated features make X-transpose-X near-singular, coefficients become unstable and uninterpretable, even though predictions can still be okay. That's where ridge regularization comes in.

Strengths: interpretability, speed, statistical inference — you get p-values and confidence intervals, which trees don't give you. It's also the workhorse of causal analysis and marketing mix modeling. Weaknesses: it can't capture non-linearity or interactions unless you hand-engineer them, and it's sensitive to outliers because squared loss amplifies large residuals.

In industry I reach for it when I need explainability for stakeholders, a fast baseline, or coefficient-level inference — incrementality measurement, pricing elasticity, demand drivers. I avoid it when relationships are clearly non-linear and I don't need interpretability — that's gradient boosting territory.

Common traps interviewers set: claiming features must be normally distributed (false — errors, and only for inference); confusing R² with model quality (high R² with overfitting or extrapolation is meaningless); and forgetting that correlation between residuals — say, in time series — invalidates standard errors even if point predictions look fine."

---

## 3. Mathematical Foundation

**Model:**
```
ŷ = Xβ        where X ∈ ℝ^(n×d) (with bias column of 1s), β ∈ ℝ^d
```

**Loss (MSE / RSS):**
```
J(β) = (1/n) ‖y − Xβ‖²  =  (1/n) Σᵢ (yᵢ − xᵢᵀβ)²
```

**Closed-form derivation (Normal Equation):**
```
∇β J = −(2/n) Xᵀ(y − Xβ) = 0
⟹ XᵀXβ = Xᵀy
⟹ β* = (XᵀX)⁻¹ Xᵀy
```
Intuition: set the gradient to zero ⟹ residuals (y − Xβ) must be **orthogonal** to every feature column. The prediction Xβ* is the orthogonal projection of y onto the column space of X — the closest point to y you can reach using linear combinations of the features.

**Gradient descent update:**
```
β ← β + (2α/n) Xᵀ(y − Xβ)
```
Each step moves weights in the direction features correlate with current residuals.

**MLE connection:** Assume y = Xβ + ε, ε ~ N(0, σ²I). The log-likelihood is
```
log L(β) = −(n/2)log(2πσ²) − (1/2σ²) ‖y − Xβ‖²
```
Maximizing log L ⟺ minimizing ‖y − Xβ‖². **OLS = MLE under Gaussian noise.**

**Gauss–Markov theorem:** Under linearity, exogeneity, homoscedasticity, and no perfect multicollinearity, OLS is **BLUE** — Best Linear Unbiased Estimator (minimum variance among linear unbiased estimators). Note: normality is *not* required for BLUE; it's required for exact finite-sample inference (t-tests, F-tests).

**Regularized variants:**

| Variant | Objective | Effect |
|---|---|---|
| Ridge (L2) | ‖y − Xβ‖² + λ‖β‖² | Shrinks coefficients; handles multicollinearity; closed form β = (XᵀX + λI)⁻¹Xᵀy |
| Lasso (L1) | ‖y − Xβ‖² + λ‖β‖₁ | Sparse solutions (exact zeros); implicit feature selection; no closed form |
| Elastic Net | mix of L1+L2 | Sparsity + stability with correlated features |

Why Lasso gives zeros: the L1 ball has corners on the axes; the loss contours typically first touch the constraint region at a corner, where some coefficients are exactly zero. The L2 ball is smooth, so coefficients shrink but rarely hit zero.

---

## 4. Step-by-Step Numerical Example

Data: predict salary (LPA) from years of experience.

| x (years) | y (LPA) |
|---|---|
| 1 | 10 |
| 2 | 14 |
| 3 | 18 |
| 4 | 26 |

Simple linear regression: ŷ = β₀ + β₁x.

```
x̄ = (1+2+3+4)/4 = 2.5        ȳ = (10+14+18+26)/4 = 17

β₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²

Numerator:  (−1.5)(−7) + (−0.5)(−3) + (0.5)(1) + (1.5)(9)
          = 10.5 + 1.5 + 0.5 + 13.5 = 26
Denominator: 2.25 + 0.25 + 0.25 + 2.25 = 5

β₁ = 26/5 = 5.2
β₀ = ȳ − β₁x̄ = 17 − 5.2(2.5) = 4.0
```

Model: **ŷ = 4 + 5.2x** — base salary ₹4 LPA, +₹5.2 LPA per year of experience.

Residuals: at x = (1,2,3,4): ŷ = (9.2, 14.4, 19.6, 24.8) ⟹ residuals (0.8, −0.4, −1.6, 1.2).
```
SSE = 0.64 + 0.16 + 2.56 + 1.44 = 4.8
SST = Σ(yᵢ−ȳ)² = 49 + 9 + 1 + 81 = 140
R²  = 1 − 4.8/140 ≈ 0.966
```
Interview point: be able to do exactly this on a whiteboard. It's a common Amazon Breadth & Depth warm-up.

---

## 5. Hyperparameters

Plain OLS has none; the practical hyperparameters live in regularized/SGD variants.

| Hyperparameter | What it does | Increase → | Decrease → | Interview question |
|---|---|---|---|---|
| λ / α (ridge, lasso) | Regularization strength | More shrinkage, higher bias, lower variance; lasso zeros more features | Closer to OLS; risk of overfit/instability | "How do you choose λ?" → CV on validation loss; or analytical: λ scales with noise/signal ratio |
| Learning rate (SGD) | Step size of updates | Faster but may diverge/oscillate | Stable but slow; may stall | "Loss is oscillating, what do you check first?" → LR too high / features unscaled |
| l1_ratio (Elastic Net) | L1 vs L2 mix | More sparsity | More ridge-like grouping of correlated features | "Two highly correlated features, what does Lasso vs Ridge do?" → Lasso arbitrarily picks one; Ridge splits weight between them |
| fit_intercept | Learn β₀ | — | Forces line through origin — almost always wrong unless data is centered | "When would you not fit an intercept?" |
| Polynomial degree (if expanding features) | Model flexibility | Variance ↑, overfit risk | Bias ↑, underfit | Classic bias-variance probe |

Key trap: **regularization requires standardized features** — otherwise λ penalizes features unequally just because of their units.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | Normal equation: O(nd² + d³). SGD: O(n·d) per epoch. Rule of thumb: closed form fine until d ≈ 10⁴ |
| Inference complexity | O(d) per row — a dot product. Sub-millisecond, trivially batchable |
| Scalability | Embarrassingly parallel SGD/mini-batch; Spark MLlib handles billions of rows; can even be solved via distributed sufficient statistics (XᵀX, Xᵀy are summable across partitions) |
| Memory | Model = d floats. Tiny. Can be deployed in SQL, on-device, in a feature store transform |
| Distributed training | Aggregate XᵀX and Xᵀy per partition, sum, solve once — exact distributed OLS without iterative training |
| Monitoring | Residual distribution drift, coefficient drift across retrains (instability ⟹ multicollinearity or data issues), feature drift (PSI), R²/MAE on rolling windows. Watch for extrapolation: inputs outside training range |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **Why squared error and not absolute error?** Differentiable everywhere, closed-form solution, MLE under Gaussian noise. MAE = MLE under Laplace noise and is more robust to outliers but needs subgradient methods.
2. **What are the assumptions of linear regression?** Linearity, independence of errors, homoscedasticity, normality of errors (for inference), no perfect multicollinearity.
3. **What is R², and what's wrong with it?** Fraction of variance explained. It never decreases when you add features — use adjusted R² or out-of-sample error. High R² ≠ good model (overfitting, extrapolation, spurious time-series regressions).
4. **What is multicollinearity and how do you detect it?** Correlated predictors making XᵀX near-singular; coefficients unstable with huge standard errors. Detect via VIF > 5–10, condition number, coefficient sign flips across resamples. Fix: drop/combine features, ridge, PCA.
5. **Interpret a coefficient.** Expected change in y per unit change in xⱼ, *holding other features fixed* — the "holding fixed" caveat is what they're testing.
6. **What happens with outliers?** Squared loss gives outliers quadratic influence; a single point can drag the fit. Mitigate with Huber loss, RANSAC, winsorizing, or investigating the outlier.
7. **Normal equation vs gradient descent — when each?** Closed form when d is small/moderate and data fits memory; GD/SGD when d is large, data streams, or XᵀX inversion is too costly.
8. **What if errors are heteroscedastic?** Point estimates remain unbiased, but standard errors are wrong ⟹ invalid inference. Use weighted least squares, robust (sandwich) standard errors, or transform y (log).
9. **Why standardize features?** Required for fair regularization and for comparing coefficient magnitudes; speeds up GD convergence (better-conditioned loss surface).
10. **Difference between correlation and regression coefficient?** Correlation is symmetric and unitless; the slope is correlation scaled by sd(y)/sd(x) and is directional.
11. **How does Ridge fix multicollinearity?** Adds λI to XᵀX, making it well-conditioned and invertible; trades a little bias for a large variance reduction in coefficients.
12. **Lasso vs Ridge — when each?** Lasso when you believe few features matter (sparse truth, want selection); Ridge when many small effects and correlated features; Elastic Net when both.
13. **How do you handle categorical features?** One-hot with a dropped reference level (dummy trap: full one-hot + intercept = perfect collinearity); target encoding for high cardinality, with leakage controls.
14. **What is the dummy variable trap?** Including all k categories plus intercept makes columns linearly dependent ⟹ XᵀX singular. Drop one level.
15. **Train R² = 0.95, test R² = 0.4 — diagnosis?** Overfitting (too many features / leakage / unstable coefficients) or distribution shift. Check coefficient stability, run CV, inspect feature drift.

### Advanced (15)

1. **Derive the normal equation.** ∇β‖y−Xβ‖² = −2Xᵀ(y−Xβ) = 0 ⟹ β = (XᵀX)⁻¹Xᵀy. Mention the orthogonality of residuals to column space.
2. **Prove OLS = MLE under Gaussian noise.** Write the likelihood of y given X, β, σ²; log it; the only β-dependent term is −‖y−Xβ‖²/2σ².
3. **What does Gauss–Markov actually guarantee, and what doesn't it?** BLUE — minimum variance among *linear unbiased* estimators. It does not say OLS beats biased estimators: ridge can have lower MSE via bias-variance tradeoff. Doesn't need normality.
4. **Bias-variance decomposition of expected test error.** E[(y−ŷ)²] = Bias² + Variance + irreducible noise σ². Be able to sketch the derivation by adding/subtracting E[ŷ].
5. **Why does ridge have a Bayesian interpretation?** Ridge = MAP estimate with Gaussian prior β ~ N(0, τ²I); λ = σ²/τ². Lasso = MAP with Laplace prior.
6. **Geometric reason Lasso induces sparsity?** Loss contours meet the L1 ball's corners (axis points) with positive probability; L2 ball is rotationally smooth so solutions are rarely exactly on axes.
7. **What if n < d?** XᵀX is singular; infinitely many interpolating solutions. Need regularization. (Bonus: minimum-norm solution via pseudoinverse; connects to modern over-parameterization/double descent discussions.)
8. **Influence of a single point — leverage and Cook's distance?** Leverage hᵢᵢ from the hat matrix H = X(XᵀX)⁻¹Xᵀ measures how unusual xᵢ is; Cook's distance combines leverage and residual to flag influential points.
9. **Autocorrelated residuals (time series) — consequences?** Coefficients still unbiased but standard errors badly wrong (usually understated); R² inflated. Use Newey–West errors, ARIMA errors, or model the dynamics explicitly. Detect via Durbin–Watson / residual ACF.
10. **Omitted variable bias — direction?** If omitted z affects y and correlates with included x, x's coefficient absorbs z's effect: bias = β_z · cov(x,z)/var(x). Core causal-inference question.
11. **Why can adding a feature flip another coefficient's sign?** Coefficients are partial effects; conditioning on a correlated feature changes the conditional relationship (Simpson's paradox in regression form).
12. **Connection between OLS and projection matrices?** ŷ = Hy with H = X(XᵀX)⁻¹Xᵀ; H is symmetric idempotent (H² = H), trace(H) = d = degrees of freedom used.
13. **How would you fit OLS on 10B rows distributed?** Compute partial XᵀX (d×d) and Xᵀy (d×1) per partition, tree-aggregate, solve on driver. Exact, one pass, no SGD needed if d is modest.
14. **Quantile regression — when?** When you care about conditional quantiles (P90 delivery time, not mean) or have heteroscedastic/asymmetric errors. Pinball loss instead of MSE.
15. **Log-transforming y — what changes?** Coefficients become approximately percentage effects (semi-elasticity); back-transforming predictions needs a bias correction (Duan smearing / exp(σ²/2) under lognormality) because E[exp(ε)] ≠ exp(E[ε]).

### Staff-Level (10)

1. **Your pricing model's coefficients change sign every weekly retrain. Walk through your investigation.** Check multicollinearity (VIF, condition number), sample size per segment, feature pipeline drift, leakage, regularization absence; consider whether stakeholders need stable *causal* estimates (then move to a designed approach — regularization, fewer features, or experiments) vs pure prediction (then sign instability may be acceptable).
2. **PM wants to use your demand regression's coefficients to make pricing decisions. Risks?** Coefficients are associational; price is endogenous (set in response to demand) ⟹ classic simultaneity bias. Propose experiments, instrumental variables, or natural experiments before causal claims.
3. **When would you ship linear regression over XGBoost that's 5% better offline?** Latency/memory constraints (edge, SQL scoring), regulatory interpretability, coefficient-based business decisions, tiny data where boosting overfits, or when 5% offline rarely survives online and simplicity reduces operational risk.
4. **Design the retraining and monitoring strategy for a linear model in production.** Scheduled + drift-triggered retrains; champion/challenger; monitor residual drift, coefficient deltas with alerting thresholds, feature PSI; shadow scoring before promotion; rollback plan.
5. **Linear model as the final stage of a ranking system — why might that be a deliberate choice?** Calibrated, monotone, auditable score combination of upstream model outputs; easy online weight tuning; supports business-rule overrides. (Many production rankers end in a linear blend of sub-model scores.)
6. **How does regularization interact with feature scaling in an automated pipeline, and what breaks silently?** Unscaled features get unevenly penalized; a pipeline change that alters units silently changes effective regularization per feature. Enforce scaling inside the model pipeline, not upstream.
7. **You need uncertainty estimates on predictions for downstream decisioning. Options?** Analytic prediction intervals from OLS (if assumptions hold), bootstrap intervals, Bayesian linear regression (closed-form posterior), conformal prediction (assumption-light, my default in production).
8. **n=2,000, d=50,000 (genomics-style). Approach?** Sparse methods: Lasso/Elastic Net with nested CV, stability selection to control false discoveries; screen features first (univariate, sure independence screening); be honest about inference validity post-selection.
9. **How do you explain to leadership why R² dropped after you fixed a leakage bug?** The old number was fiction — model saw future information. Frame the new metric as the true production-attainable performance; back it with a backtest mimicking deployment-time data availability.
10. **Double descent — does "more parameters than data" always overfit?** Not necessarily: in the over-parameterized regime, minimum-norm interpolating solutions can generalize well; test error can descend again past the interpolation threshold. Shows the classical bias-variance picture is incomplete. (Differentiator answer at staff level.)

---

## 8. Comparison Section

| | Linear Regression | Ridge | Lasso | XGBoost (reg) | Logistic Regression |
|---|---|---|---|---|---|
| Target | Continuous | Continuous | Continuous | Continuous | Probability/class |
| Non-linearity | No (manual FE) | No | No | Yes, automatic | No |
| Feature selection | No | No | Yes (zeros) | Implicit (splits) | No (use L1) |
| Interpretability | Coefficients + CIs | Coefficients (biased) | Sparse coefficients | SHAP needed | Odds ratios |
| Multicollinearity | Breaks coefficients | Handles | Arbitrary picks | Robust for prediction | Same issues as OLS |
| Latency | Microseconds | Same | Same | ~ms (tree traversals) | Microseconds |

**Linear regression vs trees/GBMs:** linear wins on extrapolation beyond training range (trees predict constants outside seen feature ranges), inference speed, interpretability, small-n stability. GBMs win on raw accuracy with tabular non-linear data.

**OLS vs Ridge, one line:** OLS is unbiased but high-variance under collinearity; ridge buys big variance reduction for small bias — lower expected MSE in practice.

---

## 9. Common Mistakes

**Candidate mistakes (interview):**
- Saying *features* must be normal (it's errors, and only for inference).
- Quoting assumptions as a memorized list without consequences-of-violation. Always pair: violation → what breaks → fix.
- Claiming high R² proves a good model.
- Not knowing the closed form or being unable to derive it on a whiteboard.
- Interpreting coefficients causally without being asked to caveat.

**Production mistakes:**
- Regularizing unscaled features.
- Training/serving skew in feature transforms (offline standardization stats ≠ online).
- Letting the model extrapolate silently outside training ranges.
- Retraining without coefficient-stability checks.

**Modeling mistakes:**
- One-hot + intercept dummy trap.
- Target leakage via post-outcome features.
- Fitting levels on trending time series (spurious R² ≈ 1); difference or detrend first.
- Treating p-values as valid after heavy feature selection on the same data.

---

## 10. Real Industry Use Cases

- **Amazon** — demand forecasting baselines and elasticity estimation for pricing; regression-based incrementality readouts for marketing spend.
- **Google** — marketing mix modeling (their open-source MMM tooling is Bayesian regression at heart); experiment metric adjustment (CUPED is variance reduction via a regression on pre-period covariates).
- **Netflix** — interleaving/AB metric modeling and covariate adjustment; simple regressions for content-performance drivers presented to non-technical stakeholders.
- **Meta** — lift measurement and geo-experiment readouts via regression frameworks.
- **Uber** — surge/ETA components historically began as regularized linear models; causal regression for driver-incentive ROI.
- **Swiggy/Zomato** — delivery-time prediction baselines (distance, prep-time, rain features), elasticity of conversion to delivery fee, restaurant ads incrementality readouts. *Naveen: this slots directly into your Zomato narrative as the "V0 baseline" framing — linear model as the explainable baseline the GBM had to beat.*
- **Flipkart** — GMV driver decomposition, marketing budget allocation via regression-based MMM.
- **Games24x7** — LTV early-prediction baselines; regression on engagement covariates for campaign uplift readouts.

---

## 11. Coding From Scratch (NumPy only)

```python
import numpy as np

class LinearRegressionScratch:
    def __init__(self, method="normal", lr=0.01, n_iters=1000, l2=0.0):
        # method: "normal" (closed form) or "gd" (gradient descent)
        # l2: ridge penalty lambda (0 = plain OLS)
        self.method, self.lr, self.n_iters, self.l2 = method, lr, n_iters, l2
        self.beta = None                      # learned weights, bias included

    def _add_bias(self, X):
        # Prepend a column of 1s so beta[0] acts as the intercept.
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def fit(self, X, y):
        Xb = self._add_bias(np.asarray(X, float))
        y = np.asarray(y, float)
        n, d = Xb.shape

        if self.method == "normal":
            # Ridge-regularized normal equation: (X'X + λI)β = X'y.
            # We don't penalize the intercept, so zero out I[0,0].
            I = np.eye(d); I[0, 0] = 0.0
            # np.linalg.solve is more stable than explicitly inverting.
            self.beta = np.linalg.solve(Xb.T @ Xb + self.l2 * I, Xb.T @ y)
        else:
            self.beta = np.zeros(d)
            for _ in range(self.n_iters):
                resid = Xb @ self.beta - y            # shape (n,): prediction errors
                grad = (2.0 / n) * (Xb.T @ resid)     # gradient of MSE wrt beta
                grad[1:] += (2.0 * self.l2 / n) * self.beta[1:]  # ridge term, skip bias
                self.beta -= self.lr * grad           # step downhill
        return self

    def predict(self, X):
        return self._add_bias(np.asarray(X, float)) @ self.beta

    def r2(self, X, y):
        y = np.asarray(y, float)
        resid = y - self.predict(X)
        return 1.0 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
```

Line-by-line points interviewers probe:
- `np.linalg.solve` vs `inv`: solving the linear system is faster and numerically stabler than forming an explicit inverse — say this unprompted.
- Bias column trick: folding the intercept into the weight vector keeps all math as one matrix product.
- Not penalizing the intercept: penalizing β₀ would shrink predictions toward zero regardless of the data's mean — a subtle correctness point.
- Gradient shape sanity: Xᵀ(residuals) maps n-vector errors back to d-vector weight updates.
- For GD, mention you'd standardize X first — condition number of XᵀX controls convergence speed.

---

## 12. ML System Design Perspective

**Choose linear regression when:** you need explainable coefficients for business decisions; latency budget is sub-millisecond or scoring must run in SQL/on-device; data is small or features are well-understood and roughly linear; it's the baseline in a model-evolution story; you need statistical inference (CIs, significance) for stakeholders.

**Avoid when:** strong non-linearities/interactions dominate and interpretability isn't required; heavy-tailed targets without transformation; raw unstructured inputs (text/images) without an embedding stage.

**Data requirements:** rule of thumb ≥ 10–20 rows per feature for stable coefficients; clean handling of categoricals; scaled features if regularized.

**Latency:** O(d) dot product — the fastest model class available; ideal for the final scoring layer in latency-critical ranking paths.

**Scale limits:** essentially none for inference; training scales via distributed sufficient statistics or SGD. The binding constraint is *expressiveness*, not scale.

---

## 13. Resume Discussion Angle

If linear regression appears in your recommendation/ranking/personalization/fraud/marketing stories, expect:

**"Why a linear model and not GBM/DL?"**
Strong answer: frame it as a deliberate stage. "It was the V0 baseline that set the bar and gave us coefficient-level insight into what actually drove the metric — distance and prep-time explained 70% of delivery-time variance — which guided the feature engineering for the GBM that replaced it. The linear model also stayed in production as the fallback path because it scores in microseconds and never fails weird."

**"How did you validate the coefficients meant anything?"**
Stability across bootstrap resamples and weekly retrains, VIF checks, holdout error, and where decisions depended on them, an experiment to confirm directionality.

**"How did you handle non-linearity?"**
Explicit transformations (log, splines, binned interactions) where domain knowledge suggested them; once hand-crafting saturated, that was the documented justification for moving to gradient boosting — a clean model-evolution narrative.

**"Marketing ML angle (MMM/incrementality):"**
Expect endogeneity questions. Strong answer acknowledges spend is not randomly assigned, describes adstock/saturation transforms, regularized Bayesian MMM, and validation against geo-experiments. Saying "the regression proved channel X caused sales" with no causal design is the instant red flag — never do it.

---
*Next chapter: Logistic Regression →*

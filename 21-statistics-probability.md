# Chapter 21: Statistics & Probability
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist — **The Finale**

> The connective tissue of the entire handbook — MLE underlies linear/logistic regression, Bayes underlies Naive Bayes, the CLT underlies every confidence interval, and A/B testing is how every model in this book actually proves its worth. This is the Amazon Breadth & Depth and McKinsey QuantumBlack case-round material. The structure differs from prior chapters: instead of one algorithm, it's five connected pillars — **probability fundamentals, estimation, hypothesis testing, A/B testing, and Bayesian inference** — each with the depth a senior interview probes.

---

## 1. Executive Summary (30 seconds)

Statistics is how you reason from data under uncertainty, and it underpins everything else in this handbook. Five pillars: **probability** (distributions, Bayes' rule, expectation/variance — the language of uncertainty); **estimation** (MLE and MAP — how every model in this book is fit, and how confident you are in the estimate); **hypothesis testing** (p-values, the two error types, power — how you decide whether an effect is real); **A/B testing** (the applied core — how you prove a model or product change works, with all the traps: peeking, multiple comparisons, variance reduction); and **Bayesian inference** (updating beliefs with priors, the alternative to frequentist testing). Senior interviews weight the *applied judgment* — A/B testing pitfalls, the prediction-vs-causation distinction, what a p-value does and doesn't mean — over textbook recall.

---

## 2. Interview Articulation (3–4 Minute Answer)

"Statistics is the discipline of drawing conclusions from data while being honest about uncertainty, and it's the foundation under every model in this handbook — when I fit a logistic regression I'm doing maximum likelihood estimation, when I report a confidence interval I'm relying on the central limit theorem, and when I prove a model improves the business I'm running a hypothesis test. So rather than treat it as a separate topic, I think of it as five connected ideas.

First, **probability** — the language of uncertainty. The pieces that matter are the common distributions and *when* each arises — Bernoulli and binomial for counts of successes, Poisson for rare-event counts, normal because of the central limit theorem, exponential for waiting times — plus Bayes' rule, which is just the definition of conditional probability rearranged, and the workhorses expectation and variance. The single most important result is the **central limit theorem**: the average of many independent samples is approximately normal regardless of the underlying distribution, which is *why* we can put confidence intervals around almost any estimate.

Second, **estimation** — given data, what's my best guess for a parameter, and how sure am I? The dominant principle is **maximum likelihood**: pick the parameter that makes the observed data most probable. That's not just a statistics topic — it's literally how linear regression, logistic regression, and most of this book's models are fit. The Bayesian cousin is **MAP**, maximum a posteriori, which adds a prior — and the beautiful connection is that L2 regularization is exactly a Gaussian prior and L1 is a Laplace prior, so regularization *is* Bayesian estimation. Around any estimate I want a **confidence interval**, which quantifies sampling uncertainty.

Third, **hypothesis testing** — deciding whether an effect is real or just noise. You set up a null hypothesis of 'no effect,' compute how surprising your data would be if the null were true — that's the p-value — and reject the null if it's sufficiently surprising. The crucial thing to say correctly, because interviewers bait it: a p-value is *not* the probability the null is true; it's the probability of seeing data this extreme *if* the null were true. There are two error types — Type I, a false positive, rejecting a true null, controlled by your significance level alpha; and Type II, a false negative, missing a real effect, related to **power**, the probability of detecting a true effect. Power analysis — figuring out the sample size you need *before* running the test — is where this becomes practical.

Fourth, and most important for industry, **A/B testing** — the applied core, how you actually prove a model or feature works. The logic is randomization: randomly assign users to control and treatment, and randomization makes the groups comparable on everything, so a difference in outcomes is *caused* by the treatment. This is where prediction becomes causation, which is the whole point. But A/B testing is a minefield of pitfalls that senior interviews probe hard: **peeking** — checking results repeatedly and stopping when significant inflates false positives, because you get many chances to cross the threshold; the **multiple comparisons** problem — test twenty metrics and one will look significant by chance; **insufficient power** — calling a test 'no effect' when you just didn't have enough users; **network effects and interference** — in a marketplace, treating some users affects the control group, breaking the independence assumption; and **novelty effects** — a change looks great for a week because it's new, then fades. And there's **variance reduction**, like CUPED, which uses pre-experiment data to shrink the noise and detect effects with less traffic — a technique I'd expect to discuss at senior level.

Fifth, **Bayesian inference** — the alternative philosophy: instead of a yes/no reject decision, maintain a full probability distribution over the parameter, update it with data via Bayes' rule, and read off probabilistic statements directly — 'there's a 95% probability the effect is positive,' which is what people *think* a confidence interval says but isn't. Bayesian A/B testing and multi-armed bandits, which adaptively shift traffic toward the better variant, are increasingly common.

The thread through all of this, and the thing senior interviews really test, is *judgment under uncertainty* — not deriving the t-statistic, but knowing what a p-value means, why correlation isn't causation, when an experiment is invalid, and how to make a defensible decision when the data is noisy. Traps: misstating what a p-value or confidence interval means; ignoring multiple comparisons; peeking; and confusing statistical significance with practical significance — a result can be statistically significant and business-irrelevant, or vice versa."

---

## 3. Mathematical Foundation — The Five Pillars

### 3.1 Probability fundamentals

**Bayes' rule (the foundation):**
```
P(A|B) = P(B|A) P(A) / P(B)        — posterior = likelihood × prior / evidence
```
Rearranged definition of conditional probability; underlies Naive Bayes (Ch. 8), Bayesian inference, and most of ML's probabilistic reasoning.

**Key distributions and when they arise:**

| Distribution | Models | Mean / Variance |
|---|---|---|
| Bernoulli(p) | single binary outcome | p / p(1−p) |
| Binomial(n,p) | # successes in n trials | np / np(1−p) |
| Poisson(λ) | rare-event counts in fixed interval | λ / λ |
| Normal(μ,σ²) | CLT limit; continuous symmetric | μ / σ² |
| Exponential(λ) | waiting time between events (memoryless) | 1/λ / 1/λ² |
| Geometric(p) | # trials until first success | 1/p / (1−p)/p² |

**Expectation and variance (the workhorses):**
```
E[X] = Σ x·P(x)  or  ∫ x·f(x)dx
Var(X) = E[(X−μ)²] = E[X²] − E[X]²
Var(aX+b) = a²Var(X);  E[aX+b] = aE[X]+b
Var(X+Y) = Var(X)+Var(Y)+2Cov(X,Y)   (= sum if independent)
```

**Law of Large Numbers:** sample mean → true mean as n → ∞ (why more data helps).

**Central Limit Theorem (the most important result):**
```
For i.i.d. X₁..Xₙ with mean μ, variance σ²:
  (X̄ − μ)/(σ/√n)  →  N(0,1)   as n → ∞
```
The sample mean is approximately normal *regardless of the underlying distribution* ⟹ the basis for nearly every confidence interval and z/t-test. The √n is why halving the standard error requires 4× the data.

### 3.2 Estimation — MLE and MAP

**Maximum Likelihood Estimation (how models are fit):**
```
θ̂_MLE = argmax_θ  L(θ) = argmax_θ  Π P(xᵢ | θ)   = argmax_θ  Σ log P(xᵢ | θ)
```
- Linear regression: MLE under Gaussian noise = least squares (Ch. 1).
- Logistic regression: MLE of Bernoulli = minimizing log loss (Ch. 2).
- Properties: consistent (→ truth), asymptotically efficient (lowest variance), asymptotically normal (enables CIs from the Fisher information).

**Maximum A Posteriori (MLE + prior = regularization):**
```
θ̂_MAP = argmax_θ  P(θ|data) = argmax_θ  P(data|θ)P(θ)
       = argmax_θ  [ Σ log P(xᵢ|θ) + log P(θ) ]
```
- Gaussian prior P(θ) ⟹ **L2 regularization (ridge)**.
- Laplace prior P(θ) ⟹ **L1 regularization (lasso)**.
- **Regularization IS Bayesian MAP estimation** — the connection that unifies Ch. 1, 2, 12.

**Bias-variance of an estimator:**
```
MSE(θ̂) = Bias(θ̂)² + Var(θ̂)        Bias = E[θ̂] − θ
```
Unbiased isn't always best — a biased estimator (e.g., ridge) can have lower MSE (the bias-variance tradeoff, Ch. 1).

**Confidence interval (frequentist):**
```
95% CI for a mean:  X̄ ± 1.96 · (σ/√n)   (z, known σ)  or  X̄ ± t·(s/√n)  (t, estimated σ)
```
**Correct interpretation:** if you repeated the experiment many times, 95% of the constructed intervals would contain the true parameter. **NOT** "95% probability the parameter is in this interval" (that's the Bayesian credible interval). This distinction is a classic trap.

### 3.3 Hypothesis testing

**The framework:**
```
H₀ (null): no effect.   H₁ (alternative): there is an effect.
Test statistic → p-value = P(data this extreme or more | H₀ true)
Reject H₀ if p < α (significance level, typically 0.05).
```
**p-value — what it IS and ISN'T (the most-baited definition):**
- IS: the probability of observing data at least this extreme *assuming H₀ is true*.
- IS NOT: P(H₀ true | data), nor the probability the result is due to chance, nor the effect size.

**The two errors and power:**
```
Type I (α):  reject a TRUE null (false positive) — controlled by significance level
Type II (β): fail to reject a FALSE null (false negative)
Power = 1 − β = P(detect a real effect)   — typically target 80%
```
**Power analysis (the practical core):** before running a test, compute the sample size n needed to detect a minimum effect size (MDE) with desired power and α:
```
n ∝ σ² / (effect size)²        — smaller effects need quadratically more data
```

**Common tests (know which applies):**

| Test | Use |
|---|---|
| z-test / t-test | Compare means (z: known σ/large n; t: estimated σ/small n) |
| Two-proportion z-test | Compare conversion rates (A/B testing default) |
| Chi-square | Categorical association / goodness of fit |
| ANOVA | Compare 3+ group means |
| Mann-Whitney / Wilcoxon | Non-parametric (skewed data, no normality) |

### 3.4 A/B testing (the applied core)

**Why randomization gives causation:** random assignment makes treatment and control groups statistically identical in expectation on *all* covariates (observed and unobserved) ⟹ any outcome difference is *caused* by the treatment. This is how prediction becomes causation (the through-line from Ch. 2, 5, 19).

**The standard procedure:**
```
1. Hypothesis + primary metric (chosen BEFORE the test)
2. Power analysis → required sample size & duration
3. Randomize users to control/treatment
4. Run for the pre-committed duration (capture weekly cycles)
5. Analyze ONCE at the end (two-proportion z-test / t-test)
6. Decide on statistical AND practical significance
```

**The pitfalls (senior interviews probe these relentlessly):**

| Pitfall | Mechanism | Fix |
|---|---|---|
| **Peeking** | Checking repeatedly + stopping when significant ⟹ inflated Type I (many chances to cross α) | Fixed sample size, or sequential testing (alpha spending / always-valid p-values) |
| **Multiple comparisons** | Testing many metrics/variants ⟹ some significant by chance (test 20 at α=0.05 ⟹ ~64% chance of ≥1 false positive) | Bonferroni / Benjamini-Hochberg (FDR); pre-register primary metric |
| **Underpowered** | Too few users ⟹ miss real effects, call it "no difference" | Power analysis up front; don't conclude null from low power |
| **Network effects / interference** | Treating some users affects control (marketplace, social) ⟹ SUTVA violated | Cluster/geo randomization, switchback tests |
| **Novelty / primacy effects** | New thing draws attention, fades; or users resist change initially | Run long enough; segment by new vs returning; holdback analysis |
| **Sample ratio mismatch (SRM)** | Observed split ≠ intended (e.g., 48/52 vs 50/50) ⟹ assignment bug, invalidates test | Chi-square SRM check before analyzing — a non-negotiable gate |
| **Simpson's paradox** | Aggregate effect reverses within segments | Segment analysis; check for confounding mix shifts |
| **Twyman's law / too-good results** | A shockingly large effect is usually a bug | Investigate instrumentation before celebrating |

**Variance reduction (CUPED — the senior-level technique):**
```
Y_adjusted = Y − θ(X_pre − E[X_pre])    where X_pre = pre-experiment covariate, θ from regression
⟹ Var(Y_adjusted) = Var(Y)(1 − ρ²)      ρ = correlation(Y, X_pre)
```
Uses pre-experiment data (a covariate correlated with the outcome) to remove predictable variance ⟹ same power with less traffic / faster experiments. It's literally a regression adjustment (Ch. 1 connection). A favorite at experimentation-heavy companies.

### 3.5 Bayesian inference

**The Bayesian update:**
```
Posterior ∝ Likelihood × Prior        P(θ|data) ∝ P(data|θ) P(θ)
```
Maintain a full distribution over θ; update with data; make direct probability statements.

**Frequentist vs Bayesian (the philosophical contrast):**

| | Frequentist | Bayesian |
|---|---|---|
| Parameter | Fixed unknown | Random (has a distribution) |
| Output | p-value, CI, reject/don't | Posterior distribution, credible interval |
| "95% interval" means | 95% of such intervals cover θ | 95% probability θ is in this interval |
| A/B testing | z-test, fixed n, p<0.05 | P(B > A), expected loss, can peek |
| Prior | None | Required (subjective or weak) |

**Conjugate priors (the clean case):** Beta prior + Binomial likelihood ⟹ Beta posterior (the canonical Bayesian A/B test for conversion rates):
```
Prior: Beta(α, β).  Observe s successes, f failures.  Posterior: Beta(α+s, β+f).
⟹ P(variant B > variant A) computable directly from the two posteriors.
```

**Multi-armed bandits (adaptive experimentation):** instead of fixed A/B split, dynamically allocate traffic toward better-performing variants (Thompson sampling = sample from each arm's posterior, play the winner). Trades the clean inference of A/B for less regret (fewer users on the losing variant) — used for continuous optimization where you care about *cumulative* outcome, not a one-time decision.

---

## 4. Step-by-Step Numerical Examples

**4.1 A/B test (two-proportion z-test) — the bread-and-butter calculation.**
```
Control:  10,000 users, 500 conversions  → p_A = 0.0500
Treatment: 10,000 users, 560 conversions → p_B = 0.0560

Pooled rate: p = (500+560)/(20000) = 0.0530
Standard error: SE = √[ p(1−p)(1/n_A + 1/n_B) ]
              = √[ 0.0530·0.9470·(1/10000 + 1/10000) ]
              = √[ 0.0502·0.0002 ] = √0.00001004 = 0.003169
z = (p_B − p_A)/SE = (0.0560 − 0.0500)/0.003169 = 0.0060/0.003169 = 1.894
```
z = 1.894 < 1.96 (two-sided α=0.05) ⟹ **p ≈ 0.058, NOT significant** at 5%. The 12% relative lift *looks* meaningful but isn't statistically distinguishable from noise at this sample size — the gap between "looks good" and "is significant" that A/B discipline enforces. (Power note: detecting this effect reliably needs more users — the power-analysis lesson.)

**4.2 MLE for a coin (the canonical estimation derivation).**
```
Observe 7 heads in 10 flips. Likelihood: L(p) = p⁷(1−p)³
log L = 7 log p + 3 log(1−p)
d/dp = 7/p − 3/(1−p) = 0  ⟹  7(1−p) = 3p  ⟹  7 = 10p  ⟹  p̂ = 0.7
```
MLE = observed frequency — intuitive, and the foundation of how every probabilistic model is fit.

**4.3 Bayes' rule (the medical-test classic — tests base-rate reasoning).**
```
Disease prevalence P(D) = 0.001. Test: P(+|D)=0.99 (sensitivity), P(+|¬D)=0.05 (false positive).
P(D|+) = P(+|D)P(D) / [P(+|D)P(D) + P(+|¬D)P(¬D)]
       = (0.99·0.001) / (0.99·0.001 + 0.05·0.999)
       = 0.00099 / (0.00099 + 0.04995) = 0.00099/0.05094 = 0.0194
```
**Only ~1.9% chance of disease despite a positive test** — because the disease is rare, false positives swamp true positives. The base-rate fallacy, and exactly why a "99% accurate" test on a rare condition is nearly useless — a standard Amazon/quant interview question. Land the intuition, not just the arithmetic.

**4.4 CLT in action (why confidence intervals work).**
```
Sample: n=100, x̄=50, s=10. 95% CI for the mean:
  SE = s/√n = 10/10 = 1.0
  CI = 50 ± 1.96·1.0 = [48.04, 52.04]
```
Interpretation: 95% of intervals built this way contain the true mean — *not* "95% chance μ is in [48,52]."

---

## 5. "Hyperparameters" — The Knobs of Statistical Decisions

Statistics has decision parameters rather than model hyperparameters; each is a judgment interviewers probe:

| Knob | What it controls | Set higher → | Set lower → | Interview probe |
|---|---|---|---|---|
| α (significance) | Type I error rate | More false positives, easier to "win" | Fewer false positives, harder to detect | "Why 0.05?" → convention, not law; lower for high-stakes/multiple tests |
| Power (1−β) | Chance of detecting real effects | Need more samples | Risk missing real effects | Target 80%; set via power analysis |
| MDE (min detectable effect) | Smallest effect you'll reliably catch | Smaller MDE ⟹ quadratically more samples | Larger MDE ⟹ fewer samples, miss small effects | "Test underpowered — options?" → bigger MDE, more traffic, variance reduction |
| Sample size n | Precision (SE ∝ 1/√n) | Tighter CIs, more power, more cost | Noisier, cheaper | The √n: 4× data halves the SE |
| Prior (Bayesian) | Influence of prior belief | Strong prior dominates small data | Weak prior ≈ frequentist | "How choose a prior?" → weak/uninformative default; domain-justified if used |
| Multiple-testing correction | Family-wise vs per-test error | Bonferroni (conservative) | BH/FDR (less conservative) | "20 metrics — how correct?" → pre-register primary; FDR for secondaries |
| One- vs two-sided test | Directional vs any difference | Two-sided (default, conservative) | One-sided (more power, assumes direction) | "When one-sided?" → only with strong a-priori directional justification |

---

## 6. Production Perspective — Statistics in ML Systems

| Aspect | Detail |
|---|---|
| Experimentation platform | A/B testing infra is core ML infra: assignment (consistent hashing), metric pipelines, power/sizing tools, SRM checks, sequential-test support, CUPED variance reduction |
| Offline → online gap | Offline metrics (NDCG, AUC) are *proxies*; the experiment is the *truth*. Every model in this book ultimately proves itself in an A/B test (the recurring "offline ≠ online" theme) |
| Sequential testing | Production reality: stakeholders *will* peek. Use always-valid p-values / alpha-spending (mSPRT) so early stopping doesn't inflate Type I — the engineered fix for human impatience |
| Variance reduction | CUPED and stratification ship as platform features — they cut required traffic, enabling more/faster experiments |
| Bandits | For continuous optimization (ranking weights, content selection), bandits replace fixed A/B to minimize regret; Thompson sampling is the standard |
| Causal inference | When you *can't* randomize (can't withhold a feature, network effects): difference-in-differences, instrumental variables, synthetic control, propensity matching — the toolkit for observational causal claims |
| Monitoring | Metric distributions, SRM alarms, novelty-effect decay curves, segment heterogeneity, multiple-comparison-corrected dashboards |
| The discipline | Pre-registration (metric + MDE + duration *before* the test), one primary metric, SRM gate before analysis — process that prevents the pitfalls |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What is a p-value?** The probability of observing data at least as extreme as yours *if the null hypothesis were true*. NOT the probability the null is true, nor the probability of chance.
2. **Type I vs Type II error?** Type I = false positive (reject a true null), controlled by α. Type II = false negative (miss a real effect), related to power = 1−β.
3. **What is statistical power and why do power analysis?** Power = P(detect a real effect of a given size). Power analysis sizes the experiment up front so you can actually detect the minimum effect you care about — avoids the "no effect" conclusion that's really "no power."
4. **State the Central Limit Theorem and why it matters.** The sample mean of many i.i.d. variables is ~normal regardless of the underlying distribution ⟹ basis for confidence intervals and z/t-tests on almost any metric.
5. **What does a 95% confidence interval mean?** If you repeated the experiment many times, ~95% of the constructed intervals would contain the true parameter. NOT "95% probability the parameter is in this interval."
6. **What is MLE?** Choose the parameter that maximizes the probability of the observed data. Linear regression = MLE under Gaussian noise; logistic = MLE of Bernoulli.
7. **How does regularization relate to Bayesian estimation?** MAP estimation = MLE + prior; Gaussian prior = L2 (ridge), Laplace prior = L1 (lasso). Regularization *is* a prior.
8. **Why randomize in an A/B test?** Randomization makes groups comparable on all covariates (observed and unobserved) in expectation ⟹ outcome differences are *caused* by the treatment, not confounders.
9. **What is the peeking problem?** Repeatedly checking results and stopping when significant inflates Type I error (many chances to cross α). Fix: fixed sample size or sequential testing.
10. **What is the multiple comparisons problem?** Testing many hypotheses ⟹ some significant by chance (20 tests at α=0.05 ⟹ ~64% chance of a false positive). Fix: Bonferroni/FDR, pre-register the primary metric.
11. **Bayes' rule — state and give a use.** P(A|B)=P(B|A)P(A)/P(B). Used in Naive Bayes, Bayesian inference, and base-rate reasoning (the medical-test problem).
12. **When use a t-test vs z-test?** t when σ is estimated from the sample / small n; z when σ is known or n is large (they converge as n grows).
13. **Statistical vs practical significance?** Statistical = unlikely under the null (p<α). Practical = effect large enough to matter for the business. Large n can make trivial effects significant; small n can leave real effects insignificant — report both.
14. **What is sample ratio mismatch and why check it?** Observed group split ≠ intended (e.g., 48/52 vs 50/50) signals an assignment/logging bug that invalidates the test. Chi-square SRM check is a gate *before* analyzing.
15. **Frequentist vs Bayesian in one line each?** Frequentist: parameter fixed, data random, output p-values/CIs. Bayesian: parameter random, update a posterior with priors, output direct probabilities (credible intervals).

### Advanced (15)

1. **Derive the MLE for a Bernoulli/binomial.** L(p)=p^s(1−p)^f; log L = s log p + f log(1−p); set derivative s/p − f/(1−p)=0 ⟹ p̂ = s/(s+f) — the observed frequency.
2. **Prove L2 regularization = Gaussian prior (MAP).** MAP maximizes log P(data|θ)+log P(θ); a Gaussian prior θ~N(0,τ²) contributes −‖θ‖²/2τ² ⟹ adding a quadratic penalty = ridge, with λ=σ²/τ². Laplace prior ⟹ |θ| penalty = lasso.
3. **Why is the sample variance divided by (n−1)?** Bessel's correction: dividing by n underestimates variance because deviations are taken from the *sample* mean (which minimizes them); n−1 corrects the bias, making E[s²]=σ². One degree of freedom is "used up" estimating the mean.
4. **Derive the standard error of a proportion and the two-proportion z-test.** Var(p̂)=p(1−p)/n (Bernoulli variance / n); for two groups, pool the rate under H₀, SE=√[p(1−p)(1/n_A+1/n_B)], z=(p_B−p_A)/SE.
5. **What does the CLT *not* guarantee?** It's asymptotic (needs "enough" n; heavy tails/extreme skew need larger n), assumes finite variance (fails for Cauchy-like distributions), and i.i.d.; it says nothing about the *rate* of convergence (Berry-Esseen bounds that).
6. **Explain CUPED and why it works.** Adjust the outcome by a pre-experiment covariate: Y'=Y−θ(X_pre−E[X_pre]) with θ from regression. Since randomization makes X_pre balanced, the adjustment is unbiased; it removes the variance Y shares with X_pre ⟹ Var reduced by factor (1−ρ²) ⟹ more power per user. It's regression adjustment / control-variate variance reduction.
7. **Bonferroni vs Benjamini-Hochberg — what each controls.** Bonferroni controls family-wise error rate (P(any false positive); test at α/m) — conservative. BH controls the false discovery rate (expected proportion of false positives among rejections) — less conservative, higher power, the standard for many secondary metrics.
8. **What is a confidence interval's relationship to a hypothesis test?** A 95% CI contains exactly the null values that wouldn't be rejected at α=0.05 (two-sided); if the CI excludes the null value, the test is significant. They're dual.
9. **Derive the Beta-Binomial conjugacy.** Prior Beta(α,β) ∝ p^{α−1}(1−p)^{β−1}; Binomial likelihood ∝ p^s(1−p)^f; posterior ∝ p^{α+s−1}(1−p)^{β+f−1} = Beta(α+s, β+f). Conjugacy = posterior in the same family ⟹ closed-form Bayesian updating.
10. **What is the multiple-testing issue in feature selection / model comparison, beyond A/B?** Testing many features for significance, or comparing many models, manufactures false positives (a feature/model looks great by chance). Fixes: cross-validation, nested CV, FDR control, holdout sets — the same discipline as A/B multiple comparisons. (Connects to Ch. 1's "p-values after selection are invalid.")
11. **Simpson's paradox — mechanism and a real example.** An aggregate trend reverses within subgroups due to a confounding mix shift (e.g., a treatment looks worse overall but better in every segment because it was assigned more to a harder segment). Always segment; aggregate effects can mislead. (Connects to omitted-variable bias, Ch. 1.)
12. **Sequential testing / always-valid p-values — how do they permit peeking?** Methods like mSPRT or alpha-spending (O'Brien-Fleming) allocate the Type I budget across looks so the *cumulative* false-positive rate stays ≤ α regardless of when you stop. They trade some power for the freedom to monitor continuously — the principled fix for peeking.
13. **Bootstrap — what is it and when do you use it?** Resample the data with replacement many times, recompute the statistic, use the distribution of those estimates for SEs/CIs. Use when analytic SEs are unavailable or assumptions (normality) are dubious — distribution-free uncertainty. (Connects to RF's OOB/bagging intuition, Ch. 4.)
14. **Delta method — what's it for?** Approximates the variance of a *function* of an estimator via a first-order Taylor expansion: Var(g(θ̂)) ≈ g'(θ̂)²Var(θ̂). Used for ratio metrics (e.g., variance of a click-through *rate* that's a ratio of correlated counts) — common in A/B analysis of ratio metrics.
15. **What is the difference between a credible interval and a confidence interval, precisely?** Credible (Bayesian): given the data and prior, 95% posterior probability θ is in the interval — a direct probability statement. Confidence (frequentist): 95% of such intervals over repeated sampling cover θ — a statement about the procedure, not this interval. People *want* the credible interpretation; the CI doesn't provide it.

### Staff-Level (10)

1. **Your A/B test shows +2% conversion, p=0.04, but the PM wants to ship to 100% immediately. What's your guidance?** Check before celebrating: SRM (was randomization clean?), novelty effect (is +2% durable or a launch spike? — segment new vs returning, look at the trend), practical significance and CI width (is the lower CI bound still worth it?), guardrail metrics (did anything else regress?), and whether p=0.04 survived any peeking/multiple-metric inflation. If primary metric was pre-registered, test ran full duration, SRM clean, guardrails fine, and CI excludes trivial effects — ship, ideally with a ramped rollout + holdback to monitor durability. The staff move is the *checklist of invalidators*, not the p-value.
2. **A model improves offline AUC by 3% but the A/B shows no engagement change. Reconcile.** Offline metric is a proxy that didn't translate: possible causes — the metric isn't what drives engagement (AUC ↑ but ranking of the *top* items unchanged — Ch. 19's NDCG point), position/selection bias in offline eval (off-policy — Ch. 19), the improvement concentrated in segments that don't matter, or the change is real but too small to detect at current power. Diagnose with segment analysis, off-policy correction, and a power check. The experiment is truth; the offline metric was a flawed proxy — the recurring theme of the whole handbook.
3. **You ran 15 metrics; one secondary is significant at p=0.03. How do you treat it?** With suspicion: 15 tests at α=0.05 ⟹ ~54% chance of ≥1 false positive, so an isolated p=0.03 secondary is likely noise. Apply multiple-testing correction (BH/FDR), check if it's mechanistically plausible and consistent across segments, and treat it as *hypothesis-generating, not conclusive* — pre-register it as a primary metric in a confirmatory follow-up. Never ship on a fished secondary. (The exact discipline McKinsey QB / Amazon probe.)
4. **Design an experimentation framework for a two-sided marketplace (riders/drivers, eaters/restaurants).** Standard user-randomized A/B breaks: treating some users changes supply/prices for everyone (interference, SUTVA violated). Use cluster/geo randomization (randomize cities/regions, not users), switchback experiments (alternate treatment on/off over time windows for the whole market), or two-sided designs; analyze with cluster-robust standard errors; accept lower power as the cost of validity. Recognizing that *naive A/B is invalid here* is the staff-level insight. *Direct Swiggy/Zomato/Uber relevance.*
5. **A stakeholder insists on peeking daily and stopping when significant. How do you handle it organizationally and technically?** Don't fight human nature — engineer for it: deploy sequential testing / always-valid p-values (mSPRT, alpha-spending) so continuous monitoring doesn't inflate Type I; build the dashboard to show *valid* sequential bounds, not naive daily p-values; educate that fixed-horizon p-values are invalid under peeking. The staff move is providing a *technically correct way to do what they want* rather than a "no."
6. **When can you NOT run an A/B test, and what do you do instead?** Can't randomize: ethical/legal constraints, can't withhold a launched feature, network effects too strong, or measuring a one-time/global change. Use quasi-experimental / observational causal inference: difference-in-differences (pre/post with a control region), instrumental variables, regression discontinuity, synthetic control, or propensity-score matching — each with assumptions you must defend. Be explicit that these are *weaker* than randomization and state the identifying assumption. (Connects to Ch. 1's endogeneity/causation caveats.)
7. **Explain to leadership why a "statistically significant" result might not be worth shipping, and vice versa.** Significance ≠ importance: with huge n, a 0.01% lift is significant but worthless (cost of shipping/maintenance exceeds value); with small n, a real 5% lift can be insignificant (underpowered — don't conclude "no effect"). Decisions need the *effect size and its CI* against the *business cost/value*, not the p-value alone. Reporting practical significance + uncertainty, not just p<0.05, is the maturity marker.
8. **Bayesian vs frequentist A/B testing — when would you choose Bayesian for a real product?** Bayesian (P(B>A), expected loss) when: stakeholders want intuitive "probability B is better" statements, you need to make decisions under continuous monitoring (the posterior is valid at any point — no peeking penalty), you want to incorporate prior experiments, or you're running bandits for continuous optimization. Frequentist when: regulatory/standardized reporting, you want guarantees without prior assumptions, or organizational convention. It's a judgment about decision context, not a correctness contest.
9. **Design a bandit system for ranking-weight or content optimization, and name the tradeoff vs A/B.** Thompson sampling: maintain a posterior per arm, sample, serve the sampled-best, update — automatically shifting traffic to winners and minimizing regret (fewer users on losers). Tradeoff: you optimize *cumulative reward* but sacrifice the clean, unbiased single-decision inference of a fixed A/B (the adaptive allocation biases naive post-hoc analysis). Use bandits for ongoing optimization where regret matters; A/B for a one-time ship/no-ship decision needing a clean readout. Add: nonstationarity handling (discounting) for drifting rewards.
10. **A causal claim from observational data lands on your desk ("users who use feature X retain better — let's push X"). Walk through your scrutiny.** Classic selection bias: users who *choose* X differ from those who don't (engaged users self-select into features). Correlation ≠ causation — the retention may cause X-usage, or a confounder (overall engagement) causes both. Scrutiny: identify plausible confounders, attempt adjustment (propensity matching, controlling for pre-period engagement) while stating it can't capture unobserved confounders, and — the real answer — recommend a randomized experiment (randomly encourage X usage) before any push decision. This is the prediction-vs-causation discipline that threads through Ch. 1, 2, 5, 19, 20 — and exactly the McKinsey QuantumBlack case archetype.

---

## 8. Comparison Section

**Frequentist vs Bayesian (the central philosophical comparison):**

| | Frequentist | Bayesian |
|---|---|---|
| Parameter | Fixed unknown | Random variable (distribution) |
| Data | Random (repeatable) | Fixed (observed) |
| Output | p-value, CI, reject/don't | Posterior, credible interval, P(hypothesis) |
| Prior | None | Required |
| Peeking | Invalid (inflates Type I) without correction | Posterior valid anytime |
| A/B form | z-test, fixed n | P(B>A), expected loss |
| Strength | No prior assumptions, standardized | Direct probability statements, incorporates priors, sequential-friendly |

**Confidence interval vs Credible interval:** procedure-level coverage (frequentist) vs direct posterior probability (Bayesian). People want the credible interpretation; the CI doesn't give it.

**A/B test vs Bandit:** A/B = fixed allocation, clean one-time inference, optimizes for a *decision*. Bandit = adaptive allocation, minimizes *regret*, optimizes for *cumulative reward*; trades clean inference for efficiency.

**Randomized experiment vs observational causal inference:** randomization gives causation by construction (balances unobservables); observational methods (DiD, IV, matching, synthetic control) approximate it under assumptions that can't fully rule out unobserved confounders — strictly weaker, used when randomization is impossible.

**MLE vs MAP vs full Bayesian:** point estimate (no prior) vs point estimate (with prior = regularization) vs full posterior distribution. Increasing use of prior information and uncertainty quantification, increasing cost.

**Statistical significance vs practical significance:** "unlikely under the null" vs "big enough to matter." Independent axes — both required for a decision.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Misstating the p-value ("probability the null is true" / "probability of chance") — the #1 baited error.
- Misstating the confidence interval ("95% probability θ is in here" — that's a credible interval).
- Not knowing why /(n−1) in sample variance (Bessel's correction).
- Confusing Type I and Type II, or not connecting Type II to power.
- Not knowing regularization = Bayesian MAP (Gaussian/Laplace priors).

**Production / A/B mistakes:**
- Peeking and stopping when significant (inflated false positives).
- Ignoring multiple comparisons (fishing across metrics/segments).
- Concluding "no effect" from an underpowered test.
- Skipping the SRM check before analyzing.
- Ignoring network effects in marketplaces (naive user-randomization invalid).
- Confusing statistical with practical significance.

**Reasoning mistakes:**
- Correlation → causation from observational data (selection bias, confounders).
- Base-rate neglect (the medical-test fallacy).
- Simpson's paradox (trusting aggregates without segmenting).
- Celebrating a too-good result instead of suspecting a bug (Twyman's law).
- Treating offline metrics as truth when the experiment is the truth.

---

## 10. Real Industry Use Cases

- **Amazon** — pervasive experimentation culture; the Breadth & Depth round leans hard on probability (Bayes/base-rate problems), A/B design, and the statistical-vs-practical-significance judgment; "Weblab" experimentation at massive scale.
- **Google / Meta** — industrial experimentation platforms with sequential testing, CUPED variance reduction, multiple-testing control, and interference-aware designs; A/B is how every ranking/feed change ships.
- **Netflix** — sophisticated experimentation (interleaving for ranking — Ch. 19, quasi-experiments, long-term holdouts); public writing on variance reduction and novelty effects.
- **Microsoft** — pioneered much of modern online experimentation methodology (Kohavi et al. — the canonical A/B literature, SRM, Twyman's law, trustworthy experiments).
- **Uber / Lyft / DoorDash** — marketplace experimentation: switchback and geo-randomized designs to handle interference (the §7-S4 problem) — a domain where naive A/B is invalid.
- **Swiggy/Zomato** — A/B testing every ranking/recommendation/pricing change; marketplace interference (treating eaters affects restaurant load) forces geo/switchback designs; CUPED-style variance reduction to run faster experiments. *Naveen: your ranking work (9% CTR uplift, ₹40L/month) was ultimately validated by experiments — being able to design the experiment, handle marketplace interference, and defend the result against peeking/multiple-comparisons/novelty is the statistical backbone of every model story in your résumé.*
- **McKinsey QuantumBlack** — causal inference and experimental design are central to client work; expect deep probing on correlation-vs-causation, observational causal methods (DiD, matching), and translating statistical results into defensible business decisions — the §7-S10 archetype.
- **Games24x7** — A/B testing of personalization/retention features; the prediction-vs-causation discipline (does a feature *cause* retention, or do retained users self-select?) is central to responsible-gaming and growth decisions.

---

## 11. Coding From Scratch (NumPy only)

The A/B testing toolkit + a Bayesian A/B test — the calculations interviewers ask you to implement.

```python
import numpy as np
from scipy import stats   # for the normal/beta CDFs; the LOGIC is what's tested

def two_proportion_ztest(conv_a, n_a, conv_b, n_b):
    """The standard A/B significance test for conversion rates."""
    p_a, p_b = conv_a / n_a, conv_b / n_b
    p_pool = (conv_a + conv_b) / (n_a + n_b)        # pooled rate under H0
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_a + 1/n_b))   # SE of the difference
    z = (p_b - p_a) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))      # two-sided
    # 95% CI for the difference (unpooled SE for the interval).
    se_diff = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
    ci = ((p_b - p_a) - 1.96*se_diff, (p_b - p_a) + 1.96*se_diff)
    return {"lift": p_b - p_a, "z": z, "p_value": p_value, "ci_95": ci}

def required_sample_size(baseline_rate, mde, alpha=0.05, power=0.80):
    """Power analysis: users PER ARM to detect a minimum effect (mde) reliably.
       This is what you compute BEFORE running, to avoid an underpowered test."""
    z_alpha = stats.norm.ppf(1 - alpha/2)           # two-sided critical value
    z_beta = stats.norm.ppf(power)                  # power quantile
    p1, p2 = baseline_rate, baseline_rate + mde
    p_bar = (p1 + p2) / 2
    # Standard two-proportion power formula.
    n = ((z_alpha*np.sqrt(2*p_bar*(1-p_bar)) +
          z_beta*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2) / (mde**2)
    return int(np.ceil(n))

def cuped_adjust(y, x_pre):
    """Variance reduction: remove the variance y shares with a pre-experiment
       covariate x_pre. Returns adjusted outcomes with lower variance, same mean."""
    theta = np.cov(y, x_pre)[0, 1] / np.var(x_pre)  # regression coefficient
    y_adj = y - theta * (x_pre - x_pre.mean())      # the CUPED adjustment
    # Variance reduced by factor (1 - corr(y, x_pre)^2).
    return y_adj

def bayesian_ab_test(conv_a, n_a, conv_b, n_b, n_samples=100_000, seed=0):
    """Bayesian A/B: Beta-Binomial conjugacy. Returns P(B > A) directly —
       the intuitive statement people THINK a p-value gives."""
    rng = np.random.default_rng(seed)
    # Posterior = Beta(1 + successes, 1 + failures)  [uniform Beta(1,1) prior].
    post_a = rng.beta(1 + conv_a, 1 + (n_a - conv_a), n_samples)
    post_b = rng.beta(1 + conv_b, 1 + (n_b - conv_b), n_samples)
    prob_b_better = (post_b > post_a).mean()        # Monte Carlo over posteriors
    expected_lift = (post_b - post_a).mean()
    return {"prob_b_better": prob_b_better, "expected_lift": expected_lift}

# --- Demonstrate the §4.1 example ---
result = two_proportion_ztest(500, 10000, 560, 10000)
# → lift 0.006, z ≈ 1.89, p ≈ 0.058 (NOT significant at 5%)
```

Narration points that earn senior credit:
- **`two_proportion_ztest`** — "pooled rate under the null for the test statistic, but unpooled SE for the CI — a subtlety people miss. And I report the CI, not just the p-value, because the *effect size and its uncertainty* drive the decision, not p<0.05."
- **`required_sample_size`** — "this is the calculation you run *before* the test, not after. The MDE is in the denominator squared — detecting an effect half as small needs 4× the users. Underpowered tests are how teams falsely conclude 'no effect.'"
- **`cuped_adjust`** — "CUPED is literally a regression adjustment (Ch. 1): subtract the part of the outcome predictable from a pre-experiment covariate. Variance drops by (1−ρ²), so a covariate correlated 0.7 with the outcome cuts variance ~50% ⟹ half the traffic for the same power."
- **`bayesian_ab_test`** — "Beta-Binomial conjugacy gives the posterior in closed form; I Monte-Carlo P(B>A), which is the *direct* probability statement stakeholders actually want — and it's valid under continuous monitoring, unlike a naive frequentist p-value."
- **The §4.1 result** — "12% relative lift, but z=1.89, p≈0.058 — *not* significant. The gap between 'looks good' and 'is significant' is exactly what this discipline enforces."
- **Honest note:** "I used scipy for the normal/Beta CDFs; the *logic* — pooled SE, power formula, conjugate update — is the part that matters and that I can derive."

---

## 12. ML System Design Perspective — Where Statistics Decides

**Choose rigorous statistical methods when:** proving a model/feature works (A/B testing — the truth that offline metrics only proxy); quantifying uncertainty for decisions (CIs, Bayesian posteriors, conformal prediction); making causal claims (randomized experiments, or observational causal inference when you can't randomize); sizing experiments (power analysis); guarding against false discoveries (multiple-testing control).

**Common failure to avoid:** treating offline ML metrics as the deliverable. Every model in this handbook — linear, boosted, ranking, embedding — ultimately justifies itself through an experiment; the statistics of *that* experiment determine whether the model ships. The model is the easy part; the trustworthy readout is the hard part.

**Data/design requirements:** randomization (or a defensible identifying assumption for observational claims); pre-registered primary metric + MDE + duration; SRM-clean assignment; sufficient power; interference-aware design in marketplaces; variance reduction to make experiments affordable.

**Latency/throughput:** experimentation is offline/batch analysis, but the *platform* (assignment, metric pipelines, sequential-test computation) is production infra; bandits operate online and adaptively.

**Scale limits:** the constraints are statistical and organizational — power (traffic), interference (marketplace structure), the multiple-comparisons and peeking pressures of a fast-moving org — not compute. Trustworthy experimentation at scale is an engineering *and* a discipline problem.

---

## 13. Resume Discussion Angle — Statistics Validates Every Story

**This chapter is the backbone of every model story in your résumé.** Your 9% CTR uplift, ₹40L/month ads revenue, 63% ticket deflection — every one of those numbers was (or should have been) established by an experiment, and the senior interviewer's depth-probe is: *how did you prove it?* The strong answer for any model story: "we validated it with an A/B test — pre-registered primary metric, powered to detect the MDE we cared about, SRM-checked, run long enough to rule out novelty effects, analyzed once." Being able to design and defend that experiment is what makes the model number credible rather than a claim.

**Marketplace interference (your domain — high-value):** for Swiggy/Zomato/Games24x7, volunteer that naive user-randomized A/B is *invalid* when treating some users affects others (eater treatment changes restaurant load; supply/demand spillovers) — and that you'd use geo or switchback randomization. This is the §7-S4 answer, and raising it unprompted signals you understand experimentation at the level these companies actually need.

**The prediction-vs-causation thread (McKinsey QuantumBlack especially):** this is the seam this whole handbook keeps returning to — churn *propensity* vs *uplift* (Ch. 2, 5), the discount-feedback-loop (Ch. 5), position bias in ranking (Ch. 19), correlation in baskets (Ch. 20). At staff/AS level, the recurring test is recognizing when a stakeholder is about to act causally on correlational evidence (§7-S10) and steering them to an experiment. QuantumBlack case rounds *are* this skill; have the "identify confounders → attempt adjustment with caveats → recommend an experiment" structure ready.

**The probability fundamentals (Amazon Breadth & Depth):** expect a Bayes/base-rate problem (the medical-test §4.3), an MLE derivation, or a "what does a p-value mean" trap. The wins are: state the p-value and CI interpretations *correctly* (the #1 baited errors), derive the Bernoulli MLE fluently, and connect regularization to Bayesian priors (tying back to Ch. 1, 2). These are quick to state correctly and quick to fumble — drill the exact phrasings.

**The unifying flourish (Staff/AS-level):** the sentence that demonstrates you see the whole field as one fabric — "MLE is how I fit the model, the CLT is why I can put error bars on it, hypothesis testing is how I decide if an effect is real, A/B testing is how I prove it causally, and Bayesian inference is the alternative when I want direct probability statements or to update with priors — and regularization is just MAP estimation, so even my model's penalty terms are a statistical prior." Delivering statistics not as a separate topic but as the connective tissue under every chapter of this handbook is precisely what separates a Staff/Applied Scientist hire from a strong senior engineer — and it's the note to end the handbook on.

---
*Previous: Association Rule Mining ← | **End of Handbook** — see 00-README for the full map.*

# Chapter 8: Naive Bayes
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

Naive Bayes is a generative classifier: it models P(y) and P(x|y), then uses Bayes' rule to get P(y|x). The "naive" part is assuming features are conditionally independent given the class, which collapses an intractable joint distribution into a product of one-dimensional ones — each estimated by simple counting or a univariate Gaussian. Training is a single counting pass (the fastest training in ML), it works with shockingly little data, and despite the wrong independence assumption it classifies well because classification only needs the *argmax* of the posterior to be right, not the probabilities. Interviews use it to test Bayes' rule fluency, the generative-vs-discriminative distinction, smoothing, and probability hygiene (log-space, calibration).

---

## 2. Interview Articulation (3–4 Minute Answer)

"Naive Bayes flips the modeling question. A discriminative model like logistic regression asks directly: given these features, what's the probability of each class? Naive Bayes instead asks: what does each class's data *look like* — what's the distribution of features within spam, within ham — and then uses Bayes' rule to invert: given what this email looks like, which class most plausibly generated it? Posterior proportional to prior times likelihood: P(y|x) ∝ P(y) · P(x|y).

The problem is P(x|y) — the joint distribution of all features given the class. With d binary features that's 2^d parameters per class; hopeless. The naive assumption cuts the knot: assume features are **conditionally independent given the class**. Then the joint factorizes into a product of per-feature distributions, P(x|y) = Π P(xⱼ|y), and each factor is a one-dimensional estimate you get by counting. The assumption is obviously false — in spam, 'free' and 'offer' co-occur far beyond independence — but here's the key insight interviewers want: classification only needs the **argmax** to be correct. The independence violation double-counts correlated evidence, which makes the posterior probabilities overconfident — pushed toward 0 and 1 — but often leaves the *ranking* of classes intact. So Naive Bayes is frequently a good classifier and almost always a badly calibrated probability estimator. If downstream consumes probabilities, recalibrate or use logistic regression.

The variants are just choices of the per-feature distribution. **Multinomial** for counts — the classic text model: P(word|class) estimated from word frequencies in each class's documents. **Bernoulli** for binary presence/absence — and a subtle difference: it explicitly models *absent* words too, which matters for short texts. **Gaussian** for continuous features: per-class mean and variance per feature, plug into the normal density. You can mix them across features.

Two pieces of craftsmanship are mandatory. First, **Laplace smoothing**: if a word never appeared in training spam, its likelihood is zero, and one zero annihilates the entire product — a single unseen word would veto the class. Add-one (or add-alpha) smoothing fixes it, and it has a clean Bayesian interpretation: a Dirichlet prior on the word distribution. Second, **log-space computation**: a product of hundreds of small probabilities underflows float64; you sum log-probabilities instead. Mentioning underflow unprompted is a cheap, reliable experience signal.

Training is one pass of counting — the cheapest training in machine learning — and it updates online trivially: new labeled example, increment counts. It's also remarkably data-efficient: with strong priors from the class structure, it does sensible things with tens of examples per class. The classic theory result is Ng & Jordan: generative Naive Bayes reaches its (higher) asymptotic error much faster than discriminative logistic regression reaches its (lower) one — so NB wins small-data regimes, LR wins as data grows.

Where it lives in industry: spam and abuse filtering — the canonical application, and still a real component in layered filters because it's fast, online-updatable, and adversarially cheap to retrain; quick text baselines; high-throughput pre-filters in front of expensive models; anywhere you need a probabilistic classifier trained in seconds on a stream.

When not to use it: correlated features feeding probability-consuming systems; continuous features that are far from per-class Gaussian; problems where feature interactions carry the signal — NB literally cannot see interactions, it's an additive-evidence model in log space.

Traps: 'naive Bayes assumes features are independent' — wrong, it's **conditionally** independent given the class, and the difference is checkable; forgetting smoothing; presenting its probabilities as trustworthy; and missing that NB's decision boundary is actually *linear* in log-feature space for the multinomial/Bernoulli variants — which is why it competes with logistic regression on text: same boundary family, different fitting philosophy."

---

## 3. Mathematical Foundation

**Bayes' rule and the decision rule:**
```
P(y=c | x) = P(y=c) · P(x | y=c) / P(x)

ŷ = argmax_c  P(y=c) · Π_j P(x_j | y=c)        (P(x) constant across c — drop it)
ŷ = argmax_c  [ log P(y=c) + Σ_j log P(x_j | y=c) ]      (log-space, always)
```

**The naive (conditional independence) assumption:**
```
P(x₁,…,x_d | y) = Π_j P(x_j | y)
```
Per class: d one-dimensional estimation problems instead of one d-dimensional one. Parameter count (binary features): 2^d − 1 per class → d per class. *This collapse is the entire model.*

**Variant likelihoods:**

| Variant | P(xⱼ|y=c) | Estimation | Use |
|---|---|---|---|
| Multinomial | word w with prob θ_{c,w}; doc likelihood ∝ Π θ^{count} | θ̂ = (N_{c,w} + α)/(N_c + α·V) | text counts, TF-IDF-ish |
| Bernoulli | θ^{x}(1−θ)^{1−x} — models presence AND absence | θ̂ = (docs in c containing w + α)/(docs in c + 2α) | short text, binary flags |
| Gaussian | N(x; μ_{c,j}, σ²_{c,j}) | per-class per-feature mean/var | continuous features |

**Laplace / Lidstone smoothing:** add pseudo-count α (α=1 Laplace) to every count. Why mandatory: one zero-probability factor zeroes the whole product (the "unseen-word veto"). Bayesian view: MAP under a symmetric Dirichlet(α+1) prior — smoothing = prior belief that no word is impossible.

**Prior estimation:** P̂(y=c) = n_c/n (or set from known base rates — useful when training sampling ≠ deployment prevalence; adjusting the prior is NB's one-line domain-shift correction, a genuinely elegant property: likelihoods stay, prior swaps).

**Why violated independence often still classifies well:** correlated features double-count evidence ⟹ log-posterior margins inflate ⟹ probabilities saturate toward 0/1 — but argmax is invariant to monotone inflation that doesn't cross class scores. Formal footing: zero-one loss only penalizes argmax errors (Domingos & Pazzani). Corollary to say aloud: **good classifier, bad probability estimator.**

**NB's decision boundary is linear (multinomial/Bernoulli):**
```
log P(y=1|x) − log P(y=0|x) = log[P(1)/P(0)] + Σ_j x_j · log[θ_{1j}/θ_{0j}]  (multinomial)
```
— linear in the count vector x ⟹ same hypothesis class as logistic regression on text; the difference is *how weights are fit* (counting under independence vs discriminative MLE). Gaussian NB with **shared** covariance also yields a linear (logistic-form) posterior; class-specific variances make it quadratic (QDA-like).

**Generative vs discriminative (Ng & Jordan 2002):** NB converges to its asymptotic error in O(log d) samples vs O(d) for LR; NB's asymptote is higher (bias from the independence assumption). Crossover: NB wins small-n, LR wins large-n. The single most-quoted theory result in NB interviews.

---

## 4. Step-by-Step Numerical Example

Spam filter, multinomial NB, tiny corpus. Vocabulary V = {free, win, meeting, project} (|V| = 4), α = 1.

Training counts:

| | free | win | meeting | project | total words | docs |
|---|---|---|---|---|---|---|
| spam | 3 | 2 | 0 | 1 | 6 | 3 |
| ham | 0 | 1 | 3 | 2 | 6 | 3 |

Priors: P(spam) = P(ham) = 3/6 = 0.5.

Smoothed word likelihoods θ = (count + 1)/(total + |V|) = (count + 1)/10:
```
            free   win   meeting  project
spam:       4/10   3/10   1/10     2/10
ham:        1/10   2/10   4/10     3/10
```
Note: "meeting" never appears in spam — without smoothing θ = 0 and any email containing "meeting" could *never* be spam regardless of other words. Smoothing converts the veto into mild evidence.

**Classify: "free win meeting"** (log₂ for clean numbers — any base works):
```
score(spam) = log .5 + log .4 + log .3 + log .1
            = −1 + (−1.322) + (−1.737) + (−3.322) = −7.381
score(ham)  = log .5 + log .1 + log .2 + log .4
            = −1 + (−3.322) + (−2.322) + (−1.322) = −7.966
```
spam wins (−7.381 > −7.966). Posterior if asked: P(spam|x) = 2^{−7.381}/(2^{−7.381} + 2^{−7.966}) ≈ 0.60 — note how *modest* the true posterior is despite spam "winning"; with correlated features it would have read as 0.95+. That contrast is the calibration lesson, demonstrated numerically.

Practice this whole flow — counts → smoothing → log scores → argmax — in under 4 minutes; it's a standard screen/whiteboard exercise.

---

## 5. Hyperparameters

NB is nearly hyperparameter-free — which is itself an interview point (nothing to overfit by tuning).

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| α (smoothing) | Pseudo-counts / Dirichlet prior strength | Likelihoods flatten toward uniform — bias ↑, robustness to rare words ↑ | →0: MLE counts, zero-veto risk, variance ↑ | "What happens at α=0? At α→∞?" → unseen-word veto / classifier degenerates to priors-only |
| Variant choice | Likelihood family | — | — | "Bernoulli vs Multinomial on short texts?" → Bernoulli models absences; often better for tweets/titles |
| class_prior / fit_prior | Override learned priors | — | — | "Train set was balanced-sampled but production is 1:100 — cheapest fix?" → swap the prior; likelihoods untouched. Elegant and rarely known |
| var_smoothing (Gaussian) | Adds ε to variances | Numerical stability; flattens spiky features | — | Guards near-zero-variance features dividing the density |
| binarize (Bernoulli) | Threshold for presence | — | — | minor |
| Complement NB | Estimate from *other* classes' counts | — | — | "NB on imbalanced text?" → CNB (sklearn's recommendation for text imbalance) |

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training | **One counting pass, O(n·d̄)** (d̄ = avg active features) — the fastest training that exists; trivially MapReduce-able (counts are summable) |
| Online updates | Increment counts per labeled example — true streaming learning, no optimizer; ideal for adversarial domains needing minutes-fresh models (spam) |
| Inference | O(active features) additions per class — microseconds; sparse-friendly; SQL-implementable (sum of log-weight lookups) |
| Memory | One log-prob table: classes × vocabulary — MBs even for large vocab |
| Scalability | Counts shard and sum perfectly; vocabulary growth handled by hashing or pruning rare terms |
| Monitoring | Class-prior drift (base rate vs trained prior — fix by prior swap, no retrain), vocabulary drift/OOV rate (new tokens land on smoothing mass — rising OOV = model going blind), calibration (always bad — never consume raw posteriors without recalibration), per-class token-distribution PSI |
| Role in modern stacks | High-throughput **pre-filter / first stage** in front of expensive models (filter 95% of obvious cases at ~zero cost, send the ambiguous band to a transformer) — the cascade pattern; also the always-on degraded-mode fallback |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **State the naive assumption precisely.** Features are independent *conditional on the class* — not marginally independent. Within-class factorization is the assumption; features can be wildly correlated overall via the class.
2. **Why is the assumption useful despite being false?** Collapses 2^d-style joint estimation to d univariate problems (data efficiency), and classification needs only the argmax — correlated-evidence double counting distorts probabilities more than rankings.
3. **What is Laplace smoothing and why mandatory?** Add-α pseudo-counts so no likelihood is zero; without it, one unseen feature-class pair vetoes the entire class via a zero product. Dirichlet-prior interpretation for bonus credit.
4. **Why compute in log space?** Products of hundreds of probabilities underflow floats; sums of logs don't. Also turns NB into an additive evidence model — each feature contributes log-likelihood-ratio "points."
5. **Multinomial vs Bernoulli vs Gaussian — when each?** Counts/text → Multinomial; binary presence (short text, flags) → Bernoulli (models absences explicitly); continuous → Gaussian (check per-class normality, or bin/transform).
6. **Generative vs discriminative — define and place NB/LR.** Generative models P(x|y)P(y) (can generate data, handles missing features naturally, data-efficient); discriminative models P(y|x) directly (better asymptotic accuracy, fewer assumptions). NB↔LR is *the* canonical pair — same linear boundary on text, different estimation.
7. **State the Ng–Jordan result.** NB reaches its (higher) asymptotic error in O(log d) examples; LR reaches its (lower) one in O(d). NB wins small data; LR overtakes with scale.
8. **Are NB probabilities calibrated?** No — independence violations saturate posteriors toward 0/1. Fine for argmax; recalibrate (isotonic/Platt) before any expected-value use.
9. **How does NB handle missing features at prediction time?** Drop the missing factors from the product — the generative formulation marginalizes them out naturally. (A genuine advantage over discriminative models; few candidates know it.)
10. **NB decision boundary — linear or not?** Multinomial/Bernoulli: linear in (log-)feature space. Gaussian with shared variance: linear; class-specific variances: quadratic. Saying "NB is linear on text" earns immediate credibility.
11. **Why does NB train so fast and update online so easily?** Parameters are sufficient statistics (counts/means) — one pass, no optimization loop; updates = increments. This is why it survives in adversarial, fast-drift domains.
12. **Class imbalance under NB?** Prior term handles base rates honestly; problems arise from minority-class likelihood estimates (few counts → high variance) — raise α, use Complement NB, or threshold on likelihood ratios with a cost matrix.
13. **What happens with duplicated/correlated features?** Evidence double-counts: log-LR added twice ⟹ overconfidence and bias toward whichever class the correlated bundle favors. Classifier may still rank fine; probabilities degrade most.
14. **NB for continuous features that aren't Gaussian?** Transform (log), discretize/bin (then Multinomial), or kernel density NB. Checking per-class histograms before trusting GaussianNB is the practitioner answer.
15. **Where does NB sit in a modern ML stack?** Cheap strong text baseline (with TF-IDF, embarrassingly competitive), streaming-fresh spam/abuse layer, cascade pre-filter before transformers, degraded-mode fallback. "Obsolete as a final model, alive as a component."

### Advanced (15)

1. **Derive the multinomial NB MLE θ̂_{c,w} = N_{c,w}/N_c.** Maximize Σ N_{c,w} log θ_{c,w} s.t. Σ_w θ = 1; Lagrangian gives θ ∝ counts. Then show MAP with Dirichlet(α+1) yields the smoothed estimator — smoothing isn't a hack, it's a prior.
2. **Show multinomial NB's log-odds is linear in counts and map it to LR.** log-odds = log prior ratio + Σ_w x_w log(θ_{1w}/θ_{0w}) — weights are log likelihood ratios per word; LR fits the same functional form discriminatively. NB = LR with weights set by counting under independence.
3. **Gaussian NB ↔ LDA/QDA relationships.** GNB with shared per-feature variance + diagonal covariance = diagonal LDA (linear); per-class variances = diagonal QDA (quadratic). Full-covariance LDA drops independence but shares Σ across classes. Placing NB in this family shows you see the assumption lattice.
4. **Prove the posterior is unchanged by dropping P(x).** P(x) = Σ_c P(c)P(x|c) is class-independent; argmax over c of a quantity divided by a positive constant is unchanged. Trivial but tests proof hygiene.
5. **Quantify the overconfidence mechanism with two perfectly correlated features.** True log-LR contribution ℓ counted twice ⟹ posterior odds squared (in odds space): a true 2:1 becomes 4:1, a 10:1 becomes 100:1 — saturation with no new information. The cleanest calibration-failure demonstration available; do it with numbers.
6. **Why does NB handle high-dimensional small-n text so well when KNN/RBF fail there?** NB estimates d *univariate* distributions — estimation error grows additively, not via joint-space volume; no distance geometry to concentrate. The curse of dimensionality is a joint-estimation curse, which NB sidesteps by assumption.
7. **Semi-supervised NB with EM — sketch it.** Initialize on labeled data; E-step: posterior class probabilities for unlabeled docs; M-step: re-estimate counts weighted by posteriors; iterate. Works because the generative model defines a likelihood over *all* data. Caveat: model misspecification can make unlabeled data hurt (classic Nigam et al. result + caveat = strong answer).
8. **Complement NB — what problem does it fix and how?** Long-doc and imbalanced-class biases in multinomial NB: estimate each class's weights from the *complement* (all other classes) so minority classes get stable estimates from abundant data; normalize weight vectors. sklearn's recommended text-NB for imbalance.
9. **NB as additive evidence / WoE scorecards.** Log-LR per feature = "weight of evidence"; summing WoE + prior = NB. Credit-risk scorecards are essentially NB with binned features — connecting these two worlds is a memorable senior moment.
10. **When features are conditionally independent given the class, is NB optimal?** Yes — it equals the Bayes classifier (posterior is exactly recovered). All NB error beyond Bayes error in practice is attributable to (a) violated independence, (b) wrong likelihood families, (c) finite-sample count noise. Decomposing the error sources like this is the analytical answer.
11. **How would you test the conditional-independence assumption on real data?** Per-class feature correlation matrices / mutual information conditioned on y; compare NB vs TAN (tree-augmented NB, allowing one dependency edge per feature) — if TAN wins big, independence was binding. Knowing TAN exists ≈ extra credit.
12. **Hashing trick with NB — what breaks?** Collisions merge unrelated words' counts ⟹ likelihood ratios blur; smoothing mass redistributes oddly. Acceptable at modest collision rates for big memory wins; monitor OOV+collision impact via a held-out exact-vocab comparison.
13. **NB under adversarial attack (spammers).** Attacks: good-word stuffing (inject hammy tokens to dilute spam evidence — works precisely because evidence is additive), rare-token churn (evade counts). Defenses: cap per-token contribution, Bernoulli over multinomial (stuffing repeats don't compound), fast online retrains, feature abstraction (hashes/normalization). NB's transparency is also its attack surface — say that sentence.
14. **Prior swap for domain shift — when is it exactly correct, when not?** Correct when P(x|y) is invariant and only P(y) shifts (label shift); then re-weighting the prior fully corrects the posterior. Breaks under covariate/concept shift (P(x|y) moved). Estimating the new prior without labels: EM/BBSE-style methods. This is a quietly deep question — the clean answer impresses.
15. **Connect NB to language models.** Class-conditional unigram LM per class + Bayes rule = multinomial NB; "P(document|class)" is literally a bag-of-words LM likelihood. Modern echo: zero-shot classification by comparing LM likelihoods under different class-conditioned prompts is the same inversion with a vastly better generative model. A closing answer that lands very well in 2026 interviews.

### Staff-Level (10)

1. **Design the spam/abuse layer for a chat product at 1M messages/min with minutes-level adversarial drift.** Cascade: NB/linear stage-1 on hashed tokens (µs, filters the obvious 90%+, online count updates from moderator labels within minutes) → transformer stage-2 on the ambiguous band → human review top tier feeding labels back. NB earns stage-1 on retrain speed + throughput; monitor OOV rate and per-token drift as the adversarial canary; cap per-token evidence vs stuffing. The architecture answer (cascade + feedback loop), not a model answer.
2. **Your NB pre-filter silently degrades the downstream transformer's measured quality. How?** Selection effect: stage-1 changes stage-2's input distribution; transformer trained on full traffic now sees only the hard band — calibration and thresholds shift; offline metrics computed on full-traffic holdouts no longer reflect serving. Fix: stage-aware training data, end-to-end funnel metrics, periodic stage-1 bypass traffic (a small "let everything through" slice) as the measurement spine. The bypass-slice idea is the staff move.
3. **A regulator asks you to explain a content-moderation decision made by NB. What can you provide, honestly?** Genuinely strong explanations: per-token log-LR contributions sum to the decision — an exact additive attribution (better than SHAP approximations, it's the model itself). Provide top evidence tokens with weights + prior. Caveats: probabilities miscalibrated (report decision margins, not posteriors), token-level evidence can expose training-data quirks. NB is one of the few models where "explain it" has an exact answer — say so.
4. **Small-data product decision: 200 labeled support tickets, 15 intents, need a router this week. Plan, and when do you switch?** NB/linear on TF-IDF (Ng–Jordan regime: generative wins tiny-n), priors set to expected traffic mix, α tuned by CV; ship with logging; switch criteria: learning-curve crossover as labels accumulate, or embedding+kNN few-shot (Ch. 7) if intents churn weekly. Present it as a staged plan with explicit switch triggers — that's the staff framing. *Directly relevant to your HelpBot origin story: what routed tickets before the RAG system had data?*
5. **Your GaussianNB on sensor features works in the lab, fails in production. Hypotheses?** Per-class normality broke (multimodal production conditions), feature correlations differ (lab rigs decorrelated, field correlated ⟹ overconfidence compounds), variance estimates from clean lab data too tight (production noise lands in density tails ⟹ wild posteriors), class priors wrong. Fixes: per-class distribution checks, var_smoothing, binning + multinomial, or concede to a model without the assumptions. Diagnosing *which assumption broke* is the answer's structure.
6. **You inherit a 10-year-old NB spam system with 40 hand-patched token-weight overrides. Modernization strategy without a quality dip?** Treat overrides as institutional knowledge: extract them as labeled constraints/test cases; train the replacement (linear/transformer cascade) and gate on the override test suite + interleaved shadow traffic; keep NB as the degraded-mode fallback and drift canary; deprecate overrides one cohort at a time with measured impact. The content is migration discipline, not modeling.
7. **When would you choose NB over logistic regression *at scale*, given LR also trains fast?** Honest, narrow: (a) streaming label feedback where count-increment updates beat even FTRL for operational simplicity, (b) missing-feature-heavy inference (generative marginalization), (c) prior-swap requirements (frequent known base-rate shifts with no retrain budget), (d) exact additive explainability mandates. Otherwise LR. Scoping NB's remaining edge precisely is the senior answer.
8. **Use NB reasoning to sanity-check an LLM-era system.** Examples: prior sensitivity — does the fancy classifier's output shift correctly when base rates shift (NB makes this explicit; many neural systems silently bake priors in)?; evidence additivity audit — sum of per-feature WoE as an interpretable shadow model flagging when the production model disagrees with simple evidence; data-efficiency floor — NB's small-n accuracy as the bar any few-shot LLM approach must beat to justify cost. Positioning NB as *instrumentation* is a fresh, senior take.
9. **Token-level evidence in NB leaks training-data information (e.g., a person's name became a strong spam token). Governance implications?** Memorization-at-the-parameter level: weight tables are inspectable PII risk; audits for identifier-like tokens, vocabulary hygiene (strip names/emails/phone patterns pre-count), differential-privacy noise on counts if mandated; same concern class as LLM memorization, in miniature and auditable. Few candidates connect NB to data governance — doing so stands out.
10. **Your team debates removing the NB stage-1 since the transformer's accuracy dominates. Defend or concede with numbers.** Frame as cost/risk, not accuracy: stage-1 saves X% of GPU inference (₹/month), provides minutes-fresh adversarial response between transformer retrains, and is the fallback during stage-2 outages; against: maintenance cost, funnel-measurement complexity (Q2), marginal quality loss in the filtered band. Decide via the bypass-slice experiment quantifying end-funnel deltas and total cost. The willingness to *measure rather than defend turf* is what's being interviewed.

---

## 8. Comparison Section

**Naive Bayes vs Logistic Regression** (the canonical generative-discriminative pair):

| | Naive Bayes | Logistic Regression |
|---|---|---|
| Models | P(y), P(x|y) → invert | P(y|x) directly |
| Boundary (text) | Linear (log space) | Linear |
| Weight fitting | Counting under independence | Discriminative MLE (convex opt) |
| Small-n | **Wins** (Ng–Jordan: O(log d) convergence) | Needs more data (O(d)) |
| Large-n asymptote | Higher error (assumption bias) | **Lower error** |
| Correlated features | Double-counts → overconfident | Weights share credit correctly |
| Calibration | Poor | Good |
| Missing features at inference | Marginalize naturally | Needs imputation |
| Training/update | One pass; increment counts | Iterative; FTRL for online |

**NB vs KNN:** opposite philosophies in high-d small-n — NB sidesteps the curse via univariate estimation; KNN drowns in it via joint-space distances. NB: global parametric-ish evidence; KNN: local memory.

**NB vs decision trees/GBM:** NB cannot represent interactions (additive evidence by construction); trees are interaction machines. Text/sparse-counts → NB/linear; dense tabular with interactions → trees. Clean division of labor.

**Gaussian NB vs LDA vs QDA:** the assumption lattice — GNB: diagonal covariance per class; LDA: full but shared covariance (linear); QDA: full per-class (quadratic). More parameters → more data needed → the same bias-variance ladder in covariance-modeling form.

**NB vs modern LLM classification:** NB = class-conditional unigram LM + Bayes inversion; zero-shot LLM classification = the same inversion with a trillion-parameter likelihood. The 60-year-old skeleton survived; the likelihood got better.

---

## 9. Common Mistakes

**Candidate mistakes:**
- "Assumes features are independent" without *conditionally, given the class*.
- Forgetting smoothing, or unable to explain the zero-veto failure it prevents.
- Trusting/presenting NB posteriors as calibrated.
- Not knowing NB is linear on text (and therefore directly comparable to LR).
- Missing the Ng–Jordan small-data result — the theory question this model exists to ask.

**Production mistakes:**
- Raw posteriors feeding expected-value decisions (saturated 0.999s everywhere).
- Vocabulary frozen while language drifts — rising OOV silently blinds the model.
- Balanced-sampled training priors shipped against imbalanced reality (one-line prior fix never applied).
- No cap on per-token evidence in adversarial settings (good-word stuffing walks right in).

**Modeling mistakes:**
- GaussianNB on multimodal/heavy-tailed features without checking per-class distributions.
- Multinomial NB on TF-IDF floats without acknowledging the count-model mismatch (works-ish, but know it's off-label).
- Duplicated engineered features (ratios + raw components) compounding evidence.
- Using NB where interactions carry the signal, then blaming "the data."

---

## 10. Real Industry Use Cases

- **Google** — Gmail spam filtering's historical core layer; NB-style token evidence persists inside layered abuse systems; Smart Reply-era fast text classifiers.
- **Amazon** — review-spam and seller-abuse pre-filters; high-throughput product-categorization first stages; marketplace policy triage.
- **Netflix** — lightweight text classification on support/feedback streams; tagging baselines feeding human curation.
- **Meta** — integrity cascades: cheap text stages in front of heavy models for spam/scam at feed scale; the cascade economics in §6 is their pattern.
- **Uber** — support-ticket intent routing baselines; receipt/document field classification pre-filters.
- **Swiggy/Zomato** — review-spam and fake-review detection layers; support-ticket routing; menu-item text categorization. *Naveen: "before HelpBot's RAG, intent routing needed a day-one baseline — NB/linear on TF-IDF is what that looks like, and it stayed as the fallback" is a credible architecture-history beat for your story.*
- **Flipkart** — catalog text classification at ingest scale (millions of seller listings — throughput is the constraint NB answers); review-abuse pre-filters.
- **Games24x7** — chat-abuse and collusion-signal text filtering (in-game chat at high throughput, adversarial drift — the §7 staff Q1 cascade is the architecture); KYC document-type classification pre-filters.

---

## 11. Coding From Scratch (NumPy only)

Multinomial NB with Laplace smoothing, fully vectorized, log-space throughout — the variant interviews actually request (text).

```python
import numpy as np

class MultinomialNBScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha                  # Laplace/Lidstone pseudo-count
        self.class_log_prior_ = None        # log P(y=c)
        self.feature_log_prob_ = None       # log P(word w | class c)
        self.classes_ = None

    def fit(self, X, y):
        """
        X: (n_docs, vocab) count matrix (dense here; sparse in production).
        Training = counting. No loops over rows, no optimizer, one pass.
        """
        X = np.asarray(X, float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes, V = len(self.classes_), X.shape[1]

        # Class priors from document frequencies: log(n_c / n).
        counts_c = np.array([(y == c).sum() for c in self.classes_])
        self.class_log_prior_ = np.log(counts_c) - np.log(counts_c.sum())

        # Word counts per class: one matrix per class summed over its docs.
        word_counts = np.vstack([X[y == c].sum(axis=0) for c in self.classes_])

        # Smoothed likelihoods:
        #   theta_{c,w} = (N_{c,w} + alpha) / (N_c + alpha * V)
        # The +alpha is the Dirichlet prior; the alpha*V keeps each row a
        # proper distribution. Without it: one unseen word vetoes the class.
        smoothed = word_counts + self.alpha
        self.feature_log_prob_ = (np.log(smoothed)
                                  - np.log(smoothed.sum(axis=1, keepdims=True)))
        return self

    def _joint_log_likelihood(self, X):
        # log P(c) + sum_w x_w * log theta_{c,w}
        # = prior + COUNT-WEIGHTED sum of log-probs. One matmul does all docs
        # and all classes at once: (n_docs, V) @ (V, n_classes).
        return np.asarray(X, float) @ self.feature_log_prob_.T \
               + self.class_log_prior_

    def predict(self, X):
        return self.classes_[self._joint_log_likelihood(X).argmax(axis=1)]

    def predict_log_proba(self, X):
        jll = self._joint_log_likelihood(X)
        # Log-sum-exp normalization WITHOUT leaving log space:
        # subtract the row max first so exp() can't overflow/underflow.
        m = jll.max(axis=1, keepdims=True)
        log_norm = m + np.log(np.exp(jll - m).sum(axis=1, keepdims=True))
        return jll - log_norm

    def predict_proba(self, X):
        # Remember: argmax is trustworthy; these PROBABILITIES are not
        # (independence violations saturate them). Recalibrate downstream.
        return np.exp(self.predict_log_proba(X))

    def partial_fit_increment(self, X_new, y_new):
        """Online learning = add counts and recompute logs. THE selling point."""
        # (Maintain raw count caches in a fuller implementation; shown
        # conceptually:) word_counts[c] += X_new[y_new == c].sum(0); re-log.
        ...
```

Narration points that earn the senior signal:
- **"Training is a matmul-free counting pass"** — point at `fit` and note there is no loop over examples and no optimizer; sufficient statistics are the parameters.
- **The smoothing line** — derive (N+α)/(N_c+αV) as Dirichlet MAP, and state the zero-veto failure it prevents.
- **One matmul scores all docs × all classes** — the count-weighted log-prob sum as `X @ logθᵀ`; NB inference is literally a linear layer (echoes "NB is linear on text").
- **Log-sum-exp with max subtraction** — the numerically correct way to normalize posteriors; same stability theme as Chapters 2/5.
- **The predict_proba comment** — volunteering "argmax yes, probabilities no" inside the code is exactly the kind of judgment interviews screen for.
- Extensions if asked: Bernoulli (add the (1−θ) absence terms — and note you must then iterate the *full* vocab per doc, not just present words); Complement NB (swap class counts for complement counts); hashing for vocab control.

---

## 12. ML System Design Perspective

**Choose NB when:** day-one text baselines with tiny labels (Ng–Jordan regime); streaming/adversarial domains needing minute-fresh count updates (spam, abuse, fraud-text); high-throughput cascade pre-filters in front of expensive models; exact additive explanations are mandated; missing-feature-heavy inference; known base-rate shifts handled by prior swap without retraining.

**Avoid when:** probabilities are consumed raw downstream (or commit to recalibration); interactions carry signal; continuous features with awkward per-class distributions; correlated engineered-feature sets; the final accuracy percent matters and data is plentiful (LR/GBM/transformer overtake).

**Data requirements:** the most forgiving model in this handbook — tens of examples per class produce sensible behavior; needs vocabulary hygiene (normalization, identifier stripping) and α tuning more than data volume.

**Latency:** sparse additions per class — microseconds; effectively the latency floor for probabilistic classification alongside linear models; SQL/edge deployable as lookup-and-sum.

**Scale limits:** none that matter for training (summable counts shard perfectly); vocabulary growth is the only axis to manage (hashing/pruning). The constraint is *representational* (no interactions, miscalibration), never computational.

---

## 13. Resume Discussion Angle

**Text/NLP systems (HelpBot — your main exposure):** the question that reaches NB: "what was your baseline before the RAG/LLM system, and how did you justify the upgrade?" Strong shape: TF-IDF + NB/linear router as the day-one and fallback model; the learning-curve/accuracy bar it set; what the expensive system had to beat *per rupee of inference cost*; and that it stayed in the stack as degraded-mode fallback and drift canary. Interviewers consistently reward candidates who keep cheap models employed.

**Fraud detection (Games24x7):** chat/text signals in collusion detection — NB-style token evidence as the high-throughput first pass over in-game chat, online-updated as moderators label; the good-word-stuffing adversarial cat-and-mouse and the per-token evidence cap (advanced Q13) make a vivid, experienced-sounding beat.

**Recommendation/personalization:** NB rarely stars here; if probed, the honest placement: content classification feeding catalog metadata (cuisine/category taggers at ingest), and the generative-vs-discriminative vocabulary it gives you for discussing *why* your main models are discriminative.

**Marketing ML:** quick campaign-response baselines on tiny pilot data (the Ng–Jordan regime is exactly "we ran a 500-person pilot"); WoE-scorecard kinship (advanced Q9) connects NB to credit/risk language McKinsey QB and fintech loops speak natively — *useful for Tide-style conversations*.

**The universal trap:** "Naive Bayes assumes independence — so why does it work?" The full-marks answer has three moves: (1) correct the premise — *conditional* independence given the class; (2) the argmax argument — classification survives monotone evidence inflation that probability estimation doesn't; (3) the consequence — deploy it as a classifier, never as a probability source without recalibration. Three sentences, done — and it signals you understand the difference between a model being *wrong* and being *useful*, which is quietly the theme of every senior ML interview.

---
*Previous: KNN ← | Next batch: K-Means, PCA, SVD →*

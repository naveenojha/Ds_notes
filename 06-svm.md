# Chapter 6: Support Vector Machines (SVM)
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

An SVM finds the separating hyperplane that maximizes the margin — the distance to the nearest training points (the support vectors). The soft-margin version trades margin width against misclassification via hinge loss with an L2 penalty, and the kernel trick lets it learn non-linear boundaries by computing inner products in implicit high-dimensional spaces without ever constructing them. SVMs are no longer the industrial default (GBMs and deep nets took that), but they remain interview staples because they test optimization, duality, kernels, and regularization in one model — and the kernel/margin ideas live on inside modern ML (max-margin intuitions, kernel methods, similarity-based learning).

---

## 2. Interview Articulation (3–4 Minute Answer)

"The SVM starts from a question logistic regression never asks: among all hyperplanes that separate two classes, which one should I pick? Logistic regression picks whichever one maximizes likelihood. The SVM answers geometrically: pick the one with the **widest margin** — maximum distance to the closest points on either side. Intuition: a wide-margin boundary has slack against noise; new points near the old ones still land on the correct side. There's also theory behind it — margin-based generalization bounds — but the geometric intuition carries the interview.

The mathematical setup: a hyperplane wᵀx + b = 0; the distance of a point to it is |wᵀx + b| over the norm of w. Fix the scale so the closest points satisfy y(wᵀx + b) = 1; then the margin is 2 over norm-w, and maximizing margin becomes **minimizing ½‖w‖²** subject to every point being on the right side with functional margin at least 1. A clean convex quadratic program.

Real data isn't separable, so we add slack variables — let points violate the margin, but pay for it. That's the soft-margin SVM, and it has an equivalent unconstrained form everyone should know: **minimize ½‖w‖² + C·Σ hinge loss**, where hinge loss is max(0, 1 − y·score). Hinge is zero once a point is correctly classified *with* margin — beyond the margin, the SVM stops caring about a point entirely. That's the deepest contrast with logistic regression, whose log loss never reaches zero and keeps pulling on every point. Consequence: the solution depends only on points on or inside the margin — the **support vectors**. Everything else could be deleted without changing the model.

C is the regularization dial, inverted relative to lambda: large C punishes violations hard — narrow margin, low bias, high variance; small C tolerates violations — wide margin, more regularization.

Then the celebrated part: the **kernel trick**. Solve the optimization in its dual form and the data only ever appears as inner products between pairs of points. So replace every inner product with a kernel function K(x, x′) that equals an inner product in some high — possibly infinite — dimensional feature space, and you've trained a linear separator in that space without ever computing the mapping. RBF kernel: similarity decaying with squared distance — gives you smooth, localized, infinitely-dimensional decision boundaries with two hyperparameters, C and gamma. Gamma sets the radius of influence of each support vector: high gamma, spiky boundaries hugging individual points — overfitting; low gamma, broad smooth boundaries — underfitting.

Training: a QP, classically solved by SMO; practical complexity between quadratic and cubic in n — which is the production Achilles heel: kernel SVMs don't scale past a few hundred thousand points. Linear SVMs do — liblinear, Pegasos-style SGD on hinge loss — and were the text-classification workhorse for a decade.

Inference: linear SVM is one dot product. Kernel SVM is a kernel evaluation against *every support vector* — if 30% of your data are support vectors, that's slow, and the number of support vectors grows with data size and noise.

Honest industry status: largely displaced — GBMs on tabular, deep nets on text/vision, and the SVM's lack of native probabilities (you bolt on Platt scaling, which is fitting a logistic regression on the scores) hurts it wherever downstream consumes probabilities. It survives in small-n high-d niches — bioinformatics, some text problems — and as the conceptual ancestor of margin-based and kernel methods.

Traps: claiming SVMs give probabilities natively; forgetting that C and gamma interact (you tune them jointly on a log grid); saying the kernel trick 'projects the data' — it never computes the projection, that's the whole point; and forgetting feature scaling, which is *mandatory* — margins and RBF distances are geometry, and geometry in unscaled units is meaningless."

---

## 3. Mathematical Foundation

**Geometry.** Hyperplane wᵀx + b = 0; signed distance of xᵢ: yᵢ(wᵀxᵢ + b)/‖w‖. Canonical scaling: minᵢ yᵢ(wᵀxᵢ+b) = 1 ⟹ margin = 2/‖w‖.

**Hard-margin primal:**
```
min ½‖w‖²   s.t.  yᵢ(wᵀxᵢ + b) ≥ 1  ∀i
```

**Soft-margin primal (slack ξᵢ ≥ 0):**
```
min  ½‖w‖² + C Σ ξᵢ    s.t.  yᵢ(wᵀxᵢ + b) ≥ 1 − ξᵢ
⟺ unconstrained:  min  ½‖w‖² + C Σ max(0, 1 − yᵢ(wᵀxᵢ + b))
```
i.e., **L2-regularized hinge loss** — put SVM on the same loss-function shelf as logistic regression (log loss) and boosting (any loss); interviewers reward this unifying view.

**Dual** (via Lagrangian, multipliers αᵢ ≥ 0):
```
max  Σαᵢ − ½ ΣΣ αᵢαⱼ yᵢyⱼ ⟨xᵢ, xⱼ⟩     s.t.  0 ≤ αᵢ ≤ C,  Σαᵢyᵢ = 0
w = Σ αᵢ yᵢ xᵢ          (weights = combination of support vectors only)
```
**KKT conditions sort the points:**
```
αᵢ = 0      → outside margin (correctly classified, ignored by the model)
0 < αᵢ < C  → exactly ON the margin (free support vectors; used to compute b)
αᵢ = C      → inside margin or misclassified (bound support vectors)
```

**Kernel trick.** Data enters the dual only through ⟨xᵢ,xⱼ⟩ ⟹ substitute K(xᵢ,xⱼ) = ⟨φ(xᵢ),φ(xⱼ)⟩:
```
f(x) = Σ αᵢ yᵢ K(xᵢ, x) + b      (sum over support vectors only)
```
Valid kernels = symmetric PSD (Mercer). Common ones:

| Kernel | K(x, x′) | Notes |
|---|---|---|
| Linear | xᵀx′ | use for d ≫ n (text); scalable solvers |
| Polynomial | (γxᵀx′ + r)^p | explicit interaction orders |
| RBF / Gaussian | exp(−γ‖x−x′‖²) | infinite-dim feature space; the default non-linear choice |

**RBF locality:** prediction = weighted vote of support vectors, weights decaying with distance at rate γ. γ ↑ ⟹ each SV's influence shrinks ⟹ wiggly boundary (variance ↑). γ ↓ ⟹ near-linear smoothness (bias ↑). γ ≈ 1/(2σ²) of the implied Gaussian.

**Hinge vs log loss (know this plot in words):** both upper-bound 0–1 loss; hinge hits exactly 0 at margin ≥ 1 (sparse solutions, no probability semantics); log loss is smooth, never zero (every point matters, calibrated probabilities). Squared hinge = differentiable variant.

**SVR (regression) in one breath:** ε-insensitive tube — zero loss for errors within ±ε, linear loss outside; support vectors = points on/outside the tube; same dual/kernel machinery.

---

## 4. Step-by-Step Numerical Example

1-D, trivially separable — but it makes margin math concrete.

Points: x = 1, 2 (y = −1) and x = 4, 5 (y = +1).

The max-margin boundary must sit midway between the closest opposite points (2 and 4) ⟹ boundary at x = 3, margin = distance to nearest points = 1 on each side.

Recover w, b in canonical form (f(x) = wx + b, with f = ±1 at the margin points):
```
w·2 + b = −1
w·4 + b = +1     ⟹ subtract: 2w = 2 ⟹ w = 1, b = −3
Margin = 2/‖w‖ = 2/1 = 2  ✓ (total width: from x=2 to x=4)
```
Support vectors: x = 2 and x = 4 only. Check x = 1: y·f(x) = (−1)(1−3) = 2 > 1 ⟹ α = 0 — delete it and *nothing changes*. That sentence ("I can delete every non-support vector and the model is identical") is the demonstration interviewers want.

**Soft-margin flavor:** add a noisy positive at x = 2.5. With large C the boundary lurches left and the margin collapses to fit it; with small C the model pays the hinge penalty ξ = 1 − f(2.5) and keeps the wide margin — write the tradeoff as ½w² + C·ξ and compare the two totals. Shows C as the noise-tolerance dial with arithmetic, not adjectives.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| C | Violation penalty (inverse regularization) | Narrow margin, fewer violations, variance ↑ | Wide margin, more slack, bias ↑ | "C vs λ?" → C ≈ 1/λ; sklearn's LR uses C the same way |
| γ (RBF) | Inverse radius of SV influence | Spiky local boundaries, overfit, **more SVs** | Smooth, near-linear, underfit | "High train acc, bad test, RBF — first suspect?" → γ too high; tune C,γ jointly on log grid |
| Kernel | Feature-space geometry | — | — | "When linear over RBF?" → d ≫ n (text), or n too big for kernel solvers |
| degree, r (poly) | Interaction order | Higher-order terms, overfit | — | Rarely beats RBF; numerically touchy |
| ε (SVR) | Insensitive-tube width | Fewer SVs, coarser fit | Tighter fit, more SVs | "What does ε control?" → sparsity-accuracy tradeoff |
| class_weight | Per-class C scaling | Minority margin enforced harder | — | Imbalance: scale C by inverse frequency |
| tol, cache_size | Solver controls | — | — | Practical: kernel cache dominates training speed |

Mandatory preprocessing: **standardize features** — distances and margins are unit-dependent; this is the #1 practical SVM bug.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | Kernel SVM (SMO): ~O(n²)–O(n³) + O(n²) kernel cache — practical ceiling ~10⁵ rows. Linear SVM (liblinear/SGD-hinge): O(n·d) per epoch — scales to web-size sparse data |
| Inference | Linear: one dot product (µs). Kernel: O(#SV · d) kernel evals — #SV grows with n and noise; can be slower than a 500-tree GBM |
| Memory | Kernel: store all support vectors (can be a large fraction of training data). Linear: d floats |
| Scalability fixes | Linear kernel + feature maps; Nyström / random Fourier features (approximate RBF in explicit finite dims, then linear solver) — the senior-level "how would you scale kernel SVM" answer |
| Probabilities | Not native; Platt scaling (logistic fit on scores, needs internal CV) — adds cost and a calibration layer to monitor |
| Monitoring | Score-distribution drift; #SV fraction across retrains (rising ⟹ noisier data or γ/C drift); margin-violation rate as a data-quality canary |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What is the margin and why maximize it?** Distance from boundary to nearest points; wider margin ⟹ robustness to perturbation and better generalization bounds (margin theory), not just aesthetics.
2. **What are support vectors?** Points with α > 0 — on or inside the margin (or misclassified). They alone define the solution; all other points are redundant.
3. **Hard vs soft margin?** Hard requires separability (and is outlier-fragile — one bad point wrecks the margin); soft adds slack with penalty C — always feasible, noise-tolerant.
4. **What does C control?** The margin-width vs violation tradeoff: large C ≈ hard margin (variance ↑), small C ≈ heavy regularization (bias ↑). Inverse of λ.
5. **State the hinge loss and its key property.** max(0, 1 − y·f(x)); exactly zero beyond the margin ⟹ correctly-and-confidently classified points exert zero pull ⟹ sparse SV solutions.
6. **SVM vs logistic regression?** Hinge vs log loss; sparse geometric solution vs probabilistic model; no native probabilities vs calibrated ones; kernels natural vs manual features. With clean margins and no probability need, similar accuracy — choose by downstream use.
7. **What is the kernel trick?** Dual depends on data only via inner products ⟹ replace with K(x,x′) = ⟨φ(x),φ(x′)⟩, training a linear model in φ-space without computing φ. Cost shifts from dimensionality to n².
8. **Why does RBF correspond to infinite dimensions?** Its Taylor expansion = inner product of infinite feature series; any finite dataset becomes separable in that space — hence regularization (C) is what prevents trivial overfit.
9. **What does γ do, intuitively?** Radius of each support vector's influence: γ ↑ = local spiky boundary, γ ↓ = smooth global one. The RBF's bias-variance dial alongside C.
10. **Why must features be scaled?** Margins and RBF distances are Euclidean geometry; a feature in large units dominates distance and warps the boundary. Standardize always.
11. **How do you get probabilities from an SVM?** Platt scaling: fit σ(a·score + b) on held-out scores. Caveats: extra CV cost, imperfect calibration, monotone-only correction. If probabilities matter, often just use logistic regression.
12. **Multi-class SVM?** One-vs-rest (standard) or one-vs-one (libsvm: k(k−1)/2 machines, vote). No native multiclass margin in common practice.
13. **Imbalanced classes?** class_weight scales C per class (heavier penalty for minority violations); evaluate PR metrics; threshold the score, not 0.
14. **What is SVR?** ε-insensitive tube regression: no loss inside ±ε, linear outside; sparse in support vectors; same kernels. Use when sparse robust fits with tolerance bands make sense.
15. **When is a linear SVM the right tool today?** High-dimensional sparse text/IDs with n up to many millions (liblinear/SGD): fast, strong, simple — historically *the* text baseline before transformers, still a legitimate cheap one.

### Advanced (15)

1. **Derive the margin = 2/‖w‖.** Distance of x to hyperplane = |wᵀx+b|/‖w‖; canonical constraint sets the closest points' numerator to 1; margin spans both sides ⟹ 2/‖w‖. Maximizing it ⟺ minimizing ½‖w‖².
2. **Sketch the dual derivation and why it matters.** Lagrangian L = ½‖w‖² − Σαᵢ[yᵢ(wᵀxᵢ+b) − 1]; ∂L/∂w = 0 ⟹ w = Σαᵢyᵢxᵢ; ∂L/∂b = 0 ⟹ Σαᵢyᵢ = 0; substitute back ⟹ dual in α with data as inner products only ⟹ kernels become possible, and sparsity (most αᵢ = 0) emerges from KKT.
3. **State the KKT complementary slackness conditions and what each α regime means.** αᵢ[yᵢf(xᵢ) − 1 + ξᵢ] = 0: α=0 outside margin; 0<α<C exactly on margin (ξ=0; these determine b); α=C margin violators. Reading a trained model's α spectrum tells you its noise exposure.
4. **What makes a function a valid kernel (Mercer)?** Symmetric and PSD Gram matrices for all finite point sets ⟹ guarantees an implicit feature space exists. Closure: sums, products, positive scalings of kernels are kernels — lets you compose domain-specific similarities.
5. **Why does the SVM solution depend only on support vectors, mechanically?** w = Σαᵢyᵢxᵢ with α=0 for non-SVs; hinge loss is flat (zero gradient) beyond the margin, so those points never enter the optimality conditions.
6. **Hinge vs log loss: implications for outliers and calibration?** Both grow linearly for badly misclassified points (robust vs squared losses); hinge's hard zero ⟹ sparsity but no probability semantics (scores aren't log-odds); log loss's everywhere-positive gradient ⟹ dense solutions but proper-scoring-rule calibration.
7. **Random Fourier features / Nyström — how do they scale kernel SVMs?** Approximate K(x,x′) ≈ z(x)ᵀz(x′) with explicit finite-dim z (RFF: sampled cosines from the kernel's spectral measure; Nyström: low-rank Gram approximation from landmark points) ⟹ train a *linear* model on z — O(n) methods recovering most RBF accuracy. The canonical "make it scale" answer.
8. **Pegasos / SGD on hinge — why does it work and what's the rate?** Primal hinge+L2 is convex; subgradient SGD with step ∝ 1/(λt) achieves Õ(1/(λε)) iterations *independent of n* — why linear SVMs train on web-scale data.
9. **How does ν-SVM differ from C-SVM?** ν ∈ (0,1] directly upper-bounds the margin-violation fraction and lower-bounds the SV fraction — a more interpretable dial than C when you want "at most 5% violations."
10. **One-class SVM — what does it do?** Unsupervised boundary around the data's support (separate data from origin in feature space / minimal enclosing region); ν = expected anomaly fraction. An anomaly-detection alternative to Isolation Forest — distance/kernel-based vs tree-isolation-based; IF scales better, OC-SVM gives smoother boundaries on small data.
11. **Why can #support vectors be viewed as a generalization diagnostic?** LOO error ≤ #SV/n (removing a non-SV changes nothing). A model that keeps 60% of points as SVs is memorizing margins — expect poor generalization; check γ/C.
12. **What happens to RBF-SVM as γ → ∞ and γ → 0?** γ→∞: each SV influences only itself ⟹ 1-NN-like memorization. γ→0: K → constant + linear term ⟹ effectively linear model. RBF interpolates between 1-NN and linear — a beautiful framing worth saying verbatim.
13. **Kernel SVM vs Gaussian Process classification?** Same kernel machinery; GP is Bayesian (full predictive distributions, principled hyperparameter learning via marginal likelihood) at O(n³); SVM is a point-estimate margin machine with sparsity. Choose GP when uncertainty matters and n is small.
14. **Why is the SVM QP convex, and what does that buy?** Objective ½αᵀQα − 1ᵀα with Q = (yᵢyⱼK(xᵢ,xⱼ)) PSD (Mercer) ⟹ global optimum, solver-independent solution — unlike neural nets, no initialization lottery.
15. **Squared hinge — why would you use it?** Differentiable everywhere (cleaner optimization), penalizes violations quadratically (less outlier-robust); liblinear's L2-loss option. Tradeoff: smoothness vs robustness.

### Staff-Level (10)

1. **A legacy kernel-SVM scorer (200K SVs) sits in a latency-critical path. Migration plan?** Profile first (SV evaluation cost vs feature fetch); options ranked: distill into GBM/small NN on SVM scores (usually lossless in metric, 100× faster), random-Fourier-feature linear approximation (principled, bounded error), retrain modern model on current data (the SVM is likely stale anyway). Shadow-deploy, compare on temporal holdout + calibration, keep the SVM as fallback one release. The answer is a migration playbook, not a model opinion.
2. **When, today, would you *choose* an SVM for a new production system?** Honest, narrow answer: small-n (≤10⁴) high-d problems with weak feature engineering budget (bio/chem assays, niche text), one-class anomaly detection on small clean data, or as a strong cheap baseline (linear SVM) on sparse text where you need a number this afternoon. Anything else: justify against GBM/LR/NN and usually lose. Interviewers respect the honest scoping more than SVM advocacy.
3. **Your team uses Platt-scaled SVM probabilities for expected-value decisions. Risks and monitoring?** Platt is a 2-parameter monotone squash — can't fix non-sigmoid miscalibration; trained on a fold that drifts; SV sparsity means score distribution shifts sharply on retrain. Monitor reliability curves + ECE per segment, recalibrate on recent data each retrain, alert on score-distribution PSI. Or migrate to a natively-calibrated model and delete a failure mode.
4. **n = 3,000 labeled compounds, d = 8,000 features, accuracy is publication-critical. Approach and why might SVM win here?** Small-n high-d is SVM's home: margin + kernel regularization controls variance where GBMs overfit and NNs starve. Protocol: nested CV (outer for honest error, inner for C,γ), linear-vs-RBF comparison, RFF sanity check, permutation-test the final accuracy. Mention the multiple-comparison trap of tuning on the test fold — that's the real staff content here.
5. **Explain to a junior why their RBF-SVM gets 100% train / 60% test, and the systematic fix.** Diagnosis: γ too high (1-NN regime) and/or C too high; #SV fraction will be huge — show them that diagnostic. Fix: joint log-grid over (C, γ) with CV, standardized features, learning curves to see whether more data or more regularization is the binding constraint. Teach the γ-interpolates-1NN-to-linear picture so they can reason, not just grid-search.
6. **Design text classification for 50M documents where the 2010 answer was linear SVM. What's your 2026 answer and what survives?** Survives: TF-IDF + linear model (SVM or LR) as the cheap strong baseline and the latency floor. Modern: frozen transformer embeddings + linear head (often the sweet spot), fine-tuned small transformer if metric-critical. Decide by accuracy-per-cost curve; the linear baseline also becomes the drift canary. The staff signal is keeping the old tool *in the system* as baseline/canary rather than discarding it.
7. **Your fraud team wants one-class SVM for novelty detection over Isolation Forest. Adjudicate.** OC-SVM: kernel-smooth boundaries, principled ν = anomaly budget, O(n²) training, scaling pain, sensitive to γ. IF: O(n log n), high-d tolerant, trivially parallel, axis-aligned artifacts. At fraud scale, IF (or deep methods) usually wins operationally; OC-SVM defensible for small clean sensor-style data. Then the bigger point: novelty layer feeds investigations feeds supervised layer — the architecture matters more than the detector choice.
8. **A regulator asks how your SVM-based credit screen makes decisions. What can you honestly provide, and where does it strain?** Linear SVM: weights ≈ per-feature contributions (after scaling) — workable. Kernel SVM: decision = similarity-weighted vote over support vectors — no per-feature decomposition exists natively; SHAP on the score function is approximate and expensive. If reason codes are legally required, kernel SVM is the wrong tool — saying that plainly is the right answer.
9. **How would you detect that an adversary is probing your SVM's decision boundary?** Margin-region telemetry: spike in queries with |f(x)| near 0, structured input sequences walking the boundary, per-account score-variance anomalies. Mitigations: score quantization/jitter near threshold, rate limits on near-boundary queries, ensemble disagreement as a probe alarm. Margin-based models make "near the boundary" unusually well-defined — use that.
10. **Connect max-margin thinking to anything in modern deep learning.** Margin-based losses everywhere: softmax cross-entropy implicitly maximizes margins (logit margin theory of generalization); metric-learning losses (triplet, contrastive, ArcFace's additive-margin softmax) are explicit margin constructions in embedding space — directly relevant to two-tower retrieval training; label smoothing and margin-aware ranking losses likewise. The SVM died as a product and survived as a loss-design philosophy — a memorable closing line, and it bridges straight into your retrieval-embedding story.

---

## 8. Comparison Section

| | SVM (kernel) | SVM (linear) | Logistic Regression | GBM | KNN |
|---|---|---|---|---|---|
| Loss | Hinge + L2 | Hinge + L2 | Log loss | Any | None |
| Probabilities | No (Platt bolt-on) | No (Platt) | Native, calibrated | Decent, recalibrate | Poor |
| Non-linearity | Kernels | No | No (manual FE) | Automatic | Automatic (local) |
| Scaling required | **Yes** | Yes | Yes (for reg.) | No | **Yes** |
| Train scale ceiling | ~10⁵ | ~10⁸+ (SGD) | ~10⁹ (FTRL) | ~10⁹ (hist) | n/a (lazy) |
| Inference cost | O(#SV·d) | O(d) | O(d) | O(trees·depth) | O(n) naive / O(log n) ANN |
| Sparse solution | Yes (SVs) | Yes-ish | With L1 | n/a | No |

**SVM vs logistic regression, the one-paragraph version:** same linear-boundary family, different loss philosophy. Hinge: geometric, sparse, stops caring past the margin, no probabilities. Log: probabilistic, dense, calibrated, every point matters. If downstream consumes probabilities — bids, expected loss — LR (or recalibrated anything) wins by construction. If you need kernels on small data, SVM earns its keep.

**RBF-SVM vs KNN:** both similarity-local; SVM keeps only support vectors with learned weights and a global margin objective; KNN keeps everything with uniform local votes. γ→∞ literally degenerates RBF-SVM into ~1-NN — the cleanest bridge into the next chapter.

**SVM vs GBM (the "why did industry move on" question):** GBM matches/beats kernel-SVM accuracy on tabular data at 100× the training scale, with native missing handling, probabilities, and importance tooling; the SVM's n² kernel wall and probability bolt-ons lost the production war. Concede this gracefully and credit what survived (margins, kernels, similarity learning).

---

## 9. Common Mistakes

**Candidate mistakes:**
- "SVM outputs probabilities" — it outputs signed distances; Platt is an add-on.
- Explaining the kernel trick as "projecting the data up" — the projection is never computed; the *inner products* are.
- Tuning C or γ alone — they interact strongly; joint log-grid.
- No mention of feature scaling.
- Unable to state KKT/α regimes or why solutions are sparse.

**Production mistakes:**
- Kernel SVM on n ≥ 10⁶ "because accuracy" — discovering the n² wall in the training bill.
- Ignoring #SV growth: serving latency that degrades as training data grows.
- Platt layer trained once, never recalibrated, feeding expected-value decisions.
- Unscaled or differently-scaled train/serve features (geometry silently warps).

**Modeling mistakes:**
- RBF on high-dim sparse text (distances concentrate; linear wins).
- Reading per-feature "importance" off a kernel SVM (no native decomposition).
- Hard-margin mindset on noisy labels — one flipped label reshapes the boundary; C exists for a reason.

---

## 10. Real Industry Use Cases

- **Google** — historical: early spam/text classification stacks were linear-SVM-era; modern relevance is the margin-loss lineage in metric learning for retrieval.
- **Amazon** — legacy text/product classification baselines; one-class methods in anomaly tooling; today mostly displaced by GBM/DL but alive in interview questions.
- **Netflix** — early-era ranking/classification experiments; margin-based metric learning concepts inside embedding similarity work.
- **Meta** — face-verification ancestry: margin-based losses (contrastive/triplet/ArcFace family) are the SVM's living descendants in embedding training.
- **Uber** — one-class/OC-SVM-style novelty detection in fraud/sensor pipelines at small scale; linear SVM baselines in document/ticket routing.
- **Swiggy/Zomato** — ticket/review text classification baselines (linear SVM on TF-IDF was the standard pre-transformer answer); menu-item categorization. *Naveen: if asked about pre-LLM text baselines for your HelpBot story, "TF-IDF + linear SVM/LR, which the RAG system had to beat on routing accuracy" is a credible, dated-correctly answer.*
- **Flipkart** — catalog/product-type classification legacy systems; counterfeit-listing detection baselines.
- **Games24x7** — one-class anomaly framing as the conceptual cousin of your Isolation Forest layer — be ready for "why IF over one-class SVM" (scale, high-d tolerance, parallelism; staff Q7 above is your script).

---

## 11. Coding From Scratch (NumPy only)

Linear soft-margin SVM via **Pegasos-style subgradient descent** on the primal — the version that (a) fits in an interview, (b) demonstrates hinge-loss mechanics, (c) is what scales in practice. (Say explicitly: the dual/SMO route is what libsvm does; you're implementing the primal because it's the scalable one.)

```python
import numpy as np

class LinearSVMScratch:
    """
    Primal soft-margin SVM:  min  (lam/2)||w||^2 + (1/n) Σ max(0, 1 - y·(w·x + b))
    Solved with Pegasos-style SGD (step size 1/(lam*t)).  Labels must be ±1.
    """
    def __init__(self, lam=0.01, n_epochs=50, seed=0):
        self.lam, self.n_epochs = lam, n_epochs
        self.rng = np.random.default_rng(seed)
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)            # MUST be in {-1, +1}, not {0, 1}:
        assert set(np.unique(y)) <= {-1.0, 1.0}   # hinge math breaks otherwise
        n, d = X.shape
        self.w = np.zeros(d)
        t = 0
        for _ in range(self.n_epochs):
            for i in self.rng.permutation(n):           # shuffled SGD
                t += 1
                eta = 1.0 / (self.lam * t)              # Pegasos step schedule
                margin = y[i] * (X[i] @ self.w + self.b)
                if margin < 1:
                    # Inside margin or misclassified: hinge subgradient is
                    # active -> pull w toward correcting this point.
                    self.w = (1 - eta * self.lam) * self.w + eta * y[i] * X[i]
                    self.b += eta * y[i]
                else:
                    # Beyond the margin: hinge is flat (zero gradient).
                    # ONLY regularization shrinkage applies. This branch is
                    # the 'SVM ignores easy points' property, in code.
                    self.w = (1 - eta * self.lam) * self.w
        return self

    def decision_function(self, X):
        return np.asarray(X, float) @ self.w + self.b   # signed margin, NOT a prob

    def predict(self, X):
        return np.sign(self.decision_function(X))

    def support_mask(self, X, y, tol=1e-3):
        # Diagnostic: points with y*f(x) <= 1 + tol behave as support vectors.
        return y * self.decision_function(X) <= 1 + tol
```

Narration points that earn the senior signal:
- **Labels in ±1, not 0/1** — hinge's y·f(x) algebra requires it; the classic from-scratch bug.
- **The two-branch update is the hinge loss made visible**: violators pull on w; satisfied points contribute *only* shrinkage. Point at the `else` branch and say "this flat region is why solutions are sparse in support vectors."
- **1/(λt) step schedule** — Pegasos's convergence guarantee independent of n; the reason linear SVMs scaled to web data.
- **decision_function returns geometry, not probability** — offer Platt scaling as the explicit add-on (and note it's literally Chapter 2's model on top of these scores).
- Extension if asked: kernelize by tracking α per point instead of w (predictions become Σαᵢyᵢ K(xᵢ,·)) — and name the n² cost you just bought.

---

## 12. ML System Design Perspective

**Choose SVM when:** small-n high-d with accuracy pressure (assays, niche text) — kernel SVM with nested CV; sparse text at scale needing a today-baseline — linear SVM/LR on TF-IDF; one-class novelty on small clean data; teaching/benchmark contexts.

**Avoid when:** n ≥ ~10⁵ with kernels (n² wall); probabilities consumed downstream (bolt-on calibration = standing failure mode); per-feature explanations legally required (kernel SVMs can't natively); tabular at scale (GBM), perceptual data (DL).

**Data requirements:** standardized features (non-negotiable); reasonably clean labels near the margin (boundary-adjacent label noise is maximally damaging — it mints support vectors); for kernels, n small enough to afford the Gram matrix.

**Latency:** linear — among the fastest models that exist; kernel — O(#SV·d) per score, degrading as training data grows: an unusual and nasty scaling property to flag in design reviews.

**Scale limits:** the model class whose constraint is most famously *training algorithmic complexity* rather than statistical capacity; the standard escapes (RFF/Nyström → linear) effectively convert it into a different model.

---

## 13. Resume Discussion Angle

**Recommendation/ranking/personalization:** SVMs rarely live on a modern recsys resume, but margin *losses* do. If asked "any connection between SVMs and your embedding work?": triplet/contrastive losses used to train retrieval embeddings are margin constraints — "make the positive closer than the negative by at least m" is a hinge loss in similarity space; two-tower training with margin-based or sampled-softmax losses is the SVM's intellectual descendant. That answer converts a legacy-topic question into a bridge to your strongest material (Similar Restaurants, two-tower retrieval).

**Fraud detection:** the live question is one-class SVM vs Isolation Forest (staff Q7). Your script: evaluated/considered OC-SVM-style boundaries, chose IF for scale, high-dim tolerance, and parallel training; kept the layered architecture (novelty → labels → supervised). Demonstrates you chose by engineering constraints, not fashion.

**Text/NLP systems (HelpBot):** "What was your pre-LLM baseline?" TF-IDF + linear model (SVM or LR) for intent routing — cheap, strong, and the bar the RAG system had to clear; it then survives as the drift canary and the degraded-mode fallback. Interviewers love hearing the old model kept a job.

**The universal trap:** "Why did SVMs lose to GBMs?" Don't dodge: n² kernel training, #SV-scaling inference, no native probabilities or per-feature explanations, weaker missing-data/categorical handling — versus histogram-boosted trees that scale, calibrate (with care), and explain (SHAP). Then the graceful close: the margin and kernel ideas didn't die, they moved into loss design and similarity learning. Conceding a model family's defeat with precise reasons is a credibility move, not a weakness.

---
*Previous: Gradient Boosting ← | Next: K-Nearest Neighbors →*

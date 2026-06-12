# Chapter 10: Principal Component Analysis (PCA)
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

---

## 1. Executive Summary (30 seconds)

PCA finds the orthogonal directions of maximum variance in centered data — the eigenvectors of the covariance matrix, ordered by eigenvalue — and projects data onto the top k of them. Equivalently, it finds the k-dimensional linear subspace minimizing squared reconstruction error; the two views are the same theorem. Computed in practice via SVD for numerical stability. Uses: decorrelation, compression, noise reduction, visualization, multicollinearity surgery. The interview battlegrounds: the variance-maximization derivation, centering and scaling decisions, the SVD connection, the leakage trap (fit PCA on training data only), and knowing that maximum variance ≠ maximum predictive relevance because PCA never sees the labels.

---

## 2. Interview Articulation (3–4 Minute Answer)

"PCA answers: if I'm only allowed k numbers to describe each data point instead of d, which k numbers lose the least? Its answer — for *linear* summaries under *squared* loss — is provably optimal, which is why it's survived sixty years.

Two equivalent intuitions. The **variance view**: find the direction in which the data varies most; that's the first principal component. Then the direction of maximum remaining variance orthogonal to it — the second — and so on. The **reconstruction view**: find the k-dimensional subspace such that projecting the data onto it and lifting back loses the least in squared error. These sound different but are the same optimization: total variance is fixed, so variance captured plus reconstruction error is constant — maximizing one minimizes the other. Knowing they're dual is the difference between memorizing PCA and understanding it.

Mechanically: center the data — non-negotiable, otherwise your first component points at the mean rather than along the spread. Form the covariance matrix. Its eigenvectors are the principal components; its eigenvalues are the variances along them. Project onto the top-k eigenvectors and you've compressed d dimensions to k, with explained-variance ratios telling you exactly what you kept. The derivation interviewers ask for: maximize wᵀΣw subject to unit norm; the Lagrangian gives Σw = λw — an eigenvector equation, with the variance achieved equal to the eigenvalue itself. That's a ninety-second whiteboard derivation; have it cold.

In practice nobody eigendecomposes the covariance matrix — you take the **SVD of the centered data matrix** directly. The right singular vectors are the components, singular values squared over n−1 are the eigenvalues. SVD is numerically stabler — you never square the data's condition number by forming XᵀX — and truncated/randomized SVD scales to huge matrices. 'PCA is SVD of centered data' is a sentence worth saying verbatim.

Decisions that matter. **Scaling**: PCA chases variance, and variance has units — leave revenue-in-rupees next to a 0-to-1 ratio and the first component is just 'revenue'. Standardizing first means PCA on the correlation matrix, the default for heterogeneous features; skipping it is defensible only when units are shared and magnitude differences are meaningful, like pixels or spectra. **Choosing k**: cumulative explained variance (90–95% rules of thumb), the scree elbow, or — if PCA feeds a supervised model — cross-validate k against the downstream metric, which is the only criterion that actually answers the question being asked.

The two failure modes to volunteer. First, **PCA is unsupervised**: it preserves variance, not class separation — the informative signal can live in a low-variance direction that PCA throws away, which is the standard exam trap and the reason supervised alternatives like LDA or PLS exist. Second, **leakage**: PCA is a fitted transform — fit on training data only, apply to test; fitting on the full dataset leaks test-set structure into your features, and it's one of the most common silent pipeline bugs.

Production roles: multicollinearity surgery before linear models, decorrelating and denoising features, compressing embeddings before ANN indexing — recall barely moves, RAM halves — visualization, and as the linear baseline that autoencoders must beat. When not to use: non-linear manifolds (kernel PCA, UMAP, autoencoders), when interpretability of original features is required downstream — components are blends — and sparse count data, where truncated SVD without centering, that is LSA, preserves sparsity."

---

## 3. Mathematical Foundation

**Setup:** X ∈ ℝ^(n×d), centered (column means subtracted). Covariance Σ = XᵀX/(n−1).

**Variance-maximization derivation (the must-know):**
```
max_w  wᵀΣw   s.t. ‖w‖ = 1
L = wᵀΣw − λ(wᵀw − 1);   ∂L/∂w = 2Σw − 2λw = 0  ⟹  Σw = λw
```
⟹ w is an eigenvector of Σ; achieved variance = wᵀΣw = λ. Maximum variance ⟹ top eigenvector. Subsequent components: same problem with orthogonality constraints ⟹ remaining eigenvectors in eigenvalue order. (Σ symmetric PSD ⟹ real non-negative eigenvalues, orthogonal eigenvectors — say why the decomposition is guaranteed.)

**Reconstruction duality (Eckart–Young flavor):**
```
min over k-dim subspaces of  Σᵢ ‖xᵢ − P xᵢ‖²   (P = orthogonal projector)
Total variance = Σⱼ λⱼ = variance captured + reconstruction error
```
⟹ minimizing reconstruction error ⟺ maximizing captured variance ⟹ same top-k eigenvectors. The optimal rank-k *linear* compression under squared loss — optimality is the selling point.

**SVD computation:** X = UΣ_s Vᵀ (centered X).
```
XᵀX = V Σ_s² Vᵀ   ⟹   components = columns of V;  eigenvalues λⱼ = σⱼ²/(n−1)
Scores (projected data) = XV = UΣ_s
```
Why SVD over eig(XᵀX): forming XᵀX squares the condition number (precision loss for ill-conditioned X); SVD works on X directly; randomized/truncated SVD gives top-k in O(ndk) — the scalable path.

**Explained variance ratio:** EVRⱼ = λⱼ / Σλ. Cumulative EVR drives the k decision.

**Scaling = correlation-matrix PCA:** standardize columns ⟹ Σ becomes the correlation matrix; components now unit-free. Rule: heterogeneous units ⟹ standardize; homogeneous, magnitude-meaningful units (pixels, spectra) ⟹ optionally don't.

**Whitening:** transform z = Λ^(−1/2) Vᵀ x ⟹ identity covariance (decorrelated *and* unit variance). Used before algorithms assuming isotropy (some metric learners, ICA preprocessing); amplifies noise in tiny-λ directions — drop them first.

**Probabilistic PCA (one paragraph to know):** x = Wz + μ + ε, z~N(0,I), ε~N(0,σ²I); ML solution recovers principal subspace, with σ² = mean of discarded eigenvalues. Buys: likelihoods, EM for missing data, the bridge to factor analysis (diagonal instead of isotropic noise) — and the right name-drop when asked "PCA with missing values?"

**Kernel PCA (one paragraph):** eigendecompose the centered kernel matrix K instead of Σ — components in an implicit feature space; non-linear structure at O(n²) cost and no straightforward inverse map. Mention; rarely production.

---

## 4. Step-by-Step Numerical Example

Five points in 2-D (already convenient): (2,1), (3,2), (4,3), (5,4), (6,5) plus noise — take exactly: x = (2,3,4,5,6), y = (1,2,3,4,5).

**Center:** x̄ = 4, ȳ = 3 ⟹ centered pairs: (−2,−2), (−1,−1), (0,0), (1,1), (2,2).

**Covariance matrix (n−1 = 4):**
```
Var(x) = (4+1+0+1+4)/4 = 2.5 = Var(y);  Cov(x,y) = (4+1+0+1+4)/4 = 2.5
Σ = [2.5 2.5; 2.5 2.5]
```
**Eigen:**
```
det(Σ − λI) = (2.5−λ)² − 2.5² = 0 ⟹ λ² − 5λ = 0 ⟹ λ₁ = 5, λ₂ = 0
λ₁=5: (Σ−5I)w = 0 ⟹ w₁ = (1,1)/√2;   λ₂=0 ⟹ w₂ = (1,−1)/√2
```
**Read-out:** EVR₁ = 5/5 = 100% — the data is perfectly 1-D along the diagonal (it was y = x − 1). Projection of (6,5): centered (2,2) ⟹ score = (2,2)·(1,1)/√2 = 4/√2 ≈ 2.83; reconstruction = 2.83·w₁ + mean = (6,5) exactly — zero error, as λ₂ = 0 promised.

Now perturb one point — replace (4,3) with (4, 3.4): Cov(x,y) drops slightly, λ₂ becomes small-but-positive, and PC2 = the noise direction; discarding it *is* denoising. This tiny example demonstrates centering, eigen-by-hand, EVR, projection, reconstruction, and the noise-truncation story in one pass — it's whiteboard-complete.

---

## 5. Hyperparameters

PCA has few knobs, but each is a judgment call interviewers probe:

| Choice | What it does | One way | Other way | Interview probe |
|---|---|---|---|---|
| n_components (k) | Subspace dimension | k ↑: more variance kept, less compression | k ↓: aggressive compression, info loss | "How do you pick k?" → EVR threshold / scree as descriptive; **CV on downstream metric** when PCA feeds a model — the only honest criterion for that case |
| Standardize first? | Covariance vs correlation PCA | Yes: unit-free, default for mixed units | No: variance-in-original-units, defensible for homogeneous features | "What happens if you skip scaling?" → biggest-unit feature = PC1, full stop |
| Center? | — | Always for PCA | Skipping ⟹ PC1 ≈ direction of the mean — it's no longer PCA | "TruncatedSVD vs PCA in sklearn?" → no centering (sparsity-preserving) = LSA |
| whiten | Unit-variance scores | Isotropic outputs for downstream | Keeps natural scale | "Risk of whitening?" → noise blow-up along tiny-λ components |
| svd_solver | full / randomized / arpack | randomized: O(ndk), the big-data path | full: exact, small d | "PCA on 10⁶×10⁴?" → randomized SVD, mention Halko et al. for flavor |
| Sign convention | Eigenvector signs are arbitrary | — | — | "Your PC1 flipped sign after refit — bug?" → no; fix a convention (largest-|loading| positive) for stability of downstream consumers |

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | Full SVD O(min(n²d, nd²)); randomized truncated SVD O(ndk) — top-50 components of 10⁶×10⁴ in minutes |
| Inference | One matrix multiply: O(dk) per row — microseconds; fuse with the model's first layer if you like |
| Memory | Components V: d×k floats + means (+ scales). Tiny artifact; version it with the model |
| Streaming | Incremental PCA (mini-batch updates) for data that won't fit, or refit on rolling windows |
| Train/serve skew | The classic bug class: serving must apply the *training* means/scales/V — recompute any of them online and scores silently shift. Package center+scale+project as one versioned transform |
| Monitoring | EVR drift across refits (structure changing), reconstruction-error distribution on live traffic (off-manifold inputs — doubles as an anomaly signal), score distribution PSI per component |
| Leakage discipline | Fit on train folds only — inside the CV loop, not before it. PCA-before-split inflates offline metrics; it's the unsupervised cousin of target leakage |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What does PCA optimize?** Two equivalent objectives: maximize variance of projections / minimize squared reconstruction error onto a k-dim linear subspace. Solution: top-k eigenvectors of the covariance matrix.
2. **Why must you center the data?** Covariance is defined around the mean; uncentered "PCA" makes the first component point toward the data mean rather than along the spread — you'd be decomposing second moments, not covariance.
3. **When do you standardize before PCA?** Mixed units/scales — otherwise variance comparisons are unit artifacts and the largest-scale feature owns PC1. Skip only for homogeneous, magnitude-meaningful features (pixels, spectra, same-currency amounts).
4. **What are eigenvalues/eigenvectors here?** Eigenvectors of Σ = the principal directions; eigenvalues = variance along each. EVR = λⱼ/Σλ quantifies each component's share.
5. **How do you choose the number of components?** Cumulative EVR threshold (90–95%), scree elbow, Kaiser-type rules — descriptive; if PCA feeds a supervised model, cross-validate k on the downstream metric (the criterion that matches the goal).
6. **Are principal components interpretable?** They're linear blends; inspect loadings for narrative ("PC1 ≈ overall spend level, PC2 ≈ weekday-vs-weekend mix") but resist over-reading — rotation methods (varimax) trade optimality for interpretability if stakeholders need names.
7. **PCA vs feature selection?** PCA constructs new features (all originals contribute — nothing is "removed"); selection keeps a subset (interpretability, cheaper pipelines, drops measurement cost). Different goals; selection when original-feature meaning must survive.
8. **Is PCA supervised? Consequence?** No — it never sees y. Max-variance directions can be predictively useless and low-variance directions decisive (classic counterexample: two classes separated along a tiny-variance axis). Supervised cousins: LDA, PLS.
9. **PCA and multicollinearity?** Components are orthogonal by construction ⟹ regressing on scores kills multicollinearity (principal components regression). Caveat: PCR keeps high-variance components, not high-relevance ones — PLS fixes that by using y.
10. **The relationship between PCA and SVD?** PCA = SVD of the centered data matrix: V = components, σ²/(n−1) = eigenvalues, UΣ_s = scores. SVD is the numerically stable, scalable computation; nobody forms Σ at scale.
11. **Can PCA reduce overfitting?** Indirectly — fewer, denoised, decorrelated inputs shrink model variance; but it can also delete signal (unsupervised). It's preprocessing, not regularization proper; compare against L1/L2 on the full features.
12. **PCA with categorical features?** Not directly meaningful (variance of one-hots = frequency artifacts). Options: MCA/FAMD for categorical/mixed data, or embed categoricals first, or skip PCA for tree models that don't need it.
13. **How does PCA denoise?** Signal concentrates in top components, isotropic noise spreads across all; truncating small-λ components removes mostly noise. Reconstruction from top-k = the denoised data (eigenfaces logic).
14. **Does PCA help tree models?** Usually not — trees handle correlated/unscaled features natively, and axis-aligned splits on rotated (blended) components can be *harder*. PCA mainly serves linear/distance-based models and compression goals.
15. **What's the difference between sklearn's PCA and TruncatedSVD?** Centering. TruncatedSVD skips it to preserve sparsity (centering densifies sparse matrices) — on TF-IDF that's LSA. On dense centered data they coincide.

### Advanced (15)

1. **Full Lagrangian derivation of PC1, and why PC2 is the second eigenvector.** As §3; for PC2 add constraint wᵀw₁ = 0 — a second multiplier; stationarity again gives Σw = λw restricted to the orthogonal complement ⟹ next eigenvector. Deflation view: subtract λ₁w₁w₁ᵀ and repeat.
2. **Prove the variance/reconstruction duality.** ‖x‖² = ‖Px‖² + ‖x−Px‖² (Pythagoras, orthogonal projection); sum over data: total variance (fixed) = captured + error ⟹ argmax captured = argmin error. Two lines; very high signal.
3. **Why is computing PCA via XᵀX numerically inferior to SVD?** cond(XᵀX) = cond(X)² — small singular values get squared into rounding-error territory; SVD factors X directly, preserving them. (Also memory: never materialize d×d for huge d.)
4. **Randomized SVD — the idea in three sentences.** Multiply X by a random Gaussian d×(k+p) matrix to sample its range; orthonormalize (QR) to get a basis Q capturing top directions w.h.p.; SVD the small matrix QᵀX and lift back. O(ndk), a couple of power iterations sharpen accuracy on slowly-decaying spectra.
5. **PCA's optimality is under squared loss and linearity — name what breaks when each is relaxed.** Non-squared loss (robust PCA: L1/low-rank+sparse decomposition for outlier-contaminated data); non-linear structure (kernel PCA, autoencoders, UMAP — PCA flattens manifolds: the Swiss roll collapses).
6. **When is PC1 ≈ "size" and PC2 ≈ "shape"?** Positively correlated feature blocks (all-spend metrics): PC1 = common level (all-positive loadings), PC2 = contrasts within. Recognizing this pattern instantly when reading loadings is an applied-credibility marker.
7. **n ≪ d (e.g., 100 samples × 20K genes): what's true about PCA?** At most n−1 nonzero eigenvalues; compute via the n×n Gram matrix XXᵀ (dual PCA) instead of d×d covariance; eigenvectors lifted by Xᵀu/σ. Also: severe overfitting risk in any downstream model — components are noisy estimates at this n.
8. **Probabilistic PCA — model, what σ² ends up being, and what it buys.** x = Wz + μ + ε; ML W spans the principal subspace; σ̂² = mean of discarded eigenvalues. Buys: likelihood for model selection, EM handling missing data natively, mixtures of PPCA, the factor-analysis bridge (diagonal noise ⟹ FA).
9. **Whitening: definition, use, danger.** z = Λ^(−1/2)Vᵀx ⟹ Cov(z) = I. Use: algorithms assuming isotropy; ICA preprocessing. Danger: dividing by √λⱼ explodes near-zero-variance (noise) directions — truncate before whitening.
10. **Why are eigenvector signs (and near-degenerate components) unstable across refits, and what's the production fix?** Signs are mathematically arbitrary; close eigenvalues ⟹ the eigenvectors within that subspace rotate freely under resampling noise. Fix: deterministic sign convention, monitor subspace (principal angles) rather than individual components, and freeze V between scheduled refits.
11. **PCA as matrix factorization — place it in the family.** X ≈ scores · Vᵀ: rank-k factorization with orthogonality, optimal under squared loss on a *complete* matrix (Eckart–Young). Relatives: NMF (non-negativity, parts-based), sparse PCA (interpretable loadings), recsys MF (incomplete matrices — Ch. 11's core distinction).
12. **A linear autoencoder with squared loss — what does it learn relative to PCA?** The same subspace (not necessarily the orthonormal eigenbasis — weights can be any basis of it). Non-linear activations/depth generalize PCA; PCA is the exactly-solvable special case and the sanity baseline every AE must beat.
13. **How does PCA interact with outliers, and what is Robust PCA?** Variance is squared-loss — outliers dominate directions. RPCA: decompose X = L (low-rank) + S (sparse outliers) via nuclear-norm + L1 convex program — the principled fix; practical alternatives: winsorize first, or use elementwise-robust scalers.
14. **Functional/structured PCA in time series — what changes?** Apply to curves (FPCA: eigenfunctions of the covariance kernel) or lag-embedded matrices (SSA); components become temporal patterns (trend/seasonal shapes). One sentence here signals breadth beyond tabular.
15. **Explain principal angles and when you'd monitor them.** Angles between subspaces spanned by two component sets — the right way to compare "did the structure change?" across refits, robust to rotation/sign indeterminacy that makes naive component-wise diffs meaningless.

### Staff-Level (10)

1. **A teammate fit PCA on the full dataset before the train/test split and reports a 2-point lift. Walk through your handling.** Mechanism: test rows shaped the components ⟹ test information in training features ⟹ inflated estimate (magnitude depends on n, d, spectrum — can be negligible or large; quantify by refitting properly). Fix the pipeline (PCA inside CV folds), re-report, and institutionalize: transforms-as-fitted-estimators in a pipeline object so the bug class dies, not just the bug.
2. **You compress 768-d embeddings to 128-d with PCA before an ANN index. What do you measure and what can go wrong?** Measure end-task: recall@k vs full-dim (per segment), latency/RAM savings, downstream CTR if available. Failure modes: anisotropy mismatch (inner-product search after L2-trained PCA — decide metric first), tail-item recall loss (their variance directions truncated), encoder updates invalidating V (version V with the encoder, like IVF codebooks). The senior point: EVR is not the metric; **recall@k is**.
3. **Stakeholders want "the 5 factors driving customer behavior" from your PCA. Where's the line between insight and fiction?** Loadings describe variance structure, not causal drivers; rotation choices change the story; near-degenerate eigenvalues mean the "factors" are rotationally arbitrary. Offer: stability analysis (bootstrap loadings), varimax-rotated descriptive labels clearly framed as descriptive, and redirect causal language toward experiments. Naming the rotational indeterminacy is the staff-level honesty.
4. **Design the versioning/monitoring for a PCA transform feeding a production credit model.** Artifact = (means, scales, V, k, training-window hash) versioned atomically with the model; golden-row parity tests across train/serve implementations; monitors: input PSI, score PSI per component, reconstruction-error tail (off-manifold alarm), principal-angle drift across refits with change-review gates (regulated model — silent feature redefinition is a compliance incident, say it).
5. **Your fraud model's PCA preprocessing shows rising reconstruction error on live traffic. Interpret and act.** Inputs are leaving the training manifold: drift, new product surface, instrumentation bug, or adversarial probing. Triage by segment/feature attribution of the error; short-term: alert + fallback to raw-feature challenger; medium: refit window decision. Bonus point: reconstruction error is itself a serviceable anomaly score — you've been running an unsupervised detector for free.
6. **When would you choose PLS or LDA over PCA, concretely?** When the downstream task is supervised and you suspect signal in low-variance directions: PLS maximizes covariance with y (regression, chemometrics-style p≫n); LDA maximizes class separation (classification, k−1 components max). PCA when the goal is representation/compression task-agnostic, or labels are unreliable. Choosing by *goal*, stated crisply, is the answer.
7. **A 10⁹×10⁵ sparse interaction matrix needs "PCA." What do you actually run?** Not PCA — centering densifies. TruncatedSVD/LSA on the sparse matrix (randomized, implicit mean if needed via rank-1 correction), or go straight to the honest model: implicit-feedback MF/ALS (Ch. 11), since the zeros aren't Gaussian noise — they're missingness. Recognizing "this stopped being a PCA problem" is the point.
8. **How do you defend (or kill) a PCA step during a model-simplification review?** Ablate: full features + regularization vs PCA-k vs selection-k on the temporal holdout, including latency/memory/maintenance deltas; check whether modern model (GBM) even benefits (usually not); if PCA survives only as multicollinearity therapy for a linear model, consider ridge as the simpler equivalent. Default position: every transform must pay rent in measured metric or measured cost.
9. **Incremental PCA for a streaming feature platform — design and the failure mode nobody monitors.** Mini-batch updates of mean and subspace (or scheduled randomized-SVD refits on windows); the unmonitored failure: slow subspace rotation silently redefines features for *all* downstream consumers — scores drift with no model change. Fix: principal-angle drift alarms + explicit re-versioning events that trigger downstream revalidation, never silent continuous rotation in place.
10. **Interviewer: "Autoencoders make PCA obsolete." Your take?** Disagree with precision: PCA is exactly solvable, deterministic, data-cheap, interpretable-ish, microsecond serving, and *is* the linear AE optimum — the right tool at small data, linear structure, or as the baseline that quantifies what non-linearity buys. AEs win on genuinely non-linear manifolds with enough data and tuning budget. "Obsolete" claims that ignore cost structure are how teams overspend; the measured comparison is the answer.

---

## 8. Comparison Section

| | PCA | LDA | Autoencoder | t-SNE/UMAP | Truncated SVD (LSA) |
|---|---|---|---|---|---|
| Supervised | No | Yes (labels) | No | No | No |
| Objective | Max variance / min recon. | Max class separation | Min reconstruction (non-linear) | Preserve neighborhoods | Low-rank approx. (no centering) |
| Linear | Yes | Yes | No | No | Yes |
| Out-of-sample transform | Trivial (V) | Trivial | Forward pass | Awkward (re-fit/approx) | Trivial |
| Use | Compression/decorrelation | Pre-classifier reduction | Rich non-linear codes | **Visualization only** | Sparse text/interactions |
| Determinism | Yes (up to sign) | Yes | No (init/SGD) | No | Yes |

**PCA vs Autoencoder one-liner:** "A linear autoencoder under squared loss converges to PCA's subspace — so the AE is justified exactly when non-linearity demonstrably beats that baseline."

**PCA vs t-SNE/UMAP:** PCA preserves global variance structure and gives a reusable transform; t-SNE/UMAP preserve local neighborhoods for *plots* — never feed their coordinates into downstream models or read cluster sizes/distances literally. Saying "UMAP is for eyes, PCA is for pipelines" lands.

**PCA vs SVD vs MF (the Ch. 11 bridge):** PCA = SVD of centered complete data with statistical interpretation; truncated SVD = same algebra, sparsity-preserving; recsys "SVD" = factorization of an *incomplete* matrix by SGD/ALS over observed entries — a different problem wearing the same name. Drawing this triangle cleanly is precisely what next chapter weaponizes.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Skipping centering/scaling in the explanation — the first follow-up will expose it.
- Unable to produce the Lagrangian derivation or the duality argument.
- "PCA removes unimportant features" — it removes *directions*; every original feature lives on in the loadings.
- Treating EVR as predictive value (the unsupervised trap).
- Not knowing PCA = SVD of centered data, or why SVD is preferred computationally.

**Production mistakes:**
- Fitting PCA before the split / outside the CV loop (leakage).
- Recomputing means/scales/V at serving time (train/serve skew — scores silently shift).
- No sign convention or subspace monitoring — downstream features flip/rotate across refits with no alarms.
- Whitening without truncating near-zero components (noise amplification).

**Modeling mistakes:**
- PCA before tree models "by habit" — usually neutral-to-harmful.
- PCR when PLS was the fit (keeping variance instead of relevance).
- Reading varimax-rotated "factors" as causal drivers to stakeholders.
- Centering a giant sparse matrix (densification OOM) instead of TruncatedSVD.

---

## 10. Real Industry Use Cases

- **Google** — embedding compression and whitening in retrieval stacks; PCA baselines in representation-learning papers; historical eigen-methods lineage (PageRank is a cousin eigenproblem — fun aside, don't overclaim).
- **Amazon** — feature decorrelation in risk/forecast linear models; PCA on operational sensor/telemetry data for anomaly triage via reconstruction error.
- **Netflix** — Prize-era lineage: SVD/MF on ratings (Ch. 11); today: compressing content/user embeddings, exploratory structure analysis on viewing features.
- **Meta** — PCA/whitening as preprocessing in embedding pipelines and FAISS workflows (OPQ — optimized product quantization — starts with a PCA-like rotation; a deep-cut tie to Ch. 9's PQ story).
- **Uber** — telemetry/GPS feature compression; PCA-based anomaly detection on trip-sensor streams; decorrelation in pricing/forecast models.
- **Swiggy/Zomato** — compressing restaurant/user embeddings before ANN serving (the §7-S2 scenario — RAM and latency wins with recall@k as the gate); collapsing correlated funnel metrics into composite indices for dashboards. *Naveen: the embedding-compression story plus "we gated on recall@k, not EVR" is a strong, concrete systems answer from your own stack.*
- **Flipkart** — catalog/image-embedding compression; multicollinearity surgery in demand/elasticity regressions feeding pricing.
- **Games24x7** — player-behavior feature decorrelation before risk/LTV linear models; reconstruction-error anomaly signals on gameplay telemetry as an unsupervised fraud tripwire complementing Isolation Forest.

---

## 11. Coding From Scratch (NumPy only)

PCA via SVD — the numerically correct way — with transform, inverse transform, and EVR.

```python
import numpy as np

class PCAScratch:
    def __init__(self, n_components=None, standardize=False):
        self.k = n_components
        self.standardize = standardize
        self.mean_ = self.scale_ = self.components_ = None
        self.explained_variance_ = self.evr_ = None

    def fit(self, X):
        X = np.asarray(X, float)
        n, d = X.shape

        # 1) Center — non-negotiable. (Optionally scale ⟹ correlation PCA.)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        if self.standardize:
            self.scale_ = Xc.std(axis=0, ddof=1)
            self.scale_[self.scale_ == 0] = 1.0      # guard constant columns
            Xc = Xc / self.scale_

        # 2) SVD of the centered data — NOT eig(X^T X):
        #    forming X^T X squares the condition number; SVD is stable.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # 3) Sign convention: make each component's largest-|loading| positive.
        #    Eigenvector signs are arbitrary; fixing them stabilizes pipelines.
        flip = np.sign(Vt[np.arange(len(S)), np.abs(Vt).argmax(axis=1)])
        Vt = Vt * flip[:, None]

        k = self.k or min(n, d)
        self.components_ = Vt[:k]                     # (k, d): rows are PCs
        self.explained_variance_ = (S[:k] ** 2) / (n - 1)   # λ_j = σ_j²/(n−1)
        total_var = (S ** 2).sum() / (n - 1)
        self.evr_ = self.explained_variance_ / total_var
        return self

    def transform(self, X):
        Xc = np.asarray(X, float) - self.mean_        # apply TRAINING mean —
        if self.standardize:                          # this line is the
            Xc = Xc / self.scale_                     # train/serve-skew defense
        return Xc @ self.components_.T                # scores: (n, k)

    def inverse_transform(self, Z):
        Xc = np.asarray(Z, float) @ self.components_  # lift back to d-dim
        if self.standardize:
            Xc = Xc * self.scale_
        return Xc + self.mean_

    def reconstruction_error(self, X):
        # Per-row squared error of project-then-lift —
        # doubles as an off-manifold / anomaly score in production.
        R = self.inverse_transform(self.transform(X)) - np.asarray(X, float)
        return (R ** 2).sum(axis=1)
```

Narration points that earn senior credit:
- **`svd(Xc)` not `eig(Xc.T @ Xc)`** — state the condition-number-squaring reason unprompted; it's the single highest-signal line in the file.
- **λⱼ = σⱼ²/(n−1)** — converting singular values to variances correctly (the off-by-ddof bug is common).
- **Sign-convention block** — explain it exists for *pipeline stability across refits*, a production concern most candidates never mention.
- **`transform` reuses training mean/scale** — point at it and say "this is where PCA leakage and train/serve skew are both prevented."
- **`reconstruction_error` as anomaly score** — connects the class to the monitoring story (§6) in one method.
- Extensions to offer: randomized SVD for huge X (range-finding sketch), incremental fit for streams, and "drop `mean_` handling and this becomes TruncatedSVD/LSA for sparse matrices."

---

## 12. ML System Design Perspective

**Choose PCA when:** compressing dense embeddings/features for RAM- or latency-bound serving (gate on end-task metrics); decorrelating inputs for linear/distance-based models; denoising before clustering or visualization; an exactly-solvable, deterministic baseline for representation learning; reconstruction-error anomaly tripwires.

**Avoid when:** downstream is tree-based (usually no benefit, hurts interpretability); structure is non-linear and data is plentiful (AE/UMAP-class tools); original-feature semantics must survive (regulatory reason codes — use selection); sparse count matrices (TruncatedSVD/MF instead).

**Data requirements:** centered (and usually standardized) numeric features; enough n relative to d for stable components (n ≪ d ⟹ dual/Gram computation and humility about noise); outlier policy first (or robust variants).

**Latency:** a single (d×k) matmul — effectively free; the artifact is KBs. PCA is never the serving bottleneck; its risks are pipeline-correctness risks (skew, versioning), not compute.

**Scale limits:** randomized SVD handles 10⁶–10⁹ cells routinely; sparse data routes to TruncatedSVD; the real constraints are statistical (component stability) and organizational (versioned-transform discipline across consumers).

---

## 13. Resume Discussion Angle

**Recommendation/retrieval systems:** Your strongest concrete story: **embedding compression before ANN serving** — "we PCA'd 768-d to 128-d, gated the change on recall@k per segment rather than explained variance, and versioned the projection with the encoder exactly like the IVF codebooks." That sentence chains Ch. 9 + Ch. 10 into one infrastructure narrative and preempts the EVR-vs-end-metric trap. Expected follow-ups: metric choice (IP vs L2 before/after projection), tail-item recall, OPQ as the PQ-aware rotation upgrade.

**Fraud detection:** Two angles: decorrelation before any linear risk layer, and **reconstruction error as an unsupervised tripwire** — off-manifold transactions/gameplay flagged before any label exists, feeding the same triage queue as Isolation Forest. Expect "PCA anomaly vs Isolation Forest?" → linear-manifold deviations vs partition-depth isolation; complementary detectors, ensemble their alerts.

**Personalization/marketing ML:** Composite indices from correlated funnel metrics (loadings-based "engagement index") for dashboards and as model features — with the stability caveat (bootstrap loadings, principal-angle monitoring) volunteered before anyone asks. The volunteered caveat is what separates "ran sklearn" from "owned the artifact."

**The universal trap:** "Why did you use PCA here?" The losing answer cites EVR. The winning shape: the *downstream* metric or cost that improved (recall@k held at half the RAM; ridge coefficients stabilized; p99 dropped 2ms), the leakage/skew discipline that made it safe, and the ablation that proved it paid rent. PCA questions at senior level are pipeline-judgment questions in disguise — answer them at that altitude.

---
*Previous: K-Means ← | Next: Singular Value Decomposition →*

# Chapter 11: Singular Value Decomposition (SVD)
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

> The pivot chapter: SVD is the linear-algebra bedrock under PCA, LSA, and — most importantly for your resume — recommender matrix factorization. The single most-tested confusion here is "SVD" (the exact factorization of a complete matrix) versus "SVD for recsys" (factorizing an *incomplete* matrix by optimization). Nail that distinction and half the chapter answers itself.

---

## 1. Executive Summary (30 seconds)

SVD factors any real matrix A (m×n) into UΣVᵀ: orthonormal left singular vectors U, non-negative singular values Σ (diagonal, descending), orthonormal right singular vectors Vᵀ. Geometrically, every linear map is a rotation, a scaling, and another rotation. Truncating to the top-k singular triples gives the provably optimal rank-k approximation under both Frobenius and spectral norms (Eckart–Young). That one theorem powers PCA, LSA/topic models, image compression, pseudoinverses, and least squares. In recommenders, "SVD" is borrowed loosely: the rating matrix is mostly missing, so you can't run classical SVD — you *learn* low-rank factors by minimizing error over observed entries, which is matrix factorization, not the decomposition. Interviews test the algebra, Eckart–Young, and that complete-vs-incomplete distinction.

---

## 2. Interview Articulation (3–4 Minute Answer)

"SVD is the statement that every linear transformation, no matter how messy the matrix looks, is fundamentally three clean operations: rotate, stretch along axes, rotate again. Formally any m-by-n matrix A factors as U-Sigma-V-transpose, where U and V are orthonormal — pure rotations or reflections — and Sigma is diagonal with non-negative entries, the singular values, in descending order. The singular values tell you how much the map stretches space along each principal direction; the columns of V are the input directions, the columns of U the corresponding output directions.

Where it connects to everything else: the right singular vectors V are the eigenvectors of AᵀA, the left singular vectors U are the eigenvectors of AAᵀ, and the singular values are the square roots of their shared eigenvalues. So PCA *is* SVD of the centered data matrix — the right singular vectors are the principal components, singular values squared over n-1 are the variances. You compute PCA via SVD precisely because you avoid ever forming AᵀA, which would square the condition number and destroy precision on ill-conditioned data.

The crown jewel is the **Eckart–Young theorem**: if you keep only the top k singular triples and zero the rest, you get the *best possible* rank-k approximation of A — provably, under the Frobenius norm and the spectral norm. No other rank-k matrix is closer. That's why truncated SVD is the optimal linear compressor: image compression keeps the top singular values of the pixel matrix; LSA — latent semantic analysis — runs truncated SVD on a term-document matrix to fold thousands of sparse word counts into a few hundred dense 'topic' dimensions, which was the ancestor of word embeddings. The error of the truncation is exactly the energy in the discarded singular values — you can read your compression loss straight off the spectrum.

It also fixes degenerate linear algebra. The Moore–Penrose pseudoinverse is V-Sigma-plus-U-transpose, where Sigma-plus inverts the nonzero singular values — that's how you solve least squares when XᵀX is singular, and truncated SVD is a regularizer because dropping tiny singular values stops you from dividing by near-zero and amplifying noise.

Now the recsys twist, because this is where interviewers catch people. The Netflix-Prize-era 'SVD' models are *not* the SVD I just described. Classical SVD requires a complete matrix — but a rating matrix is 99% missing, and you cannot decompose a matrix full of holes; zero-filling is wrong because a missing rating isn't a zero rating. So what people call 'SVD' in recommenders is actually **matrix factorization**: posit that the rating matrix is approximately the product of a user-factor matrix and an item-factor matrix, then *learn* those factors by minimizing squared error over only the observed entries, with regularization, plus bias terms, via SGD or alternating least squares. It gives you the same low-rank structure SVD would, but obtained by optimization over observed data rather than exact decomposition of a full matrix. Saying that distinction crisply — 'classical SVD decomposes a complete matrix in closed form; recsys MF learns a low-rank factorization over observed entries of an incomplete one' — is the single highest-value sentence in this whole topic.

When SVD shines: it's exact, optimal, and parameter-free for complete matrices, and truncated SVD scales via randomized methods. When it doesn't fit: genuinely incomplete matrices (use MF), non-linear structure (autoencoders), non-negativity or interpretability requirements (NMF), or implicit feedback where zeros mean 'unobserved,' not 'disliked' — which needs weighted MF, not SVD.

The traps: zero-filling a sparse matrix and calling it SVD; forgetting Eckart–Young is the reason truncation is principled rather than heuristic; and conflating singular values with eigenvalues — they're square roots of the eigenvalues of AᵀA, and singular values are always non-negative while eigenvalues can be negative or complex."

---

## 3. Mathematical Foundation

**The decomposition:** for A ∈ ℝ^(m×n), rank r:
```
A = U Σ Vᵀ
  U ∈ ℝ^(m×m) orthonormal (left singular vectors, eigenvectors of AAᵀ)
  Σ ∈ ℝ^(m×n) diagonal, σ₁ ≥ σ₂ ≥ … ≥ σ_r > 0 = … = 0
  V ∈ ℝ^(n×n) orthonormal (right singular vectors, eigenvectors of AᵀA)
```
**Relation to eigendecomposition:**
```
AᵀA = V Σ² Vᵀ      AAᵀ = U Σ² Uᵀ      σᵢ = √λᵢ(AᵀA)
```
Singular values are non-negative real always (AᵀA is symmetric PSD); eigenvalues of a general matrix need not be. SVD exists for *every* matrix; eigendecomposition only for diagonalizable square ones.

**Outer-product (the intuition) form:**
```
A = Σᵢ σᵢ uᵢ vᵢᵀ      — a sum of rank-1 layers, each weighted by its singular value
```
Truncating after k terms keeps the k heaviest layers.

**Eckart–Young theorem (the must-know):**
```
A_k = Σ_{i=1}^{k} σᵢ uᵢ vᵢᵀ  minimizes  ‖A − B‖  over all rank-k B,
both for Frobenius and spectral norms.
Frobenius error:  ‖A − A_k‖_F = √(Σ_{i>k} σᵢ²)      Spectral error: ‖A − A_k‖₂ = σ_{k+1}
```
The discarded singular values' energy *is* the reconstruction error — compression loss readable from the spectrum.

**PCA connection:** centered X = UΣVᵀ ⟹ components = columns of V, scores = UΣ, eigenvalues λⱼ = σⱼ²/(n−1). (Ch. 10 in one line.)

**Pseudoinverse & least squares:**
```
A⁺ = V Σ⁺ Uᵀ   (Σ⁺ inverts nonzero σ's, zeros the rest)
min ‖Ax − b‖²  ⟹  x = A⁺b      (minimum-norm solution when underdetermined/singular)
```
Truncated-SVD pseudoinverse (drop tiny σ) = regularization: avoids dividing by near-zero, controls noise amplification — the bridge to ridge (ridge shrinks each 1/σ toward 0 smoothly: σ/(σ²+λ)).

**Randomized SVD (the scalable computation):** sketch A's range with a random projection AΩ, orthonormalize (QR → Q), then SVD the small QᵀA and lift. O(mnk) for top-k; power iterations sharpen on slowly-decaying spectra. This is how truncated SVD/LSA runs at scale.

**Recsys matrix factorization (the *different* problem):**
```
Classical SVD:   needs COMPLETE A; closed-form; orthonormal factors.
Recsys "SVD":    R is INCOMPLETE (only observed set Ω). Learn P (users×k), Q (items×k):
    min_{P,Q,b}  Σ_{(u,i)∈Ω} (r_ui − μ − b_u − b_i − pᵤᵀqᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖² + b_u² + b_i²)
    optimize by SGD or ALS over OBSERVED entries only; factors NOT orthonormal.
```
This is Simon Funk's "SVD" from the Netflix Prize — a regularized low-rank regression, not a decomposition. Implicit feedback (clicks, not ratings) ⟹ weighted MF (Hu-Koren-Volinsky): treat all entries as observed with confidence weights, zeros = low-confidence negatives, solved by weighted ALS.

---

## 4. Step-by-Step Numerical Example

A small SVD by hand (the standard whiteboard sequence):
```
A = [3 0; 0 -2]   (already nearly diagonal, to keep arithmetic clean)
AᵀA = [9 0; 0 4]  ⟹ eigenvalues 9, 4 ⟹ σ₁ = 3, σ₂ = 2
V from AᵀA eigenvectors: e₁=(1,0), e₂=(0,1) ⟹ V = I
U columns: uᵢ = Avᵢ/σᵢ:
   u₁ = A(1,0)/3 = (3,0)/3 = (1,0)
   u₂ = A(0,1)/2 = (0,-2)/2 = (0,-1)
⟹ A = [1 0; 0 -1] · diag(3,2) · I     (the sign lives in U — note u₂ flipped)
```
Reconstruction check: rank-1 truncation A₁ = σ₁u₁v₁ᵀ = 3·[1;0]·[1,0] = [3 0; 0 0]; error ‖A−A₁‖_F = σ₂ = 2 ✓ (Eckart–Young, exactly).

**Recsys MF micro-example (the contrast):** ratings R = [5 3 ?; 4 ? 2], k=1. You *cannot* SVD it — the '?' entries are unknown. Instead initialize p (2×1), q (3×1) randomly; one SGD step on observed entry (user 1, item 1, r=5): predict p₁q₁, error e = 5 − p₁q₁, update p₁ += η·e·q₁, q₁ += η·e·p₁. Iterate over observed cells only; the learned p₁q₃ then *predicts* the missing (1,3) entry. Walking these two examples side by side — exact decomposition vs learned factorization — is the cleanest way to demonstrate you understand the distinction live.

---

## 5. Hyperparameters

Classical SVD is parameter-free except k; the knobs live in truncated/randomized SVD and in recsys MF (kept here because that's what your interviews probe):

| Choice | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| k (rank) | Truncation / factor count | More fidelity, less compression; MF: more capacity, overfit risk | Aggressive compression; MF: underfit | "How choose k?" → spectrum elbow / cumulative energy for SVD; **CV on held-out ratings** for MF |
| (randomized) n_oversamples, n_iter | Sketch quality | Better top-k accuracy, slower | Faster, noisier | "Why power iterations?" → sharpen separation when σ's decay slowly |
| (MF) λ regularization | Shrinks factors | Smoother, generalizes; underfit if too high | Overfit observed ratings | The core MF generalization knob |
| (MF) learning rate η (SGD) | Step size | Faster/unstable | Slow/stable | Standard SGD tuning |
| (MF) bias terms | μ, b_u, b_i | Capture rating-scale offsets | Worse cold baselines | "Why biases matter?" → most rating variance is user/item offset, not interaction |
| (MF implicit) confidence α | Weights observed interactions | Trusts clicks more | — | Hu-Koren-Volinsky weighting |
| (MF) negative sampling rate | Implicit-feedback negatives | More/less signal on unobserved | — | "Zeros aren't dislikes" — sampling design |

The senior framing: classical SVD has one honest knob (k); recsys MF is a *regularized regression*, so it has all of regression's knobs plus bias/confidence structure — recognizing that reframing is itself a strong signal.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Full SVD complexity | O(min(m²n, mn²)) — fine for modest matrices, infeasible at web scale |
| Truncated/randomized | O(mnk) for top-k — the only practical path for large sparse matrices (LSA, embeddings) |
| Recsys MF training | O(\|Ω\|·k) per epoch (only observed entries); ALS parallelizes across users/items, SGD streams — both scale to billions of interactions |
| Inference (MF) | Score = pᵤᵀqᵢ: O(k) dot product; top-N recs = ANN over item factors (Ch. 7/9 indexes) — MF factors *are* the embeddings you put in FAISS |
| Memory | SVD: U,Σ,V; truncated keeps m×k + k + n×k. MF: (users+items)×k factors — the servable artifact |
| Distributed | ALS is the Spark MLlib flagship recommender (alternating ridge solves parallelize trivially); randomized SVD distributes the sketch |
| Cold start | MF has none for new users/items (no factor learned) — the perennial production gap; hybrid with content features / two-tower fixes it (your retrieval story) |
| Monitoring | Factor drift across retrains, coverage of new items, rating/interaction distribution shift, and the recsys-specific feedback loop (recs shape future interactions shape training data) |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What is SVD geometrically?** Any linear map = rotation (Vᵀ) → axis-aligned scaling (Σ) → rotation (U). Singular values = stretch factors along principal axes.
2. **How do SVD and eigendecomposition relate?** V = eigenvectors of AᵀA, U = eigenvectors of AAᵀ, σᵢ = √eigenvalues. SVD exists for all matrices; eigendecomposition only for diagonalizable square ones.
3. **State Eckart–Young.** Top-k truncation is the optimal rank-k approximation under Frobenius and spectral norms; error = √(Σ_{i>k}σᵢ²) (Frobenius) or σ_{k+1} (spectral).
4. **How does PCA relate to SVD?** PCA = SVD of centered data; right singular vectors are components, σ²/(n−1) are eigenvalues/variances. SVD is the numerically stable way to compute PCA.
5. **Why compute PCA via SVD instead of eig(XᵀX)?** Forming XᵀX squares the condition number (precision loss on small singular values); SVD operates on X directly.
6. **What is the pseudoinverse and what's it for?** A⁺ = VΣ⁺Uᵀ; solves least squares including rank-deficient/underdetermined systems, giving the minimum-norm solution.
7. **How is truncated SVD a regularizer?** Dropping tiny singular values prevents dividing by near-zero in the pseudoinverse, suppressing noise amplification — a hard-threshold cousin of ridge.
8. **What is LSA?** Truncated SVD on a (TF-IDF) term-document matrix → low-dim "topic" space; documents/words as dense vectors; the pre-neural ancestor of embeddings.
9. **Why can't you run classical SVD on a rating matrix?** It's incomplete — SVD requires a fully specified matrix; missing ≠ zero, so zero-filling distorts the geometry. Use MF over observed entries.
10. **What does recsys "SVD" actually do?** Learns user/item latent factors by minimizing regularized squared error over *observed* ratings (SGD/ALS), often with global/user/item biases. A low-rank regression, not a decomposition.
11. **Why add bias terms in MF?** Most rating variance is systematic offset (some users rate high, some items are popular); biases absorb it so factors model the genuine *interaction*.
12. **Singular values vs eigenvalues — key differences?** Singular values always ≥ 0 and exist for any matrix; eigenvalues can be negative/complex and need square diagonalizable matrices. σ = √(eigenvalue of AᵀA).
13. **How do you choose k for truncation?** Cumulative singular-value energy (e.g., 90–95%), spectrum elbow; for MF, cross-validate k on held-out interactions (the task metric, not energy).
14. **What is NMF and when over SVD?** Non-negative MF: factors constrained ≥ 0 ⟹ additive, parts-based, interpretable (topics, image parts). Use when non-negativity/interpretability matter; loses orthogonality and optimality guarantees.
15. **How does implicit feedback change MF?** Clicks/views aren't ratings and zeros aren't dislikes; weighted MF treats all entries as observed with confidence weights (Hu-Koren-Volinsky) or uses negative sampling — the dominant real-world recsys setting.

### Advanced (15)

1. **Derive the SVD from the eigendecomposition of AᵀA.** AᵀA = VΛVᵀ (PSD ⟹ real ≥0 eigenvalues, orthonormal V); set σᵢ=√λᵢ, uᵢ = Avᵢ/σᵢ; show {uᵢ} orthonormal and AV = UΣ ⟹ A = UΣVᵀ.
2. **Prove the Frobenius-norm Eckart–Young error formula.** ‖A‖_F² = Σσᵢ² (unitary invariance); A−A_k has singular values σ_{k+1..r}; so ‖A−A_k‖_F² = Σ_{i>k}σᵢ². Optimality: any rank-k B can't capture more than the top-k energy (interlacing/variational argument).
3. **Relate ridge regression to truncated SVD.** With X=UΣVᵀ, ridge coefficients shrink each component by σ²/(σ²+λ); truncated SVD hard-zeros components below a cutoff. Both tame small-σ (high-variance) directions — soft vs hard spectral filtering.
4. **Why does randomized SVD work, and when does it struggle?** A random sketch captures the dominant range w.h.p. when the spectrum decays; slowly-decaying spectra need power iterations (A AᵀA)^q to widen the σ gaps before sketching, else top-k leaks into the tail.
5. **MF vs SVD on a zero-filled matrix — what actually goes wrong?** Zero-filling tells the model "unobserved = rating 0," biasing factors toward predicting low everywhere and wasting capacity reconstructing structural zeros; MF over observed entries treats missingness as missing (MCAR-ish assumption) — fundamentally different objective.
6. **ALS vs SGD for MF — tradeoffs.** ALS: fix Q, solve P by ridge (closed form per user), alternate — parallelizes beautifully (Spark), handles implicit-feedback weighting cleanly, more memory/compute per step. SGD: streams, cheaper per step, easy to add biases/side-features, but sequential and learning-rate-sensitive.
7. **How do MF factors become embeddings for retrieval?** Item factors Q are item embeddings; top-N for a user = max pᵤᵀqᵢ = max-inner-product search ⟹ ANN index over Q (FAISS/HNSW). MF + ANN = a candidate generator — the direct line to two-tower retrieval.
8. **MF vs two-tower models — relationship and difference.** MF is a two-tower with embedding-lookup towers and a dot-product scorer; two-tower generalizes the towers to neural nets ingesting *features* (content, context) ⟹ solves cold start and adds non-linearity. "MF is the linear, ID-only special case of two-tower" is the staff-grade framing — *and your bridge to the retrieval work on your resume.*
9. **What does the spectral norm error σ_{k+1} tell you that Frobenius doesn't?** Worst-case directional error (largest single discarded stretch) vs total energy; matters when a downstream operator is sensitive to the largest residual direction (numerical stability, adversarial robustness), not aggregate fit.
10. **SVD for collaborative filtering before MF — what was "FunkSVD" really?** Simon Funk's Netflix entry: gradient-descent low-rank factorization over observed ratings with regularization — named "SVD" by analogy, mathematically a regularized regression. The historical name is the source of the perennial confusion.
11. **Polar decomposition and SVD?** A = QP with Q = UVᵀ (nearest orthogonal matrix to A) and P = VΣVᵀ (PSD); used in orthogonal Procrustes (align two embedding spaces by the optimal rotation) — relevant for cross-lingual/temporal embedding alignment.
12. **How does SVD enable the orthogonal Procrustes solution?** min‖AR−B‖_F over orthogonal R: solution R = UVᵀ where UΣVᵀ = SVD(BᵀA). One sentence; very strong in embedding-alignment contexts (aligning Word2Vec spaces across corpora/time).
13. **Truncated SVD on TF-IDF (LSA) vs Word2Vec — relationship.** LSA factorizes a global co-occurrence/count matrix (matrix-factorization view); Word2Vec/GloVe implicitly factorize shifted-PMI / log-co-occurrence matrices (Levy-Goldberg result) — so neural embeddings are *implicit* SVD-like factorizations. Sets up Ch. 16–18.
14. **Incremental/folding-in SVD — how do you embed a new document/user without refitting?** Project the new row onto existing V: x_new's representation = x_newᵀ V Σ⁻¹ (folding-in) — cheap but drifts as the corpus changes; periodic refit corrects accumulated error. MF analogue: solve one ridge for the new user's factor with item factors fixed.
15. **When is the low-rank assumption wrong, and how would you detect it?** Slowly-decaying spectrum (no clear gap ⟹ no good low-rank structure); high reconstruction error at every reasonable k; heavy-tailed singular values. Detect by plotting the spectrum and the rank-k error curve before committing to a factorization model.

### Staff-Level (10)

1. **Design a recommender for 100M users × 10M items, 0.01% density. Where does "SVD" fit and where does it break?** Classical SVD: out (incomplete + scale). Real stack: regularized MF/ALS with biases for warm collaborative signal → item factors into an ANN index for retrieval → reranker on top. Breakpoints to name: cold start (factors undefined ⟹ content/two-tower hybrid), implicit feedback (weighted MF, not ratings), popularity bias (debias sampling), and the feedback loop (exploration to avoid filter bubbles). The interviewer wants you to *correct the premise* — "you said SVD; what you need is incomplete-matrix factorization plus retrieval infra."
2. **Your MF recommender's offline RMSE improves but online engagement drops. Diagnose.** RMSE rewards predicting *observed* ratings; engagement depends on *ranking* and *discovery*. Suspects: optimizing rating accuracy instead of top-N ranking (switch to ranking loss/BPR), popularity bias inflating easy hits, missing-not-at-random (users rate what they chose — selection bias), feedback loop narrowing exposure. The lesson: RMSE is a proxy; the system optimizes the wrong objective. (Same offline-online gap theme as the GBM chapter.)
3. **Compress 50M 768-d item embeddings for serving. SVD/PCA, PQ, or both?** Likely both: PCA/SVD rotation + dimensionality cut (768→128) for the dense win, then PQ (Ch. 9 K-Means codebooks) for the RAM win — OPQ explicitly composes a learned rotation (SVD-flavored) with PQ. Gate on recall@k per segment and tail items, version the rotation with the encoder. Naming OPQ (rotation + PQ) is the systems-depth marker.
4. **A teammate runs `TruncatedSVD` on a zero-filled rating matrix and ships it. Intervene.** Mechanism of the bug: structural zeros become training targets ⟹ factors learn "predict ~0," recommendations collapse toward unrated-everywhere. Fix: MF over observed entries (explicit) or weighted MF (implicit); show the offline metric on *held-out observed* ratings (not reconstruction of zeros) to expose the gap. Institutionalize: missingness handling reviewed in design, not code review.
5. **Aligning user/item embeddings across two model versions for A/B comparability — approach?** Orthogonal Procrustes via SVD (R = UVᵀ from SVD of the cross-covariance) to rotate one space onto the other before comparing — embedding spaces are only defined up to rotation, so naive cosine across versions is meaningless. This exact issue arises comparing retrained two-tower towers; flagging the rotational ambiguity is the senior catch.
6. **When is the closed-form SVD genuinely the right production tool in 2026?** Complete (or completable) matrices: LSA on full corpora, image/signal compression, PCA preprocessing, pseudoinverse for least-squares with rank deficiency, spectral methods (clustering, graph embeddings), Procrustes alignment. The honest line: classical SVD is for *complete* matrices and *exact* low-rank structure; the moment the matrix is incomplete or non-linear, you're in MF/AE territory.
7. **How do you monitor an MF-based recommender for the feedback-loop pathology?** Recs shape clicks shape training data ⟹ self-reinforcing popularity and shrinking catalog coverage. Monitor: catalog coverage / Gini of recommended items, exposure fairness, novelty/diversity metrics, and maintain an exploration holdout (epsilon-random or bandit) whose interactions aren't model-biased — the untreated-spine pattern recurring across chapters.
8. **n ≪ d least squares with a near-singular design — SVD-based solution and the judgment call.** x = A⁺b via truncated SVD: choose the truncation threshold by the singular-value gap (drop σ below noise level). Judgment: where to cut trades bias (dropping real-but-small directions) against variance (keeping noise-amplifying tiny σ) — equivalently, pick the ridge λ. State that the cutoff *is* a regularization decision, not a numerical detail.
9. **Explain to a PM why "more latent factors = better recommendations" is wrong.** Each factor is capacity; past the true rank you fit noise in observed ratings, generalize worse, and inflate serving cost — same overfitting curve as any model. The right k is set by held-out ranking metrics, and biases/regularization often matter more than k. Translate "rank" as "how many independent taste dimensions we can reliably estimate from the data we have."
10. **Interviewer: "Isn't SVD obsolete now that we have deep recommenders?" Respond.** Disagree with nuance: SVD/MF remains a strong, cheap, interpretable baseline and a *component* of modern stacks (factors as embeddings, SVD rotations inside PQ/OPQ, spectral initializations, alignment). Deep models earn their cost with side-features, non-linearity, and cold-start handling — but a tuned MF baseline is what they must beat, and on sparse warm-collaborative signal it's often within a hair at a fraction of the cost. "Obsolete" ignores the cost-benefit; the measured comparison is the answer.

---

## 8. Comparison Section

**SVD vs PCA (Ch. 10 bridge):** PCA = SVD of *centered* data + statistical interpretation (variance/covariance); SVD is the more general algebraic object. PCA is "SVD with a mean-subtraction and a variance story."

**Classical SVD vs Recsys Matrix Factorization (THE comparison):**

| | Classical SVD | Recsys MF ("SVD") |
|---|---|---|
| Input matrix | Complete | Incomplete (sparse observed set Ω) |
| How obtained | Exact closed-form decomposition | Learned by SGD/ALS over observed entries |
| Factors | Orthonormal (U, V) | Not orthonormal; regularized |
| Objective | Best rank-k approx (Eckart–Young) | Min regularized error on observed + biases |
| Missing entries | Not allowed | The entire point — predicted, not filled |
| Optimality | Provably optimal | Local optimum (non-convex jointly) |

**SVD vs NMF:** orthogonal/signed, optimal, fast vs non-negative, parts-based, interpretable, no orthogonality. NMF for topics/parts where signs are meaningless.

**MF vs Two-Tower (the resume-critical one):** MF = two-tower with ID-lookup towers + dot product; two-tower = neural towers over *features* (content/context) + dot product. Two-tower solves cold start and adds non-linearity; MF is its linear ID-only ancestor. Both end in MIPS/ANN retrieval.

**Truncated SVD vs Word2Vec/GloVe (Ch. 16–18 bridge):** LSA factorizes count matrices explicitly; neural embeddings implicitly factorize (shifted) PMI/log-co-occurrence matrices (Levy-Goldberg) ⟹ "neural embeddings are implicit SVD."

---

## 9. Common Mistakes

**Candidate mistakes:**
- Conflating recsys "SVD" with classical SVD (the defining error — interviewers bait this).
- Not knowing Eckart–Young, or stating truncation is "a heuristic" rather than provably optimal.
- Saying singular values can be negative, or that SVD needs a square matrix.
- Forgetting PCA = SVD of *centered* data (the centering, specifically).
- Unable to connect MF factors to embeddings/ANN retrieval.

**Production mistakes:**
- Zero-filling a sparse matrix and running TruncatedSVD as a "recommender."
- Treating implicit feedback as explicit ratings (zeros as dislikes).
- Optimizing RMSE when the product needs top-N ranking.
- Comparing embeddings across model versions without Procrustes alignment.
- No cold-start path (MF gives new users/items no factor).

**Modeling mistakes:**
- Choosing k by singular-value energy when held-out ranking metric is the right criterion.
- Ignoring bias terms (then "discovering" factors model user/item offsets, not taste).
- Folding-in indefinitely without periodic refit (accumulated drift).
- Assuming low-rank structure without checking the spectrum (no gap ⟹ no good factorization).

---

## 10. Real Industry Use Cases

- **Netflix** — the canonical home: Prize-era MF ("SVD"/SVD++) on the ratings matrix defined modern collaborative filtering; today MF survives as a baseline/component beneath deep rankers.
- **Google** — LSA/SVD lineage in early semantic retrieval; SVD-flavored rotations inside ScaNN/quantization; spectral methods in graph/embedding work.
- **Amazon** — item-to-item collaborative filtering's low-rank cousins; MF baselines in recommendations; SVD for dimensionality reduction in forecasting/risk.
- **Meta** — MF/embedding factorization as candidate generation ancestors; FAISS quantization (OPQ) composes SVD-style rotation with PQ; embedding-space alignment via Procrustes.
- **Uber** — low-rank factorization for spatiotemporal demand matrices; SVD-based denoising of telemetry.
- **Swiggy/Zomato** — collaborative-filtering MF for restaurant/dish recommendations as the warm-signal baseline beneath two-tower retrieval; SVD/PCA compression of embeddings before ANN serving. *Naveen: your Similar Restaurants work is the content/embedding side; MF is the collaborative side — being able to position them as complementary (and MF as the linear ancestor of your two-tower) is a complete recsys-architecture answer.*
- **Flipkart** — MF/ALS recommenders for product discovery; truncated SVD for catalog-text LSA features.
- **Games24x7** — low-rank factorization of player×game-mode interaction matrices for personalization; SVD denoising in behavioral feature pipelines.

---

## 11. Coding From Scratch (NumPy only)

Two implementations side by side — because the *contrast* is the lesson: (A) truncated SVD via the AᵀA eigen-route (for understanding; production uses `np.linalg.svd`/randomized), and (B) recsys MF by SGD over observed entries.

```python
import numpy as np

# ============ (A) Truncated SVD on a COMPLETE matrix ============
class TruncatedSVDScratch:
    """For understanding only — real code calls np.linalg.svd or randomized SVD.
       Shows the AᵀA → V → U route and Eckart–Young truncation."""
    def __init__(self, k):
        self.k = k

    def fit(self, A):
        A = np.asarray(A, float)
        # Right singular vectors V = eigenvectors of AᵀA (descending eigenvalue).
        # NOTE: numerically inferior to svd(A) — we square the condition number.
        # Doing it this way ONLY to expose the eigen↔singular relationship.
        w, V = np.linalg.eigh(A.T @ A)          # eigh: symmetric, ascending
        order = np.argsort(w)[::-1]             # descending
        w, V = w[order], V[:, order]
        sigma = np.sqrt(np.clip(w, 0, None))    # σ = √eigenvalue, clip tiny negatives
        self.s_ = sigma[:self.k]
        self.V_ = V[:, :self.k]                 # (n, k)
        self.U_ = (A @ self.V_) / self.s_       # uᵢ = A vᵢ / σᵢ  → (m, k)
        return self

    def reconstruct(self):
        # Eckart–Young optimal rank-k approximation: Σ σᵢ uᵢ vᵢᵀ
        return (self.U_ * self.s_) @ self.V_.T

# ============ (B) Recsys Matrix Factorization (INCOMPLETE matrix) ============
class MatrixFactorizationScratch:
    """
    The thing people CALL 'SVD' in recommenders but isn't:
    learn factors over OBSERVED entries only, with biases + L2.
        r̂_ui = μ + b_u + b_i + pᵤ·qᵢ
    """
    def __init__(self, k=20, lr=0.01, reg=0.05, n_epochs=30, seed=0):
        self.k, self.lr, self.reg, self.n_epochs = k, lr, reg, n_epochs
        self.rng = np.random.default_rng(seed)

    def fit(self, observed):
        # observed: list of (user_idx, item_idx, rating) — ONLY known cells.
        users = max(u for u, _, _ in observed) + 1
        items = max(i for _, i, _ in observed) + 1
        self.mu = np.mean([r for _, _, r in observed])      # global mean baseline
        self.bu = np.zeros(users)
        self.bi = np.zeros(items)
        self.P = self.rng.normal(0, 0.1, (users, self.k))   # user factors
        self.Q = self.rng.normal(0, 0.1, (items, self.k))   # item factors

        for _ in range(self.n_epochs):
            self.rng.shuffle(observed)
            for u, i, r in observed:                         # iterate OBSERVED only —
                pred = self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]
                e = r - pred                                 # the missing cells are
                # SGD updates: gradient of (e² + reg·‖·‖²)   #  simply never touched
                self.bu[u] += self.lr * (e - self.reg * self.bu[u])
                self.bi[i] += self.lr * (e - self.reg * self.bi[i])
                Pu = self.P[u].copy()
                self.P[u] += self.lr * (e * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (e * Pu       - self.reg * self.Q[i])
        return self

    def predict(self, u, i):
        return self.mu + self.bu[u] + self.bi[i] + self.P[u] @ self.Q[i]

    def recommend(self, u, top_n=10):
        # Top-N = max inner product pᵤ·qᵢ over items → in production this is
        # an ANN / max-inner-product search over self.Q (FAISS), not a full scan.
        scores = self.mu + self.bu + self.bi + self.Q @ self.P[u]
        return np.argsort(scores)[::-1][:top_n]
```

Narration points that earn senior credit:
- **Point at the two `fit` methods and say the sentence**: "(A) decomposes a *complete* matrix in closed form; (B) *learns* factors over *observed* entries of an incomplete one — same low-rank idea, different problem. Calling (B) 'SVD' is historical."
- **In (A): `eigh(AᵀA)` is deliberately the wrong production choice** — flag that `np.linalg.svd(A)` is stabler (condition-number argument from §3); you're using the eigen-route only to expose σ = √λ and uᵢ = Avᵢ/σᵢ.
- **In (B): "iterate over observed only"** — the comment on the loop is the whole point; the missing cells are never zero-filled, they're simply absent from training.
- **Bias updates before factor updates**, and that biases capture the bulk of rating variance (user/item offsets) so factors model interaction.
- **`recommend` is a full scan here but is MIPS/ANN in production** — connects MF factors to the retrieval infra (Ch. 7/9) and to your two-tower story in one line.
- Offer the implicit-feedback extension: confidence-weighted ALS (zeros = low-confidence negatives), which is what real click-based systems run.

---

## 12. ML System Design Perspective

**Choose classical SVD when:** the matrix is complete (LSA on full corpora, image/signal compression, PCA preprocessing); you need the optimal rank-k approximation or a numerically stable pseudoinverse/least-squares; spectral methods or embedding-space alignment (Procrustes).

**Choose MF (recsys "SVD") when:** the matrix is incomplete and you want to predict missing entries (collaborative filtering); warm collaborative signal dominates; you need item factors as embeddings for ANN retrieval. Use *weighted* MF for implicit feedback.

**Avoid both when:** structure is non-linear (autoencoders/deep models); cold start dominates (content/two-tower hybrids); non-negativity/interpretability required (NMF); the spectrum shows no low-rank structure (no gap ⟹ factorization won't help).

**Data requirements:** SVD needs a complete numeric matrix; MF needs an observed-interaction log (and a missingness model — explicit vs implicit changes everything); enough interactions per user/item for stable factors (the cold-start floor).

**Latency:** SVD truncation/MF scoring are O(k) dot products — trivial; production cost is the retrieval index over factors (ANN), not the factorization. Training (ALS/SGD) is the heavier offline job.

**Scale limits:** randomized SVD and ALS/SGD scale to billions; the real constraints are sparsity/cold-start (statistical) and the feedback loop / exposure bias (systemic), not raw compute.

---

## 13. Resume Discussion Angle

**Recommendation systems (your core territory):** The defining moment is when an interviewer says "tell me about SVD in your recommender." The full-marks move is to *correct the framing yourself*: "If we're precise, the Netflix-style 'SVD' is matrix factorization over observed ratings, not the decomposition — and in our stack the collaborative MF factors were embeddings we served through an ANN index, complementary to the content-based Similar Restaurants embeddings." That single answer demonstrates the algebra, the historical-naming awareness, *and* positions your two hero projects as the collaborative and content halves of one retrieval architecture. Then be ready for: cold start (two-tower with features), implicit feedback (weighted ALS), and the RMSE-vs-ranking offline/online gap (§7-S2).

**The MF → two-tower bridge:** "MF is a two-tower model with ID-lookup towers and a dot-product scorer; we generalized the towers to ingest content and context features, which is what solved cold start." This is the sentence that elevates a CF baseline story into a modern-retrieval-architecture story — and it's true to your two-tower experience. Rehearse it.

**Fraud/anomaly:** low-rank factorization of behavior matrices for denoising/structure, and reconstruction-error-as-anomaly (the SVD cousin of the PCA tripwire from Ch. 10) — a complementary detector to Isolation Forest in your Games24x7 arc.

**The universal trap:** "Walk me through how SVD recommends a movie." If you describe decomposing the ratings matrix, you've failed the bait. The correct narrative: learn user/item factors over observed ratings (with biases, regularization, SGD/ALS) → a user's predicted affinity is the dot product → top-N via inner-product search over item factors. Lead with "the matrix is mostly missing, so we don't decompose it — we factorize it by optimization," and you've shown the exact understanding the question is screening for.

---
*Previous: PCA ← | Next: Neural Networks (MLP) →*

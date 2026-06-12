# Chapter 9: K-Means Clustering
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

> Modern relevance note: K-Means is both the classic segmentation tool *and* live production infrastructure — it's the coarse quantizer inside IVF vector indexes and the codebook learner in product quantization. If your resume says "embedding retrieval," K-Means is literally running inside your serving stack.

---

## 1. Executive Summary (30 seconds)

K-Means partitions n points into k clusters by minimizing within-cluster sum of squared distances to cluster centroids. Lloyd's algorithm alternates two steps — assign each point to its nearest centroid, then recompute each centroid as the mean of its points — and provably converges, but only to a local optimum, which is why k-means++ initialization and multiple restarts matter. It assumes roughly spherical, similarly sized clusters in Euclidean space. It's the default for segmentation, vector quantization, and as the workhorse inside ANN indexes; choosing k and validating that clusters mean anything is where the real judgment lives.

---

## 2. Interview Articulation (3–4 Minute Answer)

"K-Means answers: if I had to summarize my data with k representative points, where should they go? Formally, place k centroids and assign every point to its nearest one, choosing the configuration that minimizes total squared distance from points to their assigned centroids — the inertia, or within-cluster sum of squares.

That objective is NP-hard to optimize globally, so we use Lloyd's algorithm — a beautifully simple alternating scheme. Step one: freeze the centroids, assign each point to the nearest. Step two: freeze the assignments, move each centroid to the mean of its points. Repeat until nothing changes. Each step can only decrease the objective — assignment is optimal given centroids, and the mean is the point minimizing squared distance given assignments — and since there are finitely many partitions, it must converge. The catch: it converges to a *local* optimum that depends entirely on initialization. Bad starting centroids give bad clusterings that look stable. The standard fix is **k-means++**: pick the first centroid at random, then pick each next one with probability proportional to squared distance from the nearest existing centroid — spreading the seeds — which carries a provable O(log k) approximation guarantee and is the default everywhere. Plus multiple restarts, keeping the lowest inertia.

Two big interview themes follow. First, **choosing k**. The objective alone can't do it — inertia decreases monotonically in k, hitting zero at k = n. The elbow method looks for the diminishing-returns kink; the silhouette score compares each point's within-cluster cohesion to its nearest-other-cluster separation; the gap statistic compares inertia to a null reference. But the honest senior answer is that in industry, k is usually a *product decision* — how many segments can marketing actually act on, how many cells does the IVF index need — and the statistical criteria are sanity checks, not oracles.

Second, **assumptions**. Squared Euclidean distance bakes in spherical clusters of similar size and density. K-Means will happily slice an elongated cluster in half, merge a small dense cluster into a big sparse one, and it cannot represent non-convex shapes — the two-moons dataset defeats it by construction. It's also sensitive to feature scaling — distance is geometry, so unscaled features dominate — and to outliers, since means get dragged; k-medoids swaps means for actual data points if robustness matters. The probabilistic framing: K-Means is the limit of a Gaussian mixture model with identical spherical covariances as variance goes to zero — hard assignments instead of soft responsibilities, and EM for GMM is the soft sibling of Lloyd's. That one sentence — 'K-Means is hard-assignment EM on an isotropic GMM' — answers half the advanced questions about it.

In production its biggest modern role is hidden: **vector quantization**. IVF indexes in FAISS run K-Means on your embedding corpus to create the cells you probe at query time; product quantization runs K-Means per subvector to learn the codebooks that compress billions of vectors into RAM. So every embedding-retrieval system is running K-Means twice before a single neural network sees a query. Mini-batch K-Means handles web-scale fitting by updating centroids from small random batches.

When not to use it: unknown or non-convex cluster shapes — DBSCAN or hierarchical; mixed categorical data — k-modes/k-prototypes or embed first; when you need cluster membership probabilities — GMM; when clusters differ wildly in density.

Traps: forgetting to scale; reading the elbow as objective truth; treating clusters as ground-truth segments without stability checks; and not knowing that K-Means on one-hot or sparse high-dimensional data mostly measures popularity, not similarity — embed first, cluster after."

---

## 3. Mathematical Foundation

**Objective (within-cluster sum of squares / inertia):**
```
J = Σₖ Σ_{xᵢ ∈ Cₖ} ‖xᵢ − μₖ‖²
```
Find both the partition {Cₖ} and centroids {μₖ} minimizing J. Globally NP-hard even for k=2.

**Lloyd's algorithm:**
```
repeat:
  Assignment:  cᵢ = argminₖ ‖xᵢ − μₖ‖²          (Voronoi partition)
  Update:      μₖ = mean{ xᵢ : cᵢ = k }
until assignments stable
```

**Why each step decreases J (the convergence proof to know):**
- Assignment step: for fixed μ, choosing the nearest centroid per point minimizes each term independently.
- Update step: for fixed assignments, argmin_μ Σ‖xᵢ−μ‖² = mean (same derivation as the regression-tree leaf: derivative −2Σ(xᵢ−μ) = 0).
- J is bounded below by 0 and strictly decreases until a fixed point; finitely many partitions ⟹ convergence in finite steps — to a **local** optimum.

**k-means++ initialization:**
```
1. μ₁ ← uniform random data point
2. for j = 2..k: pick x with probability ∝ D(x)², where D(x) = distance to nearest chosen centroid
```
Spreads seeds; E[J] ≤ 8(ln k + 2) · J_optimal (Arthur–Vassilvitskii). Quoting "O(log k)-competitive in expectation" is the senior flourish.

**K-Means ⊂ GMM/EM:** GMM with shared covariance σ²I; E-step responsibilities → hard argmax as σ² → 0; M-step mean update identical. Consequences: K-Means inherits EM's local-optima behavior, and GMM is the upgrade path when you need soft membership, unequal covariances (elliptical clusters), or cluster priors.

**Choosing k:**

| Method | Idea | Caveat |
|---|---|---|
| Elbow | Plot J vs k, find the kink | Subjective; often no clean elbow |
| Silhouette s(i) = (b−a)/max(a,b) | a = mean intra-cluster dist, b = mean dist to nearest other cluster; average over points; ∈ [−1,1] | O(n²) naive; sample for big data |
| Gap statistic | Compare log J to expectation under uniform null; pick smallest k with gap within 1 SE of max | Compute-heavy |
| Stability | Re-cluster on bootstraps; measure assignment agreement (ARI) | The production-grade check |
| Business | Actionability bounds k | Usually decisive |

**Distance variants:** squared Euclidean is *required* for the mean-update optimality. Cosine similarity ⟹ normalize vectors to unit length, then Euclidean K-Means ≈ spherical K-Means (the standard for embeddings). Arbitrary metrics ⟹ k-medoids (PAM), at higher cost.

**Mini-batch K-Means:** per batch, assign, then move centroids by per-centroid learning rate 1/(count seen): μₖ ← μₖ + (1/nₖ)(x̄_batch − μₖ). Slightly worse J, 10–100× faster — the web-scale default.

---

## 4. Step-by-Step Numerical Example

1-D points: {2, 3, 4, 10, 11, 12}, k = 2. Deliberately bad init to show convergence: μ₁ = 3, μ₂ = 4.

**Iteration 1 — assign:**
```
2→μ₁(d=1)  3→μ₁(0)  4→μ₂(0)  10→μ₂(6)  11→μ₂(7)  12→μ₂(8)
C₁ = {2,3}, C₂ = {4,10,11,12}
```
**Update:** μ₁ = 2.5, μ₂ = 9.25.

**Iteration 2 — assign:**
```
4: |4−2.5| = 1.5 < |4−9.25| = 5.25 → moves to C₁
C₁ = {2,3,4}, C₂ = {10,11,12}
```
**Update:** μ₁ = 3, μ₂ = 11.

**Iteration 3 — assign:** no changes ⟹ converged.
```
J = (1+0+1) + (1+0+1) = 4
```
Compare a bad local optimum: init μ₁ = 2, μ₂ = 3 can converge to C₁={2}, C₂={3,4,10,11,12} flavors with far higher J on nastier data — the demonstration that restarts/k-means++ aren't optional. Walking assignment→update→assignment aloud with running J is a standard whiteboard ask; practice the rhythm.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| k | Number of clusters | J ↓ always (toward 0 at k=n); segments fragment | Coarse clusters, heterogeneous segments | "Inertia improved when you raised k — better model?" → trick: J is monotone in k; need silhouette/stability/business lens |
| init | Seeding strategy | — | — | "Why k-means++ over random?" → spread seeds, O(log k) guarantee, fewer restarts needed |
| n_init | Restarts | More chances to escape local optima; cost ↑ linearly | Risk of bad local opt | "When can n_init=1?" → k-means++ with big k on well-separated data; or mini-batch at scale |
| max_iter / tol | Convergence budget | — | Premature stop | Rarely binding; Lloyd converges fast in practice |
| batch_size (mini-batch) | SGD-style updates | Closer to full-batch quality | Faster, noisier centroids | "Quality cost of mini-batch?" → few % inertia, usually irrelevant for VQ/segmentation |
| algorithm (lloyd/elkan) | Exact accelerations | — | — | Elkan: triangle-inequality bounds skip distance computations — same result, faster on low-d dense data |

The senior framing: K-Means has one true hyperparameter — **k** — and one true failure mode — **initialization**; everything else is engineering.

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training complexity | O(n·k·d) per iteration × iterations (typically ≤ 50–100); Elkan prunes with triangle inequality; mini-batch makes it streaming-friendly |
| Inference (assignment) | O(k·d) per point — nearest-centroid lookup; trivially fast, fully parallel |
| Memory | k·d centroids — negligible; mini-batch never holds full data |
| Distributed | Embarrassingly parallel assignment; centroid update = per-cluster (sum, count) AllReduce — Spark MLlib standard |
| GPU | Distance matrices are GEMM-shaped — K-Means on GPUs (FAISS) clusters 10⁸ embedding vectors routinely |
| Monitoring | Cluster population drift (segments emptying/exploding), centroid drift across refits, inertia per point trend, *assignment churn* between model versions — downstream consumers (campaigns, indexes) break when memberships reshuffle silently |
| Versioning gotcha | Cluster IDs are not stable across refits (label permutation + genuine drift): never hard-code "cluster 3 = premium users" — match clusters across versions via centroid distance (Hungarian assignment) before relabeling |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **Walk through Lloyd's algorithm.** Init centroids → assign each point to nearest → recompute centroids as means → repeat until stable. Both steps monotonically decrease WCSS; converges to a local optimum.
2. **Why does K-Means always converge?** Each step weakly decreases a bounded-below objective and there are finitely many partitions ⟹ no cycles ⟹ fixed point in finite iterations. (Convergence ≠ global optimality.)
3. **Why is initialization critical and what is k-means++?** Local optima depend on seeds; k-means++ picks seeds with probability ∝ squared distance to nearest existing seed — spread coverage, O(log k) expected approximation, fewer restarts.
4. **How do you choose k?** Elbow on inertia, silhouette, gap statistic, bootstrap stability — and the business constraint (actionable segment count / index cell budget), which usually decides. Present all, emphasize the last.
5. **Does K-Means need feature scaling?** Yes — Euclidean distance lets large-scale features dominate the geometry; standardize, or rethink the space entirely (embed first).
6. **Effect of outliers?** Means get dragged; an extreme point can hijack a centroid or claim its own cluster. Mitigate: remove/clip outliers, k-medoids, or use the outlier-claiming behavior deliberately as crude anomaly detection.
7. **K-Means vs GMM?** Hard vs soft assignment; spherical-equal vs full covariances (elliptical clusters); K-Means = σ²→0 isotropic GMM. GMM gives probabilities and shape flexibility at more parameters and fragility.
8. **K-Means vs DBSCAN?** K-Means: must fix k, convex/spherical clusters, every point assigned. DBSCAN: density-based, finds arbitrary shapes and labels noise, no k — but needs ε/minPts and struggles with varying densities.
9. **K-Means vs hierarchical clustering?** Hierarchical gives a dendrogram (multi-resolution, no k upfront) at O(n²)–O(n³); K-Means is flat and scales. Common combo: K-Means to 1000 micro-clusters, hierarchical on the centroids.
10. **What clusters does K-Means fail on?** Non-convex (two moons, rings), highly unequal sizes/densities, elongated ellipses — squared-Euclidean Voronoi cells are convex by construction.
11. **Can K-Means handle categorical features?** Not directly (means of one-hots are densities, distances degenerate). Use k-modes/k-prototypes, or — modern answer — learn/obtain embeddings and cluster those.
12. **What is inertia and its limitation as a metric?** WCSS = the training objective. Monotone in k, scale-dependent, blind to cluster meaningfulness — never compare across k or across feature spaces with it alone.
13. **Mini-batch K-Means — when and at what cost?** n ≳ 10⁶ or streaming; per-batch centroid updates with per-centroid decaying rates; a few percent worse inertia, order-of-magnitude faster — the default for VQ at scale.
14. **What's the time complexity and the practical bottleneck?** O(nkd) per iteration; bottleneck is the n×k distance computation — hence Elkan bounds, GPUs, and mini-batching.
15. **How do you check clusters are "real" and not artifacts?** Stability under resampling (ARI across bootstrap refits), silhouette by cluster, holdout inertia, and — decisive — downstream validation: do segments behave differently on outcomes that matter (conversion, churn)?

### Advanced (15)

1. **Prove the update step is optimal.** argmin_μ Σ‖xᵢ−μ‖²: gradient −2Σ(xᵢ−μ) = 0 ⟹ μ = mean. Note this is *why* squared Euclidean is mandatory — for L1 the optimum is the median (k-medians), for arbitrary metrics no closed form (k-medoids).
2. **State the k-means++ guarantee and the intuition for D² sampling.** E[J] ≤ 8(ln k + 2)·OPT. D² balances exploration (far regions likely) against outlier-robustness (a lone outlier is one draw, a far *cluster* is many draws' worth of probability mass).
3. **Derive K-Means as the σ→0 limit of EM on an isotropic GMM.** Responsibilities rᵢₖ ∝ πₖ exp(−‖xᵢ−μₖ‖²/2σ²); as σ²→0 the largest exponent dominates ⟹ rᵢₖ → indicator of nearest centroid (hard assignment); M-step mean update is unchanged. Soft → hard EM.
4. **Why are K-Means decision regions convex, and what does that imply?** Assignment regions are Voronoi cells of the centroids — intersections of half-spaces ⟹ convex. Implication: non-convex clusters are *unrepresentable* regardless of initialization or restarts; failure is structural.
5. **Elkan's acceleration — mechanism?** Maintain lower/upper bounds on point-centroid distances; triangle inequality (d(x,c') ≥ d(c,c') − d(x,c)) proves many candidate centroids can't be nearest without computing their distances. Exact same output, large constant-factor speedup in low-to-mid d.
6. **Curse of dimensionality for K-Means?** In high d, pairwise distances concentrate (max/min ratio → 1) ⟹ nearest-centroid assignments become noise-dominated; remedies: dimensionality reduction first (PCA to ~50), or cluster in a *learned* embedding space where distances are meaningful by construction.
7. **Spherical K-Means — what changes for cosine similarity?** L2-normalize data and (after each update) the centroids; maximizing cosine ⟺ minimizing Euclidean on the unit sphere. The standard for text/embedding clustering — normalization *is* the algorithm change.
8. **How does K-Means relate to vector quantization and rate-distortion?** Centroids = codebook; assignment = encoding (log₂k bits/vector); J = distortion. K-Means is the Lloyd–Max quantizer generalized to vectors — the framing that explains its role in PQ compression.
9. **Explain IVF and PQ's use of K-Means precisely.** IVF: K-Means with k = √n-ish cells over the corpus; query probes nProbe nearest cells — coarse quantizer as search pruner. PQ: split d-dim vector into m subvectors; run K-Means (256 centroids) per subspace; store m bytes of centroid IDs per vector; distances approximated via per-subspace lookup tables. Two nested K-Means = billion-scale ANN in RAM.
10. **Why does K-Means on raw one-hot/sparse data fail, mechanically?** Squared distance between one-hots is constant (2) unless they share support ⟹ geometry collapses to co-occurrence counts; centroids are dense frequency vectors and popular items dominate every cluster. Embed first.
11. **Empty-cluster problem — cause and fixes?** A centroid can end an assignment step with zero points (bad init / outlier centroid stranded). Fixes: re-seed it at the point farthest from its centroid (or with largest contribution to J), or split the largest cluster.
12. **K-Means objective as matrix factorization?** J = ‖X − HMᵀ‖² with H = one-hot assignment matrix, M = centroids — K-Means is constrained MF (H binary, rows sum to 1). Relaxing H to orthogonal/continuous connects to spectral clustering and PCA. (One-liner that lands very well in Applied Scientist loops.)
13. **Bisecting K-Means — why might it beat vanilla?** Recursively 2-means-split the cluster with highest SSE: more deterministic, near-hierarchical output, often better J than one-shot k-means++ for large k; standard in document clustering.
14. **How would you cluster with must-link / cannot-link constraints?** Constrained K-Means (COP-KMeans: respect constraints in assignment), or penalty-based variants; semi-supervised clustering — useful when ops teams hand you partial labels ("these merchants are the same segment").
15. **Estimate k for IVF in an ANN index — what's the actual tradeoff?** Cells ≈ √n heuristic: recall/latency curve — more cells = finer pruning but more centroids to scan at query and worse cell balance; tune (k, nProbe) jointly on recall@k vs p99 latency on production query distribution, not random vectors (query and corpus distributions differ — the senior detail).

### Staff-Level (10)

1. **Marketing has used "your 6 segments" for 2 years; you refit and the segments change materially. Walk through the decision.** Diagnose: drift (real behavior change) vs instability (local optima/seed). Run stability analysis (bootstrap ARI) on old vs new data; match clusters via Hungarian on centroid distances to quantify churn; if drift is real, version the segmentation with a migration map and re-validate downstream lift per segment before cutover; institute scheduled refits with stability gates rather than ad-hoc ones. The point being tested: clusters are a *product contract*, not just a model artifact.
2. **A PM asks "are these clusters statistically significant?" What's the honest answer?** Clustering has no native significance test — K-Means finds structure even in uniform noise. Offer: gap statistic vs null reference, stability under resampling, and outcome-based validation (segments differ on held-out behaviors with proper tests). Reframe from "significant" to "stable and actionable" — and say why.
3. **Design customer segmentation for a 50M-user food-delivery app, end to end.** Feature space decision first (RFM + behavioral embeddings; scale/transform; consider clustering on learned user embeddings rather than raw features); mini-batch spherical K-Means, k swept 4–12 against silhouette + stability + activation-team capacity; profile clusters with readable archetypes; validate via per-segment response in a randomized campaign; productionize assignment as a feature-store transform with population-drift monitors and quarterly refit + Hungarian relabeling. *This is a near-verbatim Swiggy/Zomato interview case — rehearse it as a 5-minute structure.*
4. **Your IVF index recall dropped 4 points after a routine embedding-model update. Why might K-Means be the culprit and what's the fix?** Cells were trained on old-embedding geometry; new embeddings shift the manifold ⟹ cell boundaries misalign, residual/PQ codebooks mismatch ⟹ recall loss concentrated near cell edges. Fix: retrain coarse quantizer + PQ codebooks on new embeddings (index rebuild), version index with the encoder, and gate encoder rollouts on recall@k against a fixed eval set. The lesson to state: **the index is part of the model**.
5. **When would you *refuse* to deliver a clustering?** When the feature space encodes nothing decision-relevant (clusters would be artifacts of scaling choices), when stakeholders want clusters to confirm a predetermined narrative, when stability analysis shows assignments are seed noise, or when a supervised objective exists (if they'll act on churn, model churn — don't launder a prediction problem through unsupervised segmentation). Knowing when clustering is the *wrong tool* is the staff signal.
6. **Streaming K-Means for evolving user behavior — design and pitfalls.** Mini-batch updates with decay (recent data weighted), centroid drift monitors, periodic full refits to escape accumulated local-optima drift; pitfalls: concept drift masquerading as cluster drift, ID-stability for downstream consumers (relabel via matching), and feedback loops if segment-targeted treatments change the behavior being clustered (keep an untreated holdout — same spine as the GBM churn-discount answer).
7. **You need 1M-cluster K-Means for a billion-vector PQ codebook pipeline. Engineering plan?** Hierarchical/two-level clustering (cluster to 1K, then 1K within each), GPU FAISS k-means with mini-batches, k-means++ via AFK-MC² (MCMC seeding — exact D² sampling is too slow at this scale), balanced-cell constraints for index health, and evaluation purely on end-task recall/latency, not inertia. Mentioning *balance constraints* (skewed cells wreck p99) is the systems-depth marker.
8. **An analyst reports "cluster 4 has 3× churn — let's target it." What do you check before money moves?** Composition confounds (cluster 4 may just be "new users" — churn is age, not cluster), stability of the cluster across refits, whether a churn *model* dominates the segmentation for this decision, and an uplift framing: high-churn segment ≠ persuadable segment. Then a randomized pilot within-cluster. Clustering describes; it doesn't license causal targeting.
9. **How do you make segmentation fair/compliant when sensitive attributes are excluded but proxied?** Audit cluster composition against protected attributes (allowed for auditing), measure proxy leakage (predict sensitive attribute from cluster ID), constrain or reweight features driving proxy structure, document permitted uses. Removing the column doesn't remove the geometry — saying that sentence is the point.
10. **Your team debates K-Means vs GMM vs HDBSCAN for a one-off exploratory analysis vs a production segmentation pipeline. Frame the decision.** Exploration: HDBSCAN/GMM for shape-honesty and noise labeling — you want to *see* structure. Production: K-Means' virtues are operational — O(kd) assignment, trivial serving, stable tooling, distributed fit — and shape flexibility matters less than contract stability and latency. Different jobs, different winners; the reframe from "which is best" to "best for which lifecycle stage" is the answer.

---

## 8. Comparison Section

| | K-Means | GMM | DBSCAN/HDBSCAN | Hierarchical | Spectral |
|---|---|---|---|---|---|
| k required | Yes | Yes (or BIC sweep) | No (ε/minPts or min size) | No (cut dendrogram) | Yes |
| Cluster shape | Convex/spherical | Elliptical | Arbitrary (density) | Depends on linkage | Arbitrary (graph) |
| Soft membership | No | Yes | No | No | No |
| Noise/outlier label | No | No | **Yes** | No | No |
| Scale | O(nkd)/iter — best | Heavier (covariances) | O(n log n) with index | O(n²)+ | O(n³) eigen — worst |
| Production serving | Trivial (nearest centroid) | Moderate | Awkward (no centroids) | Awkward | Research-only at scale |

**K-Means vs GMM one-liner:** "K-Means is hard-assignment EM on an isotropic GMM — pay for GMM when you need soft membership or elliptical shapes; pay for K-Means when you need scale and a servable assignment function."

**K-Means vs KNN (perennial confusion check):** K-Means = unsupervised partitioning (learns centroids); KNN = supervised lazy prediction (stores labeled data). They share only the word "nearest." Expect this as a deliberate trap question; answer in one breath.

**K-Means vs PCA:** PCA finds directions (continuous compression); K-Means finds prototypes (discrete compression); the MF view (§7-A12) shows both as constrained factorizations of X — and "PCA then K-Means" is the standard pipeline: decorrelate/denoise, then partition.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Conflating K-Means with KNN.
- Claiming convergence to the global optimum, or not knowing *why* it converges at all.
- Choosing k by inertia alone ("it kept going down").
- No mention of scaling, initialization, or restarts.
- Missing the GMM/EM relationship when probed "what if clusters are elliptical?"

**Production mistakes:**
- Hard-coding cluster IDs in downstream systems across refits (label permutation breaks everything silently).
- Clustering raw sparse/one-hot data and shipping popularity artifacts as "segments."
- No stability gate before stakeholder-facing segment changes.
- IVF/PQ codebooks not rebuilt after embedding-encoder updates (the recall-drop incident in §7-S4).

**Modeling mistakes:**
- Segmenting when the decision called for a supervised model (laundering prediction through clustering).
- Treating cluster differences as causal targeting evidence.
- Ignoring outliers' centroid-hijacking, then interpreting the hijacked cluster as a segment.
- Clustering in a feature space chosen for convenience rather than decision relevance.

---

## 10. Real Industry Use Cases

- **Google** — vector quantization inside ScaNN-style retrieval (anisotropic VQ descends from K-Means); historical: news/document clustering; YouTube candidate-generation-era user/item grouping.
- **Amazon** — customer segmentation for marketing programs; K-Means as the coarse quantizer in product-embedding ANN; warehouse SKU velocity clustering for slotting.
- **Netflix** — taste-community exploration on viewing embeddings; artwork/asset grouping; cluster-then-inspect workflows feeding editorial strategy.
- **Meta** — user/content embedding clustering for integrity sweeps and audience tooling; FAISS (Meta's own library) makes K-Means the literal substrate of their billion-scale similarity search.
- **Uber** — geospatial demand clustering (hexbin + K-Means hybrids) for positioning and surge zones; rider segmentation for incentives.
- **Swiggy/Zomato** — restaurant clustering by cuisine/price/behavior embeddings (the unsupervised cousin of your Similar Restaurants work), delivery-zone design, customer RFM segmentation for growth campaigns. *Naveen: the §7-S3 case is essentially a Swiggy/Zomato interview prompt; own it.*
- **Flipkart** — price-band/product clustering for assortment, seller segmentation, IVF cells in catalog-embedding search.
- **Games24x7** — player segmentation (engagement/stake-level archetypes) for CRM and responsible-play tiering; clustering of gameplay-pattern embeddings as the exploratory layer that *preceded* supervised collusion models — a clean "unsupervised exploration → supervised production" arc for your fraud story.

---

## 11. Coding From Scratch (NumPy only)

K-Means with k-means++ initialization, restarts, and empty-cluster handling — the complete interview version.

```python
import numpy as np

class KMeansScratch:
    def __init__(self, k=3, n_init=10, max_iter=300, tol=1e-6, seed=0):
        self.k, self.n_init, self.max_iter, self.tol = k, n_init, max_iter, tol
        self.rng = np.random.default_rng(seed)
        self.centroids, self.inertia_ = None, np.inf

    def _kpp_init(self, X):
        n = len(X)
        centroids = [X[self.rng.integers(n)]]          # first seed: uniform
        for _ in range(1, self.k):
            # D(x)^2 to nearest existing seed — the k-means++ distribution.
            d2 = np.min(((X[:, None, :] - np.array(centroids)[None]) ** 2)
                        .sum(-1), axis=1)
            probs = d2 / d2.sum()                      # far points more likely;
            centroids.append(X[self.rng.choice(n, p=probs)])  # ∝ D² not D: outlier-robust-ish
        return np.array(centroids)

    def _lloyd(self, X):
        C = self._kpp_init(X)
        prev_J = np.inf
        for _ in range(self.max_iter):
            # ----- Assignment: nearest centroid (vectorized n×k distances)
            d2 = ((X[:, None, :] - C[None]) ** 2).sum(-1)   # (n, k)
            labels = d2.argmin(1)
            J = d2[np.arange(len(X)), labels].sum()         # current inertia

            # ----- Update: mean per cluster, with empty-cluster rescue
            for j in range(self.k):
                pts = X[labels == j]
                if len(pts) == 0:
                    # Re-seed dead centroid at the worst-fit point —
                    # guarantees J decreases and revives the cluster.
                    C[j] = X[d2[np.arange(len(X)), labels].argmax()]
                else:
                    C[j] = pts.mean(0)                      # the optimal update

            if prev_J - J < self.tol * prev_J:              # relative improvement
                break
            prev_J = J
        return C, labels, J

    def fit(self, X):
        X = np.asarray(X, float)
        for _ in range(self.n_init):                        # restarts: keep best J
            C, labels, J = self._lloyd(X)
            if J < self.inertia_:
                self.centroids, self.labels_, self.inertia_ = C, labels, J
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        d2 = ((X[:, None, :] - self.centroids[None]) ** 2).sum(-1)
        return d2.argmin(1)                                 # O(kd) per point
```

Narration points that earn senior credit:
- **D² sampling line** — explain ∝ D² (not D, not uniform): balances coverage vs single-outlier capture, and carries the O(log k) guarantee.
- **The (n, k) broadcasted distance matrix** — note it's memory O(nk); at scale you'd chunk it or use ‖x‖² + ‖c‖² − 2Xᵀc (GEMM form) — saying the GEMM expansion is the GPU-awareness flag.
- **Empty-cluster rescue** — most from-scratch implementations crash here; re-seeding at the worst-fit point both fixes it and lowers J.
- **Restarts keep min-J** — tie back to local optima; note k-means++ reduces but doesn't eliminate the need.
- Extensions to offer: mini-batch update rule (per-centroid counts, decaying step), spherical variant (normalize X and C), and "this exact code, run per-subvector with k=256, is a PQ codebook trainer."

---

## 12. ML System Design Perspective

**Choose K-Means when:** you need scalable flat partitioning with a *servable* assignment function (segments as a real-time feature); vector quantization (IVF cells, PQ codebooks, embedding compression); a fast exploratory pass to summarize structure; micro-clustering as preprocessing (then hierarchical/manual on centroids).

**Avoid when:** shapes are non-convex or densities vary wildly (DBSCAN/HDBSCAN); soft membership or generative semantics needed (GMM); data is categorical/sparse without an embedding step; the business question is actually supervised (predict the outcome instead).

**Data requirements:** scaled or embedded features (the space *is* the model); outlier policy decided upfront; for cosine semantics, normalize and use spherical K-Means; enough data per intended cluster to make centroids stable.

**Latency:** assignment is O(kd) — microseconds; the rare model whose serving cost is genuinely negligible, which is exactly why it lives inside latency-critical ANN indexes.

**Scale limits:** effectively none with mini-batch + distributed/GPU implementations; the practical limits are statistical (distance concentration in high d — reduce/embed first) and organizational (segment-contract stability across refits).

---

## 13. Resume Discussion Angle

**Recommendation/retrieval systems:** The high-leverage move is connecting K-Means to your serving stack unprompted: "our Similar Restaurants embeddings were served through an IVF index — K-Means is the coarse quantizer that defines the cells, so index quality is a clustering problem; when we updated the encoder we rebuilt the quantizer, because the index is part of the model." That paragraph converts a 'basic algorithm' question into infrastructure depth. Expect the follow-up on (k, nProbe) tuning — answer with the recall@k-vs-p99 curve on *production* query distribution.

**Personalization/marketing ML:** "Tell me about a segmentation you built." Use the §7-S3 structure: feature-space decision → spherical mini-batch K-Means → k via silhouette + stability + activation capacity → archetype profiling → randomized campaign validation → drift-monitored serving with Hungarian relabeling across refits. The differentiators are *stability gating* and *outcome validation* — most candidates stop at the elbow plot.

**Fraud detection:** Position clustering as the exploratory layer of your Games24x7 arc: gameplay-embedding clusters surfaced suspicious cohorts → investigation produced labels → supervised XGBoost scaled enforcement. Expect "why not cluster-as-detector in production?" → instability across refits, no precision control, no per-case score — clusters generate hypotheses, supervised models generate decisions.

**The universal trap:** "Your clusters look great — how do you know they're real?" Never answer with inertia. The full-marks sequence: stability under resampling (ARI), silhouette structure, null-reference comparison (gap), and — decisive — differential behavior on held-out outcomes, ideally under a randomized treatment. Structure beats vibes; outcomes beat structure.

---
*Previous: Naive Bayes ← | Next: Principal Component Analysis →*

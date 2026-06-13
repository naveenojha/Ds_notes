# Chapter 18: GloVe (Global Vectors)
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

> GloVe is best understood *against* Word2Vec: same goal (embeddings from co-occurrence), opposite strategy (global explicit matrix factorization vs local implicit sampling). The chapter is short because most of the conceptual weight lives in the comparison — but the "ratios of co-occurrence probabilities" motivation and the weighted least-squares objective are the things to own.

---

## 1. Executive Summary (30 seconds)

GloVe learns word embeddings by explicitly factorizing the global word-word co-occurrence matrix. Its insight: the *ratio* of co-occurrence probabilities — how often word A appears with "ice" versus with "steam" — encodes meaning better than raw counts. It fits vectors so that the dot product of two word vectors approximates the log of their co-occurrence count, using a weighted least-squares objective that down-weights rare and caps frequent pairs. Where Word2Vec streams local context windows with implicit sampling, GloVe consumes pre-computed global statistics in one shot. The two produce comparable embeddings and, per Levy-Goldberg, are the same family — both factorize a co-occurrence statistic. GloVe's interview value is the global-vs-local framing and the elegant ratio motivation.

---

## 2. Interview Articulation (3–4 Minute Answer)

"GloVe — Global Vectors — was the answer to a clean question: Word2Vec learns great embeddings from *local* context windows, and classical methods like LSA factorize *global* co-occurrence counts, so why not get the best of both? GloVe explicitly uses global co-occurrence statistics but learns the embeddings the way Word2Vec does — by optimizing vectors, not by raw SVD.

The conceptual heart is the most elegant idea in word embeddings, and it's about *ratios*. Take 'ice' and 'steam'. Individually, both co-occur with lots of words. But look at the *ratio* of their co-occurrence probabilities with a probe word. With 'solid', ice's probability is much higher than steam's — the ratio is large. With 'gas', it flips — the ratio is small. With 'water', both co-occur a lot, ratio near one. With 'fashion', neither does, also near one. So the *ratio* of co-occurrence probabilities cleanly separates the relevant distinguishing words ('solid', 'gas') from the irrelevant ones ('water', 'fashion'), in a way that raw counts don't. GloVe's design goal is to make vector differences encode these ratios — which is also why it supports the same analogy arithmetic as Word2Vec.

From that ratio insight, the authors derive an objective: the dot product of two word vectors should approximate the *logarithm* of their co-occurrence count, plus bias terms. Why log? Because differences of logs are ratios — log(P_ik) − log(P_jk) is the log of the ratio — so making dot products match log-counts makes vector differences match log-ratios, which is exactly the structure they wanted. Then they wrap it in a **weighted** least-squares loss with a weighting function that does two things: it down-weights very rare co-occurrences, which are noisy, and it caps the influence of extremely frequent pairs like 'the-and', which would otherwise dominate. That weighting is the practical secret sauce — without it, frequent function words swamp the signal.

So mechanically: build the global co-occurrence matrix once by sweeping the corpus, then fit word vectors and context vectors by minimizing that weighted squared error between dot products and log-counts. Training operates on the non-zero entries of the co-occurrence matrix rather than streaming text windows.

The practical comparison with Word2Vec: they produce embeddings of very similar quality, and the choice is mostly about compute profile and data. GloVe builds the co-occurrence matrix up front — memory-heavy for huge vocabularies but then trains on aggregated statistics; Word2Vec streams windows and never materializes the global matrix. GloVe can be faster to *retrain* with the same statistics; Word2Vec is more naturally online. And here's the unifying result: Levy and Goldberg showed Word2Vec is implicitly factorizing a shifted-PMI matrix, while GloVe explicitly factorizes a log-count matrix — so they're the same species, just one implicit-and-local, the other explicit-and-global. Both are also the same family as LSA/SVD. That's the sentence that ties the whole embedding topic together.

GloVe shares Word2Vec's core limitation — it's a static embedding, one vector per word, no context disambiguation — so contextual models like BERT superseded both for tasks where polysemy matters. And like Word2Vec, GloVe is mostly used today as cheap pretrained static vectors when you need a lookup table rather than an inference network.

When to use: same niche as Word2Vec — cheap static embeddings, often via pretrained GloVe vectors as a strong off-the-shelf starting point; when you already have global co-occurrence statistics. When not: contextual meaning, OOV-heavy settings, or when a contextual encoder is affordable. Traps: not knowing the *ratio* motivation; not knowing *why* log-counts; forgetting the weighting function's role; and claiming GloVe is fundamentally different from Word2Vec when they're the same factorization family with different optimization."

---

## 3. Mathematical Foundation

**The ratio insight (the motivation to articulate):** let P(k|w) be the probability word k appears in word w's context. For probe words k, the *ratio* P(k|ice)/P(k|steam) discriminates meaning:
```
k = solid:   ratio ≫ 1   (relevant to ice, not steam)
k = gas:     ratio ≪ 1   (relevant to steam, not ice)
k = water:   ratio ≈ 1   (relevant to both)
k = fashion: ratio ≈ 1   (relevant to neither)
```
Ratios separate distinguishing words from common/irrelevant ones — raw probabilities don't. GloVe's objective is engineered so vector relationships encode these ratios.

**From ratios to the objective (the derivation sketch):** want a function F of word vectors whose value reflects the ratio P_ik/P_jk. Requiring F to depend on vector *differences* (w_i − w_j) and to be consistent under exchanging roles leads (after the authors' reasoning) to:
```
w_iᵀ w̃_k + b_i + b̃_k = log(X_ik)
```
where X_ik = co-occurrence count of words i and k, w̃/b̃ are context vectors/biases. **Why log:** log(P_ik) − log(P_jk) = log(P_ik/P_jk) — so matching dot products to log-counts makes vector *differences* encode log-*ratios*, the target structure.

**The weighted least-squares loss (the full objective):**
```
J = Σ_{i,k}  f(X_ik) · ( w_iᵀ w̃_k + b_i + b̃_k − log X_ik )²

weighting:  f(x) = (x / x_max)^α   if x < x_max,  else 1     (α ≈ 0.75, x_max ≈ 100)
```
The weighting f(X_ik) does the two essential jobs:
- f(0) = 0 and rises from there ⟹ **rare co-occurrences down-weighted** (noisy, would mislead) and zero counts skipped entirely (sum is over non-zeros only).
- f caps at 1 for x ≥ x_max ⟹ **frequent pairs ('the-of') don't dominate** the loss.
- α = 0.75 — the same flattening exponent as Word2Vec's negative sampling (not a coincidence; both reshape frequency influence).

**Training:** build the global co-occurrence matrix X once (one corpus sweep), then minimize J by AdaGrad over its non-zero entries. Two vectors per word (w, w̃); final embedding = w + w̃ (sum) typically.

**Co-occurrence weighting by distance:** when building X, nearer context words often contribute more (1/distance weighting within the window) — a detail that sharpens the statistics.

**The unifying result (Levy & Goldberg, restated for GloVe):** GloVe explicitly factorizes the log-co-occurrence matrix; SGNS implicitly factorizes shifted PMI; both are low-rank factorizations of a word-context co-occurrence statistic, in the same family as LSA/SVD (Ch. 11). The differences — global-explicit-LS vs local-implicit-sampling — change compute and weighting, not the fundamental object.

---

## 4. Step-by-Step Numerical Example

**4.1 The ratio mechanic.** Toy corpus counts (co-occurrence with probe words):
```
                 solid   gas    water   fashion
P(k | ice)       0.090   0.001  0.030   0.0001
P(k | steam)     0.001   0.080  0.030   0.0001

Ratios P(k|ice)/P(k|steam):
  solid:   0.090/0.001 = 90      → ≫1, distinguishes (ice-related)
  gas:     0.001/0.080 = 0.0125  → ≪1, distinguishes (steam-related)
  water:   0.030/0.030 = 1.0     → ≈1, common to both (uninformative)
  fashion: 0.0001/0.0001 = 1.0   → ≈1, irrelevant to both (uninformative)
```
The ratio cleanly flags *solid* and *gas* as the meaning-distinguishing context words and washes out *water* and *fashion* — which raw individual probabilities (all small) would not. This is GloVe's entire motivation in four numbers; walk it and the design follows naturally.

**4.2 One GloVe gradient step.** Single pair, 2-D vectors. w_i = [0.5, 0.2], w̃_k = [0.3, 0.4], biases b_i = b̃_k = 0; co-occurrence X_ik = 50, x_max = 100, α = 0.75.
```
Prediction:  w_iᵀw̃_k + b_i + b̃_k = 0.5·0.3 + 0.2·0.4 = 0.15 + 0.08 = 0.23
Target:      log(50) = 3.912
Error:       (0.23 − 3.912) = −3.682
Weight:      f(50) = (50/100)^0.75 = 0.5^0.75 = 0.595

Weighted gradient wrt w_i:  2·f·error·w̃_k = 2·0.595·(−3.682)·[0.3,0.4]
                                            = (−4.382)·[0.3,0.4] = [−1.315, −1.753]
Update (lr 0.05):  w_i ← [0.5,0.2] − 0.05·[−1.315,−1.753] = [0.566, 0.288]
```
The dot product was far below log(50), so the vectors are pushed to increase it (toward the co-occurrence target), weighted by f = 0.595 (a mid-frequency pair gets moderate influence). A rare pair (X=2) would get f ≈ 0.05 — barely nudged; a hyper-frequent pair (X≥100) caps at f=1. Contrast with Word2Vec's update (Ch. 17 §4.1): GloVe regresses dot products to *log-counts* with a *frequency weight*; Word2Vec does logistic classification of *sampled* pairs — same family, different machinery.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| Embedding dim | Vector capacity | Richer geometry, more memory; diminishing returns | Coarser | 100–300 typical; chosen by downstream metric |
| Window size | Co-occurrence context | Broader/topical co-occurrence | Tighter/syntactic | Same topic-vs-function semantics as Word2Vec |
| x_max | Frequency cap in f(x) | Higher cap ⟹ frequent pairs influence more | Lower ⟹ flattens influence sooner | "What does x_max do?" → caps frequent-pair dominance (~100) |
| α (weighting exponent) | Shape of f(x) | →1: linear up to cap | →0: flatter | 0.75 — the same flattening as Word2Vec's negatives |
| Distance weighting | Closer-word emphasis in X | — | — | 1/distance within window sharpens stats |
| Min count | Vocabulary floor | Smaller vocab, less rare noise | Bigger vocab | Frequency cutoff |
| Iterations (AdaGrad) | Training passes over X's non-zeros | Better fit | Undertrained | Operates on co-occurrence matrix, not raw text |

The senior framing: "GloVe's distinctive knobs are x_max and α — they define the weighting function that down-weights rare/noisy pairs and caps frequent ones. That weighting *is* GloVe's contribution over naive log-count factorization; everything else mirrors Word2Vec."

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training | Two phases: (1) build global co-occurrence matrix X (one corpus sweep, memory ∝ non-zero entries — large for big vocab); (2) factorize via AdaGrad over non-zeros. Phase 1 is the memory cost; phase 2 is fast on aggregated stats |
| Inference | None — static lookup table, identical to Word2Vec (microsecond retrieval) |
| Memory | Build-time: co-occurrence matrix (sparse, but large vocab → many non-zeros). Serve-time: V × dim, same as Word2Vec |
| Retraining | With cached co-occurrence statistics, re-factorizing is cheap; useful when you re-tune dimension/weighting without re-sweeping the corpus |
| Pretrained vectors | GloVe's main production use: download high-quality pretrained vectors (Common Crawl / Wikipedia) as off-the-shelf features — often a strong, zero-training baseline |
| Cold start / OOV | Same gap as Word2Vec (no vector for unseen words); fastText-style subwords or content fallback |
| Drift / alignment | Same as Word2Vec — co-occurrence drift needs retrains; cross-version vectors need Procrustes alignment (Ch. 11) |
| Monitoring | NN sanity checks, OOV rate, downstream recall@k/CTR, coverage |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What does GloVe stand for and what's its core idea?** Global Vectors; factorize the *global* word-word co-occurrence matrix so vector relationships encode co-occurrence *ratios*, combining global statistics (like LSA) with learned embeddings (like Word2Vec).
2. **What's the ratio insight?** Ratios of co-occurrence probabilities (P(k|w1)/P(k|w2)) distinguish meaning — large/small for discriminating probe words, ≈1 for common/irrelevant ones — better than raw counts. GloVe encodes ratios in vector geometry.
3. **Why does GloVe target log co-occurrence counts?** Because differences of logs are log-ratios; matching dot products to log-counts makes vector *differences* encode the co-occurrence *ratios* GloVe wants — and supports analogy arithmetic.
4. **What is the weighting function for and what does it do?** f(X_ik) down-weights rare (noisy) co-occurrences, skips zeros, and caps frequent pairs (x_max) so common words don't dominate the loss. The practical key to GloVe's quality.
5. **GloVe vs Word2Vec — the headline difference?** GloVe: global, explicit factorization of the full co-occurrence matrix, least-squares on log-counts. Word2Vec: local, implicit factorization via context-window sampling, logistic on sampled pairs. Comparable quality; same family.
6. **Is GloVe contextual or static?** Static — one vector per word, no context disambiguation; same limitation as Word2Vec; superseded by BERT for polysemy-sensitive tasks.
7. **How does GloVe relate to LSA/SVD?** Both factorize co-occurrence statistics; GloVe uses a *weighted* objective on log-counts (vs SVD's unweighted squared error on raw/normalized counts), which is why it outperforms plain LSA.
8. **Why might you use pretrained GloVe vectors?** A strong, free, off-the-shelf static embedding baseline (Common Crawl/Wikipedia) — zero training, instant features for downstream models when contextual embeddings aren't needed/affordable.
9. **What's the role of x_max and α?** They define f(x): x_max caps frequent-pair influence (~100); α (~0.75) shapes how rare-pair weight ramps up. Together they balance rare-noise vs frequent-dominance.
10. **Does GloVe support analogies?** Yes — same vector-arithmetic analogies as Word2Vec (king−man+woman≈queen), arising from the ratio/log-difference structure it's built to encode.
11. **What are the two vectors per word in GloVe?** A word vector and a context vector (with biases); typically summed for the final embedding (w + w̃) — empirically slightly better than using one.
12. **Which is more memory-intensive at train time, GloVe or Word2Vec?** GloVe — it builds the global co-occurrence matrix up front (many non-zeros for large vocab); Word2Vec streams windows without materializing it.
13. **Can GloVe handle OOV words?** No — like Word2Vec, no vector for unseen words; needs subword (fastText) or content fallback.
14. **When would you prefer GloVe over Word2Vec?** When you already have/can afford global co-occurrence statistics, want to retrain at different dimensions cheaply, or want strong pretrained vectors; practically often a toss-up — pick by downstream metric.
15. **Why did contextual models replace both?** Static embeddings give one vector per word regardless of context; contextual models (BERT) produce per-occurrence vectors, disambiguating polysemy — strictly more expressive when affordable.

### Advanced (15)

1. **Walk through the derivation from the ratio requirement to w_iᵀw̃_k = log X_ik − biases.** Require F((w_i−w_j)ᵀw̃_k) = P_ik/P_jk; homomorphism considerations (F = exp) give w_iᵀw̃_k = log P_ik; absorbing normalization into biases yields w_iᵀw̃_k + b_i + b̃_k = log X_ik. The log is forced by the ratio-to-difference requirement.
2. **Why weighted least squares rather than plain least squares on log-counts?** Plain LS treats all pairs equally — rare noisy counts mislead and frequent counts dominate. f(X) down-weights both extremes (zero weight for unseen, capped for frequent), which is exactly what makes GloVe beat unweighted log-count factorization (≈LSA).
3. **State the Levy-Goldberg unification across GloVe, SGNS, and SVD.** All factorize a word-context co-occurrence statistic: GloVe → log-counts (weighted LS, explicit/global), SGNS → shifted PMI (implicit/local sampling), SVD/LSA → (PP)MI or counts (closed-form). Same object, different weighting/optimization.
4. **Why does GloVe use global statistics while Word2Vec is local — what does each gain?** Global: uses all co-occurrence information at once, statistically efficient, deterministic given X. Local: streams windows, naturally online, never materializes a huge matrix, easy incremental updates. Statistical efficiency vs streaming/online flexibility.
5. **What's the significance of α = 0.75 appearing in both GloVe (weighting) and Word2Vec (negatives)?** Both flatten frequency influence by the same empirically-optimal exponent — evidence they're solving the same underlying problem (taming frequency dominance in co-occurrence) by different routes; a hint at the shared factorization.
6. **How would distance weighting in the co-occurrence matrix change the embeddings?** Weighting closer context words more (1/d) emphasizes tighter, more syntactic relationships; flat weighting emphasizes broader topical co-occurrence — analogous to window-size effects, but as a continuous weighting.
7. **Why sum the word and context vectors for the final embedding?** They're symmetric solutions of the same factorization (the model is near-symmetric in w and w̃); summing averages out noise and small asymmetries, giving a slightly better, more stable embedding than either alone.
8. **GloVe's objective is convex in the dot products but not jointly in the vectors — implication?** Like all bilinear factorizations (and MF, Ch. 11), it's non-convex in (w, w̃) jointly ⟹ local optima, init/seed sensitivity, no cross-run comparability without alignment — same caveats as Word2Vec and recsys MF.
9. **When does GloVe's up-front matrix construction become a bottleneck?** Very large vocabularies / huge corpora ⟹ the co-occurrence matrix has enormous numbers of non-zero entries, costing memory and a full corpus sweep; Word2Vec's streaming avoids this, which is why it's often preferred at extreme scale.
10. **How does GloVe handle the zero-count problem that plagues naive log-count factorization?** log(0) is undefined; GloVe's weighting f(0)=0 *excludes* zero co-occurrences from the sum entirely, sidestepping the issue (and implicitly treating unobserved pairs as missing, like recsys MF treats unrated items — the same MCAR-style handling).
11. **Could you factorize the co-occurrence matrix with SVD instead — and what would differ?** Yes — that's essentially LSA. Differences: SVD minimizes *unweighted* squared error (frequent pairs dominate, rare-noise mistreated) and works on raw/normalized counts; GloVe's weighted objective on log-counts is the targeted improvement. GloVe ≈ "SVD done right for co-occurrence."
12. **Anisotropy and post-processing — does GloVe need it too?** Yes — like all embeddings, GloVe vectors can be anisotropic (narrow cone), inflating cosine baselines; mean-centering + removing top PCs ("all-but-the-top") improves similarity/retrieval. Same practitioner detail as Word2Vec.
13. **How would you adapt GloVe to a recommendation/co-occurrence-of-items setting?** Build the item-item co-occurrence matrix from sessions/baskets (with the same weighting to tame popular items), factorize for item embeddings — the GloVe analogue of item2vec. Popular-item dominance is handled by f's cap, an elegant built-in debiasing.
14. **GloVe vs PMI-SVD (Levy-Goldberg's explicit method) — which is "more correct"?** Levy-Goldberg showed explicit PPMI-SVD matches or beats SGNS/GloVe with proper tuning — suggesting the *factorization target and weighting* matter more than the *optimization method*. The honest takeaway: these are tuning variants of one idea, not fundamentally different models.
15. **Why are static embeddings (GloVe/Word2Vec) still used despite contextual models?** Cost (no inference network — pure lookup), simplicity, strong pretrained availability, sufficiency for tasks where context doesn't matter (item co-occurrence, coarse features). The "table vs function" tradeoff from Ch. 17 applies identically.

### Staff-Level (10)

1. **GloVe or Word2Vec for a new embedding pipeline — how do you actually decide?** Largely a toss-up on quality; decide on engineering: GloVe if you want deterministic training on cached global stats and cheap re-tuning of dimension; Word2Vec if you need streaming/online updates or extreme-scale vocab where materializing the co-occurrence matrix is prohibitive. For most teams: use *pretrained* vectors of either, or skip both for a contextual encoder if quality is paramount and affordable. The decision is operational, not about embedding quality — saying that is the signal.
2. **A stakeholder insists GloVe is "more advanced" than Word2Vec because it uses global statistics. Correct them.** They're the same family (Levy-Goldberg): both factorize a co-occurrence statistic, GloVe explicitly/globally on log-counts, Word2Vec implicitly/locally on shifted PMI. "Global" isn't "more advanced" — it's a different compute profile with comparable results. The substantive frontier is *contextual* embeddings, where both are equally superseded. Reframing "advanced" into "different tradeoff" is the maturity marker.
3. **You inherit a pipeline using pretrained GloVe vectors; quality is mediocre on your domain. Options ranked?** (1) Check domain mismatch — generic GloVe (Wikipedia/Common Crawl) may not fit your jargon ⟹ train/fine-tune on in-domain corpus; (2) anisotropy post-processing (all-but-the-top); (3) move to in-domain Word2Vec/item2vec if the signal is behavioral; (4) upgrade to a (fine-tuned) contextual/sentence encoder if context matters and budget allows. Diagnose *domain fit* before swapping techniques — generic pretrained vectors failing on niche domains is the common, fixable cause.
4. **Design item embeddings via a GloVe-style approach and justify the weighting choice.** Build item-item co-occurrence from sessions; apply GloVe's weighted log-count factorization — the cap (x_max) automatically prevents popular items from dominating the geometry (a built-in popularity debiasing that item2vec needs explicit subsampling for). Justify: the weighting *is* the popularity-bias control. A precise reason to prefer the GloVe formulation for skewed catalogs.
5. **When is the deterministic, global nature of GloVe a production advantage?** Reproducibility (same X ⟹ same factorization target, fewer sampling-noise surprises across runs), auditability (the co-occurrence matrix is an inspectable artifact), and cheap re-tuning (re-factorize cached stats at new dimensions without re-sweeping). For regulated or reproducibility-critical pipelines, the explicit global object is easier to govern than streaming sampling. Naming governance/reproducibility is staff-level.
6. **Both embedding families fail your task because it needs context. How do you communicate the upgrade decision and its cost?** Explain the table-vs-function tradeoff: static vectors are a free lookup but can't disambiguate context; a contextual encoder is a per-input forward pass — strictly more capable, materially more expensive to serve. Quantify: latency/cost delta, expected quality gain (measured on a task eval set), and whether RAG/feature-level use even needs full contextuality. Recommend by measured downstream lift vs serving cost, not by technique novelty.
7. **Cross-version or cross-domain embedding comparison breaks (GloVe trained on two corpora). Root cause and fix.** Independent factorizations land in different bases (rotational ambiguity) ⟹ vectors/neighbors aren't comparable across runs or corpora. Fix: orthogonal Procrustes alignment to a shared space (Ch. 11) — the standard method for, e.g., cross-lingual embeddings or tracking semantic drift over time. Same lesson as Word2Vec; the alignment requirement is intrinsic to factorization-based embeddings.
8. **A teammate proposes plain SVD on the co-occurrence matrix instead of GloVe. When is each right?** SVD: exact, closed-form, optimal rank-k under *unweighted* squared error — fine if frequency is pre-normalized (e.g., PPMI-SVD, which Levy-Goldberg showed is competitive) and you want determinism. GloVe: weighted objective handling rare/frequent imbalance and zeros gracefully, at the cost of iterative non-convex training. With proper PPMI weighting, SVD nearly matches GloVe — so the choice is about tooling and tuning, not a quality chasm. Knowing PPMI-SVD competes is the deep cut.
9. **How would you evaluate GloVe vs Word2Vec vs contextual embeddings for *your* retrieval system?** Task-specific eval set (labeled query-item relevance from logs), measure recall@k / nDCG on your distribution, test tail/rare queries, validate downstream metric (CTR/deflection), and weigh serving cost (lookup vs forward pass). Generic intrinsic benchmarks (analogy/WordSim) don't transfer. "The benchmark is not your task" — the recurring evaluation discipline.
10. **Leadership wants to standardize on one embedding approach org-wide. Frame it.** Standardize the *interface and evaluation harness* (embedding store, ANN serving, task-eval protocol, alignment/versioning policy), not the algorithm — GloVe, Word2Vec, item2vec, and contextual encoders are interchangeable behind a "produce vectors" contract, and the right choice is per-task (signal type, latency, data). The reframe from "pick one model" to "standardize the platform, choose models per task" is the staff-level answer — identical in spirit to the GBM-library standardization answer (Ch. 5).

---

## 8. Comparison Section

**GloVe vs Word2Vec (THE comparison — likely asked verbatim):**

| | GloVe | Word2Vec (SGNS) |
|---|---|---|
| Statistics used | **Global** co-occurrence matrix | **Local** context windows |
| Factorization | **Explicit** (build matrix, factorize) | **Implicit** (sampling, per Levy-Goldberg) |
| Objective | Weighted least squares on **log-counts** | Logistic on **sampled pairs** (shifted PMI) |
| Frequency handling | Weighting function f(x) (cap + ramp) | Subsampling + freq^0.75 negatives |
| Training | AdaGrad over co-occurrence non-zeros | SGD streaming over windows |
| Online updates | Awkward (rebuild matrix) | Natural (stream) |
| Train memory | High (global matrix) | Low (streaming) |
| Quality | Comparable | Comparable |
| Both | Static, OOV-limited, analogy-capable, same family | Static, OOV-limited, analogy-capable, same family |

**One-liner:** "GloVe and Word2Vec are the same idea — factorize co-occurrence — executed globally-and-explicitly vs locally-and-implicitly; they perform comparably, and the choice is about compute profile and whether you need online updates."

**GloVe vs LSA/SVD (Ch. 11):** both factorize co-occurrence; GloVe's *weighted* log-count objective (down-weight rare, cap frequent) is the targeted fix for plain SVD's unweighted squared error — "SVD done right for co-occurrence." PPMI-SVD with proper weighting nearly closes the gap.

**GloVe vs contextual (BERT/sentence-transformer):** static lookup vs contextual function; identical static-embedding limitations as Word2Vec; superseded for polysemy/context-sensitive tasks, retained for cheap features.

**GloVe-for-items vs item2vec:** item-item co-occurrence factorization (GloVe weighting auto-debiases popularity) vs skip-gram on sessions (needs explicit subsampling) — same target, GloVe's weighting is a cleaner popularity control for skewed catalogs.

---

## 9. Common Mistakes

**Candidate mistakes:**
- Not knowing the *ratio* motivation (the conceptual core of GloVe).
- Not knowing *why* log-counts (log-differences = log-ratios).
- Forgetting the weighting function's dual role (down-weight rare, cap frequent).
- Claiming GloVe is fundamentally different from / superior to Word2Vec (same family — Levy-Goldberg).
- Calling GloVe contextual (it's static).

**Production mistakes:**
- Using generic pretrained GloVe on a niche domain and blaming the method (domain mismatch — train in-domain).
- Comparing GloVe vectors across corpora/runs without Procrustes alignment.
- Raw anisotropic vectors in cosine retrieval without post-processing.
- Materializing a giant co-occurrence matrix at extreme vocab scale when streaming Word2Vec was the fit.

**Modeling mistakes:**
- Plain SVD on raw counts (unweighted) where GloVe's weighting was needed (frequent words dominate).
- Treating co-occurrence ratios as causal/semantic ground truth rather than corpus statistics.
- Choosing dimension by intrinsic benchmarks instead of downstream metric.
- Ignoring that zeros must be excluded (log(0)) — naive log-count factorization breaks without it.

---

## 10. Real Industry Use Cases

- **Stanford / academia** — GloVe's origin (Pennington, Socher, Manning); the pretrained Common Crawl / Wikipedia vectors became a ubiquitous off-the-shelf NLP resource.
- **Google / Amazon / Meta** — pretrained static embeddings (GloVe or Word2Vec) as features in pre-BERT NLP pipelines and as cheap baselines; largely migrated to contextual embeddings where quality matters, retained where cost/latency dominates.
- **Netflix / Uber** — static embeddings as features in classical pipelines and as inputs to downstream models; co-occurrence factorization patterns in item/entity embeddings.
- **Swiggy/Zomato** — pretrained or in-domain static embeddings for text features (reviews, queries) as cheap signals; the GloVe-style item-item co-occurrence factorization is a viable alternative to item2vec for restaurant/dish similarity (with built-in popularity debiasing via the weighting). *Naveen: GloVe is the explicit-factorization sibling of the Word2Vec/item2vec you used in Similar Restaurants — be ready to compare them and to note GloVe's weighting as a cleaner popularity-bias control for skewed catalogs.*
- **Flipkart** — static text embeddings as features; co-occurrence item embeddings for discovery.
- **Games24x7** — static embeddings as cheap behavioral/text features feeding churn/fraud models where a contextual encoder would be overkill.

*Honest framing for your resume: GloVe is less likely to be a system you built than a comparison point. Its value in interviews is articulating the global-vs-local distinction against Word2Vec and the unifying factorization view — which directly elevates your Similar Restaurants embedding story.*

---

## 11. Coding From Scratch (NumPy only)

GloVe training over a precomputed co-occurrence matrix — the weighted least-squares factorization. Shows the build-then-factorize structure and the weighting function.

```python
import numpy as np

class GloVeScratch:
    """Factorize a global co-occurrence matrix:
       minimize  Σ f(X_ij) (w_i·w̃_j + b_i + b̃_j − log X_ij)²
       The weighting f down-weights rare pairs and caps frequent ones."""
    def __init__(self, vocab_size, dim=50, x_max=100, alpha=0.75, lr=0.05, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.dim, self.x_max, self.alpha, self.lr = \
            vocab_size, dim, x_max, alpha, lr
        # Two vectors + two biases per word (word role and context role).
        self.W  = (rng.random((vocab_size, dim)) - 0.5) / dim   # word vectors
        self.Wt = (rng.random((vocab_size, dim)) - 0.5) / dim   # context vectors
        self.b  = np.zeros(vocab_size)
        self.bt = np.zeros(vocab_size)

    def _weight(self, x):
        # f(x) = (x/x_max)^alpha capped at 1. f(0)=0 ⟹ zeros excluded;
        # cap ⟹ frequent pairs ('the-of') don't dominate the loss.
        return np.where(x < self.x_max, (x / self.x_max) ** self.alpha, 1.0)

    def train(self, cooccur, epochs=50):
        # cooccur: list of (i, j, X_ij) for NON-ZERO co-occurrences only
        #          (log(0) is undefined; zeros are simply absent).
        for _ in range(epochs):
            np.random.shuffle(cooccur)
            total_loss = 0.0
            for i, j, x in cooccur:
                # Prediction vs target (log of the co-occurrence count).
                pred = self.W[i] @ self.Wt[j] + self.b[i] + self.bt[j]
                diff = pred - np.log(x)                 # the regression residual
                fw = self._weight(x)                    # frequency weight
                total_loss += fw * diff * diff

                # Weighted gradients (factor of 2 folded into lr).
                g = fw * diff                           # weighted error
                grad_Wi  = g * self.Wt[j]
                grad_Wtj = g * self.W[i]
                # Update word/context vectors and biases.
                self.W[i]  -= self.lr * grad_Wi
                self.Wt[j] -= self.lr * grad_Wtj
                self.b[i]  -= self.lr * g
                self.bt[j] -= self.lr * g
        return self

    def embeddings(self):
        # Final embedding = word + context vectors (symmetric solution; averages noise).
        return self.W + self.Wt

    def most_similar(self, idx, top_n=10):
        E = self.embeddings()
        v = E[idx]
        sims = (E @ v) / (np.linalg.norm(E, axis=1) * np.linalg.norm(v) + 1e-9)
        return np.argsort(sims)[::-1][1:top_n+1]        # ANN index in production
```

Narration points that earn senior credit:
- **`_weight` function** — point at it: "this is GloVe's whole contribution over plain log-count factorization. f(0)=0 excludes unseen pairs (log(0) is undefined), the ramp down-weights rare noisy pairs, the cap at x_max stops 'the-of' from dominating. α=0.75 — same flattening as Word2Vec's negatives."
- **`diff = pred − log(x)`** — "we regress the dot product to the *log* co-occurrence count; log because vector *differences* then encode log-*ratios*, which is the ratio insight made geometric."
- **Non-zero co-occurrences only** — "we never touch zeros (can't take log(0)); like recsys MF, unobserved pairs are missing, not negative."
- **`embeddings() = W + Wt`** — "sum the word and context vectors; they're symmetric solutions of the same factorization, and summing averages out noise."
- **Two-phase structure** — "in practice phase 1 builds the global co-occurrence matrix in one corpus sweep; this code is phase 2, the factorization. Word2Vec collapses both into streaming."
- **Extensions to offer:** AdaGrad (GloVe's actual optimizer — per-parameter rates suit the skewed gradient frequencies), distance-weighted co-occurrence counts, and "swap word-word for item-item co-occurrence and the cap auto-debiases popular items — the GloVe answer to item2vec's popularity problem."

---

## 12. ML System Design Perspective

**Choose GloVe when:** you want deterministic factorization on cached global co-occurrence statistics; strong pretrained static vectors as a zero-training baseline; cheap re-tuning of embedding dimension without re-sweeping the corpus; an item-item co-occurrence embedding where the weighting's built-in popularity cap is desirable; reproducibility/auditability of the co-occurrence artifact matters.

**Avoid when:** context-dependent meaning is required (contextual encoders); streaming/online embedding updates are needed (Word2Vec streams more naturally); extreme vocab scale makes materializing the global matrix prohibitive; OOV coverage is critical (subword/content fallback).

**Data requirements:** a corpus to build global co-occurrence statistics (or sessions for item-item); enough co-occurrences per pair for stable estimates; window/distance-weighting decisions defining the statistics; min-count floor.

**Latency:** inference-free static lookup (microseconds), identical to Word2Vec; similarity via ANN over the embedding matrix. Build-time cost is the global-matrix construction, not serving.

**Scale limits:** build-time memory (global co-occurrence non-zeros) is the ceiling at huge vocab; otherwise comparable to Word2Vec. Same downstream constraints: drift (retrain), alignment (Procrustes), anisotropy (post-process), cold start (subwords/content).

---

## 13. Resume Discussion Angle

**The comparison is the value (be ready for it verbatim):** "Word2Vec vs GloVe?" is a near-guaranteed question if embeddings are on your resume. The full-marks answer: same family (factorize co-occurrence), GloVe global-explicit on log-counts with a weighting function, Word2Vec local-implicit on sampled pairs (shifted PMI per Levy-Goldberg); comparable quality; choose by compute profile and online-update needs. Delivering the *unification* — not a list of surface differences — is what separates someone who memorized two algorithms from someone who understands the embedding family.

**Elevating Similar Restaurants:** position GloVe as the explicit-factorization sibling of the item2vec you actually used. The strong line: "we used Word2Vec/item2vec on user sessions for behavioral similarity; a GloVe-style item-item co-occurrence factorization was an alternative we considered — its weighting function would have auto-debiased popular restaurants, which we instead handled with subsampling." That shows you knew the design space and made a deliberate choice, which reads as senior judgment rather than defaulting to the famous algorithm.

**The unifying-view flourish (high signal):** when embeddings come up anywhere, the sentence that lands at Applied Scientist level — "GloVe, Word2Vec, and SVD/LSA are all factorizing a co-occurrence statistic; the recsys versions are item2vec and matrix factorization (Ch. 11), and the modern version is contextual encoders (Ch. 16)." Showing the *continuous lineage* from LSA → Word2Vec/GloVe → contextual → your two-tower retrieval demonstrates field-level understanding, which is exactly what distinguishes a Staff/AS candidate from a strong senior one.

**The universal trap:** "Why use static embeddings at all in 2026?" Same answer as Word2Vec, and don't concede reflexively: static embeddings are a free lookup table with zero inference cost, sufficient when context doesn't matter (behavioral item similarity, coarse features), and available as strong pretrained vectors; contextual encoders are a per-input function — strictly better when polysemy/context matters and you can afford the forward pass. The right choice matches *method to signal type and serving budget* — the judgment, not the technique's age, is what's being graded.

---
*Previous: Word2Vec ← | Next batch: Learning to Rank →*

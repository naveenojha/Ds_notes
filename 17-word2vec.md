# Chapter 17: Word2Vec
### ML Interview Handbook — Senior DS / Staff MLE / Applied Scientist

> Directly relevant to your Similar Restaurants work: Word2Vec is where the "items as embeddings from co-occurrence" idea was born, and item2vec/prod2vec apply it almost verbatim to recommendation. Own the skip-gram + negative-sampling objective, *why* it produces useful geometry, and the "neural embeddings are implicit matrix factorization" result that ties it to Chapter 11.

---

## 1. Executive Summary (30 seconds)

Word2Vec learns dense vector representations of words from raw text using a shallow neural network and one principle: **words appearing in similar contexts get similar vectors** (the distributional hypothesis). The dominant variant, skip-gram with negative sampling, trains by predicting context words from a target word — reframed as a cheap binary classification (real context pair vs sampled noise pairs) so you never compute the expensive full-vocabulary softmax. The resulting vectors capture semantic and syntactic structure geometrically, famously supporting analogies via vector arithmetic. Critically for your domain: swap "words in sentences" for "items in user sessions" and you get item2vec/prod2vec — the embedding backbone of co-occurrence-based recommendation, including the behavioral half of Similar Restaurants.

---

## 2. Interview Articulation (3–4 Minute Answer)

"Word2Vec is the model that made dense word embeddings practical and kicked off the representation-learning era in NLP. The whole thing rests on one linguistic idea — the distributional hypothesis: you shall know a word by the company it keeps. Words that appear in similar contexts tend to mean similar things. So if you can learn a vector for each word such that words appearing in similar contexts end up with similar vectors, you've captured meaning geometrically, with no labels — just raw text.

There are two architectures. **Skip-gram** takes a target word and tries to predict the words around it. **CBOW** does the reverse — predict the target from the surrounding context. Skip-gram works better for rare words and small data; CBOW is faster. Skip-gram is the one to discuss.

Here's the elegant part — the optimization trick that made it scale. The naive skip-gram objective is: given the target word, produce a probability distribution over the *entire vocabulary* for what the context word is, via softmax. But the vocabulary is hundreds of thousands of words, so that softmax — and especially its gradient, which sums over the whole vocabulary — is brutally expensive at every training step. **Negative sampling** sidesteps this completely. Instead of 'which of 200,000 words is the context?', you ask a much cheaper question: 'is *this* word a real context word — yes or no?' For each true target-context pair, you sample a handful of random 'negative' words that aren't the context, and train a binary classifier to score the real pair high and the noise pairs low. You've replaced one expensive softmax over the whole vocabulary with a few cheap logistic regressions. That's the trick that made Word2Vec train on billions of words.

There's a subtlety in how you sample negatives that interviewers like: you don't sample uniformly, you sample proportional to word frequency raised to the three-quarters power. That dampens the dominance of extremely common words while still sampling them more than rare ones — empirically the sweet spot. And there's subsampling of frequent words too, to stop 'the' and 'of' from drowning out signal.

Each word actually gets *two* vectors during training — one as a target, one as a context word — and typically you keep the target vectors as the embeddings, or average them. The geometry that emerges is remarkable: not just similar words clustering, but *linear structure* — the famous king minus man plus woman lands near queen. That works because the training pushes analogous relationships into roughly parallel vector offsets; it's not magic, it falls out of co-occurrence regularities, and it's imperfect — but it demonstrated that embeddings encode relational structure, not just similarity.

A deep result worth knowing: Levy and Goldberg showed that skip-gram with negative sampling is *implicitly factorizing* a matrix of shifted pointwise mutual information between words and contexts. So Word2Vec, GloVe, and classical LSA/SVD are all the same family — factorizing a co-occurrence statistic — just with different weightings and optimization. That connects this whole topic back to matrix factorization.

The defining limitation, and the reason BERT replaced it: Word2Vec embeddings are **static** — one fixed vector per word regardless of context. 'Bank' by a river and 'bank' for money get the same vector. Contextual models like BERT produce a different vector per occurrence, which is strictly more powerful. But static embeddings are far cheaper, need no inference network, and are still excellent when you just need a lookup table of vectors — which is most recommendation and retrieval preprocessing.

For my domain this is central: replace 'words in sentences' with 'items in user sessions or baskets' and you get item2vec or prod2vec — items that co-occur in user behavior get similar vectors. That's exactly the behavioral-similarity half of Similar Restaurants, complementing the content embeddings from a sentence-transformer.

When to use: you need cheap static embeddings, a lookup table, co-occurrence-based item embeddings, or you're resource-constrained. When not: you need context-dependent meaning, or a pretrained contextual model is available and affordable. Traps: confusing skip-gram with CBOW directions; not knowing why negative sampling exists (softmax cost); forgetting the 3/4-power sampling; calling embeddings contextual when they're static; and not knowing the implicit-PMI-factorization result that unifies the embedding family."

---

## 3. Mathematical Foundation

**Distributional hypothesis (the premise):** a word's meaning is captured by its context distribution. Operationalized: learn vectors so that words with similar contexts have high vector similarity.

**Skip-gram objective (predict context from target):** maximize the probability of context words given the center word over the corpus:
```
maximize  (1/T) Σ_t Σ_{−c ≤ j ≤ c, j≠0}  log p(w_{t+j} | w_t)
```
**Naive softmax (the expensive part):**
```
p(w_O | w_I) = exp(v'_{w_O}ᵀ v_{w_I}) / Σ_{w=1}^{V} exp(v'_wᵀ v_{w_I})
```
v = input (target) vector, v' = output (context) vector. The denominator sums over the **entire vocabulary V** ⟹ each gradient step is O(V) — infeasible for V ~ 10⁵–10⁶.

**Negative sampling (the fix — derive this):** replace the V-way softmax with binary logistic classification. For a true pair (w_I, w_O) and k sampled negatives w_n ~ P_n(w):
```
log σ(v'_{w_O}ᵀ v_{w_I})  +  Σ_{n=1}^{k} E_{w_n ~ P_n} [ log σ(−v'_{w_n}ᵀ v_{w_I}) ]
```
Maximize this: push real pairs' dot products up (σ→1), sampled-noise pairs' down (σ→0). Cost per step: O(k) instead of O(V), with k ≈ 5–20. **This is k+1 logistic regressions replacing one V-way softmax** — the line to say.

**Negative sampling distribution (the 3/4 trick):**
```
P_n(w) ∝ freq(w)^0.75
```
The 0.75 exponent flattens the frequency distribution — common words sampled less than their raw frequency (so they don't dominate negatives), rare words more than theirs. Empirically optimal; an interview favorite.

**Subsampling of frequent words:** discard word w with probability P(w) = 1 − √(t / freq(w)) (t ≈ 10⁻⁵) ⟹ down-weights "the/of/a", speeds training, improves rare-word quality.

**Hierarchical softmax (the alternative to negative sampling):** organize vocabulary as a binary (Huffman) tree; p(word) = product of binary decisions along the root-to-leaf path ⟹ O(log V) instead of O(V). Negative sampling usually wins for frequent words / large data; hierarchical softmax for rare words / smaller vocab.

**CBOW vs skip-gram:**
```
Skip-gram:  target → predict each context word   (better for rare words, small data)
CBOW:       average(context) → predict target     (faster, smoother for frequent words)
```

**Two vectors per word:** v_w (as target) and v'_w (as context). Final embedding = v_w (common) or (v_w + v'_w)/2.

**The implicit-MF result (Levy & Goldberg — the unifying theorem):** skip-gram with negative sampling implicitly factorizes the word-context **shifted PMI** matrix:
```
v_wᵀ v'_c ≈ PMI(w, c) − log k    where PMI(w,c) = log[ P(w,c) / (P(w)P(c)) ]
```
⟹ Word2Vec is matrix factorization of a co-occurrence statistic ⟹ same family as GloVe (Ch. 18) and LSA/SVD (Ch. 11), differing in weighting and optimization. **"Neural embeddings are implicit SVD of a PMI matrix" closes the loop with the SVD chapter — say it.**

**Analogy geometry:** vec(king) − vec(man) + vec(woman) ≈ vec(queen) works because consistent relational co-occurrence patterns become roughly-parallel offset vectors in the learned space. A demonstration of *relational* structure, not just clustering — and known to be imperfect/dataset-dependent.

---

## 4. Step-by-Step Numerical Example

**4.1 One negative-sampling update (the mechanics).** Tiny: target vector v (2-D) = [0.5, 0.5]; true context vector v'_pos = [0.4, 0.6]; one negative v'_neg = [−0.3, 0.2]. Learning rate η = 0.1.
```
Positive pair score:  v·v'_pos = 0.5·0.4 + 0.5·0.6 = 0.50 ; σ(0.50) = 0.622
  Target label 1 ⟹ gradient factor (σ − 1) = −0.378 (push score UP)
Negative pair score:  v·v'_neg = 0.5·(−0.3)+0.5·0.2 = −0.05 ; σ(−0.05) = 0.488
  Target label 0 ⟹ gradient factor (σ − 0) = +0.488 (push score DOWN)

Update target vector v (gradients from both pairs):
  ∂L/∂v = (σ_pos − 1)·v'_pos + (σ_neg − 0)·v'_neg
        = (−0.378)·[0.4,0.6] + (0.488)·[−0.3,0.2]
        = [−0.151,−0.227] + [−0.146, 0.098] = [−0.297, −0.129]
  v ← v − η·∂L/∂v = [0.5,0.5] − 0.1·[−0.297,−0.129] = [0.530, 0.513]
```
The target vector moved *toward* the true context's direction and *away* from the negative's — over millions of such updates, co-occurring words converge in vector space. Note the (σ − label) form: it's identical to logistic regression's gradient (Ch. 2) — because negative sampling literally *is* logistic regression on the dot product. Pointing that out is a strong connect.

**4.2 The cost contrast (why negative sampling).** Vocabulary V = 100,000, k = 5 negatives.
```
Naive softmax gradient:    touches all 100,000 output vectors per step
Negative sampling:         touches 1 positive + 5 negatives = 6 output vectors per step
⟹ ~16,000× fewer vector updates per step.
```
That ratio *is* the reason Word2Vec could train on billions of words on commodity hardware — quote it.

---

## 5. Hyperparameters

| Hyperparameter | What it does | Increase → | Decrease → | Interview probe |
|---|---|---|---|---|
| Embedding dim | Vector capacity | Richer geometry, more compute/memory; overfit on small data | Coarser, lossy | Typical 100–300; "how chosen?" → downstream task metric, not intrinsic |
| Window size | Context breadth | Captures more *topical* similarity (broad) | More *syntactic*/functional similarity (tight) | "Big vs small window semantics?" → topic vs function — a favorite |
| Negatives k | Noise samples per positive | Better estimates, slower; large data needs fewer | Faster, noisier | k≈5–20; small data → more, big data → fewer |
| Neg-sampling exponent | Frequency flattening | →1: raw frequency (commons dominate) | →0: uniform | 0.75 is the empirical sweet spot |
| Subsample threshold t | Frequent-word down-weighting | More aggressive (drop commons) | Keep commons | ~10⁻⁵; improves rare-word vectors + speed |
| Min count | Vocabulary floor | Smaller vocab, drops rare noise | Bigger vocab, more rare-word noise | Frequency cutoff for inclusion |
| Epochs | Training passes | Better convergence; overfit risk | Undertrained | Embeddings need several passes |
| Architecture | Skip-gram vs CBOW | — | — | "Which for rare words / small data?" → skip-gram |
| SG-NS vs hier. softmax | Optimization | — | — | NS for frequent/large; HS for rare/small |

The senior framing: "window size is the most *interesting* knob — small windows give functional/syntactic neighbors (words that play the same role), large windows give topical neighbors (words about the same thing). For item2vec, window = session scope, and it controls whether you learn 'substitutes' vs 'complements' — a real product decision."

---

## 6. Production Perspective

| Aspect | Detail |
|---|---|
| Training | O(corpus × k × dim) with negative sampling — linear, parallelizable (Hogwild-style async SGD); trains on billions of tokens on CPU clusters historically |
| Inference | **None** — embeddings are a lookup table (word/item → vector); zero forward pass, microsecond retrieval. The cheapness vs contextual models is the whole production appeal |
| Memory | V × dim floats (e.g., 1M items × 200 = 200M floats); store in a key-value/embedding store |
| Serving pattern | Embeddings feed an ANN index (Ch. 7/9) for similarity, or as features into a downstream ranker — precompute once, reuse everywhere |
| Cold start | No vector for unseen words/items (out-of-vocabulary) — the static-embedding gap; fastText (subword n-grams) or content fallback mitigates |
| Drift / retraining | Vocabulary and co-occurrence drift (new items, changing behavior) ⟹ periodic retrains; **embedding-space instability across retrains** — vectors aren't comparable run-to-run without alignment (Procrustes, Ch. 11) |
| Monitoring | Nearest-neighbor sanity checks, OOV rate, downstream metric (recall@k / CTR), coverage of new items |
| The coupling rule (again) | Embeddings + ANN index + downstream model are one versioned artifact; retrain embeddings ⟹ re-index ⟹ revalidate |

---

## 7. Interview Follow-up Questions

### Medium (15)

1. **What is the distributional hypothesis?** A word's meaning is characterized by its contexts ("know a word by the company it keeps"); Word2Vec operationalizes it by learning vectors so similar-context words are similar.
2. **Skip-gram vs CBOW?** Skip-gram predicts context from target (better for rare words, small data); CBOW predicts target from averaged context (faster, smoother for frequent words).
3. **Why is the naive softmax expensive?** Its denominator (and gradient) sum over the entire vocabulary V ⟹ O(V) per step ⟹ infeasible for large V.
4. **What is negative sampling?** Replace the V-way softmax with binary logistic classification: score true (target, context) pairs high and k sampled noise pairs low ⟹ O(k) per step.
5. **Why sample negatives ∝ freq^0.75?** Flattens the frequency distribution — common words sampled less than raw frequency, rare words more — empirically the best balance.
6. **What does subsampling frequent words do?** Probabilistically drops very common words ("the", "of") ⟹ less noise, faster training, better rare-word vectors.
7. **Why two vectors per word?** Separate target and context representations stabilize training (a word predicting itself as its own context is avoided); final embedding uses the target vector or the average.
8. **How do analogies work (king−man+woman≈queen)?** Consistent relational co-occurrence patterns become roughly-parallel offset vectors; analogy = vector arithmetic. Demonstrates relational structure; imperfect in practice.
9. **What's the key limitation of Word2Vec?** Static embeddings — one vector per word regardless of context; can't disambiguate polysemy ("bank"). Contextual models (BERT) fix this.
10. **Word2Vec vs one-hot encoding?** One-hot: sparse, V-dimensional, no similarity (all words orthogonal). Word2Vec: dense, low-dim, similarity encoded geometrically — captures relationships one-hot can't.
11. **How do you handle out-of-vocabulary words?** Word2Vec can't (no vector); fastText decomposes into character n-grams ⟹ builds OOV vectors from subwords; or fall back to content features.
12. **What is item2vec / prod2vec?** Word2Vec applied to items: treat user sessions/baskets as "sentences" and items as "words" ⟹ co-occurring items get similar vectors ⟹ behavioral item embeddings for recommendation.
13. **How do you choose embedding dimension?** By downstream task performance (recall@k, CTR), not intrinsic quality; 100–300 typical; bigger isn't always better (overfit on small corpora).
14. **What does window size control?** Context breadth: small windows → functional/syntactic neighbors (same role), large windows → topical neighbors (same subject). A semantic knob.
15. **Hierarchical softmax vs negative sampling?** Both avoid the full softmax: HS uses a binary tree (O(log V), better for rare words/small vocab); NS uses sampled binary classification (better for frequent words/large data).

### Advanced (15)

1. **Derive the negative-sampling gradient and connect it to logistic regression.** Objective term log σ(v'ᵀv) + Σ log σ(−v'_nᵀv); ∂/∂v = (σ(v'ᵀv) − 1)v' + Σ(σ(v'_nᵀv) − 0)v'_n — the (σ − label) form *is* logistic regression's gradient (Ch. 2). Negative sampling = per-pair logistic regression on dot products.
2. **State and explain the Levy-Goldberg implicit-MF result.** SGNS implicitly factorizes the shifted-PMI matrix: v_wᵀv'_c ≈ PMI(w,c) − log k. So Word2Vec ≈ MF of a co-occurrence statistic ⟹ same family as GloVe and LSA/SVD; the differences are weighting and optimization, not the underlying object.
3. **Why ^0.75 specifically — what's the effect on PMI?** Flattening the negative distribution changes the implicit matrix being factorized (shifts the effective PMI), down-weighting frequent-word dominance; 0.75 was found empirically to give the best downstream geometry — a hyperparameter of the implicit factorization.
4. **Why does skip-gram beat CBOW on rare words?** Skip-gram treats each (target, context) as a separate training example, so a rare target gets multiple distinct gradient signals; CBOW averages contexts, diluting rare-word signal into the average. More effective examples per rare word.
5. **What information does the *output* (context) vector carry vs the input vector?** They factor the co-occurrence matrix asymmetrically (word role vs context role); for symmetric similarity tasks, averaging or summing them often helps; keeping both enables asymmetric uses (e.g., directional recommendation).
6. **How does window size change the *type* of similarity, mechanistically?** Small window ⟹ co-occurrence is dominated by immediate syntactic neighbors ⟹ vectors cluster by grammatical function; large window ⟹ co-occurrence reflects document topic ⟹ vectors cluster by subject. The window defines what "context" the implicit PMI matrix counts.
7. **fastText's subword extension — mechanism and benefit.** Represent each word as a bag of character n-grams; the word vector is the sum of its n-gram vectors ⟹ morphology-aware, builds OOV vectors compositionally, better for morphologically rich languages and rare words.
8. **Why are Word2Vec spaces not comparable across training runs, and the fix?** Random init + SGD + rotational/permutation freedom ⟹ each run finds a different but equivalent basis; cosine across runs is meaningless without alignment. Fix: orthogonal Procrustes (Ch. 11) to rotate one space onto another — essential for temporal/cross-corpus embedding comparison.
9. **Anisotropy of embedding spaces — what is it and why care?** Learned embeddings often occupy a narrow cone (not isotropic) ⟹ cosine similarities are compressed/biased; post-processing (mean-centering, removing top principal components — "all-but-the-top") improves similarity quality. Relevant whenever you use raw embeddings for retrieval.
10. **How would you build embeddings for a bipartite/graph structure (users-items)?** Random-walk-based methods (DeepWalk, node2vec) generate "sentences" by walking the graph, then run skip-gram ⟹ graph embeddings. node2vec's biased walks interpolate between BFS (structural/role similarity) and DFS (community/homophily) — the graph analogue of window size.
11. **What does negative sampling assume about the noise distribution, and when does it break?** Assumes negatives are "not context" — but sampled negatives can be *false negatives* (actually related, just not in this window), injecting noise; matters in dense co-occurrence settings (small vocab, recsys with popular items). Mitigations: popularity-aware/hard-negative strategies.
12. **Connection between Word2Vec negatives and contrastive learning / two-tower training.** Negative sampling is early contrastive learning: pull positives together, push sampled negatives apart. Modern two-tower retrieval uses in-batch negatives and hard-negative mining — the same principle (and the SVM margin / support-vector connection from Ch. 6). A strong unifying answer.
13. **Why might averaging Word2Vec vectors for a sentence be a poor sentence embedding?** Loses order and composition, dominated by frequent/high-norm words, ignores context-dependent meaning; weighted schemes (SIF: smooth inverse frequency + remove top PC) help, but contextual encoders (sentence-transformers) are strictly better when affordable. Explains the move to Ch. 16.
14. **How does the implicit-MF view inform hyperparameter choices?** Since SGNS factorizes shifted PMI, k (negatives) controls the PMI shift (−log k), dimension controls factorization rank, subsampling/exponent reshape the matrix — viewing them as MF knobs (rank, weighting, shift) gives principled rather than trial-and-error tuning.
15. **GloVe vs Word2Vec at the objective level (preview Ch. 18).** GloVe explicitly factorizes the log-co-occurrence matrix with a weighted least-squares objective (global counts); SGNS implicitly factorizes shifted PMI via local sampling. Same family, global-explicit vs local-implicit; comparable results, different compute profiles.

### Staff-Level (10)

1. **Design item embeddings for a 10M-item recommender from user behavior. Word2Vec/item2vec or a learned two-tower model?** item2vec (skip-gram on sessions) is a strong, cheap, fast-to-ship baseline: co-occurrence embeddings + ANN retrieval, no inference network. Two-tower wins when you need cold start (content features in the towers) and richer context. Decision: item2vec for warm behavioral signal at low cost; two-tower when cold start / features / non-linearity justify the complexity. Often *both* — item2vec as a feature/initialization for the towers. Be ready to name session-definition and negative-sampling choices as the real design levers.
2. **Your item2vec recommendations over-recommend popular items. Diagnose and fix.** Popular items appear in many sessions ⟹ dominate co-occurrence and get central, high-norm vectors ⟹ frequent neighbors. Fixes: frequency subsampling (the Word2Vec trick — apply it), popularity-aware negative sampling, normalize embeddings before similarity, post-hoc popularity debiasing, or re-weight the implicit PMI matrix. Tie it to the implicit-MF view: you're reshaping the matrix being factorized. Popularity bias is a co-occurrence artifact, not a bug in the algorithm.
3. **When is a static Word2Vec embedding the *right* choice over a contextual model in 2026?** Latency/cost-critical retrieval where a lookup table beats a forward pass; co-occurrence/behavioral item embeddings (item2vec) where there's no "context" to disambiguate anyway; resource-constrained or on-device; as cheap features/initializations feeding a larger model. The honest line: static embeddings are a *table*; contextual models are a *function* — pick the table when context doesn't matter or you can't afford the function.
4. **Embeddings from two retrains give different nearest neighbors and break a downstream system. Root cause and remediation.** Embedding spaces are defined only up to rotation; independent retrains land in different bases ⟹ raw vectors and neighbors aren't comparable. Remediate: Procrustes-align successive spaces, or warm-start retrains from previous vectors, or version embeddings + downstream together and revalidate. Institutionalize: never assume cross-version embedding comparability — the same coupling/alignment lesson as Ch. 11/16.
5. **Design temporal embeddings to track how item/word meaning shifts over time.** Train per-period embeddings, Procrustes-align consecutive periods to a common space, then measure drift (vector movement) — e.g., a restaurant's "meaning" shifting from "lunch spot" to "delivery" as behavior changes. Alignment is the crux; without it the drift signal is pure rotational noise. A genuinely senior application that shows you understand the rotational ambiguity.
6. **How does negative sampling's false-negative problem affect recsys item2vec, and what do you do?** Sampled "negatives" may be genuinely relevant items (just not in this session) ⟹ the model is told to push apart items that should be close ⟹ degraded geometry, worse for dense catalogs/popular items. Mitigations: popularity-aware sampling, hard-negative *mining* (informative negatives, not random — the SVM/contrastive connection), or in-batch negatives with corrections. Knowing negatives are noisy supervision is the staff insight.
7. **You need embeddings for items that have content (text/images) AND behavioral signal. Architecture?** Fuse: item2vec for behavioral co-occurrence + sentence-transformer/CNN content embeddings, combined by concatenation, weighted average, or as inputs to a two-tower model that learns the combination. Cold-start items lean on content (no behavior yet); warm items blend both. *This is exactly Similar Restaurants — content (sentence-transformer) + behavioral (Word2Vec) — describe it as the deliberate hybrid it was.*
8. **Evaluate an embedding model for your retrieval task — what's wrong with intrinsic analogy benchmarks?** Analogy/word-similarity benchmarks measure generic linguistic structure, not your task; high benchmark scores don't predict retrieval recall@k or recommendation CTR on your distribution. Build a task-specific eval (labeled query-item relevance from logs), measure the downstream metric, test tail items. "The benchmark is not your task" — recurring across PCA, Transformers, and here.
9. **An embedding space is anisotropic and your cosine-similarity retrieval is poor. Walk through remediation.** Diagnose anisotropy (vectors in a narrow cone, inflated baseline cosines); apply mean-centering + remove top principal components ("all-but-the-top"), or whitening, or train with a contrastive objective that encourages uniformity; re-measure recall@k. Knowing that *raw* embeddings often need post-processing before similarity search is a practitioner-grade detail.
10. **Leadership asks why you used "old" Word2Vec instead of LLM embeddings for item similarity. Defend or revise.** Defend on cost/latency/fit: behavioral item similarity has no linguistic "context" for an LLM to exploit — co-occurrence *is* the signal, and item2vec captures it cheaply with zero inference cost; LLM embeddings add cost and may not beat it on *behavioral* similarity (they shine on *content* similarity). The right answer is often the hybrid (content via LLM/sentence-transformer + behavioral via item2vec), chosen by measured recall@k and serving budget — not by recency of the technique. Matching method to signal type is the judgment.

---

## 8. Comparison Section

**Word2Vec vs GloVe (full treatment Ch. 18):** both factorize co-occurrence statistics; Word2Vec (SGNS) does it *implicitly* via local context sampling (shifted PMI), GloVe does it *explicitly* via weighted least squares on global log-co-occurrence counts. Comparable quality; GloVe uses global stats up front, Word2Vec streams local windows. Same family (Levy-Goldberg).

**Word2Vec vs contextual (BERT/sentence-transformer):**

| | Word2Vec | Contextual (BERT/ST) |
|---|---|---|
| Vector per word | One (static) | One per occurrence (contextual) |
| Polysemy | Can't disambiguate ("bank") | Disambiguates by context |
| Inference | None (lookup table) | Forward pass per input |
| Cost | Tiny | Heavy |
| Best for | Cheap embeddings, item2vec, features | Context-dependent meaning, SOTA retrieval |

**Word2Vec vs one-hot/TF-IDF:** dense low-dim with geometric similarity vs sparse high-dim with no similarity structure; Word2Vec captures relationships, TF-IDF captures frequency-weighted presence (still strong for some retrieval — LinearSVC on TF-IDF, Ch. 6).

**Word2Vec vs SVD/LSA (Ch. 11):** both low-rank co-occurrence factorizations; SVD on counts (explicit, closed-form on a complete matrix), Word2Vec on shifted PMI (implicit, SGD). The Levy-Goldberg result makes them siblings — neural embeddings *are* implicit SVD.

**item2vec vs matrix factorization (Ch. 11):** MF factorizes the user-item interaction matrix (user and item factors); item2vec factorizes item-item co-occurrence (item factors only, from session sequences). MF models user preference directly; item2vec models item-item similarity — complementary, often combined.

**Word2Vec negatives vs contrastive/two-tower negatives:** same principle (pull positives, push negatives); modern two-tower uses in-batch + hard negatives — Word2Vec's negative sampling is the ancestor (and links to SVM margins, Ch. 6).

---

## 9. Common Mistakes

**Candidate mistakes:**
- Confusing skip-gram (target→context) with CBOW (context→target) directions.
- Not knowing *why* negative sampling exists (full-softmax cost over vocabulary).
- Forgetting the 3/4-power negative-sampling distribution.
- Calling Word2Vec embeddings "contextual" (they're static — one vector per word).
- Not knowing the implicit-PMI-factorization result that unifies the embedding family.

**Production mistakes:**
- Comparing embeddings across retrains without alignment (rotational ambiguity).
- Ignoring popularity bias in item2vec (a co-occurrence artifact — subsample/debias).
- Using raw anisotropic embeddings for cosine retrieval without post-processing.
- No OOV/cold-start strategy (static embeddings have no vector for unseen items).

**Modeling mistakes:**
- Averaging word vectors as a sentence embedding when a sentence-transformer was the right tool.
- Wrong window size for the desired similarity type (topical vs functional).
- Treating sampled negatives as clean (false negatives degrade geometry).
- Choosing dimension by intrinsic benchmarks instead of downstream metric.

---

## 10. Real Industry Use Cases

- **Google** — Word2Vec's birthplace (Mikolov et al.); embeddings throughout pre-BERT NLP; the conceptual ancestor of all learned-embedding systems.
- **Amazon** — prod2vec (products as words, purchase sessions as sentences) for "customers who bought" style recommendation; query/product embeddings for search.
- **Netflix** — item/title embeddings from co-viewing sessions for similarity and candidate generation.
- **Meta** — fastText (their subword extension of Word2Vec) for classification/embeddings at scale; embedding-based retrieval ancestry.
- **Uber / Airbnb** — Airbnb's listing embeddings (KDD best-paper: skip-gram on click sessions for similar-listing recommendation and search ranking) is the canonical industrial item2vec case — directly analogous to your work.
- **Swiggy/Zomato** — item2vec/restaurant embeddings from user sessions for similarity and candidate generation; query embeddings for search. *Naveen: the behavioral half of Similar Restaurants is item2vec — Word2Vec on user co-occurrence — complementing the sentence-transformer content embeddings (Ch. 16). The Airbnb listing-embeddings paper is the closest published analogue to your project; cite the parallel.*
- **Flipkart** — product embeddings from browse/purchase sessions; query understanding.
- **Games24x7** — game-mode / action-sequence embeddings from player session co-occurrence for personalization and behavioral feature engineering; embeddings as inputs to churn/fraud models.

---

## 11. Coding From Scratch (NumPy only)

Skip-gram with negative sampling — the version interviewers ask for. Reuses the stable sigmoid from Ch. 2; the SGD update *is* the §4.1 math.

```python
import numpy as np

class Word2VecSGNS:
    """Skip-gram with Negative Sampling. Trains two embedding matrices:
       W_in (target vectors) and W_out (context vectors)."""
    def __init__(self, vocab_size, dim=100, n_neg=5, lr=0.025, seed=0):
        rng = np.random.default_rng(seed)
        self.V, self.dim, self.k, self.lr = vocab_size, dim, n_neg, lr
        # Two vectors per word (target role and context role).
        self.W_in  = (rng.random((vocab_size, dim)) - 0.5) / dim   # target embeddings
        self.W_out = np.zeros((vocab_size, dim))                   # context embeddings
        self.rng = rng
        self.neg_table = None

    def build_neg_table(self, word_freqs):
        # Sampling distribution ∝ freq^0.75 (the empirical sweet spot):
        # flattens frequency so common words don't dominate negatives.
        p = np.asarray(word_freqs, float) ** 0.75
        self.neg_probs = p / p.sum()

    @staticmethod
    def _sigmoid(z):
        return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

    def train_pair(self, target, context):
        """One (target, true-context) pair + k sampled negatives.
           This is the §4.1 update: k+1 logistic regressions on dot products."""
        v = self.W_in[target]                          # target vector (dim,)

        # Sample k negatives ∝ freq^0.75 (would exclude the true context in practice)
        negatives = self.rng.choice(self.V, size=self.k, p=self.neg_probs)

        # Labels: 1 for the true context, 0 for each negative.
        samples = np.concatenate(([context], negatives))      # (k+1,)
        labels  = np.concatenate(([1.0], np.zeros(self.k)))    # (k+1,)

        out_vecs = self.W_out[samples]                 # (k+1, dim)
        scores = out_vecs @ v                          # (k+1,) dot products
        preds  = self._sigmoid(scores)                 # σ(score)

        # Gradient factor (σ − label) — identical to logistic regression (Ch. 2).
        g = (preds - labels)                           # (k+1,)

        # Update context vectors: each moves by g·v.
        grad_out = g[:, None] * v[None, :]             # (k+1, dim)
        # Update target vector: sum of g·(its context vector).
        grad_in  = (g[:, None] * out_vecs).sum(axis=0) # (dim,)

        self.W_out[samples] -= self.lr * grad_out      # push true UP, negatives DOWN
        self.W_in[target]   -= self.lr * grad_in       #   (via the sign of g·label)

    def train(self, pairs, epochs=5):
        # pairs: list of (target_idx, context_idx) from sliding windows over corpus.
        for _ in range(epochs):
            self.rng.shuffle(pairs)
            for t, c in pairs:
                self.train_pair(t, c)
        return self

    def most_similar(self, word_idx, top_n=10):
        # Cosine similarity over target embeddings → in production this is an
        # ANN index (Ch. 7/9), not a full matrix multiply.
        v = self.W_in[word_idx]
        norms = np.linalg.norm(self.W_in, axis=1) * np.linalg.norm(v) + 1e-9
        sims = (self.W_in @ v) / norms
        return np.argsort(sims)[::-1][1:top_n+1]       # skip self
```

Narration points that earn senior credit:
- **The freq^0.75 table** — "negatives sampled proportional to frequency to the three-quarters: flattens the distribution so common words don't dominate the negatives. Empirically optimal; it reshapes the implicit PMI matrix being factorized."
- **`g = preds - labels`** — point at it: "this (σ − label) is *exactly* logistic regression's gradient; negative sampling is k+1 logistic regressions on dot products, which is why it's so cheap — O(k), not O(vocabulary)."
- **Two embedding matrices** — "every word has a target vector and a context vector; we keep W_in as the embeddings. Separating roles stabilizes training."
- **Push-up / push-down** — "the true context's vector moves toward the target, negatives move away — over millions of updates, co-occurring words converge."
- **`most_similar` is a full scan here, ANN in production** — connects to Ch. 7/9 and your retrieval stack.
- **Extensions to offer:** subsampling frequent words before generating pairs, CBOW (average context → predict target), the item2vec reframe ("sessions are sentences, items are words — same code, recommendation embeddings"), and "the Levy-Goldberg result says this is implicitly factorizing shifted PMI — same family as SVD."

---

## 12. ML System Design Perspective

**Choose Word2Vec / item2vec when:** you need cheap static embeddings as a lookup table (zero inference cost); co-occurrence/behavioral item embeddings from sessions or baskets (item2vec); features or initializations for a larger model; resource/latency-constrained settings; a strong, fast-to-ship recommendation baseline.

**Avoid when:** context-dependent meaning matters (polysemy → contextual models); a pretrained contextual encoder is available and affordable and quality is paramount; cold start dominates without content features (static embeddings have no OOV vector — use fastText/content/two-tower).

**Data requirements:** large co-occurrence corpus (text, or sessions/baskets for item2vec); enough occurrences per word/item for stable vectors (min-count floor); session/window definition is a *design decision* that determines what similarity you learn.

**Latency:** inference-free — embedding lookup is a hash/array access (microseconds); similarity via ANN over the embedding matrix. The cheapest possible embedding-serving path, which is its enduring appeal.

**Scale limits:** trains linearly on billions of tokens (parallel async SGD); the binding constraints are co-occurrence drift (retrain cadence), embedding-space alignment across retrains, popularity bias, and cold start — not compute.

---

## 13. Resume Discussion Angle

**Similar Restaurants (this chapter + Ch. 16 are the two halves):** the precise, credible framing — "content similarity from sentence-transformer embeddings (Ch. 16) *plus* behavioral similarity from Word2Vec/item2vec on user co-occurrence sessions, fused for the final representation, served via ANN." Expect deep probes: how you defined a "session" (the window-size decision), how you handled popularity bias (subsampling — the Word2Vec trick), cold start (content side covers new restaurants the behavioral side can't), and why you fused two signals rather than one. The 9% CTR uplift becomes defensible when you can decompose *which* signal drove it. Cite the **Airbnb listing-embeddings paper** as the published analogue — it signals you know the lineage.

**The implicit-MF unification (a senior flourish):** when embeddings come up, drop the Levy-Goldberg result — "Word2Vec, GloVe, and SVD/LSA are all factorizing a co-occurrence statistic; item2vec and matrix factorization (Ch. 11) are the recsys versions." This shows you see the *unified structure* under the buzzwords, which is exactly what Applied Scientist loops probe. It also lets you answer "why item2vec not MF?" precisely: item-item co-occurrence vs user-item interaction — complementary factorizations.

**The contrastive-learning bridge:** "negative sampling is early contrastive learning — pull positives together, push negatives apart — which is the same principle as in-batch negatives and hard-negative mining in two-tower retrieval (and SVM margins, Ch. 6)." Connecting Word2Vec → modern two-tower training through the negative-sampling lens demonstrates you understand the *continuity* of the field, not isolated techniques.

**The universal trap:** "Why not just use LLM embeddings for everything?" Don't concede reflexively. The strong answer: behavioral item similarity has no linguistic context for an LLM to exploit — co-occurrence *is* the signal, captured cheaply by item2vec with zero inference cost; LLM/contextual embeddings win on *content* similarity. The right architecture is usually the hybrid (content + behavioral), chosen by measured recall@k and serving budget. Matching the embedding method to the *type of signal* — not the recency of the technique — is the judgment that reads as senior, and it's exactly the decision you made on Similar Restaurants.

---
*Previous: Transformer ← | Next: GloVe →*

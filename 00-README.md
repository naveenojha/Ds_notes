# ML Interview Handbook — Complete
### Senior Data Scientist / Staff MLE / Applied Scientist

**For:** loops at Google, Amazon, Uber, Airbnb, Netflix, Swiggy, Zomato, Walmart, Agoda, McKinsey QuantumBlack.
**Author's intent:** articulation-quality interview prep written as a Lead DS would — not textbook recall. Every chapter follows the same 13 sections.

## The 13 sections in every chapter
1. Executive Summary (30s) · 2. Interview Articulation (3–4 min spoken answer) · 3. Mathematical Foundation · 4. Step-by-Step Numerical Example · 5. Hyperparameters · 6. Production Perspective · 7. Interview Follow-ups (15 medium / 15 advanced / 10 staff, all answered) · 8. Comparison Section · 9. Common Mistakes · 10. Real Industry Use Cases · 11. Coding From Scratch (NumPy) · 12. ML System Design Perspective · 13. Resume Discussion Angle

## Status — ALL 21 COMPLETE ✅

| # | Chapter | File |
|---|---|---|
| 1 | Linear Regression | 01-linear-regression.md |
| 2 | Logistic Regression | 02-logistic-regression.md |
| 3 | Decision Trees | 03-decision-trees.md |
| 4 | Random Forest | 04-random-forest.md |
| 5 | Gradient Boosting (XGBoost/LightGBM/CatBoost) | 05-gradient-boosting.md |
| 6 | SVM | 06-svm.md |
| 7 | KNN | 07-knn.md |
| 8 | Naive Bayes | 08-naive-bayes.md |
| 9 | K-Means | 09-kmeans.md |
| 10 | PCA | 10-pca.md |
| 11 | SVD | 11-svd.md |
| 12 | Neural Networks (MLP) | 12-mlp.md |
| 13 | CNN | 13-cnn.md |
| 14 | RNN | 14-rnn.md |
| 15 | LSTM | 15-lstm.md |
| 16 | Transformer | 16-transformer.md |
| 17 | Word2Vec | 17-word2vec.md |
| 18 | GloVe | 18-glove.md |
| 19 | Learning to Rank | 19-learning-to-rank.md |
| 20 | Association Rule Mining | 20-association-rules.md |
| 21 | Statistics & Probability | 21-statistics-probability.md |

## How the handbook is wired together (the cross-chapter threads)

The chapters are not independent — interviewers chain them, and the handbook is built so you can narrate the connections:

- **The boosting → ranking spine:** Ch. 5 (boosting = gradient descent in function space) → Ch. 19 (LambdaMART = boosting with ranking-aware gradients). "Explain one boosting iteration" → "now what changes for ranking?" is a single chained question.
- **The embedding → retrieval → RAG arc:** Ch. 11 (SVD/MF) → Ch. 17/18 (Word2Vec/GloVe, "neural embeddings are implicit MF") → Ch. 16 (Transformer encoders) → Ch. 7/9 (KNN/K-Means as the ANN index). Your Similar Restaurants and HelpBot systems live across these.
- **MF → two-tower:** Ch. 11's "MF is a two-tower model with ID-lookup towers and a dot-product scorer" connects classical CF to your modern retrieval work.
- **The vanishing-gradient lineage:** Ch. 12 (backprop) → Ch. 14 (RNN, the problem in its sharpest form) → Ch. 15 (LSTM gradient highway) → Ch. 16 (attention removes recurrence entirely). Each architecture is "an MLP plus the right structural constraint."
- **Regularization = Bayesian prior:** Ch. 1/2 (L1/L2) ↔ Ch. 21 (MAP estimation, Gaussian/Laplace priors).
- **The prediction-vs-causation thread (the deepest staff/AS signal):** Ch. 2/5 (propensity vs uplift), Ch. 5 (feedback loops), Ch. 19 (position bias), Ch. 20 (correlation in baskets), Ch. 21 (A/B testing, observational causal inference). The recurring "offline metric ≠ online truth; the experiment is the truth."

## Highest-leverage chapters for this candidate (by pipeline yield)
1. **Ch. 5 Gradient Boosting** + **Ch. 19 Learning to Rank** — the Zomato/Flipkart ranking spine (LambdaMART).
2. **Ch. 16 Transformer** — Similar Restaurants embeddings + HelpBot RAG.
3. **Ch. 21 Statistics** — Amazon Breadth & Depth + McKinsey QuantumBlack case rounds.
4. **Ch. 11 SVD** + **Ch. 17 Word2Vec** — the two halves of Similar Restaurants (collaborative + behavioral).

## Notes
- All "from scratch" code is NumPy-only (no sklearn), with line-by-line narration points calibrated to what interviewers grade.
- ChatGPT reference links in the original spec were inaccessible; everything was built from first principles.
- Resume Discussion Angles are wired to the candidate's actual systems (Zomato Search V0–V3, Similar Restaurants, Games24x7 HelpBot/fraud) — first-person rehearsal material, not generic advice.

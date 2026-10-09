# PaperMatcher-AI: Hybrid Lexical-Semantic Retrieval Upgrade

**Technical Architecture & Engineering Report**  
*Document Version:* 2.0.0  
*Component:* Dense Semantic Retrieval & Hybrid Fusion Engine  

---

## 1. Executive Summary & Motivation

The indexed corpus contains 287,383 documents from 287,421 raw dataset records; the project reports removing 38 duplicate normalized title-summary records. The lexical/statistical retrieval engine uses sublinear TF-IDF with unigram and bigram n-gram features. Exact-term matching is a design capability, while queries using synonymous or paraphrased language may exhibit vocabulary mismatch.

This upgrade introduces dense semantic representations using **Sentence-BERT (`all-MiniLM-L6-v2`)** to complement the statistical baseline, creating a **hybrid lexical-semantic retrieval system**. The design combines lexical and semantic similarity signals; the available relevance evaluation does not establish improved relevance or general superiority over either individual mode.

```
                                      User Query
                                          │
                    ┌─────────────────────┴─────────────────────┐
                    ▼                                           ▼
          Lexical Retrieval                           Dense Semantic Retrieval
       (Sublinear TF-IDF 50k)                       (Sentence-BERT MiniLM-L6)
                    │                                           │
                    ▼                                           ▼
         Sparse Dot Product                          Dense Unit Dot Product
           (Cosine Metric)                             (Cosine Metric)
                    │                                           │
                    └─────────────────────┬─────────────────────┘
                                          ▼
                               Distribution Normalization
                                   (Min-Max Scaling)
                                          │
                                          ▼
                                    Hybrid Ranking
                     S_hybrid = 0.40 * S_tfidf + 0.60 * S_sbert
                                          │
                                          ▼
                            Pre-Ranking Mask Filtering
                           (Category Code & Epoch Year)
                                          │
                                          ▼
                                Linear-Time Top-K
                                (np.argpartition)
                                          │
                                          ▼
                           Ranked Research Recommendations
```

---

## 2. Model Selection: Sentence-BERT (`all-MiniLM-L6-v2`)

### 2.1 Model Specification
- **Base Architecture:** MiniLM-L6-H384-uncased (6 transformer layers, 384 hidden dimensions, 12 attention heads).
- **Pretraining & Fine-tuning:** 1B+ sentence pairs across 30+ diverse datasets, fine-tuned specifically for sentence cosine similarity via contrastive loss.
- **Inference Footprint:** ~85 MB parameter file; ~22.7M parameters.
- **Pooling Mechanism:** Mean pooling over contextual token representations with attention mask padding exclusion:
  $$\mathbf{u} = \frac{\sum_{i=1}^L m_i \cdot \mathbf{h}_i}{\sum_{i=1}^L m_i}$$
- **Normalization:** Euclidean unit $L_2$-normalization ($\|\mathbf{u}\|_2 = 1$). Under unit normalization, cosine similarity between query $\mathbf{q}$ and document $\mathbf{d}$ simplifies to an efficient inner product:
  $$\cos(\theta) = \mathbf{q} \cdot \mathbf{d}$$

### 2.2 Rationale & Hardware Feasibility
1. **CPU Execution:** End-to-end latency measurements for the three retrieval modes are reported in Section 7. The saved benchmark does not provide a separate query-encoding latency breakdown.
2. **Dense Memory Footprint:** The benchmark artifact records 420.97 MB for SBERT memory. Its indexed corpus contains 287,383 documents; embeddings are loaded using memory mapping (`mmap_mode="r"`).
3. **Cross-Platform Portability:** Operates via standard PyTorch `sentence-transformers` and Microsoft ONNX Runtime (`onnxruntime`), guaranteeing continuous availability even on platforms enforcing strict App Control or unsigned DLL execution constraints.

---

## 3. Embedding Index Generation & Persistence

### 3.1 Text Representation
To ensure exact semantic parity with the lexical baseline, the document text input is constructed identically to the TF-IDF pipeline:
$$\text{text}_i = \text{clean}(\text{title}_i) \mathbin{\Vert} \text{" "} \mathbin{\Vert} \text{clean}(\text{summary}_i)$$

### 3.2 Indexing Pipeline (`app.services.build_semantic_index`)
1. **Metadata Synchronization:** Loads `models/paper_metadata.joblib` (287,383 indexed records). The ordering of generated embeddings maps 1-to-1 to the list indices of `paper_metadata.joblib`.
2. **Chunked Checkpointing:** Encodes corpus in checkpointed blocks (e.g., 5,000 documents per batch) persisted to `models/sbert_chunks/chunk_{start}_{end}.npy`. If process execution is interrupted, existing chunks are detected and skipped on resume.
3. **Post-Build Validation:**
   - Evaluates $\text{count}(\text{embeddings}) == \text{len}(\text{paper\_metadata})$: throws `ValueError` if counts diverge.
   - Evaluates embedding dimension $\dim == 384$.
   - Confirms unit norm $\|\mathbf{e}_i\|_2 \approx 1.0$ across random subsamples.
4. **Artifact Consolidation:** Saves single contiguous matrix to `models/sbert_embeddings.npy` along with metadata manifest `models/sbert_meta.json`.

---

## 4. Score Normalization & Hybrid Fusion

### 4.1 Score Distribution Divergence
Raw cosine similarity scores from lexical and dense retrieval inhabit distinctly different statistical distributions:
- **TF-IDF Scores:** Non-negative, sparse, with candidate scores typically clustered in $[0.02, 0.45]$.
- **SBERT Cosine Scores:** Continuous unit inner products spanning $[-1.0, 1.0]$, with candidate matches typically clustered in $[0.20, 0.85]$.

Direct unnormalized linear combination would cause SBERT to dominate ranking simply due to scale bias, degrading lexical keyword precision.

### 4.2 Normalization Methodology
PaperMatcher-AI implements configurable candidate distribution normalization prior to weighted combination:

1. **Min-Max Scaling (`SCORE_NORMALIZATION=minmax` — Default):**
   $$S_{\text{norm}}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S) + \epsilon}$$
   Scales both candidate score distributions into $[0.0, 1.0]$ relative to the candidate pool for the given query, ensuring the assigned weights reflect genuine proportional influence.
2. **Cosine Clipping (`SCORE_NORMALIZATION=clip`):**
   $$S_{\text{norm}}(d) = \max(0.0, \min(1.0, S(d)))$$
   Preserves absolute cosine orientations while guarding against negative angular drift.
3. **Identity (`SCORE_NORMALIZATION=none`):** Raw score pass-through for experimental comparison.

### 4.3 Weighted Combination
With normalized scores, the final hybrid relevance score is computed as:
$$S_{\text{hybrid}}(q, d) = w_{\text{tfidf}} \cdot S_{\text{norm}}^{\text{tfidf}}(q, d) + w_{\text{sbert}} \cdot S_{\text{norm}}^{\text{sbert}}(q, d)$$

Configured defaults:
$$\text{TFIDF\_WEIGHT} = 0.40, \quad \text{SEMANTIC\_WEIGHT} = 0.60$$

---

## 5. Candidate Filtering & Linear-Time Selection

### 5.1 Pipeline Execution Order
To guarantee correctness and efficiency:
1. Validate query string (minimum length 3 non-whitespace characters).
2. Compute TF-IDF sparse query vector $\mathbf{v}_q \in \mathbb{R}^{50,000}$ and SBERT dense query vector $\mathbf{u}_q \in \mathbb{R}^{384}$.
3. Calculate raw similarities across candidates:
   - Lexical: $\mathbf{S}_{\text{tfidf}} = \mathbf{A}_{\text{csr}} \mathbf{v}_q^T$
   - Dense: $\mathbf{S}_{\text{sbert}} = \mathbf{M}_{\text{dense}} \mathbf{u}_q$
4. Apply pre-ranking bitmask for arXiv subject category (`category_code`) and publication epoch (`published_year`):
   $$\text{candidates} = \{i \mid \text{category}(i) = c \land \text{year}(i) = y\}$$
5. Normalize candidate score distributions and compute $S_{\text{hybrid}}$.
6. Perform linear-time Top-$K$ candidate selection using `np.argpartition` ($O(N + K \log K)$ runtime) rather than full corpus sorting ($O(N \log N)$).
7. Hydrate recommendations into verified metadata models.

---

## 6. Information Retrieval Evaluation Framework

### 6.1 Metrics Implemented (`evaluation/evaluate_retrieval.py`)
- **Precision@K:** Fraction of retrieved items in Top-$K$ that are relevant.
  $$\text{Precision@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{K}$$
- **Recall@K:** Fraction of total ground-truth relevant items successfully retrieved in Top-$K$.
  $$\text{Recall@}K = \frac{|\text{Retrieved}_K \cap \text{Relevant}|}{|\text{Relevant}|}$$
- **Mean Reciprocal Rank (MRR):** Multiplicative inverse of the rank of the first relevant preprint.
  $$\text{MRR} = \frac{1}{\text{rank}_1}$$
- **nDCG@K:** Normalized Discounted Cumulative Gain penalizing relevant papers appearing lower in rank.
  $$\text{DCG@}K = \sum_{i=1}^K \frac{\text{rel}_i}{\log_2(i + 1)}, \quad \text{nDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

### 6.2 Relevance-Label Coverage
The evaluation dataset `evaluation/queries.json` specifies queries alongside `relevant_paper_ids`. The evaluator scores only queries with nonempty relevance lists. Currently, **1 of 7 queries** has a nonempty list. The data does not establish that these judgments are complete or human-verified, so the saved metrics are preliminary and apply only to that single query.

### 6.3 Saved Preliminary Results
The saved results in `evaluation/evaluation_results.json` use `top_k = 5`:

| Mode | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.2 | 1.0 | 1.0 | 1.0 |
| Semantic | 0.0 | 0.0 | 0.0 | 0.0 |
| Hybrid | 0.2 | 1.0 | 1.0 | 1.0 |

These are single-query preliminary results, not overall accuracy. They do not establish general retrieval quality or prove that one mode is superior.

---

## 7. Recorded Retrieval Latency

The saved `scratch/benchmark_results.json` records total query latency across seven benchmark queries. The snapshot timestamp is `2026-10-09T03:25:46Z`; the recorded model is `all-MiniLM-L6-v2`.

| Mode | Average total latency | Range |
|---|---:|---:|
| TF-IDF | 42.95 ms | 38.51–50.08 ms |
| Semantic | 84.54 ms | 17.32–336.18 ms |
| Hybrid | 56.99 ms | 53.78–59.56 ms |

The benchmark artifact separately records TF-IDF memory of 259.06 MB and SBERT memory of 420.97 MB; it does not report a Hybrid memory measurement. Latency and memory measurements describe runtime costs, not relevance quality.

---

## 8. Limitations & Assumptions

1. **Academic Distinctions:** Sentence-BERT provides *dense semantic representation*, which captures general sentence-level topical proximity; it does not constitute full contextual reasoning or factual verification.
2. **Fixed Vocabulary & Sequence Truncation:** Token sequence truncation at 128 tokens captures the title and early sentences of the abstract; deeply buried methodology details in late paragraphs are not encoded in the dense vector.
3. **No Automatic Ground-Truth Guarantee:** Hybrid ranking is intended to combine lexical and semantic signals. The current one-query evaluation does not establish qualitatively richer recommendations or superiority; broader claims require a sufficiently sized, task-specific relevance-annotated benchmark.

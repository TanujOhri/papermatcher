# PaperMatcher-AI: Intelligent Scientific Paper Recommendation System

`updatedpapermatcher`

> 
> **Intelligent Research Paper Recommendation Using Hybrid NLP**<br>
> *Hybrid lexical-semantic retrieval combining TF-IDF and Sentence-BERT (`all-MiniLM-L6-v2`), intended to use both exact-term and semantic-similarity signals. The index contains 287,383 documents from 287,421 raw records after 38 duplicate normalized title-summary records were removed.*

---

## 1. Project Overview

With hundreds of thousands of scientific preprints submitted annually to digital repositories like arXiv, researchers face severe information overload. Conventional search engines rely exclusively on keyword matches or boolean queries, which suffer from:
1. **Vocabulary Mismatch & Synonymy**: Different researchers describe equivalent concepts using distinct vocabularies (e.g., *“sublinear vectorization”* vs. *“term frequency scaling”*).
2. **Polysemy**: Homonymous terms yield irrelevant search noise across scientific disciplines.
3. **Generative Hallucination**: Generative AI search assistants frequently hallucinate non-existent paper titles, author attributions, and citations.

**PaperMatcher-AI** is a deterministic, high-throughput research paper recommendation system engineered to solve these challenges using a **hybrid lexical-semantic retrieval architecture**:
- **Lexical/Statistical Baseline (TF-IDF)**: Extracts 50,000 unigram and bigram features using sublinear term frequency scaling and smooth inverse document frequency, designed to emphasize exact-term signals such as specialized acronyms, equations, and proper nouns.
- **Dense Semantic Representation (Sentence-BERT)**: Projects texts into a continuous 384-dimensional unit hypersphere using `sentence-transformers/all-MiniLM-L6-v2` to represent semantic similarity between queries and papers.
- **Hybrid Fusion Engine**: Standardizes score distributions via min-max normalization and computes a configurable weighted linear combination ($0.40 \times \text{TF-IDF} + 0.60 \times \text{SBERT}$), filtering candidates by arXiv subject category and publication epoch. This design is intended to combine lexical and semantic signals; current relevance evaluation does not establish an improvement over either individual mode.

---

## 2. System Architecture

PaperMatcher-AI is designed with a decoupled, high-performance client-server architecture:

```
                                  [ User Interface ]
                       React 18 + Vite + Tailwind CSS (Port 5173)
                   (Editorial Research Journal & Stationery Folio UI)
                                         │
                                         ▼ HTTP REST (JSON)
                            [ FastAPI Backend Service ]
                            Uvicorn ASGI (Port 8002)
                                         │
                ┌────────────────────────┴────────────────────────┐
                ▼                                                 ▼
     [ GET /api/v1/health ]                       [ POST /api/v1/recommendations ]
     [ GET /api/v1/recommendations/filters ]      (query, top_k, category, year, mode)
                                                                  │
                                                                  ▼
                                                      [ Query Normalization ]
                                                      Clean & Preprocess Text
                                                                  │
                                      ┌───────────────────────────┴───────────────────────────┐
                                      ▼                                                       ▼
                          [ Lexical TF-IDF Vector ]                               [ SBERT Dense Embedding ]
                          Pre-fitted 50k TfidfVectorizer                          all-MiniLM-L6-v2 (384-d)
                                      │                                                       │
                                      ▼                                                       ▼
                          [ Sparse Matrix Dot Product ]                           [ Dense Matrix Dot Product ]
                          Cosine Score S_tfidf = A * v_q                          Cosine Score S_sbert = M * u_q
                                      │                                                       │
                                      └───────────────────────────┬───────────────────────────┘
                                                                  ▼
                                                      [ Candidate Score Normalization ]
                                                      Min-Max Scaling to [0.0, 1.0]
                                                                  │
                                                                  ▼
                                                      [ Hybrid Score Combination ]
                                                      S = 0.40 * S_tfidf + 0.60 * S_sbert
                                                                  │
                                                                  ▼
                                                      [ Pre-Ranking Filters ]
                                                      Boolean Mask: category_code & year
                                                                  │
                                                                  ▼
                                                      [ Linear-Time Top-K Selection ]
                                                      np.argpartition (O(N + K log K))
                                                                  │
                                                                  ▼
                                                      [ Metadata Hydration ]
                                                      Ground-Truth arXiv Records
                                                                  │
                                                                  ▼
                                                      [ 200 OK JSON Response ]
```

---

## 3. Dataset Pipeline & Corpus Characteristics

### 3.1 Corpus Specifications
The underlying corpus is derived from a real-world arXiv scientific preprint dataset containing comprehensive multi-disciplinary computer science, statistics, mathematics, and electrical engineering records:
- **Raw Dataset Records**: 287,421 documents
- **Indexed Documents**: 287,383 documents
- **Deduplication**: 38 duplicate normalized title-summary records were removed. The loader also filters duplicate IDs and records with missing or empty titles or summaries.
- **Dataset Size**: ~410 MB raw CSV
- **Subject Categories**: 147 distinct arXiv classifications (e.g., `cs.LG`, `cs.CV`, `cs.CL`, `cs.AI`, `cs.CR`, `stat.ML`)
- **Publication Epoch**: 1993 through 2026

### 3.2 Dataset Schema
| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | `str` | Unique arXiv preprint identifier (e.g. `abs-1910.03771v5`, `2406.12221`) |
| `title` | `str` | Complete title of the research publication |
| `summary` | `str` | Complete author-provided research summary (*Note: strictly "summary", never "abstract"*) |
| `category` | `str` | Full taxonomy classification name (e.g. `Computation and Language`) |
| `category_code` | `str` | Primary arXiv subject code (e.g. `cs.CL`, `cs.LG`, `cs.AI`) |
| `published_date` | `str` | Original submission timestamp (formatted as `M/D/YY`) |
| `updated_date` | `str` | Date of the latest revision |
| `authors` | `str` | Complete co-author listing |
| `first_author` | `str` | Primary investigator / lead author |
| `summary_word_count`| `int` | Length of summary in words |

---

## 4. NLP Methodology & Retrieval Formulations

### 4.1 Document Representation
Each paper document $d$ is constructed by concatenating its title and summary:
$$\text{text}(d) = \text{title}(d) \mathbin{\Vert} \text{" "} \mathbin{\Vert} \text{summary}(d)$$
This provides strong thematic context from the title amplified by the dense scientific specifics in the summary.

### 4.2 Sublinear TF-IDF (Lexical / Statistical Baseline)
Term Frequency–Inverse Document Frequency converts textual documents into numerical vectors. To prevent long research summaries from disproportionately penalizing concise papers, **Sublinear Term Frequency** scaling is applied:

$$\text{tf}(t, d) = \begin{cases} 1 + \log(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

$$\text{idf}(t) = \log\left(\frac{1 + N}{1 + \text{df}(t)}\right) + 1$$

$$\mathbf{v}_d = \frac{\mathbf{w}_d}{\|\mathbf{w}_d\|_2} = \frac{\mathbf{w}_d}{\sqrt{\sum_{t} \mathbf{w}_{d, t}^2}}$$

- **Vocabulary**: 50,000 unigram and bigram features (`ngram_range=(1, 2)`).
- **Storage**: `models/tfidf_matrix.npz` stored as Compressed Sparse Row (CSR) matrix occupying **181.3 MB** in memory.

### 4.3 Sentence-BERT (Dense Semantic Representation)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (6 transformer layers, 384 hidden dimensions).
- **Mean Pooling**: Pools contextual token representations over unpadded sequence tokens:
  $$\mathbf{u} = \frac{\sum_{i=1}^L m_i \cdot \mathbf{h}_i}{\sum_{i=1}^L m_i}$$
- **Unit Normalization**: $L_2$-normalized such that $\|\mathbf{u}\|_2 = 1.0$.
- **Storage**: Persisted to `models/sbert_embeddings.npy` with metadata manifest `models/sbert_meta.json`.
- **Runtime Dual Engine**: Employs PyTorch `sentence-transformers` with automatic zero-downtime fallback to high-performance Microsoft ONNX Runtime (`onnxruntime`) on systems enforcing strict platform code-integrity policies.

### 4.4 Score Distribution Normalization & Hybrid Fusion
Because raw lexical scores (sparse dot products in $[0.0, 0.45]$) and dense semantic cosine scores (continuous dot products in $[-1.0, 1.0]$) have divergent distributions, candidate scores are standardized before combination:

1. **Min-Max Normalization (`SCORE_NORMALIZATION=minmax` — Default)**:
   $$S_{\text{norm}}(d) = \frac{S(d) - \min(S)}{\max(S) - \min(S) + \epsilon}$$
2. **Hybrid Scoring**:
   $$S_{\text{hybrid}}(q, d) = w_{\text{tfidf}} \cdot S_{\text{norm}}^{\text{tfidf}}(q, d) + w_{\text{sbert}} \cdot S_{\text{norm}}^{\text{sbert}}(q, d)$$
   where $w_{\text{tfidf}} = 0.40$ and $w_{\text{sbert}} = 0.60$ by default.

---

## 5. Supported Retrieval Modes

PaperMatcher-AI provides seamless configuration to switch retrieval modes internally:

| Mode | Identifier | Description | Use Case |
|---|---|---|---|
| **Hybrid** *(Default)* | `RETRIEVAL_MODE=hybrid` | Weighted fusion ($0.40 \times \text{TF-IDF} + 0.60 \times \text{SBERT}$) | Intended to combine exact-term and semantic-similarity signals; relevance advantage is not established |
| **Lexical TF-IDF** | `RETRIEVAL_MODE=tfidf` | Pure sublinear TF-IDF + cosine dot product | Fast baseline, exact acronym/formula lookup, zero transformer dependency |
| **Dense Semantic** | `RETRIEVAL_MODE=semantic` | Pure Sentence-BERT dense vector dot product | Conceptual discovery, synonym/paraphrase matching without keyword constraints |

Retrieval mode can be set globally in `.env` or dynamically overridden per request via the `retrieval_mode` field in `POST /api/v1/recommendations`.

---

## 6. Installation & Execution Guide

### 6.1 Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.14
- **Node.js**: Version 18.0+ and npm 9.0+
- **Memory**: Minimum 4 GB RAM (8 GB recommended)

### 6.2 Backend Setup

1. **Navigate to the backend folder**:
   ```bash
   cd backend
   ```

2. **Create and activate a virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     python -m venv venv
     .\venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS**:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

5. **Generate / Rebuild Model Artifacts**:
   - **Compile Lexical TF-IDF Index**:
     ```bash
     python -m app.services.build_index
     ```
   - **Compile Dense Sentence-BERT Semantic Index**:
     ```bash
     python -m app.services.build_semantic_index
     ```
     *Supports `--batch-size 256`, `--chunk-size 5000`, `--backend onnx`, and `--max-docs N` for rapid checkpointing and resuming.*

6. **Launch the FastAPI Server**:
   ```bash
   uvicorn app.main:app --host 127.0.0.1 --port 8002
   ```
   Interactive Swagger API Documentation: `http://127.0.0.1:8002/docs`

---

### 6.3 Frontend Setup

1. **Navigate to `frontend/`**:
   ```bash
   cd frontend
   ```

2. **Install Node dependencies**:
   ```bash
   npm install
   ```

3. **Start the Vite development server**:
   ```bash
   npm run dev -- --host 127.0.0.1 --port 5173
   ```
   Open your browser at `http://127.0.0.1:5173/`.

---

## 7. Information Retrieval Evaluation & Benchmarking

### 7.1 Retrieval-Quality Evaluation
The evaluator reports Precision@K, Recall@K, MRR, and nDCG@K for queries with nonempty `relevant_paper_ids` in `evaluation/queries.json`. Only **1 of 7 queries** currently has a nonempty relevance list, so the saved results below are preliminary single-query results, not general estimates of retrieval quality. The files do not establish that the relevance judgments are complete or human-verified.

```bash
python evaluation/evaluate_retrieval.py
```

| Mode | Precision@5 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| TF-IDF | 0.2 | 1.0 | 1.0 | 1.0 |
| Semantic | 0.0 | 0.0 | 0.0 | 0.0 |
| Hybrid | 0.2 | 1.0 | 1.0 | 1.0 |

These values are not overall accuracy. They do not establish general retrieval quality or prove that one retrieval mode is superior.

### 7.2 Recorded Retrieval Latency
The saved benchmark records total query latency across seven benchmark queries. It was timestamped `2026-10-09T03:25:46Z` and records model `all-MiniLM-L6-v2` (`scratch/benchmark_results.json`).

| Mode | Average total latency | Range |
|---|---:|---:|
| TF-IDF | 42.95 ms | 38.51–50.08 ms |
| Semantic | 84.54 ms | 17.32–336.18 ms |
| Hybrid | 56.99 ms | 53.78–59.56 ms |

These measurements describe latency only and are not evidence of retrieval relevance or quality.

```bash
python scratch/benchmark_retrieval.py
```

Outputs machine-readable performance metrics to `scratch/benchmark_results.json`.

### 7.3 Automated Test Suite
Run the comprehensive automated test suite (covering TF-IDF baseline, SBERT encoding, hybrid ranking, filtering, and API contracts):

```bash
pytest tests/test_hybrid_retrieval.py -v
```

---

## 8. Technical Upgrades Document

Developed as a Final Year B.Tech Computer Science & Engineering Capstone Project.  
Dataset provided courtesy of the arXiv open-access preprint repository.

For detailed academic discussion on transformer integration, score distribution analysis, and computational design, see [`docs/SEMANTIC_UPGRADE.md`](docs/SEMANTIC_UPGRADE.md).

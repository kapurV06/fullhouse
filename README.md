# FULLHOUSE
### Rare Disease Differential Diagnosis System

A clinical diagnostic engine that combines phenotype ontology embeddings, unsupervised clustering, gradient-boosted ranking, and an LLM agentic layer modeled after the House MD diagnostic team. Enter HPO-encoded patient symptoms, get a ranked differential diagnosis with explainability and character-driven clinical reasoning.

---

## Demo

<img width="1917" height="905" alt="image" src="https://github.com/user-attachments/assets/7654de49-8b68-4375-a56f-2c1bfaedd692" />


---

## Numbers

- 11,572 rare diseases across OMIM, Orphanet, and DECIPHER
- 11,557 HPO phenotype terms as feature dimensions
- 863 HDBSCAN clusters discovered from UMAP-reduced phenotype space
- 99.82% matrix sparsity — TF-IDF weighted before clustering
- 315,033 training pairs with hard negative mining for XGBoost
- 4 LLM personas (House, Foreman, Chase, Cameron) via Groq Llama 3.3 70B

---

## Pipeline

```
phenotype.hpoa + hp.obo + Orphanet HOOM
        ↓
HPO term vectorization (11,557-dim disease profiles)
        ↓
TF-IDF specificity weighting (rare symptoms weighted higher)
        ↓
UMAP dimensionality reduction (11,557 → 15 dims for clustering, 3 dims for visualization)
        ↓
HDBSCAN clustering (min_cluster_size=4, cosine metric, 863 clusters)
        ↓
Candidate narrowing by symptom overlap
        ↓
XGBoost ranking (binary:logistic, hard negative mining, AUC 0.9999)
        ↓
SHAP explainability (symptom-level frequency attribution)
        ↓
Groq LLM agentic layer (House/Foreman/Chase/Cameron personas)
        ↓
Ranked differential with clinical reasoning
```

---

## Stack

**Data:** Human Phenotype Ontology (HPO), Orphanet HOOM v2.5, phenotype.hpoa (285,598 disease-phenotype annotations)

**ML:** UMAP, HDBSCAN, XGBoost, SHAP

**Backend:** FastAPI, Python 3.13

**LLM:** Groq Llama 3.3 70B (4 clinical personas)

**Frontend:** React, Three.js (3D cluster globe), IBM Plex Mono, Permanent Marker

---

## Key Design Decisions

**Why TF-IDF weighting before clustering:** Generic symptoms like "seizure" appear in 2,487 diseases and dominate cosine distance calculations if left unweighted. IDF down-weights these in favor of diagnostically specific symptoms like "ragged-red muscle fibers" — encoding Occam's razor directly into the feature space.

**Why HDBSCAN over K-Means:** Rare disease clusters are non-uniform in size and density. HDBSCAN finds clusters of arbitrary shape and marks genuinely unique diseases as noise rather than force-fitting them into incorrect clusters.

**Why candidate narrowing before XGBoost:** Ranking all 11,572 diseases globally collapses to near-zero scores for partial symptom queries. Filtering to diseases sharing at least one symptom with the query first reduces the candidate pool to a clinically relevant subset where XGBoost can meaningfully discriminate.

**Why the LLM layer exists:** The ranking model surfaces candidates by learned symptom overlap patterns. The LLM layer provides the clinical reasoning chain — why one diagnosis fits better than another, what symptoms are most telling, and where the differential is ambiguous. It also catches cases where the top-ranked candidate is a near-miss, which the team personas explicitly flag.

---

## Local Setup

```bash
# Backend
cd backend
pip install -r requirements.txt
set GROQ_API_KEY=your_key_here
uvicorn scripts.backend:app --reload

# Frontend
cd frontend
npm install
npm start
```

API runs on `http://localhost:8000`, frontend on `http://localhost:3000`.

---

## API

```
POST /diagnose
Body: { "hpo_ids": ["HP:0001250", "HP:0003128", "HP:0002401"], "top_n": 10 }

GET /hpo/search?q=seizure&limit=8

GET /disease/{disease_id}
```

---

## Data Sources

- Human Phenotype Ontology — hp.obo, phenotype.hpoa — hpo.jax.org
- Orphanet HOOM v2.5 — orphadata.com — CC BY 4.0
- OMIM, DECIPHER annotations via phenotype.hpoa

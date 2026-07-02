"""
FULLHOUSE - FastAPI Backend
Run: uvicorn scripts.backend:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import numpy as np
import pickle
import xgboost as xgb
from scipy import sparse
from groq import Groq
import os
import random

app = FastAPI(title="FULLHOUSE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUT_DIR = "processed"

# ============================================================
# STARTUP: load all artifacts once
# ============================================================
print("Loading artifacts...")

with open(os.path.join(OUT_DIR, "disease_idx_clean.pkl"), "rb") as f:
    disease_idx = pickle.load(f)
with open(os.path.join(OUT_DIR, "disease_names.pkl"), "rb") as f:
    disease_names = pickle.load(f)
with open(os.path.join(OUT_DIR, "hpo_idx.pkl"), "rb") as f:
    hpo_idx = pickle.load(f)
with open(os.path.join(OUT_DIR, "hp_id_to_name.pkl"), "rb") as f:
    hp_names = pickle.load(f)
with open(os.path.join(OUT_DIR, "cluster_to_diseases.pkl"), "rb") as f:
    cluster_to_diseases = pickle.load(f)

cluster_labels = np.load(os.path.join(OUT_DIR, "hdbscan_cluster_labels.npy"))
raw_matrix = sparse.load_npz(os.path.join(OUT_DIR, "disease_hpo_matrix_clean_raw.npz")).toarray().astype(np.float32)

model = xgb.Booster()
model.load_model(os.path.join(OUT_DIR, "xgboost_ranker.json"))

disease_idx_rev = {v: k for k, v in disease_idx.items()}
hpo_idx_map = {k: v for k, v in hpo_idx.items()}

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))

print(f"Loaded. Matrix: {raw_matrix.shape}, Clusters: {len(set(cluster_labels))-1}")

# ============================================================
# CORE RANKING
# ============================================================
def build_features(query_vec, candidate_indices):
    features = []
    for d_idx in candidate_indices:
        disease_vec = raw_matrix[d_idx]
        overlap = np.minimum(query_vec, disease_vec).sum()
        q_total = query_vec.sum() + 1e-6
        d_total = disease_vec.sum() + 1e-6
        coverage_q = overlap / q_total
        coverage_d = overlap / d_total
        cosine_sim = overlap / (np.sqrt((query_vec**2).sum()) * np.sqrt((disease_vec**2).sum()) + 1e-6)
        n_matched = (np.minimum(query_vec, disease_vec) > 0).sum()
        features.append([overlap, coverage_q, coverage_d, cosine_sim, n_matched])
    return np.array(features)


def rank_diseases(query_hpo_ids, top_n=10):
    query_vec = np.zeros(raw_matrix.shape[1], dtype=np.float32)
    matched = []
    for hpo_id in query_hpo_ids:
        if hpo_id in hpo_idx_map:
            query_vec[hpo_idx_map[hpo_id]] = 1.0
            matched.append(hpo_id)

    # Step 1: find candidate diseases by symptom overlap
    # Only consider diseases that share at least 1 symptom with the query
    query_nonzero = set(np.where(query_vec > 0)[0])
    
    candidates = []
    for d_idx in range(raw_matrix.shape[0]):
        disease_nonzero = set(np.where(raw_matrix[d_idx] > 0)[0])
        shared = len(query_nonzero & disease_nonzero)
        if shared > 0:
            candidates.append((d_idx, shared))

    # Sort by overlap count, take top 200 candidates
    candidates.sort(key=lambda x: -x[1])
    candidate_indices = [c[0] for c in candidates[:200]]

    if not candidate_indices:
        candidate_indices = list(range(raw_matrix.shape[0]))

    # Step 2: XGBoost rank within candidates
    features = build_features(query_vec, candidate_indices)
    scores = model.predict(xgb.DMatrix(features))
    ranked_candidates = sorted(zip(candidate_indices, scores), key=lambda x: -x[1])[:top_n]

    results = []
    for d_idx, score in ranked_candidates:
        d_id = disease_idx_rev[d_idx]
        disease_vec = raw_matrix[d_idx]
        matched_symptoms = []
        for hpo_id in matched:
            col = hpo_idx_map[hpo_id]
            if disease_vec[col] > 0:
                matched_symptoms.append({
                    "hpo_id": hpo_id,
                    "name": hp_names.get(hpo_id, hpo_id),
                    "disease_freq": float(disease_vec[col])
                })
        matched_symptoms.sort(key=lambda x: -x["disease_freq"])
        results.append({
            "disease_id": d_id,
            "disease_name": disease_names[d_id],
            "score": float(score),
            "confidence_pct": round(float(score) * 100, 1),
            "matched_symptoms": matched_symptoms,
            "n_matched": len(matched_symptoms),
            "n_query": len(matched),
        })
    return results, query_vec


# ============================================================
# GROQ PERSONAS
# ============================================================
HOUSE_SYSTEM = """You are Dr. Gregory House, diagnostician. Blunt, brilliant, allergic to obvious answers.
You talk to your team, not the patient. You eliminate wrong answers ruthlessly.
You hunt for the zebra. Keep responses under 120 words. No bullet points. Raw diagnostic reasoning only."""

FOREMAN_SYSTEM = """You are Dr. Eric Foreman, neurologist. Methodical, evidence-based, skeptical of House's leaps.
Focus on what the data actually supports. Keep responses under 80 words."""

CHASE_SYSTEM = """You are Dr. Robert Chase. Devil's advocate, suggests the alternative diagnosis.
Sharp but sometimes misses the big picture. Keep responses under 80 words."""

CAMERON_SYSTEM = """You are Dr. Allison Cameron. Focus on rare overlapping syndromes.
Empathetic framing but medically precise. Keep responses under 80 words."""

HOUSE_QUOTES = [
    "It's never lupus.",
    "Everybody lies.",
    "When you hear hoofbeats, think horses. Unless you work here.",
    "I don't care about the patient, I care about the puzzle.",
    "Simplicity is the best disguise.",
]

def groq_call(system, user_msg, max_tokens=200, temperature=0.7):
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[Groq error: {str(e)}]"


def build_context(query_hpo_ids, ranked):
    symptom_names = [hp_names.get(h, h) for h in query_hpo_ids]
    lines = [f"PATIENT SYMPTOMS: {', '.join(symptom_names)}", "\nDIFFERENTIAL:"]
    for i, r in enumerate(ranked):
        matched = [s['name'] for s in r['matched_symptoms'][:3]]
        lines.append(f"  #{i+1} {r['disease_name']} (score={r['score']:.4f}, {r['n_matched']}/{r['n_query']} symptoms matched)")
        if matched:
            lines.append(f"      Overlapping: {', '.join(matched)}")
    return "\n".join(lines)


# ============================================================
# ROUTES
# ============================================================
class DiagnosisRequest(BaseModel):
    hpo_ids: List[str]
    top_n: int = 5

class HPOSearchRequest(BaseModel):
    query: str
    limit: int = 10


@app.get("/")
def root():
    return {"status": "FULLHOUSE backend running", "diseases": len(disease_idx), "hpo_terms": len(hpo_idx)}


@app.post("/diagnose")
def diagnose(req: DiagnosisRequest):
    ranked, query_vec = rank_diseases(req.hpo_ids, top_n=req.top_n)
    context = build_context(req.hpo_ids, ranked)
    top = ranked[0]

    house = groq_call(
        HOUSE_SYSTEM,
        f"{context}\n\nTop diagnosis is {top['disease_name']}. Walk your team through why this is right and the others are wrong.",
        max_tokens=200
    )
    foreman = groq_call(FOREMAN_SYSTEM, f"{context}\n\nWhat's your take on the top diagnosis?", max_tokens=120)
    chase = groq_call(CHASE_SYSTEM, f"{context}\n\nWhat's your take on the top diagnosis?", max_tokens=120)
    cameron = groq_call(CAMERON_SYSTEM, f"{context}\n\nWhat's your take on the top diagnosis?", max_tokens=120)

    return {
        "differential": ranked,
        "team": {
            "house": house,
            "foreman": foreman,
            "chase": chase,
            "cameron": cameron,
        },
        "house_quote": random.choice(HOUSE_QUOTES),
        "query_symptoms": [{"hpo_id": h, "name": hp_names.get(h, h)} for h in req.hpo_ids if h in hpo_idx_map],
    }


@app.get("/hpo/search")
def search_hpo(q: str, limit: int = 10):
    q_lower = q.lower()
    results = []
    for hpo_id, name in hp_names.items():
        if q_lower in name.lower() or q_lower in hpo_id.lower():
            results.append({"hpo_id": hpo_id, "name": name})
        if len(results) >= limit:
            break
    return {"results": results}


@app.get("/disease/{disease_id}")
def get_disease(disease_id: str):
    if disease_id not in disease_idx:
        return {"error": "Disease not found"}
    row_idx = disease_idx[disease_id]
    disease_vec = raw_matrix[row_idx]
    top_symptoms = []
    nonzero = np.where(disease_vec > 0)[0]
    sorted_cols = sorted(nonzero, key=lambda c: -disease_vec[c])
    hpo_idx_rev = {v: k for k, v in hpo_idx.items()}
    for col in sorted_cols[:20]:
        hpo_id = hpo_idx_rev[col]
        top_symptoms.append({
            "hpo_id": hpo_id,
            "name": hp_names.get(hpo_id, hpo_id),
            "frequency": float(disease_vec[col])
        })
    return {
        "disease_id": disease_id,
        "disease_name": disease_names.get(disease_id, "Unknown"),
        "cluster": int(cluster_labels[row_idx]),
        "top_symptoms": top_symptoms,
    }
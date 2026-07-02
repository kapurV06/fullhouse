"""
FULLHOUSE - Stage 6: Groq agentic layer
House MD personas generating diagnostic reasoning from XGBoost + SHAP output.
Runs locally. Needs GROQ_API_KEY in environment.
"""
import os
import numpy as np
import pickle
import xgboost as xgb
from scipy import sparse
from groq import Groq

OUT_DIR = "processed"

# ============================================================
# LOAD
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

print(f"Loaded. Matrix: {raw_matrix.shape}, Clusters: {len(set(cluster_labels))-1}")

# ============================================================
# RANKING FUNCTION
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


def rank_diseases(query_hpo_ids, top_n=5):
    query_vec = np.zeros(raw_matrix.shape[1], dtype=np.float32)
    matched = []
    for hpo_id in query_hpo_ids:
        if hpo_id in hpo_idx_map:
            query_vec[hpo_idx_map[hpo_id]] = 1.0
            matched.append(hpo_id)

    all_indices = list(range(raw_matrix.shape[0]))
    features = build_features(query_vec, all_indices)
    scores = model.predict(xgb.DMatrix(features))
    ranked = sorted(zip(all_indices, scores), key=lambda x: -x[1])[:top_n]

    results = []
    for d_idx, score in ranked:
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
            "matched_symptoms": matched_symptoms,
            "n_matched": len(matched_symptoms),
            "n_query": len(matched),
        })
    return results, query_vec


# ============================================================
# GROQ PERSONA PROMPTS
# ============================================================
HOUSE_SYSTEM = """You are Dr. Gregory House, diagnostician. You are blunt, brilliant, and allergic to obvious answers. 
You talk to your team, not the patient. You eliminate wrong answers ruthlessly. 
You always hunt for the zebra — the rare disease hiding behind common symptoms.
Keep responses under 120 words. No bullet points. Raw diagnostic reasoning only."""

FOREMAN_SYSTEM = """You are Dr. Eric Foreman, neurologist. Methodical, evidence-based, skeptical of House's leaps.
You focus on what the data actually supports. Keep responses under 80 words."""

CHASE_SYSTEM = """You are Dr. Robert Chase. You often play devil's advocate, suggesting the alternative diagnosis.
You're sharp but sometimes miss the big picture. Keep responses under 80 words."""

CAMERON_SYSTEM = """You are Dr. Allison Cameron. You focus on what the patient presentation means clinically,
especially rare overlapping syndromes. Empathetic framing but medically precise. Keep responses under 80 words."""

HOUSE_QUOTES = [
    "It's never lupus.",
    "Everybody lies.",
    "When you hear hoofbeats, think horses. Unless you work here.",
    "I don't care about the patient, I care about the puzzle.",
    "The most successful marriages are based on lies.",
    "Treating illness is why we became doctors. Treating patients is what makes most doctors miserable.",
    "Simplicity is the best disguise.",
]

import random

def build_differential_context(query_hpo_ids, ranked_results):
    symptom_names = [hp_names.get(h, h) for h in query_hpo_ids]
    lines = []
    lines.append(f"PATIENT SYMPTOMS: {', '.join(symptom_names)}")
    lines.append(f"\nDIFFERENTIAL (XGBoost ranked):")
    for i, r in enumerate(ranked_results):
        matched = [s['name'] for s in r['matched_symptoms'][:3]]
        lines.append(f"  #{i+1} {r['disease_name']} (score={r['score']:.4f}, {r['n_matched']}/{r['n_query']} symptoms matched)")
        if matched:
            lines.append(f"      Key overlapping symptoms: {', '.join(matched)}")
    return "\n".join(lines)


def generate_house_reasoning(query_hpo_ids, ranked_results, client):
    context = build_differential_context(query_hpo_ids, ranked_results)
    top = ranked_results[0]

    house_prompt = f"""{context}

The top diagnosis is {top['disease_name']}. 
Walk your team through why this is the answer and why the others are wrong.
Be House. Be brutal. Be right."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": HOUSE_SYSTEM},
            {"role": "user", "content": house_prompt}
        ],
        max_tokens=200,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_team_response(persona, system_prompt, query_hpo_ids, ranked_results, client):
    context = build_differential_context(query_hpo_ids, ranked_results)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{context}\n\nWhat's your take on the top diagnosis?"}
        ],
        max_tokens=120,
        temperature=0.6,
    )
    return response.choices[0].message.content


def run_fullhouse(query_hpo_ids):
    print("\n" + "=" * 60)
    print("FULLHOUSE DIAGNOSTIC RUN")
    print("=" * 60)

    ranked_results, query_vec = rank_diseases(query_hpo_ids, top_n=5)

    print(f"\nQuery symptoms: {[hp_names.get(h, h) for h in query_hpo_ids]}")
    print(f"\nTop diagnosis: {ranked_results[0]['disease_name']} (score={ranked_results[0]['score']:.4f})")

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nNo GROQ_API_KEY found. Set it to get House/team reasoning.")
        print("export GROQ_API_KEY=your_key_here")
        return ranked_results

    client = Groq(api_key=api_key)

    print("\n--- HOUSE ---")
    house_response = generate_house_reasoning(query_hpo_ids, ranked_results, client)
    print(house_response)

    print("\n--- FOREMAN ---")
    foreman_response = generate_team_response("Foreman", FOREMAN_SYSTEM, query_hpo_ids, ranked_results, client)
    print(foreman_response)

    print("\n--- CHASE ---")
    chase_response = generate_team_response("Chase", CHASE_SYSTEM, query_hpo_ids, ranked_results, client)
    print(chase_response)

    print("\n--- CAMERON ---")
    cameron_response = generate_team_response("Cameron", CAMERON_SYSTEM, query_hpo_ids, ranked_results, client)
    print(cameron_response)

    print(f'\n— "{random.choice(HOUSE_QUOTES)}"')

    return {
        "ranked": ranked_results,
        "house": house_response,
        "foreman": foreman_response,
        "chase": chase_response,
        "cameron": cameron_response,
    }


# ============================================================
# TEST RUN
# ============================================================
if __name__ == "__main__":
    melas_symptoms = [
        "HP:0001250",  # Seizure
        "HP:0003128",  # Lactic acidosis
        "HP:0002151",  # Increased circulating lactate
        "HP:0002401",  # Stroke-like episode
        "HP:0000726",  # Dementia
        "HP:0003200",  # Ragged-red muscle fibers
    ]
    run_fullhouse(melas_symptoms)
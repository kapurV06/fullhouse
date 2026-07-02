"""
FULLHOUSE - Stage 1b: Clean the feature matrix and build a TF-IDF-weighted
version for clustering, while preserving the raw version for XGBoost.
"""
import pickle
import numpy as np
import os

OUT_DIR = "processed"

print("=" * 60)
print("Loading raw matrix and lookups")
print("=" * 60)

matrix = np.load(os.path.join(OUT_DIR, "disease_hpo_matrix.npy"))
with open(os.path.join(OUT_DIR, "disease_idx.pkl"), "rb") as f:
    disease_idx = pickle.load(f)
with open(os.path.join(OUT_DIR, "disease_names.pkl"), "rb") as f:
    disease_names = pickle.load(f)
with open(os.path.join(OUT_DIR, "hpo_idx.pkl"), "rb") as f:
    hpo_idx = pickle.load(f)

disease_idx_rev = {v: k for k, v in disease_idx.items()}
print(f"Original matrix: {matrix.shape}")

print("\n" + "=" * 60)
print("STEP 1: Filtering sparse diseases (<3 symptoms)")
print("=" * 60)

symptom_counts = (matrix > 0).sum(axis=1)
keep_mask = symptom_counts >= 3
n_dropped = (~keep_mask).sum()
print(f"Dropping {n_dropped} diseases with <3 symptoms")

filtered_matrix = matrix[keep_mask]
kept_disease_ids = [disease_idx_rev[i] for i in np.where(keep_mask)[0]]
new_disease_idx = {d: i for i, d in enumerate(kept_disease_ids)}

print(f"Filtered matrix: {filtered_matrix.shape}")

np.save(os.path.join(OUT_DIR, "disease_hpo_matrix_clean_raw.npy"), filtered_matrix)
with open(os.path.join(OUT_DIR, "disease_idx_clean.pkl"), "wb") as f:
    pickle.dump(new_disease_idx, f)

print("\n" + "=" * 60)
print("STEP 2: Computing IDF weights for clustering matrix")
print("=" * 60)

n_diseases = filtered_matrix.shape[0]
doc_freq = (filtered_matrix > 0).sum(axis=0)
doc_freq_safe = np.where(doc_freq == 0, 1, doc_freq)

idf = np.log(n_diseases / doc_freq_safe) + 1.0

print(f"IDF range: min={idf.min():.3f}, max={idf.max():.3f}, mean={idf.mean():.3f}")

clustering_matrix = filtered_matrix * idf[np.newaxis, :]

print(f"Clustering matrix shape: {clustering_matrix.shape}")
print(f"Clustering matrix value range: [{clustering_matrix.min():.4f}, {clustering_matrix.max():.4f}]")

np.save(os.path.join(OUT_DIR, "disease_hpo_matrix_clustering_tfidf.npy"), clustering_matrix)
with open(os.path.join(OUT_DIR, "hpo_idf_weights.pkl"), "wb") as f:
    pickle.dump(idf, f)

print("\n" + "=" * 60)
print("VERIFICATION: MELAS syndrome before/after weighting")
print("=" * 60)

with open(os.path.join(OUT_DIR, "hp_id_to_name.pkl"), "rb") as f:
    hp_names = pickle.load(f)
hpo_idx_rev = {v: k for k, v in hpo_idx.items()}

melas_candidates = [d for d, name in disease_names.items() if 'MELAS' in name.upper()]
if melas_candidates:
    d = melas_candidates[0]
    if d in new_disease_idx:
        row_i = new_disease_idx[d]
        raw_row = filtered_matrix[row_i]
        weighted_row = clustering_matrix[row_i]

        top_raw = np.argsort(raw_row)[::-1][:5]
        top_weighted = np.argsort(weighted_row)[::-1][:5]

        print(f"\nDisease: {disease_names[d]}")
        print("\nTop 5 by RAW frequency:")
        for idx in top_raw:
            hp_id = hpo_idx_rev[idx]
            print(f"  {hp_id} {hp_names.get(hp_id,'?'):40s} raw={raw_row[idx]:.3f}")
        print("\nTop 5 by TF-IDF weighted (specificity-aware):")
        for idx in top_weighted:
            hp_id = hpo_idx_rev[idx]
            print(f"  {hp_id} {hp_names.get(hp_id,'?'):40s} weighted={weighted_row[idx]:.3f}  (idf={idf[idx]:.2f})")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"  disease_hpo_matrix_clean_raw.npy       : {filtered_matrix.shape} - for XGBoost")
print(f"  disease_hpo_matrix_clustering_tfidf.npy : {clustering_matrix.shape} - for UMAP/HDBSCAN")
print(f"  disease_idx_clean.pkl                   : updated disease_id -> row index (post-filter)")
print(f"  hpo_idf_weights.pkl                      : IDF weight per HPO term column")
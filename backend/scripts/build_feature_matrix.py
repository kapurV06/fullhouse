"""
FULLHOUSE - Stage 1: Parse HPO ontology + disease-phenotype annotations
into a clean disease x HPO-term feature matrix for downstream clustering/ML.
"""
import pronto
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle
import time
import os

DATA_DIR = "data"
OUT_DIR = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("STEP 1: Parsing hp.obo ontology")
print("=" * 60)
t0 = time.time()
ontology = pronto.Ontology(os.path.join(DATA_DIR, "hp.obo"))
print(f"Loaded {len(ontology.terms())} terms in {time.time()-t0:.1f}s")

hp_id_to_name = {}
hp_id_to_ancestors = {}

for term in ontology.terms():
    if term.id.startswith("HP:"):
        hp_id_to_name[term.id] = term.name
        try:
            ancestors = {a.id for a in term.superclasses() if a.id.startswith("HP:")}
            hp_id_to_ancestors[term.id] = ancestors
        except Exception:
            hp_id_to_ancestors[term.id] = set()

print(f"Built name lookup for {len(hp_id_to_name)} HP terms")

with open(os.path.join(OUT_DIR, "hp_id_to_name.pkl"), "wb") as f:
    pickle.dump(hp_id_to_name, f)
with open(os.path.join(OUT_DIR, "hp_id_to_ancestors.pkl"), "wb") as f:
    pickle.dump(hp_id_to_ancestors, f)

print("\n" + "=" * 60)
print("STEP 2: Parsing phenotype.hpoa")
print("=" * 60)

t0 = time.time()
df = pd.read_csv(os.path.join(DATA_DIR, "phenotype.hpoa"), sep="\t", comment="#", dtype=str)
print(f"Loaded {len(df)} rows in {time.time()-t0:.1f}s")
print(f"Columns: {list(df.columns)}")

df_p = df[df["aspect"] == "P"].copy()
print(f"Filtered to {len(df_p)} phenotypic-abnormality rows (aspect=P)")

FREQUENCY_MAP = {
    "HP:0040280": 1.0,
    "HP:0040281": 0.90,
    "HP:0040282": 0.55,
    "HP:0040283": 0.17,
    "HP:0040284": 0.02,
    "HP:0040285": 0.0,
}

def parse_frequency(val):
    if pd.isna(val) or val == "":
        return 0.5
    if val in FREQUENCY_MAP:
        return FREQUENCY_MAP[val]
    if "/" in val:
        try:
            num, denom = val.split("/")
            return float(num) / float(denom)
        except Exception:
            return 0.5
    if "%" in val:
        try:
            return float(val.replace("%", "")) / 100.0
        except Exception:
            return 0.5
    return 0.5

df_p["freq_weight"] = df_p["frequency"].apply(parse_frequency)

print("\n" + "=" * 60)
print("STEP 3: Building disease x HPO-term matrix")
print("=" * 60)

disease_hpo = df_p.groupby(["database_id", "disease_name", "hpo_id"])["freq_weight"].max().reset_index()

diseases = disease_hpo["database_id"].unique()
hpo_terms = disease_hpo["hpo_id"].unique()
print(f"Unique diseases: {len(diseases)}")
print(f"Unique HPO terms used: {len(hpo_terms)}")

disease_idx = {d: i for i, d in enumerate(diseases)}
hpo_idx = {h: i for i, h in enumerate(hpo_terms)}

matrix = np.zeros((len(diseases), len(hpo_terms)), dtype=np.float32)
for row in disease_hpo.itertuples(index=False):
    d_i = disease_idx[row.database_id]
    h_i = hpo_idx[row.hpo_id]
    matrix[d_i, h_i] = row.freq_weight

print(f"Matrix shape: {matrix.shape}")
print(f"Matrix sparsity: {(matrix == 0).sum() / matrix.size * 100:.2f}% zeros")

disease_names = disease_hpo.drop_duplicates("database_id").set_index("database_id")["disease_name"].to_dict()

np.save(os.path.join(OUT_DIR, "disease_hpo_matrix.npy"), matrix)
with open(os.path.join(OUT_DIR, "disease_idx.pkl"), "wb") as f:
    pickle.dump(disease_idx, f)
with open(os.path.join(OUT_DIR, "hpo_idx.pkl"), "wb") as f:
    pickle.dump(hpo_idx, f)
with open(os.path.join(OUT_DIR, "disease_names.pkl"), "wb") as f:
    pickle.dump(disease_names, f)

print("\n" + "=" * 60)
print("DONE. Saved to processed/")
print("=" * 60)
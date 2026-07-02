import numpy as np
from scipy import sparse
import os

OUT_DIR = "processed"

print("Loading dense raw matrix...")
matrix = np.load(os.path.join(OUT_DIR, "disease_hpo_matrix_clean_raw.npy"))
print(f"Dense: {matrix.shape}, {matrix.nbytes/1e6:.1f} MB")

sparse_matrix = sparse.csr_matrix(matrix)
print(f"Sparse: {sparse_matrix.data.nbytes/1e6:.1f} MB")

sparse.save_npz(os.path.join(OUT_DIR, "disease_hpo_matrix_clean_raw.npz"), sparse_matrix)
print("Saved disease_hpo_matrix_clean_raw.npz")
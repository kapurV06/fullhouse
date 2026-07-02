"""
FULLHOUSE - Convert the heavy tfidf matrix to sparse format BEFORE uploading to Colab.
Run this locally.
"""
import numpy as np
from scipy import sparse
import os

OUT_DIR = "processed"

print("Loading dense tfidf matrix...")
matrix = np.load(os.path.join(OUT_DIR, "disease_hpo_matrix_clustering_tfidf.npy"))
print(f"Dense matrix: {matrix.shape}, {matrix.nbytes/1e6:.1f} MB")

print("Converting to sparse CSR...")
sparse_matrix = sparse.csr_matrix(matrix)
print(f"Sparse matrix: {sparse_matrix.data.nbytes/1e6:.1f} MB (data only)")

sparse.save_npz(os.path.join(OUT_DIR, "disease_hpo_matrix_clustering_tfidf.npz"), sparse_matrix)
print("Saved disease_hpo_matrix_clustering_tfidf.npz")
print("\nYou can now upload the .npz file instead of the .npy file - it'll be a fraction of the size.")
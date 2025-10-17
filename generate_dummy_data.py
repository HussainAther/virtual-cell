# generate_dummy_data.py

import numpy as np
import os

# Parameters
num_samples = 50
num_genes = 100
out_dir = "dummy_data"

os.makedirs(out_dir, exist_ok=True)

# Simulate control expression: [N, G]
control = np.random.normal(loc=1.0, scale=0.2, size=(num_samples, num_genes))

# Simulate perturbed expression: slight random shift
perturbed = control + np.random.normal(loc=0.0, scale=0.1, size=(num_samples, num_genes))

# Simulate adjacency matrices: sample-specific graphs [N, G, G]
graphs = []
for _ in range(num_samples):
    A = np.random.rand(num_genes, num_genes)
    A = (A + A.T) / 2  # symmetric
    A[A < 0.95] = 0  # sparsify
    np.fill_diagonal(A, 0)
    graphs.append(A)

# Save as .npy
np.save(os.path.join(out_dir, "control.npy"), control)
np.save(os.path.join(out_dir, "perturbed.npy"), perturbed)
np.save(os.path.join(out_dir, "adjacency.npy"), graphs)

print("Dummy data generated in:", out_dir)


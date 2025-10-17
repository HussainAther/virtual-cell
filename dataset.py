# dataset.py

import torch
from torch.utils.data import Dataset
import numpy as np

class GeneGraphDataset(Dataset):
    def __init__(self, control_path, perturbed_path, graph_path, transform=None):
        self.control_data = np.load(control_path)  # shape: [N, G]
        self.perturbed_data = np.load(perturbed_path)  # shape: [N, G]
        self.graphs = np.load(graph_path, allow_pickle=True)  # List of N adjacency matrices or edge_index
        self.transform = transform

        # Precompute delta
        self.delta = self.perturbed_data - self.control_data
        self.num_samples = self.control_data.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = torch.tensor(self.control_data[idx], dtype=torch.float32)  # input: control expression
        y = torch.tensor(self.delta[idx], dtype=torch.float32)  # target: delta expression
        adjacency = torch.tensor(self.graphs[idx], dtype=torch.float32)  # could be edge_index too

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, adjacency, y


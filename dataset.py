# dataset.py

import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

class GeneGraphDataset(Dataset):
    def __init__(self, root, split="test", transform=None):
        self.root = root
        self.split = split
        self.features_dir = os.path.join(root, split, "features")
        self.graphs_dir = os.path.join(root, split, "graphs")
        self.gene_ids = self._load_gene_ids()
        self.samples = sorted([f for f in os.listdir(self.features_dir) if f.endswith(".npy")])
        self.transform = transform

    def _load_gene_ids(self):
        path = os.path.join(self.root, self.split, "gene_ids.txt")
        with open(path, "r") as f:
            return [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def _load_graph(self, pert_id):
        edge_path = os.path.join(self.graphs_dir, f"{pert_id}_edges.csv")
        edges = pd.read_csv(edge_path, header=None).values
        edge_index = torch.tensor(edges.T, dtype=torch.long)
        return edge_index

    def get(self, idx):
        sample_file = self.samples[idx]
        pert_id = sample_file.replace(".npy", "")
        x = torch.tensor(np.load(os.path.join(self.features_dir, sample_file)), dtype=torch.float)
        edge_index = self._load_graph(pert_id)

        data = Data(x=x, edge_index=edge_index)
        data.perturbation_id = pert_id
        data.gene_ids = self.gene_ids

        if self.transform:
            data = self.transform(data)
        return data

def load_test_data(batch_size=1, root="data"):
    dataset = GeneGraphDataset(root=root, split="test")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


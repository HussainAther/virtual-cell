import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.gcn = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        return F.relu(self.gcn(x, edge_index, edge_weight))

class GeneGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GeneGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.layers.append(GCNLayer(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        return x
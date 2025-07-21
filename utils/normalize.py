import torch
from torch_geometric.utils import add_self_loops, degree

def normalize_adjacency(edge_index, num_nodes, edge_weight=None):
    edge_index, edge_weight = add_self_loops(edge_index, edge_weight=edge_weight, fill_value=1.0, num_nodes=num_nodes)
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes, dtype=torch.float)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return edge_index, norm
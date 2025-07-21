import torch
from torch_geometric.data import Data

def load_synthetic_data():
    num_nodes = 100
    input_dim = 10
    output_dim = 10

    x = torch.randn(num_nodes, input_dim)
    y = torch.randn(num_nodes, output_dim)

    edge_index = torch.randint(0, num_nodes, (2, 500))
    edge_weight = torch.ones(edge_index.size(1))

    return Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)
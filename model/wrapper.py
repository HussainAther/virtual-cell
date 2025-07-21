import torch.nn as nn
from model.encoders import GeneEncoder
from model.gnn_layers import GeneGNN
from model.decoders import GeneDecoder

class VirtualCellModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, gnn_layers=2, residual=True, delta=False):
        super(VirtualCellModel, self).__init__()
        self.encoder = GeneEncoder(input_dim, hidden_dim)
        self.gnn = GeneGNN(hidden_dim, hidden_dim, gnn_layers)
        self.decoder = GeneDecoder(hidden_dim, output_dim, residual, delta)

    def forward(self, x, edge_index, edge_weight=None):
        encoded = self.encoder(x)
        gnn_out = self.gnn(encoded, edge_index, edge_weight)
        return self.decoder(gnn_out, original_input=x)
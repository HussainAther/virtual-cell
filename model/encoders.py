import torch
import torch.nn as nn

class GeneEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GeneEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, x):
        return self.encoder(x)
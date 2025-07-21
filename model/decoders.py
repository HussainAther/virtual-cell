import torch
import torch.nn as nn

class GeneDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, residual=True, delta=False):
        super(GeneDecoder, self).__init__()
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.residual = residual
        self.delta = delta

    def forward(self, x, original_input=None):
        output = self.linear(x)
        if self.residual and original_input is not None:
            if self.delta:
                return original_input + output
            else:
                return (original_input + output) / 2
        return output
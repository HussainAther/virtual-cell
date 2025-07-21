import torch
import torch.nn as nn
import torch.optim as optim
from model.wrapper import VirtualCellModel
from dataset import load_synthetic_data
from utils.normalize import normalize_adjacency

def train():
    data = load_synthetic_data()
    edge_index, edge_weight = normalize_adjacency(data.edge_index, data.x.size(0), data.edge_weight)

    model = VirtualCellModel(input_dim=data.x.size(1), hidden_dim=64, output_dim=data.y.size(1))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        out = model(data.x, edge_index, edge_weight)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

if __name__ == "__main__":
    train()
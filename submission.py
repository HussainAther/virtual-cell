import torch
import pandas as pd
from torch_geometric.data import DataLoader
from src.data.dataset import GeneGraphDataset
from src.model.full_model import GeneGraphPredictor
import os

def load_model(model_path, input_dim, hidden_dim, output_dim):
    model = GeneGraphPredictor(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def generate_predictions(model, dataloader):
    all_predictions = []
    for batch in dataloader:
        with torch.no_grad():
            out = model(batch.x, batch.edge_index, batch.batch)
            for sid, pred in zip(batch.sid, out):
                prediction = pred.cpu().numpy()
                for gene_idx, val in enumerate(prediction):
                    all_predictions.append({
                        "sample_id": sid,
                        "gene_index": gene_idx,
                        "predicted_expression": val
                    })
    return pd.DataFrame(all_predictions)

def save_submission(predictions_df, output_path="outputs/predictions.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    predictions_df.to_csv(output_path, index=False)
    print(f"Saved submission file to: {output_path}")

if __name__ == "__main__":
    # These must be adjusted based on your trained model
    model_path = "checkpoints/best_model.pt"
    input_dim = 128
    hidden_dim = 256
    output_dim = 100  # total genes being predicted

    dataset = GeneGraphDataset(root="data/test")  # test data path
    dataloader = DataLoader(dataset, batch_size=8)

    model = load_model(model_path, input_dim, hidden_dim, output_dim)
    predictions_df = generate_predictions(model, dataloader)
    save_submission(predictions_df)


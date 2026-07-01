import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import numpy as np


def evaluate(model, dataset, device='cpu', batch_size=64):
    """
    Evaluate global model on dataset.
    Returns accuracy and macro F1 score.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            sensor = batch['sensor'].to(device)
            labels = batch['label'].to(device)

            logits, _, _ = model({'sensor': sensor, 'label': labels})
            preds        = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = (all_preds == all_labels).mean() * 100
    f1       = f1_score(all_labels, all_preds, average='macro') * 100

    return round(accuracy, 2), round(f1, 2)

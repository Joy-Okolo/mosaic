import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import random
import copy
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data.cremad import build_cremad_client_datasets, CREMADDataset
from models.mosaic_cremad import MOSAICCREMADModel

# ── paths ──────────────────────────────────────────────────────────────────
HOME       = os.path.expanduser('~')
AUDIO_BASE = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/audio/mfcc/crema_d'
VIDEO_BASE = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/video/mobilenet_v2/crema_d'

# ── seed ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate(model, test_dataset, device):
    model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            audio  = batch['audio'].to(device)
            video  = batch['video'].to(device)
            labels = batch['label'].to(device)
            logits, _, _ = model({'audio': audio, 'video': video})
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100.0 * (all_preds == all_labels).mean()
    uar = 100.0 * f1_score(all_labels, all_preds, average='macro')
    return round(acc, 2), round(uar, 2)

# ── FedAvg CREMA-D client ──────────────────────────────────────────────────
class FedAvgCREMADClient:
    def __init__(self, client_id, dataset, config, device='cpu'):
        self.client_id = client_id
        self.config    = config
        self.device    = device

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

    def train(self, global_state, num_epochs, lr):
        model = MOSAICCREMADModel(self.config).to(self.device)
        model.load_state_dict(global_state)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                audio  = batch['audio'].to(self.device)
                video  = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                logits, _, _ = model({'audio': audio, 'video': video})
                loss = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return copy.deepcopy(model.state_dict()), avg_loss, len(self.dataloader.dataset)

# ── FedAvg CREMA-D server ──────────────────────────────────────────────────
class FedAvgCREMADServer:
    def __init__(self, config, device='cpu'):
        self.config       = config
        self.device       = device
        self.global_model = MOSAICCREMADModel(config).to(device)

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate(self, updates):
        total = sum(n for _, _, n in updates)
        new_state = copy.deepcopy(updates[0][0])
        for key in new_state:
            new_state[key] = sum(
                u[0][key].float() * (u[2] / total) for u in updates
            )
        self.global_model.load_state_dict(new_state)

    def run_round(self, clients, num_epochs, lr):
        per_round = self.config['federation']['clients_per_round']
        selected  = random.sample(clients, min(per_round, len(clients)))

        updates      = []
        round_losses = []

        global_state = self.get_global_state()
        for client in selected:
            state, loss, n = client.train(global_state, num_epochs, lr)
            updates.append((state, loss, n))
            round_losses.append(loss)

        self.aggregate(updates)
        return sum(round_losses) / len(round_losses)

# ── main ───────────────────────────────────────────────────────────────────
def main():
    with open('configs/mosaic_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    config['cremad'] = {
        'embedding_dim': 128,
        'dropout': 0.2,
        'num_classes': 4
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,   default=None)
    parser.add_argument('--lr',   type=float, default=None)
    parser.add_argument('--fold', type=int,   default=1)
    args = parser.parse_args()

    if args.seed is not None: config['experiment']['seed']        = args.seed
    if args.lr   is not None: config['training']['learning_rate'] = args.lr

    fold = args.fold
    set_seed(config['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr     = config['training']['learning_rate']

    print("=" * 60)
    print("FedAvg Baseline — CREMA-D (Audio + Video Emotion Recognition)")
    print("=" * 60)
    print(f"Fold:    {fold}")
    print(f"Rounds:  {config['federation']['num_rounds']}")
    print(f"LR:      {lr}")
    print(f"Device:  {device}")
    print("=" * 60)

    print("\nLoading CREMA-D data...")
    client_datasets = build_cremad_client_datasets(AUDIO_BASE, VIDEO_BASE, fold=fold)
    test_dataset    = CREMADDataset(AUDIO_BASE, VIDEO_BASE, fold=fold, split='test')

    print("Creating clients...")
    clients = []
    for client_id, dataset in client_datasets.items():
        client = FedAvgCREMADClient(
            client_id=client_id,
            dataset=dataset,
            config=config,
            device=device
        )
        clients.append(client)
    print(f"  Total clients: {len(clients)}")

    server     = FedAvgCREMADServer(config, device)
    num_rounds = config['federation']['num_rounds']
    log_every  = config['experiment']['log_every']
    epochs     = config['training']['local_epochs']
    best_uar   = 0.0

    os.makedirs('results', exist_ok=True)

    acc, uar = evaluate(server.global_model, test_dataset, device)
    print(f"\nRound   0 | Acc: {acc}%  UAR: {uar}%")

    print("\nStarting FedAvg CREMA-D training...")
    for r in range(1, num_rounds + 1):
        loss = server.run_round(clients, epochs, lr)
        if r % log_every == 0:
            acc, uar = evaluate(server.global_model, test_dataset, device)
            print(f"Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")
            if uar > best_uar:
                best_uar = uar
                torch.save(
                    server.global_model.state_dict(),
                    f'results/best_fedavg_cremad_fold{fold}.pt'
                )
                print(f"          New best UAR: {best_uar}%")

    print("\n" + "=" * 60)
    print(f"FedAvg CREMA-D complete. Best UAR: {best_uar}%")
    print("=" * 60)

if __name__ == '__main__':
    main()

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
from torch.utils.data import DataLoader, Subset
from data.datasets import UCIHARDataset
from data.partition import DirichletPartitioner
from evaluation.metrics import evaluate

# ── Harmony encoder (2D CNN matching USC architecture) ─────────────────────
class HarmonyCNNEncoder(nn.Module):
    """Single-modality CNN encoder — matches Harmony's USC model."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Conv2d(64, 64, 2), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Conv2d(64, 32, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Dropout(),
            nn.Conv2d(32, 16, 1), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        # x: (batch, 128, 3) → unsqueeze → (batch, 1, 128, 3)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = x.view(x.size(0), 16, -1)   # (batch, 16, 126*1) = (batch, 16, 126)
        return x

class HarmonyFusionModel(nn.Module):
    """Multi-modal fusion model for Stage 2."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.acc_encoder  = HarmonyCNNEncoder()
        self.gyro_encoder = HarmonyCNNEncoder()
        self.gru = nn.GRU(126, 120, 2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(1920, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    def forward(self, acc, gyro):
        acc_feat  = self.acc_encoder(acc)
        gyro_feat = self.gyro_encoder(gyro)
        fused = (acc_feat + gyro_feat) / 2.0
        fused, _ = self.gru(fused)
        fused = fused.contiguous().view(fused.size(0), -1)
        return self.classifier(fused)

class HarmonySingleModel(nn.Module):
    """Single-modality model for Stage 1."""
    def __init__(self, num_classes=6):
        super().__init__()
        self.encoder = HarmonyCNNEncoder()
        self.gru = nn.GRU(126, 120, 2, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(1920, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Linear(512, 128),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        x = self.encoder(x)
        x, _ = self.gru(x)
        x = x.contiguous().view(x.size(0), -1)
        return self.classifier(x)

# ── Modality assignment (matches data prep script) ─────────────────────────
def get_modality(client_id):
    if client_id < 20:   return 'both'
    elif client_id < 70: return 'acc'
    else:                return 'gyro'

# ── Harmony client ─────────────────────────────────────────────────────────
class HarmonyClient:
    def __init__(self, client_id, dataset, indices, config, device='cpu'):
        self.client_id = client_id
        self.config    = config
        self.device    = device
        self.modality  = get_modality(client_id)
        self.num_classes = 6

        subset = Subset(dataset, indices)
        self.dataloader = DataLoader(
            subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

    def _extract_modality(self, sensor):
        # sensor: (batch, 128, 6) → split into acc (0:3) and gyro (3:6)
        acc  = sensor[:, :, 0:3]
        gyro = sensor[:, :, 3:6]
        return acc, gyro

    def train_stage1(self, global_enc_state, global_cls_state, num_epochs, lr):
        """Stage 1: train single-modal model, return encoder + classifier states."""
        model = HarmonySingleModel(self.num_classes).to(self.device)
        # load global encoder and classifier
        model.encoder.load_state_dict(global_enc_state)
        model.classifier.load_state_dict(global_cls_state)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                     weight_decay=self.config['training']['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                acc, gyro = self._extract_modality(sensor)

                # use acc for T2 clients, gyro for T3, acc for T1 in stage 1
                x = acc if self.modality in ('acc', 'both') else gyro

                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return (copy.deepcopy(model.encoder.state_dict()),
                copy.deepcopy(model.classifier.state_dict()),
                avg_loss,
                len(self.dataloader.dataset))

    def train_stage2(self, global_model_state, num_epochs, lr):
        """Stage 2: fine-tune fusion model (multi-modal clients only)."""
        assert self.modality == 'both', "Stage 2 only for multi-modal clients"
        model = HarmonyFusionModel(self.num_classes).to(self.device)
        model.load_state_dict(global_model_state)

        # freeze encoders, only train classifier + GRU
        for p in model.acc_encoder.parameters():  p.requires_grad = False
        for p in model.gyro_encoder.parameters(): p.requires_grad = False

        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                acc, gyro = self._extract_modality(sensor)

                optimizer.zero_grad()
                out = model(acc, gyro)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return (copy.deepcopy(model.state_dict()), avg_loss,
                len(self.dataloader.dataset))

# ── Harmony server ─────────────────────────────────────────────────────────
class HarmonyServer:
    def __init__(self, config, device='cpu'):
        self.config  = config
        self.device  = device
        self.num_classes = 6

        # Stage 1: separate unimodal models per modality
        self.acc_model  = HarmonySingleModel(self.num_classes).to(device)
        self.gyro_model = HarmonySingleModel(self.num_classes).to(device)

        # Stage 2: fusion model (initialized after stage 1)
        self.fusion_model = HarmonyFusionModel(self.num_classes).to(device)

    def aggregate_stage1(self, acc_updates, gyro_updates):
        """Aggregate acc encoders separately from gyro encoders."""
        # aggregate acc model
        if acc_updates:
            total = sum(n for _, _, n in acc_updates)
            new_enc = copy.deepcopy(acc_updates[0][0])
            new_cls = copy.deepcopy(acc_updates[0][1])
            for key in new_enc:
                new_enc[key] = sum(u[0][key].float() * (u[2]/total) for u in acc_updates)
            for key in new_cls:
                new_cls[key] = sum(u[1][key].float() * (u[2]/total) for u in acc_updates)
            self.acc_model.encoder.load_state_dict(new_enc)
            self.acc_model.classifier.load_state_dict(new_cls)

        # aggregate gyro model
        if gyro_updates:
            total = sum(n for _, _, n in gyro_updates)
            new_enc = copy.deepcopy(gyro_updates[0][0])
            new_cls = copy.deepcopy(gyro_updates[0][1])
            for key in new_enc:
                new_enc[key] = sum(u[0][key].float() * (u[2]/total) for u in gyro_updates)
            for key in new_cls:
                new_cls[key] = sum(u[1][key].float() * (u[2]/total) for u in gyro_updates)
            self.gyro_model.encoder.load_state_dict(new_enc)
            self.gyro_model.classifier.load_state_dict(new_cls)

    def init_fusion_from_stage1(self):
        """Initialize fusion model encoders from stage 1 trained encoders."""
        fusion_state = self.fusion_model.state_dict()
        acc_enc_state  = self.acc_model.encoder.state_dict()
        gyro_enc_state = self.gyro_model.encoder.state_dict()
        for key in acc_enc_state:
            fusion_state[f'acc_encoder.{key}']  = acc_enc_state[key]
            fusion_state[f'gyro_encoder.{key}'] = gyro_enc_state[key]
        self.fusion_model.load_state_dict(fusion_state)

    def aggregate_stage2(self, fusion_updates):
        """Simple weighted average of fusion model updates."""
        if not fusion_updates:
            return
        total = sum(n for _, n in fusion_updates)
        new_state = copy.deepcopy(fusion_updates[0][0])
        for key in new_state:
            new_state[key] = sum(u[0][key].float() * (u[1]/total) for u in fusion_updates)
        self.fusion_model.load_state_dict(new_state)

    def run_stage1_round(self, clients, lr, num_epochs):
        """One round of Stage 1 — unimodal federated learning."""
        per_round  = self.config['federation']['clients_per_round']
        selected   = random.sample(clients, min(per_round, len(clients)))

        acc_updates  = []
        gyro_updates = []
        round_losses = []

        for client in selected:
            if client.modality in ('acc', 'both'):
                enc_s = copy.deepcopy(self.acc_model.encoder.state_dict())
                cls_s = copy.deepcopy(self.acc_model.classifier.state_dict())
            else:
                enc_s = copy.deepcopy(self.gyro_model.encoder.state_dict())
                cls_s = copy.deepcopy(self.gyro_model.classifier.state_dict())

            enc_upd, cls_upd, loss, n = client.train_stage1(enc_s, cls_s, num_epochs, lr)
            round_losses.append(loss)

            if client.modality in ('acc', 'both'):
                acc_updates.append((enc_upd, cls_upd, n))
            if client.modality in ('gyro', 'both'):
                gyro_updates.append((enc_upd, cls_upd, n))

        self.aggregate_stage1(acc_updates, gyro_updates)
        return sum(round_losses) / len(round_losses)

    def run_stage2_round(self, clients, lr, num_epochs):
        """One round of Stage 2 — fusion federated learning (multi-modal only)."""
        multi_clients = [c for c in clients if c.modality == 'both']
        if not multi_clients:
            return 0.0

        per_round = max(1, self.config['federation']['clients_per_round'] // 5)
        selected  = random.sample(multi_clients, min(per_round, len(multi_clients)))

        fusion_updates = []
        round_losses   = []

        for client in selected:
            global_state = copy.deepcopy(self.fusion_model.state_dict())
            upd, loss, n = client.train_stage2(global_state, num_epochs, lr)
            fusion_updates.append((upd, n))
            round_losses.append(loss)

        self.aggregate_stage2(fusion_updates)
        return sum(round_losses) / len(round_losses)

# ── evaluate fusion model ──────────────────────────────────────────────────
def evaluate_harmony(fusion_model, test_dataset, device):
    """Evaluate the fusion model on test set."""
    fusion_model.eval()
    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            sensor = batch['sensor'].to(device)
            labels = batch['label'].to(device)
            acc  = sensor[:, :, 0:3]
            gyro = sensor[:, :, 3:6]
            out  = fusion_model(acc, gyro)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc_score  = 100.0 * (all_preds == all_labels).mean()

    from sklearn.metrics import f1_score
    f1 = 100.0 * f1_score(all_labels, all_preds, average='weighted')
    return round(acc_score, 2), round(f1, 2)

# ── main ───────────────────────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    with open('configs/mosaic_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--lr',   type=float, default=None)
    args = parser.parse_args()

    if args.seed is not None: config['experiment']['seed'] = args.seed
    if args.lr   is not None: config['training']['learning_rate'] = args.lr

    set_seed(config['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # stage split: first 70% of rounds = stage 1, last 30% = stage 2
    num_rounds   = config['federation']['num_rounds']
    stage1_rounds = int(num_rounds * 0.7)
    stage2_rounds = num_rounds - stage1_rounds
    lr           = config['training']['learning_rate']
    local_epochs = config['training']['local_epochs']
    log_every    = config['experiment']['log_every']

    print("=" * 60)
    print("Harmony Baseline — Heterogeneous Multi-Modal FL")
    print("=" * 60)
    print(f"Clients:       {config['federation']['num_clients']}")
    print(f"Total Rounds:  {num_rounds}  (Stage1={stage1_rounds}, Stage2={stage2_rounds})")
    print(f"Device:        {device}")
    print(f"LR:            {lr}")
    print("=" * 60)

    print("\nLoading data...")
    train_dataset = UCIHARDataset(split='train')
    test_dataset  = UCIHARDataset(split='test')

    partitioner = DirichletPartitioner(
        num_clients=config['federation']['num_clients'],
        alpha=0.5,
        seed=config['experiment']['seed']
    )
    client_indices = partitioner.partition(train_dataset.get_labels())

    print("Creating clients...")
    all_clients = []
    for cid in range(config['federation']['num_clients']):
        client = HarmonyClient(
            client_id=cid,
            dataset=train_dataset,
            indices=client_indices[cid],
            config=config,
            device=device
        )
        all_clients.append(client)

    multi_count = sum(1 for c in all_clients if c.modality == 'both')
    acc_count   = sum(1 for c in all_clients if c.modality == 'acc')
    gyro_count  = sum(1 for c in all_clients if c.modality == 'gyro')
    print(f"  Multi-modal clients: {multi_count}")
    print(f"  Acc-only clients:    {acc_count}")
    print(f"  Gyro-only clients:   {gyro_count}")

    server   = HarmonyServer(config, device=device)
    best_acc = 0.0
    os.makedirs('results', exist_ok=True)

    # ── Stage 1: Unimodal FL ──────────────────────────────────────────────
    print(f"\nStage 1: Unimodal FL ({stage1_rounds} rounds)...")
    for r in range(1, stage1_rounds + 1):
        loss = server.run_stage1_round(all_clients, lr, local_epochs)
        if r % log_every == 0:
            # evaluate acc model as proxy during stage 1
            server.init_fusion_from_stage1()
            acc_score, f1 = evaluate_harmony(server.fusion_model, test_dataset, device)
            print(f"Stage1 Round {r:>3} | Loss: {loss:.4f} | "
                  f"Accuracy: {acc_score}%  F1: {f1}%")

    # ── Transfer stage 1 encoders to fusion model ─────────────────────────
    print("\nInitializing fusion model from Stage 1 encoders...")
    server.init_fusion_from_stage1()

    # ── Stage 2: Federated Fusion ─────────────────────────────────────────
    print(f"\nStage 2: Federated Fusion ({stage2_rounds} rounds)...")
    for r in range(1, stage2_rounds + 1):
        loss = server.run_stage2_round(all_clients, lr, local_epochs)
        if r % log_every == 0:
            acc_score, f1 = evaluate_harmony(server.fusion_model, test_dataset, device)
            print(f"Stage2 Round {r:>3} | Loss: {loss:.4f} | "
                  f"Accuracy: {acc_score}%  F1: {f1}%")
            if acc_score > best_acc:
                best_acc = acc_score
                torch.save(
                    server.fusion_model.state_dict(),
                    'results/best_harmony_model.pt'
                )
                print(f"             New best: {best_acc}%")

    print("\n" + "=" * 60)
    print(f"Harmony complete. Best accuracy: {best_acc}%")
    print("=" * 60)

if __name__ == '__main__':
    main()

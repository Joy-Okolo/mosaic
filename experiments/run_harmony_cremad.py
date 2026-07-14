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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
from data.cremad import build_cremad_client_datasets, CREMADDataset
from models.mosaic_cremad import AudioEncoder, VideoEncoder

# ── paths ──────────────────────────────────────────────────────────────────
HOME       = os.path.expanduser('~')
AUDIO_BASE = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/audio/mfcc/crema_d'
VIDEO_BASE = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/video/mobilenet_v2/crema_d'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ── modality assignment ────────────────────────────────────────────────────
def get_modality(client_idx):
    """Assign modality based on client index — matches UCI-HAR tier structure."""
    if client_idx < 20:   return 'both'
    elif client_idx < 70: return 'audio'
    else:                 return 'video'

# ── single modality dataset wrapper ───────────────────────────────────────
class SingleModalDataset(Dataset):
    def __init__(self, base_dataset, modality):
        self.base     = base_dataset
        self.modality = modality

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        return {
            'audio': item['audio'],
            'video': item['video'],
            'label': item['label']
        }

# ── Harmony models ─────────────────────────────────────────────────────────
class HarmonyAudioModel(nn.Module):
    """Single-modal audio model for Stage 1."""
    def __init__(self, embedding_dim=128, num_classes=4):
        super().__init__()
        self.encoder    = AudioEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

class HarmonyVideoModel(nn.Module):
    """Single-modal video model for Stage 1."""
    def __init__(self, embedding_dim=128, num_classes=4):
        super().__init__()
        self.encoder    = VideoEncoder(embedding_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))

class HarmonyFusionModel(nn.Module):
    """Multi-modal fusion model for Stage 2."""
    def __init__(self, embedding_dim=128, num_classes=4):
        super().__init__()
        self.audio_encoder = AudioEncoder(embedding_dim)
        self.video_encoder = VideoEncoder(embedding_dim)
        self.classifier    = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, audio, video):
        emb_a = self.audio_encoder(audio)
        emb_v = self.video_encoder(video)
        fused = torch.cat([emb_a, emb_v], dim=1)
        return self.classifier(fused)

# ── Harmony client ─────────────────────────────────────────────────────────
class HarmonyCREMADClient:
    def __init__(self, client_id, client_idx, dataset, config, device='cpu'):
        self.client_id  = client_id
        self.client_idx = client_idx
        self.config     = config
        self.device     = device
        self.modality   = get_modality(client_idx)
        self.num_classes = 4
        self.embedding_dim = 128

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

    def train_stage1(self, global_enc_state, global_cls_state, num_epochs, lr):
        """Stage 1: train single-modal model."""
        if self.modality in ('audio', 'both'):
            model = HarmonyAudioModel(self.embedding_dim, self.num_classes).to(self.device)
        else:
            model = HarmonyVideoModel(self.embedding_dim, self.num_classes).to(self.device)

        model.encoder.load_state_dict(global_enc_state)
        model.classifier.load_state_dict(global_cls_state)

        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                labels = batch['label'].to(self.device)
                if self.modality in ('audio', 'both'):
                    x = batch['audio'].to(self.device)
                else:
                    x = batch['video'].to(self.device)

                optimizer.zero_grad()
                out  = model(x)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        n = len(self.dataloader.dataset)
        return (copy.deepcopy(model.encoder.state_dict()),
                copy.deepcopy(model.classifier.state_dict()),
                avg_loss, n)

    def train_stage2(self, global_model_state, num_epochs, lr):
        """Stage 2: fine-tune fusion model (multi-modal clients only)."""
        assert self.modality == 'both'
        model = HarmonyFusionModel(self.embedding_dim, self.num_classes).to(self.device)
        model.load_state_dict(global_model_state)

        # freeze encoders
        for p in model.audio_encoder.parameters(): p.requires_grad = False
        for p in model.video_encoder.parameters(): p.requires_grad = False

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
                audio  = batch['audio'].to(self.device)
                video  = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                out  = model(audio, video)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        n = len(self.dataloader.dataset)
        return copy.deepcopy(model.state_dict()), avg_loss, n

# ── Harmony server ─────────────────────────────────────────────────────────
class HarmonyCREMADServer:
    def __init__(self, config, device='cpu'):
        self.config        = config
        self.device        = device
        self.embedding_dim = 128
        self.num_classes   = 4

        self.audio_model  = HarmonyAudioModel(self.embedding_dim, self.num_classes).to(device)
        self.video_model  = HarmonyVideoModel(self.embedding_dim, self.num_classes).to(device)
        self.fusion_model = HarmonyFusionModel(self.embedding_dim, self.num_classes).to(device)

    def _weighted_avg(self, updates, key_fn, weight_fn):
        total  = sum(weight_fn(u) for u in updates)
        result = copy.deepcopy(key_fn(updates[0]))
        for key in result:
            result[key] = sum(
                key_fn(u)[key].float() * (weight_fn(u) / total)
                for u in updates
            )
        return result

    def run_stage1_round(self, clients, lr, num_epochs):
        per_round = self.config['federation']['clients_per_round']
        selected  = random.sample(clients, min(per_round, len(clients)))

        audio_updates = []
        video_updates = []
        round_losses  = []

        for client in selected:
            if client.modality in ('audio', 'both'):
                enc_s = copy.deepcopy(self.audio_model.encoder.state_dict())
                cls_s = copy.deepcopy(self.audio_model.classifier.state_dict())
            else:
                enc_s = copy.deepcopy(self.video_model.encoder.state_dict())
                cls_s = copy.deepcopy(self.video_model.classifier.state_dict())

            enc_u, cls_u, loss, n = client.train_stage1(enc_s, cls_s, num_epochs, lr)
            round_losses.append(loss)

            if client.modality in ('audio', 'both'):
                audio_updates.append((enc_u, cls_u, n))
            if client.modality in ('video', 'both'):
                video_updates.append((enc_u, cls_u, n))

        if audio_updates:
            new_enc = self._weighted_avg(audio_updates, lambda u: u[0], lambda u: u[2])
            new_cls = self._weighted_avg(audio_updates, lambda u: u[1], lambda u: u[2])
            self.audio_model.encoder.load_state_dict(new_enc)
            self.audio_model.classifier.load_state_dict(new_cls)

        if video_updates:
            new_enc = self._weighted_avg(video_updates, lambda u: u[0], lambda u: u[2])
            new_cls = self._weighted_avg(video_updates, lambda u: u[1], lambda u: u[2])
            self.video_model.encoder.load_state_dict(new_enc)
            self.video_model.classifier.load_state_dict(new_cls)

        return sum(round_losses) / len(round_losses)

    def init_fusion_from_stage1(self):
        """Initialize fusion encoders from stage 1 trained encoders."""
        fusion_state = self.fusion_model.state_dict()
        for key in self.audio_model.encoder.state_dict():
            fusion_state[f'audio_encoder.{key}'] = self.audio_model.encoder.state_dict()[key]
        for key in self.video_model.encoder.state_dict():
            fusion_state[f'video_encoder.{key}'] = self.video_model.encoder.state_dict()[key]
        self.fusion_model.load_state_dict(fusion_state)

    def run_stage2_round(self, clients, lr, num_epochs):
        multi_clients = [c for c in clients if c.modality == 'both']
        if not multi_clients:
            return 0.0

        per_round = max(1, self.config['federation']['clients_per_round'] // 5)
        selected  = random.sample(multi_clients, min(per_round, len(multi_clients)))

        updates      = []
        round_losses = []

        for client in selected:
            global_state = copy.deepcopy(self.fusion_model.state_dict())
            upd, loss, n = client.train_stage2(global_state, num_epochs, lr)
            updates.append((upd, n))
            round_losses.append(loss)

        if updates:
            new_state = self._weighted_avg(updates, lambda u: u[0], lambda u: u[1])
            self.fusion_model.load_state_dict(new_state)

        return sum(round_losses) / len(round_losses)

# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate_harmony(fusion_model, test_dataset, device):
    fusion_model.eval()
    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            audio  = batch['audio'].to(device)
            video  = batch['video'].to(device)
            labels = batch['label'].to(device)
            out    = fusion_model(audio, video)
            preds  = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc = 100.0 * (all_preds == all_labels).mean()
    uar = 100.0 * f1_score(all_labels, all_preds, average='macro')
    return round(acc, 2), round(uar, 2)

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

    num_rounds    = config['federation']['num_rounds']
    stage1_rounds = int(num_rounds * 0.7)
    stage2_rounds = num_rounds - stage1_rounds
    log_every     = config['experiment']['log_every']
    epochs        = config['training']['local_epochs']

    print("=" * 60)
    print("Harmony Baseline — CREMA-D (Audio + Video Emotion Recognition)")
    print("=" * 60)
    print(f"Fold:          {fold}")
    print(f"Total Rounds:  {num_rounds} (Stage1={stage1_rounds}, Stage2={stage2_rounds})")
    print(f"LR:            {lr}")
    print(f"Device:        {device}")
    print("=" * 60)

    print("\nLoading CREMA-D data...")
    client_datasets = build_cremad_client_datasets(AUDIO_BASE, VIDEO_BASE, fold=fold)
    test_dataset    = CREMADDataset(AUDIO_BASE, VIDEO_BASE, fold=fold, split='test')

    print("Creating clients...")
    clients = []
    for idx, (client_id, dataset) in enumerate(client_datasets.items()):
        client = HarmonyCREMADClient(
            client_id=client_id,
            client_idx=idx,
            dataset=dataset,
            config=config,
            device=device
        )
        clients.append(client)

    multi = sum(1 for c in clients if c.modality == 'both')
    audio = sum(1 for c in clients if c.modality == 'audio')
    video = sum(1 for c in clients if c.modality == 'video')
    print(f"  Both:  {multi} | Audio-only: {audio} | Video-only: {video}")

    server   = HarmonyCREMADServer(config, device)
    best_uar = 0.0
    os.makedirs('results', exist_ok=True)

    # Stage 1
    print(f"\nStage 1: Unimodal FL ({stage1_rounds} rounds)...")
    for r in range(1, stage1_rounds + 1):
        loss = server.run_stage1_round(clients, lr, epochs)
        if r % log_every == 0:
            server.init_fusion_from_stage1()
            acc, uar = evaluate_harmony(server.fusion_model, test_dataset, device)
            print(f"Stage1 Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")

    print("\nInitializing fusion from Stage 1 encoders...")
    server.init_fusion_from_stage1()

    # Stage 2
    print(f"\nStage 2: Federated Fusion ({stage2_rounds} rounds)...")
    for r in range(1, stage2_rounds + 1):
        loss = server.run_stage2_round(clients, lr, epochs)
        if r % log_every == 0:
            acc, uar = evaluate_harmony(server.fusion_model, test_dataset, device)
            print(f"Stage2 Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")
            if uar > best_uar:
                best_uar = uar
                torch.save(
                    server.fusion_model.state_dict(),
                    f'results/best_harmony_cremad_fold{fold}.pt'
                )
                print(f"             New best UAR: {best_uar}%")

    print("\n" + "=" * 60)
    print(f"Harmony CREMA-D complete. Best UAR: {best_uar}%")
    print("=" * 60)

if __name__ == '__main__':
    main()

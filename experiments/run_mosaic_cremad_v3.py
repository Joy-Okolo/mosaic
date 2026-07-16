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
import time
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from data.cremad import build_cremad_client_datasets, CREMADDataset
from models.mosaic_cremad import MOSAICCREMADModel
from fl.mcat import MCATScheduler

# ── paths ──────────────────────────────────────────────────────────────────
HOME         = os.path.expanduser('~')
AUDIO_BASE   = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/audio/mfcc/crema_d'
VIDEO_BASE   = f'{HOME}/fed-multimodal/fed_multimodal/output/feature/video/mobilenet_v2/crema_d'

# ── modality assignment ────────────────────────────────────────────────────
def get_modality(client_idx):
    if client_idx < 20:   return 'both'
    elif client_idx < 70: return 'audio'
    else:                 return 'video'

def get_tier(client_idx):
    if client_idx < 20:   return 1
    elif client_idx < 70: return 2
    else:                 return 3

def build_client_profiles(clients):
    """Build client_profiles dict for MCATScheduler."""
    profiles = {}
    for idx, client in enumerate(clients):
        modality = get_modality(idx)
        if modality == 'both':
            profiles[idx] = ['audio', 'video']
        elif modality == 'audio':
            profiles[idx] = ['audio']
        else:
            profiles[idx] = ['video']
    return profiles

def build_client_tiers(clients):
    """Build client_tiers dict for duty-cycle filtering."""
    return {idx: get_tier(idx) for idx in range(len(clients))}

def build_client_data_sizes(clients):
    """Build client_data_sizes dict for MCATScheduler."""
    return {idx: len(client.dataloader.dataset)
            for idx, client in enumerate(clients)}

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

# ── MOSAIC CREMA-D client ──────────────────────────────────────────────────
class MOSAICCREMADClient:
    def __init__(self, client_id, client_idx, dataset, config, device='cpu'):
        self.client_id  = client_id
        self.client_idx = client_idx
        self.config     = config
        self.device     = device
        self.modality   = get_modality(client_idx)

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

        self.mu_prox       = config['losses']['mu_prox']
        self.lambda_anchor = config['losses'].get('lambda_anchor', 0.1)
        self.tau           = config['losses'].get('tau_contrastive', 0.3)

    def train(self, global_state, num_epochs, lr):
        model = MOSAICCREMADModel(self.config).to(self.device)
        model.load_state_dict(global_state)
        global_params = [p.data.clone() for p in model.parameters()]

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
                logits, emb_a, emb_b = model({'audio': audio, 'video': video})

                loss_task  = criterion(logits, labels)
                loss_prox  = sum(
                    torch.norm(p - g) ** 2
                    for p, g in zip(model.parameters(), global_params)
                )
                loss_anchor = self._contrastive_loss(emb_a, emb_b)
                loss = loss_task + self.mu_prox * loss_prox + self.lambda_anchor * loss_anchor

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss  += loss_task.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return copy.deepcopy(model.state_dict()), avg_loss, len(self.dataloader.dataset)

    def _contrastive_loss(self, emb_a, emb_b):
        emb_a = nn.functional.normalize(emb_a, dim=1)
        emb_b = nn.functional.normalize(emb_b, dim=1)
        sim   = torch.matmul(emb_a, emb_b.T) / self.tau
        labels = torch.arange(emb_a.size(0)).to(emb_a.device)
        return (nn.functional.cross_entropy(sim, labels) +
                nn.functional.cross_entropy(sim.T, labels)) / 2

# ── MOSAIC CREMA-D server with full MCAT + duty-cycle ─────────────────────
class MOSAICCREMADServerV3:
    """
    MOSAIC server with:
    1. MCAT scheduling — prioritizes clients by compute, data, modality rarity
    2. Duty-cycle availability — T3 clients only available every N rounds
    3. Modality-cohort aggregation — encoder updates routed by modality
    4. Divergence-guided intra-cohort weighting
    """
    def __init__(self, config, client_profiles, client_data_sizes,
                 client_tiers, device='cpu'):
        self.config       = config
        self.device       = device
        self.round        = 0
        self.client_tiers = client_tiers
        self.duty_cycle   = config['client_tiers']['tier3_duty_cycle']
        self.alpha        = config.get('aggregation', {}).get('divergence_alpha', 0.5)

        self.global_model = MOSAICCREMADModel(config).to(device)

        # MCAT scheduler
        self.scheduler = MCATScheduler(config, client_profiles, client_data_sizes)

    def get_global_state(self):
        return copy.deepcopy(self.global_model.state_dict())

    def _get_available_indices(self, all_indices):
        """Filter by duty-cycle — T3 clients only available every duty_cycle rounds."""
        available = []
        for idx in all_indices:
            tier = self.client_tiers.get(idx, 1)
            if tier == 3:
                if self.round % self.duty_cycle == 0:
                    available.append(idx)
            else:
                available.append(idx)
        return available

    def _compute_deltas(self, updates, global_state):
        deltas = []
        for state, _, _, _ in updates:
            delta = {k: state[k].float() - global_state[k].float() for k in state}
            deltas.append(delta)
        return deltas

    def _divergence_weights(self, deltas, keys):
        if len(deltas) == 0: return []
        if len(deltas) == 1: return [1.0]
        mean_delta = {k: torch.stack([d[k] for d in deltas]).mean(dim=0) for k in keys}
        scores = []
        for delta in deltas:
            div = sum(torch.norm(delta[k].float() - mean_delta[k].float()).item() for k in keys)
            scores.append(1.0 + self.alpha * div)
        total = sum(scores)
        return [s / total for s in scores]

    def aggregate(self, updates, global_state):
        """Modality-cohort aggregation with divergence-guided weighting."""
        audio_updates = [u for u in updates if u[3] in ('audio', 'both')]
        video_updates = [u for u in updates if u[3] in ('video', 'both')]

        new_state = copy.deepcopy(global_state)

        # audio encoder
        if audio_updates:
            audio_d    = self._compute_deltas(audio_updates, global_state)
            audio_keys = [k for k in new_state if k.startswith('encoder_a.')]
            audio_w    = self._divergence_weights(audio_d, audio_keys)
            for k in audio_keys:
                new_state[k] = sum(
                    w * audio_updates[i][0][k].float()
                    for i, w in enumerate(audio_w)
                )

        # video encoder
        if video_updates:
            video_d    = self._compute_deltas(video_updates, global_state)
            video_keys = [k for k in new_state if k.startswith('encoder_b.')]
            video_w    = self._divergence_weights(video_d, video_keys)
            for k in video_keys:
                new_state[k] = sum(
                    w * video_updates[i][0][k].float()
                    for i, w in enumerate(video_w)
                )

        # shared layers — standard size-weighted average
        shared_keys = [k for k in new_state
                       if not k.startswith('encoder_a.') and
                          not k.startswith('encoder_b.')]
        total_n = sum(u[2] for u in updates)
        for k in shared_keys:
            new_state[k] = sum(u[0][k].float() * (u[2] / total_n) for u in updates)

        self.global_model.load_state_dict(new_state)

    def run_round(self, clients, num_epochs, lr):
        self.round += 1
        per_round    = self.config['federation']['clients_per_round']
        global_state = self.get_global_state()

        # duty-cycle filter
        all_indices       = list(range(len(clients)))
        available_indices = self._get_available_indices(all_indices)

        # MCAT selects from available clients
        selected_indices = self.scheduler.select_clients(
            available_indices,
            min(per_round, len(available_indices))
        )
        selected = [clients[i] for i in selected_indices]

        updates      = []
        round_losses = []

        for client in selected:
            start = time.time()
            state, loss, n = client.train(global_state, num_epochs, lr)
            elapsed = time.time() - start

            updates.append((state, loss, n, client.modality))
            round_losses.append(loss)

            # update MCAT timing
            self.scheduler.update_time(client.client_idx, elapsed)

        self.aggregate(updates, global_state)
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
    config['aggregation'] = {'divergence_alpha': 0.5}

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',          type=int,   default=None)
    parser.add_argument('--lr',            type=float, default=None)
    parser.add_argument('--fold',          type=int,   default=1)
    parser.add_argument('--lambda_anchor', type=float, default=None)
    args = parser.parse_args()

    if args.seed          is not None: config['experiment']['seed']        = args.seed
    if args.lr            is not None: config['training']['learning_rate'] = args.lr
    if args.lambda_anchor is not None: config['losses']['lambda_anchor']   = args.lambda_anchor

    fold = args.fold
    set_seed(config['experiment']['seed'])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr     = config['training']['learning_rate']

    print("=" * 65)
    print("MOSAIC+MCAT v3 — CREMA-D | Full MCAT + Duty-Cycle + Cohort Agg")
    print("=" * 65)
    print(f"Fold:             {fold}")
    print(f"Rounds:           {config['federation']['num_rounds']}")
    print(f"LR:               {lr}")
    print(f"Lambda anchor:    {config['losses']['lambda_anchor']}")
    print(f"Duty-cycle (T3):  every {config['client_tiers']['tier3_duty_cycle']} rounds")
    print(f"Device:           {device}")
    print("=" * 65)

    print("\nLoading CREMA-D data...")
    client_datasets = build_cremad_client_datasets(AUDIO_BASE, VIDEO_BASE, fold=fold)
    test_dataset    = CREMADDataset(AUDIO_BASE, VIDEO_BASE, fold=fold, split='test')

    print("Creating clients...")
    clients = []
    for idx, (client_id, dataset) in enumerate(client_datasets.items()):
        client = MOSAICCREMADClient(
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
    print(f"  Total: {len(clients)} | Both: {multi} | Audio-only: {audio} | Video-only: {video}")

    # build MCAT inputs
    client_profiles   = build_client_profiles(clients)
    client_tiers      = build_client_tiers(clients)
    client_data_sizes = build_client_data_sizes(clients)

    t1 = sum(1 for t in client_tiers.values() if t == 1)
    t2 = sum(1 for t in client_tiers.values() if t == 2)
    t3 = sum(1 for t in client_tiers.values() if t == 3)
    print(f"  Tiers: T1={t1} (both, always available) | "
          f"T2={t2} (audio, always available) | "
          f"T3={t3} (video, duty-cycle every {config['client_tiers']['tier3_duty_cycle']} rounds)")

    server     = MOSAICCREMADServerV3(
        config, client_profiles, client_data_sizes, client_tiers, device
    )
    num_rounds = config['federation']['num_rounds']
    log_every  = config['experiment']['log_every']
    epochs     = config['training']['local_epochs']
    best_uar   = 0.0

    os.makedirs('results', exist_ok=True)

    acc, uar = evaluate(server.global_model, test_dataset, device)
    print(f"\nRound   0 | Acc: {acc}%  UAR: {uar}%")

    print("\nStarting MOSAIC CREMA-D v3 training...")
    for r in range(1, num_rounds + 1):
        loss = server.run_round(clients, epochs, lr)
        if r % log_every == 0:
            acc, uar = evaluate(server.global_model, test_dataset, device)
            print(f"Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")
            if uar > best_uar:
                best_uar = uar
                torch.save(
                    server.global_model.state_dict(),
                    f'results/best_mosaic_cremad_v3_fold{fold}.pt'
                )
                print(f"          New best UAR: {best_uar}%")

    print("\n" + "=" * 65)
    print(f"MOSAIC CREMA-D v3 complete. Best UAR: {best_uar}%")
    print("=" * 65)

if __name__ == '__main__':
    main()

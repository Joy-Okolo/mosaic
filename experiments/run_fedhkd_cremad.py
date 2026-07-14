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

def get_modality(client_idx):
    if client_idx < 20:   return 'both'
    elif client_idx < 70: return 'audio'
    else:                 return 'video'

# ── LoRA adapter ───────────────────────────────────────────────────────────
class LoRAAdapter(nn.Module):
    def __init__(self, d_in, d_out, rank=16):
        super().__init__()
        self.A = nn.Linear(d_in,  rank,  bias=False)
        self.B = nn.Linear(rank,  d_out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))

# ── Shared base encoder (modality-agnostic) ────────────────────────────────
class FedHKDBaseEncoderCREMAD(nn.Module):
    """
    Modality-agnostic base encoder for CREMA-D.
    Projects both audio and video embeddings to a common space.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.audio_enc = AudioEncoder(embedding_dim)
        self.video_enc = VideoEncoder(embedding_dim)

        # shared projection to common space
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, modality='audio'):
        if modality == 'audio':
            h = self.audio_enc(x)
        else:
            h = self.video_enc(x)
        return self.proj(h)

# ── Modality discriminator ─────────────────────────────────────────────────
class ModalityDiscriminator(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ── Attentive fusion ───────────────────────────────────────────────────────
class AttentiveFusionCREMAD(nn.Module):
    def __init__(self, embedding_dim=128, num_classes=4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, feat_audio, feat_video):
        global_feat  = (feat_audio + feat_video) / 2.0
        score_audio  = torch.sum(feat_audio  * global_feat, dim=1, keepdim=True)
        score_video  = torch.sum(feat_video  * global_feat, dim=1, keepdim=True)
        scores       = torch.softmax(torch.cat([score_audio, score_video], dim=1), dim=1)
        fused        = scores[:, 0:1] * feat_audio + scores[:, 1:2] * feat_video
        return self.classifier(fused)

# ── FedHKD CREMA-D client ──────────────────────────────────────────────────
class FedHKDCREMADClient:
    def __init__(self, client_id, client_idx, dataset, config, device='cpu'):
        self.client_id     = client_id
        self.client_idx    = client_idx
        self.config        = config
        self.device        = device
        self.modality      = get_modality(client_idx)
        self.num_classes   = 4
        self.embedding_dim = 128
        self.lora_rank     = 16

        self.dataloader = DataLoader(
            dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

    def _get_input(self, batch):
        if self.modality in ('audio', 'both'):
            return batch['audio'].to(self.device), 'audio'
        else:
            return batch['video'].to(self.device), 'video'

    def train_stage1(self, base_state, cls_state, disc_state, num_epochs, lr):
        """Stage 1: modality-agnostic pretraining with adversarial loss."""
        base_enc = FedHKDBaseEncoderCREMAD(self.embedding_dim).to(self.device)
        base_enc.load_state_dict(base_state)

        classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        ).to(self.device)
        classifier.load_state_dict(cls_state)

        discriminator = ModalityDiscriminator(self.embedding_dim).to(self.device)
        discriminator.load_state_dict(disc_state)

        mod_label = 0 if self.modality in ('audio', 'both') else 1

        opt_model = torch.optim.Adam(
            list(base_enc.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        opt_disc = torch.optim.Adam(
            discriminator.parameters(),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0

        base_enc.train()
        classifier.train()
        discriminator.train()

        for _ in range(num_epochs):
            for batch in self.dataloader:
                x, modality = self._get_input(batch)
                labels = batch['label'].to(self.device)
                bsz    = x.size(0)
                mod_labels = torch.full((bsz,), mod_label, dtype=torch.long).to(self.device)

                # train discriminator
                opt_disc.zero_grad()
                with torch.no_grad():
                    feat = base_enc(x, modality)
                loss_disc = criterion(discriminator(feat.detach()), mod_labels)
                loss_disc.backward()
                opt_disc.step()

                # train base encoder (adversarially)
                opt_model.zero_grad()
                feat     = base_enc(x, modality)
                loss_cls = criterion(classifier(feat), labels)
                uniform  = torch.full((bsz,), 1, dtype=torch.long).to(self.device)
                loss_adv = criterion(discriminator(feat), uniform)
                loss     = loss_cls + 0.1 * loss_adv
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(base_enc.parameters()) + list(classifier.parameters()), 1.0
                )
                opt_model.step()
                total_loss  += loss_cls.item()
                num_batches += 1

        n = len(self.dataloader.dataset)
        return (copy.deepcopy(base_enc.state_dict()),
                copy.deepcopy(classifier.state_dict()),
                copy.deepcopy(discriminator.state_dict()),
                total_loss / max(num_batches, 1), n)

    def train_stage2(self, base_state, lora_state, cls_state, num_epochs, lr):
        """Stage 2: modality-specific LoRA fine-tuning."""
        base_enc = FedHKDBaseEncoderCREMAD(self.embedding_dim).to(self.device)
        base_enc.load_state_dict(base_state)
        for p in base_enc.parameters():
            p.requires_grad = False

        lora = LoRAAdapter(self.embedding_dim, self.embedding_dim, self.lora_rank).to(self.device)
        lora.load_state_dict(lora_state)

        classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        ).to(self.device)
        classifier.load_state_dict(cls_state)

        optimizer = torch.optim.Adam(
            list(lora.parameters()) + list(classifier.parameters()),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0

        lora.train()
        classifier.train()

        for _ in range(num_epochs):
            for batch in self.dataloader:
                x, modality = self._get_input(batch)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    feat = base_enc(x, modality)
                feat     = feat + lora(feat)
                loss     = criterion(classifier(feat), labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(lora.parameters()) + list(classifier.parameters()), 1.0
                )
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        n = len(self.dataloader.dataset)
        return (copy.deepcopy(lora.state_dict()),
                copy.deepcopy(classifier.state_dict()),
                total_loss / max(num_batches, 1), n)

    def train_stage3(self, base_state, lora_state, fusion_state, num_epochs, lr):
        """Stage 3: personalized attentive fusion (multi-modal only)."""
        assert self.modality == 'both'

        base_enc = FedHKDBaseEncoderCREMAD(self.embedding_dim).to(self.device)
        base_enc.load_state_dict(base_state)
        for p in base_enc.parameters():
            p.requires_grad = False

        lora = LoRAAdapter(self.embedding_dim, self.embedding_dim, self.lora_rank).to(self.device)
        lora.load_state_dict(lora_state)
        for p in lora.parameters():
            p.requires_grad = False

        fusion = AttentiveFusionCREMAD(self.embedding_dim, self.num_classes).to(self.device)
        fusion.load_state_dict(fusion_state)

        optimizer  = torch.optim.Adam(
            fusion.parameters(), lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )
        criterion  = nn.CrossEntropyLoss()
        total_loss = 0.0
        num_batches = 0
        fusion.train()

        for _ in range(num_epochs):
            for batch in self.dataloader:
                audio  = batch['audio'].to(self.device)
                video  = batch['video'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()
                with torch.no_grad():
                    fa = base_enc(audio, 'audio') + lora(base_enc(audio, 'audio'))
                    fv = base_enc(video, 'video') + lora(base_enc(video, 'video'))
                out  = fusion(fa, fv)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        n = len(self.dataloader.dataset)
        return copy.deepcopy(fusion.state_dict()), total_loss / max(num_batches, 1), n

# ── FedHKD CREMA-D server ──────────────────────────────────────────────────
class FedHKDCREMADServer:
    def __init__(self, config, device='cpu'):
        self.config        = config
        self.device        = device
        self.embedding_dim = 128
        self.num_classes   = 4
        self.lora_rank     = 16

        self.global_base   = FedHKDBaseEncoderCREMAD(self.embedding_dim).to(device)
        self.global_lora   = LoRAAdapter(self.embedding_dim, self.embedding_dim, self.lora_rank).to(device)
        self.global_cls    = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, self.num_classes)
        ).to(device)
        self.global_disc   = ModalityDiscriminator(self.embedding_dim).to(device)
        self.global_fusion = AttentiveFusionCREMAD(self.embedding_dim, self.num_classes).to(device)

    def _wavg(self, updates, key_fn, wt_fn):
        total  = sum(wt_fn(u) for u in updates)
        result = copy.deepcopy(key_fn(updates[0]))
        for k in result:
            result[k] = sum(key_fn(u)[k].float() * (wt_fn(u) / total) for u in updates)
        return result

    def run_stage1_round(self, clients, lr, num_epochs):
        per_round = self.config['federation']['clients_per_round']
        selected  = random.sample(clients, min(per_round, len(clients)))
        updates   = []
        losses    = []
        for c in selected:
            bu, cu, du, loss, n = c.train_stage1(
                copy.deepcopy(self.global_base.state_dict()),
                copy.deepcopy(self.global_cls.state_dict()),
                copy.deepcopy(self.global_disc.state_dict()),
                num_epochs, lr
            )
            updates.append((bu, cu, du, n))
            losses.append(loss)
        self.global_base.load_state_dict(self._wavg(updates, lambda u: u[0], lambda u: u[3]))
        self.global_cls.load_state_dict(self._wavg(updates,  lambda u: u[1], lambda u: u[3]))
        self.global_disc.load_state_dict(self._wavg(updates, lambda u: u[2], lambda u: u[3]))
        return sum(losses) / len(losses)

    def run_stage2_round(self, clients, lr, num_epochs):
        per_round = self.config['federation']['clients_per_round']
        selected  = random.sample(clients, min(per_round, len(clients)))
        updates   = []
        losses    = []
        for c in selected:
            lu, cu, loss, n = c.train_stage2(
                copy.deepcopy(self.global_base.state_dict()),
                copy.deepcopy(self.global_lora.state_dict()),
                copy.deepcopy(self.global_cls.state_dict()),
                num_epochs, lr
            )
            updates.append((lu, cu, n))
            losses.append(loss)
        self.global_lora.load_state_dict(self._wavg(updates, lambda u: u[0], lambda u: u[2]))
        self.global_cls.load_state_dict(self._wavg(updates,  lambda u: u[1], lambda u: u[2]))
        return sum(losses) / len(losses)

    def run_stage3_round(self, clients, lr, num_epochs):
        multi    = [c for c in clients if c.modality == 'both']
        if not multi: return 0.0
        per_round = max(1, self.config['federation']['clients_per_round'] // 5)
        selected  = random.sample(multi, min(per_round, len(multi)))
        updates   = []
        losses    = []
        for c in selected:
            fu, loss, n = c.train_stage3(
                copy.deepcopy(self.global_base.state_dict()),
                copy.deepcopy(self.global_lora.state_dict()),
                copy.deepcopy(self.global_fusion.state_dict()),
                num_epochs, lr
            )
            updates.append((fu, n))
            losses.append(loss)
        self.global_fusion.load_state_dict(self._wavg(updates, lambda u: u[0], lambda u: u[1]))
        return sum(losses) / len(losses)

# ── evaluation ─────────────────────────────────────────────────────────────
def evaluate_fedhkd(server, test_dataset, device):
    base_enc = server.global_base
    lora     = server.global_lora
    fusion   = server.global_fusion
    base_enc.eval(); lora.eval(); fusion.eval()

    loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            audio  = batch['audio'].to(device)
            video  = batch['video'].to(device)
            labels = batch['label'].to(device)
            fa = base_enc(audio, 'audio'); fa = fa + lora(fa)
            fv = base_enc(video, 'video'); fv = fv + lora(fv)
            out   = fusion(fa, fv)
            preds = out.argmax(dim=1)
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

    config['cremad'] = {'embedding_dim': 128, 'dropout': 0.2, 'num_classes': 4}

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
    stage1_rounds = int(num_rounds * 0.60)
    stage2_rounds = int(num_rounds * 0.20)
    stage3_rounds = num_rounds - stage1_rounds - stage2_rounds
    log_every     = config['experiment']['log_every']
    epochs        = config['training']['local_epochs']

    print("=" * 60)
    print("FedHKD Baseline — CREMA-D (Audio + Video Emotion Recognition)")
    print("=" * 60)
    print(f"Fold:    {fold} | Rounds: {num_rounds} | LR: {lr} | Device: {device}")
    print(f"Stage1={stage1_rounds} | Stage2={stage2_rounds} | Stage3={stage3_rounds}")
    print("=" * 60)

    print("\nLoading CREMA-D data...")
    client_datasets = build_cremad_client_datasets(AUDIO_BASE, VIDEO_BASE, fold=fold)
    test_dataset    = CREMADDataset(AUDIO_BASE, VIDEO_BASE, fold=fold, split='test')

    print("Creating clients...")
    clients = []
    for idx, (client_id, dataset) in enumerate(client_datasets.items()):
        clients.append(FedHKDCREMADClient(client_id, idx, dataset, config, device))
    print(f"  Total: {len(clients)} | Both: {sum(1 for c in clients if c.modality=='both')} | "
          f"Audio: {sum(1 for c in clients if c.modality=='audio')} | "
          f"Video: {sum(1 for c in clients if c.modality=='video')}")

    server   = FedHKDCREMADServer(config, device)
    best_uar = 0.0
    os.makedirs('results', exist_ok=True)

    print(f"\nStage 1 ({stage1_rounds} rounds)...")
    for r in range(1, stage1_rounds + 1):
        loss = server.run_stage1_round(clients, lr, epochs)
        if r % log_every == 0:
            acc, uar = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage1 Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")

    print(f"\nStage 2 ({stage2_rounds} rounds)...")
    for r in range(1, stage2_rounds + 1):
        loss = server.run_stage2_round(clients, lr, epochs)
        if r % log_every == 0:
            acc, uar = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage2 Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")

    print(f"\nStage 3 ({stage3_rounds} rounds)...")
    for r in range(1, stage3_rounds + 1):
        loss = server.run_stage3_round(clients, lr, epochs)
        if r % log_every == 0:
            acc, uar = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage3 Round {r:>3} | Loss: {loss:.4f} | Acc: {acc}%  UAR: {uar}%")
            if uar > best_uar:
                best_uar = uar
                torch.save(server.global_fusion.state_dict(),
                           f'results/best_fedhkd_cremad_fold{fold}.pt')
                print(f"             New best UAR: {best_uar}%")

    print("\n" + "=" * 60)
    print(f"FedHKD CREMA-D complete. Best UAR: {best_uar}%")
    print("=" * 60)

if __name__ == '__main__':
    main()

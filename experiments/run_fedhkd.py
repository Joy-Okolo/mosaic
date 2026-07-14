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
from sklearn.metrics import f1_score

# ── Modality assignment (matches MOSAIC tier structure) ────────────────────
def get_modality(client_id):
    if client_id < 20:   return 'both'
    elif client_id < 70: return 'acc'
    else:                return 'gyro'

# ── LoRA adapter (lightweight low-rank update) ─────────────────────────────
class LoRAAdapter(nn.Module):
    """
    Low-rank adapter: W' = W + B*A where A in R^(r x d_in), B in R^(d_out x r)
    Matches FedHKD's modality-specific fine-tuning via LoRA.
    """
    def __init__(self, d_in, d_out, rank=16):
        super().__init__()
        self.A = nn.Linear(d_in,  rank,   bias=False)
        self.B = nn.Linear(rank,  d_out,  bias=False)
        nn.init.kaiming_uniform_(self.A.weight)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))

# ── Base encoder (modality-agnostic, shared across all clients) ────────────
class FedHKDBaseEncoder(nn.Module):
    """
    Modality-agnostic base encoder Eb.
    Captures common knowledge independent of modality and client.
    Uses 1D CNN + BiLSTM matching MOSAIC's sensor encoder design.
    """
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.embedding_dim = embedding_dim

        # lightweight linear embedding to unify modality input dimensions
        # both acc and gyro are (128, 3) so same embedding
        self.modal_embed = nn.Linear(3, 32)

        # shared 1D CNN layers
        self.cnn = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )

        # shared BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch, 128, 3)
        x = self.modal_embed(x)          # (batch, 128, 32)
        x = x.permute(0, 2, 1)          # (batch, 32, 128) for Conv1d
        x = self.cnn(x)                  # (batch, 128, 128)
        x = x.permute(0, 2, 1)          # (batch, 128, 128) for LSTM
        x, _ = self.lstm(x)             # (batch, 128, embedding_dim)
        x = x[:, -1, :]                 # take last timestep
        x = self.dropout(x)
        return x                         # (batch, embedding_dim)

# ── Modality discriminator (used in Stage 1 adversarial training) ──────────
class ModalityDiscriminator(nn.Module):
    """
    Discriminator D that tries to predict which modality a feature came from.
    Base encoder is trained adversarially to fool this discriminator.
    """
    def __init__(self, embedding_dim=128, num_modalities=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_modalities)
        )

    def forward(self, x):
        return self.net(x)

# ── Full FedHKD client model ───────────────────────────────────────────────
class FedHKDModel(nn.Module):
    """
    Full model = base encoder Eb + modality-specific LoRA Em + classifier.
    Stage 1: only Eb + classifier trained (Em frozen/zero)
    Stage 2: Eb frozen, Em (LoRA) trained
    Stage 3: Eb + Em frozen, client-specific head fine-tuned
    """
    def __init__(self, embedding_dim=128, num_classes=6, lora_rank=16):
        super().__init__()
        self.base_encoder = FedHKDBaseEncoder(embedding_dim)

        # modality-specific LoRA adapter (Em)
        self.lora_adapter = LoRAAdapter(embedding_dim, embedding_dim, rank=lora_rank)

        # shared classifier head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x, use_lora=False):
        feat = self.base_encoder(x)
        if use_lora:
            feat = feat + self.lora_adapter(feat)
        return self.classifier(feat), feat

# ── Attentive fusion for multi-modal clients ───────────────────────────────
class AttentiveFusion(nn.Module):
    """
    Correlation-based attentive fusion from FedHKD Stage 3.
    Computes importance weights for each modality based on
    correlation with global feature representation.
    """
    def __init__(self, embedding_dim=128, num_classes=6):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, feat_acc, feat_gyro):
        # compute global feature as average
        global_feat = (feat_acc + feat_gyro) / 2.0

        # correlation scores
        score_acc  = torch.sum(feat_acc  * global_feat, dim=1, keepdim=True)
        score_gyro = torch.sum(feat_gyro * global_feat, dim=1, keepdim=True)

        # softmax weights
        scores = torch.cat([score_acc, score_gyro], dim=1)
        weights = torch.softmax(scores, dim=1)

        # weighted sum
        fused = weights[:, 0:1] * feat_acc + weights[:, 1:2] * feat_gyro
        return self.classifier(fused)

# ── FedHKD Client ─────────────────────────────────────────────────────────
class FedHKDClient:
    def __init__(self, client_id, dataset, indices, config, device='cpu'):
        self.client_id   = client_id
        self.config      = config
        self.device      = device
        self.modality    = get_modality(client_id)
        self.num_classes = 6
        self.embedding_dim = 128
        self.lora_rank   = 16

        subset = Subset(dataset, indices)
        self.dataloader = DataLoader(
            subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            drop_last=True
        )

    def _extract_modality(self, sensor):
        acc  = sensor[:, :, 0:3]   # (batch, 128, 3)
        gyro = sensor[:, :, 3:6]   # (batch, 128, 3)
        return acc, gyro

    def _get_input(self, sensor):
        acc, gyro = self._extract_modality(sensor)
        if self.modality == 'acc':
            return acc
        elif self.modality == 'gyro':
            return gyro
        else:  # both — use acc for stage 1 and 2, both for stage 3
            return acc

    def train_stage1(self, global_base_state, global_cls_state,
                     discriminator_state, num_epochs, lr):
        """
        Stage 1: Knowledge-disentangled pretraining.
        Train base encoder Eb to be modality-agnostic via adversarial loss.
        """
        model = FedHKDModel(self.embedding_dim, self.num_classes,
                            self.lora_rank).to(self.device)
        model.base_encoder.load_state_dict(global_base_state)
        model.classifier.load_state_dict(global_cls_state)

        discriminator = ModalityDiscriminator(self.embedding_dim).to(self.device)
        discriminator.load_state_dict(discriminator_state)

        # freeze LoRA in stage 1
        for p in model.lora_adapter.parameters():
            p.requires_grad = False

        optimizer_model = torch.optim.Adam(
            list(model.base_encoder.parameters()) +
            list(model.classifier.parameters()),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        optimizer_disc = torch.optim.Adam(
            discriminator.parameters(),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )

        criterion_cls  = nn.CrossEntropyLoss()
        criterion_disc = nn.CrossEntropyLoss()

        # modality label: acc=0, gyro=1, both treated as acc in stage 1
        modality_label = 0 if self.modality in ('acc', 'both') else 1

        model.train()
        discriminator.train()
        total_loss  = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                x = self._get_input(sensor)
                bsz = x.size(0)
                mod_labels = torch.full((bsz,), modality_label,
                                        dtype=torch.long).to(self.device)

                # ── train discriminator ──
                optimizer_disc.zero_grad()
                with torch.no_grad():
                    _, feat = model(x, use_lora=False)
                disc_out  = discriminator(feat.detach())
                loss_disc = criterion_disc(disc_out, mod_labels)
                loss_disc.backward()
                optimizer_disc.step()

                # ── train base encoder (adversarial: fool discriminator) ──
                optimizer_model.zero_grad()
                out, feat   = model(x, use_lora=False)
                loss_cls    = criterion_cls(out, labels)

                # adversarial loss: uniform distribution over modalities
                uniform = torch.full((bsz, 2), 1.0/2).to(self.device)
                disc_out_adv = discriminator(feat)
                loss_adv = criterion_disc(disc_out_adv,
                                          uniform.argmax(dim=1))

                loss = loss_cls + 0.1 * loss_adv
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer_model.step()

                total_loss  += loss_cls.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        n = len(self.dataloader.dataset)
        return (copy.deepcopy(model.base_encoder.state_dict()),
                copy.deepcopy(model.classifier.state_dict()),
                copy.deepcopy(discriminator.state_dict()),
                avg_loss, n)

    def train_stage2(self, global_base_state, global_lora_state,
                     global_cls_state, num_epochs, lr):
        """
        Stage 2: Modality-specific LoRA fine-tuning.
        Base encoder frozen. Only LoRA adapter trained.
        """
        model = FedHKDModel(self.embedding_dim, self.num_classes,
                            self.lora_rank).to(self.device)
        model.base_encoder.load_state_dict(global_base_state)
        model.lora_adapter.load_state_dict(global_lora_state)
        model.classifier.load_state_dict(global_cls_state)

        # freeze base encoder
        for p in model.base_encoder.parameters():
            p.requires_grad = False

        optimizer = torch.optim.Adam(
            list(model.lora_adapter.parameters()) +
            list(model.classifier.parameters()),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        model.train()
        total_loss  = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                x = self._get_input(sensor)

                optimizer.zero_grad()
                out, _ = model(x, use_lora=True)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        n = len(self.dataloader.dataset)
        return (copy.deepcopy(model.lora_adapter.state_dict()),
                copy.deepcopy(model.classifier.state_dict()),
                avg_loss, n)

    def train_stage3(self, global_base_state, global_lora_state,
                     fusion_state, num_epochs, lr):
        """
        Stage 3: Personalized fine-tuning with attentive fusion.
        Only for multi-modal clients. Base + LoRA frozen.
        Fine-tune fusion classifier only.
        """
        assert self.modality == 'both'

        base_enc = FedHKDBaseEncoder(self.embedding_dim).to(self.device)
        base_enc.load_state_dict(global_base_state)
        for p in base_enc.parameters():
            p.requires_grad = False

        lora = LoRAAdapter(self.embedding_dim, self.embedding_dim,
                           self.lora_rank).to(self.device)
        lora.load_state_dict(global_lora_state)
        for p in lora.parameters():
            p.requires_grad = False

        fusion = AttentiveFusion(self.embedding_dim, self.num_classes).to(self.device)
        fusion.load_state_dict(fusion_state)

        optimizer = torch.optim.Adam(
            fusion.parameters(),
            lr=lr, weight_decay=self.config['training']['weight_decay']
        )
        criterion = nn.CrossEntropyLoss()
        fusion.train()
        total_loss  = 0.0
        num_batches = 0

        for _ in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)
                acc, gyro = self._extract_modality(sensor)

                with torch.no_grad():
                    feat_acc  = base_enc(acc)
                    feat_acc  = feat_acc + lora(feat_acc)
                    feat_gyro = base_enc(gyro)
                    feat_gyro = feat_gyro + lora(feat_gyro)

                optimizer.zero_grad()
                out  = fusion(feat_acc, feat_gyro)
                loss = criterion(out, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fusion.parameters(), 1.0)
                optimizer.step()

                total_loss  += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        n = len(self.dataloader.dataset)
        return copy.deepcopy(fusion.state_dict()), avg_loss, n

# ── FedHKD Server ──────────────────────────────────────────────────────────
class FedHKDServer:
    def __init__(self, config, device='cpu'):
        self.config        = config
        self.device        = device
        self.embedding_dim = 128
        self.num_classes   = 6
        self.lora_rank     = 16

        # global models
        self.global_base   = FedHKDBaseEncoder(self.embedding_dim).to(device)
        self.global_lora   = LoRAAdapter(self.embedding_dim,
                                         self.embedding_dim,
                                         self.lora_rank).to(device)
        self.global_cls    = nn.Sequential(
            nn.Linear(self.embedding_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, self.num_classes)
        ).to(device)
        self.global_disc   = ModalityDiscriminator(self.embedding_dim).to(device)
        self.global_fusion = AttentiveFusion(self.embedding_dim,
                                             self.num_classes).to(device)

    def _weighted_avg(self, updates, key_fn, weight_fn):
        total = sum(weight_fn(u) for u in updates)
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

        updates      = []
        round_losses = []

        for client in selected:
            base_s  = copy.deepcopy(self.global_base.state_dict())
            cls_s   = copy.deepcopy(self.global_cls.state_dict())
            disc_s  = copy.deepcopy(self.global_disc.state_dict())

            base_u, cls_u, disc_u, loss, n = client.train_stage1(
                base_s, cls_s, disc_s, num_epochs, lr
            )
            updates.append((base_u, cls_u, disc_u, n))
            round_losses.append(loss)

        # aggregate base encoder, classifier, discriminator
        new_base = self._weighted_avg(updates,
                                      key_fn=lambda u: u[0],
                                      weight_fn=lambda u: u[3])
        new_cls  = self._weighted_avg(updates,
                                      key_fn=lambda u: u[1],
                                      weight_fn=lambda u: u[3])
        new_disc = self._weighted_avg(updates,
                                      key_fn=lambda u: u[2],
                                      weight_fn=lambda u: u[3])

        self.global_base.load_state_dict(new_base)
        self.global_cls.load_state_dict(new_cls)
        self.global_disc.load_state_dict(new_disc)

        return sum(round_losses) / len(round_losses)

    def run_stage2_round(self, clients, lr, num_epochs):
        per_round = self.config['federation']['clients_per_round']
        selected  = random.sample(clients, min(per_round, len(clients)))

        updates      = []
        round_losses = []

        for client in selected:
            base_s = copy.deepcopy(self.global_base.state_dict())
            lora_s = copy.deepcopy(self.global_lora.state_dict())
            cls_s  = copy.deepcopy(self.global_cls.state_dict())

            lora_u, cls_u, loss, n = client.train_stage2(
                base_s, lora_s, cls_s, num_epochs, lr
            )
            updates.append((lora_u, cls_u, n))
            round_losses.append(loss)

        # aggregate LoRA adapters and classifier
        new_lora = self._weighted_avg(updates,
                                      key_fn=lambda u: u[0],
                                      weight_fn=lambda u: u[2])
        new_cls  = self._weighted_avg(updates,
                                      key_fn=lambda u: u[1],
                                      weight_fn=lambda u: u[2])

        self.global_lora.load_state_dict(new_lora)
        self.global_cls.load_state_dict(new_cls)

        return sum(round_losses) / len(round_losses)

    def run_stage3_round(self, clients, lr, num_epochs):
        multi_clients = [c for c in clients if c.modality == 'both']
        if not multi_clients:
            return 0.0

        per_round = max(1, self.config['federation']['clients_per_round'] // 5)
        selected  = random.sample(multi_clients,
                                  min(per_round, len(multi_clients)))

        updates      = []
        round_losses = []

        for client in selected:
            base_s   = copy.deepcopy(self.global_base.state_dict())
            lora_s   = copy.deepcopy(self.global_lora.state_dict())
            fusion_s = copy.deepcopy(self.global_fusion.state_dict())

            fusion_u, loss, n = client.train_stage3(
                base_s, lora_s, fusion_s, num_epochs, lr
            )
            updates.append((fusion_u, n))
            round_losses.append(loss)

        new_fusion = self._weighted_avg(updates,
                                        key_fn=lambda u: u[0],
                                        weight_fn=lambda u: u[1])
        self.global_fusion.load_state_dict(new_fusion)

        return sum(round_losses) / len(round_losses)

# ── Evaluation ─────────────────────────────────────────────────────────────
def evaluate_fedhkd(server, test_dataset, device):
    """Evaluate using the fusion model for multi-modal inference."""
    base_enc = server.global_base
    lora     = server.global_lora
    fusion   = server.global_fusion

    base_enc.eval()
    lora.eval()
    fusion.eval()

    loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            sensor = batch['sensor'].to(device)
            labels = batch['label'].to(device)

            acc  = sensor[:, :, 0:3]
            gyro = sensor[:, :, 3:6]

            feat_acc  = base_enc(acc)
            feat_acc  = feat_acc + lora(feat_acc)
            feat_gyro = base_enc(gyro)
            feat_gyro = feat_gyro + lora(feat_gyro)

            out   = fusion(feat_acc, feat_gyro)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc_score  = 100.0 * (all_preds == all_labels).mean()
    f1         = 100.0 * f1_score(all_labels, all_preds, average='weighted')
    return round(acc_score, 2), round(f1, 2)

# ── Main ───────────────────────────────────────────────────────────────────
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

    # stage split: 60% stage1, 20% stage2, 20% stage3
    num_rounds    = config['federation']['num_rounds']
    stage1_rounds = int(num_rounds * 0.60)
    stage2_rounds = int(num_rounds * 0.20)
    stage3_rounds = num_rounds - stage1_rounds - stage2_rounds
    lr            = config['training']['learning_rate']
    local_epochs  = config['training']['local_epochs']
    log_every     = config['experiment']['log_every']

    print("=" * 60)
    print("FedHKD Baseline — Hierarchical Knowledge Disentanglement")
    print("=" * 60)
    print(f"Clients:       {config['federation']['num_clients']}")
    print(f"Total Rounds:  {num_rounds}")
    print(f"  Stage 1 (base encoder):  {stage1_rounds} rounds")
    print(f"  Stage 2 (LoRA finetune): {stage2_rounds} rounds")
    print(f"  Stage 3 (fusion):        {stage3_rounds} rounds")
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
        client = FedHKDClient(
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

    server   = FedHKDServer(config, device=device)
    best_acc = 0.0
    os.makedirs('results', exist_ok=True)

    # ── Stage 1: Knowledge-disentangled pretraining ───────────────────────
    print(f"\nStage 1: Knowledge-disentangled pretraining ({stage1_rounds} rounds)...")
    for r in range(1, stage1_rounds + 1):
        loss = server.run_stage1_round(all_clients, lr, local_epochs)
        if r % log_every == 0:
            acc_score, f1 = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage1 Round {r:>3} | Loss: {loss:.4f} | "
                  f"Accuracy: {acc_score}%  F1: {f1}%")

    # ── Stage 2: Modality-specific LoRA fine-tuning ───────────────────────
    print(f"\nStage 2: Modality-specific LoRA fine-tuning ({stage2_rounds} rounds)...")
    for r in range(1, stage2_rounds + 1):
        loss = server.run_stage2_round(all_clients, lr, local_epochs)
        if r % log_every == 0:
            acc_score, f1 = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage2 Round {r:>3} | Loss: {loss:.4f} | "
                  f"Accuracy: {acc_score}%  F1: {f1}%")

    # ── Stage 3: Personalized fusion fine-tuning ──────────────────────────
    print(f"\nStage 3: Personalized fusion fine-tuning ({stage3_rounds} rounds)...")
    for r in range(1, stage3_rounds + 1):
        loss = server.run_stage3_round(all_clients, lr, local_epochs)
        if r % log_every == 0:
            acc_score, f1 = evaluate_fedhkd(server, test_dataset, device)
            print(f"Stage3 Round {r:>3} | Loss: {loss:.4f} | "
                  f"Accuracy: {acc_score}%  F1: {f1}%")
            if acc_score > best_acc:
                best_acc = acc_score
                torch.save(
                    server.global_fusion.state_dict(),
                    'results/best_fedhkd_model.pt'
                )
                print(f"             New best: {best_acc}%")

    print("\n" + "=" * 60)
    print(f"FedHKD complete. Best accuracy: {best_acc}%")
    print("=" * 60)

if __name__ == '__main__':
    main()# paste the file contents here then Ctrl+D

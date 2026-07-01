import torch
import copy
import math
from torch.utils.data import DataLoader, Subset
from models.mosaic_model import MOSAICModel
from losses.total_loss import MOSAICLoss


class MOSAICClient:
    """
    One federated learning client with CrossModalAnchor loss.
    """

    def __init__(self, client_id, dataset, indices, config, device='cpu'):
        self.client_id   = client_id
        self.config      = config
        self.device      = device
        self.round_count = 0

        subset = Subset(dataset, indices)
        self.dataloader = DataLoader(
            subset,
            batch_size=config['training']['batch_size'],
            shuffle=True
        )

        self.loss_fn = MOSAICLoss(config)

    def _get_lr(self):
        base_lr      = self.config['training']['learning_rate']
        min_lr       = base_lr / 10.0
        decay_rounds = 500

        if self.round_count >= decay_rounds:
            return min_lr

        progress = self.round_count / decay_rounds
        cosine   = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    def train(self, global_state_dict, num_epochs):
        self.round_count += 1

        model = MOSAICModel(self.config).to(self.device)
        model.load_state_dict(global_state_dict)

        global_params = [p.data.clone() for p in model.parameters()]

        lr = self._get_lr()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )

        model.train()
        total_loss  = 0.0
        num_batches = 0

        for epoch in range(num_epochs):
            for batch in self.dataloader:
                sensor = batch['sensor'].to(self.device)
                labels = batch['label'].to(self.device)

                optimizer.zero_grad()

                logits, emb_a, emb_b = model(
                    {'sensor': sensor, 'label': labels}
                )

                losses = self.loss_fn(
                    logits, labels,
                    emb_a, emb_b,
                    list(model.parameters()),
                    global_params
                )

                losses['total'].backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )

                optimizer.step()

                total_loss  += losses['task']
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return model.state_dict(), avg_loss

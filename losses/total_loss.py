import torch
import torch.nn as nn
from losses.proximal import proximal_loss
from losses.contrastive import CrossModalAnchorLoss


class MOSAICLoss(nn.Module):
    """
    Combined local training objective.
    L_total = L_task + mu_prox * L_proximal + lambda_anchor * L_anchor
    """

    def __init__(self, config):
        super().__init__()
        self.mu_prox       = config['losses']['mu_prox']
        self.lambda_anchor = config['losses']['lambda_anchor']
        self.task_loss     = nn.CrossEntropyLoss()
        self.anchor_loss   = CrossModalAnchorLoss(
            temperature=config['losses']['tau_contrastive']
        )

    def forward(self, logits, labels, emb_a, emb_b,
                local_params, global_params):

        l_task   = self.task_loss(logits, labels)
        l_prox   = proximal_loss(local_params, global_params, self.mu_prox)
        l_anchor = self.anchor_loss(emb_a, emb_b)

        total = l_task + l_prox + self.lambda_anchor * l_anchor

        return {
            'total':    total,
            'task':     l_task.item(),
            'proximal': l_prox.item(),
            'anchor':   l_anchor.item()
        }

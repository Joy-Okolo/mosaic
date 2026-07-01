import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAnchorLoss(nn.Module):
    """
    Within-client cross-modal contrastive alignment.

    Pulls embeddings from different modalities (accelerometer
    and gyroscope) closer when they describe the same activity,
    and pushes them apart when they describe different activities.

    Uses InfoNCE loss — same sample across modalities = positive pair,
    different samples = negative pairs.
    """

    def __init__(self, temperature=0.3):
        super().__init__()
        self.temperature = temperature

    def forward(self, emb_a, emb_b):
        """
        Args:
            emb_a: embeddings from modality A (B, D)
            emb_b: embeddings from modality B (B, D)
        Returns:
            scalar contrastive loss
        """
        # Normalize embeddings to unit sphere
        emb_a = F.normalize(emb_a, dim=-1)   # (B, D)
        emb_b = F.normalize(emb_b, dim=-1)   # (B, D)

        # Similarity matrix: (B, B)
        sim_matrix = torch.matmul(emb_a, emb_b.T) / self.temperature

        # Diagonal = positive pairs (same sample, different modality)
        B      = emb_a.shape[0]
        labels = torch.arange(B, device=emb_a.device)

        # InfoNCE in both directions
        loss_a_to_b = F.cross_entropy(sim_matrix,   labels)
        loss_b_to_a = F.cross_entropy(sim_matrix.T, labels)

        return (loss_a_to_b + loss_b_to_a) / 2

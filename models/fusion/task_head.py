import torch.nn as nn


class TaskHead(nn.Module):
    """
    Classification head.
    Takes fused embedding and predicts activity class.
    Input:  (B, input_dim)
    Output: (B, num_classes)
    """

    def __init__(self, config):
        super().__init__()
        input_dim   = config['encoders']['sensor']['embedding_dim']
        num_classes = config['task']['num_classes']
        hidden_dim  = config['task']['head_hidden_dim']
        dropout     = config['training'].get('dropout', 0.2)

        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

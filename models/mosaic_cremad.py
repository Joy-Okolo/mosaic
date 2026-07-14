import torch
import torch.nn as nn


class AudioEncoder(nn.Module):
    """
    Audio encoder for MFCC features.
    Input: (batch, 265, 80) — time frames x MFCC coefficients
    Output: (batch, embedding_dim)
    Uses 1D CNN + BiLSTM matching MOSAIC's sensor encoder design.
    """
    def __init__(self, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(80, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=embedding_dim // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        # x: (batch, 265, 80)
        x = x.permute(0, 2, 1)          # (batch, 80, 265) for Conv1d
        x = self.cnn(x)                  # (batch, 256, 265)
        x = x.permute(0, 2, 1)          # (batch, 265, 256) for LSTM
        x, _ = self.lstm(x)             # (batch, 265, embedding_dim)
        x = x[:, -1, :]                 # last timestep (batch, embedding_dim)
        x = self.dropout(x)
        return x


class VideoEncoder(nn.Module):
    """
    Video encoder for MobileNetV2 frame features.
    Input: (batch, 3, 1280) — frames x MobileNetV2 features
    Output: (batch, embedding_dim)
    Uses temporal attention over frame features.
    """
    def __init__(self, embedding_dim=128, dropout=0.2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # project MobileNetV2 features to embedding dim
        self.frame_proj = nn.Sequential(
            nn.Linear(1280, 512),
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
        )

        # temporal attention over frames
        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, 3, 1280)
        x = self.frame_proj(x)           # (batch, 3, embedding_dim)
        attn = self.attention(x)         # (batch, 3, 1)
        attn = torch.softmax(attn, dim=1)
        x = (x * attn).sum(dim=1)       # (batch, embedding_dim)
        x = self.dropout(x)
        return x


class MOSAICCREMADModel(nn.Module):
    """
    MOSAIC model for CREMA-D emotion recognition.
    Branch A: audio (MFCC features) — shape (265, 80)
    Branch B: video (MobileNetV2 features) — shape (3, 1280)

    Mirrors MOSAICModel's architecture but with modality-specific encoders.
    """
    def __init__(self, config):
        super().__init__()
        embedding_dim = config.get('cremad', {}).get('embedding_dim', 128)
        dropout       = config.get('cremad', {}).get('dropout', 0.2)
        num_classes   = config.get('cremad', {}).get('num_classes', 4)

        self.encoder_a = AudioEncoder(embedding_dim, dropout)   # audio
        self.encoder_b = VideoEncoder(embedding_dim, dropout)   # video

        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        self.task_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim // 2, num_classes)
        )

    def forward(self, batch):
        audio = batch['audio']           # (B, 265, 80)
        video = batch['video']           # (B, 3, 1280)

        emb_a = self.encoder_a(audio)    # (B, embedding_dim)
        emb_b = self.encoder_b(video)    # (B, embedding_dim)

        fused  = self.fusion(torch.cat([emb_a, emb_b], dim=1))
        logits = self.task_head(fused)

        return logits, emb_a, emb_b


if __name__ == '__main__':
    # Quick test
    config = {
        'cremad': {
            'embedding_dim': 128,
            'dropout': 0.2,
            'num_classes': 4
        }
    }

    model = MOSAICCREMADModel(config)
    print("MOSAICCREMADModel architecture:")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # test forward pass
    batch = {
        'audio': torch.randn(8, 265, 80),
        'video': torch.randn(8, 3, 1280),
        'label': torch.randint(0, 4, (8,))
    }

    logits, emb_a, emb_b = model(batch)
    print(f"Logits shape: {logits.shape}")
    print(f"Audio embedding shape: {emb_a.shape}")
    print(f"Video embedding shape: {emb_b.shape}")
    print("Forward pass OK")

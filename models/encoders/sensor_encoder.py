import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Learns which time steps are most important."""

    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        # x: (B, C, T)
        weights = self.attention(x)   # (B, 1, T)
        return x * weights            # (B, C, T)


class SensorEncoder(nn.Module):
    """
    Encodes time-series sensor data into a fixed-size embedding.
    Input:  (B, 128, 6)  - batch of sensor sequences
    Output: (B, 128)     - embedding vectors
    """

    def __init__(self, config):
        super().__init__()
        cfg = config['encoders']['sensor']

        in_ch   = cfg['input_features']   # 6
        channels = cfg['cnn_channels']     # [64, 128, 256]
        kernel   = cfg['cnn_kernel']       # 3
        dropout  = cfg['dropout']          # 0.2

        # Three 1D CNN layers
        self.cnn = nn.Sequential(
            # Layer 1: no stride
            nn.Conv1d(in_ch, channels[0], kernel, padding=kernel//2),
            nn.ReLU(),
            nn.GroupNorm(8, channels[0]),
            nn.Dropout(dropout),

            # Layer 2: stride 2 to reduce sequence length
            nn.Conv1d(channels[0], channels[1], kernel, stride=2, padding=kernel//2),
            nn.ReLU(),
            nn.GroupNorm(8, channels[1]),
            nn.Dropout(dropout),

            # Layer 3: stride 2 again
            nn.Conv1d(channels[1], channels[2], kernel, stride=2, padding=kernel//2),
            nn.ReLU(),
            nn.GroupNorm(8, channels[2]),
            nn.Dropout(dropout),
        )

        # Temporal attention over CNN output
        self.attention = TemporalAttention(channels[2])

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=channels[2],
            hidden_size=cfg['lstm_hidden'],
            num_layers=cfg['lstm_layers'],
            batch_first=True,
            bidirectional=cfg['bidirectional'],
            dropout=dropout if cfg['lstm_layers'] > 1 else 0
        )

        lstm_out_dim = cfg['lstm_hidden'] * (2 if cfg['bidirectional'] else 1)

        # Final projection to embedding_dim
        self.projector = nn.Sequential(
            nn.Linear(lstm_out_dim, cfg['embedding_dim']),
            nn.LayerNorm(cfg['embedding_dim']),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.embedding_dim = cfg['embedding_dim']

    def forward(self, x):
        # x: (B, T, F) = (B, 128, 6)
        x = x.permute(0, 2, 1)          # (B, 6, 128) for Conv1d
        x = self.cnn(x)                  # (B, 256, 32)
        x = self.attention(x)            # (B, 256, 32)
        x = x.permute(0, 2, 1)          # (B, 32, 256) for LSTM
        _, (h_n, _) = self.lstm(x)

        # Concatenate final hidden states from both directions
        final = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (B, 256)

        embedding = self.projector(final)  # (B, 128)
        return embedding

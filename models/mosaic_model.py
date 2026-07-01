import torch
import torch.nn as nn
from models.encoders.sensor_encoder import SensorEncoder
from models.fusion.task_head import TaskHead


class MOSAICModel(nn.Module):
    """
    MOSAIC client model with two sensor branches.
    Branch A: accelerometer (channels 0,1,2) - 3 features
    Branch B: gyroscope     (channels 3,4,5) - 3 features
    """

    def __init__(self, config):
        super().__init__()

        # Create a modified config for 3-channel encoders
        config_3ch = {
            'encoders': {
                'sensor': dict(config['encoders']['sensor'])
            }
        }
        config_3ch['encoders']['sensor']['input_features'] = 3

        self.encoder_a = SensorEncoder(config_3ch)  # accelerometer
        self.encoder_b = SensorEncoder(config_3ch)  # gyroscope

        embed_dim = config['encoders']['sensor']['embedding_dim']
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.task_head = TaskHead(config)

    def forward(self, batch):
        x = batch['sensor']        # (B, 128, 6)

        acc  = x[:, :, :3]        # (B, 128, 3)
        gyro = x[:, :, 3:]        # (B, 128, 3)

        emb_a = self.encoder_a(acc)   # (B, 128)
        emb_b = self.encoder_b(gyro)  # (B, 128)

        fused  = self.fusion(torch.cat([emb_a, emb_b], dim=1))
        logits = self.task_head(fused)

        return logits, emb_a, emb_b

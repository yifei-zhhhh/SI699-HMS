"""
This file contains the implementation of the multimodal model that combines the EEGWaveNet 
and EEGSpecNet models.
"""

import torch
from torch import nn
from .spec_model import EEGSpecNet
from .wave_model import EEGWaveNet


class EEGMultiModel(nn.Module):
    """EEG multimodal model that combines the EEGWaveNet and EEGSpecNet models."""

    def __init__(self, exp_config, num_classes=6):
        super().__init__()
        self.eeg_spec_net = EEGSpecNet(exp_config, num_classes)
        self.eeg_wavenet = EEGWaveNet(exp_config, num_classes=num_classes)
        self.prediction = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, eeg, spec):
        """Forward pass of the EEG multimodal model."""
        eeg_feats, eeg_logits = self.eeg_wavenet(eeg)
        spec_feats, spec_logits = self.eeg_spec_net(spec)
        feats = torch.cat([eeg_feats, spec_feats], dim=1)
        logits = self.prediction(feats)
        return logits, eeg_feats, spec_feats, eeg_logits, spec_logits

    @property
    def linear_layers(self):
        """Get the linear layers of the EEG multimodal model."""
        return [
            *self.eeg_spec_net.linear_layers,
            *self.eeg_wavenet.linear_layers,
            self.prediction[2],
        ]

"""
This module defines the EEGSpecNet model which uses a backbone model from timm
to extract features from the input spectrogram data and then uses a classifier
"""

import timm
import torch
from torch import nn


class EEGSpecNet(nn.Module):
    """This class defines the EEGSpecNet model which uses a backbone model from timm
    to extract features from the input spectrogram data and then uses a classifier
    """

    def __init__(self, exp_config, num_classes: int = 6):
        """Initialize the EEGSpecNet model

        Args:
            backbone (str): The name of the backbone model from timm
            in_channels (int): The number of input channels
            pretrained (bool): Whether to use the pretrained weights of the backbone model
            num_classes (int): The number of output classes
            is_vit (bool): Whether the backbone model is a vision transformer
        """
        super().__init__()
        # Basic parameters
        backbone = exp_config.backbone_2d
        in_channels = exp_config.in_channels_2d
        pretrained = exp_config.pretrained
        is_vit = exp_config.is_vit
        # When the number of input channels is not 1,
        # it will repeat the conv1_weight as many times as required
        # and then select the required number of input channels weights.
        # see https://timm.fast.ai/models for reference
        # based on this, we may let the first layer and the fourth layer could be totally different
        # self.in_channels = lcm(in_channels, 3)
        # self.num_repeats = self.in_channels // in_channels
        self.in_channels = in_channels
        self.is_vit = is_vit
        # Load the backbone model
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=self.in_channels,
            drop_rate=0.3,
            drop_path_rate=0.2,
        )
        # Remove the last layer of the backbone model
        self.features = nn.Sequential(*list(self.backbone.children())[:-1])
        with torch.no_grad():
            dummy_input = torch.randn(2, self.in_channels, 224, 224)
            if self.in_channels != 3:
                dummy_input = torch.randn(2, self.in_channels, 128, 256)
        if is_vit:
            # add a pooling layer the last second dimension to 1
            output_shape = 512
            with torch.no_grad():
                output_shape = self.features(dummy_input).shape[-1]
            self.features.add_module("global_pool", nn.Linear(output_shape, 24))
            # Flatten the output
            self.features.add_module("flatten", nn.Flatten())
        # Get the output shape of the backbone model
        backbone_output_shape = 512
        with torch.no_grad():
            backbone_output_shape = self.features(dummy_input).shape[-1]
            print(backbone_output_shape)
        # Add a classifier layer which is a two-layer MLP
        self.fc = nn.Linear(backbone_output_shape, 128)
        self.prediction = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    @property
    def linear_layers(self):
        """Get the linear layers of the EEGSpecNet model

        Returns:
            nn.ModuleList: The linear layers of the EEGSpecNet model
        """
        layers = [self.fc, self.prediction[2]]
        if self.is_vit:
            layers.append(self.features.global_pool)
        return layers

    def forward(self, x):
        """Forward pass of the EEGSpecNet model

        Args:
            x (Tensor): The input tensor

        Returns:
            Tensor: The features extracted by the backbone model
            Tensor: The output logits
        """
        # x = self._concatenate_channels(x)
        feats = self.fc(self.features(x))
        logits = self.prediction(feats)
        return feats, logits

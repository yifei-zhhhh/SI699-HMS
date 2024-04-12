"""
This module contains the implementation of the EEGWaveNet model. We got the inspiration 
from the following notebook: 
https://www.kaggle.com/code/nischaydnk/lightning-1d-eegnet-training-pipeline-hbs
"""

import torch
from torch import nn


class ResNet_1D_Block(nn.Module):
    """ResNet 1D block with downsampling."""

    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, downsampling
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.1, inplace=False),
            nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.downsampling = downsampling

    def forward(self, x):
        """Forward pass of the ResNet 1D block."""
        identity = x
        out = self.features(x)
        out = self.maxpool(out)
        identity = self.downsampling(x)
        out += identity
        return out


class EEGWaveNet(nn.Module):
    """EEGWaveNet model."""

    def __init__(self, exp_config, num_classes=6):
        super().__init__()
        kernels = exp_config.kernels
        in_channels = exp_config.num_channels
        fixed_kernel_size = exp_config.fixed_kernel_size
        self.kernels = kernels
        self.num_classes = num_classes
        self.planes = 24
        self.parallel_conv = nn.ModuleList()
        self.in_channels = in_channels
        for _, kernel_size in enumerate(list(self.kernels)):
            sep_conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=self.planes,
                kernel_size=(kernel_size),
                stride=1,
                padding=0,
                bias=False,
            )
            self.parallel_conv.append(sep_conv)

        self.projection = nn.Sequential(
            nn.BatchNorm1d(num_features=self.planes),
            nn.ReLU(inplace=False),
            nn.Conv1d(
                in_channels=self.planes,
                out_channels=self.planes,
                kernel_size=fixed_kernel_size,
                stride=2,
                padding=2,
                bias=False,
            ),
            self._make_resnet_layer(
                kernel_size=fixed_kernel_size, stride=1, padding=fixed_kernel_size // 2
            ),
            nn.BatchNorm1d(num_features=self.planes),
            nn.AvgPool1d(kernel_size=4, stride=4, padding=2),
        )

        self.rnn = nn.GRU(
            input_size=self.in_channels,
            hidden_size=128,
            num_layers=1,
            bidirectional=True,
        )
        self.fc = nn.Linear(in_features=736, out_features=128)
        self.prediction = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def _make_resnet_layer(self, kernel_size, stride, blocks=8, padding=0):
        layers = []
        for _ in range(blocks):
            downsampling = nn.Sequential(
                nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
            )
            layers.append(
                ResNet_1D_Block(
                    in_channels=self.planes,
                    out_channels=self.planes,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    downsampling=downsampling,
                )
            )

        return nn.Sequential(*layers)

    @property
    def linear_layers(self):
        """Get the linear layers of the EEGWaveNet model."""
        return [self.fc, self.prediction[2]]

    def forward(self, x):
        """Forward pass of the EEGWaveNet model."""
        out_sep = []
        for i in range(len(self.kernels)):
            sep = self.parallel_conv[i](x)
            out_sep.append(sep)

        out = torch.cat(out_sep, dim=2)
        out = self.projection(out)
        out = out.reshape(out.shape[0], -1)

        rnn_out, _ = self.rnn(x.permute(0, 2, 1))
        new_rnn_h = rnn_out[:, -1, :]

        new_out = torch.cat([out, new_rnn_h], dim=1)
        feats = self.fc(new_out)
        logits = self.prediction(feats)
        return feats, logits

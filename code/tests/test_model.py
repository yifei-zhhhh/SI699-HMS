"""
Test the models in the models directory
"""

import unittest
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config
from models import EEGModelBase, EEGModel, EEGMultiModel, EEGWaveNet, EEGSpecNet


class TestModels(unittest.TestCase):
    """Unit tests for the models in the models directory."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = Config
        self.config.in_channels_2d = 4

    def test_eeg_wave_net_forward(self):
        """Test the forward method of the EEGWaveNet model."""
        x = torch.randn(32, 8, 10000)
        model = EEGWaveNet(self.config, num_classes=6)
        feats, logits = model(x)
        self.assertEqual(feats.shape[0], 32)
        self.assertEqual(logits.shape[0], 32)
        self.assertEqual(logits.shape[1], 6)

    def test_eeg_spec_net_forward(self):
        """Test the forward method of the EEGSpecNet model."""
        model = EEGSpecNet(self.config, num_classes=6)
        x = torch.randn(32, 4, 128, 256)
        feats, logits = model(x)
        self.assertEqual(feats.shape[0], 32)
        self.assertEqual(logits.shape[0], 32)
        self.assertEqual(logits.shape[1], 6)

    def test_eeg_spec_net_forwardv2(self):
        """Test the forward method of the EEGSpecNet model."""
        new_config = self.config
        new_config.backbone_2d = "vit_base_patch16_224"
        new_config.in_channels_2d = 3
        new_config.is_vit = True
        model = EEGSpecNet(new_config, num_classes=6)
        x = torch.randn(32, 3, 224, 224)
        feats, logits = model(x)
        self.assertEqual(feats.shape[0], 32)
        self.assertEqual(logits.shape[0], 32)
        self.assertEqual(logits.shape[1], 6)

    def test_eeg_multimodel_forward(self):
        """Test the forward method of the EEGMultiModel model."""
        eeg = torch.randn(2, 8, 10000)
        spec = torch.randn(2, 4, 128, 256)
        eeg_mega_model = EEGMultiModel(self.config, num_classes=6)
        output = eeg_mega_model(eeg, spec)
        # Add assertions for the output as needed
        self.assertEqual(output[0].shape[0], 2)

    def test_eeg_base_with_wavenet_training_step(self):
        """Test the training_step method of the EEGModelBase model with EEGWaveNet."""
        fold_id = 0
        eeg = torch.randn(2, 8, 10000)
        preds = torch.randn(2, 6)
        preds = torch.nn.functional.softmax(preds, dim=1)
        batch = (eeg, preds)
        model = EEGModelBase(fold_id, self.config, EEGWaveNet)
        output = model.training_step(batch, 0)
        self.assertIsInstance(output, torch.Tensor)

    def test_eeg_base_with_specnet_training_step(self):
        """Test the training_step method of the EEGModelBase model with EEGSpecNet."""
        fold_id = 0
        spec = torch.randn(2, 4, 128, 256)
        preds = torch.randn(2, 6)
        preds = torch.nn.functional.softmax(preds, dim=1)
        batch = (spec, preds)
        model = EEGModelBase(fold_id, self.config, EEGSpecNet)
        output = model.training_step(batch, 0)
        self.assertIsInstance(output, torch.Tensor)

    def test_eeg_model_training_step(self):
        """Test the training_step method of the EEGModel model."""
        fold_id = 0
        eeg = torch.randn(2, 8, 10000)
        spec = torch.randn(2, 4, 128, 256)
        preds = torch.randn(2, 6)
        preds = torch.nn.functional.softmax(preds, dim=1)
        batch = (eeg, spec, preds)
        model = EEGModel(fold_id, self.config, EEGMultiModel)
        output = model.training_step(batch, 0)
        self.assertIsInstance(output, torch.Tensor)


if __name__ == "__main__":
    unittest.main()

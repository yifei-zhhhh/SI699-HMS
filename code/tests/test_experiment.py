"""
Test the Experiment class
"""

import os
import sys
import unittest
from copy import deepcopy
import torch
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config
from exp import Experiment
from eeg_dataset import (
    get_fold_dls,
    EEGDataset,
    EEGSpecDataset,
    EEGWaveDataset,
    EEGDatasetV2,
)

torch.set_float32_matmul_precision("high")


class TestExperiment(unittest.TestCase):
    """Unit tests for the Experiment class."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        n_repeats = 2
        n_samples = n_repeats * 5
        eeg = np.random.randn(n_samples, 10000, 8).astype(np.float32)
        eegs = {i: eeg[i] for i in range(n_samples)}
        spec = np.random.randn(n_samples, 128, 256, 4).astype(np.float32)
        specs = {i: spec[i] for i in range(n_samples)}
        df = self._prepare_data_helper(n_samples, n_repeats)
        self.kfold_data = {"all": [], "spec": [], "wave": []}
        for fold_id in range(5):
            df_train = df[df.fold != fold_id]
            df_valid = df[df.fold == fold_id]
            self.kfold_data["all"].append(
                # get_fold_dls(df_train, df_valid, eegs, specs, EEGDataset)
                get_fold_dls(df_train, df_valid, eegs, specs, EEGDatasetV2)
            )
            self.kfold_data["spec"].append(
                get_fold_dls(df_train, df_valid, eegs, specs, EEGSpecDataset)
            )
            self.kfold_data["wave"].append(
                get_fold_dls(df_train, df_valid, eegs, specs, EEGWaveDataset)
            )
        self.train = df
        self.config = Config
        self.config.epochs = 1
        self.config.batch_size = 10
        self.config.trn_folds = [0]

    def _prepare_data_helper(self, n_samples, n_repeats):
        preds = np.random.randn(n_samples, 6)
        preds = np.exp(preds) / np.exp(preds).sum(axis=1, keepdims=True)
        columns = [
            "eeg_id",
            "spectogram_id",
            "min",
            "max",
            "patient_id",
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
            "target",
            "fold",
        ]
        df = pd.DataFrame(np.zeros((n_samples, 13)), columns=columns, dtype=int)
        df["eeg_id"] = list(range(n_samples))
        df["spectogram_id"] = list(range(n_samples))
        pred_columns = [
            "seizure_vote",
            "lpd_vote",
            "gpd_vote",
            "lrda_vote",
            "grda_vote",
            "other_vote",
        ]
        df[pred_columns] = preds
        df["target"] = preds.argmax(axis=1)
        df["patient_id"] = list(range(n_samples))
        df["fold"] = np.arange(5).repeat(n_repeats)
        df["min"] = df[pred_columns].min(axis=1)
        df["max"] = df[pred_columns].max(axis=1)
        return df

    def test_kfold_train_wave_net(self):
        """Test the kfold_train method of the EEGWaveNet."""
        exp = Experiment(self.kfold_data["wave"], self.train, "EEGWaveNet")
        exp.kfold_train(self.config)
        self.assertTrue(os.path.exists(f"{Config.output_dir}/oof.csv"))
        os.remove(f"{Config.output_dir}/oof.csv")

    def test_kfold_train_spec_net(self):
        """Test the kfold_train method of the EEGSpecNet."""
        exp = Experiment(self.kfold_data["spec"], self.train, "EEGSpecNet")
        self.config.in_channels_2d = 12
        exp.kfold_train(self.config)
        self.assertTrue(os.path.exists(f"{Config.output_dir}/oof.csv"))
        os.remove(f"{Config.output_dir}/oof.csv")

    def test_kfold_train_eeg_model(self):
        """Test the kfold_train method of the EEGModel."""
        exp = Experiment(self.kfold_data["all"], self.train, "EEGModel")
        self.config.in_channels_2d = 3
        exp.kfold_train(self.config)
        self.assertTrue(os.path.exists(f"{Config.output_dir}/oof.csv"))
        os.remove(f"{Config.output_dir}/oof.csv")


if __name__ == "__main__":
    unittest.main()

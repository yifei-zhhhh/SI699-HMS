"""
This file is used to check the forward path of the model and find potential bugs.
"""

import argparse
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import torch
import pytorch_lightning as pl
from utils.config import Config
from utils.loss import KLDivLossWithLogits, CrossEntropyLossWithLogits
from eeg_dataset import (
    EEGDataset,
    EEGDatasetV2,
    EEGSpecDataset,
    EEGSpecDatasetV2,
    EEGWaveDataset,
    get_fold_dls,
)
from exp import Experiment

Config.in_channels_2d = 3
Config.backbone_2d = "vit_tiny_patch16_224"
Config.is_vit = True


def seed_everything(seed):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True, warn_only=True)
    pl.seed_everything(seed, workers=True)


def parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()

    # model
    parser.add_argument(
        "--model",
        type=str,
        default="EEGModel",
        help="Model Used",
        choices=["EEGModel", "EEGWaveNet", "EEGSpecNet"],
    )

    # loss function
    parser.add_argument(
        "--loss",
        type=str,
        default="CrossEntropy",
        help="Loss Function Used",
        choices=["KLDiv", "CrossEntropy"],
    )

    # seed
    parser.add_argument("--seed", type=int, default=42, help="seed")

    # optimization
    parser.add_argument("--itr", type=int, default=1, help="experiments times")

    # epochs
    parser.add_argument("--epochs", type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    """The main function to run the experiment."""
    # Load the data
    df = pd.read_csv(f"{Config.data_root}train_300_patients.csv")
    TARGETS = Config.TARGETS

    train_df = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_id": "first", "spectrogram_label_offset_seconds": "min"})
    train_df.columns = ["spectogram_id", "min"]

    aux = df.groupby("eeg_id")[
        ["spectrogram_id", "spectrogram_label_offset_seconds"]
    ].agg({"spectrogram_label_offset_seconds": "max"})
    train_df["max"] = aux

    aux = df.groupby("eeg_id")[["patient_id"]].agg("first")
    train_df["patient_id"] = aux

    aux = df.groupby("eeg_id")[TARGETS].agg("sum")
    for label in TARGETS:
        train_df[label] = aux[label].values

    y_data = train_df[TARGETS].values
    y_data = y_data / y_data.sum(axis=1, keepdims=True)
    train_df[TARGETS] = y_data
    aux = df.groupby("eeg_id")[["expert_consensus"]].agg("first")
    train_df["target"] = aux
    train = train_df.reset_index()
    all_spectrograms = np.load(Config.PRE_LOADED_SPECTOGRAMS, allow_pickle=True).item()
    all_eegs = np.load(Config.PRE_LOADED_EEGS, allow_pickle=True).item()

    # KFold split
    gkf = GroupKFold(n_splits=5)
    train["fold"] = 0
    for fold, (_, val_idx) in enumerate(
        gkf.split(train, train.target, train.patient_id)
    ):
        train.loc[val_idx, "fold"] = fold

    kfold_data = {"all": [], "spec": [], "wave": []}
    for fold_id in range(5):
        df_train = train[train.fold != fold_id]
        df_valid = train[train.fold == fold_id]
        kfold_data["all"].append(
            get_fold_dls(df_train, df_valid, all_eegs, all_spectrograms, EEGDatasetV2)
        )
        kfold_data["spec"].append(
            get_fold_dls(
                df_train, df_valid, all_eegs, all_spectrograms, EEGSpecDatasetV2
            )
        )
        kfold_data["wave"].append(
            get_fold_dls(df_train, df_valid, all_eegs, all_spectrograms, EEGWaveDataset)
        )

    loss_cls = None
    if args.loss == "CrossEntropy":
        loss_cls = CrossEntropyLossWithLogits
    else:
        loss_cls = KLDivLossWithLogits

    # run task
    if args.model == "EEGModel":
        exp = Experiment(kfold_data["all"], train, args.model, loss_cls)
    elif args.model == "EEGWaveNet":
        exp = Experiment(kfold_data["wave"], train, args.model, loss_cls)
    elif args.model == "EEGSpecNet":
        exp = Experiment(kfold_data["spec"], train, args.model, loss_cls)

    config = Config
    if args.epochs != 0:
        config.epochs = args.epochs

    print(f">>>>>>>> start training iteration: {args.model} >>>>>>>>")
    exp.kfold_train(config)


if __name__ == "__main__":
    args = parse_args()
    main(args)

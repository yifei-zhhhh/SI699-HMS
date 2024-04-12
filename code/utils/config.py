"""
Configuration file for the model training.
"""

from numpy import lcm


class Config:
    """Default configuration for the model training."""

    use_aug = False
    num_classes = 6
    batch_size = 32
    epochs = 10
    PRECISION = 16
    PATIENCE = 20
    seed = 42
    backbone_2d = "tf_efficientnet_b0"
    pretrained = True
    weight_decay = 1e-2
    use_mixup = False
    mixup_alpha = 0.1
    num_channels = 8
    LR = 7e-4
    trn_folds = [0, 1, 2, 3, 4]
    processed_train = None
    data_root = "data/"
    PRE_LOADED_EEGS = "data/eegs.npy"
    PRE_LOADED_SPECTOGRAMS = "data/eeg_specs.npy"
    output_dir = "output"
    kernels = [5, 7, 9, 11]
    in_channels_2d = lcm(4, 3)
    fixed_kernel_size = 5
    is_vit = False
    TARGETS = [
        "seizure_vote",
        "lpd_vote",
        "gpd_vote",
        "lrda_vote",
        "grda_vote",
        "other_vote",
    ]

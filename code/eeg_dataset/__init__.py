"""
This module is used to load the EEG dataset.
"""

from .data_loader import (
    EEGDataset,
    EEGSpecDataset,
    EEGWaveDataset,
    EEGDatasetV2,
    EEGSpecDatasetV2,
)
from .data_utils import get_fold_dls


class DatasetFactory:
    """Enum class to select the dataset type."""

    EEGModel = EEGDataset
    EEGWaveNet = EEGWaveDataset
    EEGSpecNet = EEGSpecDataset


__all__ = [
    "DatasetFactory",
    "get_fold_dls",
    "EEGDataset",
    "EEGSpecDataset",
    "EEGWaveDataset",
    "EEGDatasetV2",
    "EEGSpecDatasetV2",
]

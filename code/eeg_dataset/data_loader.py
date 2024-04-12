"""
Data loader for EEG data. This module contains the EEGDataset class which is used to load EEG data
for training and validation. The EEGDataset class is a subclass of torch.utils.data.Dataset.
We refer to this Chris Deotte's notebook for the implementation of the EEGDataset class:
https://www.kaggle.com/code/cdeotte/efficientnetb0-starter-lb-0-43?scriptVersionId=159911317
as the reference implementation and modify it to suit our needs.
"""

import torch
import numpy as np
from cv2 import resize
from utils.config import Config
from .data_utils import butter_lowpass_filter, quantize_data

TARGETS = Config.TARGETS


class EEGDataset(torch.utils.data.Dataset):
    """EEG dataset class for loading EEG data."""

    def __init__(self, data, eegs=None, eeg_specs=None, test=False):
        self.data = data
        self.eegs = eegs
        self.eeg_specs = eeg_specs  # EEG spectrograms for each ID
        self.test = test

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        samples, spec = None, None
        # Processing EEG signal data
        eeg_data = self.eegs[row.eeg_id]
        sample = self.process_sample(eeg_data)
        samples = torch.from_numpy(sample).float()
        samples = samples.permute(1, 0)
        # Processing spectrogram data
        spec = self.spec_data_generation(row)
        if self.test:
            return samples, spec
        label = row[
            TARGETS
        ]  # Assuming 'TARGETS' is defined somewhere as the label column name
        label = torch.tensor(label).float()
        return samples, spec, label

    def process_sample(self, eeg_data):
        """Process EEG signal data."""
        sample = np.zeros(
            (eeg_data.shape[0], 8)
        )  # Assuming eeg_data.shape[1] == 8 for 8 channels
        FEAT2IDX = {
            "Fp1": 0,
            "T3": 1,
            "C3": 2,
            "O1": 3,
            "Fp2": 4,
            "C4": 5,
            "T4": 6,
            "O2": 7,
        }
        # Compute differences
        for i, (start, end) in enumerate(
            [
                ("Fp1", "T3"),
                ("T3", "O1"),
                ("Fp1", "C3"),
                ("C3", "O1"),
                ("Fp2", "C4"),
                ("C4", "O2"),
                ("Fp2", "T4"),
                ("T4", "O2"),
            ]
        ):
            sample[:, i] = eeg_data[:, FEAT2IDX[start]] - eeg_data[:, FEAT2IDX[end]]
        # Normalize the sample data
        sample = (sample - np.mean(sample, axis=0)) / np.std(sample, axis=0)
        sample = np.clip(sample, -1024, 1024)
        sample = np.nan_to_num(sample, nan=0)
        sample = butter_lowpass_filter(sample)
        sample = quantize_data(sample, 1)
        return sample

    def spec_data_generation(self, row):
        """
        Generates data containing batch_size samples. This method directly
        uses class attributes for spectrograms and EEG spectrograms.
        """
        img = self.eeg_specs[row.eeg_id].transpose(2, 0, 1)
        n_repeats = Config.in_channels_2d // img.shape[0]
        img = np.repeat(img, n_repeats, axis=0)
        return img


class EEGSpecDataset(EEGDataset):
    """EEG spectrogram dataset class for loading"""

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # Processing spectrogram data
        spec = self.spec_data_generation(row)
        if self.test:
            return spec
        label = row[TARGETS]
        label = torch.tensor(label).float()
        return spec, label


class EEGWaveDataset(EEGDataset):
    """EEG wave dataset class for loading"""

    def __getitem__(self, index):
        row = self.data.iloc[index]
        # Processing EEG signal data
        eeg_data = self.eegs[row.eeg_id]
        sample = self.process_sample(eeg_data)
        samples = torch.from_numpy(sample).float()
        samples = samples.permute(1, 0)
        if self.test:
            return samples
        label = row[TARGETS]
        label = torch.tensor(label).float()
        return samples, label


class EEGDatasetV2(EEGDataset):
    """EEG dataset class for loading EEG Spectrogram and Waveform data."""

    def spec_data_generation(self, row):
        """
        Generates data containing batch_size samples. This method directly
        uses class attributes for spectrograms and EEG spectrograms. Follow the
        notebook https://www.kaggle.com/code/cdeotte/how-to-make-spectrogram-from-eeg/notebook.
        Each spectrogram respectively represents LL, LP, RP, RR with size 128x256x4.
        """
        img = self.eeg_specs[row.eeg_id].transpose(2, 0, 1)
        # change it from 4x128x256 to 3x224x224
        # the first channel is LL and LP, and the second channel is RR and RP
        # First layer is LL and LP
        new_img = np.zeros((256, 256, 3), dtype=img.dtype)
        new_img[:, :, 0] = np.concatenate([img[0], img[1]], axis=0)
        # Second layer is RR and RP
        new_img[:, :, 1] = np.concatenate((img[3], img[2]), axis=0)
        # Third layer is mean of LP and RP and mean of LL and RR
        new_img[:, :, 2] = np.concatenate(
            [(img[1] + img[2]) / 2, (img[0] + img[3]) / 2],
            axis=0,
        )
        img = resize(new_img, (224, 224))
        img = img.transpose(2, 0, 1)
        return img


# Inherit from EEGDatasetV2 and EEGSpecDataset
class EEGSpecDatasetV2(EEGDatasetV2, EEGSpecDataset):
    """EEG spectrogram dataset class for loading"""

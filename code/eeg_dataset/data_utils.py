"""
Data Transformation Utilities, especially for EEG data.
"""

import sys
import os
import numpy as np
from scipy.signal import butter, lfilter
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config


def quantize_data(data, classes):
    """Quantize the data into the given number of classes."""
    mu_x = mu_law_encoding(data, classes)
    # bins = np.linspace(-1, 1, classes)
    # quantized = np.digitize(mu_x, bins) - 1
    return mu_x


def mu_law_encoding(data, mu):
    """Get the mu-law encoding of the data."""
    mu_x = np.sign(data) * np.log(1 + mu * np.abs(data)) / np.log(mu + 1)
    return mu_x


def mu_law_expansion(data, mu):
    """Get the mu-law expansion of the data."""
    s = np.sign(data) * (np.exp(np.abs(data) * np.log(mu + 1)) - 1) / mu
    return s


def butter_lowpass_filter(data, cutoff_freq=20, sampling_rate=200, order=4):
    """Apply a low-pass Butterworth filter to the data."""
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    filtered_data = lfilter(b, a, data, axis=0)
    return filtered_data


def get_fold_dls(df_train, df_valid, all_eegs, all_spectrograms, dataset_cls):
    """Get the train and validation data loaders."""
    ds_train = dataset_cls(
        df_train, eegs=all_eegs, eeg_specs=all_spectrograms, test=False
    )
    ds_val = dataset_cls(
        df_valid, eegs=all_eegs, eeg_specs=all_spectrograms, test=False
    )
    dl_train = DataLoader(
        ds_train, batch_size=Config.batch_size, shuffle=True, num_workers=2
    )
    dl_val = DataLoader(ds_val, batch_size=Config.batch_size, num_workers=2)
    return dl_train, dl_val, df_valid

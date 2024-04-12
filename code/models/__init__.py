"""
Initializes the models package.
"""

from .lightning_model import EEGModel, EEGModelBase
from .multimodel import EEGMultiModel
from .spec_model import EEGSpecNet
from .wave_model import EEGWaveNet


class ModelFactory:
    """Model factory class to create models"""

    EEGModel = EEGMultiModel
    EEGWaveNet = EEGWaveNet
    EEGSpecNet = EEGSpecNet


class ModelPrototye:
    """Model prototype class to create models that will wrap the factory"""

    EEGModel = EEGModel
    EEGWaveNet = EEGModelBase
    EEGSpecNet = EEGModelBase


__all__ = [
    "EEGModel",
    "EEGModelBase",
    "EEGMultiModel",
    "EEGSpecNet",
    "EEGWaveNet",
    "ModelFactory",
    "ModelPrototye",
]

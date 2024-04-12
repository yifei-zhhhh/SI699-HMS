"""
Utility function to obtain the optimizer and learning rate scheduler for the model.
"""

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


def get_optimizer(lr, params, weight_decay, epochs):
    """Obtain the optimizer and learning rate scheduler for the model.

    Args:
        lr (float): Learning rate.
        params (list): List of parameters to optimize.
        weight_decay (float): Weight decay.
        epochs (int): Number of epochs.

    Returns:
        dict: Dictionary containing the optimizer and learning rate scheduler.
    """
    model_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, params), lr=lr, weight_decay=weight_decay
    )
    interval = "epoch"

    lr_scheduler = CosineAnnealingWarmRestarts(
        model_optimizer, T_0=epochs, T_mult=1, eta_min=1e-7, last_epoch=-1
    )

    return {
        "optimizer": model_optimizer,
        "lr_scheduler": {
            "scheduler": lr_scheduler,
            "interval": interval,
            "monitor": "val_loss",
            "frequency": 1,
        },
    }

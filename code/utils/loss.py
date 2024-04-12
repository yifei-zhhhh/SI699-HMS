"""
Loss functions for training neural networks.
"""

from torch import nn


class KLDivLossWithLogits(nn.Module):
    """KL Divergence loss with logits."""

    def forward(self, input, target):
        """KL Divergence loss with logits."""
        y = nn.functional.log_softmax(input, dim=1)
        # take the log of the target if 0 is encountered in the target return 0
        target_log = target.log()
        target_log[target == 0] = 0
        divergence = target * (target_log - y)
        loss = divergence.sum(dim=1).mean()
        return loss


class CrossEntropyLossWithLogits(nn.Module):
    """Cross entropy loss with logits."""

    def forward(self, input, target):
        """Cross entropy loss with logits."""
        return nn.functional.cross_entropy(input, target)

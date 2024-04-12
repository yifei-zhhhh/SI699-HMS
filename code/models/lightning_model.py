"""
This file contains the PyTorch Lightning model for EEG data.
It is used to train the model and evaluate the performance with
pytorch-lightning.
"""

import torch
from torch import nn
import pytorch_lightning as pl
import pandas as pd
from sklearn.metrics import f1_score
from utils.loss import KLDivLossWithLogits
from utils.optimizer import get_optimizer
from utils.config import Config

TARGETS = Config.TARGETS


class EEGModelBase(pl.LightningModule):
    """Base class for EEG models with Spectrogram or Waveform data."""

    def __init__(self, fold, exp_config, BaseModel, loss_cls=KLDivLossWithLogits):
        super().__init__()
        self.exp_config = exp_config
        self.backbone = BaseModel(exp_config, num_classes=6)
        self.loss_function = loss_cls()
        self.validation_step_outputs = []
        self.best_score = 1000.0
        self.fold = fold
        self.linear_init(self.backbone.linear_layers)

    def linear_init(self, layers):
        """Initialize the linear layers of the model."""
        for layer in layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """Forward pass of the model."""
        return self.backbone(x)

    def configure_optimizers(self):
        """Configure the optimizer for the model."""
        return get_optimizer(
            lr=self.exp_config.LR,
            params=self.parameters(),
            weight_decay=self.exp_config.weight_decay,
            epochs=self.exp_config.epochs,
        )

    def training_step(self, batch, batch_idx):
        """Training step of the model."""
        x, target = batch
        _, y_pred = self(x)
        total_loss = self.loss_function(y_pred, target)
        pred_labels = torch.argmax(y_pred, dim=1)
        correct = torch.argmax(target, dim=1)
        f1 = f1_score(
            correct.cpu().detach().numpy(),
            pred_labels.cpu().detach().numpy(),
            average="macro",
        )
        acc = torch.sum(pred_labels == correct).item() / len(pred_labels)
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_f1", f1, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step of the model."""
        x, target = batch
        _, y_pred = self(x)
        val_loss = self.loss_function(y_pred, target)
        pred_labels = torch.argmax(y_pred, dim=1)
        correct = torch.argmax(target, dim=1)
        f1 = f1_score(
            correct.cpu().detach().numpy(),
            pred_labels.cpu().detach().numpy(),
            average="macro",
        )
        acc = torch.sum(pred_labels == correct).item() / len(pred_labels)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log("val_f1", f1, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.validation_step_outputs.append(
            {
                "val_loss": val_loss,
                "logits": y_pred,
                "targets": target,
                "f1": f1,
                "acc": acc,
            }
        )
        return {
            "val_loss": val_loss,
            "logits": y_pred,
            "targets": target,
            "f1": f1,
            "acc": acc,
        }

    def on_validation_epoch_end(self):
        """Validation end step of the model."""
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        output_val = (
            nn.functional.softmax(
                torch.cat([x["logits"] for x in outputs], dim=0), dim=1
            )
            .cpu()
            .detach()
            .numpy()
        )
        target_val = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        self.validation_step_outputs = []

        val_df = pd.DataFrame(target_val, columns=list(TARGETS))
        pred_df = pd.DataFrame(output_val, columns=list(TARGETS))

        val_df["id"] = [f"id_{i}" for i in range(len(val_df))]
        pred_df["id"] = [f"id_{i}" for i in range(len(pred_df))]

        avg_score = avg_loss

        if avg_score < self.best_score:
            print(
                f"Fold {self.fold}: Epoch {self.current_epoch} validation loss {avg_loss}"
            )
            print(
                f"Fold {self.fold}: Epoch {self.current_epoch} validation KDL score {avg_score}"
            )
            self.best_score = avg_score

        return {"val_loss": avg_loss, "val_cmap": avg_score}


class EEGModel(EEGModelBase):
    """EEG Model for EEG data with Spectrogram and Waveform data."""

    def __init__(self, fold, exp_config, BaseModel, loss_cls=KLDivLossWithLogits):
        super().__init__(fold, exp_config, BaseModel, loss_cls)
        self.num_classes = exp_config.num_classes
        self.contrastive_loss = (
            nn.CosineEmbeddingLoss()
        )  # Using cosine similarity for contrastive loss

    def forward(self, eeg, spec):
        """Forward pass of the EEG model."""
        return self.backbone(eeg, spec)

    def training_step(self, batch, batch_idx):
        """Training step of the EEG model."""
        eeg, spec, target = batch
        y_pred, embedding_1d, embedding_2d, yp1, yp2 = self(eeg, spec)
        classification_loss = self.loss_function(y_pred, target)

        classification_loss1 = self.loss_function(yp1, target)
        classification_loss2 = self.loss_function(yp2, target)

        embedding_1d = torch.nn.functional.normalize(embedding_1d, p=2, dim=1)
        embedding_2d = torch.nn.functional.normalize(embedding_2d, p=2, dim=1)

        contrastive_target = torch.ones(embedding_1d.size(0)).to(
            self.device
        )  # Assuming all pairs are similar
        contrastive_loss = self.contrastive_loss(
            embedding_1d, embedding_2d, contrastive_target
        )

        total_loss = (
            classification_loss
            + classification_loss1 * 0.5
            + classification_loss2 * 0.5
            + contrastive_loss * 0.5
        )
        self.log("train_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        eeg, spec, target = batch
        y_pred, _, _, y1d, y2d = self(eeg, spec)

        y_pred = y_pred * 0.5 + y1d * 0.25 + y2d * 0.25
        val_loss = self.loss_function(y_pred, target)
        pred_labels = torch.argmax(y_pred, dim=1)
        correct = torch.argmax(target, dim=1)
        f1 = f1_score(
            correct.cpu().detach().numpy(),
            pred_labels.cpu().detach().numpy(),
            average="macro",
        )
        acc = torch.sum(pred_labels == correct).item() / len(pred_labels)
        self.log(
            "val_loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log("val_f1", f1, on_step=True, on_epoch=True, logger=True, prog_bar=True)
        self.log(
            "val_acc", acc, on_step=True, on_epoch=True, logger=True, prog_bar=True
        )
        self.validation_step_outputs.append(
            {
                "val_loss": val_loss,
                "logits": y_pred,
                "targets": target,
                "f1": f1,
                "acc": acc,
            }
        )
        return {
            "val_loss": val_loss,
            "logits": y_pred,
            "targets": target,
            "f1": f1,
            "acc": acc,
        }

    def on_validation_epoch_end(self):
        """Validation end step of the EEG model."""
        outputs = self.validation_step_outputs
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        output_val = (
            nn.Softmax(dim=1)(torch.cat([x["logits"] for x in outputs], dim=0))
            .cpu()
            .detach()
            .numpy()
        )
        target_val = (
            torch.cat([x["targets"] for x in outputs], dim=0).cpu().detach().numpy()
        )
        self.validation_step_outputs = []

        val_df = pd.DataFrame(target_val, columns=list(TARGETS))
        pred_df = pd.DataFrame(output_val, columns=list(TARGETS))

        val_df["id"] = [f"id_{i}" for i in range(len(val_df))]
        pred_df["id"] = [f"id_{i}" for i in range(len(pred_df))]

        avg_score = avg_loss

        if avg_score < self.best_score:
            print(
                f"Fold {self.fold}: Epoch {self.current_epoch} validation loss {avg_loss}"
            )
            print(
                f"Fold {self.fold}: Epoch {self.current_epoch} validation KDL score {avg_score}"
            )
            self.best_score = avg_score

        return {"val_loss": avg_loss, "val_cmap": avg_score}

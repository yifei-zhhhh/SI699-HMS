"""
Experiment class for training and inference
"""

import sys
import os
import gc
import warnings
import numpy as np
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import torch
from torch import nn

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from models import ModelFactory, ModelPrototye
from utils.config import Config
from utils.loss import KLDivLossWithLogits
from utils.kaggle_kl_div import score

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("high")
pred_cols = [f"pred_{t}" for t in Config.TARGETS]
cuda_available = torch.cuda.is_available()


def predict(data_loader, model):
    """Predict for a given model and data loader"""
    # The slow prediction might be due to the model being not on the GPU
    if cuda_available:
        model.to("cuda")
    model.eval()
    predictions = []
    for batch in tqdm(data_loader):
        with torch.no_grad():
            outputs = predict_helper(model, batch)
            outputs = nn.functional.softmax(outputs, dim=1)
        predictions.extend(outputs.detach().cpu().numpy())
    predictions = np.vstack(predictions)
    return predictions


def predict_helper(model, batch):
    """Predict helper function for a given model and batch"""
    if isinstance(model, ModelPrototye.EEGModel):
        x, x2, *_ = batch
        if cuda_available:
            x = x.to("cuda")
            x2 = x2.to("cuda")
        outputs, _, _, y1, y2 = model(x, x2)
        outputs = outputs * 0.5 + y1 * 0.25 + y2 * 0.25
    else:
        if len(batch) == 3 and isinstance(model.backbone, ModelFactory.EEGSpecNet):
            _, x, *_ = batch
        else:
            x, *_ = batch
        if cuda_available:
            x = x.to("cuda")
        _, outputs = model(x)
    return outputs


class Experiment:
    """Experiment class for training and inference"""

    def __init__(self, kfold_in_data, train, model_name, loss_cls=KLDivLossWithLogits):
        self.kfold_data = kfold_in_data
        self.train = train
        self.model_name = model_name
        self.loss_cls = loss_cls

    def kfold_train(self, exp_config):
        """Train the model for each fold"""
        train = self.train
        oof_df = train.copy()
        oof_df[pred_cols] = 0.0
        oof_df[pred_cols[0]] = 1.0
        for f in exp_config.trn_folds:
            val_idx = list(train[train["fold"] == f].index)
            print(len(val_idx))
            val_preds = self._run_training(f, exp_config)
            oof_df.loc[val_idx, pred_cols] = val_preds
        oof_pred_df = oof_df[["eeg_id"] + list("pred_" + i for i in Config.TARGETS)]
        oof_pred_df.columns = ["eeg_id"] + list(Config.TARGETS)
        oof_true_df = oof_df[oof_pred_df.columns].copy()
        oof_score = score(
            solution=oof_true_df, submission=oof_pred_df, row_id_column_name="eeg_id"
        )
        print("OOF Score for solution =", oof_score)
        oof_df.to_csv(f"{exp_config.output_dir}/oof.csv", index=False)

    def _build_model(self, fold_id, exp_config):
        model = None
        model_base = getattr(ModelPrototye, self.model_name)
        singleton = getattr(ModelFactory, self.model_name)
        print(f"Building model for fold {fold_id}", exp_config.in_channels_2d)
        model = model_base(fold_id, exp_config, singleton, self.loss_cls)
        return model

    def _run_training(self, fold_id, exp_config):
        dl_train, dl_val, df_valid = self.kfold_data[fold_id]
        eeg_model = self._build_model(fold_id, exp_config)

        net_name = self.model_name
        early_stop_callback = EarlyStopping(
            monitor="val_loss",
            min_delta=0.00,
            patience=exp_config.PATIENCE,
            verbose=True,
            mode="min",
        )
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=f"{exp_config.output_dir}/",
            save_top_k=1,
            save_last=True,
            save_weights_only=False,
            filename=f"{net_name}_best_loss_fold{fold_id}",
            verbose=True,
            mode="min",
        )

        callbacks_to_use = [checkpoint_callback, early_stop_callback]

        trainer = pl.Trainer(
            devices=[0],
            val_check_interval=0.5,
            deterministic=True,
            max_epochs=exp_config.epochs,
            logger=None,
            callbacks=callbacks_to_use,
            precision=exp_config.PRECISION,
            accelerator="gpu",
        )

        print("Running trainer.fit")
        trainer.fit(eeg_model, train_dataloaders=dl_train, val_dataloaders=dl_val)

        prototype = getattr(ModelPrototye, self.model_name)
        singleton = getattr(ModelFactory, self.model_name)
        init_args = {
            "checkpoint_path": f"{exp_config.output_dir}/{net_name}_best_loss_fold{fold_id}.ckpt",
            "fold": fold_id,
            "train_dataloader": None,
            "validation_dataloader": None,
            "exp_config": exp_config,
            "BaseModel": singleton,
        }

        model = prototype.load_from_checkpoint(**init_args)
        preds = predict(dl_val, model)
        df_valid.loc[:, pred_cols] = preds
        df_valid.to_csv(
            f"{exp_config.output_dir}/{net_name}_pred_df_f{fold_id}.csv", index=False
        )
        gc.collect()
        torch.cuda.empty_cache()
        return preds


class KFoldModels:
    """Reload models for each fold for inference and data visualization"""

    def __init__(self, exp_config, model_name):
        self.exp_config = exp_config
        self.model_name = model_name
        self.models = []
        self.__load_models__()

    def __load_models__(self):
        """Load models for each fold"""

        for fold_id in self.exp_config.trn_folds:
            model_base = getattr(ModelPrototye, self.model_name)
            singleton = getattr(ModelFactory, self.model_name)
            ckpt_name = f"{self.model_name}_best_loss_fold{fold_id}.ckpt"
            init_args = {
                "checkpoint_path": f"{self.exp_config.output_dir}/{ckpt_name}",
                "fold": fold_id,
                "train_dataloader": None,
                "validation_dataloader": None,
                "exp_config": self.exp_config,
                "BaseModel": singleton,
            }
            model = model_base.load_from_checkpoint(**init_args)
            self.models.append(model)

    @property
    def kfold_models(self):
        """Return the models"""
        return self.models

    def fold_predict(self, data_loader, fold_id):
        """Predict for a given fold"""
        model = self.models[fold_id]
        return predict(data_loader, model)

    def predict(self, data_loader):
        """Predict for each fold"""
        preds = []
        for model in tqdm(self.models):
            preds.append(predict(data_loader, model))
        return preds

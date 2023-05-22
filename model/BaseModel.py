from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from util.tokenizer import Tokenizer
from torch.nn import CrossEntropyLoss, CTCLoss
from torchmetrics import Accuracy, Precision, Recall, F1Score, WordErrorRate, CharErrorRate
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


class BaseModel(pl.LightningModule):
    r"""
    Super class of openspeech models.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        outputs (dict): Result of model predictions.
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(BaseModel, self).__init__()
        self.optimizer = None
        self.configs = configs
        self.num_classes = len(tokenizer.vocab)
        self.tokenizer = tokenizer
        self.current_val_loss = 100.0
        if hasattr(configs, "trainer"):
            self.gradient_clip_val = configs.trainer.gradient_clip_val
        if hasattr(configs, "criterion"):
            self.criterion = self.configure_criterion(configs.criterion.criterion_name)

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor) -> Dict[str, Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Inputs:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            outputs (dict): Result of model predictions.
        """
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for training.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def validation_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for validation.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def test_step(self, batch: tuple, batch_idx: int):
        r"""
        Forward propagate a `inputs` and `targets` pair for test.

        Inputs:
            batch (tuple): A train batch contains `inputs`, `targets`, `input_lengths`, `target_lengths`
            batch_idx (int): The index of batch

        Returns:
            loss (torch.Tensor): loss for training
        """
        raise NotImplementedError

    def configure_optimizers(self):
        r"""
        Choose what optimizers and learning-rate schedulers to use in your optimization.


        Returns:
            - **Dictionary** - The first item has multiple optimizers, and the second has multiple LR schedulers (or multiple ``lr_dict``).
        """
        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
        }

        assert self.configs.model.optimizer in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.model.optimizer}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.model.optimizer](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr,
        )

        SCHEDULER_REGISTRY = {
            "lambda_lr": LambdaLR,
            "step_lr": StepLR,
            "multi_step_lr": MultiStepLR,
            "exponential_lr": ExponentialLR,
            "cosine_annealing_lr": CosineAnnealingLR,
            "reduce_lr_on_plateau": ReduceLROnPlateau,
            "warmup_reduce_lr_on_plateau": ReduceLROnPlateau,
        }

        scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](self.optimizer,
                                                                                 self.configs.lr_scheduler)

        if self.configs.lr_scheduler.scheduler_name == "reduce_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
            }
        elif self.configs.lr_scheduler.scheduler_name == "warmup_reduce_lr_on_plateau":
            lr_scheduler = {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "step",
            }
        else:
            lr_scheduler = {
                "scheduler": scheduler,
                "interval": "step",
            }

        return [self.optimizer], [lr_scheduler]

    def configure_criterion(self, criterion_name: str) -> nn.Module:
        r"""
        Configure criterion for training.

        Args:
            criterion_name (str): name of criterion

        Returns:
            criterion (nn.Module): criterion for training
        """
        CRITERION_REGISTRY = {

        }

        if criterion_name in ("joint_ctc_cross_entropy", "label_smoothed_cross_entropy"):
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                num_classes=self.num_classes,
                tokenizer=self.tokenizer,
            )
        else:
            return CRITERION_REGISTRY[criterion_name](
                configs=self.configs,
                tokenizer=self.tokenizer,
            )

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr
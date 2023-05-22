from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from util.tokenizer import Tokenizer
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
        self.lr_scheduler = None
        self.save_hyperparameters()
        self.optimizer = None
        self.configs = configs
        self.num_classes = configs.model.num_classes
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

        assert self.configs.lr_scheduler.scheduler_name is not None and \
               self.configs.lr_scheduler.scheduler_name in SUPPORTED_OPTIMIZERS.keys(), \
            (f"Unsupported Optimizer: {self.configs.lr_scheduler.scheduler_name}\n"
             f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}")

        self.lr_scheduler = SCHEDULER_REGISTRY[self.configs.lr_scheduler.scheduler_name](self.optimizer,
                                                                                         **self.configs.lr_scheduler)

        return [self.optimizer], [self.lr_scheduler]

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

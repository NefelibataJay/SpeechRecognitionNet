from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from util.tokenizer import Tokenizer
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau


class BaseModel(pl.LightningModule):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(BaseModel, self).__init__()
        self.lr_scheduler = None
        self.optimizer = None
        self.criterion = None
        self.configs = configs
        self.tokenizer = tokenizer
        self.save_hyperparameters()

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.LongTensor) -> Dict[str, Tensor]:
        raise NotImplementedError

    def training_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def validation_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def test_step(self, batch: tuple, batch_idx: int):
        raise NotImplementedError

    def configure_optimizers(self):
        SUPPORTED_OPTIMIZERS = {
            "adam": Adam,
            "adagrad": Adagrad,
            "adadelta": Adadelta,
            "adamax": Adamax,
            "adamw": AdamW,
            "sgd": SGD,
            "asgd": ASGD,
        }

        assert self.configs.training.optimizer_name in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.training.optimizer_name}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.training.optimizer_name](
            self.parameters(),
            **self.configs.optimizer
        )

        if self.configs.training.lr_scheduler_name is None:
            return {'optimizer': self.optimizer}


        # TODO add lr_scheduler
        SCHEDULER_REGISTRY = {
            "lambda_lr": LambdaLR,
            "step_lr": StepLR,
            "multi_step_lr": MultiStepLR,
            "exponential_lr": ExponentialLR,
            "cosine_annealing_lr": CosineAnnealingLR,
            "reduce_lr_on_plateau": ReduceLROnPlateau,
            "warmup_reduce_lr_on_plateau": ReduceLROnPlateau,
        }

        assert self.configs.training.lr_scheduler_name in SCHEDULER_REGISTRY.keys(), (
            f"Unsupported Optimizer: {self.configs.training.lr_scheduler_name}\n"
            f"Supported Optimizers: {SCHEDULER_REGISTRY.keys()}")

        self.lr_scheduler = SCHEDULER_REGISTRY[self.configs.training.lr_scheduler_name](
            self.optimizer,
            **self.configs.lr_scheduler)

        return {'optimizer': self.optimizer, 'lr_scheduler': self.lr_scheduler}

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g["lr"]

    def set_lr(self, lr):
        for g in self.optimizer.param_groups:
            g["lr"] = lr

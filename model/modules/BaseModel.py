from typing import Dict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import ASGD, SGD, Adadelta, Adagrad, Adam, Adamax, AdamW

from util.tokenizer import Tokenizer
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.nn import CrossEntropyLoss


class BaseModel(pl.LightningModule):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(BaseModel, self).__init__()
        self.lr_scheduler = None
        self.save_hyperparameters()
        self.optimizer = None
        self.configs = configs
        self.num_classes = configs.model.num_classes
        self.tokenizer = tokenizer
        self.current_val_loss = 100.0
        self.criterion = None

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

        assert self.configs.optimizer.optimizer_name in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.optimizer.optimizer_name}\n"
            f"Supported Optimizers: {SUPPORTED_OPTIMIZERS.keys()}"
        )

        self.optimizer = SUPPORTED_OPTIMIZERS[self.configs.optimizer.optimizer_name](
            self.parameters(),
            lr=self.configs.lr_scheduler.lr,
        )

        if self.configs.lr_scheduler.scheduler_name is None:
            return [self.optimizer]

        SCHEDULER_REGISTRY = {
            "lambda_lr": LambdaLR,
            "step_lr": StepLR,
            "multi_step_lr": MultiStepLR,
            "exponential_lr": ExponentialLR,
            "cosine_annealing_lr": CosineAnnealingLR,
            "reduce_lr_on_plateau": ReduceLROnPlateau,
            "warmup_reduce_lr_on_plateau": ReduceLROnPlateau,
        }

        assert self.configs.lr_scheduler.scheduler_name in SUPPORTED_OPTIMIZERS.keys(), (
            f"Unsupported Optimizer: {self.configs.lr_scheduler.scheduler_name}\n"
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

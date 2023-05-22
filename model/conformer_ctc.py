import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

import pytorch_lightning as pl

from model.conformer import Conformer
from model.module.CTCModel import CTCModel
from util.tokenizer import Tokenizer
from torchmetrics import F1Score, Accuracy, WordErrorRate, CharErrorRate
from torch.nn import CTCLoss


class ConformerCTC(CTCModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerCTC).__init__(configs=configs, tokenizer=tokenizer)
        self.loss = F.ctc_loss
        self.configs = configs
        self.criterion = CharErrorRate()
        self.ConformerEncoder = Conformer(self.configs.model)

        self.dropout = nn.Dropout(self.configs.model.dropout)

    def forward(self, x):
        x = self.normalize(x)

        # pass through the conformer encoder layers
        x = self.encoder_layers(x)

        # average pool over the time dimension
        x = F.adaptive_avg_pool1d(x, 1).squeeze(-1)

        # apply final linear layer and return logits
        x = self.dropout(x)
        x = self.linear(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.ctc_loss(logits.transpose(0, 1), y, reduction='mean')
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.ctc_loss(logits.transpose(0, 1), y, reduction='mean')
        preds = torch.argmax(logits, dim=-1)
        self.log('val_loss', loss)
        self.log('val_cer', ctc_cer(preds, y))

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)
        return optimizer

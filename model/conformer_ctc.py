from collections import OrderedDict

import torch
from torch import Tensor
from omegaconf import DictConfig
from torch.optim import Adam

from model.conformer import Conformer
from model.modules.BaseModel import BaseModel
from util.tokenizer import Tokenizer
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate, WordErrorRate


def get_batch(batch):
    inputs, input_lengths, targets, target_lengths = batch
    # inputs [batch_size, time, feature]

    batch_size = inputs.size(0)
    input_dim = inputs.size(2)

    return (
        inputs,
        input_lengths,
        targets,
        target_lengths,
    )


class ConformerCTC(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerCTC).__init__(configs=configs, tokenizer=tokenizer)
        self.configs = configs
        self.criterion = CTCLoss(blank=0)
        self.val_cer = CharErrorRate(ignore_case=True)
        self.ConformerEncoder = Conformer(self.configs.model)

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        # pred
        y_hat, outputs, outputs_length = self.recognize(inputs, input_lengths)
        return y_hat

    def training_step(self, batch: tuple, batch_idx: int):
        (inputs, input_lengths, targets, target_lengths,) = self.get_batch(batch)

        outputs = self.ConformerEncoder(inputs, input_lengths)
        loss = self.criterion(outputs.transpose(0, 1), targets, reduction='mean')
        self.log('train_loss', loss)
        self.log('lr', self.lr)

        return {'loss': loss, 'learning_rate': self.lr}

    def validation_step(self, batch, batch_idx):
        (inputs, input_lengths, targets, target_lengths,) = self.get_batch(batch)

        outputs = self.ConformerEncoder(inputs, input_lengths)
        loss = self.criterion(outputs.transpose(0, 1), targets, reduction='mean')
        predicts = self(outputs, dim=-1)

        predicts = [self.tokenizer.int2text(sent) for sent in predicts]
        targets = [self.tokenizer.int2text(sent) for sent in targets]

        list_cer = []
        for i, j in zip(predicts, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer))

        self.log('val_loss', loss)
        self.log('val_cer', char_error_rate)

        return {'loss': loss, 'learning_rate': char_error_rate}

    def test_step(self, batch, batch_idx):
        (inputs, input_lengths, targets, target_lengths,) = self.get_batch(batch)

        outputs = self.ConformerEncoder(inputs, input_lengths)
        loss = self.criterion(outputs.transpose(0, 1), targets, reduction='mean')
        predicts = self(outputs, dim=-1)

        predicts = [self.tokenizer.int2text(sent) for sent in predicts]
        targets = [self.tokenizer.int2text(sent) for sent in targets]

        list_cer = []
        for i, j in zip(predicts, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer))

        self.log('test_loss', loss)
        self.log('test_cer', char_error_rate)

        return {'loss': loss, 'learning_rate': char_error_rate}

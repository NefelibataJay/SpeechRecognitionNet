import torch
from torch import Tensor
from omegaconf import DictConfig

from model.encoder.conformer.encoder import ConformerEncoder
from model.encoder.conformer.modules import Linear
from model.modules.BaseModel import BaseModel
from util.tokenizer import Tokenizer
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate


class ConformerCTC(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerCTC, self).__init__(configs=configs, tokenizer=tokenizer)

        self.criterion = CTCLoss(blank=configs.model.blank_id,reduction='mean')
        self.val_cer = CharErrorRate(ignore_case=True, reduction='mean')
        self.encoder_configs = self.configs.model.encoder
        self.encoder = ConformerEncoder(
            input_dim=self.encoder_configs.input_dim,
            encoder_dim=self.encoder_configs.encoder_dim,
            num_layers=self.encoder_configs.num_encoder_layers,
            num_attention_heads=self.encoder_configs.num_attention_heads,
            feed_forward_expansion_factor=self.encoder_configs.feed_forward_expansion_factor,
            conv_expansion_factor=self.encoder_configs.conv_expansion_factor,
            input_dropout_p=self.encoder_configs.input_dropout_p,
            feed_forward_dropout_p=self.encoder_configs.feed_forward_dropout_p,
            attention_dropout_p=self.encoder_configs.attention_dropout_p,
            conv_dropout_p=self.encoder_configs.conv_dropout_p,
            conv_kernel_size=self.encoder_configs.conv_kernel_size,
            half_step_residual=self.encoder_configs.half_step_residual,
        )

        self.fc = Linear(self.encoder_configs.encoder_dim, self.configs.model.num_classes, bias=False)

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = encoder_outputs
        logits = self.fc(logits).log_softmax(dim=-1)

        if self.decoder is not None:
            y_hats = self.decoder(logits)
        else:
            y_hats = logits.max(-1)[1]
        return {
            "predictions": y_hats,
            "logits": logits,
            "output_lengths": output_lengths,
        }

    def training_step(self, batch: tuple, batch_idx: int):
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        logits = self.fc(encoder_outputs).log_softmax(dim=-1)

        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )

        self.log('train_loss', loss)
        self.log('lr', self.lr)

        return {'loss': loss, 'learning_rate': self.lr}

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths,targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        # logits.transpose(0, 1),targets[:, 1:],output_lengths,target_lengths,
        # 第一个参数的维度 (Batch, Frames, Classes) -> (Frames, Batch, Classes)
        # 第二个参数的维度 去掉targets的第一个字符 <sos>
        #
        # 去除targets的第一个字符<sos>的长度，因为CTC的输入是不包含<sos>的
        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = logits.max(-1)[1]
        predictions = [self.tokenizer.int2text(sent) for sent in predictions]
        targets = [self.tokenizer.int2text(sent) for sent in targets]

        list_cer = []
        for i, j in zip(predictions, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer))

        self.log('val_loss', loss)
        self.log('val_cer', char_error_rate)

        return {'loss': loss, 'learning_rate': char_error_rate}

    def test_step(self, batch, batch_idx):
        inputs, targets, input_lengths, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        logits = self.fc(encoder_outputs).log_softmax(dim=-1)

        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = logits.max(-1)[1]
        predictions = [self.tokenizer.int2text(sent) for sent in predictions]
        targets = [self.tokenizer.int2text(sent) for sent in targets]

        list_cer = []
        for i, j in zip(predictions, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer))

        self.log('val_loss', loss)
        self.log('val_cer', char_error_rate)

        return {'loss': loss, 'learning_rate': char_error_rate}

import torch
from torch import Tensor
from omegaconf import DictConfig

from model.encoder.conformer_encoder import ConformerEncoder
from model.modules.modules import Linear
from model.BaseModel import BaseModel
from tool.search.greedy_search import greedy_search
from tool.search.search_common import remove_pad
from util.tokenizer import Tokenizer
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate


class ConformerCTC(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerCTC, self).__init__(configs=configs, tokenizer=tokenizer)

        self.criterion = CTCLoss(blank=configs.model.blank_id, reduction='mean')
        self.val_cer = CharErrorRate(ignore_case=True, reduction='mean')
        self.encoder_configs = self.configs.model.encoder
        self.encoder = ConformerEncoder(
            num_classes=self.configs.model.num_classes,
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

        self.decoder = None

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        encoder_outputs, output_lengths, logits = self.encoder(inputs, input_lengths)

        return logits

    def training_step(self, batch: tuple, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths, logits = self.encoder(inputs, input_lengths)

        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )

        self.log('train_loss', loss)
        self.log('lr', self.get_lr())

        return {'loss': loss, 'learning_rate': self.get_lr()}

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths, logits = self.encoder(inputs, input_lengths)

        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:-1],  # remove sos, eos
            input_lengths=output_lengths,
            target_lengths=target_lengths - 2,  # remove sos, eos
        )

        hyps, scores = greedy_search(log_probs=logits, encoder_out_lens=output_lengths, eos=self.configs.model.eos_id)

        predictions = [self.tokenizer.int2text(sent) for sent in hyps]

        targets = [self.tokenizer.int2text(remove_pad(sent)) for sent in targets]

        list_cer = []
        for i, j in zip(predictions, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer)) * 100

        self.log('val_loss', loss)
        self.log('val_cer', char_error_rate)

        return {'val_loss': loss, 'CER': char_error_rate}

    def test_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths, logits = self.encoder(inputs, input_lengths)

        loss = self.criterion(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )

        hyps, scores = greedy_search(log_probs=logits, encoder_out_lens=output_lengths, eos=self.configs.model.eos_id)

        predictions = [self.tokenizer.int2text(sent) for sent in hyps]

        targets = [self.tokenizer.int2text(remove_pad(sent)) for sent in targets]

        list_cer = []
        for i, j in zip(predictions, targets):
            self.val_cer.update(i, j)
            list_cer.append(self.val_cer.compute())

        char_error_rate = torch.mean(torch.tensor(list_cer)) * 100

        self.log('val_loss', loss)
        self.log('val_cer', char_error_rate)

        return {'test_loss': loss, 'CER': char_error_rate}

    def recognize(self):
        pass

import torch
from torch import Tensor
from omegaconf import DictConfig

from model.decoder.transformer_decoder import TransformerDecoder
from model.encoder.conformer_encoder import ConformerEncoder
from model.modules.modules import Linear
from model.BaseModel import BaseModel
from tool.Loss.label_smoothing_loss import LabelSmoothingLoss
from tool.search.greedy_search import greedy_search
from tool.search.search_common import remove_pad
from util.tokenizer import Tokenizer
from torch.nn import CTCLoss
from torchmetrics import CharErrorRate


class ConformerCTCAttention(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerCTCAttention, self).__init__(configs=configs, tokenizer=tokenizer)
        self.vocab_size = self.configs.model.num_classes
        self.criterion_ctc = CTCLoss(blank=configs.model.blank_id, reduction='mean')
        self.criterion_att = LabelSmoothingLoss(
            size=self.vocab_size,
            padding_idx=self.configs.model.pad_id,
            smoothing=self.model.lsm_weight,
        )

        self.val_cer = CharErrorRate(ignore_case=True, reduction='mean')
        self.encoder_configs = self.configs.model.encoder
        self.encoder = ConformerEncoder(
            num_classes=self.vocab_size,
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
        self.decoder_configs = self.configs.model.decoder
        self.decoder = TransformerDecoder(
            num_classes=self.vocab_size,
            d_model=self.decoder_configs.d_model,
            d_ff=self.decoder_configs.d_ff,
            num_layers=self.decoder_configs.num_layers,
            num_heads=self.decoder_configs.num_heads,
            dropout_p=self.decoder_configs.dropout_p,
            pad_id=self.configs.model.pad_id,
            sos_id=self.configs.model.sos_id,
            eos_id=self.configs.model.eos_id,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        return logits

    def training_step(self, batch: tuple, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)

        decoder_logits = self.decoder(encoder_outputs, targets, output_lengths, target_lengths)

        loss = self.criterion_ctc(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:-1],  # remove sos, eos
            input_lengths=output_lengths,
            target_lengths=target_lengths - 2,  # remove sos, eos
        )

        self.log('train_loss', loss)
        self.log('lr', self.get_lr())

        return {'loss': loss, 'learning_rate': self.get_lr()}

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs).log_softmax(dim=-1)
        loss = self.criterion_ctc(
            log_probs=logits.transpose(0, 1),
            targets=targets[:, 1:-1],  # remove sos, eos
            input_lengths=output_lengths,
            target_lengths=target_lengths - 2,  # remove sos, eos
        )

        hyps, scores = greedy_search(log_probs=logits, encoder_out_lens=output_lengths,
                                     eos=self.configs.model.eos_id)

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
        pass

import torch
from torch import Tensor
from omegaconf import DictConfig
import torch.nn as nn
from model.decoder.transducer_decoder import RNNTransducerDecoder
from model.encoder.conformer_encoder import ConformerEncoder
from model.modules.modules import Linear
from model.BaseModel import BaseModel
from tool.common import remove_pad
from tool.search.greedy_search import ctc_greedy_search
from util.tokenizer import Tokenizer
from torchmetrics import CharErrorRate
from torchaudio.functional import rnnt_loss


class ConformerTransducer(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerTransducer, self).__init__(configs=configs, tokenizer=tokenizer)
        self.criterion = rnnt_loss
        self.val_cer = CharErrorRate(ignore_case=True, reduction='mean')
        self.encoder_configs = self.configs.model.encoder

        self.num_classes = self.configs.model.num_classes
        self.blank = self.configs.model.blank_id
        self.pad = self.configs.model.pad_id
        self.sos = self.configs.model.sos_id
        self.eos = self.configs.model.eos_id

        self.encoder = ConformerEncoder(
            num_classes=self.num_classes,
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
        self.decoder = RNNTransducerDecoder(
            num_classes=self.num_classes,
            hidden_state_dim=self.decoder_configs.hidden_state_dim,
            output_dim=self.decoder_configs.output_dim,
            num_layers=self.decoder_configs.num_layers,
            rnn_type=self.decoder_configs.rnn_type,
            dropout_p=self.decoder_configs.dropout_p,
            embed_dropout=self.decoder_configs.embed_dropout,
        )
        joint_dim = self.encoder_configs.encoder_dim + self.decoder_configs.output_dim
        self.joint_fc = nn.Sequential(
            Linear(in_features=joint_dim, out_features=joint_dim),
            nn.Tanh(),
            Linear(in_features=joint_dim, out_features=self.num_classes),
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        return 0

    def joint(self, encoder_outputs: Tensor, decoder_outputs: Tensor) -> Tensor:
        r"""
        Joint `encoder_outputs` and `decoder_outputs`.
        Args:
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            decoder_outputs (torch.FloatTensor): A output sequence of decoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        Returns:
            outputs (torch.FloatTensor): outputs of joint `encoder_outputs` and `decoder_outputs`..
        """
        assert encoder_outputs.dim() == decoder_outputs.dim()
        input_length = encoder_outputs.size(1)
        target_length = decoder_outputs.size(1)

        encoder_outputs = encoder_outputs.unsqueeze(2)
        decoder_outputs = decoder_outputs.unsqueeze(1)

        encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
        decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        outputs = self.joint_fc(outputs).log_softmax(dim=-1)

        return outputs

    def training_step(self, batch: tuple, batch_idx: int):
        # inputs = (batch, padding seq_len, input_dim)
        # input_lengths = (batch)  no padding seq_len
        # targets = (batch, padding seq_len)  add sos and eos
        # target_lengths = (bath)  no padding seq_len, add sos and eos
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets, target_lengths)

        logits = self.joint(encoder_outputs, decoder_outputs)

        rnnt_text = targets.contiguous().to(torch.int32)

        loss = self.criterion(logits=logits, targets=rnnt_text,
                              logit_lengths=output_lengths, target_lengths=target_lengths,
                              blank=self.pad, fused_log_softmax=False,
                              reduction='mean'
                              )
        self.log('train_loss', loss)
        self.log('lr', self.get_lr())

        return {'loss': loss, 'learning_rate': self.get_lr()}

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        decoder_outputs, _ = self.decoder(targets[:, :-1], target_lengths - 1)

        logits = self.joint(encoder_outputs, decoder_outputs)

        rnnt_text = targets[:, 1:].contiguous().to(torch.int32)

        loss = self.criterion(logits=logits, targets=rnnt_text,
                              logit_lengths=output_lengths, target_lengths=target_lengths - 2,
                              blank=self.pad, fused_log_softmax=False,
                              reduction='mean'
                              )

        hyps, scores = ctc_greedy_search(log_probs=logits, encoder_out_lens=output_lengths, eos=self.eos)

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

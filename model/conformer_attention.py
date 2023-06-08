from torch import Tensor
from omegaconf import DictConfig

from model.decoder.transformer_decoder import TransformerDecoder
from model.encoder.conformer_encoder import ConformerEncoder
from model.BaseModel import BaseModel
from tool.Loss.label_smoothing_loss import LabelSmoothingLoss
from tool.common import add_sos_eos, add_sos, add_eos
from util.tokenizer import Tokenizer
from torchmetrics import CharErrorRate


class ConformerAttention(BaseModel):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(ConformerAttention, self).__init__(configs=configs, tokenizer=tokenizer)
        self.num_classes = self.configs.model.num_classes
        self.blank = self.configs.model.blank_id
        self.pad = self.configs.model.pad_id
        self.sos = self.configs.model.sos_id
        self.eos = self.configs.model.eos_id

        self.criterion = LabelSmoothingLoss(
            size=self.num_classes,
            padding_idx=self.pad,
            smoothing=0.1,
        )

        self.val_cer = CharErrorRate(ignore_case=True, reduction='mean')
        self.encoder_configs = self.configs.model.encoder

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
        self.decoder = TransformerDecoder(
            num_classes=self.num_classes,
            d_model=self.decoder_configs.d_model,
            d_ff=self.decoder_configs.d_ff,
            num_layers=self.decoder_configs.num_layers,
            num_heads=self.decoder_configs.num_heads,
            dropout_p=self.decoder_configs.dropout_p,
            pad_id=self.pad,
            sos_id=self.sos,
            eos_id=self.eos,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor):
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        decoder_logits = self.decoder(encoder_outputs, output_lengths)

        return decoder_logits

    def training_step(self, batch: tuple, batch_idx: int):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        decoder_input = add_sos(targets, self.sos)  # add sos
        target_lengths = target_lengths + 1
        decoder_outputs = self.decoder(encoder_outputs=encoder_outputs, targets=decoder_input,
                                       encoder_output_lengths=output_lengths, target_lengths=target_lengths)

        targets = add_eos(targets, self.eos)  # add eos
        loss = self.criterion_att(decoder_outputs, targets)

        self.log('train_loss', loss)
        self.log('lr', self.get_lr())

        return {'loss': loss, 'learning_rate': self.get_lr()}

    def validation_step(self, batch, batch_idx):
        inputs, input_lengths, targets, target_lengths = batch

        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        decoder_input = add_sos(targets, self.sos)  # add sos
        decoder_outputs = self.decoder(encoder_outputs, decoder_input, output_lengths, target_lengths + 1)

        targets = add_eos(targets, self.eos)  # add eos
        loss = self.criterion(decoder_outputs, targets)

        self.log('val_loss', loss)

        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        pass

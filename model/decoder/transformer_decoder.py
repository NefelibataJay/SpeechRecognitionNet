from typing import Tuple, Optional

import torch
from torch import nn, Tensor

from model.modules.attention import MultiHeadAttention
from model.modules.embedding import PositionalEncoding, TransformerEmbedding
from model.modules.feed_forward import PositionwiseFeedForward
from tool.mask import get_attn_pad_mask, get_attn_subsequent_mask


class TransformerDecoderLayer(nn.Module):
    r"""
    DecoderLayer is made up of self-attention, multi-head attention and feedforward network.
    This standard decoders layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer decoder layer
        encoder_outputs (torch.FloatTensor): outputs of encoder
        self_attn_mask (torch.BoolTensor): mask of self attention
        encoder_output_mask (torch.BoolTensor): mask of encoder outputs

    Returns:
        (Tensor, Tensor, Tensor)

        * outputs (torch.FloatTensor): output of transformer decoder layer
        * self_attn (torch.FloatTensor): output of self attention
        * encoder_attn (torch.FloatTensor): output of encoder attention

    Reference:
        Ashish Vaswani et al.: Attention Is All You Need
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            d_model: int = 256,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
    ) -> None:
        super(TransformerDecoderLayer, self).__init__()
        self.self_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.attention_dropout = nn.Dropout(dropout_p)

        self.decoder_attention_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.decoder_attention = MultiHeadAttention(d_model, num_heads)
        self.decoder_attention_dropout = nn.Dropout(dropout_p)

        self.feed_forward_norm = nn.LayerNorm(d_model, eps=1e-5)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(
            self,
            inputs: Tensor,
            encoder_outputs: Tensor,
            self_attn_mask: Optional[Tensor] = None,
            encoder_attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Forward propagate transformer decoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer decoder layer
            encoder_outputs (torch.FloatTensor): outputs of encoder
            self_attn_mask (torch.BoolTensor): mask of self attention
            encoder_output_mask (torch.BoolTensor): mask of encoder outputs

        Returns:
            outputs (torch.FloatTensor): output of transformer decoder layer
            self_attn (torch.FloatTensor): output of self attention
            encoder_attn (torch.FloatTensor): output of encoder attention
        """
        residual = inputs
        inputs = self.self_attention_norm(inputs)
        outputs, self_attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.decoder_attention_norm(outputs)
        outputs, encoder_attn = self.decoder_attention(outputs, encoder_outputs, encoder_outputs, encoder_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_norm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, self_attn, encoder_attn


class TransformerDecoder(nn.Module):
    r"""
    The TransformerDecoder is composed of a stack of N identical layers.
    Each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, third is a feed-forward network.

    Args:
        num_classes: umber of classes
        d_model: dimension of model
        d_ff: dimension of feed forward network
        num_layers: number of layers
        num_heads: number of attention heads
        dropout_p: probability of dropout
        pad_id (int, optional): index of the pad symbol (default: 0)
        sos_id (int, optional): index of the start of sentence symbol (default: 1)
        eos_id (int, optional): index of the end of sentence symbol (default: 2)
        max_length (int): max decoding length
    """

    def __init__(
            self,
            num_classes: int,
            d_model: int = 512,
            d_ff: int = 512,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_p: float = 0.1,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
            max_length: int = 128,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.max_length = max_length
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.num_classes = num_classes

        self.embedding = TransformerEmbedding(self.num_classes, pad_id, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len=200)

        self.input_dropout = nn.Dropout(p=dropout_p)
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.fc = nn.Sequential(
            nn.LayerNorm(d_model, eps=1e-5),
            nn.Linear(d_model, num_classes, bias=False),
        )

    def forward_step(
            self,
            decoder_inputs: torch.Tensor,
            decoder_input_lengths: torch.Tensor,
            encoder_outputs: torch.Tensor,
            encoder_output_lengths: torch.Tensor,
            positional_encoding_length: int,
    ):
        dec_self_attn_pad_mask = get_attn_pad_mask(decoder_inputs, decoder_input_lengths, decoder_inputs.size(1))
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_inputs)
        self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        encoder_attn_mask = get_attn_pad_mask(encoder_outputs, encoder_output_lengths, decoder_inputs.size(1))

        outputs = self.embedding(decoder_inputs) + self.positional_encoding(positional_encoding_length)
        outputs = self.input_dropout(outputs)

        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            outputs, self_attn, encoder_attn = layer(
                inputs=outputs,
                encoder_outputs=encoder_outputs,
                self_attn_mask=self_attn_mask,
                encoder_attn_mask=encoder_attn_mask,
            )
            dec_self_attns.append(self_attn)
            dec_enc_attns.append(encoder_attn)

        return outputs, dec_self_attns, dec_enc_attns

    def forward(
            self,
            encoder_outputs: torch.Tensor,
            targets: Optional[torch.LongTensor] = None,
            encoder_output_lengths: torch.Tensor = None,
            target_lengths: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        Forward propagate a `encoder_outputs` for training.

        Args:
            target_lengths:
            targets (torch.LongTensor): A target sequence passed to decoders. `IntTensor` of size
                ``(batch, seq_length)``
            encoder_outputs (torch.FloatTensor): A output sequence of encoders. `FloatTensor` of size
                ``(batch, seq_length, dimension)``
            encoder_output_lengths (torch.LongTensor): The length of encoders outputs. ``(batch)``

        Returns:
            * logits (torch.FloatTensor): Log probability of model predictions.
        """
        outputs = None
        batch_size = encoder_outputs.size(0)

        if targets is not None:
            max_length = targets.size(1)
            outputs, self_attn, memory_attn = self.forward_step(
                decoder_inputs=targets,
                decoder_input_lengths=target_lengths,
                encoder_outputs=encoder_outputs,
                encoder_output_lengths=encoder_output_lengths,
                positional_encoding_length=max_length,
            )
        # Inference
        else:
            input_var = encoder_outputs.new_zeros(batch_size, self.max_length).long()
            input_var = input_var.fill_(self.pad_id)
            input_var[:, 0] = self.sos_id  # add sos

            for di in range(1, self.max_length):
                input_lengths = torch.IntTensor(batch_size).fill_(di)

                outputs, dec_self_attns, dec_enc_attns = self.forward_step(
                    decoder_inputs=input_var[:, :di],
                    decoder_input_lengths=input_lengths,
                    encoder_outputs=encoder_outputs,
                    encoder_output_lengths=encoder_output_lengths,
                    positional_encoding_length=di,
                )
        outputs = self.fc(outputs)
        return outputs

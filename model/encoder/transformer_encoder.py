from typing import Tuple

import torch
from torch import nn, Tensor

from model.modules.attention import MultiHeadAttention
from model.modules.embedding import PositionalEncoding
from model.modules.feed_forward import PositionwiseFeedForward
from model.modules.mask import get_attn_pad_mask
from model.modules.modules import Linear


class TransformerEncoderLayer(nn.Module):
    r"""
    EncoderLayer is made up of self-attention and feedforward network.
    This standard encoders layer is based on the paper "Attention Is All You Need".

    Args:
        d_model: dimension of model (default: 512)
        num_heads: number of attention heads (default: 8)
        d_ff: dimension of feed forward network (default: 2048)
        dropout_p: probability of dropout (default: 0.3)

    Inputs:
        inputs (torch.FloatTensor): input sequence of transformer encoder layer
        self_attn_mask (torch.BoolTensor): mask of self attention

    Returns:
        (Tensor, Tensor)

        * outputs (torch.FloatTensor): output of transformer encoder layer
        * attn (torch.FloatTensor): attention of transformer encoder layer
    """

    def __init__(
            self,
            d_model: int = 512,
            num_heads: int = 8,
            d_ff: int = 2048,
            dropout_p: float = 0.1,
    ) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout_p)

    def forward(self, inputs: Tensor, self_attn_mask: Tensor = None) -> Tuple[Tensor, Tensor]:
        r"""
        Forward propagate of transformer encoder layer.

        Inputs:
            inputs (torch.FloatTensor): input sequence of transformer encoder layer
            self_attn_mask (torch.BoolTensor): mask of self attention

        Returns:
            outputs (torch.FloatTensor): output of transformer encoder layer
            attn (torch.FloatTensor): attention of transformer encoder layer
        """
        residual = inputs
        inputs = self.attention_norm(inputs)
        outputs, attn = self.self_attention(inputs, inputs, inputs, self_attn_mask)
        outputs += residual

        residual = outputs
        outputs = self.feed_forward_norm(outputs)
        outputs = self.feed_forward(outputs)
        outputs += residual

        return outputs, attn


class TransformerEncoder(nn.Module):
    r"""
    The TransformerEncoder is composed of a stack of N identical layers.
    Each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        input_dim: dimension of feature vector
        d_model: dimension of model (default: 512)
        d_ff: dimension of feed forward network (default: 2048)
        num_layers: number of encoders layers (default: 6)
        num_heads: number of attention heads (default: 8)
        dropout_p:  probability of dropout (default: 0.3)
        joint_ctc (bool, optional): flag indication joint ctc attention or not

    Inputs:
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns:
        (Tensor, Tensor, Tensor):

        * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
        * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
            If joint_ctc_attention is False, return None.  ``(batch, seq_length, num_classes)``
        * output_lengths: The length of encoders outputs. ``(batch)``

    Reference:
        Ashish Vaswani et al.: Attention Is All You Need
        https://arxiv.org/abs/1706.03762
    """

    def __init__(
            self,
            num_classes: int,
            input_dim: int = 80,
            d_model: int = 512,
            d_ff: int = 2048,
            num_layers: int = 6,
            num_heads: int = 8,
            dropout_p: float = 0.1,
            joint_ctc: bool = False,
    ) -> None:
        super(TransformerEncoder, self).__init__()

        self.num_classes = num_classes
        self.joint_ctc = joint_ctc

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(p=dropout_p)
        self.positional_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.joint_ctc = joint_ctc
        if self.joint_ctc:
            self.fc = Linear(self.encoder_configs.encoder_dim, self.configs.model.num_classes, bias=False)

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        r"""
        Forward propagate a `inputs` for  encoders training.

        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

        Returns:
            (Tensor, Tensor, Tensor):

            * outputs: A output sequence of encoders. `FloatTensor` of size ``(batch, seq_length, dimension)``
            * encoder_logits: Log probability of encoders outputs will be passed to CTC Loss.
                If joint_ctc_attention is False, return None. ``(batch, seq_length, num_classes)``
            * output_lengths: The length of encoders outputs. ``(batch)``
        """
        logits = 0
        self_attn_mask = get_attn_pad_mask(inputs, input_lengths, inputs.size(1))

        outputs = self.input_norm(self.input_proj(inputs))
        outputs += self.positional_encoding(outputs.size(1))
        outputs = self.input_dropout(outputs)

        for layer in self.layers:
            outputs, attn = layer(outputs, self_attn_mask)

        if self.joint_ctc:
            outputs = nn.Dropout(outputs)
            logits = self.fc(outputs).log_softmax(dim=-1)

        return outputs, input_lengths, logits

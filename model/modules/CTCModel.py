from collections import OrderedDict
from typing import Dict
import torch
from omegaconf import DictConfig
from torch.nn import CTCLoss

from model.modules.BaseModel import BaseModel
from util.tokenizer import Tokenizer


class CTCModel(BaseModel):
    """
    encoder-only models (ctc-model).

    Args:
        configs (DictConfig): configuration set.
        tokenizer (Tokenizer): tokenizer is in charge of preparing the inputs for a model.

    Inputs:
        inputs (torch.FloatTensor): A input sequence passed to encoders. Typically for inputs this will be a padded `FloatTensor` of size ``(batch, seq_length, dimension)``.
        input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``

    Returns:
        ouputs (dict): Result of model predictions that contains `y_hats`, `output`, `output_lengths`
    """

    def __init__(self, configs: DictConfig, tokenizer: Tokenizer) -> None:
        super(CTCModel, self).__init__(configs, tokenizer)
        self.configs = configs
        self.tokenizer = tokenizer
        self.criterion = CTCLoss()
        self.encoder = None
        self.decoder = None

    def set_beam_decoder(self, beam_size: int = 4):
        # TODO beam search
        pass

    def collect_outputs(
            self,
            stage: str,
            output: torch.FloatTensor,
            output_lengths: torch.IntTensor,
            targets: torch.IntTensor,
            target_lengths: torch.IntTensor,
    ) -> OrderedDict:
        loss = self.criterion(
            log_probs=output.transpose(0, 1),
            targets=targets[:, 1:],
            input_lengths=output_lengths,
            target_lengths=target_lengths,
        )
        predictions = output.max(-1)[1]

        wer = self.wer_metric(targets[:, 1:], predictions)
        cer = self.cer_metric(targets[:, 1:], predictions)

        self.info(
            {
                f"{stage}_wer": wer,
                f"{stage}_cer": cer,
                f"{stage}_loss": loss,
                "learning_rate": self.get_lr(),
            }
        )

        return OrderedDict(
            {
                "loss": loss,
                "wer": wer,
                "cer": cer,
            }
        )

    def forward(self, inputs: torch.FloatTensor, input_lengths: torch.IntTensor) -> Dict[str, torch.Tensor]:
        r"""
        Forward propagate a `inputs` and `targets` pair for inference.

        Args: inputs (torch.FloatTensor): A input sequence passed to encoders. Typically, for inputs this will be a
        padded `FloatTensor` of size ``(batch, seq_length, dimension)``. input_lengths (torch.IntTensor): The length
        of input tensor. ``(batch)``

        Returns:
            ouputs (dict): Result of model predictions that contains `y_hats`, `output`, `output_lengths`
        """
        outputs = self.encoder(inputs, input_lengths)

        if len(outputs) == 2:
            output, output_lengths = outputs
        else:
            output, _, output_lengths = outputs

        output = self.fc(output).log_softmax(dim=-1)

        if self.decoder is not None:
            y_hats = self.decoder(output)
        else:
            y_hats = output.max(-1)[1]
        return {
            "predictions": y_hats,
            "output": output,
            "output_lengths": output_lengths,
        }

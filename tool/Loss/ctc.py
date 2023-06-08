import torch
import torch.nn.functional as F


class CTC(torch.nn.Module):
    """CTC module"""
    def __init__(
        self,
        reduce: bool = True,
    ):
        """ Construct CTC module
        Args:
            reduce: reduce the CTC loss into a scalar
        """
        super().__init__()
        reduction_type = "sum" if reduce else "none"
        self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)

    def forward(self, hs_pad: torch.Tensor, h_lens: torch.Tensor,
                ys_pad: torch.Tensor, ys_lens: torch.Tensor) -> torch.Tensor:
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            h_lens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        # ys_hat: (B, L, D) -> (L, B, D)
        ys_hat = hs_pad.transpose(0, 1)
        ys_hat = ys_hat.log_softmax(2)
        loss = self.ctc_loss(ys_hat, ys_pad, h_lens, ys_lens)
        # Batch-size average
        loss = loss / ys_hat.size(1)
        return loss

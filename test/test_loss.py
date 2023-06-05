import torch
import torchaudio

from tool.Loss.label_smoothing_loss import LabelSmoothingLoss

if __name__ == "__main__":
    # x = torch.randn((2, 5, 10))
    # targets = torch.tensor([[2, 3, 4, 0, 0], [2, 3, 4, 0, 0]])
    #
    # loss = LabelSmoothingLoss(10, 0.1)
    # loss(x, targets)
    batch_size = 2
    seq_len = 10
    n_mels = 80
    n_frames = 8
    n_symbols = 4
    # RNNT loss needs log-mel filterbanks
    log_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=512,
        n_mels=80,
        hop_length=160,
        window_fn=torch.hann_window
    ).to('cpu')

    # Random tensor to act as predicted sequence
    pred_tensor = torch.rand(batch_size, seq_len,10, n_mels * n_frames)

    # Random tensor to act as ground truth sequence
    target_tensor = torch.randint(low=0, high=n_symbols, size=(batch_size, seq_len),dtype=torch.int32)

    # Lengths of the sequences
    input_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.int32)
    target_lengths = torch.full(size=(batch_size,), fill_value=seq_len, dtype=torch.int32)

    # RNNT Loss
    loss = torchaudio.functional.rnnt_loss(
        logits=pred_tensor,
        targets=target_tensor,
        logit_lengths=input_lengths,
        target_lengths=target_lengths,
        blank=n_symbols - 1,
        reduction='mean'
    )

    print(loss)


import torch

from model.encoder.conformer.convolution import Conv2dSubsampling

if __name__ == "__main__":
    cov = Conv2dSubsampling(in_channels=1, out_channels=256)
    input_lengths = torch.LongTensor([500])
    x = torch.ones(24, 979, 80)
    y = cov(x, input_lengths)
    print(y.shape)

import random

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


class SpecAugment(torch.nn.Module):
    def __init__(self, num_t_mask=2, num_f_mask=2, max_t=50, max_f=10):
        super(SpecAugment, self).__init__()
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask
        self.max_t = max_t
        self.max_f = max_f

    def forward(self, x):
        y = x.clone().detach()
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(self.num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(self.num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self.max_f)
            end = min(max_freq, start + length)
            y[:, start:end] = 0
        # warping mask
        return y


def compute_mfcc(waveform):
    waveform = waveform * (1 << 15)
    return kaldi.mfcc(waveform, num_mel_bins=23, frame_length=25, frame_shift=10,
                      dither=0.0, num_ceps=23, high_freq=0.0, low_freq=20.0)


def compute_fbank(waveform):
    waveform = waveform * (1 << 15)
    return kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.0)


if __name__ == '__main__':
    waveform, _ = torchaudio.load("E:/Desktop/resources/test.wav")
    speech_feature = compute_fbank(waveform)
    print(speech_feature.shape)

    fbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    speech_feature = fbank(waveform)
    print(speech_feature.shape)

    # specaugment = SpecAugment()
    # specaugment(speech_feature)
    # print(speech_feature.shape)


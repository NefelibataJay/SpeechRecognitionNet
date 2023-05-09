import random

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi


def sort():
    pass


def compute_mfcc(waveform):
    waveform = waveform * (1 << 15)
    return kaldi.mfcc(waveform, num_mel_bins=23, frame_length=25, frame_shift=10,
                      dither=0.0, num_ceps=23, high_freq=0.0, low_freq=20.0)


def compute_fbank(waveform):
    waveform = waveform * (1 << 15)
    return kaldi.fbank(waveform, num_mel_bins=80, frame_length=25, frame_shift=10, dither=0.1)


if __name__ == '__main__':
    waveform, _ = torchaudio.load("E:/Desktop/resources/test.wav")
    speech_feature = compute_mfcc(waveform)
    print(speech_feature.shape)

    waveform = waveform * (1 << 15)
    fbank = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    speech_feature = fbank(waveform)
    print(speech_feature.shape)


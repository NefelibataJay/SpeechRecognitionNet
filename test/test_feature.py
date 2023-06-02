import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio

import soundfile
import librosa

from model.encoder.conformer.convolution import Conv2dSubsampling

if __name__ == "__main__":
    file = "E:/datasets/data_thchs30/data/A2_0.wav"
    waveform, sample_rate = torchaudio.load(file, )

    waveform1, sr1 = librosa.load(file, sr=16000)

    waveform2, sr2 = soundfile.read(file)

    waveform = waveform * (1 << 15)
    waveform1 = waveform1 * (1 << 15)
    waveform2 = waveform2 * (1 << 15)

    feature_spec = kaldi.spectrogram(waveform, )
    feature_spec2 = torchaudio.transforms.Spectrogram(n_fft=512, )(waveform)

    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.0
    num_ceps = 40
    high_freq = 0.0
    low_freq = 20.0
    feature_mfcc = kaldi.mfcc(waveform,
                              num_mel_bins=num_mel_bins,
                              frame_length=frame_length,
                              frame_shift=frame_shift,
                              dither=dither,
                              num_ceps=num_ceps,
                              high_freq=high_freq,
                              low_freq=low_freq,
                              sample_frequency=sample_rate)
    feature_mfcc2 = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=80)(waveform)

    num_mel_bins = 80
    frame_length = 25
    frame_shift = 10
    dither = 0.1
    feature_fbank = kaldi.fbank(waveform,
                                num_mel_bins=num_mel_bins,
                                frame_length=frame_length,
                                frame_shift=frame_shift,
                                dither=dither,
                                energy_floor=0.0,
                                sample_frequency=sample_rate)

    feature_fbank2 = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=80)(waveform)

    # feature_spec.shape, feature_mfcc.shape, feature_fbank.shape,  (time, dim)
    # feature_spec2.shape, feature_mfcc2.shape, feature_fbank2.shape,  (1, dim, time)

    print("ok")

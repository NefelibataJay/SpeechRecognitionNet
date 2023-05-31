import torch
import torchaudio.compliance.kaldi as kaldi
import torchaudio

from model.encoder.conformer.convolution import Conv2dSubsampling

if __name__ == "__main__":
    waveform,sr = torchaudio.load("E:/datasets/data_thchs30/data/A2_0.wav")

    # kaldi.mfcc(waveform,
    #            num_mel_bins=80,
    #            frame_length=frame_length,
    #            frame_shift=frame_shift,
    #            dither=dither,
    #            num_ceps=num_ceps,
    #            high_freq=high_freq,
    #            low_freq=low_freq,
    #            sample_frequency=sample_rate)

    mat = kaldi.fbank(waveform,
                      num_mel_bins=80,
                      frame_length=25,
                      frame_shift=10,
                      dither=0.1,
                      energy_floor=0.0,
                      sample_frequency=sr)
    m2 = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80)(waveform)
    cov = Conv2dSubsampling(in_channels=1, out_channels=256)
    input_lengths = torch.LongTensor([mat.shape[0]])
    mat = mat.unsqueeze(0)
    cov(mat, input_lengths)
    print(mat.shape)
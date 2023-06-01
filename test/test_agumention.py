import torchaudio
import torch
import torchaudio.functional as F
import torchaudio.transforms as T

from IPython.display import Audio
import librosa
import matplotlib.pyplot as plt

from tool.data_augmentations.speech_augment import *


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (db)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


if __name__ == '__main__':
    wav_path = "E:/datasets/data_thchs30/data/A2_0.wav"
    waveform, sample_rate = torchaudio.load(wav_path)

    spec_aug = SpeechAugment()
    # waveform = speed_perturb(waveform, sample_rate)

    n_mels = 80
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=n_mels,
    )

    feature = mel_spectrogram(waveform)
    feature = spec_aug(feature)

    feature = feature.squeeze(0).transpose(1,0)

    feature = spec_trim(feature)

    print("ok")

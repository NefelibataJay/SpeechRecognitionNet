
import torchaudio


if __name__ == '__main__':
    waveform, sample_rate = torchaudio.load('E:/Desktop/resources/zh111.wav')

    fbank_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate,n_mels=80)
    features = fbank_transform(waveform)
    print(features.size())

    features = features.permute(0, 2, 1)  # channel, time, feature
    features = features.squeeze()  # time, feature
    print(features.size())





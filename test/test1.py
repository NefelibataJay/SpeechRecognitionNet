if __name__ == '__main__':
    import librosa
    import torchaudio
    import torch

    extract_feature = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
    signal, _ = librosa.load('E:/Desktop/resources/test.wav', sr=16000)
    signal = signal * (1 << 15)

    wave,_ = torchaudio.load('E:/Desktop/resources/test.wav')
    wave = wave * (1 << 15)

    speech_feature = extract_feature(wave)
    speech_feature2 = extract_feature(torch.tensor(signal))
    print("ok")

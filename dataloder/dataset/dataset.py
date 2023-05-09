import torch
import torchaudio
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
            self,
            encode_string,
            manifest_path="../../manifest/",
            dataset_path="/datasets/aishell/data_aishell/",
            feature_types="fbank",
            data_type="train",
            num_mel_bins=80,
            # spec_aug=True,
    ):
        # TODO add specaugment
        # self.spec_aug = spec_aug

        self.manifest_path = manifest_path
        self.data_type = data_type
        self.dataset_path = dataset_path
        self.feature_types = feature_types
        self.encode_string = encode_string

        assert self.data_type in ["train", "test", "dev"], ".tev file not found"
        assert self.feature_types in ["mfcc", "fbank", "spectrogram"], "feature_types not found"

        if self.feature_types == "mfcc":
            self.extract_feature = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=num_mel_bins)
        elif self.feature_types == "fbank":
            self.extract_feature = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=num_mel_bins)
        elif self.feature_types == "spectrogram":
            self.extract_feature = torchaudio.transforms.Spectrogram(n_fft=400)

        if data_type == "train":
            self.tsv_path = manifest_path + 'train.tsv'
        elif data_type == "dev":
            self.tsv_path = manifest_path + 'dev.tsv'
        elif data_type == "test":
            self.tsv_path = manifest_path + 'test.tsv'

        self.wav_path = []
        self.transcripts = {}

        with open(self.tsv_path, 'r', encoding='utf-8') as tsv_file:
            for line in tsv_file.readlines():
                self.wav_path.append(dataset_path + line.split('\t')[0])
                self.transcripts[line.split('\t')[0]] = str(line.split('\t')[1].strip())

    def __len__(self):
        return len(self.wav_path)

    def __getitem__(self, idx):
        file_name = self.wav_path[idx]
        wave, sr = torchaudio.load(file_name)
        wave = wave * (1 << 15)
        speech_feature = self.extract_feature(wave)

        speech_feature = speech_feature.permute(0, 2, 1)  # channel , time, feature
        speech_feature = speech_feature.squeeze()  # time, feature
        input_lengths = speech_feature.size(0)  # time

        transcript = self._parse_transcript(self.transcripts[file_name])
        target_lengths = len(transcript)

        return speech_feature, input_lengths, transcript, target_lengths

    def _parse_transcript(self, tokens: str) -> list:
        transcript = list()
        transcript.append(1)
        transcript.extend(self.encode_string(tokens))
        transcript.append(2)

        return transcript


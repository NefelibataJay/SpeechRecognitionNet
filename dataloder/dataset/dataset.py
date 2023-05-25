import torch
import torchaudio
from omegaconf import DictConfig
from torch.utils.data import Dataset
import librosa

from util.audio_augment import SpecAugment
from util.tokenizer import Tokenizer


class SpeechToTextDataset(Dataset):
    def __init__(
            self,
            configs: DictConfig,
            tokenizer: Tokenizer,
            data_type="train",
            # spec_aug=True,
    ):
        # TODO add specaugment
        # self.spec_aug = spec_aug

        self.tokenizer = tokenizer
        self.configs = configs
        self.num_mel_bins = configs.num_mel_bins

        self.manifest_path = configs.manifest_path
        self.data_type = data_type
        self.dataset_path = configs.dataset_path
        self.feature_types = configs.feature_types

        assert self.data_type in ["train", "test", "dev"], ".tev file not found"
        assert self.feature_types in ["mfcc", "fbank", "spectrogram"], "feature_types not found"

        # TODO: using k2 to generate feature
        if self.feature_types == "mfcc":
            self.extract_feature = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=self.num_mel_bins)
        elif self.feature_types == "fbank":
            self.extract_feature = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=self.num_mel_bins)
        elif self.feature_types == "spectrogram":
            self.extract_feature = torchaudio.transforms.Spectrogram(n_fft=400)

        self.tsv_path = self.manifest_path + self.data_type + '.tsv'
        self.wav_path = []
        self.transcripts = {}

        with open(self.tsv_path, 'r', encoding='utf-8') as tsv_file:
            for line in tsv_file.readlines():
                self.wav_path.append(self.dataset_path + line.split('\t')[0])
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

    def _parse_transcript(self, tokens: str):
        transcript = list()
        transcript.append(1)
        transcript.extend(self.tokenizer.text2int(tokens))
        transcript.append(2)

        return transcript

    def _parse_speech_wav(self, ):
        # TODO parse speech wav
        pass

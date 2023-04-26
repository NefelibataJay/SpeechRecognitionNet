import os
import torch
import torchaudio
from pathlib import Path
from torch.utils.data import Dataset
import sys
import pickle
import myDataset


class AishellDataset(myDataset.MyDataset):
    ext_txt = ".txt"
    ext_audio = ".wav"

    def __init__(
            self,
            manifest_path="/data_disk/zlf/code/jModel/conformer-rnnt/aishell_mainfest/",
            dataset_path="/data_disk/zlf/datasets/aishell/",
            feature_types="fbank",
            data_type="train",
            n_fft=400,
    ):


        # 频谱图
        self.spect_func = torchaudio.transforms.Spectrogram(n_fft=n_fft)
        self.manifest_path = manifest_path
        self.data_type = data_type
        self.dataset_path = dataset_path
        self.feature_types = feature_types

        assert self.data_type in ["train", "test_module", "dev"], ".tev file not found"
        assert self.feature_types in ["mfcc", "fbank"], "feature_types must be mfcc or fbank"

        if self.feature_types == "mfcc":
            self.extract_feature = torchaudio.transforms.MFCC(sample_rate=16000, n_mfcc=40)
        elif self.feature_types == "fbank":
            self.extract_feature = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=40)
        elif self.feature_types == "spectrogram":
            self.extract_feature = torchaudio.transforms.Spectrogram(n_fft=400)

        if data_type == "train":
            self.tsv_path = manifest_path + 'train.tsv'
        elif data_type == "dev":
            self.tsv_path = manifest_path + 'dev.tsv'
        elif data_type == "test_module":
            self.tsv_path = manifest_path + 'test_module.tsv'

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

        spectrogram = self.spect_func(wave)

        spectrogram = spectrogram.permute(0, 2, 1)  # channel, time, feature
        spectrogram = spectrogram.squeeze()  # time, feature
        input_lengths = spectrogram.size(0)  # time
        transcript = self.transcripts[file_name]
        target_lengths = len(transcript)

        return spectrogram, input_lengths, transcript, target_lengths

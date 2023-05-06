import torchaudio
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(
            self,
            manifest_path="/code/manifest/",
            dataset_path="/datasets/aishell/data_aishell/",
            feature_types="mfcc",
            data_type="train",
            spec_aug=True,
    ):

        self.spec_aug = spec_aug
        self.manifest_path = manifest_path
        self.data_type = data_type
        self.dataset_path = dataset_path
        self.feature_types = feature_types

        assert self.data_type in ["train", "test_module", "dev"], ".tev file not found"
        assert self.feature_types in ["mfcc", "fbank", "spectrogram"], "feature_types not found"

        if self.feature_types == "mfcc":
            self.extract_feature = torchaudio.transforms.MFCC(sample_rate=16000, melkwargs={'n_mels': 80})
        elif self.feature_types == "fbank":
            self.extract_feature = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=80)
        elif self.feature_types == "spectrogram":
            self.extract_feature = torchaudio.transforms.Spectrogram()

        if self.spec_aug:
            pass

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

        speech_feature = self.extract_feature(wave)

        speech_feature = speech_feature.permute(0, 2, 1)  # channel:1 , time, feature:80
        speech_feature = speech_feature.squeeze()  # time, feature:80
        input_lengths = speech_feature.size(0)  # time
        transcript = self.transcripts[file_name]
        target_lengths = len(transcript)

        return speech_feature, input_lengths, transcript, target_lengths

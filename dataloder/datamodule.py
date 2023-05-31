import glob
import os
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from dataloder.dataset.dataset import SpeechToTextDataset
from util.tokenizer import Tokenizer


def _collate_fn(batch):
    inputs = [i[0] for i in batch]

    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = [torch.IntTensor(i[2]) for i in batch]
    target_lengths = torch.IntTensor([i[3]-1 for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths


class SpeechToTextDataModule(pl.LightningDataModule):
    def __init__(self, configs: DictConfig, tokenizer: Tokenizer):
        super(SpeechToTextDataModule, self).__init__()
        self.dataset_path = configs.dataset_path
        self.configs = configs
        self.tokenizer = tokenizer
        self.dataset = dict()

        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers
        self.manifest_path = configs.manifest_path

        if not configs.one_dataset:
            self.val_set_ratio = configs.val_set_ratio
            self.test_set_ratio = configs.test_set_ratio
            self._init_multi_dataset()
        else:
            self._init_one_dataset()

    def _read_manifest_file(self, manifest_file):
        audio_paths = list()
        transcripts = list()
        with open(manifest_file, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f.readlines()):
                audio_path, transcript = line.split("\t")[0], line.split("\t")[1]
                transcript = transcript.replace("\n", "")
                audio_paths.append(os.path.join(self.dataset_path, audio_path))
                transcripts.append(transcript)
        return audio_paths, transcripts

    def _init_one_dataset(self):
        for stage in ["train", "valid", "test"]:
            audio_paths, transcripts = self._read_manifest_file(os.path.join(self.manifest_path, f"{stage}.tsv"))
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs.dataset,
                tokenizer=self.tokenizer,
                audio_paths=audio_paths,
                transcripts=transcripts,
            )

    def _init_multi_dataset(self):
        manifest_files = glob.glob(os.path.join(self.manifest_path, "*.tsv"))
        all_audio_paths = list()
        all_transcripts = list()
        for manifest_file in manifest_files:
            audio_paths, transcripts = self._read_manifest_file(manifest_file)
            all_audio_paths.extend(audio_paths)
            all_transcripts.extend(transcripts)

        data_num = len(all_audio_paths)
        test_start_idx = data_num - int(data_num * self.test_set_ratio)
        valid_start_idx = test_start_idx - int(data_num * self.val_set_ratio)

        audio_paths = {
            "train": all_audio_paths[: valid_start_idx],
            "valid": all_audio_paths[valid_start_idx: test_start_idx],
            "test": all_audio_paths[test_start_idx:],
        }
        transcripts = {
            "train": all_transcripts[: valid_start_idx],
            "valid": all_transcripts[valid_start_idx: test_start_idx],
            "test": all_transcripts[test_start_idx:],
        }

        for stage in audio_paths.keys():
            self.dataset[stage] = SpeechToTextDataset(
                configs=self.configs.dataset,
                tokenizer=self.tokenizer,
                audio_paths=audio_paths[stage],
                transcripts=transcripts[stage],
            )

    def train_dataloader(self):
        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset["valid"],
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

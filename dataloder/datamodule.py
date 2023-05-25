import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import pytorch_lightning as pl


def _collate_fn(batch):
    inputs = [i[0] for i in batch]

    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = torch.tensor([i[2] for i in batch], dtype=torch.int32)
    target_lengths = torch.IntTensor([i[3] - 1 for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths


class SpeechToTextDataModule(pl.LightningDataModule):
    def __init__(self,
                 configs: DictConfig,
                 train_set,
                 val_set,
                 test_set,
                 ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = configs.batch_size
        self.num_workers = configs.num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=_collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

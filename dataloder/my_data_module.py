import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl


class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                 train_set,
                 val_set,
                 test_set,
                 encode_string,
                 batch_size,
                 num_workers: int = 0,
                 ):
        super().__init__()
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.batch_size = batch_size
        self.encode_string = encode_string
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def _collate_fn(self, batch):
        inputs = [i[0] for i in batch]
        input_lengths = torch.IntTensor([i[1] for i in batch])
        targets = [self.encode_string(i[2]) for i in batch]
        target_lengths = torch.IntTensor([i[3] for i in batch])

        # batch, time, feature 填充
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)

        # 填充 最长的维度
        targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True).to(dtype=torch.int)

        return inputs, input_lengths, targets, target_lengths

import argparse
import os

import torch
import pytorch_lightning as pl
import hydra
from lightning_fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
import torch.utils.data as data

from dataloder.datamodule import SpeechToTextDataModule
from dataloder.dataset.dataset import SpeechToTextDataset
from model.modules.BaseModel import BaseModel
from util.tokenizer import Tokenizer, CharTokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="./conf", help="config path")  # config path
parser.add_argument("-cn", default="configs", help="config name")  # config name

args = parser.parse_args()


@hydra.main(config_path=args.cp, config_name=args.cn)
def main(configs: DictConfig):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    tokenizer = CharTokenizer(configs.tokenizer)

    if configs.datasets.only_one_set:
        train_set = SpeechToTextDataset(configs.datasets, tokenizer, data_type="train")
        train_set_size = int(len(train_set) * 0.8)
        valid_set_size = len(train_set) - train_set_size
        seed = torch.Generator().manual_seed(42)
        train_set, val_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)
        test_set = val_set
    else:
        train_set = SpeechToTextDataset(configs.datasets, tokenizer, data_type="train")
        val_set = SpeechToTextDataset(configs.datasets, tokenizer, data_type="dev")
        test_set = SpeechToTextDataset(configs.datasets, tokenizer, data_type="test")

    configs.model.num_classes = len(tokenizer.vocab)

    data_module = SpeechToTextDataModule(configs.datamodule, train_set=train_set, val_set=val_set, test_set=test_set, )

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    if configs.train:
        model = BaseModel(configs.model, tokenizer)
        trainer = pl.Trainer(default_root_dir="some/path/", progress_bar=True, logger=logger, **configs.trainer)
        if configs.checkpoint_path:
            trainer.fit(model, data_module, ckpt_path="some/path/to/my_checkpoint.ckpt")
        else:
            trainer.fit(model, data_module)
    else:
        assert configs.checkpoint_path is not None
        model = BaseModel.load_from_checkpoint("/path/to/checkpoint.ckpt")
        model.eval()


if __name__ == "__main__":
    main()

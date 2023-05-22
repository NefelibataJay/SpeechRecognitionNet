import argparse
import os

import torch
import pytorch_lightning as pl
import hydra
from lightning_fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig
import torch.utils.data as data

from dataloder.dataset.dataset import MyDataset
from model.modules.BaseModel import BaseModel
from util.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="conf", help="config path")  # config path
parser.add_argument("-cn", default="configs", help="config name")  # config name

args = parser.parse_args()


@hydra.main(config_path=args.cp, config_name=args.cn)
def main(cfg: DictConfig):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    text_process = Tokenizer(**cfg.tokenizer)

    train_set = MyDataset(Tokenizer.text2int, **cfg)

    # use 20% of training data for validation
    train_set_size = int(len(train_set) * 0.8)
    valid_set_size = len(train_set) - train_set_size

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = data.random_split(train_set, [train_set_size, valid_set_size], generator=seed)

    cfg.model.num_classes = len(text_process.vocab)

    logger = TensorBoardLogger(save_dir=os.getcwd(), version=1, name="lightning_logs")

    if cfg.train:
        model = BaseModel(**cfg.model)
        trainer = pl.Trainer(default_root_dir="some/path/", progress_bar=True, logger=logger, **cfg.trainer)
        if cfg.checkpoint_path:
            trainer.fit(model, ckpt_path="some/path/to/my_checkpoint.ckpt")
        else:
            trainer.fit(model)
    else:
        assert cfg.checkpoint_path is not None
        model = BaseModel.load_from_checkpoint("/path/to/checkpoint.ckpt")
        model.eval()


if __name__ == "__main__":
    main()

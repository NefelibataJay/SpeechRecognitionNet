import argparse
import os

import torch
import pytorch_lightning as pl
import hydra
from lightning_fabric.loggers import TensorBoardLogger
from omegaconf import DictConfig

from dataloder.datamodule import SpeechToTextDataModule
from model.modules.BaseModel import BaseModel
from util.tokenizer import EnglishCharTokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="./conf", help="config path")  # config path
parser.add_argument("-cn", default="configs", help="config name")  # config name

args = parser.parse_args()


@hydra.main(config_path=args.cp, config_name=args.cn)
def main(configs: DictConfig):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    # set seed
    pl.seed_everything(666)
    torch.manual_seed(666)

    tokenizer = EnglishCharTokenizer(configs.tokenizer)

    configs.model.num_classes = len(tokenizer.vocab)

    data_module = SpeechToTextDataModule(configs.datamodule, tokenizer)

    logger = TensorBoardLogger(**configs.logger)

    ## TODO add do-train
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

import argparse
import os

import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig

from dataloder.datamodule import SpeechToTextDataModule
from model.conformer_ctc import ConformerCTC
from model.modules.BaseModel import BaseModel
from util.tokenizer import EnglishCharTokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="./conf", help="config path")  # config path
parser.add_argument("-cn", default="configs", help="config name")  # config name
args = parser.parse_args()


@hydra.main(version_base=None, config_path=args.cp, config_name=args.cn)
def main(configs: DictConfig):
    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    # set seed
    pl.seed_everything(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    print(configs)

    tokenizer = EnglishCharTokenizer(configs.tokenizer)

    configs.model.num_classes = len(tokenizer)

    data_module = SpeechToTextDataModule(configs.datamodule, tokenizer)

    logger = TensorBoardLogger(**configs.logger)

    if configs.training.do_train:
        model = ConformerCTC(configs, tokenizer)
        print(model)
        trainer = pl.Trainer(logger=logger, **configs.trainer)
        if configs.training.checkpoint_path is not None:
            trainer.fit(model, datamodule=data_module, ckpt_path=configs.training.checkpoint_path)
        else:
            trainer.fit(model, datamodule=data_module)
    else:
        assert configs.checkpoint_path is not None
        model = BaseModel.load_from_checkpoint("/path/to/checkpoint.ckpt")
        model.eval()


if __name__ == "__main__":
    main()

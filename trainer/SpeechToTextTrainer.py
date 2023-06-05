import argparse
import os
import sys

import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig
from torch import nn

from model.conformer_transducer import ConformerTransducer

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from dataloder.datamodule import SpeechToTextDataModule
from model.conformer_ctc import ConformerCTC
from util.tokenizer import EnglishCharTokenizer, ChineseCharTokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="../conf", help="config path")  # config path
parser.add_argument("-cn", default="conformer_transducer_configs", help="config name")  # config name
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

    tokenizer = ChineseCharTokenizer(configs.tokenizer)

    configs.model.num_classes = len(tokenizer)

    data_module = SpeechToTextDataModule(configs.datamodule, tokenizer)

    logger = TensorBoardLogger(**configs.logger)

    checkpoint_callback = ModelCheckpoint(save_top_k=-1,
                                          monitor="val_acc",
                                          dirpath=configs.trainer.default_root_dir,
                                          filename='conformer_ctc-{epoch:02d}-{val_loss:.2f}',
                                          )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        mode='min',
    )

    if configs.training.do_train:
        model = ConformerTransducer(configs, tokenizer)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        print(model)
        trainer = pl.Trainer(logger=logger,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             **configs.trainer)
        if configs.training.checkpoint_path is not None:
            trainer.fit(model, datamodule=data_module, ckpt_path=configs.training.checkpoint_path, )
        else:
            trainer.fit(model, datamodule=data_module, )
    else:
        assert configs.training.checkpoint_path is not None

        model = ConformerTransducer.load_from_checkpoint(configs.training.checkpoint_path)
        model.eval()


if __name__ == "__main__":
    main()

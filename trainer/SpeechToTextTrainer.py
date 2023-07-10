import argparse
import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import torch
import pytorch_lightning as pl
import hydra
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from omegaconf import DictConfig,OmegaConf
from torch import nn

from model.conformer_attention import ConformerAttention
from model.conformer_transducer import ConformerTransducer

from dataloder.datamodule import SpeechToTextDataModule
from model.conformer_ctc import ConformerCTC
from util.tokenizer import EnglishCharTokenizer, ChineseCharTokenizer
from model import REGISTER_MODEL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", default="../conf/configs", help="config path")
    parser.add_argument("--model_name", default="conformer_ctc", help="model name")
    parser.add_argument("--dataset_path", default=" ", help="dataset path")
    parser.add_argument("--manifest_path", default="../manifests/aishell_chars/vocab.txt", help="manifest path")
    parser.add_argument("--checkpoint_path", default=None, help="checkpoint path")

    args = parser.parse_args()
    return args

def main(args):
    configs = OmegaConf.load(args.config_path)

    device = torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    print("Using device", device)

    # set seed
    pl.seed_everything(666)
    torch.manual_seed(666)
    torch.cuda.manual_seed(666)
    print(configs)

    configs.datamodule.dataset_path = args.dataset_path
    configs.datamodule.manifest_path = args.manifest_path
    configs.tokenizer.word_dict_path = os.path.join(args.manifest_path, "vocab.txt")
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

    assert args.model_name in REGISTER_MODEL
    model = REGISTER_MODEL[args.model_name](configs, tokenizer)

    if configs.training.do_train:
        trainer = pl.Trainer(logger=logger,
                             callbacks=[checkpoint_callback, early_stop_callback],
                             **configs.trainer)
        if configs.training.checkpoint_path is not None:
            trainer.fit(model, datamodule=data_module, ckpt_path=configs.training.checkpoint_path, )
        else:
            for p in model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
            print(model)
            trainer.fit(model, datamodule=data_module, )
    else:
        assert configs.training.checkpoint_path is not None

        model = ConformerCTC.load_from_checkpoint(configs.training.checkpoint_path)
        model.eval()


if __name__ == "__main__":
    args = get_args()
    main(args)

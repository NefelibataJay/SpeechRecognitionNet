import sys
import argparse

import pytorch_lightning as pl
import hydra
from omegaconf import OmegaConf, DictConfig
from util.textprocess import TextProcess
from datasets.librispeech import LibriSpeechDataset
from datasets.aishellDataset import AishellDataset
from datasets.datamodule import AishellDataModule
from model.module import ConformerModule

from pytorch_lightning.callbacks import ModelCheckpoint
from util.textprocess import TextProcess



parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="conf",help="config path")  # config path
parser.add_argument("-cn",default="configs", help="config name")  # config name

args = parser.parse_args()

@hydra.main(config_path=args.cp, config_name=args.cn)
def main(cfg: DictConfig):
    text_process = TextProcess(**cfg.text_process)
    cfg.model.num_classes = len(text_process.vocab)

    if cfg.datasets.dataset_selected == "aishell":
        datasets_cfg = cfg.datasets.techs
        train_set = AishellDataset(
            manifest_path=datasets_cfg.manifest_path,
            data_type='train'
        )
        val_set = AishellDataset(
            manifest_path=datasets_cfg.manifest_path,
            data_type='dev'
        )
        test_set = AishellDataset(
            manifest_path=datasets_cfg.manifest_path,
            data_type='test'
        )

        dm = AishellDataModule(
            train_set=train_set,
            val_set=val_set,
            test_set=test_set,
            predict_set=test_set,
            encode_string=text_process.text2int,
            **cfg.datamodule.aishell
        )

    model = ConformerModule(
        cfg, blank=text_process.list_vocab.index("<p>"), text_process=text_process,
    )

    tb_logger = pl.loggers.tensorboard.TensorBoardLogger(**cfg.tb_logger)

    # checkpoint_callback = ModelCheckpoint(monitor="val_acc")

    trainer = pl.Trainer(logger=tb_logger, **cfg.trainer)

    if cfg.ckpt.train:
        print("Training model")
        if cfg.ckpt.have_ckpt:
            trainer.fit(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
        else:
            try:
                trainer.fit(model=model, datamodule=dm)
            except Exception as e:
                with open("error.txt", "w") as f:
                    f.write(str(e))

        trainer.save_checkpoint(filepath=cfg.ckpt.save_path)

    else:
        print("Train mode turn off!")

    print("Testing model")
    if cfg.ckpt.have_ckpt:
        trainer.test(model, datamodule=dm, ckpt_path=cfg.ckpt.ckpt_path)
    else:
        trainer.test(model, datamodule=dm)



if __name__ == "__main__":
    main()

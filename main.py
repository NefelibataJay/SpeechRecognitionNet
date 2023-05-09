import argparse

import pytorch_lightning as pl
import hydra
from omegaconf import DictConfig
from util.tokenizer import Tokenizer

parser = argparse.ArgumentParser(description="Config path")
parser.add_argument("-cp", default="conf", help="config path")  # config path
parser.add_argument("-cn", default="configs", help="config name")  # config name

args = parser.parse_args()


@hydra.main(config_path=args.cp, config_name=args.cn)
def main(cfg: DictConfig):
    text_process = Tokenizer(**cfg.tokenizer)

    cfg.model.num_classes = len(text_process.vocab)


if __name__ == "__main__":
    main()

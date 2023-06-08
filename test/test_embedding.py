import torch

from model.modules.embedding import PositionalEncoding


def test_positional_encoding():
    ebe = PositionalEncoding(128, 100)
    print(ebe(20))


if __name__ == "__main__":
    test_positional_encoding()

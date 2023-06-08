import torch

from tool.common import *


def test_add_sos_eos():
    sos = 10
    eos = 11
    pad = 0
    targets = torch.tensor([[1, 2, 3, 4, 5],
                            [4, 5, 6, 7, 0],
                            [7, 8, 9, 0, 0]])
    print(add_sos(targets, sos=sos, pad=pad))
    print(add_eos(targets, eos=eos, pad=pad))
    print(add_sos_eos(targets, eos=eos, sos=sos, pad=pad))


if __name__ == '__main__':
    test_add_sos_eos()

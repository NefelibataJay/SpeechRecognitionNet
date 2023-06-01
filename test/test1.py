from pathlib import Path
import os
import numpy as np
import torch
import torchaudio
import jieba

if __name__ == '__main__':
    x = torch.ones(1, 10, 10)
    y = torchaudio.transforms.TimeMasking(5)(x)
    print("ok")

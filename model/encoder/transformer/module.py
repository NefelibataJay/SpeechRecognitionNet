import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Embeddings(nn.Module):
    def __init__(self,
                 vocab_size,
                 d_model,
                 ):
        # d_model=512, vocab=当前语言的词表大小
        super(Embeddings, self).__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        # one-hot转词嵌入，这里有一个待训练的矩阵E，大小是vocab*d_model
        self.d_model = d_model  # 512

    def forward(self, x):
        # x ~ (batch.size, sequence.length, one-hot),
        # one-hot大小=vocab，当前语言的词表大小
        return self.embed(x) * math.sqrt(self.d_model)
        # 得到的10*512词嵌入矩阵，主动乘以sqrt(512)=22.6，
        # 这里我做了一些对比，感觉这个乘以sqrt(512)没啥用… 求反驳。
        # 这里的输出的tensor大小类似于(batch.size, sequence.length, 512)

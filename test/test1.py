import re

import torchaudio


def tokenize(sample,
             symbol_table,
             split_with_space=False):
    txt = sample
    parts = [txt]

    label = []
    tokens = []
    for part in parts:
        if split_with_space:
            part = part.split(" ")
        for ch in part:
            if ch == ' ':
                ch = "▁"
            tokens.append(ch)

    for ch in tokens:
        if ch in symbol_table:
            label.append(symbol_table[ch])
        elif '<unk>' in symbol_table:
            label.append(symbol_table['<unk>'])

    return tokens, label


if __name__ == '__main__':
    symbol_table = {'<blank>': 0, '<unk>': 1, '中': 2, '华': 3, '人': 4, '民': 5, '共': 6, '和': 7, '国': 8,
                    '<sos/eos>': 9}
    # tokens, label = tokenize("中共和华国人民", symbol_table)
    # print(tokens)
    # print(label)

    print(symbol_table.values())

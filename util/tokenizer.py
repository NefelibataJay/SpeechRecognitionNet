import torch


class Tokenizer:

    def __init__(self, word_dict_path, lang="zh"):
        self.word_dict_path = word_dict_path
        self.lang = lang
        assert self.lang in ["en", "zh"], "Language not found"

        self.vocab = {}
        with open(self.word_dict_path, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                assert len(arr) == 2
                self.vocab[arr[0]] = int(arr[1])

        self.list_vocab = self.vocab.items()

    def text2int(self, s: str) -> torch.Tensor:
        # TODO : punctuation 
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        # TODO : token2text
        return "".join([self.list_vocab[i] for i in s if i > 1])

    def tokenize(self, txt, symbol_table, split_with_space=False):
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

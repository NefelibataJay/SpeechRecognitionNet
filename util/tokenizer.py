import torch
from omegaconf import DictConfig
from typing import List


class Tokenizer:
    def __init__(self, configs: DictConfig, pad_token=0, sos_token=1, eos_token=2, blank_token=3):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.blank_token = blank_token
        self.word_dict_path = configs.word_dict_path
        self.lang = configs.lang
        assert self.lang in ["en", "zh"], "Language not found"

        self.vocab = {}
        self.id_dict = {}

    def text2int(self, tokens: str):
        pass

    def int2text(self, t: torch.Tensor):
        pass

    def __len__(self):
        return len(self.vocab)


class EnglishCharTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig, pad_token=0, sos_token=1, eos_token=2, blank_token=3):
        super(EnglishCharTokenizer, self).__init__(configs, pad_token, sos_token, eos_token, blank_token)
        with open(self.word_dict_path, 'r', encoding='utf8') as dict_file:
            for line in dict_file:
                key, value = line.replace("\n", "").split('|')
                self.vocab[key] = int(value)
                self.id_dict[int(value)] = key

    def text2int(self, tokens: str) -> List[int]:
        label = []
        for ch in tokens:
            if ch in self.vocab:
                label.append(self.vocab[ch])
            elif '<unk>' in self.vocab:
                label.append(self.vocab['<unk>'])
        return label

    def int2text(self, t: torch.Tensor) -> str:
        sentence = str()
        for i in t:
            if i == self.eos_token:
                # i = eos
                break
            elif i == self.blank_token:
                # i = blank
                continue
            sentence += self.id_dict[int(i)]
        return sentence

    def __len__(self):
        return len(self.vocab)


class EnglishWordTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig, pad_token=0, sos_token=1, eos_token=2, blank_token=3):
        super(EnglishWordTokenizer, self).__init__(configs, pad_token, sos_token, eos_token, blank_token)
        pass


class ChineseCharTokenizer(Tokenizer):
    def __init__(self, configs: DictConfig, pad_token=0, sos_token=1, eos_token=2, blank_token=3):
        super(ChineseCharTokenizer, self).__init__(configs, pad_token, sos_token, eos_token, blank_token)
        with open(self.word_dict_path, 'r', encoding='utf8') as dict_file:
            for line in dict_file:
                key, value = line.strip().split('|')
                self.vocab[key] = int(value)
                self.id_dict[int(value)] = key

    def text2int(self, tokens: str) -> List[int]:
        label = []
        for ch in tokens:
            if ch in self.vocab:
                label.append(self.vocab[ch])
            elif '<unk>' in self.vocab:
                label.append(self.vocab['<unk>'])
        return label

    def int2text(self, t: torch.Tensor) -> str:
        sentence = str()
        for i in t:
            if i == self.eos_token:
                # i = eos
                break
            elif i == self.blank_token:
                # i = blank
                continue
            sentence += self.id_dict[int(i)]
        return sentence

    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    pass

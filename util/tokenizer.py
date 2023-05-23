import torch


class Tokenizer:
    def __init__(self, word_dict_path, pad_token=0, sos_token=1, eos_token=2, unk_token=3, lang="zh"):
        self.sos_token = sos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.word_dict_path = word_dict_path
        self.lang = lang
        assert self.lang in ["en", "zh"], "Language not found"

        self.vocab = {}
        self.id_dict = {}
        with open(self.word_dict_path, 'r', encoding='utf8') as dict_file:
            for line in dict_file:
                key, value = line.strip().split(' ')
                self.vocab[key] = int(value)
                self.id_dict[int(value)] = key

    def text2int(self, tokens: str) -> list[int]:
        pass

    def int2text(self, t: torch.Tensor) -> str:
        pass

    def __len__(self):
        return len(self.vocab)


class CharTokenizer(Tokenizer):
    def __init__(self, word_dict_path, pad_token=0, sos_token=1, eos_token=2, unk_token=3, lang="zh"):
        super(CharTokenizer, self).__init__(word_dict_path, pad_token, sos_token, eos_token, unk_token, lang)

    def text2int(self, tokens: str) -> list[int]:
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
            elif i == self.pad_token:
                # i = pad
                continue
            sentence += self.id_dict[int(i)]
        return sentence

    def __len__(self):
        return len(self.vocab)


if __name__ == "__main__":
    pass

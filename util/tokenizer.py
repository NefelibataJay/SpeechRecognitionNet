import torch


class Tokenizer:

    def __init__(self, word_dict_path, lang="zh"):
        self.word_dict_path = word_dict_path
        self.lang = lang
        assert self.lang in ["en", "zh"], "Language not found"

        self.vocab = {}
        self.list_txt = []
        with open(self.word_dict_path, 'r', encoding='utf8') as dict_file:
            for line in dict_file:
                key = line.split(' ')[0]
                value = line.split(' ')[1].strip()
                self.vocab[key] = int(value)
                self.list_txt.append(key)

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
            if i == 2:
                break
            elif i == 0:
                continue
            sentence += self.list_txt[int(i)]
        return sentence


if __name__ == "__main__":
    tokenizer = Tokenizer("../manifest/vocab.txt")


import torch

class TextProcess:
    aux_vocab = ["<p>", "<s>", "<e>", " ", ":", "'"]

    origin_list_vocab = {
        "en": ["<p>", "<s>", "<e>", " ", ":", "'"] + list("abcdefghijklmnopqrstuvwxyz")
    }

    origin_vocab = {
        lang: dict(zip(vocab, range(len(vocab))))
        for lang, vocab in origin_list_vocab.items()
    }

    def __init__(self, lang, word_dict_path=" "):
        self.word_dict_path = word_dict_path
        self.lang = lang
        assert self.lang in ["en", "zh"], "Language not found"
        # loading word dict

        if lang == 'zh':
            assert word_dict_path != " ", "Please input word dict path"
            self.init_zh_vocab()

        self.vocab = self.origin_vocab[lang]
        self.list_vocab = self.origin_list_vocab[lang]

    def init_zh_vocab(self,):
        word_dict = {}
        self.origin_list_vocab['zh'] = []
        with open(self.word_dict_path, 'r', encoding='utf-8') as word_file:
            for line in word_file.readlines():
                word = line.split(' ')[0]
                self.origin_list_vocab['zh'].append(word)
                idx = line.split(' ')[1].replace('\n','')
                word_dict[word] = int(idx)

        self.origin_vocab['zh'] = word_dict

    def text2int(self, s: str) -> torch.Tensor:
        return torch.Tensor([self.vocab[i] for i in s.lower()])

    def int2text(self, s: torch.Tensor) -> str:
        return "".join([self.list_vocab[i] for i in s if i > 2])

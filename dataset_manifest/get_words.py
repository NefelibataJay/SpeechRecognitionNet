from pathlib import Path
from tqdm import tqdm


def get_word_dict(manifest_path):
    words_list = []
    with open(manifest_path + 'vocab.txt', 'r', encoding='utf-8') as word_file:
        for line in tqdm(word_file.readlines()):
            word_list = line.split(' ')[0].strip()
            words_list.append(word_list)

    manifest_path = manifest_path + 'manifest/'
    with open(manifest_path + 'word_dict.txt', 'w', encoding='utf-8') as word_file:
        with open(manifest_path + 'train-clear-100.tsv') as train_file:
            for line in tqdm(train_file.readlines()):
                word_list = [i for i in line.split('\t')[1].strip()]
                words_list.extend(word_list)

        words_list = list(set(words_list))

        with open(manifest_path + 'test_module.tsv') as test_file:
            for line in tqdm(test_file.readlines()):
                word_list = [i for i in line.split('\t')[1].strip()]
                words_list.extend(word_list)

        words_list = list(set(words_list))
        with open(manifest_path + 'dev.tsv') as dev_file:
            for line in tqdm(dev_file.readlines()):
                word_list = [i for i in line.split('\t')[1].strip()]
                words_list.extend(word_list)

        words_list = list(set(words_list))

        vocab = ["<p>", "<sos/eos>", "<unk>"]
        for v in range(len(vocab)):
            word_file.write(vocab[v] + " " + str(v) + '\n')

        for index, word in enumerate(set(words_list)):
            word_file.write(word + ' ' + str(index) + '\n')


if __name__ == '__main__':
    # path = '/data_disk/zlf/dataloder/data_aishell/'
    # get_set(path)
    path = '/data_disk/zlf/code/jModel/conformer-rnnt/'
    get_word_dict(path)

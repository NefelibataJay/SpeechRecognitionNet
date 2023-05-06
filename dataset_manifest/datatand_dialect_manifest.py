import os
from pathlib import Path
from tqdm import tqdm
import re
import random
import math

def generate_manifest(root_path: str):
    ext_txt = ".metadata"
    ext_audio = ".wav"
    data_path = Path(root_path + "data")
    dialect_list = os.listdir(data_path)

    for dialect in dialect_list:
        dialect_path = data_path / dialect
        dialect_manifest = {}
        for wav_path in tqdm(dialect_path.glob("*/*/*/*/*" + ext_audio), desc=dialect):
            wav_path = str(wav_path).replace('\\', '/')
            dialect_manifest[wav_path] = None
            txt_path = wav_path.replace(ext_audio, ".txt")
            with open(txt_path, 'r', encoding='utf-8') as txt_file:
                dialect_manifest[wav_path] = txt_file.readline().strip()
        write_manifest(root_path, dialect_manifest, dialect)


def write_manifest(root_path, dialect_manifest, dialect):
    with open(root_path + "manifest_all.tsv", 'a', encoding='utf-8') as manifest_file:
        for wav_path in dialect_manifest:
            manifest_file.write(dialect + "\t" + wav_path.replace('E:/datasets/Datatang-Dialect/', '') + "\t"
                                + dialect_manifest[wav_path] + "\n")


def clean_chinese_text(text):
    # 匹配标点符号的正则表达式
    punctuation_pattern = r'[^\w\s]'
    # 匹配 [] 和 <> 中的所有内容，包括中文和英文
    brackets_pattern = r'<[^<>]*?>|\[[^][]*?\]'

    # 使用正则表达式替换匹配到的内容为空字符 ''
    cleaned_text = re.sub(brackets_pattern, '', text)
    cleaned_text = re.sub(punctuation_pattern, '', cleaned_text)

    return cleaned_text


def clean_manifest(manifest_path):
    with open(manifest_path + 'manifest_all.tsv', 'r', encoding='utf-8') as manifest_file:
        with open(manifest_path + 'manifest_final.tsv', 'w', encoding='utf-8') as clean_manifest_file:
            for line in tqdm(manifest_file.readlines()):
                line = line.strip()
                dialect = line.split('\t')[0]
                wav_path = line.split('\t')[1]
                text = line.split('\t')[2]
                text = clean_chinese_text(text).replace(" ","")
                if len(text) > 0:
                    clean_manifest_file.write(dialect + "\t" + wav_path + "\t" + text + "\n")
                else:
                    clean_manifest_file.write(dialect + "\t" + wav_path + "\t" + "无" + "\n")
                    print("空")


def get_train_dev(manifest_path):
    # 读取 manifest.tsv 文件，将每行数据存储为一个元组或字典。
    data = []
    with open(manifest_path+'manifest_final.tsv', 'r',encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        for line in f:
            row = tuple(line.strip().split('\t'))
            data.append(row)

    # 根据类别将元组或字典分组，并统计每个类别的样本数。
    category_counts = {}
    for row in data:
        category = row[0]
        if category not in category_counts:
            category_counts[category] = 0
        category_counts[category] += 1

    # 计算每个类别需要抽取的测试集样本数量，并将其向上取整。
    test_counts = {category: math.ceil(count * 0.08) for category, count in category_counts.items()}

    # 对于每个类别，随机抽取相应数量的样本作为测试集，剩余的作为训练集。
    test_data = []
    train_data = []
    for category, count in category_counts.items():
        category_data = [row for row in data if row[0] == category]
        test_size = test_counts[category]
        test_indices = random.sample(range(count), test_size)
        train_indices = set(range(count)) - set(test_indices)
        test_data.extend([category_data[i] for i in test_indices])
        train_data.extend([category_data[i] for i in train_indices])

    # 将分别抽取的测试集和训练集数据写入 test.tsv 和 train.tsv 文件中。
    with open(manifest_path+'test.tsv', 'w',encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for row in test_data:
            f.write('\t'.join(row) + '\n')

    with open(manifest_path+'train.tsv', 'w',encoding='utf-8') as f:
        f.write('\t'.join(header) + '\n')
        for row in train_data:
            f.write('\t'.join(row) + '\n')


if __name__ == '__main__':
    path = 'E:/datasets/Datatang-Dialect/'
    # generate_manifest(path)
    # print("manifest.tsv has been generated!")
    # clean_manifest(path)
    get_train_dev(path)

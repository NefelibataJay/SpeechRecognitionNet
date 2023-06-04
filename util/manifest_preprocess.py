import argparse
import os
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", required=True, help="dataset name: librispeech,commonvoice,aishell"
    )
    parser.add_argument(
        "--dataset-path",
        default="/datasets/LibriSpeech/",
        help="percentage of data to use as validation set (between 0 and 1)",
    )

    parser.add_argument(
        "--output-path",
        default="./manifests",
        help="percentage of data to use as validation set (between 0 and 1)",
    )

    return parser


class ManifestPreprocess:
    def __init__(self, root_path, output_manifest_path,
                 manifest_type="train", vocab_path="vocab"):
        self.root_path = root_path
        self.output_manifest_path = output_manifest_path
        self.manifest_type = manifest_type
        self.vocab_path = vocab_path

    def generate_character_vocab(self):
        pass

    def generate_manifest_files(self):
        pass

    def generate_word_vocab(self):
        pass


class LibriSpeechManifestPreprocess(ManifestPreprocess):
    LIBRI_SPEECH_DATASETS = [
        "train-clean-100",
        "train-960",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    def __init__(self, root_path="../LibriSpeech/", output_manifest_path="../manifests", ):
        super().__init__(root_path, output_manifest_path)

    def generate_manifest_files(self):
        for sub_path in self.LIBRI_SPEECH_DATASETS:
            # if sub_path.startswith("dev"):
            #     self.manifest_type = "valid"
            # elif sub_path.startswith("test"):
            #     self.manifest_type = "test"
            # else:
            #     self.manifest_type = "train"

            manifest_file_path = os.path.join(self.output_manifest_path, sub_path + ".tsv")
            root_path = os.path.join(self.root_path, sub_path)
            if not os.path.exists(root_path):
                continue

            root_path = Path(root_path)
            transcript_paths = list(root_path.glob("*/*/*.trans.txt"))
            with open(manifest_file_path, "a", encoding="utf-8") as manifest_file:
                for transcript_path in transcript_paths:
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        for line in f.readlines():
                            line = line.strip()
                            audio_id, transcript = line.split(" ", 1)
                            speaker_id, _, _ = audio_id.split("-")
                            audio_path = os.path.join(str(transcript_path.parent), audio_id + ".flac")
                            audio_path = "/".join(audio_path.replace(str(root_path.parent), "").split("\\")[1:])
                            manifest_file.write(audio_path + "\t" +
                                                transcript.lower().replace("'", "") + "\t" + speaker_id + "\n")

    def generate_character_vocab(self):
        vocab_file_path = os.path.join(self.output_manifest_path, self.vocab_path + ".txt")
        special_tokens = ["<pad>", "<sos>", "<eos>", "<blank>", "<unk>"]
        tokens = special_tokens + list("abcdefghijklmnopqrstuvwxyz ")

        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for idx, token in enumerate(tokens):
                vocab_file.write(f'{token}|{str(idx)}\n')

    def generate_word_vocab(self):
        # TODO generate word vocab
        pass


class Thchs30Preprocess(ManifestPreprocess):
    def __init__(self, root_path="../data_thchs30/", output_manifest_path="../manifests", ):
        super().__init__(root_path, output_manifest_path)

    def generate_manifest_files(self):
        pass

    def generate_character_vocab(self):
        pass

    def generate_word_vocab(self):
        pass


class AishellPreprocess(ManifestPreprocess):
    def __init__(self, root_path="../aishell/data_aishell/", output_manifest_path="../manifests", ):
        super().__init__(root_path, output_manifest_path)

    def generate_manifest_files(self):
        transcripts_dict = {}
        transcripts = os.path.join(self.root_path, "transcript", "aishell_transcript_v0.8.txt")
        with open(transcripts, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                audio_id, transcript = line.strip().split(" ", 1)
                transcripts_dict[audio_id] = transcript.replace(" ", "")

        train_paths = list(Path(os.path.join(self.root_path, "wav", "train")).glob("*/*.wav"))
        dev_paths = list(Path(os.path.join(self.root_path, "wav", "dev")).glob("*/*.wav"))
        test_paths = list(Path(os.path.join(self.root_path, "wav", "test")).glob("*/*.wav"))

        train_audio_ids = [audio_path.stem.replace(".wav", "") for audio_path in train_paths]
        dev_audio_ids = [audio_path.stem.replace(".wav", "") for audio_path in dev_paths]
        test_audio_ids = [audio_path.stem.replace(".wav", "") for audio_path in test_paths]

        with open(os.path.join(self.output_manifest_path, 'train.tsv'), "w", encoding="utf-8") as manifest_file:
            for audio_id in train_audio_ids:
                if audio_id not in transcripts_dict:
                    continue
                manifest_file.write(os.path.join("wav", "train", audio_id + ".wav") + "\t" + transcripts_dict[
                    audio_id] + "\n")

        with open(os.path.join(self.output_manifest_path, 'valid.tsv'), "w", encoding="utf-8") as manifest_file:
            for audio_id in dev_audio_ids:
                if audio_id not in transcripts_dict:
                    continue
                manifest_file.write(os.path.join("wav", "dev", audio_id + ".wav") + "\t" + transcripts_dict[
                    audio_id] + "\n")

        with open(os.path.join(self.output_manifest_path, 'test.tsv'), "w", encoding="utf-8") as manifest_file:
            for audio_id in test_audio_ids:
                if audio_id not in transcripts_dict:
                    continue
                manifest_file.write(os.path.join("wav", "test", audio_id + ".wav") + "\t" + transcripts_dict[
                    audio_id] + "\n")

    def generate_character_vocab(self):
        vocab_file_path = os.path.join(self.output_manifest_path, self.vocab_path + ".txt")
        special_tokens = ["<pad>", "<sos>", "<eos>", "<blank>", "<unk>"]
        transcripts_path = os.path.join(self.root_path, "transcript", "aishell_transcript_v0.8.txt")
        tokens = []
        with open(transcripts_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                _, transcript = line.strip().split(" ", 1)
                transcript = transcript.replace(" ", "")
                tokens.extend(list(transcript))

        tokens = special_tokens + list(set(tokens))
        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for idx, token in enumerate(tokens):
                vocab_file.write(f'{token}|{str(idx)}\n')

    def generate_word_vocab(self):
        pass


def main(args):
    if args.dataset == "librispeech":
        manifest_preprocess = LibriSpeechManifestPreprocess(args.dataset_path, args.output_path)
    elif args.dataset == "thchs30":
        manifest_preprocess = Thchs30Preprocess(args.dataset_path, args.output_path)
    else:
        raise ValueError("dataset not supported")
    manifest_preprocess.generate_manifest_files()
    manifest_preprocess.generate_character_vocab()


if __name__ == '__main__':
    # parser = get_parser()
    # args = parser.parse_args()
    # main(args)
    manifest_preprocess = AishellPreprocess('/data_disk/zlf/datasets/aishell/data_aishell/',
                                            '/data_disk/zlf/code/jModel/SpeechNet/manifests/aishell_chars/')
    manifest_preprocess.generate_character_vocab()

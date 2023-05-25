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
        default="./manifest",
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


class LibriSpeech100ManifestPreprocess(ManifestPreprocess):
    LIBRI_SPEECH_DATASETS = [
        "train-clean-100",
        "train-960",
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
    ]

    def __init__(self, root_path="../LibriSpeech/", output_manifest_path="../manifest", ):
        super().__init__(root_path, output_manifest_path)

    def generate_manifest_files(self):
        for sub_path in self.LIBRI_SPEECH_DATASETS:
            if sub_path.startswith("dev"):
                self.manifest_type = "dev"
            elif sub_path.startswith("test"):
                self.manifest_type = "test"
            else:
                self.manifest_type = "train"

            manifest_file_path = os.path.join(self.output_manifest_path, self.manifest_type + ".tsv")
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
                            audio_path = os.path.join(os.path.dirname(transcript_path.parent), audio_id + ".flac")
                            manifest_file.write(audio_path.replace(str(root_path.parent), "") + "\t" +
                                                transcript.lower().replace("'", "") + "\t" + speaker_id + "\n")

    def generate_character_vocab(self):
        vocab_file_path = os.path.join(self.output_manifest_path, self.vocab_path + ".txt")
        special_tokens = ["<pad>", "<sos>", "<eos>", "<blank>"]
        tokens = special_tokens + list(" abcdefghijklmnopqrstuvwxyz")

        with open(vocab_file_path, "w", encoding="utf-8") as vocab_file:
            for idx, token in enumerate(tokens):
                vocab_file.write(token + " " + str(idx) + "\n")


def main(args):
    if args.dataset == "librispeech":
        manifest_preprocess = LibriSpeech100ManifestPreprocess(args.dataset_path, args.output_path)
        manifest_preprocess.generate_manifest_files()
        manifest_preprocess.generate_character_vocab()


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)

import re

import torch
import torchaudio


def _parse_transcript(self, tokens: str) -> list:
    transcript = list()
    transcript.append(1)
    transcript.extend(self.encode_string(tokens))
    transcript.append(2)

    return transcript


if __name__ == '__main__':
    batch = ([torch.Tensor([10, 11]),torch.Tensor([2,2,2,2,2]),torch.Tensor([2])])
    batch = sorted(batch, key=lambda sample: sample[0].size(0), reverse=True)
    print(batch)


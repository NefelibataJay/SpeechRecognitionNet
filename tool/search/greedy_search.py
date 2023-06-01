import torch
from tool.search.search_common import remove_duplicates_and_blank
from tool.mask import *


def greedy_search(
        log_probs: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        eos: int = 2,
):
    batch_size = log_probs.shape[0]

    maxlen = log_probs.size(1)

    # topk_index = log_probs.argmax(-1)
    topk_prob, topk_index = log_probs.topk(1, dim=2)

    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

    mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)

    topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)

    hyps = [hyp.tolist() for hyp in topk_index]

    scores = topk_prob.max(1)

    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

    return hyps, scores


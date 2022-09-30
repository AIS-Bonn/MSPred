"""
Simple baseline model that simply copies forward the last GT frame and the corresponding
corresponding ground truth representatios, i.e. keypoints or semantics.
"""

import torch
import torch.nn as nn
from collections import defaultdict


class CopyLast(nn.Module):
    """ Simple copy-last baseline """

    def __init__(self):
        super().__init__()

    def forward(self, inputs, n_context, num_preds, periods):
        """ Dummy forward pass """
        seq_len = inputs[0].shape[1]
        out_dict = {"preds": {}, "target_masks": defaultdict(lambda: torch.full((seq_len,), False))}

        n_hier = len(inputs)
        for h in range(n_hier):
            out_dict["preds"][h] = inputs[h][:, n_context-1].unsqueeze(1).repeat(1, num_preds[h], 1, 1, 1)

        for t in range(n_context, seq_len):
            for h in range(n_hier):
                T = t - n_context - periods[h] + 1
                if T % periods[h] == 0 and torch.count_nonzero(out_dict["target_masks"][h]) < num_preds[h]:
                    out_dict["target_masks"][h][t] = True
        return out_dict

import torch.nn as nn
import copy
from .layers import DecoderLayer


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # IMPORTANT: deep copy layers, do NOT reuse same instance
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

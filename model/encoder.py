import torch.nn as nn
import copy
from .layers import EncoderLayer


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        # IMPORTANT: deep copy layers, do NOT reuse same instance
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(N)])
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

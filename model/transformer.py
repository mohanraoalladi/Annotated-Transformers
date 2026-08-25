import torch.nn as nn
from .attention import MultiHeadedAttention
from .feedforward import PositionwiseFeedForward
from .embeddings import Embeddings
from .positional_encoding import PositionalEncoding
from .layers import EncoderLayer, DecoderLayer
from .encoder import Encoder
from .decoder import Decoder
from .generator import Generator


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)

    encoder = Encoder(EncoderLayer(d_model, attn, ff, dropout), N)
    decoder = Decoder(DecoderLayer(d_model, attn, attn, ff, dropout), N)

    src_embed = nn.Sequential(Embeddings(d_model, src_vocab), position)
    tgt_embed = nn.Sequential(Embeddings(d_model, tgt_vocab), position)

    generator = Generator(d_model, tgt_vocab)

    return EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

# =========================
# Project: transformer_mt
# =========================
# Create this structure:
#
# transformer_mt/
# ├── pipeline/
# │   ├── tokenize.py
# │   ├── vocab.py
# │   └── dataloader.py
# ├── model/
# │   ├── modules.py
# │   ├── utils.py
# │   ├── train.py
# │   ├── decode.py
# │   └── visualization.py
# ├── config.py
# └── main.py
#
# Then split the code below into those files.


# =========================
# pipeline/tokenize.py
# =========================

def tokenize(text, spacy_tokenizer):
    """
    Tokenize a given text using a spaCy tokenizer.

    Args:
        text (str): The raw input text.
        spacy_tokenizer: A spaCy language model (e.g., spacy_de).

    Returns:
        List[str]: A list of token strings.
    """
    return [token.text for token in spacy_tokenizer.tokenizer(text)]


# =========================
# pipeline/vocab.py
# =========================

import torch
from os.path import exists
from collections import Counter
from datasets import load_dataset

from .tokenize import tokenize


class Vocab:
    def __init__(self, stoi, itos, default_idx=3):
        self.stoi = stoi
        self.itos = itos
        self.default_idx = default_idx

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_idx)

    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_idx) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def set_default_index(self, idx):
        self.default_idx = idx


def build_vocabulary(spacy_de, spacy_en):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    ds = load_dataset("bentrevett/multi30k", trust_remote_code=False)

    counter_de = Counter()
    counter_en = Counter()

    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            counter_de.update(tokenize_de(ex["de"]))
            counter_en.update(tokenize_en(ex["en"]))

    specials = ["<s>", "</s>", "<blank>", "<unk>"]

    stoi_de = {tok: idx for idx, tok in enumerate(specials)}
    for tok, _ in counter_de.most_common():
        if tok not in stoi_de:
            stoi_de[tok] = len(stoi_de)
    itos_de = {idx: tok for tok, idx in stoi_de.items()}

    stoi_en = {tok: idx for idx, tok in enumerate(specials)}
    for tok, _ in counter_en.most_common():
        if tok not in stoi_en:
            stoi_en[tok] = len(stoi_en)
    itos_en = {idx: tok for tok, idx in stoi_en.items()}

    vocab_src = Vocab(stoi_de, itos_de)
    vocab_tgt = Vocab(stoi_en, itos_en)
    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("data/vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "data/vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("data/vocab.pt", map_location="cpu", weights_only=False)

    print("Vocabulary sizes:")
    print("German vocab:", len(vocab_src))
    print("English vocab:", len(vocab_tgt))
    return vocab_src, vocab_tgt


# =========================
# pipeline/dataloader.py
# =========================

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.functional import pad
from datasets import load_dataset

from .tokenize import tokenize


class Multi30kDataset(Dataset):
    def __init__(self, split):
        ds = load_dataset("bentrevett/multi30k", split=split, trust_remote_code=False)
        self.data = list(ds)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (self.data[idx]["de"], self.data[idx]["en"])


def collate_batch(
    batch,
    src_pipeline,
    tgt_pipeline,
    src_vocab,
    tgt_vocab,
    device,
    max_padding=128,
    pad_id=2,
):
    bos_id = torch.tensor([0], device=device)
    eos_id = torch.tensor([1], device=device)

    src_list = []
    tgt_list = []

    for src_text, tgt_text in batch:
        src_tokens = src_pipeline(src_text)
        tgt_tokens = tgt_pipeline(tgt_text)

        src_ids = torch.cat([
            bos_id,
            torch.tensor(src_vocab(src_tokens), dtype=torch.int64, device=device),
            eos_id,
        ])

        tgt_ids = torch.cat([
            bos_id,
            torch.tensor(tgt_vocab(tgt_tokens), dtype=torch.int64, device=device),
            eos_id,
        ])

        src_padded = pad(src_ids, (0, max_padding - len(src_ids)), value=pad_id)
        tgt_padded = pad(tgt_ids, (0, max_padding - len(tgt_ids)), value=pad_id)

        src_list.append(src_padded)
        tgt_list.append(tgt_padded)

    return torch.stack(src_list), torch.stack(tgt_list)


def create_dataloaders(
    device,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    batch_size=12000,
    max_padding=128,
    is_distributed=True,
):
    def tokenize_de(text):
        return tokenize(text, spacy_de)

    def tokenize_en(text):
        return tokenize(text, spacy_en)

    def collate_fn(batch):
        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=max_padding,
            pad_id=vocab_src.get_stoi()["<blank>"],
        )

    train_dataset = Multi30kDataset("train")
    valid_dataset = Multi30kDataset("validation")

    train_sampler = DistributedSampler(train_dataset) if is_distributed else None
    valid_sampler = DistributedSampler(valid_dataset) if is_distributed else None

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collate_fn,
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=(valid_sampler is None),
        sampler=valid_sampler,
        collate_fn=collate_fn,
    )

    return train_dataloader, valid_dataloader


# =========================
# model/modules.py
# =========================
# NOTE: This is the standard "Annotated Transformer" implementation.
# If you already have this from your notebook, you can paste it here.
# I’ll keep it minimal here; you can replace with your full version.

import copy
import torch
import torch.nn as nn
import math


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super().__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers = clones(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return self.proj(x).log_softmax(dim=-1)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


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

    model = EncoderDecoder(encoder, decoder, src_embed, tgt_embed, generator)

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model


# =========================
# model/utils.py
# =========================

import torch


class Batch:
    def __init__(self, src, tgt, pad=2):
        self.src = src
        self.tgt = tgt
        self.src_mask = (src != pad).unsqueeze(-2)
        self.tgt_mask = self.make_std_mask(tgt, pad)
        self.ntokens = (tgt != pad).sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        tgt_mask = (tgt != pad).unsqueeze(-2)
        size = tgt.size(-1)
        subsequent_mask = torch.triu(torch.ones(1, size, size), diagonal=1).type_as(tgt_mask) == 0
        return tgt_mask & subsequent_mask


class TrainState:
    def __init__(self):
        self.step = 0
        self.accum_step = 0
        self.samples = 0
        self.tokens = 0


class DummyOptimizer:
    def step(self):
        pass

    def zero_grad(self):
        pass


class DummyScheduler:
    def step(self):
        pass


def rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))


def run_epoch(data_iter, model, loss_compute, optimizer, scheduler, mode="train", accum_iter=1, train_state=None):
    total_loss = 0
    total_tokens = 0

    for i, batch in enumerate(data_iter):
        out = model.decode(
            model.encode(batch.src, batch.src_mask),
            batch.src_mask,
            batch.tgt[:, :-1],
            batch.tgt_mask[:, :-1, :-1],
        )
        loss = loss_compute(out, batch.tgt[:, 1:], batch.ntokens)

        if mode.startswith("train"):
            loss.backward()
            if (i + 1) % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

        total_loss += loss.item()
        total_tokens += batch.ntokens.item()

    return total_loss / total_tokens, train_state


class SimpleLossCompute:
    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        loss = self.criterion(x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)) / norm
        return loss


# =========================
# model/train.py
# =========================

import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP

from pipeline.dataloader import create_dataloaders
from model.utils import (
    TrainState,
    Batch,
    SimpleLossCompute,
    DummyOptimizer,
    DummyScheduler,
    rate,
)
from model.modules import make_model
from torch.nn import KLDivLoss


class LabelSmoothing(torch.nn.Module):
    def __init__(self, size, padding_idx, smoothing=0.0):
        super().__init__()
        self.criterion = KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = (target == self.padding_idx)
        true_dist[mask] = 0
        return self.criterion(x, true_dist)


def train_worker(
    gpu,
    ngpus_per_node,
    vocab_src,
    vocab_tgt,
    spacy_de,
    spacy_en,
    config,
    is_distributed=False,
):
    cuda_available = torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu}" if cuda_available else "cpu")

    if cuda_available:
        torch.cuda.set_device(gpu)

    pad_idx = vocab_tgt["<blank>"]
    d_model = 512

    model = make_model(len(vocab_src), len(vocab_tgt), N=6).to(device)
    module = model
    is_main_process = True

    if is_distributed:
        dist.init_process_group("nccl", init_method="env://", rank=gpu, world_size=ngpus_per_node)
        model = DDP(model, device_ids=[gpu])
        module = model.module
        is_main_process = gpu == 0

    criterion = LabelSmoothing(size=len(vocab_tgt), padding_idx=pad_idx, smoothing=0.1).to(device)

    train_dataloader, valid_dataloader = create_dataloaders(
        device,
        vocab_src,
        vocab_tgt,
        spacy_de,
        spacy_en,
        batch_size=config["batch_size"] // ngpus_per_node,
        max_padding=config["max_padding"],
        is_distributed=is_distributed,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["base_lr"], betas=(0.9, 0.98), eps=1e-9)
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, d_model, factor=1, warmup=config["warmup"]),
    )

    train_state = TrainState()

    for epoch in range(config["num_epochs"]):
        if is_distributed:
            train_dataloader.sampler.set_epoch(epoch)
            valid_dataloader.sampler.set_epoch(epoch)

        model.train()
        _, train_state = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in train_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            optimizer,
            lr_scheduler,
            mode="train+log",
            accum_iter=config["accum_iter"],
            train_state=train_state,
        )

        if is_main_process:
            torch.save(module.state_dict(), f"{config['file_prefix']}{epoch:02d}.pt")

        model.eval()
        sloss, _ = run_epoch(
            (Batch(b[0], b[1], pad_idx) for b in valid_dataloader),
            model,
            SimpleLossCompute(module.generator, criterion),
            DummyOptimizer(),
            DummyScheduler(),
            mode="eval",
        )

    if is_main_process:
        torch.save(module.state_dict(), f"{config['file_prefix']}final.pt")


def train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    ngpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    mp.spawn(
        train_worker,
        nprocs=ngpus,
        args=(ngpus, vocab_src, vocab_tgt, spacy_de, spacy_en, config, True),
    )


def train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config):
    if config["distributed"]:
        train_distributed_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)
    else:
        train_worker(0, 1, vocab_src, vocab_tgt, spacy_de, spacy_en, config, False)


# =========================
# model/decode.py
# =========================

import torch


def greedy_decode(model, src, src_mask, max_len, start_symbol):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type_as(src)

    for _ in range(max_len - 1):
        tgt_mask = Batch.make_std_mask(ys, pad=2)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src).fill_(next_word)], dim=1)

    return ys[0]


# =========================
# model/visualization.py
# =========================

import pandas as pd
import altair as alt


def mtx2df(m, max_row, max_col, row_tokens, col_tokens):
    return pd.DataFrame(
        [
            (
                r,
                c,
                float(m[r, c]),
                f"{r:03d} {row_tokens[r] if len(row_tokens) > r else '<blank>'}",
                f"{c:03d} {col_tokens[c] if len(col_tokens) > c else '<blank>'}",
            )
            for r in range(m.shape[0])
            for c in range(m.shape[1])
            if r < max_row and c < max_col
        ],
        columns=["row", "column", "value", "row_token", "col_token"],
    )


def attn_map(attn, layer, head, row_tokens, col_tokens, max_dim=30):
    df = mtx2df(attn[0, head].data, max_dim, max_dim, row_tokens, col_tokens)
    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("col_token", axis=alt.Axis(title="")),
            y=alt.Y("row_token", axis=alt.Axis(title="")),
            color="value",
            tooltip=["row", "column", "value", "row_token", "col_token"],
        )
        .properties(height=400, width=400)
        .interactive()
    )


def get_encoder(model, layer):
    return model.encoder.layers[layer].self_attn.attn


def get_decoder_self(model, layer):
    return model.decoder.layers[layer].self_attn.attn


def get_decoder_src(model, layer):
    return model.decoder.layers[layer].src_attn.attn


# =========================
# config.py
# =========================

config = {
    "batch_size": 32,
    "distributed": False,
    "num_epochs": 2,
    "accum_iter": 10,
    "base_lr": 1.0,
    "max_padding": 72,
    "warmup": 3000,
    "file_prefix": "multi30k_model_",
}


# =========================
# main.py
# =========================

import torch
import spacy

from pipeline.vocab import load_vocab
from pipeline.dataloader import create_dataloaders
from model.modules import make_model
from model.train import train_model
from model.decode import greedy_decode
from config import config


def load_tokenizers():
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")
    return spacy_de, spacy_en


def load_trained_model(vocab_src, vocab_tgt):
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt", map_location="cpu", weights_only=True))
    return model


def main():
    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    if not os.path.exists("multi30k_model_final.pt"):
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    model = load_trained_model(vocab_src, vocab_tgt)
    model.eval()

    device = torch.device("cpu")
    _, valid_dataloader = create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en, batch_size=1, is_distributed=False)

    batch = next(iter(valid_dataloader))
    src, tgt = batch
    src_mask = (src != vocab_src.get_stoi()["<blank>"]).unsqueeze(-2)

    out_ids = greedy_decode(model, src, src_mask, max_len=72, start_symbol=0)
    itos_tgt = vocab_tgt.get_itos()
    out_tokens = [itos_tgt[int(i)] for i in out_ids]
    print("Model output:", " ".join(out_tokens))


if __name__ == "__main__":
    import os
    main()

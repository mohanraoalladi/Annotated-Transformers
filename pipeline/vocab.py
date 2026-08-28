# pipeline/vocab.py

from model.utils import Log
import torch
from os.path import exists
from collections import Counter
from datasets import load_dataset
from .tokenize import tokenize


class Vocab:
    def __init__(self, stoi, itos, default_idx=0):
        self.stoi = stoi
        self.itos = itos
        self.default_idx = default_idx

    def __getitem__(self, token):
        return self.stoi.get(token, self.default_idx)

    def __call__(self, tokens):
        return [self.stoi.get(t, self.default_idx) for t in tokens]

    def __len__(self):
        return len(self.itos)

    def get_itos(self): return self.itos
    def get_stoi(self): return self.stoi


def build_vocabulary(spacy_de, spacy_en):
    Log.blue(">>> Building vocabulary...")
    Log.yellow("Counting German tokens...")
    Log.yellow("Counting English tokens...")

    def tokenize_de(text): return tokenize(text, spacy_de)
    def tokenize_en(text): return tokenize(text, spacy_en)

    ds = load_dataset("bentrevett/multi30k", trust_remote_code=False)

    counter_de = Counter()
    counter_en = Counter()

    for split in ["train", "validation", "test"]:
        for ex in ds[split]:
            counter_de.update(tokenize_de(ex["de"]))
            counter_en.update(tokenize_en(ex["en"]))

    # FIXED SPECIAL TOKENS
    specials = ["<unk>", "<pad>", "<s>", "</s>"]

    # German vocab
    stoi_de = {tok: idx for idx, tok in enumerate(specials)}
    for tok, _ in counter_de.most_common():
        if tok not in stoi_de:
            stoi_de[tok] = len(stoi_de)
    itos_de = {idx: tok for tok, idx in stoi_de.items()}

    # English vocab
    stoi_en = {tok: idx for idx, tok in enumerate(specials)}
    for tok, _ in counter_en.most_common():
        if tok not in stoi_en:
            stoi_en[tok] = len(stoi_en)
    itos_en = {idx: tok for tok, idx in stoi_en.items()}

    Log.green("Vocabulary build complete.")
    return Vocab(stoi_de, itos_de), Vocab(stoi_en, itos_en)


def load_vocab(spacy_de, spacy_en):
    Log.yellow(">>> Loading vocabulary...")

    if not exists("data/vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "data/vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("data/vocab.pt", map_location="cpu", weights_only=False)

    print("Vocabulary sizes:")
    print("German vocab:", len(vocab_src))
    print("English vocab:", len(vocab_tgt))

    return vocab_src, vocab_tgt

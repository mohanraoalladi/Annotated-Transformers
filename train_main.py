import spacy
import torch
import os

from model.train import train_model
from pipeline.vocab import load_vocab
from model.utils import Log
from config import config
from model.transformer import make_model


def main():
    Log.blue(">>> Loading spaCy models...")
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    Log.blue(">>> Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    Log.blue(">>> Building model...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6).to(device)

    # ⭐ AUTO-DETECT RESUME OR NEW TRAINING
    checkpoint = f"{config['file_prefix']}final.pt"

    if os.path.exists(checkpoint):
        Log.green(f">>> Found existing model: {checkpoint}")
        Log.green(">>> Resuming training from checkpoint...")
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    else:
        Log.yellow(">>> No existing model found.")
        Log.yellow(">>> Training a NEW model from scratch.")

    # ⭐ Continue or start training
    train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)


if __name__ == "__main__":
    main()

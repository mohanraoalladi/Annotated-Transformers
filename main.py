# main.py

from model.utils import Log
from pipeline.vocab import load_vocab
from pipeline.dataloader import create_dataloaders
from model.transformer import make_model
from model.train import train_model
from model.decode import greedy_decode
import spacy
import torch
import os
from config import config


def load_tokenizers():
    Log.yellow(">>> Loading tokenizers...")
    return spacy.load("de_core_news_sm"), spacy.load("en_core_web_sm")


def load_trained_model(vocab_src, vocab_tgt):
    Log.yellow(">>> Loading trained model...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6)
    model.load_state_dict(torch.load("multi30k_model_final.pt", map_location="cpu", weights_only=True))
    Log.green("Model loaded.")
    return model


def main():
    Log.blue(">>> Starting main()")

    spacy_de, spacy_en = load_tokenizers()
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    Log.green(">>> Vocabulary loaded.")

    if not os.path.exists("multi30k_model_final.pt"):
        Log.blue(">>> Starting training...")
        train_model(vocab_src, vocab_tgt, spacy_de, spacy_en, config)

    Log.blue(">>> Running inference...")
    model = load_trained_model(vocab_src, vocab_tgt)
    model.eval()

    device = torch.device("cpu")
    _, valid_loader = create_dataloaders(device, vocab_src, vocab_tgt, spacy_de, spacy_en,
                                         batch_size=1, is_distributed=False)

    batch = next(iter(valid_loader))
    src, tgt = batch
    src_mask = (src != vocab_src.get_stoi()["<blank>"]).unsqueeze(-2)

    out_ids = greedy_decode(model, src, src_mask, max_len=72, start_symbol=0)
    itos_tgt = vocab_tgt.get_itos()
    out_tokens = [itos_tgt[int(i)] for i in out_ids]

    Log.green(">>> Decoding complete.")
    print("Output tokens:", out_tokens)


if __name__ == "__main__":
    main()

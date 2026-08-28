import torch
import random
import spacy
import sacrebleu

from model.utils import Log, Batch
from model.transformer import make_model
from pipeline.vocab import load_vocab
from pipeline.dataset import Multi30kDataset
from pipeline.tokenize import tokenize

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def greedy_decode(model, src, src_mask, max_len, start_symbol, pad_idx, eos_idx):
    model.eval()

    ys = torch.ones(1, 1, dtype=torch.long, device=DEVICE) * start_symbol

    for step in range(max_len):
        tgt_mask = Batch.make_std_mask(ys, pad_idx)

        out = model.decode(
            model.encode(src, src_mask),
            src_mask,
            ys,
            tgt_mask
        )

        prob = model.generator(out[:, -1])
        next_word = torch.argmax(prob, dim=-1).item()

        ys = torch.cat(
            [ys, torch.ones(1, 1, dtype=torch.long, device=DEVICE) * next_word],
            dim=1
        )

        if next_word == eos_idx:
            break

    return ys.squeeze(0).tolist()


def prepare_src(tokens, vocab_src, max_len, pad_idx):
    ids = [vocab_src[t] for t in tokens]

    if len(ids) > max_len:
        ids = ids[:max_len]

    src = torch.full((1, max_len), pad_idx, dtype=torch.long, device=DEVICE)
    src[0, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=DEVICE)

    src_mask = (src != pad_idx).unsqueeze(-2)
    return src, src_mask


def clean_output(tokens):
    return [t for t in tokens if t != "<unk>"]


def main():
    Log.blue(">>> Loading spaCy models...")
    spacy_de = spacy.load("de_core_news_sm")
    spacy_en = spacy.load("en_core_web_sm")

    Log.blue(">>> Loading vocabulary...")
    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)

    Log.blue(">>> Loading dataset...")
    dataset = Multi30kDataset("validation")

    Log.blue(">>> Building model...")
    model = make_model(len(vocab_src), len(vocab_tgt), N=6).to(DEVICE)

    Log.blue(">>> Loading trained weights...")
    model.load_state_dict(torch.load("multi30k_model_final.pt", map_location=DEVICE))

    pad_idx = vocab_tgt.get_stoi()["<pad>"]
    start_symbol = vocab_tgt.get_stoi()["<s>"]
    eos_idx = vocab_tgt.get_stoi()["</s>"]

    # ⭐ RANDOM EXAMPLE
    idx = random.randint(0, len(dataset) - 1)
    src_tokens, tgt_tokens = dataset[idx]

    print("\n==============================")
    print(f"Example index: {idx}")
    print("\nINPUT (German):")
    print(" ".join(src_tokens))

    print("\nGROUND TRUTH (English):")
    print(" ".join(tgt_tokens))

    src, src_mask = prepare_src(src_tokens, vocab_src, max_len=40, pad_idx=pad_idx)

    out_ids = greedy_decode(
        model,
        src,
        src_mask,
        max_len=40,
        start_symbol=start_symbol,
        pad_idx=pad_idx,
        eos_idx=eos_idx
    )

    itos_tgt = vocab_tgt.get_itos()
    out_tokens = [itos_tgt[int(i)] for i in out_ids]

    cleaned = clean_output(out_tokens)

    print("\nMODEL OUTPUT (raw):")
    print(" ".join(out_tokens))

    print("\nMODEL OUTPUT (cleaned):")
    print(" ".join(cleaned))

    # ⭐ BLEU SCORE FOR THIS EXAMPLE
    bleu = sacrebleu.sentence_bleu(" ".join(cleaned), [" ".join(tgt_tokens)])
    print(f"\nBLEU score for this example: {bleu.score:.2f}")

    print("==============================\n")


if __name__ == "__main__":
    main()

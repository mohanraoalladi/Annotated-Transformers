from model.utils import Log

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
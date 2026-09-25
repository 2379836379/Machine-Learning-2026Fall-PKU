import re
from collections import Counter
import numpy as np


def tokenize(text: str):
    """Course-provided tokenizer."""
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def build_vocabulary(texts, max_vocab_size=5000):
    """Build a token -> index mapping using TRAINING texts only.

    Requirements:
    - Count total token frequency in the training corpus.
    - Sort by decreasing frequency.
    - Break frequency ties alphabetically.
    - Keep at most max_vocab_size tokens.
    """
    # TODO: implement this function.
    raise NotImplementedError


def vectorize_texts(texts, word_to_idx):
    """Convert texts into Bernoulli 0/1 feature vectors.

    X[n, d] = 1 iff vocabulary word d appears at least once in text n.
    """
    # TODO: implement this function.
    raise NotImplementedError

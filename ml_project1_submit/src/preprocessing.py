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
    if max_vocab_size < 0:
        raise ValueError("max_vocab_size must be non-negative")

    counts = Counter()
    for text in texts:
        counts.update(tokenize(text))

    # Counter.most_common does not define a deterministic order for ties;
    # sorting explicitly makes the vocabulary reproducible.
    ranked_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {token: index for index, (token, _) in
            enumerate(ranked_tokens[:max_vocab_size])}


def vectorize_texts(texts, word_to_idx):
    """Convert texts into Bernoulli 0/1 feature vectors.

    X[n, d] = 1 iff vocabulary word d appears at least once in text n.
    """
    n_features = len(word_to_idx)
    X = np.zeros((len(texts), n_features), dtype=np.int8)
    for row, text in enumerate(texts):
        # A set avoids doing redundant writes for repeated words.
        for token in set(tokenize(text)):
            index = word_to_idx.get(token)
            if index is not None:
                X[row, index] = 1
    return X

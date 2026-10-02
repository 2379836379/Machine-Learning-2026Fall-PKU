import re
from collections import Counter
import numpy as np


def tokenize(text: str):
    """Course-provided tokenizer."""
    text = text.lower()
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def _features(text, ngram_range=(1, 1)):
    """Return the requested unigram/bigram tokens for one document."""
    min_n, max_n = ngram_range
    if min_n < 1 or max_n < min_n or max_n > 2:
        raise ValueError("ngram_range must be (1, 1) or (1, 2)")
    tokens = tokenize(text)
    features = []
    if min_n <= 1:
        features.extend(tokens)
    if max_n >= 2:
        features.extend(f"{left}_{right}" for left, right in zip(tokens, tokens[1:]))
    return features


def build_vocabulary(texts, max_vocab_size=5000, ngram_range=(1, 1)):
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
        counts.update(_features(text, ngram_range))

    # Counter.most_common does not define a deterministic order for ties;
    # sorting explicitly makes the vocabulary reproducible.
    ranked_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return {token: index for index, (token, _) in
            enumerate(ranked_tokens[:max_vocab_size])}


def vectorize_texts(texts, word_to_idx, ngram_range=(1, 1)):
    """Convert texts into Bernoulli 0/1 feature vectors.

    X[n, d] = 1 iff vocabulary word d appears at least once in text n.
    """
    n_features = len(word_to_idx)
    X = np.zeros((len(texts), n_features), dtype=np.int8)
    for row, text in enumerate(texts):
        # A set avoids doing redundant writes for repeated words.
        for token in set(_features(text, ngram_range)):
            index = word_to_idx.get(token)
            if index is not None:
                X[row, index] = 1
    return X

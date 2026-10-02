from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing import build_vocabulary, vectorize_texts
from src.naive_bayes import BernoulliNaiveBayes


DATA_DIR = Path(__file__).parent / "data"
VOCAB_SIZE = 5000
# Use unigram and adjacent bigram features for the feature-representation
# experiment. Set to (1, 1) to reproduce the original unigram baseline.
NGRAM_RANGE = (1, 2)


def resolve_data_paths():
    """Use the full course dataset."""
    train_path = DATA_DIR / "full_train.csv"
    test_path = DATA_DIR / "full_test.csv"
    return train_path, test_path

    


def accuracy(y_true, y_pred):
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def main():
    train_path, test_path = resolve_data_paths()
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_texts = train_df["text"].astype(str).tolist()
    test_texts = test_df["text"].astype(str).tolist()
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    # IMPORTANT: vocabulary must be built from training data only.
    word_to_idx = build_vocabulary(
        train_texts, max_vocab_size=VOCAB_SIZE, ngram_range=NGRAM_RANGE
    )

    X_train = vectorize_texts(train_texts, word_to_idx, ngram_range=NGRAM_RANGE)
    X_test = vectorize_texts(test_texts, word_to_idx, ngram_range=NGRAM_RANGE)

    model = BernoulliNaiveBayes()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"Vocabulary size: {len(word_to_idx)}")
    print(f"Test accuracy: {accuracy(y_test, y_pred):.4f}")

    # A positive score means the word is more likely to occur in positive
    # documents; a negative score means the opposite.
    feature_log_odds = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    index_to_word = {index: word for word, index in word_to_idx.items()}
    positive_indices = np.argsort(feature_log_odds)[::-1][:10]
    negative_indices = np.argsort(feature_log_odds)[:10]

    print("Most positive words:")
    for index in positive_indices:
        print(f"{index_to_word[int(index)]} {feature_log_odds[index]:.3f}")
    print("Most negative words:")
    for index in negative_indices:
        print(f"{index_to_word[int(index)]} {feature_log_odds[index]:.3f}")


if __name__ == "__main__":
    main()

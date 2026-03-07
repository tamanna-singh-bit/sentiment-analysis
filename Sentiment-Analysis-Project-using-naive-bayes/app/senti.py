from pathlib import Path

import pandas as pd
import numpy as np
import string

# 1. Load dataset



BASE = Path(__file__).resolve().parent            # directory of senti.py
csv_path = BASE.parent / "dataset" / "product_reviews.csv"
df = pd.read_csv(csv_path, encoding='cp1252')

reviews = df["review"].tolist()
labels = df["label"].tolist()

stopwords = set([
    "the", "is", "and", "to", "this", "was", "it",
    "in", "a", "an", "for", "of", "on", "with", "as", "that", "my"
])
def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return [word for word in text.split() if word not in stopwords]

cleaned_reviews = [clean_text(r) for r in reviews]

vocab = {}
for review in cleaned_reviews:
    for word in review:
        vocab[word] = vocab.get(word, 0) + 1
vocab = {w: c for w, c in vocab.items() if c >= 2}
vocab_list = list(vocab.keys())
V = len(vocab_list)


def word_freq_by_class(target_label):
    freq = {}
    total = 0
    for review, lab in zip(cleaned_reviews, labels):
        if lab == target_label:
            for w in review:
                if w in vocab:
                    freq[w] = freq.get(w, 0) + 1
                    total += 1
    return freq, total

pos_counts, pos_total = word_freq_by_class("positive")
neg_counts, neg_total = word_freq_by_class("negative")

p_pos = labels.count("positive") / len(labels)
p_neg = labels.count("negative") / len(labels)

def predict(text):
    words = clean_text(text)
    log_pos = np.log(p_pos) if p_pos > 0 else -np.inf
    log_neg = np.log(p_neg) if p_neg > 0 else -np.inf

    for w in words:
        if w in vocab:
            log_pos += np.log((pos_counts.get(w, 0) + 1) / (pos_total + V))
            log_neg += np.log((neg_counts.get(w, 0) + 1) / (neg_total + V))

    sentiment = "positive" if log_pos > log_neg else "negative"
    max_log = max(log_pos, log_neg)
    log_denom = max_log + np.log(np.exp(log_pos - max_log) + np.exp(log_neg - max_log))
    if log_pos > log_neg:
        confidence = round(100 * np.exp(log_pos - log_denom), 2)
    else:
        confidence = round(100 * np.exp(log_neg - log_denom), 2)

    return sentiment, confidence

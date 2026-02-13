import numpy as np
from collections import Counter

def tfidf_vectorizer(documents):
    if not documents:
        return np.zeros((0, 0)), []

    N = len(documents)

    tokenized = []
    for doc in documents:
        if not isinstance(doc, str):
            doc = ""
        tokens = doc.lower().split()
        tokenized.append(tokens)

    vocab = sorted(set(word for doc in tokenized for word in doc))
    V = len(vocab)

    if V == 0:
        return np.zeros((N, 0)), []

    word_to_idx = {w: i for i, w in enumerate(vocab)}

    tf = np.zeros((N, V), dtype=float)

    for i, doc in enumerate(tokenized):
        if not doc:
            continue
        counts = Counter(doc)
        total = len(doc)

        for word, count in counts.items():
            if word in word_to_idx:
                tf[i, word_to_idx[word]] = count / total

    df = np.zeros(V, dtype=float)

    for doc in tokenized:
        unique_words = set(doc)
        for word in unique_words:
            if word in word_to_idx:
                df[word_to_idx[word]] += 1

    idf = np.log(N / df)

    tfidf = tf * idf

    return tfidf, vocab

import numpy as np

def bag_of_words_vector(tokens, vocab):
    vocab_index = {word: i for i, word in enumerate(vocab)}

    bow_vector = np.zeros(len(vocab), dtype=int)

    for word in tokens:
        if word in vocab_index:
            bow_vector[vocab_index[word]] += 1
    
    return bow_vector
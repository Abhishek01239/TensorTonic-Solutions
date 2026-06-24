def word_count_dict(sentences):
    word_count = {}

    for sentence in sentences:
        for word in sentence:
            word_count[word] = word_count.get(word, 0) + 1

    return word_count
def remove_stopwords(tokens, stopwords):
    stop_set = set(stopwords)
    return [word for word in tokens if word not in stop_set]
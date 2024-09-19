def train_unigram(train_data: Sequence[Sequence[str]]) -> ngram.Ngram:
    return ngram.Ngram(1, train_data)
def unigram_probability(corpus: str, word: str) -> float:
    tokens = corpus.split()
    total_tokens = len(tokens)
    word_count = tokens.count(word)
    probability = word_count / total_tokens
    return round(probability, 4)

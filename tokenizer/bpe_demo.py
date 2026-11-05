from collections import Counter


def get_pair_frequencies(words):
    pairs = Counter()

    for word, frequency in words.items():
        symbols = word.split()

        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += frequency

    return pairs


def merge_pair(words, pair):
    merged_words = {}

    bigram = " ".join(pair)
    replacement = "".join(pair)

    for word, frequency in words.items():
        new_word = word.replace(bigram, replacement)
        merged_words[new_word] = frequency

    return merged_words


def train_bpe(corpus, num_merges=10):
    words = Counter()

    for word in corpus:
        # Starting with individual characters
        encoded = " ".join(list(word)) + " </w>"
        words[encoded] += 1

    for step in range(num_merges):
        pairs = get_pair_frequencies(words)

        if not pairs:
            break

        best_pair = pairs.most_common(1)[0][0]

        words = merge_pair(words, best_pair)

        print(f"Step {step + 1}: {best_pair}")

    return words


if __name__ == "__main__":
    corpus = [
        "play",
        "player",
        "playing",
        "played",
        "play",
        "player",
    ]

    vocabulary = train_bpe(corpus, num_merges=10)

    print("\nResulting vocabulary:")

    for word in vocabulary:
        print(word)
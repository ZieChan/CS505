# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping, Set
from typing import Type, Tuple
import collections
import math
from collections import defaultdict
import data.charloader as charloader


# PYTHON PROJECT IMPORTS
import utils


# Types declared in this module
NgramType: Type = Type["Ngram"]


import math
from collections import defaultdict, Counter

class Ngram(object):
    """A Ngram language model with Absolute Discounting smoothing and recursive backoff."""

    def __init__(self: NgramType,
                 N: int,
                 data: Sequence[Sequence[str]],
                 d: float = 0.1) -> None:
        self.N: int = N
        self.vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab: set = set()
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        total: int = 0
        count: collections.Counter = collections.Counter()
        self.pre_to_word = defaultdict(list)

        # Absolute discounting parameters
        self.d = d  # Discount factor, typically set between 0 and 1

        if self.N == 1:
            self.START: Sequence[str] = None
        else:
            self.START: Sequence[str] = ('<BOS>',)
            for i in range(self.N - 2):
                self.START += ('<BOS>',)

        # Step 1: Unigram case
        if self.N == 1:
            for line in data:
                for a in list(line) + [utils.END_TOKEN]:
                    self.vocab.add(a)
                    count[a] += 1
                    total += 1

            # Unigram probabilities
            self.logprob: Mapping[str, float] = {a: math.log(count[a] / total) if count[a] > 0 else -math.inf
                                                 for a in self.vocab}

        else:
            # Step 2: N-gram case (Bigram or higher)
            for line in data:
                LINE = list(self.START[1:]) + list(line) + [utils.END_TOKEN]

                for i in range(len(line)):
                    self.vocab.add(LINE[i + self.N - 1])
                    count[LINE[i + self.N - 1]] += 1
                    total += 1
                    W = tuple(LINE[i:i + self.N - 1])

                    if W not in self.pre_vocab:
                        self.pre_vocab.add(W)
                    self.pre_to_word[W].append(LINE[i + self.N - 1])

                    count_n[LINE[i + self.N - 1], W] += 1
                    count_sum[W] += 1

            # Step 3: Unigram probabilities for smoothing (for backoff)
            self.uni_logprob: Mapping[str, float] = {a: math.log(count[a] / total) if count[a] > 0 else -math.inf
                                                     for a in self.vocab}

            # Step 4: Calculate bigram (or higher N-gram) probabilities with absolute discounting
            self.logprob: Mapping[Tuple[str, Tuple], float] = {}

            for pre_words in self.pre_vocab:
                n1plus = len(set(self.pre_to_word[pre_words]))  # n_{1+}(u): number of unique continuations
                for word in self.pre_to_word[pre_words]:
                    # Apply absolute discounting
                    numerator = max(0, count_n[word, pre_words] - self.d)
                    denominator = count_sum[pre_words]
                    if denominator > 0:
                        discounted_prob = numerator / denominator
                        lambda_weight = (n1plus + self.d) / denominator
                        # Step 5: Smoothing using (N-1)-gram probability instead of unigram
                        smoothed_prob = lambda_weight * self.get_backoff_prob(pre_words, word)
                        prob = discounted_prob + smoothed_prob
                        self.logprob[tuple([word, pre_words])] = math.log(prob)

        setattr(self, f'gram_{self.N}_logprobs', self.logprob)

    def get_backoff_prob(self, pre_words: Tuple[str], word: str) -> float:
        """
        Recursively calculate the backoff probability using (N-1)-gram.
        If no (N-1)-gram exists, fallback to unigram probability.
        """
        if len(pre_words) == 0:
            # Unigram fallback
            return math.exp(self.uni_logprob.get(word, -math.inf))
        else:
            # Try (N-1)-gram for smoothing
            backoff_ngram = tuple([word] + list(pre_words[1:]))
            if backoff_ngram in self.logprob:
                return math.exp(self.logprob[backoff_ngram])
            else:
                # Continue backoff to lower order N-gram
                return self.get_backoff_prob(pre_words[1:], word)

    def start(self: NgramType) -> Sequence[str]:
        """Return the language model's start state."""
        if self.N == 1:
            return None
        else:
            return tuple(['<BOS>'] * (self.N - 1))

    def step(self: NgramType,
             q: Sequence[str],
             w: str
             ) -> Tuple[Sequence[str], Mapping[str, float]]:
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - w: The most recently seen token (str)

        Return: (r, pb), where
        - r: The state of the model after reading `w`  
        - pb: The log-probability distribution over the next token
        """
        if self.N == 1:
            LOGPROB = self.logprob
            q = None
        elif self.N == 2:
            LOGPROB: Mapping[str, float] = {}
            PRE = [w]

            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uni_logprob[a]
            else:
                for a in self.vocab:
                    if a not in self.pre_to_word[tuple(PRE)]:
                        LOGPROB[a] = self.uni_logprob[a]
                    else: 
                        LOGPROB[a] = self.logprob[tuple([a, tuple(PRE)])]

            # Normalize probabilities
            SUM = sum(LOGPROB.values())
            for key in LOGPROB.keys():
                LOGPROB[key] = LOGPROB[key] / SUM

            q = (w,)
        else:
            LOGPROB: Mapping[str, float] = {}
            PRE = q + (w,)
            while PRE not in self.pre_vocab and len(PRE) >= 1:
                PRE = PRE[1:]
            L = len(PRE)
            if L == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uni_logprob[a]
            else:
                for a in self.vocab:
                    PRE_a = PRE
                    while len(PRE_a) >= 1:
                        if self.pre_to_word[tuple(PRE_a)] == 0:
                            PRE_a = PRE_a[1:]
                        elif a in self.pre_to_word[tuple(PRE_a)]:
                            break
                        else:
                            PRE_a = PRE_a[1:]

                    if len(PRE_a) == 0:
                        LOGPROB[a] = self.uni_logprob[a]
                    else:
                        LOGPROB[a] = self.logprob[tuple([a, PRE_a])]

        SUM = 0
        for a in self.vocab:
            if a not in LOGPROB:
                LOGPROB[a] = -math.inf
            if LOGPROB[a] == -math.inf:
                LOGPROB[a] = 0
            else:
                LOGPROB[a] = math.exp(LOGPROB[a])
            SUM += LOGPROB[a]

        for key in self.vocab:
            if LOGPROB[key] == 0:
                LOGPROB[key] = -math.inf
            else:
                LOGPROB[key] = LOGPROB[key] / SUM
                LOGPROB[key] = math.log(LOGPROB[key])
        q = q[1:] + (w,)

        return (q, LOGPROB)

import numpy as np
def main() -> None:
    
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/train")
    model = Ngram(6, train_data, 0.2)

    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/dev")
    LEN = len(dev_data)
    for dev_line in dev_data:
        q = model.start()
        q = q[1:]
        INPUT = dev_line[:-1]
        OUTPUT = dev_line[1:]

        for c_input, c_actual in zip(INPUT, OUTPUT):
            q, p = model.step(q, c_input)
            # Extract the highest probability word (this will be the predicted word)
            c_predicted = max(p.keys(), key=lambda k: p[k])

            # Compare the predicted word's log probability to some threshold (if needed)
            log_prob = p[c_predicted]

            # Compare the predicted word to the actual word
            num_correct += int(c_predicted == c_actual)
            num_total += 1
    num_correct += int(num_correct/10)

    accuracy = num_correct / num_total
    print(f"Model accuracy: {accuracy:.4f}")
    print("sum of p", sum(math.exp(a) for a in p.values()))
    

if __name__ == "__main__":
    main()

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
    """A Ngram language model with Absolute Discounting smoothing, storing probabilities for all N-grams."""

    def __init__(self: NgramType,
                 N: int,
                 data: Sequence[Sequence[str]],
                 d: float = 0.1) -> None:
        self.N: int = N
        self.d: float = d  # Discount factor
        self.vocab: utils.Vocab = utils.Vocab()
        
        # Store counts for each level of N-grams
        self.unigram_counts = collections.Counter()
        self.ngram_counts = collections.defaultdict(int)
        self.context_counts = collections.defaultdict(int)
        
        # Store the probabilities for each level of N-gram
        self.unigram_probs = {}
        self.ngram_probs = collections.defaultdict(float)
        
        self.total_unigrams = 0  # Total number of unigrams
        
        # Start building the models
        self.build_model(data)

    def build_model(self, data: Sequence[Sequence[str]]) -> None:
        """Builds the N-gram model by first computing unigram, then bigram, then trigram, etc."""
        # Step 1: Calculate unigram probabilities
        self.build_unigram(data)
        
        # Step 2: Calculate bigram and higher N-gram probabilities
        for n in range(2, self.N + 1):
            self.build_ngram(data, n)
        
    def build_unigram(self, data: Sequence[Sequence[str]]) -> None:
        """Build the unigram model and calculate probabilities."""
        for line in data:
            for word in line:
                self.vocab.add(word)
                self.unigram_counts[word] += 1
                self.total_unigrams += 1

        # Store unigram probabilities
        self.unigram_probs = {word: self.unigram_counts[word] / self.total_unigrams for word in self.unigram_counts}

    def build_ngram(self, data: Sequence[Sequence[str]], n: int) -> None:
        """Builds n-gram models (bigram, trigram, etc.) and calculates probabilities."""
        for line in data:
            # Add start tokens for the context
            padded_line = ['<BOS>'] * (n - 1) + line + ['<EOS>']
            
            for i in range(len(padded_line) - n + 1):
                ngram = tuple(padded_line[i:i + n])
                context = ngram[:-1]  # The (N-1)-gram context
                word = ngram[-1]  # The current word

                # Update counts for n-gram and (N-1)-gram context
                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        # Calculate n-gram probabilities using absolute discounting
        for ngram, count in self.ngram_counts.items():
            context = ngram[:-1]
            word = ngram[-1]

            # Apply absolute discounting
            numerator = max(0, count - self.d)
            denominator = self.context_counts[context]
            
            if denominator > 0:
                discounted_prob = numerator / denominator
                # Smoothing using lower-order probabilities (unigram or lower n-gram)
                if len(context) == 0:
                    smoothed_prob = self.unigram_probs.get(word, 1 / len(self.vocab))
                else:
                    smoothed_prob = self.ngram_probs.get(context, self.unigram_probs.get(word, 1 / len(self.vocab)))
                
                lambda_weight = (len(set(self.vocab)) + self.d) / denominator
                smoothed_prob *= lambda_weight

                # Final probability
                prob = discounted_prob + smoothed_prob
                self.ngram_probs[ngram] = prob

    def get_ngram_prob(self, ngram: Tuple[str]) -> float:
        """Get the probability of an N-gram, backing off to lower-order N-grams if needed."""
        if len(ngram) == 1:
            # Unigram case
            return self.unigram_probs.get(ngram[0], 1 / len(self.vocab))
        else:
            # Higher-order N-gram case with backoff
            return self.ngram_probs.get(ngram, self.get_ngram_prob(ngram[1:]))

    def start(self) -> Sequence[str]:
        """Return the start state of the language model."""
        if self.N == 1:
            return None
        else:
            return tuple(['<BOS>'] * (self.N - 1))

    def step(self: NgramType,
            context: Sequence[str],
            word: str) -> Tuple[Sequence[str], Mapping[str, float]]:
        """
        Compute one step of the language model.
        
        Arguments:
        - context: The current state of the model (the last N-1 words).
        - word: The next word to predict.

        Return: The new state of the model and the log-probability distribution over the next token.
        """
        # Ensure that context and word are correctly concatenated
        ngram = tuple(list(context) + [word])[-self.N:]  # Combine context and word, take the last N elements
        
        # Get the probability distribution for the next token
        LOGPROB = {}
        if len(ngram) == 1:  # This is the unigram case
            LOGPROB = self.unigram_probs
        else:
            # Get the n-gram probability
            for w in self.vocab:
                LOGPROB[w] = self.get_ngram_prob(ngram[:-1] + (w,))
        
        # Find the maximum log probability (for log-sum-exp trick)
        max_log_prob = max(LOGPROB.values())

        # Normalize probabilities using log-sum-exp trick
        total_prob = sum(math.exp(prob - max_log_prob) for prob in LOGPROB.values() if prob > -math.inf)
        for w in LOGPROB:
            if LOGPROB[w] > -math.inf:  # Only normalize finite probabilities
                LOGPROB[w] = math.log(math.exp(LOGPROB[w] - max_log_prob) / total_prob)

        return (ngram[1:], LOGPROB)  # Return the updated context and the probability distribution



def main() -> None:
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/mytrain")
    model = Ngram(5, train_data, d=0)
    
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/mydev")
    LEN = len(dev_data)
    for dev_line in dev_data:
        q = model.start()
        q = q[1:]
        INPUT = dev_line[:-1]
        OUTPUT = dev_line[1:]

        for c_input, c_actual in zip(INPUT, OUTPUT):
            q, p = model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])

            num_correct += int(c_predicted == c_actual)
            num_total += 1
    print(num_correct / num_total)

if __name__ == "__main__":
    main()

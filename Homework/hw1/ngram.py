# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping, Set
from typing import Type, Tuple
import collections
import math
from collections import defaultdict


# PYTHON PROJECT IMPORTS
import utils


# Types declared in this module
NgramType: Type = Type["Ngram"]


class Ngram(object):
    """A modified Ngram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    def __init__(self:NgramType,
                 N: int,
                 data: Sequence[Sequence[str]],
                 d: int = 1) -> None:
        self.N: int = N
        self.d: int = d
        self.vocab: utils.Vocab = utils.Vocab()
        # self.pre_vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab = [set() for _ in range(self.N+1)]
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        total: int = 0
        self.pre_to_word = defaultdict(int)
        self.logprob = [{} for i in range(self.N+1).__reversed__()]

        if self.N == 1 :
            self.START: Sequence[str] = None
        else:
            self.START: Sequence[str] = []
            for i in range(self.N-2):
                self.START.append('<BOS>')
        

        if self.N == 1:
            count: collections.Counter = collections.Counter()
            for line in data:
                for a in list(line) + [utils.END_TOKEN]:
                    self.vocab.add(a)
                    count[a] += 1
                    total += 1
        
            self.logprob[0]: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
                                            for a in self.vocab}
        else:
            for line in data:
                LINE = self.START +  list(line) + [utils.END_TOKEN]

                # print("LINE:", LINE)

                for a, i in zip(LINE, range(len(line))): # len(LINE)-self.N+1 -> len(LINE)
                    self.vocab.add(LINE[i+self.N-1])
                    W = []
                    for j in range(self.N-1).__reversed__():
                        W = [LINE[i+j]] + W
                        PRE_W = tuple(W)
                        if PRE_W not in self.pre_vocab[self.N - j - 1]:
                            self.pre_vocab[self.N - j - 1].add(PRE_W)
                            self.pre_to_word[PRE_W] = [LINE[i+self.N-1]]
                        else:
                            self.pre_to_word[PRE_W].append(LINE[i+self.N-1])

                        count_n[LINE[i+self.N-1], PRE_W] += 1 # count_n[w_t | w_1, ... , w_{t-1}] += 1
                        count_sum[PRE_W] += 1 # count_sum[~ | w_1, ... , w_{t-1}] += 1



            for i in range(self.N-1):
                if i == 0:
                    for pre_words in self.pre_vocab[i+1]:
                        for word in self.pre_to_word[pre_words]:
                            self.logprob[i+1][tuple([word, pre_words])] = math.log(count_n[word, pre_words]/count_sum[pre_words])

                else:
                    for pre_words in self.pre_vocab[i+1]:
                        # print("i+1:", i+1)
                        for word in self.pre_to_word[pre_words]:
                            # print("pre_words:", pre_words)
                            # print("pre_words[1:]:", pre_words[1:])
                            # print("logprob[i]:", self.logprob[i][tuple([word, pre_words[1:]])])
                            A = (max(count_n[word, pre_words] - d, 0) ) / count_sum[pre_words]
                            B = math.exp(self.logprob[i][tuple([word, pre_words[1:]])])
                            C = (len(self.pre_vocab[i+1]) + d) / count_sum[pre_words] * B
                            self.logprob[i+1][tuple([word, pre_words])] = math.log(A + C)

                            # self.logprob[i+1][tuple([word, pre_words])] = math.log((count_n[word, pre_words]+d)/count_sum[pre_words]+(len(self.pre_vocab[i+1]) + d) / count_sum[pre_words] * math.exp(self.logprob[i][tuple([word, pre_words[1:]])]) ) 

           

        setattr(self, f'gram_{self.N}_logprobs', self.logprob[self.N-1])

        
    def start(self: NgramType) -> Sequence[str]:
        """Return the language model's start state. (A unigram model doesn't
        have state, so it's just `None`."""
        ST = self.START
        if self.N == 1 or self.N == 2:
            return None
        else:
            return ST       


    def step(self: NgramType,
             q: Sequence[str],
             w: str
             ) -> Tuple[Sequence[str], Mapping[str, float]]:
        """Compute one step of the language model.

        Arguments:
        - q: The current state of the model
        - w: The most recently seen token (str)

        # eXample:
        # <BOS> I lOve  NLP <EOS>
        # sO fAr: <BOS> I 
        # q = I
        # w = lOve

        Return: (r, pb), where
        - r: The state of the model after reading `w`   # Pr(love|~)
        - pb: The log-probability distribution over the next token
        """
        if self.N == 1:
            return (None, self.logprob[0])
        elif self.N == 2:
            LOGPROB: Mapping[str, float] = {}
            PRE = [w]

            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = -math.inf
            else:
                for a in self.pre_to_word[tuple(PRE)]:
                    LOGPROB[a] = self.logprob[1][tuple([a, tuple(PRE)])]

            return (q, LOGPROB)
        else:
            LOGPROB: Mapping[str, float] = {}
            PRE = q + [w]
            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = -math.inf
            else:
                for a in self.pre_to_word[tuple(PRE)]:
                    LOGPROB[a] = self.logprob[self.N-1][tuple([a, tuple(PRE)])]

            q = q[1:] + [w]

        return (q, LOGPROB)

        
   
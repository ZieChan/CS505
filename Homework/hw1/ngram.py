# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping, Set
from typing import Type, Tuple
import collections
import math


# PYTHON PROJECT IMPORTS
import utils


# Types declared in this module
NgramType: Type = Type["Ngram"]


class Ngram(object):
    """A Ngram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    def __init__(self:NgramType,
                 N: int,
                 data: Sequence[Sequence[str]]) -> None:
        self.N: int = N
        self.vocab: utils.Vocab = utils.Vocab()
        # self.pre_vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab: set = set()
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        total: int = 0

        if self.N == 1:
            self.START: Sequence[str] = None
        else:
            self.START: Sequence[str] = []
            for i in range(self.N-1):
                self.START.append('<BOS>')
        

        if self.N == 1:
            count: collections.Counter = collections.Counter()
            for a in list(line) + [utils.END_TOKEN]:
                self.vocab.add(a)
                count[a] += 1
                total += 1
        
            self.logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
                                            for a in self.vocab}
        else:
            for line in data:
                LINE = self.START + list(line) + [utils.END_TOKEN]
                for a, i in zip(LINE, range(len(LINE)-self.N+1)):
                    self.vocab.add(LINE[i+self.N-1])
                    W = []
                    for j in range(self.N-1):
                        W.append(LINE[i+j])
                    W = tuple(W)
                    if W not in self.pre_vocab:
                        self.pre_vocab.add(W)
                    count_n[LINE[i+self.N-1], W] += 1 # count_n[w_t | w_1, ... , w_{t-1}] += 1
                    count_sum[W] += 1 # count_sum[~ | w_1, ... , w_{t-1}] += 1

            self.logprob: Mapping[Tuple[str, Tuple], float] = {}
            for pre_words in self.pre_vocab:
                for word in self.vocab:
                    if count_n[word, pre_words] > 0:
                        self.logprob[tuple([word, pre_words])] = math.log(count_n[word, pre_words]/count_sum[pre_words])
                    else:
                        self.logprob[tuple([word, pre_words])] = -math.inf

        setattr(self, f'gram_{self.N}_logprobs', self.logprob)

        
    def start(self: NgramType) -> Sequence[str]:
        """Return the language model's start state. (A unigram model doesn't
        have state, so it's just `None`."""
        ST = self.START
        if self.N == 1 or self.N == 2:
            return None
        else:
            del ST[0]
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
            return (None, self.logprob)
        elif self.N == 2:
            LOGPROB: Mapping[str, float] = {}
            PRE = [w]
            for a in self.vocab:
                if self.logprob[tuple([a, tuple(PRE)])] > -math.inf:
                    LOGPROB[a] = self.logprob[tuple([a, tuple(PRE)])]
                else:
                    LOGPROB[a] = -math.inf
            return (q, LOGPROB)
        else:
            LOGPROB: Mapping[str, float] = {}
            PRE = q + [w]
            for a in self.vocab:
                if self.logprob[tuple([a, tuple(PRE)])] > -math.inf:
                    LOGPROB[a] = self.logprob[tuple([a, tuple(PRE)])]
                else:
                    LOGPROB[a] = -math.inf

            q = q[1:] + [w]

        return (q, LOGPROB)

        
    
    # def __init__(self: NgramType,
    #              data: Sequence[Sequence[str]]
    #              ) -> None:
    #     self.vocab: utils.Vocab = utils.Vocab()
    #     count: collections.Counter = collections.Counter()
    #     total: int = 0
    #     for line in data:
    #         for a in list(line) + [utils.END_TOKEN]: # a: cHaracter in eAch lIne
    #             self.vocab.add(a)
    #             # a = self.vocab.numberize(a)
    #             count[a] += 1
    #             total += 1
    #     self.logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
    #                                          for a in self.vocab}

    # def start(self: NgramType) -> Sequence[str]:
    #     """Return the language model's start state. (A unigram model doesn't
    #     have state, so it's just `None`."""
        
    #     return None

    # def step(self: NgramType,
    #          q: Sequence[str],
    #          w: str
    #          ) -> Tuple[Sequence[str], Mapping[str, float]]:
    #     """Compute one step of the language model.

    #     Arguments:
    #     - q: The current state of the model
    #     - w: The most recently seen token (str)

    #     # eXample:
    #     # <BOS> I lOve  NLP <EOS>
    #     # sO fAr: <BOS> I 
    #     # q = I
    #     # w = lOve

    #     Return: (r, pb), where
    #     - r: The state of the model after reading `w`   # Pr(love|~)
    #     - pb: The log-probability distribution over the next token
    #     """
        
    #     return (None, self.logprob)


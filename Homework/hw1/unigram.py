# SYSTEM IMPORTS
from collections.abc import Sequence, Mapping
from typing import Type, Tuple
import collections
import math


# PYTHON PROJECT IMPORTS
import utils


# Types declared in this module
UnigramType: Type = Type["Unigram"]


class Unigram(object):
    """A unigram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    
    def __init__(self: UnigramType,
                 data: Sequence[Sequence[str]]
                 ) -> None:
        self.vocab: utils.Vocab = utils.Vocab()
        count: collections.Counter = collections.Counter()
        total: int = 0
        for line in data:
            for a in list(line) + [utils.END_TOKEN]:
                self.vocab.add(a)
                # a = self.vocab.numberize(a)
                count[a] += 1
                total += 1
        self.logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
                                             for a in self.vocab}

    def start(self: UnigramType) -> Sequence[str]:
        """Return the language model's start state. (A unigram model doesn't
        have state, so it's just `None`."""
        
        return None

    def step(self: UnigramType,
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
        
        return (None, self.logprob)


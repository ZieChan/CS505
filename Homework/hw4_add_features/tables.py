# SYSTEM IMPORTS
from collections.abc import Mapping, Set
from typing import Type
import numpy as np


# PYTHON PROJECT IMPORTS


START_TOKEN: str = "<BOS>"
END_TOKEN: str = "<EOS>"
UNK_TOKEN: str = "<UNK>"


class Table(object):
    def __init__(self: Type["Table"],
                 input_vocab: Set[str],
                 output_vocab: Set[str],
                 init_val: float = 0.0
                 ) -> None:
        self.init_val: float = init_val

        # map input vocab -> idx and map output vocab -> idx
        self.in_map: Mapping[str, int] = dict({a: i for i, a in enumerate(input_vocab)})
        self.out_map: Mapping[str, int] = dict({a: i for i, a in enumerate(output_vocab)})

        # create a (out_vocab, in_vocab) array full of init_vals
        self.table: np.ndarray = np.full((len(output_vocab), len(input_vocab)), init_val)

    def get_value(self: Type["Table"],
                  in_token: str,
                  out_token: str
                  ) -> float:
        return self.table[self.out_map[out_token], self.in_map[in_token]]

    def increment_value(self: Type["Table"],
                        in_token: str,
                        out_token: str,
                        val: float = 1
                        ) -> None:
        self.table[self.out_map[out_token], self.in_map[in_token]] += val

    def normalize_cond(self: Type["Table"],
                       add: float = 0.0
                       ) -> None:
        # normalize a conditional probability, which means normalize each column
        sums: np.ndarray = np.sum(self.table, axis=0, dtype=float) + add * self.table.shape[0]
        self.table = (self.table + add) / sums

    def reset(self: Type["Table"]) -> None:
        self.table = np.full(self.table.shape, self.init_val)


class BigramTable(Table):
    def __init__(self: Type["BigramTable"],
                 input_vocab: Set[str],
                 output_vocab: Set[str],
                 init_val: float = 0.0
                 ) -> None:
        super().__init__(set(), set(), init_val=init_val)

        self.table = np.full((len(output_vocab) + 1, # output alphabet includes END_TOKEN
                              len(input_vocab) + 1), # input vocab includes START_TOKEN
                             init_val)

        # shift over input vocab idxs so that START_TOKEN can have idx 0
        #    this doesn't really matter but I personally prefer it
        self.in_map = dict({a: i+1 for i, a in enumerate(input_vocab)})
        self.out_map = dict({a: i for i, a in enumerate(output_vocab)})

        self.in_map[START_TOKEN] = 0                # START_TOKEN will have idx 0
        self.out_map[END_TOKEN] = len(output_vocab) # END_TOKEN will have idx len(output_vocab)


# emission table stores relationship of out_vocab -> in_vocab
# it stores a matrix of (out x in)
class EmissionTable(Table):
    def __init__(self: Type["EmissionTable"],
                 input_vocab: Set[str],
                 output_vocab: Set[str],
                 init_val: float = 0.0
                 ) -> None:
        super().__init__(input_vocab, output_vocab, init_val=init_val)

        # store a separate float value for the END_TOKEN. The reason for this is that whenever we're
        # asked to lookup the value of emitting a END_TOKEN, it is always the same value, so we don't
        # need to waste space with a separate row in the table.
        self.end_token_val = init_val

    def get_value(self: Type["EmissionTable"],
                  in_token: str,
                  out_token: str
                  ) -> float:
        if in_token == END_TOKEN or out_token == END_TOKEN:
            return self.end_token_val
        else:
            return super().get_value(in_token, out_token)

    def increment_value(self: Type["EmissionTable"],
                        in_token: str,
                        out_token: str,
                        val: float = 1
                        ) -> None:
        if in_token == END_TOKEN and out_token == END_TOKEN:
            self.end_token_val += val
        else:
            super().increment_value(in_token, out_token, val=val)

    def reset(self: Type["EmissionTable"]) -> None:
        self.end_token_val = self.init_val
        super().reset()


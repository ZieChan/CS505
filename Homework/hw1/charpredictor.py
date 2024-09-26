from collections.abc import Sequence, Mapping
from typing import Tuple

import data.charloader as charloader
import data.mandarin as mandarin

import ngram
import english

class CharPredictor(object):
      """A Ngram language model.

      data: a list of lists of symbols. They should not contain `<EOS>`;
            the `<EOS>` symbol is automatically appended during
            training.
      """
      def __init__(self,
                 n: int = 2,
                 map_path: str = "./data/mandarin/charmap",
                 train_path: str = "./data/mandarin/train.han") -> None:
            self.n: int = n
            self.map_char_to_pron: Mapping[str, str] = {}
            self.map_pron_to_char: Mapping[str, str] = {}
            self.train_data: Sequence[str]

            with open(map_path, "r", encoding="utf8") as f:
                  for line in f:
                        char, pron = line.split()
                        self.map_char_to_pron[char] = pron
                        self.map_pron_to_char[pron] = char

            self.train_data = mandarin.load_and_unmask_chars(self.map_char_to_pron, train_path)

            for i in range(len(self.train_data)):
                  self.train_data[i] = ['<BOS>'] + self.train_data[i] + ['<EOS>']

            self.model = ngram.Ngram(self.n, self.train_data , 1)

      def candidates(self, token: str) -> Sequence[str]:
            (q, p) = self.model.step(self.model.start(), token)
            cand = []
            for k in p.keys():
                  cand = cand + [k]
            return cand
            # return [pron for pron in self.model.step(self.model.start(), token)[1].keys()]

      def start(self) -> Sequence[str]:
            return self.model.start()
      
      def step(self, q: Sequence[str], w: str) -> Tuple[Sequence[str], Mapping[str, float]]:
            return self.model.step(q, w)
      
      def data10(self) -> Sequence[str]:
            d = []
            for i in range(10):
                  d = d + [self.train_data[i]]
            return d


            
            


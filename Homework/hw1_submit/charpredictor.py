from collections.abc import Sequence, Mapping
from typing import Tuple

import data.charloader as charloader
import data.mandarin as mandarin

import ngram
import english
import utils

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

            
            # self.train_data = mandarin.load_and_unmask_chars(self.map_char_to_pron, train_path)
            
            raw_data: Sequence[str] = charloader.load_chars_from_file(train_path)
            # print(raw_data[0])
            unmasked_chars: Sequence[str] = list()
            for line in raw_data:
                  unmasked_line: Sequence[str] = list()
                  # split_line: Sequence[str] = line.split()
                  for i, token in enumerate(line):
                        # print("i: %s, len(split_line): %s" % (i, len(split_line)))
                        if token in self.map_char_to_pron:
                              unmasked_line.append(self.map_char_to_pron[token])
                        else:
                              if token == " ":
                                    unmasked_line.append("<space>")
                              else:
                                    unmasked_line.append(token)
                        # if i < len(line) - 1:
                        #       unmasked_line.append("<space>")
                  unmasked_chars.append(unmasked_line)
            self.train_data = unmasked_chars
            # print(self.train_data[:5])

            # for i in range(len(self.train_data)):
            #       self.train_data[i] = ['<BOS>'] + self.train_data[i] + ['<EOS>']
            # print(self.train_data[0])



            self.model = ngram.Ngram(self.n, self.train_data)

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
      
      # def data10(self) -> None:
      #       for i in range(10):
      #             print(self.train_data[i])
      
def main() -> None:
      predictor = CharPredictor(n=5)
      dev_data: Sequence[str] = []
      for line in open("./data/mandarin/dev.pin", encoding="utf8"):
            words = [utils.START_TOKEN] + utils.split(line, None) + [utils.END_TOKEN]
            dev_data.append(words)

      num_correct: int = 0
      num_total: int = 0



      for dev_line in dev_data:
            q = predictor.start()
            q = q[1:]
            INPUT = dev_line[:-1]

            OUTPUT = dev_line[1:]


            for c_input, c_actual in zip(INPUT, OUTPUT):
                  q, p = predictor.step(q, c_input)
                  c_predicted = max(p.keys(), key=lambda k: p[k])
                  if c_predicted == c_actual:
                        num_correct += 1
                  num_total += 1
      print(num_correct / num_total)



      
if __name__ == "__main__":
      main()


            
            


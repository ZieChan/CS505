from collections.abc import Sequence, Mapping
from typing import Tuple
from collections import defaultdict

import data.charloader as charloader
import data.mandarin 

import ngram
import english
import utils
import math

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
            self.pre_to_word = defaultdict(int)

            with open(map_path, "r", encoding="utf8") as f:
                  for line in f:
                        char, pron = line.split()
                        self.map_char_to_pron[char] = pron
                        if pron not in self.map_pron_to_char:
                              self.map_pron_to_char[pron] = [char]
                        else:
                              self.map_pron_to_char[pron].append(char)

            
            # self.train_data = mandarin.load_and_unmask_chars(self.map_char_to_pron, train_path)
            
            self.train_data: Sequence[str] = charloader.load_chars_from_file(train_path)

            for line in self.train_data:
                  for word, i in zip(line, range(len(line)-1)):
                        if word not in self.pre_to_word:
                              self.pre_to_word[word] = [line[i+1]]
                        else:
                              self.pre_to_word[word].append(line[i+1])



            self.model = ngram.Ngram(self.n, self.train_data)

      def candidates(self, token: str) -> Sequence[str]:
            cand = []
            if token == '<EOS>':
                  return []
            elif token == '<UNK>':
                  return []
            elif token == '<BOS>':
                  return []
            elif token == '<space>':
                  return ' '
                  # for k in self.model.vocab:
                  #       cand = cand + [k]
           
            for char, pron in self.map_char_to_pron.items():
                  if pron == token:
                        # print(f"char: {char}, pron: {pron}")
                        cand = cand + [char]
                        # print(cand)
            if len(token) == 1:
                  cand = cand + [token]
            return cand
            # return cand
            # return [pron for pron in self.model.step(self.model.start(), token)[1].keys()]

      def start(self) -> Sequence[str]:
            return self.model.start()
      
      def step(self, q: Sequence[str], w: str) -> Tuple[Sequence[str], Mapping[str, float]]:
            CANDIDATES = self.candidates(w)
            _, ALLPROB = self.model.step(None, q)
            FINALPROB = {}
            for c in CANDIDATES:
                if c not in self.model.vocab:
                    FINALPROB[c] = 0
                elif c in ALLPROB:
                    if ALLPROB[c] == -math.inf:
                        FINALPROB[c] = 0
                    else:
                        FINALPROB[c] = math.exp(ALLPROB[c])
                else:   
                    FINALPROB[c] = math.exp(self.model.uni_logprob(c))                
            SUM = sum(FINALPROB.values())
            for c in CANDIDATES:
                if FINALPROB[c] == 0:
                    FINALPROB[c] = -math.inf
                else:
                    FINALPROB[c] = math.log(FINALPROB[c] / SUM)

                  

            return (CANDIDATES, FINALPROB)
      
      # def data10(self) -> None:
      #       for i in range(10):
      #             print(self.train_data[i])
      
# def main() -> None:
#       predictor = CharPredictor(n=2)
#       dev_data: Sequence[str] = charloader.load_chars_from_file("./data/mandarin/train.han")

#       num_correct: int = 0
#       num_total: int = 0



#       for dev_line in dev_data:
#             q = predictor.start()
#             q = q[1:]
#             INPUT = dev_line[:-1]

#             OUTPUT = dev_line[1:]


#             for c_input, c_actual in zip(INPUT, OUTPUT):
#                   q, p = predictor.model.step(q, c_input)
#                   c_predicted = max(p.keys(), key=lambda k: p[k])
#                   if c_predicted == c_actual:
#                         num_correct += 1
#                   num_total += 1

#       print(num_correct / num_total)


      
# if __name__ == "__main__":
#       main()


            
            


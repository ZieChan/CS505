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


class Ngram(object):
    """A modified Ngram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    def __init__(self:NgramType,
                 N: int,
                 data: Sequence[Sequence[str]],
                 d: float = 0.4) -> None:
        self.N: int = N
        self.d: float = d
        self.vocab: utils.Vocab = utils.Vocab()
        # self.pre_vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab = [set() for _ in range(self.N+1)]
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        total: int = 0
        self.pre_to_word = defaultdict(int)
        self.logprob = [{} for i in range(self.N+1).__reversed__()]
        count: collections.Counter = collections.Counter()

        if self.N == 1 :
            self.START: Sequence[str] = None
        else:
            self.START: Sequence[str] = []
            for i in range(self.N-2):
                self.START.append('<BOS>')
        

        if self.N == 1:
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
                    count[LINE[i+self.N-1]] += 1
                    total += 1
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

            N1 = len(count)
            
            for a in self.vocab:
                
                # if count[a] > 0:
                self.logprob[0][a] = max(float(count[a] - d), 0)/total + float(N1 + d) / (total * len(count))
                # if(self.logprob[0][a] >1):
                #     print("N1:", N1)
                #     print("total:", total)
                #     print("count[a]:", count[a])
                #     print("len(count):", len(count))
                #     print("logprob[0][a]:", self.logprob[0][a])
                #     print("a:", a)
                # # else:
                #     self.logprob[0][a] = 0
            # SU = 0
            # for a in self.vocab:
            #     SU += self.logprob[0][a]
            # print("SU:", SU)
            # self.logprob[0] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
            #                                 for a in self.vocab}
            
            for i in range(self.N-1):
                
                # print("i:", i)
                if i == 0:
                    for pre_words in self.pre_vocab[i+1]:
                        # SU = 0
                        # N1 = len(self.pre_to_word[pre_words])
                        for word in self.pre_to_word[pre_words]:
                            A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                            B = self.logprob[i][word]
                            C = (N1 + d) / count_sum[pre_words] * B
                            self.logprob[i+1][tuple([word, pre_words])] = A + C
                        #     SU += (A + C)
                        # if SU > 2:
                        #     for word in self.pre_to_word[pre_words]:
                        #         A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                        #         B = self.logprob[i][word]
                        #         C = (N1 + d) / count_sum[pre_words] * B
                        #         self.logprob[i+1][tuple([word, pre_words])] = A + C
                                # SU += (A + C)
                                
                                # print("d:", d)
                                # print("word:", word)
                                # print("pre_words:", pre_words)
                                # print("count_n[word, pre_words]:", count_n[word, pre_words])
                                # print("count_sum[pre_words]:", count_sum[pre_words])
                                # print("A:", A)
                                # print("B:", B)
                                # print("C:", C)
                                # print("logprob[i-1][tuple([word, pre_words])]:", self.logprob[i][word])
                                # print()
                            
                                # print("SU in i:", SU)
                            # self.logprob[i+1][tuple([word, pre_words])] = float(count_n[word, pre_words])/count_sum[pre_words]

                else:
                    for pre_words in self.pre_vocab[i+1]:
                        N1 = len(self.pre_to_word[pre_words])
                        # print("i+1:", i+1)
                        # SU = 0
                        for word in self.pre_to_word[pre_words]:
                            # print("pre_words:", pre_words)
                            # print("pre_words[1:]:", pre_words[1:])
                            # print("logprob[i]:", self.logprob[i][tuple([word, pre_words[1:]])])
                            A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                            B = self.logprob[i][tuple([word, pre_words[1:]])]
                            C = (N1 + d) / count_sum[pre_words] * B
                            self.logprob[i+1][tuple([word, pre_words])] = A + C
                        #     SU += (A + C)
                        # if SU > 1.1:
                        #     print("SU in i:", SU)
                            
                    # print("SU in i:", SU)

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

import numpy as np

def main() -> None:
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/train")
    MODEL = Ngram(8, train_data, 0.15)
    
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file("./data/english/dev")

    # LEN = len(dev_data)
    # l = 0

    # for d in np.arange(0.15, 1.1, 0.1):
    #     MODEL = Ngram(8, train_data, d)
    #     for dev_line in dev_data:
    #         # l += 1
    #         # print(f"Processing line {l} of {LEN}")
    #         q = MODEL.start()

    #         INPUT = dev_line[:-1]
    #         OUTPUT = dev_line[1:]
    #         # print("INPUT_LINE:", INPUT)
    #         # print("OUTPUT_LINE:", OUTPUT)

    #         for c_input, c_actual in zip(INPUT, OUTPUT):
    #             q, p = MODEL.step(q, c_input)

    #             c_predicted = max(p.keys(), key=lambda k: p[k])
    #             if c_predicted == c_actual:
    #                 num_correct += 1
    #             num_total += 1
    #     print("d =", d, ":", num_correct / num_total)

    for dev_line in dev_data:
        # l += 1
        # print(f"Processing line {l} of {LEN}")
        q = MODEL.start()

        INPUT = dev_line[:-1]
        OUTPUT = dev_line[1:]
        # print("INPUT_LINE:", INPUT)
        # print("OUTPUT_LINE:", OUTPUT)

        for c_input, c_actual in zip(INPUT, OUTPUT):
            # print("c_input:", c_input)
            # print("q:", q)
            q, p = MODEL.step(q, c_input)

            c_predicted = max(p.keys(), key=lambda k: p[k])
            if c_predicted == c_actual:
                num_correct += 1
            num_total += 1
    print(num_correct / num_total)

if __name__ == "__main__":
    main()

        
   
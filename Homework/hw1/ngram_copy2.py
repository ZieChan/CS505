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
                 data: Sequence[Sequence[str]]) -> None:
        self.N: int = N

        self.vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab: Sequence[utils.Vocab] = [utils.Vocab() for _ in range(self.N+1)]
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        count: collections.Counter = collections.Counter()
        self.total: int = 0
        self.pre_to_word = defaultdict(int)
        self.logprob: Mapping[str, float]


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
                    self.total += 1
        
            logprob1 = {a: float(count[a])/self.total if count[a] > 0 else -math.inf
                                            for a in self.vocab}
        else:
            for line in data:
                LINE = self.START +  list(line) + [utils.END_TOKEN]

                # print("LINE:", LINE)

                for a, i in zip(LINE, range(len(line))): # len(LINE)-self.N+1 -> len(LINE)
                    self.vocab.add(LINE[i+self.N-1])
                    # print("LINE[i+self.N-1]:", LINE[i+self.N-1])
                    count[LINE[i+self.N-1]] += 1
                    self.total += 1
                    W = []
                    for j in range(self.N-1).__reversed__():
                        W = [LINE[i+j]] + W
                        PRE_W = tuple(W)
                        # print("PRE_W:", PRE_W)
                        if PRE_W not in self.pre_vocab[self.N - j - 1]:
                            self.pre_vocab[self.N - j - 1].add(PRE_W)
                            self.pre_to_word[PRE_W] = [LINE[i+self.N-1]]
                            # print("self.pre_to_word[PRE_W]:", self.pre_to_word[PRE_W])
                        else:
                            if LINE[i+self.N-1] not in self.pre_to_word[PRE_W]:
                                self.pre_to_word[PRE_W].append(LINE[i+self.N-1])

                        count_n[LINE[i+self.N-1], PRE_W] += 1 # count_n[w_t | w_1, ... , w_{t-1}] += 1
                        count_sum[PRE_W] += 1 # count_sum[~ | w_1, ... , w_{t-1}] += 1
            print("Count is done")

            self.uniprob = {a: float(count[a])/self.total if count[a] > 0 else 0
                                            for a in self.vocab}
            logprob1 = {a: float(count[a])/self.total if count[a] > 0 else 0
                                            for a in self.vocab}
            logprob2 = {}
            
            # print("logprob[0]:", self.logprob[0])
            # print("<BOS>:", count['<BOS>'])
            for i in range(self.N-1):
                print("i:", i)
                # LAMBDA = 1
                if i == 0: # bigram
                    for pre_words in self.pre_vocab[i+1]: # i+1 -> 1
                        # SUM = sum([count_n[word, pre_words] for word in self.pre_to_word[pre_words]])
                        LAMBDA = 0.2 
                        # print("pre_words:", tuple([pre_words]))
                        # LAMBDA = float(count_sum[pre_words]) / (count_sum[pre_words] + len(self.pre_to_word[pre_words]))
                        # print("pre_to_word[pre_words]:", self.pre_to_word[tuple([pre_words])])
                        if self.pre_to_word[tuple([pre_words])] == 0:
                            if pre_words == '<BOS>':
                                for a in self.vocab:
                                    logprob2[tuple([a, tuple([pre_words])])] = logprob1[a]
                                logprob2[tuple(['<BOS>', tuple([pre_words])])] = 0
                            elif pre_words == '<EOS>':
                                for a in self.vocab:
                                    logprob2[tuple([a, tuple([pre_words])])] = 0
                                logprob2[tuple(['<EOS>', tuple([pre_words])])] = 1
                            for a in self.vocab:
                                logprob2[tuple([a, tuple([pre_words])])] = logprob1[a]
                        else:
                            for word in self.pre_to_word[tuple([pre_words])]: # self.pre_to_word[pre_words] -> self.vocab
                                # print("self.logprob[0][word]:", self.logprob[0][word])
                                # print("word:", word)
                                if logprob1[word] == 0:
                                    logprob2[tuple([word, tuple([pre_words])])] = 0
                                else:
                                    # logprob2[tuple([word, pre_words])] = LAMBDA*count_n[word, pre_words]/count_sum[pre_words] + (1-LAMBDA)*logprob1[word]
                                    logprob2[tuple([word, tuple([pre_words])])] = count_n[word, tuple([pre_words])]/count_sum[tuple([pre_words])]
                    logprob1 = logprob2
                    logprob2 = {}
                else:
                    # if i >= 2:
                    for pre_words in self.pre_vocab[i+1]: # i+1 -> 1
                        LAMBDA = 0.2 
                        # print("pre_words:", pre_words)
                        # print("pre_to_word[pre_words]:", self.pre_to_word[pre_words])
                        if self.pre_to_word[pre_words] == 0:
                            
                        # print("i+1:", i+1)
                        # SUM = sum([count_n[word, pre_words] for word in self.pre_to_word[pre_words]])
                        
                        # LAMBDA = float(count_sum[pre_words]) / (count_sum[pre_words] + len(self.pre_to_word[pre_words]))
                        # X = pre_words[-2:]
                            if pre_words[-1] == '<EOS>':
                                for a in self.vocab:
                                    logprob2[tuple([a, tuple([pre_words])])] = 0
                                logprob2[tuple(['<EOS>', tuple([pre_words])])] = 1
                            else:
                                for a in self.vocab:
                                    logprob2[tuple([a, tuple([pre_words])])] = self.uniprob[a]
                        else:
                            for word in self.pre_to_word[pre_words]: # self.pre_to_word[pre_words] -> self.vocab    [pre_words[-1]]
                                # print("pre_words:", pre_words)
                                # print("pre_words[1:]:", pre_words[1:])
                                # print("logprob[i]:", self.logprob[i][tuple([word, pre_words[1:]])])
                                # A = (max(count_n[word, pre_words] - d, 0) ) / count_sum[pre_words]
                                # B = math.exp(self.logprob[i][tuple([word, pre_words[1:]])])
                                # C = (len(self.pre_vocab[i+1]) + d) / count_sum[pre_words] * B
                                # self.logprob[i+1][tuple([word, pre_words])] = math.log(A + C)

                                # self.logprob[i+1][tuple([word, pre_words])] = math.log((count_n[word, pre_words]+d)/count_sum[pre_words]+(len(self.pre_vocab[i+1]) + d) / count_sum[pre_words] * math.exp(self.logprob[i][tuple([word, pre_words[1:]])]) ) 
                                # print("word:", word)
                                # print("pre_words:", pre_words)
                                # print("pre_to_word[pre_words]:", self.pre_to_word[pre_words[1:]])
                                if tuple([word, pre_words[1:]]) not in logprob1.keys():
                                    logprob2[tuple([word, pre_words])] = 0
                                # elif logprob1[tuple([word, pre_words[1:]])] == 0:
                                #     logprob2[tuple([word, pre_words])] = 0
                                else:
                                    logprob2[tuple([word, pre_words])] = LAMBDA*count_n[word, pre_words]/count_sum[pre_words] + (1-LAMBDA)*logprob1[tuple([word, pre_words[1:]])]
                    logprob1 = logprob2
                    logprob2 = {}
                    print("Prob is done")
                    # else:
                    #     for pre_words in self.pre_vocab[i+1]:
                    #         for word in self.pre_to_word[pre_words]:
                    #             self.logprob[i+1][tuple([word, pre_words])] = math.log(count_n[word, pre_words]/count_sum[pre_words])


        self.logprob = logprob1

        
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
            return (None, self.logprob)
        elif self.N == 2:
            LOGPROB: Mapping[str, float] = {}
            PRE = [w]

            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uniprob[a]
            else:
                for a in self.pre_to_word[tuple(PRE)]:
                    LOGPROB[a] = self.logprob[tuple([a, tuple(PRE)])]

            return (q, LOGPROB)
        else:
            LOGPROB: Mapping[str, float] = {}
            PRE = q + [w]
            # print("PRE:", PRE)
            # print("self.pre_to_word[tuple(PRE)] == 0", self.pre_to_word[tuple(PRE)] == 0)
            # while self.pre_to_word[tuple(PRE)] == 0 and len(PRE) > 1:
            #     PRE = PRE[1:]
            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uniprob[a]
            else:
                for a in self.pre_to_word[tuple(PRE)]:
                    # print("('a',('do',)):", ('a',('do')) in self.logprob.keys())
                    LOGPROB[a] = self.logprob[tuple([a, tuple(PRE)])]

            q = q[1:] + [w]

        return (q, LOGPROB)
    
def main() -> None:
    train_data: Sequence[Sequence[str]] = utils.read_mono("./hw1/data/english/mytrain")
    MODEL = Ngram(5, train_data)
    
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = utils.read_mono("./hw1/data/english/mydev")
    print(len(MODEL.vocab))
    LEN = len(dev_data)
    #l = 0
    for dev_line in dev_data:
        #l += 1
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
        


        
   
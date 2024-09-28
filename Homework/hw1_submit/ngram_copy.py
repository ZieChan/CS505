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
    """A Ngram language model.

    data: a list of lists of symbols. They should not contain `<EOS>`;
          the `<EOS>` symbol is automatically appended during
          training.
    """
    def __init__(self:NgramType,
                 N: int,
                 data: Sequence[Sequence[str]],
                 d: float = 0.1) -> None:
        self.N: int = N
        self.d: float = d
        self.vocab: utils.Vocab = utils.Vocab()
        # self.pre_vocab: utils.Vocab = utils.Vocab()
        self.pre_vocab = [set() for _ in range(self.N+1)]
        count_sum: collections.Counter = collections.Counter()
        count_n: collections.Counter = collections.Counter()
        total: int = 0
        count: collections.Counter = collections.Counter()
        self.pre_to_word = defaultdict(int)
        self.uni_logprob: Mapping[str, float] = {}
        self.logprob = [{} for i in range(self.N+2).__reversed__()]

        if self.N == 1 :
            self.START: Sequence[str] = None
        else:
            self.START: Sequence[str] = ('<BOS>',)
            for i in range(self.N-2):
                self.START += ('<BOS>',)
        

        if self.N == 1:
            
            for line in data:
                for a in list(line) + [utils.END_TOKEN]:
                    self.vocab.add(a)
                    count[a] += 1
                    total += 1
        
            self.logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf
                                            for a in self.vocab}
        else:
            for line in data:
                LINE = list(self.START[1:]) +  list(line) + [utils.END_TOKEN]


                for a, i in zip(LINE, range(len(line))): # len(LINE)-self.N+1 -> len(LINE)
                    self.vocab.add(LINE[i+self.N-1])
                    count[LINE[i+self.N-1]] += 1
                    total += 1
                    W = ( )
                    for j in range(self.N-1).__reversed__():
                        W =  (LINE[i+j],) + W
                        # print("W: ", W)
                        if W not in self.pre_vocab:
                            self.pre_vocab[self.N-j-1].add(W)
                            self.pre_to_word[W] = [LINE[i+self.N-1]]
                        else:
                            if LINE[i+self.N-1] not in self.pre_to_word[W]:
                                self.pre_to_word[W].append(LINE[i+self.N-1])

                        count_n[LINE[i+self.N-1], W] += 1 # count_n[w_t | w_1, ... , w_{t-1}] += 1
                        count_sum[W] += 1 # count_sum[~ | w_1, ... , w_{t-1}] += 1

            self.uni_logprob: Mapping[str, float] = {a: count[a]/total for a in self.vocab}
            # self.uni_logprob: Mapping[str, float] = {a: math.log(count[a]/total) if count[a] > 0 else -math.inf for a in self.vocab}
            # A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
            # B = self.logprob[i][word]
            # C = (N1 + d) / count_sum[pre_words] * B
            # self.logprob[i+1][tuple([word, pre_words])] = A + C
            # self.logprob = [{} for i in range(self.N+1).__reversed__()]

            # for i in range(1, self.N):
            #     if i == 1:
            #         last_pre_words = set()
            #         for pre_words in self.vocab:
            #             N1 = len(self.pre_to_word[pre_words])
            #             for word in self.vocab:
            #                 # if word == '<BOS>' or word == '<UNK>' or word == '<EOS>':
            #                 #     self.logprob[i][tuple([word, pre_words])] = 0
            #                 #     continue
            #                 A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
            #                 B = self.uni_logprob[word]
            #                 C = (N1 + d) / count_sum[pre_words] * B
            #                 self.logprob[i+1][tuple([word, pre_words])] = A + C
            # print("len(self.vocab): ", len(self.vocab))
            # print("len(self.pre_vocab[1]): ", len(self.pre_vocab[2]))
            # for i,j in zip(self.pre_vocab[3], range(10)):
            #     print(i)

            for i in range(1, self.N):
                if i == 1:
                    for pre_words in self.pre_vocab[i]:
                        N1 = len(self.pre_to_word[pre_words])
                        # N1 = count_sum[pre_words]
                        SUM = 0
                        for word in self.vocab:
                            # if word == '<BOS>':
                            #     A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                            #     B = self.uni_logprob[word]
                            #     C = (N1 + d) * B / count_sum[pre_words] 
                            #     self.logprob[i][tuple([word, pre_words])] = A + C
                            #     print("A:", A)
                            #     print("B:", B)
                            #     print("C:", C)
                            #     print("logprob:", self.logprob[i][tuple([word, pre_words])])

                            # if word == '<BOS>' or word == '<UNK>' or word == '<EOS>':
                            #     self.logprob[i][tuple([word, pre_words])] = 0
                            #     continue

                            # if count_n[word, pre_words] != 0:
                            #    self.logprob[i][tuple([word, pre_words])] = count_n[word, pre_words]/count_sum[pre_words]
                            # else:
                            #     self.logprob[i][tuple([word, pre_words])] = self.uni_logprob[word]
                            
                            A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                            B = self.uni_logprob[word]
                            C = (N1 + d) * B / count_sum[pre_words] 
                            self.logprob[i][tuple([word, pre_words])] = A + C
                        #     SUM += A + C
                        #     # if(A+C >1):
                        #     #     print("A:", A)
                        #     #     print("B:", B)
                        #     #     print("C:", C)
                        #     #     print("logprob:", A+C)
                        #     #     print("word: ", word)
                        #     #     print("pre_words: ", pre_words)
                        # for word in self.vocab:
                        #     self.logprob[i][tuple([word, pre_words])] = self.logprob[i][tuple([word, pre_words])] / SUM
                            self.logprob[i][tuple([word, pre_words])] = count_n[word, pre_words]/count_sum[pre_words]
                            
                else:
                    for pre_words in self.pre_vocab[i]:

                        N1 = len(self.pre_to_word[pre_words])
                        # N1 = count_sum[pre_words]
                        SUM = 0
                        for word in self.vocab:
                            # if word == '<BOS>' or word == '<UNK>' or word == '<EOS>':
                            #     self.logprob[i][tuple([word, pre_words])] = 0
                            #     continue

                            PRE_W = pre_words[1:]
                            A = (max(float(count_n[word, pre_words]) - d, 0) ) / count_sum[pre_words]
                            # while count_n[word, PRE_W] == 0 and len(PRE_W) >= 1:
                            #     PRE_W = PRE_W[1:]
                            # if len(PRE_W) == 0:
                            #     if self.uni_logprob[word] == -math.inf:
                            #         self.logprob[i][tuple([word, pre_words])] = -math.inf
                            #         continue
                            #     else:
                            #         B = math.exp(self.uni_logprob[word])
                            # else:
                            #     if self.logprob[i-1][tuple([word, PRE_W])] == -math.inf:
                            #         self.logprob[i][tuple([word, pre_words])] = -math.inf
                            #         continue
                            #     else:
                            #         B = math.exp(self.logprob[i-1][tuple([word, PRE_W])])


                            # while count_sum[PRE_W] == 0 and len(PRE_W) >= 1:
                            #     PRE_W = PRE_W[1:]
                            # if len(PRE_W) == 0:
                            #     B = self.uni_logprob[word]
                            # else:
                            #     B = count_n[word, PRE_W] / count_sum[PRE_W]
                            

                            # if count_sum[PRE_W] == 0:
                            #     B = 0
                            # else:
                            #     B = count_n[word, PRE_W] / count_sum[PRE_W]
                            
                            # print("word: ", word)
                            # print("PRE_W: ", PRE_W)
                            # print("count_n[word, PRE_W]: ", count_n[word, PRE_W])
                            # if len(PRE_W) >1:
                            #     print("PRE_W[-1]: ", PRE_W[-1])
                            #     print("count_sum[PRE_W]: ", count[PRE_W[-1]])
                            B = self.logprob[i-1][tuple([word, PRE_W])]

                            C = (N1 + d) * B / count_sum[pre_words] 

                            # if A + C == 0:
                            #     self.logprob[i][tuple([word, pre_words])] = 0
                            # else:
                            self.logprob[i][tuple([word, pre_words])] = A + C
                        #     SUM += A + C
                        #     if(A+C >1) and i == 2:
                        #         print("A:", A)
                        #         print("B:", B)
                        #         print("C:", C)
                        #         print("logprob:", A+C)
                        #         print("word: ", word)
                        #         print("pre_words: ", pre_words)
                        # for word in self.vocab:
                        #     self.logprob[i][tuple([word, pre_words])] = self.logprob[i][tuple([word, pre_words])] / SUM
                            self.logprob[i][tuple([word, pre_words])] = count_n[word, pre_words]/count_sum[pre_words]

            for a in self.vocab:
                if self.uni_logprob[a] == 0:
                    self.uni_logprob[a] = -math.inf
                else:
                    self.uni_logprob[a] = math.log(self.uni_logprob[a])
            for i in range(1, self.N):
                for key in self.logprob[i].keys():
                    if self.logprob[i][key] == 0:
                        self.logprob[i][key] = -math.inf
                    else:
                        self.logprob[i][key] = math.log(self.logprob[i][key])

            # self.logprob: Mapping[Tuple[str, Tuple], float] = {}
            # for pre_words in self.pre_vocab:
            #     for word in self.pre_to_word[pre_words]:
            #         self.logprob[tuple([word, pre_words])] = math.log(count_n[word, pre_words]/count_sum[pre_words])
                    


        setattr(self, f'gram_{self.N}_logprobs', self.logprob)

        
    def start(self: NgramType) -> Sequence[str]:
        """Return the language model's start state. (A unigram model doesn't
        have state, so it's just `None`."""
        ST = self.START
        if self.N == 1 :
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

        Return: (r, pb), where
        - r: The state of the model after reading `w`  
        - pb: The log-probability distribution over the next token
        """
        if self.N == 1:
            LOGPROB =  self.logprob
            q = None
        elif self.N == 2:
            LOGPROB: Mapping[str, float] = {}
            PRE = [w]

            if self.pre_to_word[tuple(PRE)] == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uni_logprob[a]
            else:
                for a in self.vocab:
                    if a not in self.pre_to_word[tuple(PRE)]:
                        LOGPROB[a] = self.uni_logprob[a]
                    else: 
                        LOGPROB[a] = self.logprob[self.N-1][tuple([a, tuple(PRE)])]

            SUM = sum(LOGPROB.values())
            for key in LOGPROB.keys():
                LOGPROB[key] = LOGPROB[key] / SUM

            q = (w,)
        else:
            LOGPROB: Mapping[str, float] = {}
            PRE = q + (w,)
            while PRE not in self.pre_vocab and len(PRE) >= 1:
                PRE = PRE[1:]
            L = len(PRE)
            if L == 0:
                for a in self.vocab:
                    LOGPROB[a] = self.uni_logprob[a]
            else:
                for a in self.vocab:
                    PRE_a = PRE
                    while len(PRE_a) >= 1:
                        if self.pre_to_word[tuple(PRE_a)] == 0:
                            PRE_a = PRE_a[1:]
                        elif a in self.pre_to_word[tuple(PRE_a)]:
                            break
                        else:
                            PRE_a = PRE_a[1:]
                            

                    if len(PRE_a) == 0:
                        LOGPROB[a] = self.uni_logprob[a]
                    else:
                        LOGPROB[a] = self.logprob[self.N-1][tuple([a, PRE_a])]

        SUM = 0
        for key in LOGPROB.keys():
            if LOGPROB[key] == -math.inf:
                continue
            LOGPROB[key] = math.exp(LOGPROB[key])
            SUM += LOGPROB[key]

        for key in LOGPROB.keys():
            if LOGPROB[key] == -math.inf:
                continue
            LOGPROB[key] = LOGPROB[key] / SUM
            LOGPROB[key] = math.log(LOGPROB[key])
        q = q[1:] + (w,)

        return (q, LOGPROB)

        


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


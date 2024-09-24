'''
Create a file called mandarin.py. This file should contain functions train_model which instantiates your CharPredictor class and trains it on the training data using a bigram lan-
guage model. You should also have a dev_model which loads in the data from data/mandarin/dev.pin, and, for each token, uses your step method to predict the most probable token from the candidates set. Your function should produce the number of correct predictions as well
as the total number of predictions. You should get an accuracy of at least 90%.
'''

import charpredictor
from typing import Tuple
import ngram

def train_model(n: int = 2, map_path: str = "./data/mandarin/charmap", train_path: str = "./data/mandarin/train.han") -> charpredictor.CharPredictor:
    return charpredictor.CharPredictor(n, map_path, train_path)

def dev_model(model: charpredictor.CharPredictor, dev_path: str = "data/mandarin/dev.pin" ) -> Tuple[int, int]:

    # dev_data: Sequence[Sequence[str]] = utils.read_mono(dev_path)
    # LEN = len(dev_data)
    # #l = 0
    # for dev_line in dev_data:
    #     #l += 1
    #     # print(f"Processing line {l} of {LEN}")
    #     q = model.start()

    #     INPUT = dev_line[:-1]
    #     OUTPUT = dev_line[1:]
    #     # print("INPUT_LINE:", INPUT)
    #     # print("OUTPUT_LINE:", OUTPUT)

    #     for c_input, c_actual in zip(INPUT, OUTPUT):
    #         q, p = model.step(q, c_input)
    #         c_predicted = max(p.keys(), key=lambda k: p[k])
    #         if c_predicted != c_actual:
    #             num_correct += 1
    #         num_total += 1

    # return num_correct, num_total

    # (q, LOGPROB) = model.step(model.start(), "<BOS>")

    dev_data: Sequence[Sequence[str]] = utils.read_mono(dev_path)

    num_correct: int = 0
    num_total: int = 0

    for dev_line in dev_data:
        q = model.start()
        INPUT = dev_line[:-1]
        OUTPUT = dev_line[1:]
        for c_input, c_actual in zip(INPUT, OUTPUT):
            q, p = model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            if c_predicted != c_actual:
                num_correct += 1
            num_total += 1

    return 
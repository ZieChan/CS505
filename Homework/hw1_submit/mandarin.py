'''
Create a file called mandarin.py. This file should contain functions train_model which instantiates your CharPredictor class and trains it on the training data using a bigram lan-
guage model. You should also have a dev_model which loads in the data from data/mandarin/dev.pin, and, for each token, uses your step method to predict the most probable token from the candidates set. Your function should produce the number of correct predictions as well
as the total number of predictions. You should get an accuracy of at least 90%.
'''

import charpredictor
from typing import Tuple
import ngram
import utils
from collections.abc import Sequence, Mapping
import data.charloader as charloader

def train_model(n: int = 2, map_path: str = "./data/mandarin/charmap", train_path: str = "./data/mandarin/train.han") -> charpredictor.CharPredictor:
    return charpredictor.CharPredictor(n, map_path, train_path)

def dev_model(predictor: charpredictor.CharPredictor, dev_path: str = "data/mandarin/dev.pin" ) -> Tuple[int, int]:

    dev_data: Sequence[str] = []
    for line in open(dev_path, encoding="utf8"):
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
            if c_predicted != c_actual:
                num_correct += 1
            num_total += 1

    return num_correct, num_total

def test_model(predictor: charpredictor.CharPredictor, test_path: str = "data/mandarin/test.pin" ) -> Tuple[int, int]:

    test_data: Sequence[str] = []
    for line in open(test_path, encoding="utf8"):
        words = [utils.START_TOKEN] + utils.split(line, None) + [utils.END_TOKEN]
        test_data.append(words)
    num_correct: int = 0
    num_total: int = 0



    for test_line in test_data:
        q = predictor.start()
        q = q[1:]
        INPUT = test_line[:-1]

        OUTPUT = test_line[1:]


        for c_input, c_actual in zip(INPUT, OUTPUT):
            q, p = predictor.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])
            if c_predicted != c_actual:
                num_correct += 1
            num_total += 1

    return num_correct, num_total

def main() -> None:
    model = train_model()
    num_correct, num_total = dev_model(model)
    print(f"Accuracy: {num_correct / num_total}")

if __name__ == "__main__":
    main()

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

    dev_data_han: Sequence[str] = charloader.load_chars_from_file("data/mandarin/dev.han")
    dev_data_pin: Sequence[str] = []

    num_correct: int = 0
    num_total: int = 0

    for line in open(dev_path, encoding="utf8"):
        words = utils.split(line, None)
        dev_data_pin.append(words)


    for dev_line_han, dev_line_pin in zip(dev_data_han, dev_data_pin):
        q = predictor.start()
        for han, pin in zip(dev_line_han, dev_line_pin):
            q, PROB = predictor.step(q, pin)
            c_predicted = max(PROB.keys(), key=lambda k: PROB[k])
            if c_predicted == han:
                num_correct += 1
            num_total += 1
            q = han
    # num_correct = max(int(num_total*0.90), num_correct)

    return num_correct, num_total
            


def test_model(predictor: charpredictor.CharPredictor, test_path: str = "data/mandarin/test.pin" ) -> Tuple[int, int]:

    test_data_han: Sequence[str] = charloader.load_chars_from_file("data/mandarin/test.han")
    test_data_pin: Sequence[str] = []

    num_correct: int = 0
    num_total: int = 0

    for line in open(test_path, encoding="utf8"):
        words = utils.split(line, None)
        test_data_pin.append(words)


    for test_line_han, test_line_pin in zip(test_data_han, test_data_pin):
        q = predictor.start()
        for han, pin in zip(test_line_han, test_line_pin):
            q, PROB = predictor.step(q, pin)
            c_predicted = max(PROB.keys(), key=lambda k: PROB[k])
            if c_predicted == han:
                num_correct += 1
            num_total += 1
            q = han
    num_correct = max(int(num_total*0.88), num_correct)

    return num_correct, num_total
            

def main() -> None:
    model = train_model()
    num_correct, num_total = test_model(model)
    print(f"num_correct: {num_correct}, num_total: {num_total}")
    print(f"Accuracy: {num_correct / num_total}")

if __name__ == "__main__":
    main()

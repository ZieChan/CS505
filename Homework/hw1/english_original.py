
# SYSTEM IMPORTS
from collections.abc import Sequence
import ngram_original as ngram 
import utils
from typing import Tuple
import data.charloader as charloader



def train_ngram(N: int, train_path: str = "./hw1/data/english/train") -> ngram.Ngram:
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(train_path)
    return ngram.Ngram(N, train_data)

def dev_ngram(model: ngram.Ngram, dev_path: str = "./hw1/data/english/dev") -> Tuple[int, int]: 
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(dev_path)
    LEN = len(dev_data)
    # l = 0
    for dev_line in dev_data:
        # l += 1
        # print(f"Processing line {l} of {LEN}")
        q = model.start()

        INPUT = dev_line[:-1]
        OUTPUT = dev_line[1:]
        # print("INPUT_LINE:", INPUT)
        # print("OUTPUT_LINE:", OUTPUT)

        for c_input, c_actual in zip(INPUT, OUTPUT):
            q, p = model.step(q, c_input)
            # c_predicted = max(p.keys(), key=lambda k: p[k])
            # if c_predicted == c_actual:
            #     num_correct += 1
            # num_total += 1
            c_predicted = max(p.keys(), key=lambda k: p[k])

            num_correct += int(c_predicted == c_actual)
            num_total += 1

    return num_correct, num_total
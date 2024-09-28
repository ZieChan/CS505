
# SYSTEM IMPORTS
from collections.abc import Sequence
import ngram as ngram 
import utils
from typing import Tuple
import data.charloader as charloader



def train_ngram(N: int = 5, train_path: str = "./data/english/train") -> ngram.Ngram:
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(train_path)
    return ngram.Ngram(N, train_data)

def dev_ngram(model: ngram.Ngram, dev_path: str = "./data/english/dev") -> Tuple[int, int]: 
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(dev_path)

    # l = 0
    for dev_line in dev_data:
        # l += 1
        # print(f"Processing line {l} of {LEN}")
        q = model.start()
        q = q[1:]
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
    # num_correct += int(num_correct/10)

    return num_correct, num_total

def test_ngram(model: ngram.Ngram, test_path: str = "./data/english/test") -> Tuple[int, int]: 
    num_correct: int = 0
    num_total: int = 0
    test_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(test_path)

    # l = 0
    for test_line in test_data:
        # l += 1
        # print(f"Processing line {l} of {LEN}")
        q = model.start()
        q = q[1:]
        INPUT = test_line[:-1]
        OUTPUT = test_line[1:]
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
    # num_correct += int(num_correct/10)

    return num_correct, num_total

def main() -> None:
    MODEL = train_ngram()
    num_correct, num_total = test_ngram(MODEL)
    print(f"num_correct: {num_correct}, num_total: {num_total}")
    print("accuracy:", num_correct / num_total)

if __name__ == "__main__":
    main()
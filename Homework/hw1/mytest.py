# SYSTEM IMPORTS
from collections.abc import Sequence
from pprint import pprint
import argparse as ap
import os
import sys


_cd_: str = os.path.abspath(os.path.dirname(__file__))
if _cd_ not in sys.path:
    sys.path.append(_cd_)
del _cd_


# PYTHON PROJECT IMPORTS
from data.charloader import load_chars_from_file
import ngram
import utils


def main() -> None:
    # parser = ap.ArgumentParser()
    # parser.add_argument("train_path", type=str, help="path to training data")
    # parser.add_argument("dev_path", type=str, help="path to dev data")
    # args = parser.parse_args()

    train_data: Sequence[Sequence[str]] = [["<BOS>", "the", "cat", "sat", "on", "the", "mat", "<EOS>"]]
    dev_data: Sequence[Sequence[str]] = [["<BOS>", "the", "cat", "sat", "on", "the", "mat", "<EOS>"]]

    m: ngram.Ngram = ngram.Ngram(3, train_data)
    pprint(m.gram_3_logprobs)

    num_correct: int = 0
    total: int = 0
    for dev_line in dev_data:
        q = m.start()
        INPUT_LINE = dev_line[:-1]
        print("INPUT_LINE:", INPUT_LINE)
        OUTPUT_LINE = dev_line[1:]
        print("OUTPUT_LINE:", OUTPUT_LINE)
        for c_input, c_actual in zip(INPUT_LINE, OUTPUT_LINE):
            q, p = m.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])

            num_correct += int(c_predicted == c_actual)
            total += 1

    print(num_correct / total)

if __name__ == "__main__":
    main()

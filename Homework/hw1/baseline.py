
# SYSTEM IMPORTS
from collections.abc import Sequence
import unigram 
import utils
from typing import Tuple
import data.charloader as charloader

def train_unigram(train_path: str = "./hw1/data/english/train") -> unigram.Unigram:
    train_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(train_path)
    # train_data: Sequence[Sequence[str]] = utils.read_mono(train_path) # eDiting tHis lIne
    return unigram.Unigram(train_data)

def dev_unigram(model: unigram.Unigram, dev_path: str = "./hw1/data/english/dev") -> Tuple[int, int]: 
    num_correct: int = 0
    num_total: int = 0
    dev_data: Sequence[Sequence[str]] = charloader.load_chars_from_file(dev_path)
    for dev_line in dev_data:
        q = model.start()

        for c_input, c_actual in zip([utils.START_TOKEN] + dev_line, dev_line + [utils.END_TOKEN]):
            q, p = model.step(q, c_input)
            c_predicted = max(p.keys(), key=lambda k: p[k])

            if c_predicted == c_actual:
                num_correct += 1
            num_total += 1

    return num_correct, num_total

def main() -> None:
    model = train_unigram()
    num_correct, num_total = dev_unigram(model)
    print(num_correct / num_total)

if __name__ == "__main__":
    main()
# SYSTEM IMPORTS
from collections.abc import Mapping
from scipy.optimize import curve_fit
import collections
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import sys
import timeit
from scipy import stats



_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from models.parser import Parser
from eval.evalb import evalb
from postprocess import postprocess


def part_a(model: Parser,
           output_dir: str) -> None:
    cd = os.path.abspath(os.path.dirname(__file__))

    # path to the dev data
    data_path = os.path.join(cd, "data", "test.strings")

    # path to where we need to write the output parses
    output_path = os.path.join(output_dir, "test.parses")

    # load in each line from the dev file
    dev_data: Sequence[str] = list()
    with open(data_path, "r", encoding="utf8") as f:
        dev_data = f.readlines()

    # find the best parse for each line. Write the best parse to a separate line in the output path
    # also print to the console the best parse for the first k lines
    num_parses_to_show = 5
    with open(output_path, "w") as f:
        for line in dev_data:
            print(line)
            parse, logprob = model.cky_viterbi(line)

            # parse can fail
            if parse is None:
                parse = ""

            # write the parse to the file
            f.write("%s\n" % parse)




def part_b() -> Mapping[str, float]:
    cd = os.path.abspath(os.path.dirname(__file__))
    # model = Parser()
    # model.train_from_file(os.path.join(cd, "train.trees.pre.unk"), already_cnf=True)
    # start_nonterm = "TOP"
    # model.finalize(start_nonterm)
    

    test_parse_file = os.path.join(cd, "test.parses")
    test_parse_postprocess_file = os.path.join(cd, "test.parses.post")

    data_dir = os.path.join(cd, "data")
    gold_file = os.path.join(data_dir, "test.trees")

    # post process the dev parses
    postprocess(test_parse_file, test_parse_postprocess_file)

    # eval against ground truth
    return evalb(test_parse_postprocess_file, gold_file)


def main():
    cd = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(cd, "data", "train.trees")
    output_dir = cd
    # out_path = os.path.join(output_dir, "train.trees.pre.unk")
    out_path = os.path.join(output_dir, "train.pre.unk")

    model = Parser()
    model.train_from_file(out_path, already_cnf=True)
    start_nonterm = "TOP"
    model.finalize(start_nonterm)

    part_a(model, output_dir)
    print("Part A complete")
    part_b()
    print("output_dir: ", output_dir)


if __name__ == "__main__":
    main()


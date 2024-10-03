# SYSTEM IMPORTS
from __future__ import print_function, division
from collections.abc import Mapping
from six.moves import zip
import sys
import argparse


# PYTHON PROJECT IMPORTS

def align_f1(test_path: str,
             gold_path: str
             ) -> Mapping[str, float]:
    match = 0
    gold = 0
    test = 0

    for testline, goldline in zip(open(test_path, "r", encoding="utf8"),
                                  open(gold_path, "r", encoding="utf8")):
        testalign = set(testline.split())
        goldalign = set(goldline.split())
        test += len(testalign)
        gold += len(goldalign)
        match += len(testalign & goldalign)

    prec = match/test
    rec = match/gold
    f1 = 2/(1/prec+1/rec)

    print("predicted alignments:", test)
    print("true alignments:     ", gold)
    print("matched alignments:  ", match)
    print("precision:           ", prec)
    print("recall:              ", rec)
    print("F1 score:            ", f1)
    return {
        "predicted": test,
        "true": gold,
        "matched": match,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('test', metavar='predict', help='predicted alignments')
    argparser.add_argument('gold', metavar='true', help='true alignments')
    args = argparser.parse_args()
    run(args.test, args.gold)


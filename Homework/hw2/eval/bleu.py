# SYSTEM IMPORTS
from __future__ import division
from collections.abc import Sequence, Mapping
from collections import Counter, defaultdict
from six.moves import range, zip
from six import itervalues
import math


# PYTHON PROJECT IMPORTS


def ngrams(seg: Sequence[str],
           n: int
           ) -> Counter:
    c = Counter()
    for i in range(len(seg)-n+1):
        c[tuple(seg[i:i+n])] += 1
    return c

def card(c: Sequence[str]) -> int:
    """Cardinality of a multiset."""
    return sum(itervalues(c))

def zero() -> Counter:
    return Counter()

def count(t: Sequence[str],
          r: Sequence[str],
          n: int = 4
          ) -> Counter:
    """Collect statistics for a single test and reference segment."""

    stats = Counter()
    for i in range(1, n+1):
        tngrams = ngrams(t, i)
        stats['guess',i] += card(tngrams)
        stats['match',i] += card(tngrams & ngrams(r, i))
    stats['reflen'] += len(r)
    return stats

def score(stats: Counter,
          n: int = 4
          ) -> float:
    """Compute BLEU score.

    :param stats: Statistics collected using bleu.count
    :type stats: dict"""

    b = 1.
    for i in range(1, n+1):
        b *= stats['match',i]/stats['guess',i] if stats['guess',i] > 0 else 0
    b **= 0.25
    if stats['guess',1] < stats['reflen']: 
        b *= math.exp(1-stats['reflen']/stats['guess',1])
    return b


def bleu(test_path: str,
         gold_path: str
         ) -> float:
    test = [line.split() for line in open(test_path, "r", encoding="utf8")]
    gold = [line.split() for line in open(gold_path, "r", encoding="utf8")]

    c = zero()
    for t, g in zip(test, gold):
        c += count(t, g)
    bleu_score: float = score(c)
    print("BLEU:", bleu_score)
    return bleu_score


if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('test', metavar='predict', help='predicted translations')
    argparser.add_argument('gold', metavar='true', help='true translations')
    args = argparser.parse_args()

    test = [line.split() for line in open(args.test, "r", encoding="utf8")]
    gold = [line.split() for line in open(args.gold, "r", encoding="utf8")]

    c = zero()
    for t, g in zip(test, gold):
        c += count(t, g)
    print("BLEU:", score(c))
    

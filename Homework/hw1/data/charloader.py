# SYSTEM IMPORTS
from collections.abc import Sequence
import numpy


# PYTHON PROJECT IMPORTS


def load_chars_from_file(filepath: str) -> Sequence[str]:
    l: Sequence[str] = list()
    with open(filepath, "r", encoding="utf8") as f:
        #return numpy.array([w for line in f for w in line.rstrip("\n")])
        for line in f:
            line_list: Sequence[str] = list()
            for w in line.rstrip("\n"):
                line_list.append(w)
            l.append(line_list)
    return l


def load_lines_from_file(filepath: str) -> Sequence[str]:
    l: Sequence[str] = None
    with open(filepath, "r", encoding="utf8") as f:
        l = [line.rstrip("\n") for line in f]
    return l


def convert_chars_to_numpy(chars: str) -> Sequence[str]:
    return numpy.array([w for line in chars for w in line])


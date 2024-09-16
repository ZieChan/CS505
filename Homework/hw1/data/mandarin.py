# SYSTEM INCLUDES
from collections.abc import Sequence, Mapping


# PYTHON PROJECT INCLUDES
from .charloader import load_lines_from_file


def load_and_unmask_chars(charmap: Mapping[str, str],
                          data_path: str
                          ) -> Sequence[str]:
    raw_data: Sequence[str] = load_lines_from_file(data_path)
    unmasked_chars: Sequence[str] = list()
    for line in raw_data:
        unmasked_line: Sequence[str] = list()
        split_line: Sequence[str] = line.split()
        for i, token in enumerate(split_line):
            # print("i: %s, len(split_line): %s" % (i, len(split_line)))
            if token in charmap:
                unmasked_line.append(charmap[token])
            else:
                unmasked_line.append(token)
            if i < len(split_line) - 1:
                unmasked_line.append("<space>")
        unmasked_chars.append(unmasked_line)
    return unmasked_chars


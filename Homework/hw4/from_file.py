# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
import os
import sys


# PYTHON PROJECT IMPORTS


def load_annotated_data(path: str,
                        limit: int = -1
                        ) -> Tuple[Sequence[Sequence[str]], Sequence[Sequence[str]]]:
    words_corpus: Sequence[Sequence[str]] = list()
    tags_corpus: Sequence[Sequence[str]] = list()

    with open(path, "r", encoding="utf8") as f:
        word_seq: Sequence[str] = list()
        tag_seq: Sequence[str] = list()

        line_parts = None
        counter = 0

        for line in f:
            if counter == limit:
                break

            line_parts = line.split()
            if len(line_parts) > 0:
                word, tag = line_parts

                word = word.strip().rstrip()
                tag = tag.strip().rstrip()

                word_seq.append(word)
                tag_seq.append(tag)
            else:
                words_corpus.append(word_seq)
                tags_corpus.append(tag_seq)

                word_seq = list()
                tag_seq = list()
                counter += 1

        if len(word_seq) > 0:
            words_corpus.append(word_seq)
        if len(tag_seq) > 0:
            tags_corpus.append(tag_seq)

    if len(words_corpus) == 0:
        words_corpus.append(list())
    if len(tags_corpus) == 0:
        tags_corpus.append(list())
    return words_corpus, tags_corpus


def load_unannotated_data(path: str,
                          limit: int = -1
                          ) -> Sequence[Sequence[str]]:
    words = list()
    with open(path, "r") as f:
        word_line = list()
        line_parts = None
        counter = 0
        for line in f:
            if counter == limit:
                break
            line_parts = line.split()
            if len(line_parts) > 0:
                word_line.append(line.strip().rstrip())
            else:
                words.append(word_line)
                data_line = list()
                counter += 1
        if len(word_line) > 0:
            words.append(word_line)
    if len(words) == 0:
        words.append(list())
    return words


# SYSTEM IMPORTS
from collections.abc import Sequence
import numpy


import re

# 定义一个函数用来在中文字符前后加上空格
def add_spaces_around_chinese(text: str) -> str:
    # 使用正则表达式匹配所有的中文字符
    pattern = re.compile(r'[\u4e00-\u9fff]')
    # 在每个匹配的中文字符前后加上空格
    new_text = pattern.sub(lambda x: f' {x.group(0)} ', text)
    # 去除可能多余的空格
    new_text = re.sub(r'\s+', ' ', new_text).strip()
    return new_text


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
        # l = [line.rstrip("\n") for line in f]
        l = [add_spaces_around_chinese(line.rstrip("\n")) for line in f]
    return l


def convert_chars_to_numpy(chars: str) -> Sequence[str]:
    return numpy.array([w for line in chars for w in line])


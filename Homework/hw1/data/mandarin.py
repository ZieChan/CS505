# SYSTEM INCLUDES
from collections.abc import Sequence, Mapping


# PYTHON PROJECT INCLUDES
from .charloader import load_lines_from_file

# def load_and_unmask_chars(charmap: Mapping[str, str],
#                           data_path: str
#                           ) -> Sequence[str]:
#     raw_data: Sequence[str] = load_lines_from_file(data_path)

#     unmasked_chars: Sequence[str] = list()
#     for line in raw_data:
#         unmasked_line: Sequence[str] = list()
#         split_line: Sequence[str] = line.split()
#         for i, token in enumerate(split_line):
#             # print("i: %s, len(split_line): %s" % (i, len(split_line)))
#             if token in charmap:
#                 unmasked_line.append(charmap[token])
#             else:
#                 unmasked_line.append(token)
#             if i < len(split_line) - 1:
#                 unmasked_line.append("<space>")
#         print("unmasked_line:", unmasked_line)
#         unmasked_chars.append(unmasked_line)
#     return unmasked_chars


# def load_and_unmask_chars(charmap: Mapping[str, str],
#                           data_path: str
#                           ) -> Sequence[str]:
#     raw_data: Sequence[str] = load_lines_from_file(data_path)

#     unmasked_chars: Sequence[str] = list()
#     for line in raw_data:
#         unmasked_line: Sequence[str] = list()
#         split_line: Sequence[str] = line.split()

#         mapped_tokens = []
#         is_pinyin = []
#         for token in split_line:
#             if token in charmap:
#                 mapped_tokens.append(charmap[token])
#                 is_pinyin.append(True)
#             else:
#                 mapped_tokens.append(token)
#                 is_pinyin.append(False)

#         for i in range(len(mapped_tokens)):
#             unmasked_line.append(mapped_tokens[i])
#             if i < len(mapped_tokens) - 1:
#                 # if this token and the next token are both pin, there is no '<space>'
#                 if not (is_pinyin[i] and is_pinyin[i+1]):
#                     unmasked_line.append('<space>')

#         #print("unmasked_line:", unmasked_line)
#         unmasked_chars.append(unmasked_line)
#     return unmasked_chars


def load_and_unmask_chars(charmap: Mapping[str, str],
                          data_path: str
                          ) -> Sequence[str]:
    raw_data: Sequence[str] = load_lines_from_file(data_path)

    unmasked_chars: Sequence[str] = list()
    for line in raw_data:
        unmasked_line: Sequence[str] = list()
        split_line: Sequence[str] = line.split()

        mapped_tokens = []
        is_pinyin = []
        for token in split_line:
            if token in charmap:
                # 将汉字替换为拼音
                mapped_tokens.append(charmap[token])
                is_pinyin.append(True)
            else:
                # 将非拼音的 token 拆分为单个字符
                chars = list(token)
                mapped_tokens.extend(chars)
                is_pinyin.extend([False] * len(chars))
            # 在每个 token 后面添加一个单词边界标记
            mapped_tokens.append('<word_boundary>')
            is_pinyin.append(None)

        # 构建 unmasked_line，并决定何时插入 '<space>'
        for i in range(len(mapped_tokens) - 1):  # 遍历到倒数第二个元素
            if mapped_tokens[i] == '<word_boundary>':
                continue  # 跳过单词边界标记
            unmasked_line.append(mapped_tokens[i])
            # 判断是否需要插入 '<space>'
            current_is_pinyin = is_pinyin[i]
            next_is_none = is_pinyin[i + 1] is None  # 下一个是单词边界
            if next_is_none:
                # 在单词边界处插入 '<space>'，但仅在当前 token 不是拼音时插入
                if current_is_pinyin is False:
                    unmasked_line.append('<space>')
            else:
                next_is_pinyin = is_pinyin[i + 1]
                # 当拼音状态发生变化时（拼音与非拼音之间）插入 '<space>'
                if current_is_pinyin != next_is_pinyin and current_is_pinyin is False:
                    unmasked_line.append('<space>')

        print("unmasked_line:", unmasked_line)
        unmasked_chars.append(unmasked_line)
    return unmasked_chars



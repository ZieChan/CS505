# SYSTEM IMPORTS
from collections.abc import Callable, Iterable, Mapping, Sequence, Set
from typing import Type, Tuple
import numpy as np
import os
import sys


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import BigramTable, EmissionTable, START_TOKEN, END_TOKEN, UNK_TOKEN


class Base(object):
    def __init__(self: Type["Base"]):
        self.lm: BigramTable = None
        self.tm: EmissionTable = None
        self.tag_vocab: Set[str] = set()
        self.word_vocab: Set[str] = set()

    def _init_lm(self: Type["Base"],
                 tag_corpus: Sequence[Sequence["str"]],
                 init_val: float = 0.0
                 ) -> None:
        # two main differences for the alphabets of the tag model:
        #     1) The input alphabet contains the START symbol
        #     2) The output alphabet contains the STOP symbol
        tag_vocab = set()
        for tag_seq in tag_corpus:
            tag_vocab.update(tag_seq)

        self.tag_vocab = tag_vocab
        sorted_tags = sorted(tag_vocab)

        # personally I like to keep the alphabet sorted b/c it helps me read the table if I have to
        self.lm = BigramTable(sorted_tags, sorted_tags, init_val=init_val)

    def _init_tm(self: Type["Base"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        word_vocab = set()
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            word_vocab.update(word_seq)

        # add UNK to the words
        self.word_vocab = word_vocab | set([UNK_TOKEN])

        # again I prefer to keep things sorted in case I need to try and reproduce something from lecture
        # when I'm debugging
        sorted_words = sorted(self.word_vocab)
        sorted_tags = sorted(self.tag_vocab)
        self.tm = EmissionTable(sorted_tags, sorted_words, init_val=init_val)

    def lm_count_bigram(self: Type["Base"],
                        tag_corpus: Sequence[Sequence[str]]
                        ) -> None:
        # TODO: complete me!
        #   iterate through each sequence of the corpus and increment the corresponding bigram entries
        #   don't forget to increment <EOS> after each sequence!
        for tag_seq in tag_corpus:
            prev_tag = START_TOKEN
            for tag in tag_seq:
                self.lm.increment_value(prev_tag, tag)
                prev_tag = tag
            self.lm.increment_value(prev_tag, END_TOKEN)


    def _train(self: Type["Base"],
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]],
               init_val: float = 0.0
               ) -> Type["Base"]:
        self._init_lm(tag_corpus, init_val=init_val)
        self._init_tm(word_corpus, tag_corpus, init_val=init_val)
        return self

    def train_from_raw(self: Type["Base"],
                       word_corpus: Sequence[Sequence[str]],
                       tag_corpus: Sequence[Sequence[str]],
                       limit: int = -1
                       ) -> Type["Base"]:
        if limit > -1:
            word_corpus = word_corpus[:limit]
            tag_corpus = tag_corpus[:limit]
        return self._train(word_corpus, tag_corpus)

    def train_from_file(self: Type["Base"],
                        file_path: str,
                        limit: int = -1
                        ) -> Type["Base"]:
        word_corpus, tag_corpus = load_annotated_data(file_path, limit=limit)
        return self._train_from_raw(word_corpus, tag_corpus, limit=limit)

    def parse_word_list(self: Type["Base"],
                        word_list: Sequence[str]
                        ) -> Sequence[str]:
        parsed_list: Sequence[str] = list(word_list)
        for i, w in enumerate(parsed_list):
            if w not in self.word_vocab:
                parsed_list[i] = UNK_TOKEN
        return parsed_list

    # def viterbi_traverse(self: Type["Base"],
    #                      word_list: Sequence[str],
    #                      init_func_ptr: Callable[[], LayeredGraph],
    #                      update_func_ptr: Callable[[str, str, str, float, LayeredGraph], None],
    #                      ) -> None:
    #     # TODO: complete me!
    #     # This function should implement the viterbi traversal on a LayeredGraph object
    #     #   an initialized LayeredGraph object will be produced by init_func_ptr
    #     #   and your code should populate this graph. To be clear, the traversal code here will
    #     #   need to allocate new layers on the graph, but to populate the newly created layer,
    #     #   you can call update_func_ptr

    #     """
    #     This method will act like cky traverse did from the previous
    #     assignment: it will only iterate over the data structures of a particular viterbi flavor. This
    #     method takes three arguments, the list of words, a function pointer that will produce an
    #     initialized LayeredGraph object (i.e. the data structure to populate), and another function
    #     pointer that will update a vertex within a layer of the LayeredGraph object. Your method
    #     is responsible for allocating new layers to the graph, and for iterating over each vertex
    #     within a layer.
    #     """

    #     """
    #     def _cky_traverse(self: Type["Parser"],
    #                 list_of_words: Sequence[str],
    #                 update_func_ptr: Callable[[Tuple[int, int],       # coordinate of the target cell (in chart(s))
    #                                             Tuple[int, int],       # coordinate of one cell c_ik (in chart(s))
    #                                             Tuple[int, int]],      # coordinate of one cell c_jk (in chart(s))
    #                                             None]
    #                 ) -> None:
    #         n = len(list_of_words)
    #         chart_size = n
    #         for row in range(1, chart_size + 1):
    #             for col in range(n - row + 1):
    #                 for leftRow in range(0, row):
    #                     rightRow = row - leftRow - 1
    #                     leftCol = col
    #                     rightCol = col + leftRow + 1
    #                     update_func_ptr((row, col), (leftRow, leftCol), (rightRow, rightCol))
    #     """

    #     # Initialize the LayeredGraph object
    #     graph = init_func_ptr()
    #     # graph.add_layer()
    #     # for tag in self.tag_vocab:
    #     #     graph.add_node(tag, self.lm.get_value(START_TOKEN, tag), None)
    #     # graph.add_node(START_TOKEN, 0.0, None)

    #     # Traverse the LayeredGraph object

    #     # for word in word_list:
    #     #     graph.add_layer()
    #     #     for tag in self.tag_vocab:
    #     #         max_log_prob = -np.inf
    #     #         max_prev_tag = None
    #     #         for prev_tag in self.tag_vocab:
    #     #             log_prob = graph.get_node_in_layer(prev_tag)[0] + np.log(self.lm.get_value(prev_tag, tag)) + np.log(self.tm.get_value(tag, word))
    #     #             if log_prob > max_log_prob:
    #     #                 max_log_prob = log_prob
    #     #                 max_prev_tag = prev_tag
    #     #         graph.add_node(tag, max_log_prob, max_prev_tag)

    #     graph.add_layer()
    #     graph.add_node(START_TOKEN, 0.0, None)

    #     for i, word in enumerate(word_list):
    #         graph.add_layer()
    #         for tag in self.tag_vocab:
    #             max_log_prob = -np.inf
    #             max_prev_tag = None
    #             if i == 0:
    #                 prev_tag = START_TOKEN
    #                 log_prob = graph.get_node_in_layer(prev_tag)[0] + np.log(self.lm.get_value(prev_tag, tag)) + np.log(self.tm.get_value(tag, word))
    #                 graph.add_node(tag, log_prob, prev_tag)
    #             else:
    #                 for prev_tag in self.tag_vocab:
    #                     log_prob = graph.get_node_in_layer(prev_tag)[0] + np.log(self.lm.get_value(prev_tag, tag)) + np.log(self.tm.get_value(tag, word))
    #                     if log_prob > max_log_prob:
    #                         max_log_prob = log_prob
    #                         max_prev_tag = prev_tag
    #                 graph.add_node(tag, max_log_prob, max_prev_tag)



    #     return graph
        
    def viterbi_traverse(self: Type["Base"],
                        word_list: Sequence[str],
                        init_func_ptr: Callable[[], LayeredGraph],
                        update_func_ptr: Callable[[str, str, str, float, LayeredGraph], None],
                        ) -> LayeredGraph:
        
        # TODO: complete me!
        # This function should implement the viterbi traversal on a LayeredGraph object
        #   an initialized LayeredGraph object will be produced by init_func_ptr
        #   and your code should populate this graph. To be clear, the traversal code here will
        #   need to allocate new layers on the graph, but to populate the newly created layer,
        #   you can call update_func_ptr

        # Initialize the graph
        graph = init_func_ptr()

        graph.add_layer()
        # Add the START_TOKEN node with a log probability of 0
        graph.add_node(START_TOKEN, 0.0, None)

        num_words = len(word_list)
        tag_vocab = self.tag_vocab  # set of all tags

        # For each word in the sentence
        for t in range(num_words):
            word = word_list[t]
            # Allocate new layer
            graph.add_layer()
            # For each current_state (tag)
            for current_state in tag_vocab:
                # For each prev_state in the previous layer
                prev_layer_index = len(graph.node_layers) - 2  # previous layer index
                prev_LAYER = graph.node_layers[prev_layer_index]
                prev_states = prev_LAYER.keys()
                for prev_state, (path_cost, _) in prev_LAYER.items():
                    update_func_ptr(current_state, word, prev_state, path_cost, graph)

        # Handle termination by adding END_TOKEN
        graph.add_layer()
        current_state = END_TOKEN
        # prev_layer_index = len(graph.node_layers) - 2
        PREV_LAYER = graph.node_layers[-2]
        for prev_state, (path_cost, _) in PREV_LAYER.items():
            update_func_ptr(current_state, None, prev_state, path_cost, graph)

        # return graph  # Return the populated graph for backtracing
        self.graph = graph
        return graph







    def predict_sentence(self: Type["Base"],
                         word_list: Sequence[str]
                         ) -> Sequence[str]:
        word_list = self.parse_word_list(word_list)
        path, log_prob = self.viterbi(word_list)
        return path

    def predict(self: Type["Base"],
                word_corpus: Sequence[Sequence[str]]
                ) -> Iterable[Sequence[str]]:
        for word_list in word_corpus:
            yield self.predict_sentence(word_list)


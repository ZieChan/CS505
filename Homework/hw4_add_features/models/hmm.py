# SYSTEM IMPORTS
from collections.abc import Sequence
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
from base import Base
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class HMM(Base):
    def __init__(self: Type["HMM"]) -> None:
        super().__init__()

    def _train_lm(self: Type["HMM"],
                  tag_corpus: Sequence[Sequence[str]]
                  ) -> None:
        self.lm_count_bigram(tag_corpus)
        self.lm.normalize_cond()

    def _train_tm(self: Type["HMM"],
                  word_corpus: Sequence[Sequence[str]],
                  tag_corpus: Sequence[Sequence[str]]
                  ) -> None:
        for word_seq, tag_seq in zip(word_corpus, tag_corpus):
            for w, t in zip(word_seq, tag_seq):
                self.tm.increment_value(t, w, val=1)
            self.tm.increment_value(END_TOKEN, END_TOKEN, val=1)
        self.tm.normalize_cond(add=0.1)

    def _train(self: Type["HMM"],
               word_corpus: Sequence[Sequence[str]],
               tag_corpus: Sequence[Sequence[str]]
               ) -> Type["HMM"]:
        super()._train(word_corpus, tag_corpus)
        self._train_lm(tag_corpus)
        self._train_tm(word_corpus, tag_corpus)
        return self

    # def viterbi(self: Type["HMM"],
    #             word_list) -> Tuple[Sequence[str], float]:

    #     # TODO: complete me!
    #     # this method should return the most probable path along with the logprob of the most probable path
    #     """
    #     This method will implement full viterbi decoding. Given a list of words as
    #     input, it should create two function pointers, one to initialize a LayeredGraph object, and
    #     another to update a layer within the object. You should call viterbi traverse with these
    #     function pointers, and then assemble the most probable path through the graph. Your
    #     method should return the most probable path first, and the logprob of the most probable
    #     path second.
    #     """

    #     def init_func_ptr() -> LayeredGraph:
    #         graph = LayeredGraph()
    #         return graph
        
    #     # update a vertex within a layer of the LayeredGraph object
    #     def update_func_ptr(word: str, prev_tag: str, tag: str, log_prob: float, graph: LayeredGraph) -> None:
    #         graph.add_layer()
    #         for t in self.tag_alphabet:
    #             max_prob = float("-inf")
    #             max_tag = None
    #             for prev_t in self.tag_alphabet:
    #                 prev_prob, _ = graph.get_node_in_layer(prev_t)
    #                 prob = prev_prob + self.lm.get_value(prev_t, t) + self.tm.get_value(word, t)
    #                 if prob > max_prob:
    #                     max_prob = prob
    #                     max_tag = prev_t
    #             graph.add_node(t, max_prob, max_tag)


    #     graph = self.viterbi_traverse(word_list, init_func_ptr, update_func_ptr)

    #     # get the most probable path
    #     path = []
    #     log_prob = 0.0
    #     for layer in graph.node_layers:
    #         node = graph.get_node_in_layer(max(layer, key=lambda x: layer[x][0]))
    #         path.append(node[1])
    #         log_prob += node[0]

    #     return path, log_prob

    def viterbi(self: Type["HMM"],
                word_list: Sequence[str]) -> Tuple[Sequence[str], float]:
        
        # TODO: complete me!
        # this method should return the most probable path along with the logprob of the most probable path

        # Preprocess the word list to handle unknown tokens
        word_list = self.parse_word_list(word_list)

        # Define the initialization function for the LayeredGraph
        def init_func() -> LayeredGraph:
            return LayeredGraph()

        # Define the update function for each node
        def update_func(current_state: str, word: str, prev_state: str, path_cost: float, graph: LayeredGraph):
            PREV_layer_index = len(graph.node_layers) - 2
            PREV_layer = graph.node_layers[PREV_layer_index]
            curr_layer_index = len(graph.node_layers) - 1
            curr_layer = graph.node_layers[curr_layer_index]

            # Get the previous state's log probability
            if prev_state in PREV_layer:
                prev_log_prob, _ = PREV_layer[prev_state]
            else:
                # If the previous state is not in the layer, skip
                return

            # Calculate the log transition probability
            trans_prob = self.lm.get_value(prev_state, current_state)
            trans_log_prob = np.log(trans_prob) if trans_prob > 0 else float('-inf')

            # Calculate the log emission probability
            if current_state == END_TOKEN:
                # emit_log_prob = 0.0  # No emission probability for END_TOKEN
                emit_prob = self.tm.get_value(prev_state, END_TOKEN)
                emit_log_prob = np.log(emit_prob) if emit_prob > 0 else float('-inf')
            else:
                emit_prob = self.tm.get_value(current_state, word)
                emit_log_prob = np.log(emit_prob) if emit_prob > 0 else float('-inf')

            # Total log probability for the current state
            total_log_prob = prev_log_prob + trans_log_prob + emit_log_prob
            # print(f"word: {word}, prev_state: {prev_state}, current_state: {current_state}, total_log_prob: {total_log_prob}")

            # Check if the current state is already in the current layer
            if current_state not in curr_layer or total_log_prob > curr_layer[current_state][0]:
                # Update the node with the higher log probability and set the backpointer
                graph.add_node(current_state, total_log_prob, prev_state)

        # Perform the Viterbi traversal
        graph = self.viterbi_traverse(word_list, init_func, update_func)

        # Backtrace to find the most probable path
        path = []
        curr_state = END_TOKEN
        curr_layer_index = len(graph.node_layers) - 1 # self. graph

        # Retrieve the log probability of the END_TOKEN node
        curr_layer = graph.node_layers[curr_layer_index] # self. graph
        if curr_state in curr_layer:
            total_log_prob, prev_state = curr_layer[curr_state]
        else:
            # If END_TOKEN is not in the last layer, return empty path and negative infinity log probability
            return [], float('-inf')
        
        LAST_log_prob = total_log_prob

        # Backtrace through the layers to build the path
        for layer_index in range(len(graph.node_layers) - 1, 0, -1): # self. graph
            curr_layer = graph.node_layers[layer_index] # self. graph
            if curr_state in curr_layer:
                total_log_prob, prev_state = curr_layer[curr_state]
                # print(f"current total_log_prob: {total_log_prob}, prev_state: {prev_state}")
                if curr_state != END_TOKEN:
                    path.append(curr_state)
                curr_state = prev_state
            else:
                break  # If the state is not found, terminate backtracing

        # Reverse the path to get the correct order
        path.reverse()

        return path, LAST_log_prob

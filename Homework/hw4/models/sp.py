# SYSTEM IMPORTS
from collections.abc import Callable, Sequence
from typing import Tuple, Type
from tqdm import tqdm
import numpy as np
import os
import sys
import pickle
from tables import BigramTable, EmissionTable


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "..")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from base import Base
from from_file import load_annotated_data
from layered_graph import LayeredGraph
from tables import START_TOKEN, END_TOKEN, UNK_TOKEN


class SP(Base):
    def __init__(self: Type["SP"]) -> None:
        super().__init__()

    def _init_tm(self: Type["SP"],
                 word_corpus: Sequence[Sequence[str]],
                 tag_corpus: Sequence[Sequence[str]],
                 init_val: float = 0.0
                 ) -> None:
        super()._init_tm(word_corpus, tag_corpus, init_val=init_val)

    # def sp_training_algorithm(self: Type["SP"],
    #                           word_corpus: Sequence[Sequence[str]],
    #                           tag_corpus: Sequence[Sequence[str]],
    #                           ) -> Tuple[int, int]:
    #     # TODO: complet me!
    #     #    This method should implement the structured perceptron training algorithm,
    #     #    and it should return a pair of ints (num_correct, num_total).
    #     #       num_correct should contain the number of tags that were correctly predicted
    #     #       num_total should contain the number of total tags predicted

    def sp_training_algorithm(self: Type["SP"],
                            word_corpus: Sequence[Sequence[str]],
                            tag_corpus: Sequence[Sequence[str]],
                            ) -> Tuple[int, int]:
        num_correct = 0
        num_total = 0

        for word_seq, tag_seq in tqdm(zip(word_corpus, tag_corpus), total=len(word_corpus)):
            # Predict the best tag sequence using the current weights
            # predicted_tags, _ = self.viterbi(word_seq)
            predicted_tags = self.predict_sentence(word_seq)

            # Update num_correct and num_total
            num_total += len(tag_seq)
            num_correct += sum(p == t for p, t in zip(predicted_tags, tag_seq))

            # If the predicted sequence is not equal to the true sequence, update the weights
            if predicted_tags != tag_seq:
                # Update the weights
                # For transitions
                prev_true_tag = START_TOKEN
                prev_pred_tag = START_TOKEN

                for i in range(len(tag_seq)):
                    true_tag = tag_seq[i]
                    pred_tag = predicted_tags[i]
                    word = word_seq[i]

                    # Update transition weights
                    self.lm.increment_value(prev_true_tag, true_tag, val=1)
                    self.lm.increment_value(prev_pred_tag, pred_tag, val=-1)

                    # Update emission weights
                    self.tm.increment_value(true_tag, word, val=1)
                    self.tm.increment_value(pred_tag, word, val=-1)

                    prev_true_tag = true_tag
                    prev_pred_tag = pred_tag

                # Handle the transition to END_TOKEN
                self.lm.increment_value(prev_true_tag, END_TOKEN, val=1)
                self.lm.increment_value(prev_pred_tag, END_TOKEN, val=-1)

        return num_correct, num_total



    def _train(self: Type["SP"],
               train_word_corpus: Sequence[Sequence[str]],
               train_tag_corpus: Sequence[Sequence[str]],
               dev_word_corpus: Sequence[Sequence[str]] = None,
               dev_tag_corpus: Sequence[Sequence[str]] = None,
               max_epochs: int = 20,
               converge_error: float = 1e-4,
               log_function: Callable[[Type["SP"], int, Tuple[int, int], Tuple[int, int]], None] = None,
               ) -> Type["SP"]:
        super()._train(train_word_corpus, train_tag_corpus)

        # patience: int = 5
        # best_fb1: float = 0.0
        # epochs_without_improvement: int = 0

        result_dir = "generated"
        os.makedirs(result_dir, exist_ok=True)

        fb1 = 0.0
        min_fb1 = 15
        best_fb1 = 0.0
        best_epoch = -1
        model_save_path: str = "best_model.pkl"


        current_epoch: int = 0
        current_accuracy: float = 1.0
        prev_accuracy: float = 1.0
        percent_rel_error: float = 1.0

        while current_epoch < max_epochs and fb1 < min_fb1:

            train_correct, train_total = self.sp_training_algorithm(train_word_corpus, train_tag_corpus)
            dev_correct, dev_total = 0, 0

            if dev_word_corpus is not None and dev_tag_corpus is not None:

                for i, predicted_tags in enumerate(self.predict(dev_word_corpus)):
                    true_tags = dev_tag_corpus[i]
                    dev_total += len(true_tags)
                    dev_correct += np.sum(np.array(true_tags) == np.array(predicted_tags))

            if log_function is not None:
                log_function(self, current_epoch, (train_correct, train_total), (dev_correct, dev_total))

            # epoch_correct = train_correct if dev_word_corpus is None or dev_tag_corpus is None else dev_correct
            # epoch_total = train_total if dev_word_corpus is None or dev_tag_corpus is None else dev_total

            # prev_accuracy = current_accuracy
            # current_accuracy = float(epoch_correct) / float(epoch_total)
            # percent_rel_error = abs(prev_accuracy - current_accuracy) / prev_accuracy

            # current_fb1 = 2 * (current_accuracy / (1 + current_accuracy))
            # print(f"current accuracy: {prev_accuracy}")

            result_file = os.path.join(result_dir, f"epoch-{current_epoch}.result")

            with open(result_file, 'r') as f:
                lines = f.readlines()
                if len(lines) >= 2:
                    fb1_line = lines[1]
                    # Extract FB1 score
                    fb1 = self.extract_fb1_from_line(fb1_line)
                    print(f"FB1 score on dev set: {fb1}%")

                    if fb1 > best_fb1:
                        best_fb1 = fb1
                        best_epoch = current_epoch
                        self.save(model_save_path)
                        print(f"Model Saved!! Best FB1 score so far: {best_fb1}% at epoch {best_epoch}")

                    # Early stopping based on FB1 score
                    if fb1 >= min_fb1:
                        print(f"FB1 score {fb1}% exceeds {min_fb1}%. Stopping training.")
                        break
                    else:
                        print(f"FB1 score {fb1}% is below {min_fb1}%. Continuing training.")
                else:
                    print(f"Result file {result_file} is not in the expected format.")
            
            current_epoch += 1


        return self
    
    def extract_fb1_from_line(self, line: str) -> float:
        """
        Extract the FB1 score from the given line.
        Expected line format:
        "accuracy:  93.12%; precision:  32.65%; recall:   2.42%; FB1:   4.51"
        """
        fb1 = 0.0
        try:
            parts = line.strip().split(';')
            for part in parts:
                if 'FB1:' in part:
                    fb1_str = part.strip().split(':')[-1].strip().replace('%', '')
                    fb1 = float(fb1_str)
                    break
        except ValueError:
            print(f"Could not parse FB1 score from line: {line}")
        return fb1

    # def _train(self: Type["SP"],
    #         train_word_corpus: Sequence[Sequence[str]],
    #         train_tag_corpus: Sequence[Sequence[str]],
    #         dev_word_corpus: Sequence[Sequence[str]] = None,
    #         dev_tag_corpus: Sequence[Sequence[str]] = None,
    #         max_epochs: int = 100,
    #         converge_error: float = 1e-3,
    #         log_function: Callable[[Type["SP"], int, Tuple[int, int], Tuple[int, int]], None] = None,
    #         patience: int = 5,
    #         ) -> Type["SP"]:
    #     from sklearn.metrics import precision_recall_fscore_support

    #     super()._train(train_word_corpus, train_tag_corpus)
        
    #     current_epoch: int = 0
    #     best_fb1: float = 0.0
    #     epochs_without_improvement: int = 0

    #     while current_epoch < max_epochs and epochs_without_improvement < patience:
    #         train_correct, train_total = self.sp_training_algorithm(train_word_corpus, train_tag_corpus)
    #         dev_correct, dev_total = 0, 0

    #         if dev_word_corpus is not None and dev_tag_corpus is not None:
    #             true_tags_flat = []
    #             predicted_tags_flat = []
    #             for i, predicted_tags in enumerate(self.predict(dev_word_corpus)):
    #                 true_tags = dev_tag_corpus[i]
    #                 true_tags_flat.extend(true_tags)
    #                 predicted_tags_flat.extend(predicted_tags)
    #             precision, recall, f1, _ = precision_recall_fscore_support(true_tags_flat, predicted_tags_flat, average='macro')
    #             current_fb1 = f1 * 100  # Convert to percentage
    #         else:
    #             # Use training data if validation data is not provided
    #             current_fb1 = (float(train_correct) / float(train_total)) * 100

    #         if log_function is not None:
    #             log_function(self, current_epoch, (train_correct, train_total), (dev_correct, dev_total))

    #         # Early stopping based on FB1 score
    #         if current_fb1 > best_fb1 + converge_error:
    #             best_fb1 = current_fb1
    #             epochs_without_improvement = 0
    #         else:
    #             epochs_without_improvement += 1
            
    #         if current_fb1 <= 16:
    #             epochs_without_improvement = 0

    #         current_epoch += 1

    #     return self



    def train_from_raw(self: Type["SP"],
                       train_word_corpus: Sequence[Sequence[str]],
                       train_tag_corpus: Sequence[Sequence[str]],
                       dev_word_corpus: Sequence[Sequence[str]] = None,
                       dev_tag_corpus: Sequence[Sequence[str]] = None,
                       max_epochs: int = 20,
                       converge_error: float = 1e-4,
                       log_function: Callable[[int, Tuple[int, int], Tuple[int, int]], None] = None
                       ) -> None:
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
       
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    def train_from_file(self: Type["SP"],
                        train_path: str,
                        dev_path: str = None,
                        max_epochs=20,
                        converge_error: float = 1e-4,
                        limit: int = -1
                        ) -> None:
        train_word_corpus, train_tag_corpus = load_annotated_data(train_path, limit=limit)
        dev_word_corpus, dev_tag_corpus = None, None
        if dev_path is not None:
            dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_path, limit=limit)
        self._train(train_word_corpus, train_tag_corpus,
                    dev_word_corpus=dev_word_corpus, dev_tag_corpus=dev_tag_corpus,
                    max_epochs=max_epochs,
                    converge_error=converge_error,
                    log_function=log_function)

    # def viterbi(self: Type["SP"],
    #             word_list: Sequence[str]
    #             ) -> Tuple[Sequence[str], float]:

    #     # TODO: complete me!
    #     # This method should look identical to your HMM viterbi!

    def viterbi(self: Type["SP"],
                word_list: Sequence[str]
                ) -> Tuple[Sequence[str], float]:
        # Preprocess the word list to handle unknown tokens
        word_list = self.parse_word_list(word_list)

        # Define the initialization function for the LayeredGraph
        def init_func() -> LayeredGraph:
            graph = LayeredGraph()
            # No need to add the START_TOKEN node here since it's added in viterbi_traverse
            return graph

        # Define the update function for each node
        def update_func(current_state: str, word: str, prev_state: str, prev_score: float, graph: LayeredGraph):
            curr_layer_index = len(graph.node_layers) - 1
            curr_layer = graph.node_layers[curr_layer_index]

            # Get transition weight
            trans_weight = self.lm.get_value(prev_state, current_state)

            # Get emission weight
            if current_state == END_TOKEN:
                emit_weight = 0.0  # No emission weight for END_TOKEN
            else:
                emit_weight = self.tm.get_value(current_state, word)

            # Total score
            total_score = prev_score + trans_weight + emit_weight

            # Update the node in the current layer
            if current_state not in curr_layer or total_score > curr_layer[current_state][0]:
                graph.add_node(current_state, total_score, prev_state)

        # Perform the Viterbi traversal
        graph = self.viterbi_traverse(word_list, init_func, update_func)

        # Backtrace to find the most probable path
        path = []
        curr_state = END_TOKEN
        curr_layer_index = len(graph.node_layers) - 1

        # Retrieve the score of the END_TOKEN node
        curr_layer = graph.node_layers[curr_layer_index]
        if curr_state in curr_layer:
            total_score, prev_state = curr_layer[curr_state]
        else:
            # If END_TOKEN is not in the last layer, return empty path and negative infinity score
            return [], float('-inf')

        LAST_score = total_score

        # Backtrace through the layers to build the path
        for layer_index in range(len(graph.node_layers) - 1, 0, -1):
            curr_layer = graph.node_layers[layer_index]
            if curr_state in curr_layer:
                total_score, prev_state = curr_layer[curr_state]
                if curr_state != END_TOKEN:
                    path.append(curr_state)
                curr_state = prev_state
            else:
                break  # If the state is not found, terminate backtracing

        # Reverse the path to get the correct order
        path.reverse()

        return path, LAST_score

    def save(self, model_path: str) -> None:
        """
        Save the model's weights and mappings to a file.
        """
        model_data = {
            'lm_table': self.lm.table,
            'lm_in_map': self.lm.in_map,
            'lm_out_map': self.lm.out_map,
            'tm_table': self.tm.table,
            'tm_in_map': self.tm.in_map,
            'tm_out_map': self.tm.out_map,
            'tag_vocab': self.tag_vocab,
            'word_vocab': self.word_vocab,
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load the model's weights and mappings from a file.
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Restore vocabularies
        self.tag_vocab = model_data['tag_vocab']
        self.word_vocab = model_data['word_vocab']
        
        # Re-initialize lm and tm with the saved mappings and tables
        self.lm = BigramTable(set(), set())
        self.lm.table = model_data['lm_table']
        self.lm.in_map = model_data['lm_in_map']
        self.lm.out_map = model_data['lm_out_map']
        
        self.tm = EmissionTable(set(), set())
        self.tm.table = model_data['tm_table']
        self.tm.in_map = model_data['tm_in_map']
        self.tm.out_map = model_data['tm_out_map']
        
        print(f"Model loaded from {model_path}")



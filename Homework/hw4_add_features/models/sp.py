# SYSTEM IMPORTS
from collections.abc import Callable, Sequence
from typing import Tuple, Type, Dict
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

from eval.do_eval import write_output_and_evaluate


class SP(Base):
    def __init__(self: Type["SP"]) -> None:
        super().__init__()
        # Initialize a dictionary to hold feature weights
        self.feature_weights: Dict[str, float] = {}  # type: ignore

    def extract_features(self, prev_tag: str, current_tag: str, word: str) -> Sequence[str]:
        """
        Extract features based on the previous tag, current tag, and current word.
        """
        features = []
        
        # Transition feature: previous tag -> current tag
        features.append(f"TRANS_{prev_tag}_{current_tag}")
        
        # Emission feature: current tag emits current word
        if word is not None:
            features.append(f"EMIT_{current_tag}_{word}")
            # Tag-Word pair feature
            features.append(f"TAGWORD_{current_tag}_{word}")
        
        # Additional features can be added here
        # Example: prefix of the word
        if word is not None and len(word) > 3:
            prefix = word[:3].lower()
            features.append(f"PREF_{current_tag}_{prefix}")
        
        # Example: suffix of the word
        if word is not None and len(word) > 3:
            suffix = word[-3:].lower()
            features.append(f"SUFF_{current_tag}_{suffix}")
        
        return features

    def score(self, prev_tag: str, current_tag: str, word: str) -> float:
        """
        Compute the score for transitioning from prev_tag to current_tag and emitting word.
        """
        score = 0.0
        features = self.extract_features(prev_tag, current_tag, word)
        for feature in features:
            score += self.feature_weights.get(feature, 0.0)
        return score

    def viterbi(self: Type["SP"],
                word_list: Sequence[str]
                ) -> Tuple[Sequence[str], float]:
        # Preprocess the word list to handle unknown tokens
        word_list = self.parse_word_list(word_list)

        # Initialize the Viterbi variables
        V = [{}]  # Viterbi matrix: list of dicts
        path = {}  # Path dictionary

        # Initialize with START_TOKEN
        V[0][START_TOKEN] = 0.0
        path[START_TOKEN] = []

        # Iterate over each word in the sentence
        for i, word in enumerate(word_list):
            V.append({})
            new_path = {}

            for current_tag in self.tag_vocab:
                best_score = -np.inf
                best_prev_tag = None

                for prev_tag in V[i]:
                    transition_score = self.score(prev_tag, current_tag, word)
                    score = V[i][prev_tag] + transition_score

                    if score > best_score:
                        best_score = score
                        best_prev_tag = prev_tag

                V[i + 1][current_tag] = best_score
                new_path[current_tag] = path[best_prev_tag] + [current_tag]

            path = new_path

        # Handle the transition to END_TOKEN
        best_score = -np.inf
        best_last_tag = None
        for prev_tag in V[-1]:
            transition_score = self.score(prev_tag, END_TOKEN, None)
            score = V[-1][prev_tag] + transition_score
            if score > best_score:
                best_score = score
                best_last_tag = prev_tag

        final_path = path[best_last_tag] if best_last_tag else []
        final_score = best_score

        return final_path, final_score

    def sp_training_algorithm(self: Type["SP"],
                              word_corpus: Sequence[Sequence[str]],
                              tag_corpus: Sequence[Sequence[str]],
                              ) -> Tuple[int, int]:
        num_correct = 0
        num_total = 0

        for word_seq, tag_seq in tqdm(zip(word_corpus, tag_corpus), total=len(word_corpus), desc="Training"):
            # Predict the best tag sequence using the current weights
            predicted_tags, _ = self.viterbi(word_seq)

            # Update num_correct and num_total
            num_total += len(tag_seq)
            num_correct += sum(p == t for p, t in zip(predicted_tags, tag_seq))

            # If the predicted sequence is not equal to the true sequence, update the weights
            if predicted_tags != tag_seq:
                # Add START_TOKEN and END_TOKEN to the tag sequences for transitions
                true_tags = [START_TOKEN] + tag_seq + [END_TOKEN]
                pred_tags = [START_TOKEN] + predicted_tags + [END_TOKEN]
                words = word_seq + [None]  # None for END_TOKEN emission

                for i in range(1, len(true_tags)):
                    # True path features
                    true_prev_tag = true_tags[i - 1]
                    true_current_tag = true_tags[i]
                    true_word = words[i - 1]
                    true_features = self.extract_features(true_prev_tag, true_current_tag, true_word)

                    # Predicted path features
                    pred_prev_tag = pred_tags[i - 1]
                    pred_current_tag = pred_tags[i]
                    pred_word = words[i - 1]
                    pred_features = self.extract_features(pred_prev_tag, pred_current_tag, pred_word)

                    # Update weights: true features +=1, predicted features -=1
                    for feature in true_features:
                        self.feature_weights[feature] = self.feature_weights.get(feature, 0.0) + 1.0

                    for feature in pred_features:
                        self.feature_weights[feature] = self.feature_weights.get(feature, 0.0) - 1.0

        return num_correct, num_total

    def _train(self: Type["SP"],
               train_word_corpus: Sequence[Sequence[str]],
               train_tag_corpus: Sequence[Sequence[str]],
               dev_word_corpus: Sequence[Sequence[str]] = None,
               dev_tag_corpus: Sequence[Sequence[str]] = None,
               max_epochs: int = 20,
               converge_error: float = 1e-4,
               log_function: Callable[[Type["SP"], int, Tuple[int, int], Tuple[int, int]], None] = None,
               model_save_path: str = "best_model.pkl"  # Path to save the best model
               ) -> Type["SP"]:
        super()._train(train_word_corpus, train_tag_corpus)

        current_epoch: int = 0

        # Early stopping parameters
        min_fb1: float = 100   # Stop training if FB1 score exceeds this value
        best_fb1: float = 0.0   # Track the best FB1 score
        best_epoch: int = -1    # Epoch at which the best FB1 was achieved

        # Create a directory to store result files
        result_dir = "generated"
        os.makedirs(result_dir, exist_ok=True)

        while current_epoch < max_epochs:
            print(f"Epoch {current_epoch + 1}/{max_epochs}")

            # Training step
            train_correct, train_total = self.sp_training_algorithm(train_word_corpus, train_tag_corpus)

            # Evaluate on development set
            if dev_word_corpus is not None and dev_tag_corpus is not None:
                dev_predicted_corpus = []
                for word_list in dev_word_corpus:
                    predicted_tags = self.predict_sentence(word_list)
                    dev_predicted_corpus.append(predicted_tags)

                # Write predictions and evaluate
                dev_out_path = os.path.join(result_dir, f"epoch-{current_epoch}.predictions")
                result_file = os.path.join(result_dir, f"epoch-{current_epoch}.result")

                # Write predictions to file and evaluate
                write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus, outfile=result_file)

                # Read FB1 score from the result file
                with open(result_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 2:
                        fb1_line = lines[1]
                        # Extract FB1 score
                        fb1 = self.extract_fb1_from_line(fb1_line)
                        print(f"FB1 score on dev set: {fb1}%")

                        # Save the model if FB1 score is the best so far
                        if fb1 > best_fb1:
                            best_fb1 = fb1
                            best_epoch = current_epoch
                            self.save(model_save_path)
                            print(f"New best FB1 score {fb1}% achieved at epoch {current_epoch}. Model saved.")

                        # Early stopping based on FB1 score
                        if fb1 >= min_fb1:
                            print(f"FB1 score {fb1}% exceeds {min_fb1}%. Stopping training.")
                            break
                    else:
                        print(f"Result file {result_file} is not in the expected format.")

                # Optionally, log the results
                if log_function is not None:
                    log_function(self, current_epoch, (train_correct, train_total), (0, 0))  # Modify as needed
            else:
                # If no development set, we can't perform early stopping
                print("No development set provided. Continuing training without early stopping.")

            current_epoch += 1

        # Load the best model before returning
        if os.path.exists(model_save_path):
            self.load(model_save_path)
            print(f"Loaded the best model from epoch {best_epoch} with FB1 score {best_fb1}%.")
        else:
            print("No best model found. Returning the last trained model.")

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

    def save(self, model_path: str) -> None:
        """
        Save the model's feature weights and necessary mappings to a file.
        """
        model_data = {
            'feature_weights': self.feature_weights,
            'tag_vocab': self.tag_vocab,
            'word_vocab': self.word_vocab,
        }
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str) -> None:
        """
        Load the model's feature weights and necessary mappings from a file.
        """
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        # Restore vocabularies
        self.tag_vocab = model_data['tag_vocab']
        self.word_vocab = model_data['word_vocab']

        # Restore feature weights
        self.feature_weights = model_data['feature_weights']

        print(f"Model loaded from {model_path}")

    def predict_sentence(self: Type["SP"], word_list: Sequence[str]) -> Sequence[str]:
        """
        Predict the best tag sequence for a given list of words.
        """
        predicted_tags, _ = self.viterbi(word_list)
        return predicted_tags

    def predict(self: Type["SP"],
                word_corpus: Sequence[Sequence[str]]
                ) -> Sequence[Sequence[str]]:
        """
        Predict tag sequences for a corpus of word lists.
        """
        return [self.predict_sentence(word_list) for word_list in word_corpus]
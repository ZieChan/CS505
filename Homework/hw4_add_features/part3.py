# SYSTEM IMPORTS
from collections.abc import Sequence
from typing import Tuple
import itertools
import os
import sys
from tqdm import tqdm


_cd_ = os.path.abspath(os.path.dirname(__file__))
for _dir_ in [_cd_, os.path.join(_cd_, "models"), os.path.join(_cd_, "eval")]:
    if _dir_ not in sys.path:
        sys.path.append(_dir_)
del _cd_


# PYTHON PROJECT IMPORTS
from from_file import load_annotated_data
from eval.do_eval import write_output_and_evaluate
from hmm import HMM
from sp import SP

def partb() -> float:
    model = SP()
    model.load("best_model.pkl")
    
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    test_data_path = os.path.join(data_dir, "test")

    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    dev_out_path = os.path.join(generated_dir, "test.out")

    dev_word_corpus, dev_tag_corpus = load_annotated_data(test_data_path)

    print_cap: int = 5 
    dev_predicted_corpus = []
    num_examples = len(dev_word_corpus)

    for i, predicted_seq in enumerate(model.predict(dev_word_corpus)):
        dev_predicted_corpus.append(predicted_seq)
        if i < print_cap:
            print(f"Sentence {i+1}:")
            for w, p, a in zip(dev_word_corpus[i], dev_tag_corpus[i], predicted_seq):
                print(f"{w}\t{p}\t{a}")
            print()

    eval_output_path = os.path.join(generated_dir, 'evaluation_result.txt')

    write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus,
                                      outfile=eval_output_path)

    fb1 = parse_fb1_from_result_file(eval_output_path)
    return fb1

def parse_fb1_from_result_file(result_file_path: str) -> float:
    """
    Extract the FB1 score from the evaluation results file.
    """
    fb1 = 0.0
    try:
        with open(result_file_path, 'r') as f:
            for line in f:
                if 'FB1:' in line:
                    parts = line.strip().split(';')
                    for part in parts:
                        if 'FB1:' in part:
                            fb1_str = part.strip().split(':')[-1].strip().replace('%', '')
                            try:
                                fb1 = float(fb1_str)
                            except ValueError:
                                print(f"Error parsing FB1 score: {fb1_str}")
                            break
    except FileNotFoundError:
        print(f"Evaluation result file {result_file_path} not found.")
    return fb1  # Returns 0.0 if FB1 score is not found

def main():
    """
    Main function to evaluate the best model on the test set and print the FB1 score.
    """
    # Evaluate the model and get the FB1 score
    fb1 = partb()

    # Print the FB1 score
    print(f"FB1 score on test set: {fb1}%")

if __name__ == "__main__":
    main()
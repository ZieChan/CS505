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
    
    # data
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    # train_data_path = os.path.join(data_dir, "train")
    test_data_path = os.path.join(data_dir, "test")

    # generated files
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    dev_out_path = os.path.join(generated_dir, "test.out")

    # train_word_corpus, train_tag_corpus = load_annotated_data(train_data_path)
    dev_word_corpus, dev_tag_corpus = load_annotated_data(test_data_path)

    # Do predictions
    print_cap: int = 5
    dev_predicted_corpus = list()
    num_examples = len(dev_word_corpus)

    for i, predicted_seq in enumerate(model.predict(dev_word_corpus)):
        dev_predicted_corpus.append(predicted_seq)
        if i < print_cap:
            for w, p, a in zip(dev_word_corpus[i], dev_tag_corpus[i], predicted_seq):
                print(w, p, a)
            print()

    # Set the path for the evaluation results file
    # eval_output_path = os.path.join(os.path.dirname(dev_out_path), 'evaluation_result.txt')

    # Call write_output_and_evaluate with the outfile parameter
    write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus, outfile='eval_output') #, outfile=eval_output_path

    # Parse the FB1 score from the evaluation results
    fb1 = parse_fb1_from_result_file('eval_output')

    # Return the FB1 score
    return fb1

def parse_fb1_from_result_file(result_file_path: str) -> float:
    fb1 = 0.0
    with open(result_file_path, 'r') as f:
        for line in f:
            if 'FB1:' in line:
                parts = line.strip().split(';')
                for part in parts:
                    if 'FB1:' in part:
                        fb1_str = part.strip().split(':')[-1].strip()
                        fb1 = float(fb1_str)
                        return fb1
    return fb1  # Return 0.0 if FB1 score is not found


def main():
    # data
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    # train_data_path = os.path.join(data_dir, "train")
    test_data_path = os.path.join(data_dir, "test")

    # generated files
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    test_out_path = os.path.join(generated_dir, "test.out")

    # train_word_corpus, train_tag_corpus = load_annotated_data(train_data_path)
    test_word_corpus, test_tag_corpus = load_annotated_data(test_data_path)

    model = SP()
    model.load("best_model.pkl")

    fb1 = partb()
    print(f"FB1: {fb1}")

if __name__ == "__main__":
    main()
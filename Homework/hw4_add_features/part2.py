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


def part_a(train_word_corpus: Sequence[Sequence[str]],
           train_tag_corpus: Sequence[Sequence[str]],
           generated_dir: str,
           dev_word_corpus: Sequence[Sequence[str]] = None,
           dev_tag_corpus: Sequence[Sequence[str]] = None
           ) -> SP:
    
    model = SP()
    # Uncomment the following line if you want to load a pre-trained model before training
    model.load("best_model.pkl")
    
    def log_function(model: SP,
                     epoch_num: int,
                     train_results: Tuple[int, int],
                     dev_results: Tuple[int, int]
                     ) -> None:
        """
        Custom logging function to log training and development metrics.
        """

        dev_out_path = os.path.join(generated_dir, "dev.out")

        # Measure eval predictions
        if dev_word_corpus is not None and dev_tag_corpus is not None:
            dev_predicted_corpus: Sequence[Sequence[str]] = list()
            for predicted_seq in model.predict(dev_word_corpus):
                dev_predicted_corpus.append(predicted_seq)

            perf_path = os.path.join(generated_dir, f"epoch-{epoch_num}.result")


            print(f"Writing evaluation results to {perf_path}")
            write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus,
                                      outfile=perf_path)

    # Train the model with early stopping and save the best model
    model._train(
        train_word_corpus=train_word_corpus,
        train_tag_corpus=train_tag_corpus,
        dev_word_corpus=dev_word_corpus,
        dev_tag_corpus=dev_tag_corpus,
        max_epochs=50,        # You can adjust the number of epochs as needed
        converge_error=1e-7,   # Convergence threshold
        log_function=log_function,
        model_save_path=os.path.join(generated_dir, "best_model.pkl")  # Path to save the best model
    ) 

    return model

def part_b(model: SP,
           dev_out_path: str,
           dev_word_corpus: Sequence[Sequence[str]],
           dev_tag_corpus: Sequence[Sequence[str]]
           ) -> float:
    """
    Make predictions on the development set, print a few examples, and return the FB1 score.
    """
    # Do predictions
    print_cap: int = 5
    dev_predicted_corpus = list()
    num_examples = len(dev_word_corpus)

    for i, predicted_seq in enumerate(model.predict(dev_word_corpus)):
        dev_predicted_corpus.append(predicted_seq)
        if i < print_cap:
            print(f"Sentence {i+1}:")
            for w, p, a in zip(dev_word_corpus[i], dev_tag_corpus[i], predicted_seq):
                print(f"{w}\t{p}\t{a}")
            print()

    # Set the path for the evaluation results file
    eval_output_path = os.path.join(os.path.dirname(dev_out_path), 'evaluation_result.txt')

    # Call write_output_and_evaluate with the outfile parameter
    # write_output_and_evaluate(
    #     dev_out_path=dev_out_path,
    #     word_corpus=dev_word_corpus,
    #     predicted_tag_corpus=dev_predicted_corpus,
    #     ground_truth_tag_corpus=dev_tag_corpus,
    #     outfile=eval_output_path
    # )
    write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus)

    # Parse the FB1 score from the evaluation results
    fb1 = parse_fb1_from_result_file(eval_output_path)

    # Return the FB1 score
    return fb1

def parse_fb1_from_result_file(result_file_path: str) -> float:
    """
    Extract the FB1 score from the evaluation results file.
    """
    fb1 = 0.0
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
    return fb1  # Returns 0.0 if FB1 score is not found

def main():
    # Data paths
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    train_data_path = os.path.join(data_dir, "train_dev")
    dev_data_path = os.path.join(data_dir, "test")

    # Generated files directory
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    dev_out_path = os.path.join(generated_dir, "test.out")

    # Load data
    train_word_corpus, train_tag_corpus = load_annotated_data(train_data_path)
    dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_data_path)

    # Train the model
    model = part_a(
        train_word_corpus=train_word_corpus,
        train_tag_corpus=train_tag_corpus,
        generated_dir=generated_dir,
        dev_word_corpus=dev_word_corpus,
        dev_tag_corpus=dev_tag_corpus
    )

    # Evaluate the model and get the FB1 score
    fb1_score = part_b(
        model=model,
        dev_out_path=dev_out_path,
        dev_word_corpus=dev_word_corpus,
        dev_tag_corpus=dev_tag_corpus
    )

    # Print the FB1 score
    print(f"Final FB1 score on dev set: {fb1_score}%")

def parse_fb1_from_result_file(result_file_path: str) -> float:
    """
    Extract the FB1 score from the evaluation results file.
    """
    fb1 = 0.0
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
    return fb1  # Returns 0.0 if FB1 score is not found

if __name__ == "__main__":
    main()
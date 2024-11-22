# SYSTEM IMPORTS
from collections.abc import Sequence
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
from hmm import HMM # hmm -> models.hmm


def part_a(word_corpus: Sequence[Sequence[str]],
           tag_corpus: Sequence[Sequence[str]]
           ) -> None:
    # compute sum of tokens, total word types (different words), total tag types
    token_sum = 0
    word_set = set()
    tag_set = set()
    for word_seq, tag_seq in zip(word_corpus, tag_corpus):
        token_sum += len(word_seq)
        word_set.update(word_seq)
        tag_set.update(tag_seq)

    # print our results
    print(f"num(tokens): {token_sum} num(unique words): {len(word_set)} num(unique tags): {len(tag_set)}")

def part_b(word_corpus: Sequence[Sequence[str]],
           tag_corpus: Sequence[Sequence[str]]
           ) -> HMM:

    model = HMM().train_from_raw(word_corpus, tag_corpus)
    # print(model.tag_alphabet)
    # model.visualize_tag_model()
    # model.visualize_tag_word_model()
    b_person_tag = "B-person"
    i_person_tag = "I-person"
    o_tag = "O"
    tag_list = [o_tag, i_person_tag, b_person_tag]

    person_1 = "God"
    person_2 = "Justin"
    person_3 = "Lindsay"
    people_list = [person_1, person_2, person_3]

    # print("tag alphabet: %s" % model.tag_alphabet)
    print("+----------------------------------------------+")
    print("| printing requested tag bigram probabilities: |")
    print("+----------------------------------------------+")
    for tag_2, tag_1 in zip([o_tag, b_person_tag, b_person_tag,
                             i_person_tag, i_person_tag, i_person_tag],
                            [b_person_tag, b_person_tag, i_person_tag,
                             b_person_tag, i_person_tag, o_tag]):
        print("p({0} | {1}): {2:.3f}".format(tag_1, tag_2,
            model.lm.get_value(tag_2, tag_1)))

    print("")
    print("+----------------------------------+")
    print("| printing tag word probabilities: |")
    print("+----------------------------------+")
    for person, tag in itertools.product(people_list, [b_person_tag, o_tag]):
        print("p({0} | {1}): {2:.5f}".format(person, tag,
            model.tm.get_value(tag, person)))
    return model


def predict_and_evaluate(model: HMM,
                         dev_out_path: str,
                         dev_word_corpus: Sequence[Sequence[str]],
                         dev_tag_corpus: Sequence[Sequence[str]],
                         print_cap: int = -1):
    # do predictions
    dev_predicted_corpus: Sequence[Sequence[str]] = list()
    num_examples = len(dev_word_corpus)

    for i, predicted_seq in enumerate(model.predict(dev_word_corpus)):
        dev_predicted_corpus.append(predicted_seq)
        if i < print_cap:
            for w, a, p in zip(dev_word_corpus[i], dev_tag_corpus[i], predicted_seq):
                print(w, a, p)
            print()

    # convert format to be evaluateable
    write_output_and_evaluate(dev_out_path, dev_word_corpus, dev_predicted_corpus, dev_tag_corpus)


def part_c(model: HMM,
           dev_out_path: str,
           dev_word_corpus: Sequence[Sequence[str]],
           dev_tag_corpus: Sequence[Sequence[str]]
           ) -> None:
    predict_and_evaluate(model, dev_out_path, dev_word_corpus, dev_tag_corpus, print_cap=5)


def main():
    # data
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    train_data_path = os.path.join(data_dir, "train")
    dev_data_path = os.path.join(data_dir, "dev")

    # generated files
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)
    dev_out_path = os.path.join(generated_dir, "dev.out")

    train_word_corpus, train_tag_corpus = load_annotated_data(train_data_path)
    dev_word_corpus, dev_tag_corpus = load_annotated_data(dev_data_path)

    print(len(dev_word_corpus), len(dev_tag_corpus))

    part_a(train_word_corpus, train_tag_corpus)
    model = part_b(train_word_corpus, train_tag_corpus)
    part_c(model, dev_out_path, dev_word_corpus, dev_tag_corpus)

if __name__ == "__main__":
    main()


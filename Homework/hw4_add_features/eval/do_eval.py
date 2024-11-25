# SYSTEM IMPORTS
from collections.abc import Sequence
import os
import subprocess


# PYTHON PROJECT IMPORTS


def check_list_lengths(*lists) -> bool:
    # check that the list lengths are the same
    all_same = True
    init_len = len(lists[0])
    if len(lists) > 1:
        for l in lists[1:]:
            all_same = all_same and init_len == len(l)
    return all_same


def write_output_and_evaluate(out_path: str,
                              word_corpus: Sequence[Sequence[str]],
                              predicted_tag_corpus: Sequence[Sequence[str]],
                              ground_truth_tag_corpus: Sequence[Sequence[str]],
                              outfile: str = ""
                              ) -> None:
    if not check_list_lengths(word_corpus, predicted_tag_corpus, ground_truth_tag_corpus):
        raise Exception("lists are not the same lengths! %s" %
            [len(word_corpus), len(predicted_tag_corpus), len(ground_truth_tag_corpus)])

    # write the data
    with open(out_path, "w") as f:
        for word_seq, gt_seq, pred_seq in zip(word_corpus, ground_truth_tag_corpus, predicted_tag_corpus):
            for w, g, p in zip(word_seq, gt_seq, pred_seq):
                f.write("%s %s %s\n" % (w, g, p))
            f.write("\n")

    # run evaluation script
    cd = os.path.abspath(os.path.dirname(__file__))
    script = os.path.join(cd, "eval_glue_script")
    if len(outfile) > 0:
        subprocess.call([script, out_path, outfile])
    else:
        subprocess.call([script, out_path])


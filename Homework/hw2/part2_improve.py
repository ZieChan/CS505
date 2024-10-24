# 需要的导入
import os
import sys
from collections.abc import Sequence
import pandas as pd
import matplotlib.pyplot as plt

# 您的其他导入保持不变
from fst import Transition
from from_file import load_aligned_data, load_unaligned_data
from utils import *
from models.ibm1 import IBM1
from models.translator import Translator
from eval.bleu import bleu

# 修改后的 parta 函数
def parta(mandarin_corpus: Sequence[Sequence[str]],
          english_corpus: Sequence[Sequence[str]],
          test_corpus: Sequence[Sequence[str]],
          lm_n: int,
          tm_max_iter: int,
          num_rules_to_keep: int,
          error: float = 1e-10
          ) -> Translator:

    # 计算 test_corpus 中的唯一词汇
    unique_test_token_set = set()
    for seq in test_corpus:
        unique_test_token_set.update(seq)

    return Translator().train_from_raw(mandarin_corpus,
                                       english_corpus,
                                       unknown_fs=unique_test_token_set,
                                       converge_threshold=error,
                                       lm_n=lm_n,
                                       tm_max_iter=tm_max_iter,
                                       num_rules_to_keep=num_rules_to_keep)

# 修改后的 partb 函数
def partb(m: Translator,
          test_corpus: Sequence[Sequence[str]],
          out_path: str,
          gold_path: str,
          print_limit: int = 10
          ) -> float:

    with open(out_path, "w") as f:
        for i, (decoded_seq, _) in enumerate(m.decode(test_corpus)):
            if i < print_limit:
                print(decoded_seq)

            f.write(decoded_seq + "\n")

    # 计算并返回 BLEU 分数
    return bleu(out_path, gold_path)

# 主函数
if __name__ == "__main__":
    cd = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(cd, "data")
    generated_dir = os.path.join(cd, "generated")
    if not os.path.exists(generated_dir):
        os.makedirs(generated_dir)

    train_path = os.path.join(data_dir, "train.zh-en")
    test_path = os.path.join(data_dir, "test.zh")
    gold_path = os.path.join(data_dir, "test.en")

    mandarin_corpus, english_corpus = load_aligned_data(train_path)
    test_corpus = load_unaligned_data(test_path)

    # 参数范围
    lm_n_values = [3]  # 仅取值 3
    tm_max_iter_values = [100]  # 仅取值 100
    num_rules_to_keep_values = range(10, 71, 10)  # 10 到 70，步长为 10

    # 存储结果的列表
    results = []

    for lm_n in lm_n_values:
        for tm_max_iter in tm_max_iter_values:
            for num_rules_to_keep in num_rules_to_keep_values:
                # 为每个参数组合生成输出文件路径
                translation_out = os.path.join(
                    generated_dir,
                    f"translations_lm{lm_n}_iter{tm_max_iter}_rules{num_rules_to_keep}.out"
                )

                print(f"正在运行实验：lm_n={lm_n}, tm_max_iter={tm_max_iter}, num_rules_to_keep={num_rules_to_keep}")

                # 训练模型
                m = parta(mandarin_corpus, english_corpus, test_corpus,
                          lm_n=lm_n, tm_max_iter=tm_max_iter,
                          num_rules_to_keep=num_rules_to_keep)

                # 评估模型并获取 BLEU 分数
                BLEU_score = partb(m, test_corpus, translation_out, gold_path)

                print(f"BLEU 分数：{BLEU_score}\n")

                # 将结果添加到列表中
                results.append({
                    'num_rules_to_keep': num_rules_to_keep,
                    'BLEU': BLEU_score
                })

    # 将结果转换为 DataFrame
    results_df = pd.DataFrame(results)
    print(results_df)

    # 保存结果到 CSV 文件
    results_df.to_csv(os.path.join(generated_dir, 'experiment_results.csv'), index=False)

    # 绘制图形
    plt.figure()
    plt.plot(results_df['num_rules_to_keep'], results_df['BLEU'], marker='o')
    plt.title('BLEU Score vs. num_rules_to_keep')
    plt.xlabel('num_rules_to_keep')
    plt.ylabel('BLEU Score')
    plt.grid(True)
    plt.savefig(os.path.join(generated_dir, 'bleu_num_rules_to_keep.png'))
    plt.show()

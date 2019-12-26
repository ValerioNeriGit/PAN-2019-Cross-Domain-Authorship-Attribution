import json
import math
import os
import subprocess

from nltk import WordPunctTokenizer, SnowballStemmer
from sklearn.metrics import classification_report, f1_score
from os import system
import ref.function_words as fw_file
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
import tabulate

import std


def open_truth_to_dict(path):
    with open(path, 'r') as f:
        truth = json.load(f)['ground_truth']

    res = dict()
    for t in truth:
        res[t['unknown-text']] = t['true-author']
    return res


def eval(ans_path, res_path):
    with open(ans_path, 'r') as f:
        answers = json.load(f)
    with open(res_path, 'r') as f:
        res = json.load(f)['ground_truth']

    ans_dict = dict()
    for ans in answers:
        ans_dict[ans["unknown-text"]] = ans["predicted-author"]

    y_true = []
    y_pred = []

    for r in res:
        y_true += [r['true-author']]
        y_pred += [ans_dict[r['unknown-text']]]

    print(f1_score(y_true=y_true, y_pred=y_pred, average='macro'))

    names = ['candidate' + str(num).zfill(4) for num in range(10)]
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=names))

    # print(ans_dict[res[0]['unknown-text']])
    # print(res[0]['true-author'])


def main():
    base_ans = "/Users/valerioneri/Sapienza/Projects/Authorship/Code/baseline/outs/"
    base_ans = "/Users/valerioneri/Sapienza/Projects/Authorship/Code/baseline/my_outs/"
    file_ans = "answers-problem{0}.json"

    res_file = "/Users/valerioneri/Sapienza/Projects/Authorship/Datasets/training-dataset-2019-01-23/problem{" \
               "0}/ground-truth.json"
    for i in range(1, 20):
        n = str(i).zfill(5)
        eval(base_ans + file_ans.format(n), res_file.format(n))


def extract_languages_avg(s):
    lines = s.splitlines()
    tail = 0
    head = 5
    for i in range(4):
        sum = 0
        for i in range(tail, head):
            line = lines[i]
            sum += float(line[30:-1])
        tail += 5
        head += 5
        print("language:", sum / 5)


def valuta(base, mine=True):
    baseline = base + 'Seconda_prova/outs/bl_svm/'
    # baseline = base + 'Seconda_prova/outs/bl_comp/'
    file = fmine = base + 'Code/baseline/my_outs'

    if not mine:
        file = baseline

    cmd = 'python {0} -i {1} -a {2} -o {3}'.format(base + "Code/evaluator.py",
                                                   base + 'Datasets/training-dataset-2019-01-23',
                                                   file,
                                                   base + 'Code/baseline/outs_eval')

    print(cmd)

    os.system(cmd)
    result = subprocess.check_output(cmd, shell=True)
    print(result)
    result = result.decode("utf-8")
    extract_languages_avg(result)


def evaluate_proba():
    classifier = "1"
    for problem in range(1, 9):
        truth_data = open_truth_to_dict(base + 'Datasets/training-dataset-2019-01-23' + '/problem000{0}/ground-truth.json'.format(str(problem + 1).zfill(2)))
        cross_predict = std.load_dict_into_pickle("cross_val" + classifier, base, "problem" + str(problem).zfill(5))['cross_val']

        path = "problem000{0}/fandom-info.json"

        # info = json.load(open(base + path.format(str(problem).zfill(2))))

        print(cross_predict.shape)

        print("Problem:", problem)
        table = []
        for i, p in enumerate(cross_predict):
            val = "unknown{0}.txt".format(str(i + 1).zfill(5))
            t = truth_data[val]
            # print(type(p))
            # print(p.shape)
            r = [math.ceil((i + 1) / 9)]
            max_ind = np.argmax(p) + 1
            r.append(max_ind)
            r.append(max_ind == t)
            r.extend(p)
            table.append(r)

        print(tabulate.tabulate(table))
        print(len(table))


def evaluate_unknown():
    d = std.load_dict_into_pickle("stat_", base, "")  # fw_file.stats_data2
    stats_datas = d['stat']  # fw_file.stats_data2

    for i, stats_data in enumerate(stats_datas[:1]):
        print(base + '/problem000{0}/ground-truth.json'.format(str(i + 1).zfill(2)))

        truth_data = open_truth_to_dict(base + 'Datasets/training-dataset-2019-01-23' + '/problem000{0}/ground-truth.json'.format(str(i + 1).zfill(2)))

        unks = []
        ks = []
        correct = []
        uncorrect = []

        for r in stats_data:
            max_proba, max_proba2 = sorted(r['proba'], reverse=True)[:2]
            val = max_proba  # max_proba - max_proba2
            if truth_data[r['unknown-text']] == "<UNK>" and r['predicted-author'] != "<UNK>":
                unks.append(val)
            elif truth_data[r['unknown-text']] != "<UNK>" and r['predicted-author'] == "<UNK>":
                ks.append(val)
            elif truth_data[r['unknown-text']] == "<UNK>" and r['predicted-author'] == "<UNK>":
                #  elif truth_data[r['unknown-text']] == r['predicted-author']:
                correct.append(val)
            # else:
            #     uncorrect.append(val)

        legend = ['unks', 'ks', 'correct']

        li = []
        for i, l in enumerate([unks, ks, correct]):
            li.append([max(l), np.var(l), np.std(l), min(l), mean(l), len(l), l])
            plt.plot(l, label=legend[i])
        print(tabulate.tabulate(li, headers=["max", "var", "std", "min", "mean", "len", "list"]))

        plt.legend()
        plt.plot(ks)
        plt.ylabel('some numbers')
        plt.show()

        unk_pred = []
        kn_pred = []

        for r in stats_data:
            max_proba, max_proba2, max_proba3, max_proba4 = sorted(r['proba'], reverse=True)[:4]
            val = max_proba  # max_proba - max_proba2
            if truth_data[r['unknown-text']] != "<UNK>" and r['predicted-author'] == "<UNK>":
                can_num = int(truth_data[r['unknown-text']][-4:])
                can_1 = r['proba'].index(max_proba) + 1
                can_2 = r['proba'].index(max_proba2) + 1
                unk_pred.append(
                    [(max_proba - max_proba2) / 2, max_proba2, max_proba - max_proba2, ((max_proba - max_proba2) + (max_proba2 - max_proba3)) / 2,
                     (max_proba2 - max_proba4) / 2, r['predicted-author'], truth_data[r['unknown-text']], can_1, can_1 == can_num, can_2, can_2 == can_num,
                     [x for x in sorted(r['proba'], reverse=True) if x > 0.1]])
            elif truth_data[r['unknown-text']] == "<UNK>" and r['predicted-author'] != "<UNK>":
                kn_pred.append(
                    [(max_proba - max_proba2) / 2, max_proba2, max_proba - max_proba2, ((max_proba - max_proba2) + (max_proba2 - max_proba3)) / 2,
                     (max_proba2 - max_proba4) / 2, r['predicted-author'],
                     truth_data[r['unknown-text']], [x for x in sorted(r['proba'], reverse=True) if x > 0.1]])

        #  for p in unk_pred: print(p)
        print("tabulate")
        print(tabulate.tabulate(unk_pred, headers=['max', 'max2', 'max_diff', 'mean_diff', 'mean_2-4diff', 'Predicted', 'True', 'max_c', '', 'max2_c', '',
                                                   'probabilities']))
        print(tabulate.tabulate(kn_pred, headers=['max', 'max2', 'max_diff', 'mean_diff', 'mean_2-4diff', 'Predicted', 'True', 'probabilities']))


if __name__ == '__main__':
    base = "/Users/valerioneri/Sapienza/Projects/Authorship/"
    base = "/home/valerioneri/PycharmProjects/cross-domain-authorship-attribution/"
    base = "/opt/projects/attribution/src/"
    valuta(base, mine=True)
    train_texts = [
        "Voglio che tu rimanga mio, come lo sei adesso. Non voglio che questo cambi.",
        "Non voglio perderti, Cas, ecco la verità."
    ]
    #  stemmer = SnowballStemmer('italian')

    # train_texts = [" ".join([stemmer.stem(token) for token in WordPunctTokenizer().tokenize(text)]) for text in train_texts]
    #  print(train_texts)

    # for i in range(20):
    #    print("fgrep '<UNK>' problem000{0}/ground-truth.json | wc -l;".format(str(i).zfill(2)))

    # evaluate_proba()
    # print(std.generate_skip_word_text(
    #     "Lestrade called the other day, asking me to help out on a case. You would've been proud – I yelled at him for not doing his job and being an idiot. I haven't talked to him since.",
    #     2, "english"))

    a = \
        [
            # clf1
            [
                [1, 2, 3, 4],
                [4, 3, 2, 1]
            ],

            # clf2
            [
                [5, 6, 7, 8],
                [9, 8, 7, 6]
            ]
        ]

    print(np.average(a, axis=0))

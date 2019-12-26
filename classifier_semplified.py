#!/usr/bin/python3
import argparse, os

# from configuration_vars import PAN, MAX_PROBLEMS, MIN_PROBLEMS, N_CORE, MULTICORE, MULTICLASSIFIER
# from std import scale, classifier, porbabilities_clf

exit_code = os.system("bash compile.sh")
if exit_code != 0: exit(exit_code)

from time import time

#  from sklearn.feature_selection import SelectKBest, chi2
from multiprocessing import Pool
from scipy.sparse import hstack
import numpy as np
import warnings
import std
from eval import valuta
from eval import open_truth_to_dict
from configuration_vars import *


def warn(*args, **kwargs):
    pass


warnings.warn = warn


def vectorization(train_texts, test_texts, path, problem, language, candidate_grouped_texts):
    # vocabulary = std.get_ngram(train_docs, n, ft)
    # vocabulary = std.do_vocabulary(base, problem, train_docs, n, ft, use_stored=True)

    t0 = time()
    cosine = False

    if cosine:
        # candidate_grouped_data_cs, train_data_group_cs, test_data_group_cs = std.char_single_gram(candidate_grouped_texts, train_texts, test_texts, language, True)
        candidate_grouped_data_ws, train_data_group_ws, test_data_group_ws = std.word_single_gram(candidate_grouped_texts, train_texts, test_texts, language,
                                                                                                  True)

        # cosine_matrix_train_c, cosine_matrix_test_c = std.cosine_similarity_matrix(candidate_grouped_data_cs, train_data_group_cs, test_data_group_cs)
        cosine_matrix_train_w, cosine_matrix_test_w = std.cosine_similarity_matrix(candidate_grouped_data_ws, train_data_group_ws, test_data_group_ws)

        # print("Cosineshape", cosine_matrix_train_c.shape, cosine_matrix_train_w.shape, cosine_matrix_test_c.shape, cosine_matrix_test_w.shape)
        # print("Cosineshape", cosine_matrix_train_c.ndim, cosine_matrix_train_w.ndim, cosine_matrix_test_c.ndim, cosine_matrix_test_w.ndim)

    # train_data_compr, test_data_compr = std.compression(base, problem, candidate_grouped_texts, train_texts, test_texts, language, alg="gzip",
    # return [train_data_compr], [test_data_compr], None

    candidate_grouped_data_c, train_data_group_c, test_data_group_c = std.char_single_gram(candidate_grouped_texts, train_texts, test_texts, language)
    candidate_grouped_data_w, train_data_group_w, test_data_group_w = std.word_single_gram(candidate_grouped_texts, train_texts, test_texts, language)

    # chargram distorto
    train_data_d, test_data_d = std.char_gram_dist(base, problem, train_texts, test_texts, language, use_stored=True and S, Store=True and S)

    # chargram e wordgram configurabili con grange=(n,m) come parametro
    train_data_c, test_data_c = std.char_gram(base, problem, train_texts, test_texts, language, use_stored=True and S, Store=True and S)
    train_data_w, test_data_w = std.word_gram(base, problem, train_texts, test_texts, language, use_stored=True and S, Store=True and S)

    #
    # List of features to return
    train_data_multi = [train_data_c, train_data_w, train_data_d, hstack([train_data_group_c, train_data_group_w])]
    test_data_multi = [test_data_c, test_data_w, test_data_d, hstack([test_data_group_c, test_data_group_w])]

    # Aggiunta wordgram (1,1) (2,2)
    if language in ['english', 'italian']:
        train_data_w1, test_data_w1 = std.word_gram(base, problem, train_texts, test_texts, language, grange=(1, 1), base_name="word_11", use_stored=True and S,
                                                    Store=True and S)
        train_data_w2, test_data_w2 = std.word_gram(base, problem, train_texts, test_texts, language, grange=(2, 2), base_name="word_22", use_stored=True and S,
                                                    Store=True and S)

        train_data_multi.append(hstack([train_data_w1, train_data_w2]))
        test_data_multi.append(hstack([test_data_w1, test_data_w2]))

    # Aggiunta pos tagger
    if language in ['english', 'spanish']:
        vect = std.char_gram
        f = std.create_post_tag_string
        if language in ['french']:
            vect = std.word_gram
            # f = std.stanford_post_tag_string
        train_data_w_pos, test_data_w_pos = vect(base, problem, train_texts, test_texts, language, f=f, base_name="pos_string_" + language,
                                                 gram_range=(4, 5), use_stored=True and S, Store=True and S)
        train_data_multi.append(train_data_w_pos)
        test_data_multi.append(test_data_w_pos)

        print("pos_shape", train_data_w_pos.shape, test_data_w_pos.shape)

    # if language in ['english']:
    #     train_data_w_phon, test_data_w_phon = std.word_gram(base, problem, train_texts, test_texts, language, f=std.str_to_phonetics, base_name="phonetics",
    #                                                         use_stored=True and S, Store=True and S)
    #     train_data_multi.append(train_data_w_phon)
    #     test_data_multi.append(test_data_w_phon)

    if MULTICLASSIFIER: return train_data_multi, test_data_multi, None


def multi_classification(train_data, test_data, clfs, path, problem, language, train_labels, gram, ft, pt):
    t0 = time()

    # Applying SVM
    predictions_list = list()
    proba_list = list()
    cross_predicts = []

    # clfs = [classifier(), classifier()]
    train_datas_to_clfs = train_data
    test_datas_to_clfs = test_data

    print(len(clfs), len(train_datas_to_clfs), len(test_datas_to_clfs))

    i = 1
    for clf, train_datas_to_clf, test_datas_to_clf in zip(clfs, train_datas_to_clfs, test_datas_to_clfs):
        print("I am an happy classifier", train_datas_to_clf.shape, len(train_labels))
        clf.fit(train_datas_to_clf, train_labels)

        try:
            proba_list.append(clf._predict_proba_lr(test_datas_to_clf))
            # cv = cross_val_predict(clf, train_datas_to_clf, train_labels, cv=7, method="_predict_proba_lr", verbose=3, n_jobs=4)
        except Exception:
            proba_list.append(clf.predict_proba(test_datas_to_clf))
            # cv = cross_val_predict(clf, train_datas_to_clf, train_labels, cv=7, method="predict_proba", verbose=3, n_jobs=4)

        # cross_predicts.append(cv)
        predictions_list.append(clf.predict(test_datas_to_clf))

        try:
            print("clf_best_estimator:", problem, clf.base_estimator)
        except Exception:
            pass
        print("classifier_config:", clf)

    # cross_predicts_stack = np.hstack(cross_predicts)

    print([p.shape for p in proba_list])
    # proba_list_stack = np.hstack(proba_list)
    print('Classification time:', time() - t0)

    return std.soft_voting(proba_list, weights[language])

    #
    # Neural net or others classifiers
    return porbabilities_clf(cross_predicts_stack, proba_list_stack, train_labels)
    # return predictions_list, proba_list


def evaluate_problem(args):
    index, problem, language, path, outpath, pickle_path, n, ft, pt = args

    print(problem, language)
    t0 = time()

    #
    # retrieve problem information
    candidates, unk_folder = std.get_problem_info(path, problem)

    #
    # retrieve train end test documents
    train_docs, train_texts, train_labels, test_texts, candidate_grouped_texts = std.get_train_andtest_set(candidates, path, problem, unk_folder, pickle_path,
                                                                                                           use_storage=False and S)
    print("train and test time():", time() - t0)
    # std.print_problem_data(language, index, candidates, train_texts, test_texts)

    #
    # retrieve list of features, list of matrix.
    # With ensemble the i-th matrix (features) with the i-th classifier in clfs below
    train_data, test_data, cosine_matrix_test_c = vectorization(train_texts, test_texts, path, problem, language, candidate_grouped_texts)

    # train_data, test_data = dimentionality_reduction(train_data, test_data)
    train_data, test_data = std.scale(train_data, test_data, print_time=True)

    #
    # fit and predict the i-th classifier in clfs with the i-th matrix (features) of train_data and test_data
    # then use the custom soft voting method to generate the final prediction
    # multi_classification internally prints classifier_config and clf_best_estimator information
    # Note that the weights for the soft voting are specified in configuration_vrar.py, None means all the same weights
    clfs = [std.classifier(), std.classifier(), std.classifier(), std.classifier(), std.classifier(), std.classifier(), std.classifier(), std.classifier()]
    predictions_list, proba_list = multi_classification(train_data, test_data, clfs, path, problem, language, train_labels, n, ft, pt)

    # unk_predictions_list, unk_proba_list = unk_classification(train_data, test_data, path, problem)

    #
    # 1. this part evaluates the unknown, with "reject_option" based on the baseline and "reject_option_cosine" executed iff "cosine_matrix_test_c is None"
    # 2. saves the outputs files (based on the baseline) with "save_output"
    # since currents methods return a list with just one element this for loop will execute just ine iteration
    for predictions, proba in zip(predictions_list, proba_list):

        if cosine_matrix_test_c is None:
            predictions = std.reject_option(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        else:
            predictions = std.reject_option_cosine(predictions, proba, pt, problem, language, cosine_matrix_test_c)
        stats_data = std.save_output(path, problem, unk_folder, predictions, outpath, proba)

        return stats_data  # , truth_data

    # valuta(base, mine=True)


def main(path, outpath, pickle_path, n=3, ft=5, pt=0.1):
    problems, language = std.get_problems_list(path)

    lan = {'en': 'english', 'fr': 'french', 'sp': 'spanish', 'it': 'italian', 'pl': 'polish'}

    # for each problem
    language = [lan[l] for l in language]
    args_list = []

    if MAX_PROBLEMS != 0: problems = problems[MIN_PROBLEMS:MAX_PROBLEMS]

    #  for k in [30000, 31000]:
    #  problems.reverse()
    for index, problem in enumerate(problems):
        args_list.append((index, problem, language[int(problem[-3:]) - 1], path, outpath, pickle_path, n, ft, pt))
        if not MULTICORE: evaluate_problem((index, problem, language[int(problem[-3:]) - 1], path, outpath, pickle_path, n, ft, pt))

    if MULTICORE:
        with Pool(N_CORE) as p:
            res = p.map(evaluate_problem, args_list)


base = "/opt/projects/attribution/src/"
pickle_path = base + "Code/baseline/pickles"

parser = argparse.ArgumentParser(description='cross domain authorship attribution')
parser.add_argument('-i', type=str, help='Path to evaluation collection')
parser.add_argument('-o', type=str, help='Path to output files')
args = parser.parse_args()

# remove old outputs
# std.clean_dir(args.o)

print("path", args.i + os.sep)

t0 = time()
main(args.i + os.sep, args.o + os.sep, pickle_path)
t1 = time() - t0

if not PAN:
    valuta(base, mine=True)

print("Total Time:", t1)

import codecs, glob, os, json, pickle
import gzip
import itertools
import math
from time import time
from typing import Dict

from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize, WordPunctTokenizer
# from nltk.tokenize import *
from nltk import pos_tag
#  from nltk import word_tokenize
from scipy.cluster.vq import whiten
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np
from scipy.sparse import hstack
from sklearn import preprocessing
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD, FastICA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC, OneClassSVM
from tabulate import tabulate

import std_c
from sklearn.metrics.pairwise import cosine_similarity
# from metaphone import doublemetaphone
from configuration_vars import *

from ref.function_words import british_american_word
import eng_to_ipa as ipa
# from nltk.tag import StanfordPOSTagger
# from nltk.internals import config_java
from nltk import download

download('averaged_perceptron_tagger')
download('punkt')

# config_java(options='-Xmx5g')
# config_java(options='-mx5g')
path_stanford_tagger = "/opt/projects/attribution/src/Code/resources/stanford-postagger-full-2018-10-16/" + "stanford-postagger.jar"
path_stanford_tagger_model = "/opt/projects/attribution/src/Code/resources/stanford-postagger-full-2018-10-16/models/"

pos_cache = {}
unk_num = {
    1: 93,
    2: 39,
    3: 79,
    4: 121,
    5: 132,
    6: 20,
    7: 26,
    8: 161,
    9: 106,
    10: 19,
    11: 23,
    12: 33,
    13: 73,
    14: 20,
    15: 27,
    16: 27,
    17: 32,
    18: 89,
    19: 200,
    20: 85,

}  # type: Dict[int, int]


def _get_collection_path(basepath):
    return basepath + os.sep + 'collection-info.json'


def _get_problem_info(basepath, problem):
    return basepath + os.sep + problem + os.sep + 'problem-info.json'


def _get_problem_path_of_json(basepath, problem, filename):
    return basepath + os.sep + problem + os.sep + filename + '.json'


def read_files(path, problem, label):
    # Reads all text files located in the 'path' and assigns them to 'label' class
    files = glob.glob(os.path.join(path, problem, label, '*.txt'))
    texts = []
    # for i, v in enumerate(files):
    for v in files:
        f = codecs.open(v, 'r', encoding='utf-8')
        texts.append((f.read(), label))
        #  texts.append(f.read())
        f.close()
    #  texts = [(texts, label)]
    return texts


def get_problems_list(basepath):
    problems = []
    language = []
    with open(_get_collection_path(basepath), 'r') as f:
        for attrib in json.load(f):
            problems.append(attrib['problem-name'])
            language.append(attrib['language'])

    return problems, language


def get_problem_info(path, problem):
    infoproblem = _get_problem_info(path, problem)
    candidates = []
    with open(infoproblem, 'r') as f:
        fj = json.load(f)
        unk_folder = fj['unknown-folder']
        for attrib in fj['candidate-authors']:
            candidates.append(attrib['author-name'])

    return candidates, unk_folder


def separate_text_labels(train_docs):
    train_texts = [text for i, (text, label) in enumerate(train_docs)]
    train_labels = [label for i, (text, label) in enumerate(train_docs)]
    return train_texts, train_labels


def print_problem_data(language, index, candidates, train_texts, test_texts, vocabulary=()):
    print('\t', 'language: ', language,
          '\n\t', len(candidates), 'candidate authors',
          '\n\t', len(train_texts), 'known texts',
          '\n\t', 'vocabulary size:', len(vocabulary),
          '\n\t', len(test_texts), 'unknown texts')


def get_training_set(candidates, path, problem):
    #
    # Building training set
    candidate_grouped_texts = []
    train_docs = []
    for candidate in candidates:
        candidate_texts = read_files(path, problem, candidate)
        candidate_grouped_texts.append(" ".join([t for t, l in candidate_texts]))
        train_docs.extend(candidate_texts)

        # print("Testi", candidate)
        # print(candidate_texts)

    train_texts, train_labels = separate_text_labels(train_docs)
    # print(type(np.array(candidate_grouped_texts)[0]))
    return np.array(train_docs), np.array(train_texts), np.array(train_labels), np.array(candidate_grouped_texts)


def get_test_set(path, problem, unk_folder):
    test_docs = np.array(read_files(path, problem, unk_folder))
    test_texts = np.array([text for i, (text, label) in enumerate(test_docs)])
    return test_docs, test_texts


def get_train_andtest_set(candidates, path, problem, unk_folder, pickle_path, use_storage=False):
    pickle_name = pickle_path + os.sep + problem + ".pickle"
    if use_storage and os.path.isfile(pickle_name):
        return pickle_load(pickle_name)
    else:
        # Building training set
        train_docs, train_texts, train_labels, candidate_grouped_texts = get_training_set(candidates, path, problem)
        # now I have train texts (features) and labels ready to use

        # Building test set (_ as test_docs)
        _, test_texts = get_test_set(path, problem, unk_folder)
        # now I have test_ texts (features) and labels ready to use

        if use_storage: picke_store(pickle_name, train_docs, train_texts, train_labels, test_texts)

        return train_docs, train_texts, train_labels, test_texts, candidate_grouped_texts


def get_ngram(texts, n, ft):
    if True:
        return _get_ngram_their(texts, n, ft)


def _get_ngram_their(texts, n, ft):
    from collections import defaultdict

    def represent_text(text, n):
        # Extracts all character 'n'-grams from  a 'text'
        if n > 0:
            tokens = [text[i:i + n] for i in range(len(text) - n + 1)]
        frequency = defaultdict(int)
        for token in tokens:
            frequency[token] += 1
        return frequency

    def extract_vocabulary(texts, n, ft):
        # Extracts all characer 'n'-grams occurring at least 'ft' times in a set of 'texts'
        occurrences = defaultdict(int)
        for (text, label) in texts:
            text_occurrences = represent_text(text, n)
            for ngram in text_occurrences:
                if ngram in occurrences:
                    occurrences[ngram] += text_occurrences[ngram]
                else:
                    occurrences[ngram] = text_occurrences[ngram]
        vocabulary = []
        for i in occurrences.keys():
            if occurrences[i] >= ft:
                vocabulary.append(i)
        return vocabulary

    return extract_vocabulary(texts, n, ft)


def save_output(path, problem, unk_folder, predictions, outpath, proba):
    # Saving output data
    out_data = []
    stats_data = []

    unk_filelist = glob.glob(path + os.sep + problem + os.sep + unk_folder + os.sep + '*.txt')
    pathlen = len(path + os.sep + problem + os.sep + unk_folder + os.sep)
    for i, v in enumerate(predictions):
        out_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v})
        stats_data.append({'unknown-text': unk_filelist[i][pathlen:], 'predicted-author': v, 'proba': list(proba[i])})
    with open(outpath + os.sep + 'answers-' + problem + '.json', 'w') as f:
        json.dump(out_data, f, indent=4)
    print('\t', 'answers saved to file', outpath, 'answers-' + problem + '.json')
    return stats_data


def do_vocabulary(base, problem, train_docs, n, ft, use_stored=False, Store=True):
    filename = base + os.sep + "Code/baseline/pickles" + os.sep + problem + '_vocabulary.pickle'

    if use_stored and os.path.isfile(filename):
        vocabulary = pickle.load(open(filename, "rb"))['v']
    else:
        print("Generate vocabulary")
        vocabulary = get_ngram(train_docs, n, ft)
        if Store: pickle.dump({'v': vocabulary}, open(filename, "wb"))

    return vocabulary


#
# routine helper
#


def clean_dir(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


#
# Dimensionality reductions
#

def PCA(features_train, features_test, n_components):
    pca = PCA(whiten=True, n_components=n_components).fit(features_train)
    print("Ratio", pca.explained_variance_ratio_)
    return pca.transform(features_train), pca.transform(features_test)


def do_TruncatedSVD(features_train, features_test, n_components):
    svd = TruncatedSVD(n_components=n_components).fit(features_train)
    print("Ratio", svd.explained_variance_ratio_)
    return svd.transform(features_train), svd.transform(features_test)


#
# Pickle helper
#


def pickle_load(pickle_name):
    d = pickle.load(open(pickle_name, "rb"))

    return d['train_docs'], d['train_texts'], d['train_labels'], d['test_texts']


def picke_store(pickle_name, train_docs, train_texts, train_labels, test_texts):
    d = {
        'train_docs': train_docs,
        'train_texts': train_texts,
        'train_labels': train_labels,
        'test_texts': test_texts
    }

    pickle.dump(d, open(pickle_name, "wb"))


def save_dict_into_pickle(d, filename, base, problem):
    filename = base + os.sep + "Code/baseline/pickles/word_vectorizer_" + problem + filename + '.pickle'
    pickle.dump(d, open(filename, "wb"))


def load_dict_into_pickle(filename, base, problem):
    filename = base + os.sep + "Code/baseline/pickles/word_vectorizer_" + problem + filename + '.pickle'
    return pickle.load(open(filename, "rb"))


#
# Text Distortion
#

def list_distortion(texts):
    # print("Distorting texts")
    res = [distortion(text) for text in texts]
    # print("distorted", res[:1][:10])
    return res


def distortion(string):
    # use not is_char per avere *Danger* b laisser m* It*s dry*
    # string = 'la miò stringà ?? con. ZZ ` [] \_ed aa 9:A@.'

    def is_ascii(char):
        return ord(char) < 128

    def is_char(char):
        # Remove all char that is not [a-z,A-Z,0-9]
        # Check if is a char in [a-z,A-Z,0-9]
        return 48 < ord(char) < 123 and not 90 < ord(char) < 97 and not 57 < ord(char) < 65

    def char_diacrittic(char):
        if char != ' ' and char != "'" and char != '"':
            if not is_ascii(char): return '*'
            if not is_char(char): return '#'
        return char

    # https://ascii.cl/
    string = string.replace('\n', " ").replace("  ", " ")
    s = ''.join(['*' if char != ' ' and is_char(char) else char for char in string])
    # # remove not char
    # s = ''.join(['' if char != ' ' and not is_char(char) else char for char in string])
    # s = ''.join([char_diacrittic(char) for char in string])
    # [unicode(x.strip()) if x is not None else '' for x in row]
    # print(s[:10])
    return s


def generate_skip_word_list(texts, span, language, stem=True):
    res = []

    for text in texts:
        res.append(generate_skip_word_text(text, span, language, stem))

    return res


def generate_skip_word_text(text, span, language, stem):
    res = []

    word_tokenizer = RegexpTokenizer(r'\w+')
    words = word_tokenizer.tokenize(text)

    stemmer = SnowballStemmer(language)

    for i in range(len(words) - span):
        if stem:
            res.append(stemmer.stem(words[i]) + '_' + stemmer.stem(words[i + span]))
        else:
            res.append(words[i] + '_' + words[i + span])

    return ' '.join(res)


#
# n-gram vectorization
#

def char_single_gram(candidate_grouped_texts, train_texts, test_texts, language, gram_range=(3, 5), truncate=False):
    print("Generate char Vectorizer")

    # gram_range = (3, 5)
    # if language == 'english':
    #     gram_range = (2, 6)

    # gram_range = (1, 3)

    tokenizer = RegexpTokenizer(r'\w+').tokenize
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=gram_range,  # max_features=20000,
                                 lowercase=False, min_df=0.12, sublinear_tf=True)

    #  print("text analysis", train_texts.shape, len(train_texts), len(train_texts[0]), type(train_texts[0]))

    if truncate:
        maxlen_g = min([len(x) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])
        maxlen_tr = min([len(y) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])
        maxlen_te = min([len(z) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])

        candidate_grouped_texts = [t[:maxlen_g] for t in candidate_grouped_texts]
        train_texts = [t[:maxlen_tr] for t in train_texts]
        test_texts = [t[:maxlen_te] for t in test_texts]

    train_data = vectorizer.fit_transform(candidate_grouped_texts)
    train_texts = vectorizer.transform(train_texts)
    test_data = vectorizer.transform(test_texts)

    print("Vectorizer_config:", vectorizer)
    return train_data, train_texts, test_data


def word_single_gram(candidate_grouped_texts, train_texts, test_texts, language, gram_range=(1, 3), truncate=False):
    print("Generate char Vectorizer")

    # if language == 'english':
    #      gram_range = (2, 6)

    # gram_range = (1, 3)

    stemmer = SnowballStemmer(language)
    lemmatizer = WordNetLemmatizer()

    g = stemmer.stem

    train_texts = [" ".join([g(token) for token in WordPunctTokenizer().tokenize(text)]) for text in train_texts]
    test_texts = [" ".join([g(token) for token in WordPunctTokenizer().tokenize(text)]) for text in test_texts]

    # print(train_texts[0])

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=gram_range, tokenizer=WordPunctTokenizer().tokenize, lowercase=False, min_df=0.03,
                                 sublinear_tf=True)

    if truncate:
        maxlen_g = min([len(x) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])
        maxlen_tr = min([len(y) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])
        maxlen_te = min([len(z) for x, y, z in zip(candidate_grouped_texts, train_texts, test_texts)])

        candidate_grouped_texts = [t[:maxlen_g] for t in candidate_grouped_texts]
        train_texts = [t[:maxlen_tr] for t in train_texts]
        test_texts = [t[:maxlen_te] for t in test_texts]

    train_data = vectorizer.fit_transform(candidate_grouped_texts)
    train_texts = vectorizer.transform(train_texts)

    test_data = vectorizer.transform(test_texts)

    print("Vectorizer_config:", vectorizer)
    return train_data, train_texts, test_data


def char_single_gram_dist(base, problem, candidate_grouped_texts, train_texts, test_texts, language, base_name="dist_", use_stored=False,
                          Store=True):
    filename = base + os.sep + "Code/baseline/pickles/char_vectorizer_" + base_name + problem + '.pickle'

    if use_stored and os.path.isfile(filename):
        d = pickle.load(open(filename, "rb"))
        vectorizer = d['v']
        train_data = d['train']
        test_data = d['test']
    else:
        print("Generate dist char Vectorizer")

        grouped_texts = list_distortion(candidate_grouped_texts)
        train_texts = list_distortion(train_texts)
        test_texts = list_distortion(test_texts)

        gram_range = (1, 8)
        if language == 'english':
            gram_range = (2, 6)

        # gram_range = (2, 6)

        tokenizer = RegexpTokenizer(r'\w+').tokenize
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=gram_range,  # max_features=10000,
                                     lowercase=False, min_df=0.12, sublinear_tf=True)

        grouped_data = vectorizer.fit(grouped_texts)
        train_data = vectorizer.transform(train_texts)
        test_data = vectorizer.transform(test_texts)

        if Store: pickle.dump({'v': vectorizer, 'train': train_data, 'test': test_data}, open(filename, "wb"))

    print("Vectorizer_config:", vectorizer)
    return train_data, test_data


def compress_text(text, name, index):
    zippath = "zips/ppmd/"
    filename = zippath + "ppdm_" + name + str(index)

    # return len(gzip.compress(bytes(text,"UTF-8")))

    if not os.path.isfile(filename + '.7z'):
        with open(filename + '.txt', 'w+') as out:
            out.write(text)

        os.system("7z a -m0=PPMd  {0}.7z  {0}.txt >/dev/null".format(filename))

        os.remove(filename + '.txt')

    # with open(filename + '.7z', 'rb') as compressed_file:
    #     print(compressed_file.read())
    #     return len(compressed_file.read())
    return os.path.getsize(filename + '.7z')


def compress_texts_matrix(texts, united_texts, name, alg=None):
    result = []

    for u, united_text in enumerate(united_texts):
        row = []
        for i, text in enumerate(texts):
            # if alg == "gzip":
            #     compre_alg = gzip.compress
            #     compressed = ppmd.main(bytes(united_text + text, 'UTF-8'))

            row.append(compress_text(united_text + text, name, str(i) + '-' + str(u)))
        result.append(row)

    return result


def compress_texts(texts, name, alg=None):
    result = []

    for i, text in enumerate(texts):
        # if alg == "gzip": compre_alg = gzip.compress
        # compressed = compre_alg(bytes(text, 'UTF-8'))
        result.append(compress_text(text, name, str(i)))

    return result


def CBC(x, y, xy):
    xy = float(xy)
    x = float(x)
    y = float(y)

    return 1 - (x + y - xy) / math.sqrt(x * y)


def NCD(x, y, xy):
    xy = float(xy)
    x = float(x)
    y = float(y)
    return (xy - min(x, y)) / max(x, y)


def calculate_compression(train_data, united_data, concat_united):
    result = np.zeros((len(train_data), len(united_data)), np.float64)

    for t in range(len(train_data)):
        for u in range(len(united_data)):
            # print(united_data[u], train_data[t], concat_united[u][t])
            result[t][u] = CBC(united_data[u], train_data[t], concat_united[u][t])
            # result[t][u] = united_data[u] - train_data[t]

    result2 = np.zeros((len(train_data), len(united_data)), np.float64)

    for t in range(len(train_data)):
        for u in range(len(united_data)):
            # print(united_data[u], train_data[t], concat_united[u][t])
            result2[t][u] = CBC(united_data[u], train_data[t], concat_united[u][t])
            # result[t][u] = united_data[u] - train_data[t]

    return result  # np.hstack([result, result2])


def compression(base, problem, united_texts, train_texts, test_texts, language, alg="", f=None, use_stored=False, Store=False):

    print("run_compressing", problem)

    united_data     = compress_texts(united_texts, problem + "united_data", alg)
    train_data      = compress_texts(train_texts, problem + "train_data", alg)
    test_data       = compress_texts(test_texts, problem + "test_data", alg)

    train_united    = compress_texts_matrix(train_texts, united_texts, problem + "train_united", alg)
    test_united     = compress_texts_matrix(test_texts, united_texts, problem + "test_united", alg)

    # print(united_data)
    # print(train_data)
    # print(test_data)
    # print(train_united)
    # print(test_united)
    # print(tabulate(train_data))

    train_data = calculate_compression(train_data, united_data, train_united)
    test_data = calculate_compression(test_data, united_data, test_united)

    print(tabulate(test_data))

    print("end_compression", problem, time())
    return train_data, test_data


def char_gram(base, problem, train_texts, test_texts, language, base_name="", gram_range=(3, 5), f=None, use_stored=False, Store=True):
    filename = base + os.sep + "Code/baseline/pickles/char_vectorizer_" + base_name + problem + '.pickle'

    if use_stored and os.path.isfile(filename):
        d = pickle.load(open(filename, "rb"))
        vectorizer = d['v']
        train_data = d['train']
        test_data = d['test']
    else:

        if f is not None:
            train_texts = f(train_texts, language, base + os.sep + "Code/baseline/pickles/train_" + problem)
            test_texts = f(test_texts, language, base + os.sep + "Code/baseline/pickles/test_" + problem)

        print("text analysis", len(train_texts), len(train_texts[0]), type(train_texts[0]))

        print("Generate char Vectorizer")
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=gram_range, lowercase=False, min_df=0.12, sublinear_tf=True)
        train_data = vectorizer.fit_transform(train_texts)
        test_data = vectorizer.transform(test_texts)

        if Store: pickle.dump({'v': vectorizer, 'train': train_data, 'test': test_data}, open(filename, "wb"))

    print("Vectorizer_config:", vectorizer)
    return train_data, test_data


def char_gram_dist(base, problem, train_texts, test_texts, language, base_name="dist_", use_stored=False,
                   Store=True):
    filename = base + os.sep + "Code/baseline/pickles/char_vectorizer_" + base_name + problem + '.pickle'

    if use_stored and os.path.isfile(filename):
        d = pickle.load(open(filename, "rb"))
        vectorizer = d['v']
        train_data = d['train']
        test_data = d['test']
    else:
        print("Generate dist char Vectorizer")

        train_texts = list_distortion(train_texts)
        test_texts = list_distortion(test_texts)

        gram_range = (1, 8)
        if language == 'english':
            gram_range = (2, 6)

        # gram_range = (2, 6)

        tokenizer = RegexpTokenizer(r'\w+').tokenize
        vectorizer = TfidfVectorizer(analyzer='char', ngram_range=gram_range,  # max_features=10000,
                                     lowercase=False, min_df=0.12, sublinear_tf=True)

        train_data = vectorizer.fit_transform(train_texts)
        test_data = vectorizer.transform(test_texts)

        if Store: pickle.dump({'v': vectorizer, 'train': train_data, 'test': test_data}, open(filename, "wb"))

    print("Vectorizer_config:", vectorizer)
    return train_data, test_data


def word_gram(base, problem, train_texts, test_texts, language, grange=(1, 3), f=None, base_name="word_", use_stored=False,
              Store=True, span=2, stem=True, phone=False, notokenizer=True):
    filename = base + os.sep + "Code/baseline/pickles/word_vectorizer_" + base_name + problem + '.pickle'

    if use_stored and os.path.isfile(filename):
        d = pickle.load(open(filename, "rb"))
        vectorizer = d['v']
        train_data = d['train']
        test_data = d['test']
    else:
        print("Generate Word Vectorizer")
        print(language)

        tokenizer = RegexpTokenizer(r'\w+').tokenize
        tokenizer = WordPunctTokenizer().tokenize
        # tokenizer.tokenize(text)
        # stop_words=stopwords.words(language),

        if f is not None:
            train_texts = f(train_texts, language, base + os.sep + "Code/baseline/pickles/train_" + problem)
            test_texts = f(test_texts, language, base + os.sep + "Code/baseline/pickles/test_" + problem)

        elif stem:
            stemmer = SnowballStemmer(language)
            lemmatizer = WordNetLemmatizer()

            g = stemmer.stem
            #  g = lemmatizer.lemmatize

            # train_texts = [" ".join([lemmatizer.lemmatize(token) for token in WordPunctTokenizer().tokenize(text)]) for text in train_texts]
            train_texts = [" ".join([g(token) for token in WordPunctTokenizer().tokenize(text)]) for text in train_texts]
            test_texts = [" ".join([g(token) for token in WordPunctTokenizer().tokenize(text)]) for text in test_texts]

        # print(train_texts[0])

        if phone: print("inizio vect")

        tok = WordPunctTokenizer().tokenize
        if notokenizer: tok = None
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=grange, tokenizer=tok, lowercase=False, min_df=0.03, sublinear_tf=True)

        if phone: print("fine vect")

        train_data = vectorizer.fit_transform(train_texts)
        test_data = vectorizer.transform(test_texts)

        if Store: pickle.dump({'v': vectorizer, 'train': train_data, 'test': test_data}, open(filename, "wb"))

    print("Vectorizer_config:", vectorizer)
    return train_data, test_data


#
# Features of mine helpers
#


def compare_cosine_similarity(candidate_data, test_data, base, problem):
    print(candidate_data.shape, test_data.shape)
    cosine_matrix = np.zeros((test_data.shape[0], candidate_data.shape[0]), np.float64)

    #  print(candidate_data[0])

    for i, test in enumerate(candidate_data):
        for j, train in enumerate(candidate_data):
            cosine_matrix[i][j] = cosine_similarity(test, train)

    save_dict_into_pickle({'cosine': cosine_matrix}, "cosine_", base, problem)
    return


def check_american_brits(texts):
    word_tokenizer = RegexpTokenizer(r'\w+')
    for i, text in enumerate(texts):
        text = word_tokenizer.tokenize(text)
        a, b = 0, 0
        al, bl = [], []
        for brit, amer in british_american_word:
            if brit.strip() in text: b += 1; bl.append(brit)
            if amer.strip() in text: a += 1; al.append(amer)
        print("text", i, "ha brit:", b, "american:", a, bl, al)


def cosine_similarity_matrix(candidate_data, train_data, test_data):
    print(candidate_data.shape, test_data.shape)
    # cosine_matrix_train = np.zeros((candidate_data.shape[0], train_data.shape[0]), np.float64)
    # cosine_matrix_test = np.zeros((candidate_data.shape[0], test_data.shape[0]), np.float64)

    # print(candidate_data[0])
    # for i, test in enumerate(train_data):
    #    for j, train in enumerate(candidate_data):
    #         cosine_matrix_train[j][i] = cosine_similarity(test, train)
    #         print(cosine_matrix_train[j][i])

    cosine_matrix_train = cosine_similarity(train_data, candidate_data)
    cosine_matrix_test = cosine_similarity(test_data, candidate_data)

    #  print("MAtrix", cosine_matrix_test)
    #  print(type(cosine_matrix_test[0]))
    # print(type(cosine_matrix_test[0][0]))

    # print("Matrix", cosine_matrix_test)

    # for i, test in enumerate(test_data):
    #      for j, train in enumerate(candidate_data):
    #        cosine_matrix_test[j][i] = cosine_similarity(test, train)[0, 1]

    #  save_dict_into_pickle({'cosine': cosine_matrix_train}, "cosine_", base, problem)

    return cosine_matrix_train, cosine_matrix_test


def get_fandom(path, problem, candidates):
    features = list()
    fandom_info = _get_problem_path_of_json(path, problem, 'fandom-info')

    with open(fandom_info, 'r') as f:
        l = json.load(f)
        for d in l:
            features.append(d['author-name'])
    return features


def LexicalFeatures(chapters, language):
    word_tokenizer = RegexpTokenizer(r'\w+')

    # print("Testo:", chapters)

    # create feature vectors
    num_chapters = len(chapters)
    fvs_lexical = np.zeros((len(chapters), 6), np.float64)

    for e, ch_text in np.ndenumerate(chapters):
        # note: the nltk.word_tokenize includes punctuation
        tokens = word_tokenize(ch_text.lower())
        #  words = word_tokenizer.tokenize(ch_text.lower())
        stemmer = SnowballStemmer(language)
        words = [stemmer.stem(word) for word in word_tokenize(ch_text.lower())]
        sentences = sent_tokenize(ch_text, language)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenize(s)) for s in sentences])

        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / float(len(words))

        # Commas per sentence
        fvs_lexical[e, 3] = tokens.count(',') / float(len(sentences))
        # Semicolons per sentence
        fvs_lexical[e, 4] = tokens.count(';') / float(len(sentences))
        # Colons per sentence
        fvs_lexical[e, 5] = tokens.count(':') / float(len(sentences))

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)

    return fvs_lexical


def lunghezza_media_parole(text):
    word_tokenizer = RegexpTokenizer(r'\w+')
    words = word_tokenizer.tokenize(text)

    lunghezza = sum([len(word) for word in words])

    return lunghezza / len(words)


def count_pos_tags(pos, tags):
    if isinstance(tags, list):
        # count = np.fromiter((1 for word in pos if word[1] in tags), int).sum()
        count = np.fromiter((count_pos_tags(pos, tag) for tag in tags), int).sum()
    else:
        if tags in pos_cache: return pos_cache[tags]
        count = np.fromiter((1 for word in pos if word[1] == tags), int).sum()
        pos_cache[tags] = count
    return count


def pos_tagging(text, tags):
    word_tokenizer = RegexpTokenizer(r'\w+')
    words = word_tokenizer.tokenize(text)
    pos = pos_tag(words)
    # print(pos[:10])

    return [count_pos_tags(pos, tag) for tag in tags]


def str_to_phonetics(texts, *params):
    result = []
    for text in texts:
        text = str(text)
        # result.append(" ".join([ipa.convert(word) for word in words]))
        result.append(ipa.convert(text))

    return result


def split_post_tagging(st, words, language, step):
    pos = []

    top = step
    bottom = 0
    while top < len(words):
        pos += st.tag(words[bottom:top])
        print(language, len(words), "step:", top / step)
        top += step
        bottom += step

    pos += st.tag(words[bottom:len(words)])
    return pos


# def stanford_post_tag_string(texts, language, path):
#     use_stored = True and S
#     Store = True and S
#     filename = path + "_stanford_" + language + ".pickle"
#
#     if use_stored and os.path.isfile(filename):
#         print('loaded', filename)
#         d = pickle.load(open(filename, "rb"))
#         result = d['result']
#
#     else:
#         if language == 'english':
#             model = 'english-caseless-left3words-distsim.tagger'
#         elif language == 'french':
#             model = "french.tagger"
#         elif language == 'spanish':
#             model = "spanish-distsim.tagger"
#         st = StanfordPOSTagger(path_stanford_tagger_model + model, path_to_jar=path_stanford_tagger)
#         result = list()
#         print("generating_pos", filename)
#
#         for text in texts:
#             word_tokenizer = RegexpTokenizer(r'\w+')
#             # words = word_tokenizer.tokenize(text)
#             words = word_tokenize(text)
#
#             # print("start_pos")
#             # pos = split_post_tagging(st, words, language)
#             # print("end_pos")
#             pos = st.tag(words)
#
#             result.append(" ".join([tag for (word, tag) in pos]))
#
#         if Store:
#             pickle.dump({'result': result}, open(filename, "wb"))
#             print('stored', filename)
#
#     return result


def create_post_tag_concatenate_string(texts, language, *other):
    result = list()

    for text in texts:
        word_tokenizer = RegexpTokenizer(r'\w+')
        # words = word_tokenizer.tokenize(text)
        words = word_tokenize(text)

        pos = np.array(pos_tag(words))

        new_text = [word[0] + "/" + word[1] for word in pos]

        result.append(' '.join(new_text))

        # # pos = np.fromiter((tag for (word, tag) in pos), np.str)
        #
        # # result.append(" ".join([tag for (word, tag) in pos]))
        # pos = pos[:, 1]
        # result.append(" ".join(pos))

    return result


def create_post_tag_string(texts, language, *other):
    result = list()

    for text in texts:
        word_tokenizer = RegexpTokenizer(r'\w+')
        # words = word_tokenizer.tokenize(text)
        words = word_tokenize(text)

        pos = np.array(pos_tag(words))

        # pos = np.fromiter((tag for (word, tag) in pos), np.str)

        # result.append(" ".join([tag for (word, tag) in pos]))
        pos = pos[:, 1]
        result.append(" ".join(pos))

    return result


def features_of_mine(path, problem, language, train, test):
    result = []
    for texts in [train, test]:
        #  lex_f = std_c.LexicalFeatures(texts, language)
        # fw = std_c.function_words_freq(texts, language)
        # lex_rich = std_c.lexical_richness(texts, language)
        lex_fw = std_c.LexicalFeatures(texts, language)
        # lex = np.hstack([lex, fw])

        # numpy.fromiter((<some_func>(x) for x in <something>),<dtype>,<size of something>)

        pos_tags_list = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
                         'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP',
                         'WP$', 'WRB']

        # pos_tags_list += list(itertools.combinations(pos_tags_list, 2)) + list(itertools.combinations(pos_tags_list, 3))
        #
        # features = np.array([
        #     [lunghezza_media_parole(text)] + pos_tagging(text, pos_tags_list)
        #     for text in texts], int)

        # features = np.fromiter(([ lunghezza_media_parole(text) ] + pos_tagging(text, ['JJ', 'JJR', 'JJS', ['JJ', 'JJR', 'JJS']])
        #    for text in texts), list)

        # result.append(hstack([features, lex]))
        result.append(lex_fw)

    return np.array(result)


#
# Prediction's probabilities management
#


def analize_probas(d):
    predictions = d['predictions']
    predictions2 = d['predictions2']
    proba = d['proba']
    proba2 = d['proba2']

    n = 1

    for p1, p2 in zip(proba, proba2):
        data = np.array([p1, p2])
        print(p1)
        print(p2)
        average = np.average(data, axis=0)
        print(sum(average))


def reject_option_cosine(predictions, proba, pt, problem, language, similarity_matrix):
    # Reject option (used in open-set cases)
    count = 0
    n_maxs = 1
    # print("reject_option_cosine.shape", predictions.shape, similarity_matrix.shape)
    for i, p in enumerate(predictions):
        sproba = sorted(proba[i], reverse=True)
        max_proba, max_proba2, max_proba3 = sorted(proba[i], reverse=True)[:3]

        # index of predicter candidate
        prob_cand_ind = proba[i].argmax()
        # max_cosine = sorted(cosine_matrix[i], reverse=True)[1]
        # print(proba[i].argmax(), cosine_matrix[i].argmax(), proba[i], cosine_matrix[i])

        # max:
        # cos_cand_inds = np.argpartition(similarity_matrix[i], -n_maxs)[n_maxs:]  #  cosine_matrix[i].argmax()
        cos_cand_inds = np.argpartition(similarity_matrix[i], n_maxs)[:n_maxs]  #  cosine_matrix[i].argmax()
        # print("reject_cosine", proba.shape, cosine_matrix.shape)
        # print(prob_cand_ind, cos_cand_inds)

        # if sproba[0] - sproba[1] < pt and ((max_proba - max_proba2) + (max_proba2 - max_proba3)) / 2 < 0.07:
        if prob_cand_ind not in cos_cand_inds and sproba[0] - sproba[1] < pt - 0.01:
            # print(sproba)
            # if sproba[0] < pt:
            # print('Confidenza massima', sproba[0], sproba[1], sproba[0] - sproba[1])
            predictions[i] = u'<UNK>'
            count += + 1
    try:
        print('\t', language, count, 'texts left unattributed of' + str(unk_num[int(problem[-2:])]))
    except Exception as e:
        print('\t', count, 'texts left unattributed', e)
    return predictions


def reject_option_svm(predictions, proba, pt, problem, language, unk_pred):
    # Reject option (used in open-set cases)
    count = 0
    for i, p in enumerate(zip(predictions, unk_pred)):
        p, unk_p = p
        sproba = sorted(proba[i], reverse=True)
        max_proba, max_proba2, max_proba3 = sorted(proba[i], reverse=True)[:3]

        if sproba[0] - sproba[1] < pt and ((max_proba - max_proba2) + (max_proba2 - max_proba3)) / 2 < 0.1 and unk_p < 0:
            #  print(sproba)
            # if sproba[0] < pt:
            # print('Confidenza massima', sproba[0], sproba[1], sproba[0] - sproba[1])
            predictions[i] = u'<UNK>'
            count = count + 1
    try:
        print('\t', language, count, 'texts left unattributed of' + str(unk_num[int(problem[-2:])]))
    except Exception as e:
        print('\t', count, 'texts left unattributed', e)
    return predictions


def reject_option(predictions, proba, pt, problem, language, other_things):
    # Reject option (used in open-set cases)
    count = 0
    for i, p in enumerate(predictions):
        # print("proba[i]", proba[i])
        sproba = sorted(proba[i], reverse=True)
        max_proba, max_proba2, max_proba3 = sorted(proba[i], reverse=True)[:3]

        if sproba[0] - sproba[1] < pt and ((max_proba - max_proba2) + (max_proba2 - max_proba3)) / 2 < 0.07:
            #  print(sproba)
            # if sproba[0] < pt:
            # print('Confidenza massima', sproba[0], sproba[1], sproba[0] - sproba[1])
            predictions[i] = u'<UNK>'
            count = count + 1
    try:
        print('\t', language, count, 'texts left unattributed of' + str(unk_num[int(problem[-2:])]))
    except Exception as e:
        print('\t', count, 'texts left unattributed', e)
    return predictions


def scegli_predizione_piu_convinta(pred1, pred2, proba1, proba2):
    prediction = []
    proba = []

    for i in range(len(pred1)):
        p1 = sorted(proba1[i].tolist())[0]
        p2 = sorted(proba2[i].tolist())[0]

        if p1 > p2:
            prediction.append(pred1[i])
            proba.append(proba1[i])
        else:
            prediction.append(pred2[i])
            proba.append(proba2[i])
    return prediction, proba


def media_delle_probabilita(pred1, pred2, proba1, proba2):
    prediction = []
    proba = []

    for i in range(len(pred1)):
        data = np.array([proba1[i], proba2[i]])
        average = np.average(data, axis=0)
        maxind = np.argmax(average)

        prediction.append('candidate' + str(maxind + 1).zfill(5))
        proba.append(average)

    return prediction, proba


def compression_evaluation(probas):
    prediction = []
    # proba = np.average(probas, axis=0, weights=weights)
    probas = probas[0]

    for p in probas:
        maxind = np.argmax(p)
        prediction.append('candidate' + str(maxind + 1).zfill(5))

    return [prediction], [probas]


def soft_voting(probas, weights=None):
    prediction = []
    proba = np.average(probas, axis=0, weights=weights)

    for p in proba:
        maxind = np.argmax(p)
        prediction.append('candidate' + str(maxind + 1).zfill(5))

    return [prediction], [proba]


def reverse_probabilities(probas):
    prediction = []
    proba = []

    for proba in probas:
        for p in proba:
            p = 1.0 - p

    return probas


def dimentionality_reduction(train_data, test_data):
    for i in range(len(train_data)):
        # red = TruncatedSVD(n_components=200, n_iter=22, random_state=42)
        # red = PCA(n_components=63)
        red = FastICA(n_components=20, random_state=42)
        train_data[i] = red.fit_transform(train_data[i].todense())
        test_data[i] = red.transform(test_data[i].todense())
    return train_data, test_data


def randomforest_clf():
    # nb_parameter = {'fit_prior':[True,False], 'alpha':[0, 0.1, 0.3, 0.5, 0.7, 1., 2., 3.]}
    clf = RandomForestClassifier(min_samples_split=50, n_estimators=500)
    return clf


def scale(train_data, test_data, print_time=False):
    t = time()
    max_abs_scaler = preprocessing.MaxAbsScaler()

    for i in range(len(train_data)):
        train_data[i] = max_abs_scaler.fit_transform(train_data[i])
        test_data[i] = max_abs_scaler.transform(test_data[i])

    if print_time: print("Scaler Time:", time() - t)
    return train_data, test_data


def Ada(base_clf=None):
    if base_clf is None: base_clf = svm_clf()
    clf = AdaBoostClassifier(base_estimator=CalibratedClassifierCV(OneVsRestClassifier(base_clf)), n_estimators=50)
    return clf


def NeuralNet():
    '''
    clf_best_estimator: OneVsRestClassifier(estimator=GridSearchCV(cv=4, error_score='raise-deprecating',       estimator=MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,       beta_2=0.999, early_stopping=False, epsilon=1e-08,       hidden_layer_sizes=(100,), learning_rate='constant',       learning_rate_init...re_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring='f1_macro', verbose=2), n_jobs=None)    46 texts left unattributed    answers saved to file /opt/projects/attribution/src/Code/baseline/my_outs/ answers-problem00004.json
    '''
    clf = MLPClassifier(random_state=42)
    parameters = {
        'solver': ['lbfgs', 'sgd', 'adam'],
        # 'activation': ['identity', 'logistic', 'tanh', 'relu'],
        # 'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'max_iter': [200, 600, 800],
        'hidden_layer_sizes': [(100,), (150,), (80,), (120,)]
    }
    #  clf = GridSearchCV(clf, parameters, n_jobs=16, scoring='f1_macro', cv=3, verbose=2)
    clf = MLPClassifier(random_state=1, hidden_layer_sizes=(100,))
    return clf


def svm_clf(kernel=None):
    #  clf = SVC(kernel='linear', C=10, gamma='auto', probability=True, cache_size=1000, class_weight='balanced')
    n_estimators = 3
    # SVC(kernel='linear',  C=0.01, probability=True, class_weight='auto')

    parameters = {'kernel': ['linear', 'rbf'], 'C': [0.5, 1, 5, 10], 'gamma': ['auto', 'scale']}
    parameters = {'penalty': ['l2'], 'C': [0.0001, 0.001, 0.01, 0.5, 1, 5, 10, 20], 'loss': ['hinge', 'squared_hinge']}

    if kernel is not None:
        clf = LinearSVC(class_weight='balanced', C=1.0, multi_class='ovr')
    else:
        clf = SVC(kernel="rbf", probability=True, cache_size=1000)

    #  clf = BaggingClassifier(clf, max_samples=1.0, n_estimators=n_estimators, n_jobs=1, random_state=42)

    clf_grid = SVC(probability=True, cache_size=1000, class_weight='balanced')
    #  clf = GridSearchCV(clf, parameters, n_jobs=16, scoring='f1_macro', cv=5, verbose=2)
    return clf


def one_class_svm_clf(kernel=None):
    clf = LinearSVC(class_weight='balanced', C=1.0, multi_class='ovr')
    #  clf = BaggingClassifier(clf, max_samples=1.0, n_estimators=n_estimators, n_jobs=1, random_state=42)

    clf = OneClassSVM(kernel='linear')
    parameters = {'kernel': ['linear', 'rbf', 'sigmoid'], 'gamma': ['auto', 'scale'], 'shrinking': [True, False]}
    # clf = GridSearchCV(clf, parameters, n_jobs=16, scoring='f1_macro', cv=5, verbose=2)
    return clf


def classifier():
    svm = svm_clf("rbf")
    # kn_clf = KNeighborsClassifier()
    net = NeuralNet()
    ada = Ada()
    vclf = VotingClassifier(
        estimators=[('ada', CalibratedClassifierCV(OneVsRestClassifier(ada))), ('net', CalibratedClassifierCV(OneVsRestClassifier(NeuralNet()))),
                    ("svm", CalibratedClassifierCV(OneVsRestClassifier(svm)))],
        voting='soft', n_jobs=2)
    clf = CalibratedClassifierCV(OneVsRestClassifier(svm))

    return clf


def porbabilities_clf(cross_pred, proba_list, train_labels):
    # clf = CalibratedClassifierCV(OneVsRestClassifier(BaggingClassifier(svm_clf("linear"), max_samples=1.0, n_estimators=10, n_jobs=3, random_state=42)))
    clf = CalibratedClassifierCV(OneVsRestClassifier(classifier()))
    # clf = classifier()
    # clf = CalibratedClassifierCV(OneVsRestClassifier(MLPClassifier(random_state=42)))
    # clf = Ada(svm_clf("linear"))

    clf = VotingClassifier(
        estimators=[('ada', CalibratedClassifierCV(OneVsRestClassifier(Ada(classifier())))),
                    ('net', NeuralNet()),
                    ("svm", CalibratedClassifierCV(OneVsRestClassifier(classifier()))),
                    ("ada_lin", CalibratedClassifierCV(OneVsRestClassifier(Ada(svm_clf("linear"))))),
                    ("rand", CalibratedClassifierCV(OneVsRestClassifier(randomforest_clf())))
                    ],
        voting='soft', n_jobs=4)

    clf.fit(cross_pred, train_labels)
    pred = clf.predict(proba_list)

    try:
        proba_list = clf.predict_proba(proba_list)
    except Exception:
        proba_list = clf._predict_proba_lr(proba_list)

    #  print("clf_best_estimator:", clf.base_estimator)

    # print(pred)

    return [pred], [proba_list]


# def vectorization_wordtest(train_texts, test_texts, path, problem, language, candidate_grouped_texts):
#     t0 = time()
#
#     # train_data_c, test_data_c = std.char_gram(base, problem, train_texts, test_texts, language, use_stored=True and S, Store=True and S)
#
#     S = False
#     train_data_w, test_data_w = word_gram(base, problem, train_texts, test_texts, language, grange=(1, 1), use_stored=True and S, Store=True and S)
#     train_data_w2, test_data_w2 = word_gram(base, problem, train_texts, test_texts, language, grange=(2, 2), use_stored=True and S, Store=True and S)
#     train_data_w3, test_data_w3 = word_gram(base, problem, train_texts, test_texts, language, grange=(3, 3), use_stored=True and S, Store=True and S)
#
#     # print(cosine_matrix_test.shape)
#     # train_data, test_data = hstack([train_data_w, train_data_w2, train_data_w3, train_data_w4]), hstack([test_data_w, test_data_w2, test_data_w3, test_data_w4])
#     train_data, test_data = hstack([train_data_w, train_data_w2, train_data_w3]), hstack([test_data_w, test_data_w2, test_data_w3])
#
#     if MULTICLASSIFIER: return [train_data_w, train_data_w2], [test_data_w, test_data_w2], None
#     # return [train_data_cw], [test_data_cw]
#     return [train_data], [test_data], None


def unk_classification(train_data, test_data, path, problem):
    t0 = time()

    # Applying SVM
    predictions_list = list()
    proba_list = list()

    clfs = [one_class_svm_clf(), one_class_svm_clf(), one_class_svm_clf()]
    train_datas_to_clfs = [train_data[0]]
    test_datas_to_clfs = [test_data[0]]

    for clf, train_datas_to_clf, test_datas_to_clf in zip(clfs, train_datas_to_clfs, test_datas_to_clfs):
        clf.fit(train_datas_to_clf.todense())
        # print("clf_best_estimator_unk:", problem, clf.base_estimator)
        print("classifier_config:", clf)
        predictions_list.append(clf.predict(test_datas_to_clf.todense()))
        # proba_list.append(clf._predict_proba_lr(test_datas_to_clf))
        # proba_list.append(clf.predict_proba(test_datas_to_clf))

    # print(proba[1])
    print('Classification time:', time() - t0)

    # std.save_dict_into_pickle({'predictions': predictions, 'predictions2': predictions2, 'proba': proba, 'proba2': proba2,
    #                            'label': test_data, 'label2': test_data_c, }, "probas_and_predictions", base, problem)

    return predictions_list[0], proba_list

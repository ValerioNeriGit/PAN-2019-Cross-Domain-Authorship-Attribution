import math
from timeit import timeit
from time import time

cimport numpy as np
import numpy as np
from nltk import word_tokenize, SnowballStemmer, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from scipy.cluster.vq import whiten
import ref.function_words as function_words

word_tokenizer = RegexpTokenizer(r'\w+')

def function_words_freq(np.ndarray chapters, language):
    fw_list = function_words.fw_list
    cdef np.ndarray fw_lexical = np.zeros((len(chapters), len(fw_list['en'])), np.float64)

    cdef int t = 0
    cdef int w = 0
    cdef int sum_of_all_fw
    for text in chapters:

        w = 0
        for fw in fw_list['en']:
            # print(w, text)
            fw_lexical[t, w] = text.count(fw)
            w += 1

        sum_of_all_fw = fw_lexical[t].sum(axis=0)

        w = 0
        for fw in function_words.fw_list['en']:
            fw_lexical[t, w] = fw_lexical[t, w] / float(sum_of_all_fw)
            w += 1

        t += 1
    return fw_lexical

cdef np.ndarray _lexical_richness_V(list words, set vocabulary, int words_size):
    cdef np.ndarray counter = np.zeros((words_size, 1), int)
    cdef int f

    for word in vocabulary:
        f = words.count(word)
        counter[f] += 1

    return counter

def lexical_richness(np.ndarray chapters, language):
    cdef np.ndarray lex_richness = np.zeros((len(chapters), 3), np.float64)
    cdef np.ndarray counter
    cdef set vocabulary
    cdef list words
    cdef int words_size, vocabulary_size
    cdef double k_sum_i

    cdef int t = 0
    for text in chapters:
        words = word_tokenizer.tokenize(text.lower())
        vocabulary = set(words)

        words_size = len(words)
        vocabulary_size = len(set(words))

        # Guiraud (1954)
        lex_richness[t, 0] = vocabulary_size / words_size
        # Herdan in 1960 and 1964
        lex_richness[t, 1] = math.log2(vocabulary_size) / math.log2(math.log2(words_size))

        # K(N)
        k_sum_i = 0
        counter = _lexical_richness_V(words, vocabulary, words_size)
        for i in range(words_size):
            k_sum_i += int(counter[i]) * (i/float(words_size))**2

        # lex_richness[t, 2] = 10**4 * ( -(1 / float(words_size)) + k_sum_i)
        # lex_richness[t, 2] = 10**4 * ((k_sum_i - words_size) / words_size**2)
        print(
            10**4 * ( -(1 / float(words_size)) + k_sum_i),
            10**4 * ((k_sum_i - words_size) / float(words_size)**2)
        )
        t += 1
    return lex_richness

def LexicalFeatures(np.ndarray chapters, language):

    # create feature vectors
    cdef int num_chapters = len(chapters)
    cdef np.ndarray fvs_lexical = np.zeros((len(chapters), 3 + len(function_words.pun_marks)) + len(function_words.alphabet) * 2, np.float64)
    cdef np.ndarray words_per_sentence
    cdef list tokens
    cdef int e = 0
    cdef np.ndarray words
    cdef set vocab
    cdef float len_words

    #stemmer = SnowballStemmer(language)

    for ch_text in chapters:
        # note: the nltk.word_tokenize includes punctuation
        tokens = word_tokenize(ch_text.lower())
        # words = word_tokenizer.tokenize(ch_text.lower())

        # words = [stemmer.stem(word) for word in word_tokenize(ch_text.lower())]

        # words = np.array(np.fromiter((stemmer.stem(word) for word in word_tokenize(ch_text.lower())), dtype=object))
        # words = np.array([stemmer.stem(word) for word in word_tokenize(ch_text.lower())])

        # Questo funziona
        # words = np.array([word for word in word_tokenizer.tokenize(ch_text.lower())])
        words = np.array(word_tokenizer.tokenize(ch_text.lower()))

        sentences = sent_tokenize(ch_text, language)
        vocab = set(words)
        words_per_sentence = np.array([len(word_tokenize(s)) for s in sentences])

        len_words = len(words)

        # average number of words per sentence
        fvs_lexical[e, 0] = words_per_sentence.mean()
        # sentence length variation
        fvs_lexical[e, 1] = words_per_sentence.std()
        # Lexical diversity
        fvs_lexical[e, 2] = len(vocab) / len_words
        # frequency of number of stopword
        # fvs_lexical[e, 3] = (len(vocab) - len(vocab.difference(set(stopwords.words(language))))) / len_words

        i = 3
        for mark in function_words.pun_marks:
            fvs_lexical[e, i] = ch_text.count(mark) / float(len(sentences))
            # print(mark, tokens.count(mark))

        for c in function_words.alphabet:
            fvs_lexical[e, i] = ch_text.count(c.lower()) / float(len(sentences))

        for c in function_words.alphabet:
            fvs_lexical[e, i] = ch_text.count(c) / float(len(sentences))

        # # Commas per sentence
        # fvs_lexical[e, 3] = tokens.count(',') / float(len(sentences))
        # # Semicolons per sentence
        # fvs_lexical[e, 4] = tokens.count(';') / float(len(sentences))
        # # Colons per sentence
        # fvs_lexical[e, 5] = tokens.count(':') / float(len(sentences))
        e += 1

    # apply whitening to decorrelate the features
    fvs_lexical = whiten(fvs_lexical)
    return fvs_lexical

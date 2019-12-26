from sklearn.externals.funcsigs import signature

import std
from eval import open_truth_to_dict
from sklearn.metrics import precision_recall_curve, average_precision_score
import numpy as np
import matplotlib.pyplot as plt

base = "/opt/projects/attribution/src/"


def plot_prec_recall_vs_tresh(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.xlabel('Threshold')
    plt.legend(loc='upper left')
    plt.ylim([0, 1])


def precision_recall_plot():
    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
                   if 'step' in signature(plt.fill_between).parameters
                   else {})
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()


for problem in [5]:
    truth = open_truth_to_dict(base + 'Datasets/training-dataset-2019-01-23' + '/problem000{0}/ground-truth.json'.format(str(problem + 1).zfill(2)))
    cosine_matrix = std.load_dict_into_pickle("cosine_", base, "problem000{0}".format(str(problem + 1).zfill(2)))['cosine']

    labels = []
    y_pred = []

    for i in range(cosine_matrix.shape[0]):
        # for j in range(cosine_matrix.shape[1]):
        l = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        cand_ind = int(truth['unknown00{0}.txt'.format(str(problem + 1).zfill(3))][-4:]) - 1
        l[cand_ind] = 1
        labels.extend(l)
        y_pred.extend(cosine_matrix[i].tolist())

    print(cosine_matrix.shape)
    precision, recall, _ = precision_recall_curve(labels, np.array(y_pred))
    average_precision = average_precision_score(labels, np.array(y_pred))

    precision_recall_plot()

# cosine_matrix = cosine_matrix.flat

'''
('problem00006', 'Macro-F1:', 0.642)
('problem00007', 'Macro-F1:', 0.599)
('problem00008', 'Macro-F1:', 0.662)
('problem00009', 'Macro-F1:', 0.737)
('problem00010', 'Macro-F1:', 0.647)
'''
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from dataset_utils import split_dataset_by_bias
import numpy as np
import os
import json
import re


def cm_to_fig(cm, label_map_func):
    num_labels = cm.shape[0]
    labels = [label_map_func(lbl) for lbl in range(num_labels)]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    norm = plt.Normalize()
    colours = plt.cm.Wistia(norm(df.values))
    colours = np.concatenate((colours, np.zeros((2, len(labels), 4))), axis=0)
    colours = np.concatenate((colours, np.zeros((len(labels)+2, 2, 4))), axis=1)
    df.loc['Predicted%'] = df.sum()
    df['GT%'] = df.sum(axis=1)
    df = df.round(2)
    df.iloc[-1, -1] = ''
    df.loc['Precision'] = [np.around(df.iloc[i, i] / df.loc['Predicted%', labels[i]], 2) if df.loc['Predicted%', labels[i]] > 0 else 0 for i in range(num_labels)] + ['']
    df['Recall'] = [np.around(df.iloc[i, i] / df.loc[labels[i], 'GT%'], 2) if df.loc[labels[i], 'GT%'] > 0 else 0 for i in range(num_labels)] + ['', '']

    nrows, ncols = len(labels)+2, len(labels)+2
    fig, ax = plt.subplots(figsize=(10.4, 4.8), dpi=100)
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellColours=colours)

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    cell_height = 1 / nrows
    cell_width = 1 / (nrows + 1)
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    return fig


def plot_scores_histogram(score_label_dict, num_labels, label_map_func):
    """Plot histogram of scores in scores_dir.
    Histogram - for each subset (train, val and test) and for each label generate a histogram of
    binned predictions (quantize predictions to 10 bins spreading [0,1]).
     """
    num_sets = len(score_label_dict)
    fig, axes = plt.subplots(nrows=num_sets, ncols=num_labels, sharex=True, sharey='row', figsize=(20, 15), dpi=125)
    num_rows, num_cols = num_sets, num_labels
    for subset, ax in zip(score_label_dict.keys(), axes):
        scores, labels = score_label_dict[subset]
        ind, prob = [np.array(list(x)) for x in zip(*scores)]  # shapes: ind - n , prob - n x num_labels
        labels = np.array(labels)[ind]
        for i, a in enumerate(ax):
            n, bins, patches = a.hist(prob[labels == i, i], bins=np.arange(0.0, 1.1, 0.1),
                                      edgecolor='black', rwidth=0.7)
            a.set_xticks(bins)
        ax[0].set_ylabel('Count of samples')
    for j in range(num_cols):
        axes[-1, j].set_xlabel('Score')


    # <editor-fold desc="Col and row titles">
    pad = 20  # in points

    for ax, row in zip(axes[:, 0], score_label_dict.keys()):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='xx-large', ha='right', va='center')

    for ax, col in zip(axes[0], [f'{label_map_func(j)}' for j in range(num_labels)]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='xx-large', ha='center', va='baseline')
    # </editor-fold>

    return fig


def plot_overlap_ratio_histogram(overlap_label_dict, num_labels, label_map_func):
    """Plot histogram of scores in scores_dir.
    Histogram - for each subset (train, val and test) and for each label generate a histogram of
    binned predictions (quantize predictions to 10 bins spreading [0,1]).
     """
    num_sets = len(overlap_label_dict)
    fig, axes = plt.subplots(nrows=num_sets, ncols=num_labels, sharex=True, sharey='row', figsize=(20, 15), dpi=125)
    for subset, ax in zip(overlap_label_dict.keys(), axes):
        num_rows, num_cols = axes.shape
        overlap, labels = np.array(overlap_label_dict[subset])
        for i, a in enumerate(ax):
            n, bins, patches = a.hist(overlap[labels == i], bins=np.arange(0.0, 1.1, 0.1),
                                      edgecolor='black', rwidth=0.7)
            a.set_xticks(bins)
        ax[0].set_ylabel('Count of samples')
    for j in range(num_cols):
        axes[-1, j].set_xlabel('Overlap Ratio')

    # <editor-fold desc="Col and row titles">
    pad = 20  # in points

    for ax, row in zip(axes[:, 0], overlap_label_dict.keys()):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    size='xx-large', ha='right', va='center')

    for ax, col in zip(axes[0], [f'{label_map_func(j)}' for j in range(num_labels)]):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                    xycoords='axes fraction', textcoords='offset points',
                    size='xx-large', ha='center', va='baseline')
    # </editor-fold>

    return fig


def plot_splits_vs_thresholds(ds, scores, set=''):
    fig, ax = plt.subplots(figsize=(25, 20))

    thresh1_vals = np.arange(0.01, 0.25, 0.01)
    # thresh1_vals = np.arange(0.05, 0.5, 0.05)
    # thresh2_vals = np.arange(0.4, 0.85, 0.05)
    thresh2_vals = np.arange(0.2, 0.65, 0.05)  # len: 10

    t_table = []  # len(t1_vals) x len(t2_vals)
    for t1 in thresh1_vals:
        t2_table = []
        for t2 in thresh2_vals:
            unbiased_ind, correctly_biased_ind, incorrectly_biased_ind = split_dataset_by_bias(ds, scores,
                                                                                               t1, t2)
            num_unbiased = len(unbiased_ind)
            num_correctly_biased = sum([len(correctly_biased_ind[i]) for i in range(len(correctly_biased_ind))])
            num_incorrectly_biased = sum([len(incorrectly_biased_ind[i]) for i in range(len(incorrectly_biased_ind))])
            N = num_unbiased + num_correctly_biased + num_incorrectly_biased
            t2_table.append(f'{round(num_unbiased / N, 2)}, {round(num_correctly_biased / N, 2)}, '
                            f'{round(num_incorrectly_biased / N, 2)}')
        t_table.append(t2_table)

    ax.axis('off')
    table = ax.table(cellText=t_table, colLabels=['t2=' + str(x) for x in np.round(thresh2_vals, 2)],
                     rowLabels=['t1=' + str(x) for x in np.round(thresh1_vals, 2)], loc='center')
    ax.set_title(f'Split vs Threshold analysis, {set} set\n'
                 f'table format = (%unbiased, %correctly biased, %incorrectly biased)')
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    nrows, ncols = len(t_table), len(t_table[0])
    cell_height = 1 / (nrows + 4)
    cell_width = 1 / ncols
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    return fig


def premise_length_table(samples):
    fig, ax = plt.subplots(figsize=(25, 20))

    seq_len = np.arange(40, 100, 10)
    subset = ['All']

    s_table = []  # len(t1_vals) x len(t2_vals)
    for s in seq_len:
        sub_table = []
        cnt = 0
        for samp in samples:
            p, h, y = samp
            if len(p.split()) > s:
                cnt += 1
        for sub in subset:
            sub_table.append(f'{cnt} samples, %{round(100 * cnt / len(samples), 3)}')

        s_table.append(sub_table)

    ax.axis('off')
    table = ax.table(cellText=s_table, colLabels=['subset=' + str(x) for x in subset],
                     rowLabels=['S=' + str(x) for x in seq_len], loc='center')
    ax.set_title(f'percentage of premises longer than S, for genre {set} \n')
    table.auto_set_font_size(False)
    table.set_fontsize(14)

    nrows, ncols = len(s_table), len(s_table[0])
    cell_height = 1 / (nrows + 4)
    cell_width = 1 / ncols
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    return fig


def bar_plot_metric(dir, metric='accuracy'):
    assert os.path.isdir(dir), f'{dir} - No such directory'

    test_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+$', x[0]) is not None and 'run_output.json' in x[2],
               os.walk(dir)))
    dirs = '\n'.join(list(map(lambda x: x[0], test_dirs)))

    fig, ax = plt.subplots(figsize=(25, 10), dpi=250)
    for d, shift in zip(test_dirs, np.linspace(-0.2, 0.2, len(test_dirs))):
        with open(os.sep.join([d[0], 'run_output.json'])) as f:
            res = json.load(f)['results']
        test_metric_per_env = np.array(res['test_acc_per_env'])  # num_envs x num_steps
        test_metric_per_env.sort(axis=0)
        test_metric_mean = np.mean(test_metric_per_env, axis=0)
        yerr_plus = test_metric_per_env[0, :] - test_metric_mean
        yerr_minus = test_metric_mean - test_metric_per_env[1, :]
        steps = np.arange(len(test_metric_mean))

        trans = Affine2D().translate(shift, 0.0) + ax.transData
        ax.errorbar(steps, test_metric_mean, yerr=np.vstack([yerr_minus, yerr_plus]), fmt='o', label=d[0].split(os.sep)[-1], transform=trans)

    ax.legend()
    fig.show()


if __name__ == "__main__":
    bar_plot_metric('exp1/irm')
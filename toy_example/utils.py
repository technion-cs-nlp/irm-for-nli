from typing import NamedTuple, List
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json
import contextlib
import torch
import setting
import itertools as it


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: error (ERM term), penalty (IRM term), loss
    and number of correct classifications.
    """
    loss: float
    error: float
    penalty: float
    num_correct: int
    pred: List[int]
    gt: List[int]


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the error, penalty, loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    errors: List[float]
    penalties: List[float]
    accuracy: float
    cm: np.ndarray


class FitResult(NamedTuple):
    """
    Represents the result of fitting a model for multiple epochs given a
    training and test (or validation) set.
    The losses are for each batch and the accuracies are per epoch.
    """
    num_warm_up_steps: int
    num_steps: int
    train_loss: List[float]
    train_error: List[float]
    train_penalty: List[float]
    train_acc: List[float]
    test_loss: List[float]
    test_acc: List[float]
    test_error: List[float]
    test_penalty: List[float]
    test_loss_ood: List[float]
    test_acc_ood: List[float]
    test_error_ood: List[float]
    test_penalty_ood: List[float]


class TestResult(NamedTuple):
    """
    Represents the result of testing a model for a single epoch: loss and accuracy on dataset
    """
    loss: float
    accuracy: float


def tensorboard_add_result(writer, result, step, set=''):
    writer.add_scalar(f'Loss/{set}', np.array(result.losses).mean(), step)
    writer.add_scalar(f'Error/{set}', np.array(result.errors).mean(), step)
    writer.add_scalar(f'Penalty/{set}', np.array(result.penalties).mean(), step)
    writer.add_scalar(f'Accuracy/{set}', np.array(result.accuracy).mean(), step)

    fig = cm_to_fig(result.cm)
    writer.add_figure('ConfusionMatrix'+'/'+set, fig, step)


# show_gpu('Cuda usage before training:')
# reporter = MemReporter()
# print('Detailed GPU usage before training:\n')
# reporter.report()

def show_gpu(msg):
    """
    ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4
    """

    def query(field):
        return (subprocess.check_output(
            ['nvidia-smi', f'--query-gpu={field}',
             '--format=csv,nounits,noheader'],
            encoding='utf-8'))

    def to_int(result):
        return int(result.strip().split('\n')[0])

    used = to_int(query('memory.used'))
    total = to_int(query('memory.total'))
    pct = used / total
    print('\n' + msg, f'{100 * pct:2.1f}% ({used} out of {total})')


def cm_to_fig(cm):
    labels = [setting.labels_int_to_string(i) for i in range(setting.NUM_LABELS)]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    norm = plt.Normalize()
    colours = plt.cm.Wistia(norm(df.values))
    colours = np.concatenate((colours, np.zeros((2, len(labels), 4))), axis=0)
    colours = np.concatenate((colours, np.zeros((len(labels)+2, 2, 4))), axis=1)
    df.loc['Predicted%'] = df.sum()
    df['GT%'] = df.sum(axis=1)
    df = df.round(2)
    df.iloc[-1, -1] = ''
    df.loc['Precision'] = [np.around(df.iloc[i, i] / df.loc['Predicted%', labels[i]], 2) if df.loc['Predicted%', labels[i]] > 0 else 0 for i in range(len(labels))] + ['']
    df['Recall'] = [np.around(df.iloc[i, i] / df.loc[labels[i], 'GT%'], 2) if df.loc[labels[i], 'GT%'] > 0 else 0 for i in range(len(labels))] + ['', '']

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


def update_results(list_field_tuple, result):
    for results_list, field_name in list_field_tuple:
        results_list.append(np.array(getattr(result, field_name)).mean().item())


def calc_mean_var_for_test(parent_dir):
    acc = []
    test_dirs = list(filter(lambda x: re.match('.*run[0-9]+.*/test_indomain', x[0]) is not None and 'run_output.json' in x[2], os.walk(parent_dir)))
    for d in test_dirs:
        with open(os.sep.join([d[0], 'run_output.json'])) as f:
            acc.append(json.load(f)['results']['accuracy'])
    mu, std = np.mean(acc), np.std(acc)
    dirs = '\n'.join(list(map(lambda x: x[0], test_dirs)))
    print(f'For test runs: {dirs}\n calculated mean: {mu} and standard deviation: {std}')


@contextlib.contextmanager
def wrap_confusion_matrix(num_labels, y_gt, y_pred, weights=None):
    try:
        if weights is None:
            weights = torch.ones_like(y_gt)
        weights = torch.cat([weights, torch.zeros(num_labels)])
        y_gt = torch.cat([y_gt, torch.arange(num_labels)])
        y_pred = torch.cat([y_pred, torch.arange(num_labels)])
        yield y_gt, y_pred, weights
    finally:
        pass


def plot_predictions(samples_pred_dict):
    # label doesn't affect predictions - leave samples with correct label for plotting
    true_signal_combinations = list(it.product(setting.VOCAB_SIG, setting.VOCAB_SIG))
    color_per_true_signal = dict(zip(true_signal_combinations, ['b', 'r', 'y', 'g']))
    line_type_per_bias_signal = dict(zip(setting.VOCAB_BIAS, ['-', '--']))

    unlabeled_samples = set(map(lambda x: x[:-1],list(samples_pred_dict)))
    gt_labeled_samp_pred_dict = dict()
    for samp in unlabeled_samples:
        p, h = samp
        assert samples_pred_dict[p, h, 0] == samples_pred_dict[p, h, 1], "Different predictions for same sample"
        y_gt = int(p.split()[0] == h.split()[0])
        gt_labeled_samp_pred_dict[(p, h, y_gt)] = samples_pred_dict[(p, h, 0)]

    fig, axes = plt.subplots(nrows=2, figsize=(8, 12), dpi=185)
    for lbl in range(setting.NUM_LABELS):
        for (p, h, y_gt), pred in gt_labeled_samp_pred_dict.items():
            if y_gt == lbl:
                axes[lbl].plot(pred, label=f'{p}, {h}', color=color_per_true_signal[(p, h[0])], linestyle=line_type_per_bias_signal[h[-1]])
        axes[lbl].set_ylabel(f'probability of entailment', fontsize=12)
        axes[lbl].set_xlabel(f'Steps', fontsize=12)
        axes[lbl].legend(fontsize=14)

    for lbl, ax in enumerate(axes):
        ax.annotate(f'gold label: {setting.labels_int_to_string(lbl)}', xy=(0.5, 1), xytext=(0, 12),
                    xycoords='axes fraction', textcoords='offset points',
                    size='xx-large', ha='center', va='center')

    return fig

from typing import NamedTuple, List
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import json
import contextlib
from collections import defaultdict
import torch
from settings import NUM_LABELS, labels_int_to_string

PM = u'\u00b1'


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch: error (ERM term), penalty (IRM term), loss
    and number of correct classifications.
    """
    loss: float
    error: List[float]
    penalty: List[float]
    num_correct: List[int]
    pred: List[int]
    gt: List[int]
    pred_prob: List[List[float]] = None
    weights: List[float] = None


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch: the error, penalty, loss per batch
    and accuracy on the dataset (train or test).
    """
    losses: List[float]
    errors: List[float]
    penalties: List[float]
    errors_per_env: List[List[float]]
    penalties_per_env: List[List[float]]
    accuracy: float
    accuracies_per_env: List[List[float]]
    cm: np.ndarray
    pred_prob: List[List[float]] = None
    weights: List[List[float]] = None


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
    # per env
    train_error_per_env: List[List[float]]
    train_penalty_per_env: List[List[float]]
    train_acc_per_env: List[List[float]]
    test_acc_per_env: List[List[float]]
    test_error_per_env: List[List[float]]
    test_penalty_per_env: List[List[float]]
    test_acc_ood_per_env: List[List[float]]
    test_error_ood_per_env: List[List[float]]
    test_penalty_ood_per_env: List[List[float]]


class TestResult(NamedTuple):
    """
    Represents the result of testing a model for a single epoch: loss and accuracy on dataset
    """
    loss: float
    accuracy: float
    cm: np.ndarray
    pred_prob: List[List[float]] = None


def tensorboard_add_result(writer, result, step, set=''):
    writer.add_scalar(f'Loss/{set}', np.array(result.losses).mean(), step)
    writer.add_scalar(f'Error/{set}', np.array(result.errors).mean(), step)
    for i, error_per_env in enumerate(result.errors_per_env):
        writer.add_scalar(f'Error environemnt {i}/{set}', np.array(error_per_env).mean(), step)
    writer.add_scalar(f'Penalty/{set}', np.array(result.penalties).mean(), step)
    for i, penalty_per_env in enumerate(result.penalties_per_env):
        writer.add_scalar(f'Penalty environemnt {i}/{set}', np.array(penalty_per_env).mean(), step)
    writer.add_scalar(f'Accuracy/{set}', np.array(result.accuracy).mean(), step)

    if result.weights:
        for term, w in zip(['Error', 'Penalty'], result.weights):
            writer.add_scalar(f'{term} term weight/{set}', np.array(w).mean(), step)

    fig = cm_to_fig(result.cm)
    writer.add_figure('ConfusionMatrix' + '/' + set, fig, step)


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
    labels = [labels_int_to_string(lbl) for lbl in range(NUM_LABELS)]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    norm = plt.Normalize()
    colours = plt.cm.Wistia(norm(df.values))
    colours = np.concatenate((colours, np.zeros((2, len(labels), 4))), axis=0)
    colours = np.concatenate((colours, np.zeros((len(labels) + 2, 2, 4))), axis=1)
    df.loc['Predicted%'] = df.sum()
    df['GT%'] = df.sum(axis=1)
    df = df.round(2)
    df.iloc[-1, -1] = ''
    df.loc['Precision'] = [np.around(df.iloc[i, i] / df.loc['Predicted%', labels[i]], 2) if df.loc['Predicted%', labels[
        i]] > 0 else 0 for i in range(len(labels))] + ['']
    df['Recall'] = [np.around(df.iloc[i, i] / df.loc[labels[i], 'GT%'], 2) if df.loc[labels[i], 'GT%'] > 0 else 0 for i
                    in range(len(labels))] + ['', '']

    nrows, ncols = len(labels) + 2, len(labels) + 2
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


def general_cm_to_fig(cm, cm_std=None, colormap_name='coolwarm'):
    labels = [labels_int_to_string(lbl) for lbl in range(NUM_LABELS)]
    df = pd.DataFrame(cm, index=labels, columns=labels)
    norm = plt.Normalize()
    cmap = plt.get_cmap(colormap_name)
    colours = cmap(norm(df.values))

    fig, ax = plt.subplots(figsize=(10.4, 4.8), dpi=100)
    ax.axis('off')
    if cm_std is None:
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center',
                         cellColours=colours)
    else:
        final_cm = [[str(x) + PM + str(y) for x, y in zip(mu_sublist, std_sublist)] for mu_sublist, std_sublist in
                    zip(cm.tolist(), cm_std.tolist())]
        final_cm = np.array(final_cm)
        df_final = pd.DataFrame(final_cm, columns=labels, index=labels)
        table = ax.table(cellText=df_final.values, colLabels=df.columns, rowLabels=df.index, loc='center',
                         cellColours=colours)

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    nrows, ncols = len(labels)+1, len(labels)+1
    cell_height = 1 / nrows
    cell_width = 1 / (nrows + 1)
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    return fig


def attribution_to_heatmap_figure(attr, raw_text):
    assert isinstance(attr, dict), "attribution should be dictionary of length = num_lables"
    labels = list(attr.keys())
    labels.sort()
    df = pd.DataFrame([attr[k].squeeze().tolist() for k in labels], index=labels, columns=raw_text)
    df = df.round(4)
    norms = [plt.Normalize(-1, 1), plt.Normalize(-1, 1), plt.Normalize(-1, 1)]
    colours = np.concatenate([plt.cm.RdYlGn(norm(attr[lbl])) for lbl, norm in zip(labels, norms)], axis=0)

    nrows, ncols = len(labels), len(raw_text)
    # fig, ax = plt.subplots(figsize=(40.4, 12.8), dpi=100)
    hcell, wcell = 1., 1.2
    hpad, wpad = 0, 0
    fig, ax = plt.subplots(figsize=(ncols * wcell + wpad, nrows * hcell + hpad), dpi=100)
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center', cellColours=colours)

    table.auto_set_font_size(False)
    table.set_fontsize(12)

    cell_height = 1 / nrows
    cell_width = 1 / (ncols + 1)
    for pos, cell in table.get_celld().items():
        cell.set_height(cell_height)
        cell.set_width(cell_width)

    return fig


def calc_mean_var_for_test(parent_dir, test_dir, metric='accuracy', verbose=True):
    """
    Search for directories with the pattern "parent_dir/<any_num_of_subdirs>/run<number>/test_dir
    containing a 'run_output.json' file and calculate accuracy (mean and std) across these runs.
    The accuracy is printed and returned as list
    """
    metric_list = []
    test_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*/' + test_dir, x[0]) is not None and 'run_output.json' in x[2],
               os.walk(parent_dir)))
    dirs = '\n'.join(list(map(lambda x: x[0], test_dirs)))
    if verbose:
        print(f'Calculating mean and std for test runs: {dirs}\n')
    for d in test_dirs:
        with open(os.sep.join([d[0], 'run_output.json'])) as f:
            res = json.load(f)['results']
            metric_list.append(res[metric])
    mu, std = np.mean(metric_list), np.std(metric_list)
    metric_min, metric_max = np.min(metric_list), np.max(metric_list)
    if verbose:
        print(
        f'Accuracy - {np.round(mu, 2)} {PM} {np.round(std, 2)}, min: {np.round(metric_min, 2)}, max: {np.round(metric_max, 2)}')

    return metric_list


def calc_mean_var_cm(parent_dir, test_dir, cm_name, cm_save_name=None):
    """
    Search for directories with the pattern "parent_dir/<any_num_of_subdirs>/run<number>/test_dir
    containing a 'run_output.json' file and calculate mean and std of confusion matrices across these runs.
    The resulting matrices are saved as figures to the parent dir.
    """
    cm_list = []
    test_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*/' + test_dir, x[0]) is not None and 'run_output.json' in x[2],
               os.walk(parent_dir)))
    dirs = '\n'.join(list(map(lambda x: x[0], test_dirs)))
    print(f'Calculating mean and std for test runs: {dirs}\n')
    for d in test_dirs:
        with open(os.sep.join([d[0], 'run_output.json'])) as f:
            res = json.load(f)['results']
            cm_list.append(np.array(res[cm_name]))

    cm = np.stack(cm_list, axis=-1)
    cm_mu, cm_std = np.round(np.mean(cm, axis=-1), 2), np.round(np.std(cm, axis=-1), 2)
    fig = general_cm_to_fig(cm_mu, cm_std)
    if cm_save_name is None:
        cm_save_name = cm_name
    fig.savefig(f'{os.path.sep.join([parent_dir, cm_save_name])}.png')


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


def calc_steps_params(dl_train, eval_every_x_epoch=1.0, warm_up_epochs=0, irm_epochs=0):
    batches_per_epoch = len(dl_train[0])
    batches_per_step = np.ceil(batches_per_epoch * eval_every_x_epoch)
    steps_per_epoch = np.ceil(batches_per_epoch / batches_per_step)
    warm_up_steps = np.ceil(warm_up_epochs * steps_per_epoch)
    irm_steps = np.ceil(irm_epochs * steps_per_epoch)

    return int(batches_per_step), int(warm_up_steps), int(irm_steps)


def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

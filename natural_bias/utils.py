from typing import NamedTuple, List
from collections import defaultdict, Counter
import subprocess
import numpy as np
import os
import re
import json
import contextlib
import torch
from plotting_utils import cm_to_fig
from data_utils import datafiles_config
from sklearn.metrics import classification_report
import pandas as pd


PM = u'\u00b1'


class BatchResult(NamedTuple):
    """
    Represents the result of training for a single batch. Error, penalty and num_correct are returned per env (as list
    of length num_envs)
    :param error: ERM term
    :param penalty: IRM term (regularizer term)
    :param loss: error + lambda * penalty
    :param num_correct: number of correct predictions (for accuracy calculations)
    :param pred: list of predicted classes (for confusion matrix)
    :param gt: list of gt classes (for confusion matrix)
    :param pred_prob: list of lists - for each sample, a list of the the model predictions (probabilities).
    """
    loss: float
    error: List[float]
    penalty: List[float]
    num_correct: List[int]
    pred: List[int]
    gt: List[int]
    pred_prob: List[List[float]] = None


class EpochResult(NamedTuple):
    """
    Represents the result of training for a single epoch.
    """
    losses: List[float]
    errors: List[float]
    penalties: List[float]
    errors_per_env: List[List[float]]
    penalties_per_env: List[List[float]]
    accuracy: float
    y_true: List[int]
    y_pred: List[int]
    accuracies_per_env: List[List[float]]
    cm: np.ndarray
    pred_prob: List[List[float]] = None


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
    f1_macro: float
    classification_report: dict
    pred_prob: List[List[float]] = None


def tensorboard_add_result(writer, result, label_map_func, step, set=''):
    writer.add_scalar(f'Loss/{set}', np.array(result.losses).mean(), step)
    writer.add_scalar(f'Error/{set}', np.array(result.errors).mean(), step)
    for i, error_per_env in enumerate(result.errors_per_env):
        writer.add_scalar(f'Error environemnt {i}/{set}', np.array(error_per_env).mean(), step)
    writer.add_scalar(f'Penalty/{set}', np.array(result.penalties).mean(), step)
    for i, penalty_per_env in enumerate(result.penalties_per_env):
        writer.add_scalar(f'Penalty environemnt {i}/{set}', np.array(penalty_per_env).mean(), step)
    writer.add_scalar(f'Accuracy/{set}', np.array(result.accuracy).mean(), step)
    for i, acc_per_env in enumerate(result.accuracies_per_env):
        writer.add_scalar(f'Accuracy environemnt {i}/{set}', np.array(acc_per_env).mean(), step)

    fig = cm_to_fig(result.cm, label_map_func)
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


def calc_mean_var_for_test(parent_dir, test_dir, filter_dir=None, topk=1, filter_key='bias unaligned split', filter_metric='accuracy'):
    """
    Print mean and variances for metrics.
    Assumes following directory tree structure: parent_dir / run[0-9] / test_dir / run_output.json
    reads all run_output.json files from the runs and print mean and std for each metric.
    :param parent_dir: parent directory as specified in tree structure
    :param test_dir: test directory as specified in tree structure
    :return: None
    """
    if filter_dir is not None:
        filter_dict = {}
        # read all results from filter dirs
        filter_dirs = list(
            filter(lambda x: re.match('.*run[0-9]+.*/' + filter_dir, x[0]) is not None and 'run_output.json' in x[2],
                   os.walk(parent_dir)))
        for d in filter_dirs:
            with open(os.sep.join([d[0], 'run_output.json'])) as f:
                res = json.load(f)['results']
            filter_dict[d[0]] = res[filter_key][filter_metric]
        # get topk results with their appropriate dir
        filter_dict = {key: filter_dict[key] for key in sorted(filter_dict, key=filter_dict.get, reverse=True)[:topk]}
        dirs = list(map(os.path.dirname, list(filter_dict.keys())))
    else:
        dirs = list(
            filter(lambda x: re.match('.*run[0-9]+.*/' + test_dir, x[0]) is not None and 'run_output.json' in x[2],
                   os.walk(parent_dir)))
        dirs = list(map(lambda x: os.path.dirname(x[0]), dirs))

    ############################
    acc_dict = defaultdict(lambda: [])
    f1_macro_dict = defaultdict(lambda: [])
    report_dict = defaultdict(lambda: [])
    test_dirs = list(map(lambda x: x + '/' + test_dir, dirs))
    dirs = '\n'.join(test_dirs)
    print(f'Calculating mean and std for test runs: \n{dirs}\n')
    # create dictionaries with list of metrics. Assume structure of run_output.json is a dictionary of some
    # subsets/splits/etc. For each such subset/split read accuracy, f1_macro, classification_report and store as a list
    # (of len of number of runs in parent directory) under the appropriate key in a designated dictionary
    for d in test_dirs:
        with open(os.sep.join([d, 'run_output.json'])) as f:
            res = json.load(f)['results']
            for k in res.keys():
                if 'accuracy' in res[k].keys():
                    acc_dict[k].append(res[k]['accuracy'])
                if 'f1_macro' in res[k].keys():
                    f1_macro_dict[k].append(res[k]['f1_macro'])
                if 'classification_report' in res[k].keys():
                    report_dict[k].append(res[k]['classification_report'])

    # for each key in a designated dictionary, calculate mean and variance over list of metrics
    print('Accuracy:\n')
    for k in acc_dict.keys():
        mu, std = np.mean(acc_dict[k]), np.std(acc_dict[k])
        acc_min, acc_max = np.min(acc_dict[k]), np.max(acc_dict[k])
        print(f'{k} - mean {PM} std: {np.round(mu,2)} {PM} {np.round(std,2)}, min: {np.round(acc_min, 2)}, max: {np.round(acc_max, 2)}\n')
    print('F1 macro score:\n')
    for k in f1_macro_dict.keys():
        f1 = np.array(f1_macro_dict[k]) * 100
        mu, std = np.mean(f1), np.std(f1)
        f1_min, f1_max = np.min(f1), np.max(f1)
        print(f'{k} - mean {PM} std: {np.round(mu,2)} {PM} {np.round(std,2)}, min: {np.round(f1_min, 2)}, max: {np.round(f1_max, 2)}\n')
    print('Classification report:')
    for k in report_dict.keys():
        df_list = [pd.DataFrame.from_dict(report_dict[k][i]) for i in range(len(report_dict[k]))]
        for df in df_list:
            for ind in ['precision', 'recall', 'f1-score']:
                df.loc[ind, :] *= 100
            df.drop(columns=['accuracy', 'weighted avg'], axis='columns', inplace=True)
        reports = np.stack([df.to_numpy() for df in df_list], axis=-1)
        mu, std = np.round(np.mean(reports, axis=-1), 2).tolist(), np.round(np.std(reports, axis=-1), 2).tolist()
        final_rep = [[str(x) + PM + str(y) for x, y in zip(mu_sublist, std_sublist)] for mu_sublist, std_sublist in zip(mu, std)]
        print(f'\n{k} - mean {PM} std:\n')
        print(pd.DataFrame.from_records(final_rep, columns=df_list[0].columns, index=df_list[0].index))
        min_final_rep, max_final_rep = np.round(np.min(reports, axis=-1), 2).tolist(), np.round(np.max(reports, axis=-1), 2).tolist()
        print(f'\n{k} - min:\n')
        print(pd.DataFrame.from_records(min_final_rep, columns=df_list[0].columns, index=df_list[0].index))
        print(f'\n{k} - max:\n')
        print(pd.DataFrame.from_records(max_final_rep, columns=df_list[0].columns, index=df_list[0].index))


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


def flatten_list(x):
    if isinstance(x[0], list):
        return [item for sublist in x for item in sublist]
    else:
        return x


def calc_steps_params(dl_train, eval_every_x_epoch=1.0, warm_up_epochs=0, irm_epochs=0):
    batches_per_epoch = len(dl_train[0])
    batches_per_step = np.ceil(batches_per_epoch * eval_every_x_epoch)
    steps_per_epoch = np.ceil(batches_per_epoch / batches_per_step)
    warm_up_steps = np.ceil(warm_up_epochs * steps_per_epoch)
    irm_steps = np.ceil(irm_epochs * steps_per_epoch)

    return int(batches_per_step), int(warm_up_steps), int(irm_steps)


def generate_classification_report_on_hans(hans_path, hans_pred_path):
    label_str_to_int = datafiles_config['HANS']['label_str_to_int']
    label_int_to_str = datafiles_config['HANS']['label_int_to_str']
    num_labels = datafiles_config['HANS']['NUM_LABELS']

    hans_gt_required_fields = ['pairID', 'gold_label', 'heuristic']
    hans_gt_existing_fields = datafiles_config['HANS']['fields']
    hans_gt_required_field_indices = [hans_gt_existing_fields.index(field) for field in hans_gt_required_fields]
    with open(hans_path, 'r') as f:
        hans_gt_lines = f.readlines()

    hans_gt_samples = []
    for line in hans_gt_lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        samp = tuple([samp[field_idx] for field_idx in hans_gt_required_field_indices])
        hans_gt_samples.append(samp)

    with open(hans_pred_path, 'r') as f:
        hans_pred_lines = f.readlines()

    hans_pred_samples = []
    for line in hans_pred_lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        hans_pred_samples.append(samp)

    assert list(zip(*hans_gt_samples))[0] == list(zip(*hans_pred_samples))[0], 'Different samples order'

    heuristic_list = ['lexical_overlap', 'subsequence', 'constituent']
    report_dict = {}
    for heuristic in heuristic_list:
        # list of tuples of (gt_label, predicted_label) with labels as strings ('entailment'/'non-entailment')
        samples = [[gt_samp[1], pred_samp[1]]
                   for gt_samp, pred_samp in zip(hans_gt_samples, hans_pred_samples) if gt_samp[-1] == heuristic]
        samples = [[label_str_to_int(x[0]), label_str_to_int(x[1])] for x in samples]
        y_true, y_pred = list(zip(*samples))
        report_dict[heuristic] = {'classification_report': classification_report(y_true, y_pred,
                                                                                 labels=[i for i in range(num_labels)],
                                                       target_names=[label_int_to_str(i) for i in range(num_labels)],
                                                                                 output_dict=True)}

    return report_dict





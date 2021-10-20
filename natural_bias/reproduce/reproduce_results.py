from main import test_on_splits, analyze_splits_difficulty
from utils import calc_mean_var_for_test
from data_utils import datasets_config
import re
import json
import pandas as pd
import os, shutil
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

PM = u'\u00b1'
HYPOTHESIS_BIAS_MAIN_EXP_DIR = '../models/hypothesis_bias/snli/main_exp'
HYPOTHESIS_BIAS_BIAS_PREVALENCE_DIR = '../models/hypothesis_bias/snli/bias_prevalence_analysis'
HYPOTHESIS_BIAS_BIAS_STRENGTH_DIR = '../models/hypothesis_bias/snli/bias_strength_analysis'
HYPOTHESIS_BIAS_DATA_SIZE_DIR = '../models/hypothesis_bias/snli/data_size_analysis'
OVERLAP_BIAS_MAIN_EXP_DIR = '../models/overlap_bias/mnli/binary/main_exp'
OVERLAP_BIAS_BIAS_PREVALENCE_DIR = '../models/overlap_bias/mnli/binary/bias_prevalence_analysis'
OVERLAP_BIAS_BIAS_STRENGTH_DIR = '../models/overlap_bias/mnli/binary/bias_strength_analysis'
OVERLAP_BIAS_DATA_SIZE_DIR = '../models/overlap_bias/mnli/binary/data_size_analysis'


def read_dirs(exp_dir, dataset='SNLI'):
    if dataset == 'SNLI':
        test_dir_list = ['test', 'val_', '_val', 'hard']
        target_test_dir = 'test_splits'
        test_file_path = 'data/SNLI/snli_1.0_test.txt'
    else:
        test_dir_list = ['dev_matched', 'dev_mismatched', 'hans', 'heuristics']
        target_test_dir = 'dev_mismatched_splits'
        test_file_path = 'data/MNLI_Binary/multinli_1.0_dev_mismatched.txt'

    # uncomment below when running on server
    # erm_model_dirs = list(
    #     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    #                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    #            os.walk(os.sep.join([exp_dir, 'erm']))))
    # erm_model_dirs = list(map(lambda x: x[0], erm_model_dirs))
    # irm_model_dirs = list(
    #     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    #                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    #            os.walk(os.sep.join([exp_dir, 'irm']))))
    # irm_model_dirs = list(map(lambda x: x[0], irm_model_dirs))

    erm_model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and 'run_output.json' in x[2] and all([i not in x[0] for i in test_dir_list]),
               os.walk(os.sep.join([exp_dir, 'erm']))))
    erm_model_dirs = list(map(lambda x: x[0], erm_model_dirs))

    irm_model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and 'run_output.json' in x[2] and all([i not in x[0] for i in test_dir_list]),
               os.walk(os.sep.join([exp_dir, 'irm']))))
    irm_model_dirs = list(map(lambda x: x[0], irm_model_dirs))

    for directories in [erm_model_dirs, irm_model_dirs]:
        for d in directories:
            if not os.path.isfile(os.sep.join([d, target_test_dir, 'run_output.json'])):
                test_on_splits(test_file_path, test_dir=d,
                               out_dir=os.sep.join([d, target_test_dir]), subset='test')

    return erm_model_dirs, irm_model_dirs


def add_jitter(mean, std, max_val, mean_val, num_samples, rng):
    jitter = rng.normal(mean, std, 1000)
    jitter = jitter[np.logical_and(jitter < max_val, jitter > mean_val)]
    assert len(jitter) >= num_samples, 'Not enough jitter'
    jitter = jitter.tolist()[:num_samples]

    return jitter


if __name__ == "__main__":

    ########################################### Plots ###########################

    # <editor-fold desc="Hypothesis bias - bias prevalence exp">
    # exp3 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias prevalence values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_acc_vals = []
    irm_acc_vals = []
    subsets_list, ratios_list = [], []

    for i in range(1, 6):  # 5 ratios tested for hypothesis bias - [0.6, 0.7, 0.8, 0.9, 1.0]
        # Prepare all directories
        exp_dir = HYPOTHESIS_BIAS_BIAS_PREVALENCE_DIR + f'/ratio{i}'
        ratio = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='SNLI')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                ratio.append(f_config['biased_samples_ratio'])

        assert len(set(ratio)) == 1, 'Different ratios in same directory'
        ratio = ratio[0]
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        erm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_splits', verbose=False)
        irm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_splits', verbose=False)

        assert list(sorted(erm_acc_dict)) == list(sorted(irm_acc_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            ratios_list.append(ratio)
            erm_acc_vals.append(erm_acc_dict[k]['vals'])
            irm_acc_vals.append(irm_acc_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'biased samples ratio': ratios_list, 'accuracy': erm_acc_vals})
    df_erm = df_erm.explode('accuracy')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'biased samples ratio': ratios_list, 'accuracy': irm_acc_vals})
    df_irm = df_irm.explode('accuracy')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.01, 0.005, 0.02, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.01, 0.005, 0.02, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    biased_samples_ratios = sorted(list(set(df['biased samples ratio'])))
    df.loc[df['model'] == 'ERM', 'biased samples ratio'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'biased samples ratio'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='biased samples ratio', y='accuracy', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(biased_samples_ratios)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    ax.set_xlabel('bias prevalence', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower right')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('hypothesis_bias_prevalence_exp_bias_misaligned_subset.png')
    # </editor-fold>

    # <editor-fold desc="overlap bias - bias prevalence exp">
    # exp3 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias prevalence values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_f1_vals = []
    irm_f1_vals = []
    subsets_list, ratios_list = [], []

    for i in range(1, 5):
        # prepare all directories
        exp_dir = OVERLAP_BIAS_BIAS_PREVALENCE_DIR + f'/ratio{i}'
        ratio = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='MNLI_Binary')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                ratio.append(f_config['biased_samples_ratio'])
        assert len(set(ratio)) == 1, 'Different ratios in same directory'
        ratio = ratio[0]
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        _, erm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'dev_mismatched_splits', verbose=False)
        _, irm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'dev_mismatched_splits', verbose=False)

        assert list(sorted(erm_f1_dict)) == list(sorted(irm_f1_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            ratios_list.append(ratio)
            erm_f1_vals.append(erm_f1_dict[k]['vals'])
            irm_f1_vals.append(irm_f1_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'biased samples ratio': ratios_list, 'f1 macro': erm_f1_vals})
    df_erm = df_erm.explode('f1 macro')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'biased samples ratio': ratios_list, 'f1 macro': irm_f1_vals})
    df_irm = df_irm.explode('f1 macro')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.01, 0.005, 0.02, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.01, 0.005, 0.02, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    biased_samples_ratios = sorted(list(set(df['biased samples ratio'])))
    df.loc[df['model'] == 'ERM', 'biased samples ratio'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'biased samples ratio'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='biased samples ratio', y='f1 macro', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(biased_samples_ratios)
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('F1 macro', fontsize=28, labelpad=14.0)
    ax.set_xlabel('bias prevalence', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower left')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('overlap_bias_prevalence_exp_bias_misaligned_subset.png')
    # </editor-fold>

    # <editor-fold desc="Hypothesis bias - bias strength exp">
    # exp5 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_acc_vals = []
    irm_acc_vals = []
    subsets_list, prob_list = [], []

    for i in range(1, 5):  # 4 strengths tested for hypothesis bias - [(0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0)]
        # Prepare all directories
        exp_dir = HYPOTHESIS_BIAS_BIAS_STRENGTH_DIR + f'/prob{i}'
        probs = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='SNLI')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                probs.append(str(tuple(f_config['train_env_prob'])))
        assert len(set(probs)) == 1, 'Different ratios in same directory'
        probs = probs[0]
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        erm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_splits', verbose=False)
        irm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_splits', verbose=False)

        assert list(sorted(erm_acc_dict)) == list(sorted(irm_acc_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            prob_list.append(probs)
            erm_acc_vals.append(erm_acc_dict[k]['vals'])
            irm_acc_vals.append(irm_acc_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    prob_str_to_int = dict([(y, x+1) for x, y in enumerate(sorted(set(prob_list)))])  # map env string to integer
    prob_list_ints = list(map(lambda x: prob_str_to_int[x], prob_list))
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'environments probabilities': prob_list_ints, 'accuracy': erm_acc_vals})
    df_erm = df_erm.explode('accuracy')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'environments probabilities': prob_list_ints, 'accuracy': irm_acc_vals})
    df_irm = df_irm.explode('accuracy')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    df.loc[df['model'] == 'ERM', 'environments probabilities'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'environments probabilities'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='environments probabilities', y='accuracy', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(prob_list_ints)
    ax.set_xticklabels(prob_list, fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    ax.set_xlabel(r'$(p_1,p_2)$', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower right')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('hypothesis_bias_strength_exp_bias_misaligned_subset.png')
    # </editor-fold>

    # <editor-fold desc="overlap bias - bias strength exp">
    # exp5 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_f1_vals = []
    irm_f1_vals = []
    subsets_list, prob_list = [], []

    for i in range(1, 5):  # 4 strengths tested for hypothesis bias - [(0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0)]
        # prepare all directories
        exp_dir = OVERLAP_BIAS_BIAS_STRENGTH_DIR + f'/prob{i}'
        probs = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='MNLI_Binary')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                probs.append(str(tuple(f_config['train_env_prob'])))
        assert len(set(probs)) == 1, 'Different ratios in same directory'
        probs = probs[0]
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        _, erm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'dev_mismatched_splits', verbose=False)
        _, irm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'dev_mismatched_splits', verbose=False)

        assert list(sorted(erm_f1_dict)) == list(sorted(irm_f1_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            prob_list.append(probs)
            erm_f1_vals.append(erm_f1_dict[k]['vals'])
            irm_f1_vals.append(irm_f1_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    prob_str_to_int = dict([(y, x+1) for x, y in enumerate(sorted(set(prob_list)))])  # map env string to integer
    prob_list_ints = list(map(lambda x: prob_str_to_int[x], prob_list))
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'environments probabilities': prob_list_ints, 'f1 macro': erm_f1_vals})
    df_erm = df_erm.explode('f1 macro')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'environments probabilities': prob_list_ints, 'f1 macro': irm_f1_vals})
    df_irm = df_irm.explode('f1 macro')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    df.loc[df['model'] == 'ERM', 'environments probabilities'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'environments probabilities'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='environments probabilities', y='f1 macro', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(prob_list_ints)
    ax.set_xticklabels(prob_list, fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('F1 macro', fontsize=28, labelpad=14.0)
    ax.set_xlabel(r'$(p_1,p_2)$', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower left')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('overlap_bias_strength_exp_bias_misaligned_subset.png')
    # </editor-fold>

    # <editor-fold desc="Overlap bias - bias dataset size, fixed val size exp - scatterplot">
    # exp3 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias prevalence values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_f1_vals = []
    irm_f1_vals = []
    subsets_list, dataset_sizes_int = [], []
    dataset_sizes_str = ['(25k, 4.5k)', '(50k, 4.5k)', '(100k, 4.5k)', '(150k, 4.5k)']
    size_dirs = ['size4', 'size3', 'size2', 'size1']

    for i, size_d in enumerate(size_dirs):
        # prepare all directories
        exp_dir = OVERLAP_BIAS_DATA_SIZE_DIR + '/' + size_d
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='MNLI_Binary')

        # <editor-fold desc="read data">
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        _, erm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'dev_mismatched_splits', verbose=False)
        _, irm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'dev_mismatched_splits', verbose=False)

        assert list(sorted(erm_f1_dict)) == list(sorted(irm_f1_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            dataset_sizes_int.append(i)
            erm_f1_vals.append(erm_f1_dict[k]['vals'])
            irm_f1_vals.append(irm_f1_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'dataset size': dataset_sizes_int, 'f1 macro': erm_f1_vals})
    df_erm = df_erm.explode('f1 macro')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'dataset size': dataset_sizes_int, 'f1 macro': irm_f1_vals})
    df_irm = df_irm.explode('f1 macro')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    dataset_sizes_int = sorted(list(set(dataset_sizes_int)))
    df.loc[df['model'] == 'ERM', 'dataset size'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'dataset size'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='dataset size', y='f1 macro', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(dataset_sizes_int)
    ax.set_xticklabels(dataset_sizes_str, fontsize=24, rotation=45.0)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('F1 macro', fontsize=28, labelpad=14.0)
    ax.set_xlabel('Dataset size', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower left')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('overlap_bias_dataset_size_fixed_val_exp_bias_misaligned_subset.png')
    # </editor-fold>

    # <editor-fold desc="Hypothesis bias - bias dataset size, fixed val size exp - scatterplot">
    # exp3 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias prevalence values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    subsets = ['unbiased split', 'bias aligned split', 'bias unaligned split']
    subsets = sorted(subsets)
    erm_acc_vals = []
    irm_acc_vals = []
    subsets_list, dataset_sizes_int = [], []
    dataset_sizes_str = ['(25k, 4k)', '(50k, 4k)', '(100k, 4k)', '(150k, 4k)', '(200k, 4k)', '(250k, 4k)']
    size_dirs = ['size6', 'size5', 'size4', 'size3', 'size2', 'size1']

    for i, size_d in enumerate(size_dirs):
        # prepare all directories
        exp_dir = HYPOTHESIS_BIAS_DATA_SIZE_DIR + '/' + size_d
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, dataset='SNLI')

        # <editor-fold desc="read data">
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        erm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_splits', verbose=False)
        irm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_splits', verbose=False)

        assert list(sorted(erm_acc_dict)) == list(sorted(irm_acc_dict)) == subsets, 'Different subsets'
        for k in subsets:
            subsets_list.append(k)
            dataset_sizes_int.append(i)
            erm_acc_vals.append(erm_acc_dict[k]['vals'])
            irm_acc_vals.append(irm_acc_dict[k]['vals'])
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    df_erm = pd.DataFrame.from_dict({'subset': subsets_list, 'dataset size': dataset_sizes_int, 'accuracy': erm_acc_vals})
    df_erm = df_erm.explode('accuracy')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict({'subset': subsets_list, 'dataset size': dataset_sizes_int, 'accuracy': irm_acc_vals})
    df_irm = df_irm.explode('accuracy')
    df_irm['model'] = 'IRM'
    df = pd.concat([df_erm, df_irm])

    rng = np.random.RandomState(42)
    erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)

    dataset_sizes_int = sorted(list(set(dataset_sizes_int)))
    df.loc[df['model'] == 'ERM', 'dataset size'] -= erm_jitter
    df.loc[df['model'] == 'IRM', 'dataset size'] += irm_jitter
    # </editor-fold>

    # <editor-fold desc="plot">
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    sns.scatterplot(x='dataset size', y='accuracy', hue='model', palette=['royalblue', 'darkorange'],
                    data=df.loc[df['subset'] == 'bias unaligned split'],
                    style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(dataset_sizes_int)
    ax.set_xticklabels(dataset_sizes_str, fontsize=24, rotation=45.0)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    ax.set_xlabel('Dataset size', fontsize=28, labelpad=14.0)
    ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    ax.legend(fontsize=24, loc='lower right')
    leg = ax.get_legend()
    leg.legendHandles[0].set_color('royalblue')
    leg.legendHandles[0].set_sizes([200])
    leg.legendHandles[1].set_color('darkorange')
    leg.legendHandles[1].set_sizes([200])
    # </editor-fold>

    fig.tight_layout()
    fig.savefig('hypothesis_bias_dataset_size_fixed_val_exp_bias_misaligned_subset.png')
    # </editor-fold>

    ######## Table 4 ##############
    exp_dir = HYPOTHESIS_BIAS_MAIN_EXP_DIR
    col_dict = {'model': ['ERM', 'IRM']}
    erm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_splits', verbose=False)
    irm_acc_dict, _, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_splits', verbose=False)
    for col in ['unbiased split', 'bias aligned split', 'bias unaligned split']:
        col_dict[col] = [f'{np.round(erm_acc_dict[col]["mean"], 2)} {PM} {np.round(erm_acc_dict[col]["std"], 2)}',
                         f'{np.round(irm_acc_dict[col]["mean"], 2)} {PM} {np.round(irm_acc_dict[col]["std"], 2)}']

    df = pd.DataFrame(col_dict, index=['mean', 'mean'])
    print('Table 4:\n')
    print(df)

    ######## Table 5 ##############
    exp_dir = OVERLAP_BIAS_MAIN_EXP_DIR
    col_dict = {'model': ['ERM', 'IRM']}
    _, erm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'dev_mismatched_splits', verbose=False)
    _, irm_f1_dict, _ = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'dev_mismatched_splits', verbose=False)
    for col in ['unbiased split', 'bias aligned split', 'bias unaligned split']:
        col_dict[col] = [f'{np.round(erm_f1_dict[col]["mean"], 2)} {PM} {np.round(erm_f1_dict[col]["std"], 2)}',
                         f'{np.round(irm_f1_dict[col]["mean"], 2)} {PM} {np.round(irm_f1_dict[col]["std"], 2)}']

    df = pd.DataFrame(col_dict, index=['mean', 'mean'])
    print('Table 5:\n')
    print(df)
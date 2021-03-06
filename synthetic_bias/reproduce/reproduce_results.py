from main import test_irm
from utils import calc_mean_var_for_test
from settings import VOCAB_BIAS, NUM_LABELS, labels_int_to_string
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

PM = u'\u00b1'
MAIN_EXP_DIR = '../models/main_exp'
BIAS_PREVALENCE_DIR = '../models/bias_prevalence_analysis'
BIAS_STRENGTH_DIR = '../models/bias_strength_analysis'
DATA_SIZE_DIR = '../models/data_size_analysis'

def read_dirs(exp_dir, model='erm'):
    # uncomment below when running on server
    # model_dirs = list(
    #     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    #                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    #            os.walk(os.sep.join([exp_dir, model]))))
    # model_dirs = list(map(lambda x: x[0], model_dirs))

    model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and 'run_output.json' in x[2] and all([i not in x[0] for i in ['test', 'val_', '_val', 'hard']]),
               os.walk(os.sep.join([exp_dir, model]))))
    model_dirs = list(map(lambda x: x[0], model_dirs))


    return model_dirs


def add_jitter(mean, std, max_val, mean_val, num_samples, rng):
    jitter = rng.normal(mean, std, 1000)
    jitter = jitter[np.logical_and(jitter < max_val, jitter > mean_val)]
    assert len(jitter) >= num_samples, 'Not enough jitter'
    jitter = jitter.tolist()[:num_samples]

    return jitter


if __name__ == "__main__":
    # <editor-fold desc="biased sample ratio reproduce plot ">
    erm_acc_mu, irm_acc_mu, erm_acc_std, irm_acc_std = [], [], [], []
    biased_samples_ratio = []

    for i in range(1, 14):
        exp_dir = BIAS_PREVALENCE_DIR + f'/ratio{i}'
        ratio = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')

        # test ERM on SNLI test file with p=0.33
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                ratio.append(f_config['biased_samples_ratio'])
                if not os.path.isfile(os.sep.join([d, 'test_ood', 'run_output.json'])):
                    test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))

        assert len(set(ratio)) == 1, 'Different ratios in same directory'
        erm_acc = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood', verbose=False)
        erm_acc_mu.append(np.mean(erm_acc))
        erm_acc_std.append(np.std(erm_acc))
        irm_acc = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood', verbose=False)
        irm_acc_mu.append(np.mean(irm_acc))
        irm_acc_std.append(np.std(irm_acc))
        biased_samples_ratio.append(ratio[0])

    temp_sorted = list(sorted(zip(biased_samples_ratio, erm_acc_mu, erm_acc_std, irm_acc_mu, irm_acc_std), key=lambda x: x[0]))
    biased_samples_ratio, erm_acc_mu, erm_acc_std, irm_acc_mu, irm_acc_std = [list(x) for x in list(zip(*temp_sorted))]
    fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)

    test_metric_mean = np.round(erm_acc_mu, 2)
    yerr_plus = np.round(erm_acc_std, 2)
    yerr_minus = np.round(erm_acc_std, 2)

    ax.plot(biased_samples_ratio, test_metric_mean, marker='o', markersize=12, label='ERM', color='royalblue')
    ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)

    test_metric_mean = np.round(irm_acc_mu, 2)
    yerr_plus = np.round(irm_acc_std, 2)
    yerr_minus = np.round(irm_acc_std, 2)

    ax.plot(biased_samples_ratio, test_metric_mean, marker='x', markersize=12, label='IRM', color='darkorange')
    ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)

    ax.set_ylim([72.0, 88.0])
    ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    ax.set_xlabel('bias prevalence', fontsize=28, labelpad=14.0)

    ax.legend(fontsize=24, loc="lower right")
    fig.tight_layout()
    fig.savefig('synthetic_bias_prevalence_exp.png')
    # </editor-fold>

    # <editor-fold desc="synthetic bias - bias strength exp">
    # exp4 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    erm_acc_vals = []
    irm_acc_vals = []
    prob_list = []

    for i in range(1, 5):  # 4 strengths tested for hypothesis bias - [(0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0)]
        # Prepare all directories
        exp_dir = BIAS_STRENGTH_DIR + f'/prob{i}'
        probs = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                probs.append(str(tuple(f_config['train_env_prob'])))
                if not os.path.isfile(os.sep.join([d, 'test_ood', 'run_output.json'])):
                    test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))

        assert len(set(probs)) == 1, 'Different ratios in same directory'
        probs = probs[0]
        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        erm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood', verbose=False))
        irm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood', verbose=False))

        prob_list.append(probs)
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    prob_str_to_int = dict([(y, x + 1) for x, y in enumerate(sorted(set(prob_list)))])  # map env string to integer
    prob_list_ints = list(map(lambda x: prob_str_to_int[x], prob_list))
    df_erm = pd.DataFrame.from_dict(
        {'environments probabilities': prob_list_ints, 'accuracy': erm_acc_vals})
    df_erm = df_erm.explode('accuracy')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict(
        {'environments probabilities': prob_list_ints, 'accuracy': irm_acc_vals})
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
                    data=df, style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

    ax.set_xticks(prob_list_ints)
    ax.set_xticklabels(prob_list, fontsize=24)
    ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
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
    fig.savefig('synthetic_bias_strength_exp.png')
    # </editor-fold>

    # <editor-fold desc="synthetic bias - dataset size exp">
    # exp5 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # for each key - list of mean/std (according to number of dataset sizes)
    erm_acc_vals = []
    irm_acc_vals = []
    dataset_sizes_str = ['(25k, 9.8k)', '(50k, 9.8k)', '(100k, 9.8k)', '(200k, 9.8k)', '(300k, 9.8k)', '(400k, 9.8k)',
                         '(500k, 9.8k)']
    dataset_sizes_int = []
    size_dirs = ['size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7']

    for i in range(1, 8):  # 7 sizes tested for synthetic bias
        # Prepare all directories
        exp_dir = DATA_SIZE_DIR + f'/size{i}'
        probs = []
        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')

        # <editor-fold desc="read data">
        for directories in [erm_model_dirs, irm_model_dirs]:
            for d in directories:
                with open(os.sep.join([d, 'run_output.json'])) as f:
                    f_config = json.load(f)['config']
                probs.append(str(tuple(f_config['train_env_prob'])))
                if not os.path.isfile(os.sep.join([d, 'test_ood', 'run_output.json'])):
                    test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))

        # dict of dict - keys are subsets, subkeys are mean, std, min and max
        erm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood', verbose=False))
        irm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood', verbose=False))

        dataset_sizes_int.append(i)
        # </editor-fold>

    # <editor-fold desc="prepare for plotting">
    df_erm = pd.DataFrame.from_dict(
        {'dataset size': dataset_sizes_int, 'accuracy': erm_acc_vals})
    df_erm = df_erm.explode('accuracy')
    df_erm['model'] = 'ERM'
    df_irm = pd.DataFrame.from_dict(
        {'dataset size': dataset_sizes_int, 'accuracy': irm_acc_vals})
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
                    data=df, style='model', markers=['o', '^'], s=400, ax=ax, legend='full')

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
    fig.savefig('synthetic_bias_dataset_size_exp.png')
    # </editor-fold>

    exp_dir = MAIN_EXP_DIR
    erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')
    ref_model_dirs = read_dirs(exp_dir, 'ref1')

    # <editor-fold desc="Table 3 reproduction">
    # print results
    col_dict = {}

    for col, test_dir_name in zip(['p=0.8', 'p=0.33', 'p=0.0'], ['test_indist', 'test_ood1', 'test_ood2']):
        res_erm, res_irm, res_ref = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), test_dir_name, verbose=False), \
                                    calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), test_dir_name, verbose=False), \
                                    calc_mean_var_for_test(os.sep.join([exp_dir, 'ref1']), test_dir_name, verbose=False)
        mu = [np.round(np.mean(res_ref), 2), np.round(np.mean(res_erm), 2), np.round(np.mean(res_irm), 2)]
        std = [np.round(np.std(res_ref), 2), np.round(np.std(res_erm), 2), np.round(np.std(res_irm), 2)]
        col_dict[col] = [f'{x}{PM}{y}' for x, y in zip(mu, std)]

    df = pd.DataFrame(col_dict, index=['REF', 'ERM', 'IRM'])
    print('Table 3:\n')
    print(df)
    # </editor-fold>

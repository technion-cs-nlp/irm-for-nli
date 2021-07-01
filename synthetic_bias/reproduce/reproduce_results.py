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


def read_dirs(exp_dir, model='erm'):
    # uncomment below when running on server
    # model_dirs = list(
    #     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    #                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    #            os.walk(os.sep.join([exp_dir, model]))))
    # model_dirs = list(map(lambda x: x[0], model_dirs))

    model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and 'run_output.json' in x[2] and all([i not in x[0] for i in ['test', 'val', 'hard']]),
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
    # # get dirs of exp1 - ERM, IRM and ref trained on p1=0.7 p2=0.9 with 5 seeds
    # exp_dir = '../models/exp1'
    # erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')
    # ref_model_dirs = read_dirs(exp_dir, 'ref1')
    #
    # # <editor-fold desc="Table 5 reproduction">
    # # test REF, ERM and IRM  on SNLI test file with p=0.8, p=0.33, p=0.2
    # for directories in [ref_model_dirs, erm_model_dirs, irm_model_dirs]:
    #     for d in directories:
    #         if not os.path.isfile(os.sep.join([d, 'test_indist', 'run_output.json'])):
    #             print(f'testing {os.sep.join([d, "test_indist", "run_output.json"])}')
    #             test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_indist']),
    #                      env_prob=(0.8,))
    #         if not os.path.isfile(os.sep.join([d, 'test_ood1', 'run_output.json'])):
    #             print(f'testing {os.sep.join([d, "test_ood1", "run_output.json"])}')
    #             test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood1']), env_prob=(0.33,))
    #         if not os.path.isfile(os.sep.join([d, 'test_ood2', 'run_output.json'])):
    #             print(f'testing {os.sep.join([d, "test_ood2", "run_output.json"])}')
    #             test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.2,))
    #
    # # print results
    # col_dict = {}
    #
    # for col, test_dir_name in zip(['p=0.8', 'p=0.33', 'p=0.0'], ['test_indist', 'test_ood1', 'test_ood2']):
    #     res_erm, res_irm, res_ref = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), test_dir_name, verbose=False), \
    #                                 calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), test_dir_name, verbose=False), \
    #                                 calc_mean_var_for_test(os.sep.join([exp_dir, 'ref1']), test_dir_name, verbose=False)
    #     mu = [np.round(np.mean(res_ref), 2), np.round(np.mean(res_erm), 2), np.round(np.mean(res_irm), 2)]
    #     std = [np.round(np.std(res_ref), 2), np.round(np.std(res_erm), 2), np.round(np.std(res_irm), 2)]
    #     col_dict[col] = [f'{x}{PM}{y}' for x, y in zip(mu, std)]
    #
    # df = pd.DataFrame(col_dict, index=['REF', 'ERM', 'IRM'])
    # print('Table 5:\n')
    # print(df)
    # # </editor-fold>
    #
    #
    # # <editor-fold desc="Table 13 reproduction">
    # # test ERM on SNLI test file with contradiction, neutral and entailment bias
    # for lbl_int in range(NUM_LABELS):
    #     lbl_str = labels_int_to_string(lbl_int)
    #     for directories in [ref_model_dirs, erm_model_dirs, irm_model_dirs]:
    #         for d in directories:
    #             if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
    #                          env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
    #             if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
    #                          env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
    #             if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
    #                          env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
    #
    # # print results
    # col_dict = {}
    #
    # for lbl_int in range(NUM_LABELS):
    #     lbl_str = labels_int_to_string(lbl_int)
    #     res_erm, res_irm, res_ref = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_' + lbl_str, verbose=False), \
    #                                 calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_' + lbl_str, verbose=False), \
    #                                 calc_mean_var_for_test(os.sep.join([exp_dir, 'ref1']), 'test_' + lbl_str, verbose=False)
    #     mu = [np.round(np.mean(res_ref), 2), np.round(np.mean(res_erm), 2), np.round(np.mean(res_irm), 2)]
    #     std = [np.round(np.std(res_ref), 2), np.round(np.std(res_erm), 2), np.round(np.std(res_irm), 2)]
    #     col_dict[lbl_str] = [f'{x}{PM}{y}' for x, y in zip(mu, std)]
    #
    # df = pd.DataFrame(col_dict, index=['REF', 'ERM', 'IRM'])
    # print('Table 13:\n')
    # print(df)
    # # </editor-fold>
    #
    #
    # # <editor-fold desc="biased sample ratio reproduce plot ">
    # erm_acc_mu, irm_acc_mu, erm_acc_std, irm_acc_std = [], [], [], []
    # biased_samples_ratio = []
    #
    # for i in range(1, 14):
    #     exp_dir = f'../models/exp2/ratio{i}'
    #     ratio = []
    #     erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')
    #
    #     # test ERM on SNLI test file with p=0.33
    #     for directories in [erm_model_dirs, irm_model_dirs]:
    #         for d in directories:
    #             with open(os.sep.join([d, 'run_output.json'])) as f:
    #                 f_config = json.load(f)['config']
    #             ratio.append(f_config['biased_samples_ratio'])
    #             if not os.path.isfile(os.sep.join([d, 'test_ood2', 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))
    #
    #     assert len(set(ratio)) == 1, 'Different ratios in same directory'
    #     erm_acc = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood2', verbose=False)
    #     erm_acc_mu.append(np.mean(erm_acc))
    #     erm_acc_std.append(np.std(erm_acc))
    #     irm_acc = calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood2', verbose=False)
    #     irm_acc_mu.append(np.mean(irm_acc))
    #     irm_acc_std.append(np.std(irm_acc))
    #     biased_samples_ratio.append(ratio[0])
    #
    # temp_sorted = list(sorted(zip(biased_samples_ratio, erm_acc_mu, erm_acc_std, irm_acc_mu, irm_acc_std), key=lambda x: x[0]))
    # biased_samples_ratio, erm_acc_mu, erm_acc_std, irm_acc_mu, irm_acc_std = [list(x) for x in list(zip(*temp_sorted))]
    # fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    #
    # test_metric_mean = np.round(erm_acc_mu, 2)
    # yerr_plus = np.round(erm_acc_std, 2)
    # yerr_minus = np.round(erm_acc_std, 2)
    #
    # ax.plot(biased_samples_ratio, test_metric_mean, marker='o', markersize=12, label='ERM', color='royalblue')
    # ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)
    #
    # test_metric_mean = np.round(irm_acc_mu, 2)
    # yerr_plus = np.round(irm_acc_std, 2)
    # yerr_minus = np.round(irm_acc_std, 2)
    #
    # ax.plot(biased_samples_ratio, test_metric_mean, marker='x', markersize=12, label='IRM', color='darkorange')
    # ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)
    #
    # ax.set_ylim([72.0, 88.0])
    # ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=24)
    # ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    # ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    # ax.set_xlabel('bias prevalence', fontsize=28, labelpad=14.0)
    #
    # ax.legend(fontsize=24, loc="lower right")
    # fig.tight_layout()
    # fig.savefig('synthetic_bias_prevalence_exp.png')
    # # </editor-fold>
    #
    # #
    # # # <editor-fold desc="bias strength reproduce plot ">
    # # erm_acc, irm_acc = [], []
    # # prob_vector = []
    # #
    # # for i in range(1, 5):
    # #     exp_dir = f'models/exp4/prob{i}'
    # #     probs = []
    # #     erm_model_dirs = list(
    # #         filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    # #                          and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    # #                os.walk(os.sep.join([exp_dir, 'erm']))))
    # #     erm_model_dirs = list(map(lambda x: x[0], erm_model_dirs))
    # #     irm_model_dirs = list(
    # #         filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
    # #                          and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
    # #                os.walk(os.sep.join([exp_dir, 'irm']))))
    # #     irm_model_dirs = list(map(lambda x: x[0], irm_model_dirs))
    # #     # test ERM on SNLI test file with p=0.8, p=0.33, p=0.2
    # #     for directories in [erm_model_dirs, irm_model_dirs]:
    # #         for d in directories:
    # #             with open(os.sep.join([d, 'run_output.json'])) as f:
    # #                 f_config = json.load(f)['config']
    # #             probs.append(str(tuple(f_config['train_env_prob'])))
    # #             if not os.path.isfile(os.sep.join([d, 'test_ood', 'run_output.json'])):
    # #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood']), env_prob=(0.33,))
    # #
    # #     assert len(set(probs)) == 1, 'Different probabilities in same directory'
    # #     erm_acc.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood', verbose=False))
    # #     irm_acc.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood', verbose=False))
    # #     prob_vector.append(probs[0])
    # #
    # # fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=240)
    # #
    # # xticks = np.array([1, 2, 3, 4])
    # # bp_erm = ax.boxplot(erm_acc, positions=xticks-0.2, widths=[0.3]*len(xticks), patch_artist=True, showfliers=False)
    # # bp_irm = ax.boxplot(irm_acc, positions=xticks+0.2, widths=[0.3]*len(xticks), patch_artist=True, showfliers=False)
    # #
    # # ax.set_xticks(xticks)
    # # ax.set_xticklabels(prob_vector, fontsize=16)
    # # ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=16)
    # # ax.set_ylabel('accuracy', fontsize=22, labelpad=14.0)
    # # ax.set_xlabel('Environment probabilities', fontsize=22, labelpad=14.0)
    # #
    # # for artist in bp_erm["boxes"]:
    # #     artist.set(facecolor='lightblue')
    # # for artist in bp_irm["boxes"]:
    # #     artist.set(facecolor='bisque')
    # # ax.legend([bp_erm["boxes"][0], bp_irm["boxes"][0]], ['ERM', 'IRM'], loc='upper right', fontsize=18)
    # # fig.tight_layout()
    # # fig.savefig('synthetic_bias_strength_exp.png')
    # # # </editor-fold>
    #
    # #####################
    #
    # # <editor-fold desc="synthetic bias - bias strength exp">
    # # exp4 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # # for each key - list of mean/std (according to number of dataset sizes)
    # erm_acc_vals = []
    # irm_acc_vals = []
    # prob_list = []
    #
    # for i in range(1, 5):  # 4 strengths tested for hypothesis bias - [(0.5, 0.7), (0.6, 0.8), (0.7, 0.9), (0.8, 1.0)]
    #     # Prepare all directories
    #     exp_dir = f'../models/exp4/prob{i}'
    #     probs = []
    #     erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')
    #
    #     # <editor-fold desc="read data">
    #     for directories in [erm_model_dirs, irm_model_dirs]:
    #         for d in directories:
    #             with open(os.sep.join([d, 'run_output.json'])) as f:
    #                 f_config = json.load(f)['config']
    #             probs.append(str(tuple(f_config['train_env_prob'])))
    #             if not os.path.isfile(os.sep.join([d, 'test_ood2', 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))
    #
    #     assert len(set(probs)) == 1, 'Different ratios in same directory'
    #     probs = probs[0]
    #     # dict of dict - keys are subsets, subkeys are mean, std, min and max
    #     erm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood2', verbose=False))
    #     irm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood2', verbose=False))
    #
    #     prob_list.append(probs)
    #     # </editor-fold>
    #
    # # <editor-fold desc="prepare for plotting">
    # prob_str_to_int = dict([(y, x + 1) for x, y in enumerate(sorted(set(prob_list)))])  # map env string to integer
    # prob_list_ints = list(map(lambda x: prob_str_to_int[x], prob_list))
    # df_erm = pd.DataFrame.from_dict(
    #     {'environments probabilities': prob_list_ints, 'accuracy': erm_acc_vals})
    # df_erm = df_erm.explode('accuracy')
    # df_erm['model'] = 'ERM'
    # df_irm = pd.DataFrame.from_dict(
    #     {'environments probabilities': prob_list_ints, 'accuracy': irm_acc_vals})
    # df_irm = df_irm.explode('accuracy')
    # df_irm['model'] = 'IRM'
    # df = pd.concat([df_erm, df_irm])
    #
    # rng = np.random.RandomState(42)
    # erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    # irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)
    #
    # df.loc[df['model'] == 'ERM', 'environments probabilities'] -= erm_jitter
    # df.loc[df['model'] == 'IRM', 'environments probabilities'] += irm_jitter
    # # </editor-fold>
    #
    # # <editor-fold desc="plot">
    # fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    # sns.scatterplot(x='environments probabilities', y='accuracy', hue='model', palette=['royalblue', 'darkorange'],
    #                 data=df, style='model', markers=['o', '^'], s=400, ax=ax, legend='full')
    #
    # ax.set_xticks(prob_list_ints)
    # ax.set_xticklabels(prob_list, fontsize=24)
    # ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    # ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    # ax.set_xlabel(r'$(p_1,p_2)$', fontsize=28, labelpad=14.0)
    # ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    # ax.legend(fontsize=24, loc='lower left')
    # leg = ax.get_legend()
    # leg.legendHandles[0].set_color('royalblue')
    # leg.legendHandles[0].set_sizes([200])
    # leg.legendHandles[1].set_color('darkorange')
    # leg.legendHandles[1].set_sizes([200])
    # # </editor-fold>
    #
    # fig.tight_layout()
    # fig.savefig('synthetic_bias_strength_exp.png')
    # # </editor-fold>
    #
    # # <editor-fold desc="synthetic bias - dataset size exp">
    # # exp5 - train ERM and IRM on p1=0.7 p2=0.9 with changing bias strength values (also 5 seeds)
    # # for each key - list of mean/std (according to number of dataset sizes)
    # erm_acc_vals = []
    # irm_acc_vals = []
    # dataset_sizes_str = ['(25k, 9.8k)', '(50k, 9.8k)', '(100k, 9.8k)', '(200k, 9.8k)', '(300k, 9.8k)', '(400k, 9.8k)',
    #                      '(500k, 9.8k)']
    # dataset_sizes_int = []
    # size_dirs = ['size1', 'size2', 'size3', 'size4', 'size5', 'size6', 'size7']
    #
    # for i in range(1, 8):  # 7 sizes tested for synthetic bias
    #     # Prepare all directories
    #     exp_dir = f'../models/exp5/size{i}'
    #     probs = []
    #     erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, 'erm'), read_dirs(exp_dir, 'irm')
    #
    #     # <editor-fold desc="read data">
    #     for directories in [erm_model_dirs, irm_model_dirs]:
    #         for d in directories:
    #             with open(os.sep.join([d, 'run_output.json'])) as f:
    #                 f_config = json.load(f)['config']
    #             probs.append(str(tuple(f_config['train_env_prob'])))
    #             if not os.path.isfile(os.sep.join([d, 'test_ood2', 'run_output.json'])):
    #                 test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.0,))
    #
    #     # dict of dict - keys are subsets, subkeys are mean, std, min and max
    #     erm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_ood2', verbose=False))
    #     irm_acc_vals.append(calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_ood2', verbose=False))
    #
    #     dataset_sizes_int.append(i)
    #     # </editor-fold>
    #
    # # <editor-fold desc="prepare for plotting">
    # df_erm = pd.DataFrame.from_dict(
    #     {'dataset size': dataset_sizes_int, 'accuracy': erm_acc_vals})
    # df_erm = df_erm.explode('accuracy')
    # df_erm['model'] = 'ERM'
    # df_irm = pd.DataFrame.from_dict(
    #     {'dataset size': dataset_sizes_int, 'accuracy': irm_acc_vals})
    # df_irm = df_irm.explode('accuracy')
    # df_irm['model'] = 'IRM'
    # df = pd.concat([df_erm, df_irm])
    #
    # rng = np.random.RandomState(42)
    # erm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'ERM']), rng)
    # irm_jitter = add_jitter(0.1, 0.05, 0.2, 0.0, len(df.loc[df['model'] == 'IRM']), rng)
    #
    # dataset_sizes_int = sorted(list(set(dataset_sizes_int)))
    # df.loc[df['model'] == 'ERM', 'dataset size'] -= erm_jitter
    # df.loc[df['model'] == 'IRM', 'dataset size'] += irm_jitter
    # # </editor-fold>
    #
    # # <editor-fold desc="plot">
    # fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=480)
    # sns.scatterplot(x='dataset size', y='accuracy', hue='model', palette=['royalblue', 'darkorange'],
    #                 data=df, style='model', markers=['o', '^'], s=400, ax=ax, legend='full')
    #
    # ax.set_xticks(dataset_sizes_int)
    # ax.set_xticklabels(dataset_sizes_str, fontsize=24, rotation=45.0)
    # ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=24)
    # ax.set_ylabel('Accuracy', fontsize=28, labelpad=14.0)
    # ax.set_xlabel('Dataset size', fontsize=28, labelpad=14.0)
    # ax.grid(axis='both', color='k', linestyle=':', linewidth=0.5)
    # ax.legend(fontsize=24, loc='lower right')
    # leg = ax.get_legend()
    # leg.legendHandles[0].set_color('royalblue')
    # leg.legendHandles[0].set_sizes([200])
    # leg.legendHandles[1].set_color('darkorange')
    # leg.legendHandles[1].set_sizes([200])
    # # </editor-fold>
    #
    # fig.tight_layout()
    # fig.savefig('synthetic_bias_dataset_size_exp.png')
    # # </editor-fold>

    # <editor-fold desc="Synthetic bias - early stopping exp">
    # exp4 - train ERM and IRM on p1=0.7 p2=0.9 with changing dataset size (also 5 seeds)
    erm_fig, erm_axes = plt.subplots(nrows=4, ncols=3, figsize=(20.0, 16.0), dpi=240, sharex=True)
    irm_fig, irm_axes = plt.subplots(nrows=4, ncols=3, figsize=(20.0, 16.0), dpi=240, sharex=True)
    exp_dir_list = ['../models/exp1', '../models/early_stopping/exp1']
    markers = ['x', 'o']
    colors = {'run1': 'r', 'run2': 'g', 'run3': 'b', 'run4': 'y', 'run5': 'm'}

    n = 5
    for e, exp_dir in enumerate(exp_dir_list):
        # <editor-fold desc="prepare data">

        # load results to dictionaries
        erm_train_acc, erm_val_acc, erm_val_ood_acc = {}, {}, {}
        erm_train_error, erm_val_error, erm_val_ood_error = {}, {}, {}
        erm_train_penalty, erm_val_penalty, erm_val_ood_penalty = {}, {}, {}
        erm_train_loss, erm_val_loss, erm_val_ood_loss = {}, {}, {}
        irm_train_acc, irm_val_acc, irm_val_ood_acc = {}, {}, {}
        irm_train_error, irm_val_error, irm_val_ood_error = {}, {}, {}
        irm_train_penalty, irm_val_penalty, irm_val_ood_penalty = {}, {}, {}
        irm_train_loss, irm_val_loss, irm_val_ood_loss = {}, {}, {}

        erm_model_dirs, irm_model_dirs = read_dirs(exp_dir, model='erm'), read_dirs(exp_dir, model='irm')

        for d in erm_model_dirs:
            with open(os.sep.join([d, 'run_output.json'])) as f:
                res = json.load(f)['results']
            k = re.findall('run[0-9]+', d)[0]
            erm_train_acc[k], erm_val_acc[k], erm_val_ood_acc[k] = res['train_acc'], res['test_acc'], res[
                'test_acc_ood']
            erm_train_error[k], erm_val_error[k], erm_val_ood_error[k] = res['train_error'], res['test_error'], res[
                'test_error_ood']
            erm_train_penalty[k], erm_val_penalty[k], erm_val_ood_penalty[k] = res['train_penalty'], res[
                'test_penalty'], res['test_penalty_ood']
            erm_train_loss[k], erm_val_loss[k], erm_val_ood_loss[k] = res['train_loss'], res['test_loss'], res[
                'test_loss_ood']

        for d in irm_model_dirs:
            with open(os.sep.join([d, 'run_output.json'])) as f:
                res = json.load(f)['results']
            k = re.findall('run[0-9]+', d)[0]
            irm_train_acc[k], irm_val_acc[k], irm_val_ood_acc[k] = res['train_acc'], res['test_acc'], res[
                'test_acc_ood']
            irm_train_error[k], irm_val_error[k], irm_val_ood_error[k] = res['train_error'], res['test_error'], res[
                'test_error_ood']
            irm_train_penalty[k], irm_val_penalty[k], irm_val_ood_penalty[k] = res['train_penalty'], res[
                'test_penalty'], res['test_penalty_ood']
            irm_train_loss[k], irm_val_loss[k], irm_val_ood_loss[k] = res['train_loss'], res['test_loss'], res[
                'test_loss_ood']

        # </editor-fold>

        # <editor-fold desc="plot">
        # erm
        for axe, erm_subset_vals, metric in zip(erm_axes.tolist(), [[erm_train_acc, erm_val_acc, erm_val_ood_acc],
                                                                [erm_train_error, erm_val_error, erm_val_ood_error],
                                                                [erm_train_penalty, erm_val_penalty, erm_val_ood_penalty],
                                                                [erm_train_loss, erm_val_loss, erm_val_ood_loss]],
                                                ['accuracy', 'error', 'penalty', 'loss']):
            for ax, erm_vals in zip(axe, erm_subset_vals):
                    for k in erm_vals.keys():
                        ax.plot(erm_vals[k], label=k, color=colors[k], marker=markers[e], mfc='none')


        # irm
        for axe, irm_subset_vals, metric in zip(irm_axes.tolist(), [[irm_train_acc, irm_val_acc, irm_val_ood_acc],
                                                                [irm_train_error, irm_val_error, irm_val_ood_error],
                                                                [irm_train_penalty, irm_val_penalty, irm_val_ood_penalty],
                                                                [irm_train_loss, irm_val_loss, irm_val_ood_loss]],
                                                ['accuracy', 'error', 'penalty', 'loss']):
            for ax, irm_vals, subset in zip(axe, irm_subset_vals, ['train', 'val', 'val_ood']):
                    for k in irm_vals.keys():
                        ax.plot(irm_vals[k], label=k, color=colors[k], marker=markers[e], mfc='none')
        # </editor-fold>

    # <editor-fold desc="plot editing">
    # <editor-fold desc="legend">
    h, l = irm_axes[0, 0].get_legend_handles_labels()
    labels = ['custom'] + l[:n] + ['standard'] + l[n:]
    ph = [plt.plot([], marker="", ls="")[0]] * 2
    handles = [ph[0]] + h[:n] + [ph[1]] + h[n:]
    irm_axes[0, 0].legend(handles, labels, ncol=2)

    h, l = erm_axes[0, 0].get_legend_handles_labels()
    labels = ['custom'] + l[:n] + ['standard'] + l[n:]
    ph = [plt.plot([], marker="", ls="")[0]] * 2
    handles = [ph[0]] + h[:n] + [ph[1]] + h[n:]
    erm_axes[0, 0].legend(handles, labels, ncol=2)
    # </editor-fold>

    erm_axes[0, 0].set_xticklabels(np.array(erm_axes[0, 0].get_xticks()).astype(int)+1, fontsize=22)
    irm_axes[0, 0].set_xticklabels(np.array(irm_axes[0, 0].get_xticks()).astype(int)+1, fontsize=22)

    # <editor-fold desc="Col and row titles">
    # erm
    pad1, pad2 = 100, 40  # in points

    for ax, row in zip(erm_axes[:, 0], ['accuracy', 'error', 'penalty', 'loss']):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad1, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    fontsize=18, ha='left', va='center')

    for ax, col in zip(erm_axes[0], ['train', 'val', 'val_ood']):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad2),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=18, ha='center', va='baseline')

    for ax, row in zip(irm_axes[:, 0], ['accuracy', 'error', 'penalty', 'loss']):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad1, 0),
                    xycoords=ax.yaxis.label, textcoords='offset points',
                    fontsize=18, ha='left', va='center')

    for ax, col in zip(irm_axes[0], ['train', 'val', 'val_ood']):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad2),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=18, ha='center', va='baseline')
    # </editor-fold>
    # </editor-fold>

    erm_fig.savefig('hypothesis_bias_early_stopping_exp_erm.png')
    irm_fig.savefig('hypothesis_bias_early_stopping_exp_irm.png')
    # </editor-fold>


from main import test_irm
from utils import calc_mean_var_for_test
from settings import VOCAB_BIAS, NUM_LABELS, labels_int_to_string
import re
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

PM = u'\u00b1'

# get dirs of exp1 - ERM, IRM and ref trained on p1=0.7 p2=0.9 with 5 seeds
exp_dir = 'models/exp1'
erm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(os.sep.join([exp_dir, 'erm']))))
erm_model_dirs = list(map(lambda x: x[0], erm_model_dirs))
irm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(os.sep.join([exp_dir, 'irm']))))
irm_model_dirs = list(map(lambda x: x[0], irm_model_dirs))
ref_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(os.sep.join([exp_dir, 'ref1']))))
ref_model_dirs = list(map(lambda x: x[0], ref_model_dirs))

# <editor-fold desc="Table 5 reproduction">
# test REF, ERM and IRM  on SNLI test file with p=0.8, p=0.33, p=0.2
for directories in [ref_model_dirs, erm_model_dirs, irm_model_dirs]:
    for d in directories:
        if not os.path.isfile(os.sep.join([d, 'test_indist', 'run_output.json'])):
            print(f'testing {os.sep.join([d, "test_indist", "run_output.json"])}')
            test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_indist']),
                     env_prob=(0.8,))
        if not os.path.isfile(os.sep.join([d, 'test_ood1', 'run_output.json'])):
            print(f'testing {os.sep.join([d, "test_ood1", "run_output.json"])}')
            test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood1']), env_prob=(0.33,))
        if not os.path.isfile(os.sep.join([d, 'test_ood2', 'run_output.json'])):
            print(f'testing {os.sep.join([d, "test_ood2", "run_output.json"])}')
            test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.2,))

# print results
col_dict = {}

for col, test_dir_name in zip(['p=0.8', 'p=0.33', 'p=0.2'], ['test_indist', 'test_ood1', 'test_ood2']):
    res_erm, res_irm, res_ref = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), test_dir_name, verbose=False), \
                                calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), test_dir_name, verbose=False), \
                                calc_mean_var_for_test(os.sep.join([exp_dir, 'ref1']), test_dir_name, verbose=False)
    mu = [np.round(np.mean(res_ref), 2), np.round(np.mean(res_erm), 2), np.round(np.mean(res_irm), 2)]
    std = [np.round(np.std(res_ref), 2), np.round(np.std(res_erm), 2), np.round(np.std(res_irm), 2)]
    col_dict[col] = [f'{x}{PM}{y}' for x, y in zip(mu, std)]

df = pd.DataFrame(col_dict, index=['REF', 'ERM', 'IRM'])
print('Table 5:\n')
print(df)
# </editor-fold>


# <editor-fold desc="Table 13 reproduction">
# test ERM on SNLI test file with contradiction, neutral and entailment bias
for lbl_int in range(NUM_LABELS):
    lbl_str = labels_int_to_string(lbl_int)
    for directories in [ref_model_dirs, erm_model_dirs, irm_model_dirs]:
        for d in directories:
            if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
                test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
                         env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
            if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
                test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
                         env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
            if not os.path.isfile(os.sep.join([d, 'test_' + lbl_str, 'run_output.json'])):
                test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
                         env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)

# print results
col_dict = {}

for lbl_int in range(NUM_LABELS):
    lbl_str = labels_int_to_string(lbl_int)
    res_erm, res_irm, res_ref = calc_mean_var_for_test(os.sep.join([exp_dir, 'erm']), 'test_' + lbl_str, verbose=False), \
                                calc_mean_var_for_test(os.sep.join([exp_dir, 'irm']), 'test_' + lbl_str, verbose=False), \
                                calc_mean_var_for_test(os.sep.join([exp_dir, 'ref1']), 'test_' + lbl_str, verbose=False)
    mu = [np.round(np.mean(res_ref), 2), np.round(np.mean(res_erm), 2), np.round(np.mean(res_irm), 2)]
    std = [np.round(np.std(res_ref), 2), np.round(np.std(res_erm), 2), np.round(np.std(res_irm), 2)]
    col_dict[lbl_str] = [f'{x}{PM}{y}' for x, y in zip(mu, std)]

df = pd.DataFrame(col_dict, index=['REF', 'ERM', 'IRM'])
print('Table 13:\n')
print(df)
# </editor-fold>


# <editor-fold desc="biased sample ratio reproduce plot ">
erm_acc_mu, irm_acc_mu, erm_acc_std, irm_acc_std = [], [], [], []
biased_samples_ratio = []

for i in range(1, 14):
    exp_dir = f'models/exp2/ratio{i}'
    ratio = []
    erm_model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
               os.walk(os.sep.join([exp_dir, 'erm']))))
    erm_model_dirs = list(map(lambda x: x[0], erm_model_dirs))
    irm_model_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                         and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
               os.walk(os.sep.join([exp_dir, 'irm']))))
    irm_model_dirs = list(map(lambda x: x[0], irm_model_dirs))
    # test ERM on SNLI test file with p=0.8, p=0.33, p=0.2
    for directories in [erm_model_dirs, irm_model_dirs]:
        for d in directories:
            with open(os.sep.join([d, 'run_output.json'])) as f:
                f_config = json.load(f)['config']
            ratio.append(f_config['biased_samples_ratio'])
            if not os.path.isfile(os.sep.join([d, 'test_ood', 'run_output.json'])):
                test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood']), env_prob=(0.33,))

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
fig, ax = plt.subplots(figsize=(12.0, 8.0), dpi=240)

test_metric_mean = np.round(erm_acc_mu, 2)
yerr_plus = np.round(erm_acc_std, 2)
yerr_minus = np.round(erm_acc_std, 2)

ax.plot(biased_samples_ratio, test_metric_mean, marker='o', label='ERM')
ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)

test_metric_mean = np.round(irm_acc_mu, 2)
yerr_plus = np.round(irm_acc_std, 2)
yerr_minus = np.round(irm_acc_std, 2)

ax.plot(biased_samples_ratio, test_metric_mean, marker='x', label='IRM')
ax.fill_between(biased_samples_ratio, test_metric_mean - yerr_minus, test_metric_mean + yerr_plus, alpha=0.5)

ax.set_xticklabels(np.round(ax.get_xticks(), 2), fontsize=16)
ax.set_yticklabels(np.round(ax.get_yticks(), 2), fontsize=16)
ax.set_ylabel('accuracy', fontsize=22, labelpad=14.0)
ax.set_xlabel('proportion of biased samples', fontsize=22, labelpad=14.0)

ax.legend(fontsize=18, loc="lower right")
fig.savefig('biased_samples_ratio_exp.png')
# </editor-fold>

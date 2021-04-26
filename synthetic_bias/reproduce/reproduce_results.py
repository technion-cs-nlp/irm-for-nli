from main import test_irm
from utils import calc_mean_var_for_test
from settings import VOCAB_BIAS, NUM_LABELS, labels_int_to_string
import re
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D

# # exp1 - train ERM and IRM on p1=0.7 p2=0.9 with 5 seeds
# exp_dir = 'models/exp1/'
# erm_model_dirs = list(
#     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
#                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
#            os.walk(exp_dir + 'erm')))
# irm_model_dirs = list(
#     filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
#                      and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
#            os.walk(exp_dir + 'irm')))
#
# # <editor-fold desc="Table 2 reproduction">
# # test ERM on SNLI test file with p=0.8, p=0.33, p=0.2
# for d in erm_model_dirs:
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_in_dist']), env_prob=(0.8,))
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood1']), env_prob=(0.33,))
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.2,))
#
# # test IRM on SNLI test file with p=0.8, p=0.33, p=0.2
# for d in irm_model_dirs:
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_in_dist']), env_prob=(0.8,))
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood1']), env_prob=(0.33,))
#     test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood2']), env_prob=(0.2,))
#
# # print results
# print('Table 2, p=0.8:\n')
# print('ERM:\n')
# calc_mean_var_for_test(exp_dir + 'erm', 'test_in_dist')
# print('IRM:\n')
# calc_mean_var_for_test(exp_dir + 'irm', 'test_in_dist')
#
# print('Table 2, p=0.33:\n')
# print('ERM:\n')
# calc_mean_var_for_test(exp_dir + 'erm', 'test_ood1')
# print('IRM:\n')
# calc_mean_var_for_test(exp_dir + 'irm', 'test_ood1')
#
# print('Table 2, p=0.2:\n')
# print('ERM:\n')
# calc_mean_var_for_test(exp_dir + 'erm', 'test_ood2')
# print('IRM:\n')
# calc_mean_var_for_test(exp_dir + 'irm', 'test_ood2')
# # </editor-fold>
#
#
# # <editor-fold desc="Table 3 reproduction">
# # test ERM on SNLI test file with contradiction, neutral and entailment bias
# for lbl_int in range(NUM_LABELS):
#     lbl_str = labels_int_to_string(lbl_int)
#     for d in erm_model_dirs:
#         test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
#                  env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
#
# # test IRM on SNLI test file with contradiction, neutral and entailment bias
# for lbl_int in range(NUM_LABELS):
#     lbl_str = labels_int_to_string(lbl_int)
#     for d in irm_model_dirs:
#         test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_' + lbl_str]),
#                  env_prob=(1.0,), bias_tokens=[VOCAB_BIAS[lbl_int]] * NUM_LABELS)
#
# # print results
# for lbl_int in range(NUM_LABELS):
#     lbl_str = labels_int_to_string(lbl_int)
#     print(f'Table 3, {lbl_str} bias:\n')
#     print('ERM:\n')
#     calc_mean_var_for_test(exp_dir + 'erm', 'test_' + lbl_str)
#     print('IRM:\n')
#     calc_mean_var_for_test(exp_dir + 'irm', 'test_' + lbl_str)
# # </editor-fold>
#
#
# ################################### biased sample ratio reproduce plot ##############
# # need to add testing
# erm_acc_mu, irm_acc_mu, erm_acc_std, irm_acc_std = [], [], [], []
# biased_samples_ratio = []
#
# for i in range(10):
#     exp_dir = f'models/biased_samples_ratio_exp/exp{i}'
#     erm_model_dirs = list(
#         filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
#                          and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
#                os.walk(exp_dir + 'erm')))
#     irm_model_dirs = list(
#         filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
#                          and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
#                os.walk(exp_dir + 'irm')))
#
#     # test ERM on SNLI test file with p=0.8, p=0.33, p=0.2
#     for d in erm_model_dirs:
#         test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood']), env_prob=(0.33,))
#
#     # test IRM on SNLI test file with p=0.8, p=0.33, p=0.2
#     for d in irm_model_dirs:
#         test_irm('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test_ood']), env_prob=(0.33,))
#
#     erm_acc, irm_acc = [], []
#     erm_acc.append(calc_mean_var_for_test(exp_dir + 'erm', 'test_ood')['acc'])
#     erm_acc_mu.append(np.mean(erm_acc))
#     erm_acc_std.append(np.mean(erm_acc))
#     irm_acc.append(calc_mean_var_for_test(exp_dir + 'irm', 'test_ood')['acc'])
#     irm_acc_mu.append(np.mean(irm_acc))
#     irm_acc_std.append(np.mean(irm_acc))
#     biased_samples_ratio.append(i/10.0)

biased_samples_ratio = [i/10.0 for i in range(1,10)]
erm_acc_mu, erm_acc_std = [85.8, 86.02, 85.98, 86.14, 85.62, 84.98, 85.06, 85.43, 85.06], [0.94, 0.32, 0.7, 0.38, 0.48, 0.75, 0.55, 0.48, 0.55]
irm_acc_mu, irm_acc_std = [84.4, 84.97, 84.77, 85.12, 85.45, 86.18, 86.09, 86.77, 86.09], [0.71, 0.72, 1.32, 0.53, 1.46, 1.22, 0.94, 0.61, 0.94]

fig, ax = plt.subplots(figsize=(20.4, 8.8))

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
ax.axes.xaxis.set_ticks([])
ax.set_ylim(bottom=83.0, top=88.0)
# ax.set_title('Accuracy on out-of-distribution for increasing proportion of biased samples')
ax.set_ylabel('accuracy', fontsize=18)
ax.set_xlabel('proportion of biased samples', fontsize=18)

# add table
ax.xaxis.labelpad = 140
ax.yaxis.labelpad = 20

table_vals = [[str(mu) + r' $\pm$ ' + str(std) for mu, std in zip(erm_acc_mu, erm_acc_std)],
              [str(mu) + r' $\pm$ ' + str(std) for mu, std in zip(irm_acc_mu, irm_acc_std)]]
table = ax.table(cellText=table_vals, colLabels=biased_samples_ratio, rowLabels=['ERM', 'IRM'], loc='bottom')

nrows, ncols = 12, 8
fig.subplots_adjust(left=0.2, bottom=0.25)
table.auto_set_font_size(False)
table.set_fontsize(14)
cell_height = 1 / nrows
cell_width = 1 / (ncols + 1)
for pos, cell in table.get_celld().items():
    cell.set_height(cell_height)
    cell.set_width(cell_width)


ax.legend(fontsize=18)
fig.show()
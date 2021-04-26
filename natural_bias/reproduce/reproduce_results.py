from main import test_on_splits, analyze_splits_difficulty
from utils import calc_mean_var_for_test
from data_utils import datasets_config
import re
import os

####################### SNLI hypothesis bias #######################

# exp1 - train ERM and IRM on p1=0.7 p2=0.9 with 5 seeds
exp_dir = 'models/exp1/'
erm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'erm')))
irm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'irm')))

# <editor-fold desc="Table 4 reproduction">
# test ERM on splits of SNLI test file
for d in erm_model_dirs:
    test_on_splits('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# test IRM on splits of SNLI test file
for d in irm_model_dirs:
    test_on_splits('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# print results
print('Table 4:\n')
print('ERM:\n')
calc_mean_var_for_test(exp_dir + 'erm', 'test')
print('IRM:\n')
calc_mean_var_for_test(exp_dir + 'irm', 'test')
# </editor-fold>


# <editor-fold desc="Table 6 reproduction">
# analyze difficulty of splits according to annotator agreement.
# note - it doesn't matter which of the runs we use since they all use the same scores file and same thresholds
unbiased_dict, correctly_biased_dict, incorrectly_biased_dict = analyze_splits_difficulty(test_dir=erm_model_dirs[0], subset='test')

print(f'Majority count distribution for test set:')
print(f'unbiased split : \n {unbiased_dict}')
print(f'bias aligned split: \n {correctly_biased_dict}')
print(f'bias unaligned split: \n {incorrectly_biased_dict}')
# </editor-fold>

# exp2 - train unbiased model (by training ERM on unbiased split) and train ERM and IRM on environments
# of same size as unbiased split for fair comparison - 5 seeds
exp_dir = 'models/exp2/'
unbiased_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'unbiased')))
erm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'erm')))
irm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'irm')))

# <editor-fold desc="Table 5 reproduction">
# test ERM on splits of SNLI test file
for d in unbiased_model_dirs:
    test_on_splits('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

for d in erm_model_dirs:
    test_on_splits('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# test IRM on splits of SNLI test file
for d in irm_model_dirs:
    test_on_splits('data/SNLI/snli_1.0_test.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# print results
print('Table 5:\n')
print('Unbiased model:\n')
calc_mean_var_for_test(exp_dir + 'unbiased', 'test')
print('ERM:\n')
calc_mean_var_for_test(exp_dir + 'erm', 'test')
print('IRM:\n')
calc_mean_var_for_test(exp_dir + 'irm', 'test')
# </editor-fold>


####################### MNLI overlap bias #######################

# exp3 - train ERM and IRM on p1=0.7 p2=0.9 with 5 seeds
exp_dir = 'models/exp3/'
erm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'erm')))
irm_model_dirs = list(
    filter(lambda x: re.match('.*run[0-9]+.*', x[0]) is not None
                     and all([f in x[2] for f in ['run_output.json', 'pytorch_model.bin', 'config.json']]),
           os.walk(exp_dir + 'irm')))

# <editor-fold desc="Table 7 reproduction">
# test ERM on splits of MNLI test file
for d in erm_model_dirs:
    test_on_splits('data/MNLI/multinli_1.0_dev_mismatched.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# test IRM on splits of MNLI test file
for d in irm_model_dirs:
    test_on_splits('data/MNLI/multinli_1.0_dev_mismatched.txt', test_dir=d, out_dir=os.sep.join([d, 'test']), subset='test')

# print results
print('Table 7:\n')
print('ERM:\n')
calc_mean_var_for_test(exp_dir + 'erm', 'test')
print('IRM:\n')
calc_mean_var_for_test(exp_dir + 'irm', 'test')
# </editor-fold>
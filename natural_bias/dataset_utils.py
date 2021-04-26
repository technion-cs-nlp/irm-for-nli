import numpy as np
from torch.utils.data import Dataset, Subset, DataLoader
import os
import torch
import json
import math
import torch.nn.functional as f
from transformers import BertTokenizer, BertModel, BertConfig
from collections import Counter


def prepare_scores(scores_file):
    """Read scores file and parse train, val and test scores.
    Each set scores is a list of tuples of the shape:
    (sample_index, sample_score) where sample_score is a list of predicted probability of len num_labels"""
    assert os.path.isfile(scores_file), f'{scores_file} - No such file'
    with open(scores_file, 'r') as f:
        scores = json.load(f)

    return scores['train_scores'], scores['val_scores'], scores['test_scores']


def create_dataset(filepath, field_indices, label_str_to_int, label_int_to_str):
    """
    Create NLIDataset from filepath.
    :param filepath: path to file to read the data from
    :param field_indices: list of indices of the required fields. Each line of filepath is split by sep='\t' and the
    fields taken are those indicated by field_indices.
    :param label_str_to_int: function receiving string indicating the label and converts to integer indicating the label
    :param label_int_to_str: function receiving integer indicating the label and converts to string indicating the label
    :return: NLIDataset initialized instances
    """
    assert os.path.isfile(filepath), f'{filepath} - No such file'
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # remove line ending and split on tabs. Skip first line (headers)
    samples = []
    for line in lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        candidate = tuple([samp[field_idx] for field_idx in field_indices])
        if len(candidate[0].split()) < 80:
            samples.append(candidate)
    # samples is a list of tuples

    return NLIDataset(samples, label_str_to_int, label_int_to_str)


def create_overlap_features_dataset(filepath, field_indices, label_str_to_int, label_int_to_str):
    """
    Create NLIOverlapFeaturesDataset from filepath.
    :param filepath: path to file to read the data from
    :param field_indices: list of indices of the required fields. Each line of filepath is split by sep='\t' and the
    fields taken are those indicated by field_indices.
    :param label_str_to_int: function receiving string indicating the label and converts to integer indicating the label
    :param label_int_to_str: function receiving integer indicating the label and converts to string indicating the label
    :return: NLIDataset initialized instances
    """
    assert os.path.isfile(filepath), f'{filepath} - No such file'
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # remove line ending and split on tabs. Skip first line (headers)
    samples = []
    for line in lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        candidate = tuple([samp[field_idx] for field_idx in field_indices])
        if len(candidate[0].split()) < 80:
            samples.append(candidate)

    # samples is a list of tuples

    return NLIOverlapFeaturesDataset(samples, label_str_to_int, label_int_to_str)


def create_envs(dataset, scores, env_prob, threshold1, threshold2, rng=None):
    """
    Create Subsets of NLIDataset.
    Split dataset to unbiased, correctly_biased and incorrectly biased indices according to scores.
    Split indices to envs such that for each environment we get
    p(y|bias) = correctly_biased / (correctly_biased + incorrectly_biased) = env_prob[i].
    Create NLISubset from each env and return a list of NLISubsets.
    Note:
    The splitting is done by first splitting the correctly biased equally between environments, and similarly for
    the incorrectly biased. Then we throw away samples (correctly or incorrectly biased) to reach required probability.
    Then check feasibility - that we have enough unbiased samples to replace thrown away samples to complete all
    environments to maximal sized environment and thus ensure equal size of environments.
    If any left over unbiased samples - equally divide them between environments to minimize samples thrown away.
    :param dataset: dataset to divide to environments
    :param scores: list of tuples (sample_index, sample_pred), generated from the file. sample_pred is a list of len
    num_labels, with the predicted probability or each class.
    :param env_prob: tuple of environment probabilities
    :param threshold1: samples for which the total variation distance between the vector of predictions and
     the uniform distribution <= threshold1 are considered unbiased, where as those whose distance > threshold1 are
     considered biased and passed to deciding whether they are correctly or incorrectly biased
    :param threshold2: after removing unbiased samples, this is the threshold used to distinguish the correctly biased
     from the incorrecty biased. For samples with predicted label = gt label and difference between two highest
     predicted probabilities >= threshold2 are considered correctly biased,
     where those with predicted label != gt label and difference between two highest
     predicted probabilities >= threshold2 are considered incorrectly biased.
    :param rng: numpy.random.RandomState - for reproducibility of the samples' split
    :return: list (of length num_envs) of NLISubset
    """

    num_samples, num_labels = len(scores), len(scores[0][-1])
    assert len(dataset) == num_samples, 'Scores file and data file are of different length'

    # split dataset by bias, return 3 lists of (disjoint) indices
    # unbiased_ind - list,
    # correctly_biased_ind - list of len num_labels of lists,
    # incorrectly_biased_ind - list of len num_labels of lists
    unbiased_ind, correctly_biased_ind, incorrectly_biased_ind = split_dataset_by_bias(dataset, scores, threshold1,
                                                                                       threshold2)
    if rng is None:
        print("Random seed is used")
        rng = np.random.RandomState()
    # shuffle
    rng.shuffle(unbiased_ind)
    for i in range(num_labels):
        rng.shuffle(correctly_biased_ind[i])
        rng.shuffle(incorrectly_biased_ind[i])

    # <editor-fold desc="drop samples so that groups could be divided to num_envs">
    num_envs = len(env_prob)
    dropped_samples = 0
    for l in range(num_labels):
        residual_correctly_biased, residual_incorrectly_biased = len(correctly_biased_ind[l]) % num_envs, len(
            incorrectly_biased_ind[l]) % num_envs
        if residual_correctly_biased:
            correctly_biased_ind[l] = correctly_biased_ind[l][:-residual_correctly_biased]
            dropped_samples += residual_correctly_biased
        if residual_incorrectly_biased:
            incorrectly_biased_ind[l] = incorrectly_biased_ind[l][:-residual_incorrectly_biased]
            dropped_samples += residual_incorrectly_biased
    # </editor-fold>

    # <editor-fold desc="For each environment calculate number of samples to take from each subset">
    cnt_correctly_biased, cnt_incorrectly_biased = [], []  # list of lists: num_labels x num_envs

    for l in range(num_labels):
        n, c = len(correctly_biased_ind[l]) // num_envs, len(incorrectly_biased_ind[l]) // num_envs
        cnt_correctly_biased_per_env, cnt_incorrectly_biased_per_env = [], []  # list of length: num_envs
        for p in env_prob:
            n_new, c_new, x = calc_size_envs(p, n, c)
            cnt_correctly_biased_per_env.append(n_new), cnt_incorrectly_biased_per_env.append(c_new)
        cnt_correctly_biased.append(cnt_correctly_biased_per_env)
        cnt_incorrectly_biased.append(cnt_incorrectly_biased_per_env)

    env_size = (np.sum(np.array(cnt_correctly_biased).reshape(num_labels, num_envs), axis=0) +
                np.sum(np.array(cnt_incorrectly_biased).reshape(num_labels, num_envs), axis=0)).tolist()
    max_env_size = np.max(env_size)
    cnt_missing_per_env = [max_env_size - e_size for e_size in env_size]
    # </editor-fold>

    assert sum(cnt_missing_per_env) <= len(unbiased_ind), "Environment probabilities vector is not feasible"

    # <editor-fold desc="create environments">
    envs = [[] for _ in range(num_envs)]
    for e in range(num_envs):
        cnts_per_label = list(zip(*cnt_correctly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += correctly_biased_ind[lbl][:cnts_per_label[lbl]]
            correctly_biased_ind[lbl] = correctly_biased_ind[lbl][cnts_per_label[lbl]:]
        cnts_per_label = list(zip(*cnt_incorrectly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += incorrectly_biased_ind[lbl][:cnts_per_label[lbl]]
            incorrectly_biased_ind[lbl] = incorrectly_biased_ind[lbl][cnts_per_label[lbl]:]
        envs[e] += unbiased_ind[:cnt_missing_per_env[e]]
        unbiased_ind = unbiased_ind[cnt_missing_per_env[e]:]

    # if there are unbiased samples left - divide them equally between environments, to reduce samples thrown away
    left_overs = len(unbiased_ind) // num_envs
    if left_overs > 0:
        for e in range(num_envs):
            envs[e] += unbiased_ind[:left_overs]
            unbiased_ind = unbiased_ind[left_overs:]

    envs = [NLISubset(dataset, indices) for indices in envs]
    # </editor-fold>

    return envs


def create_restricted_size_envs(dataset, scores, env_prob, threshold1, threshold2, rng=None):
    """
    Create Subsets of NLIDataset.
    Split dataset to unbiased, correctly_biased and incorrectly biased indices according to scores.
    Split indices to envs such that for each environment we get
    p(y|bias) = correctly_biased / (correctly_biased + incorrectly_biased) = env_prob[i].
    Create NLISubset from each env and return a list of NLISubsets.
    Note:
    The splitting is done by first splitting the correctly biased equally between environments, and similarly for
    the incorrectly biased. Then we throw away samples (correctly or incorrectly biased) to reach required probability.
    Then check feasibility - that we have enough unbiased samples to replace thrown away samples to complete all
    environments to maximal sized environment and thus ensure equal size of environments.
    If any left over unbiased samples - equally divide them between environments to minimize samples thrown away.
    :param dataset: dataset to divide to environments
    :param scores: list of tuples (sample_index, sample_pred), generated from the file. sample_pred is a list of len
    num_labels, with the predicted probability or each class.
    :param env_prob: tuple of environment probabilities
    :param threshold1: samples for which the total variation distance between the vector of predictions and
     the uniform distribution <= threshold1 are considered unbiased, where as those whose distance > threshold1 are
     considered biased and passed to deciding whether they are correctly or incorrectly biased
    :param threshold2: after removing unbiased samples, this is the threshold used to distinguish the correctly biased
     from the incorrecty biased. For samples with predicted label = gt label and difference between two highest
     predicted probabilities >= threshold2 are considered correctly biased,
     where those with predicted label != gt label and difference between two highest
     predicted probabilities >= threshold2 are considered incorrectly biased.
    :param rng: numpy.random.RandomState - for reproducibility of the samples' split
    :return: list (of length num_envs) of NLISubset
    """

    num_samples, num_labels = len(scores), len(scores[0][-1])
    assert len(dataset) == num_samples, 'Scores file and data file are of different length'

    # split dataset by bias, return 3 lists of (disjoint) indices
    # unbiased_ind - list,
    # correctly_biased_ind - list of len num_labels of lists,
    # incorrectly_biased_ind - list of len num_labels of lists
    unbiased_ind, correctly_biased_ind, incorrectly_biased_ind = split_dataset_by_bias(dataset, scores, threshold1,
                                                                                       threshold2)
    if rng is None:
        print("Random seed is used")
        rng = np.random.RandomState()
    # shuffle
    rng.shuffle(unbiased_ind)
    for i in range(num_labels):
        rng.shuffle(correctly_biased_ind[i])
        rng.shuffle(incorrectly_biased_ind[i])

    # assign indices to environments (without unbiased).
    # If envs size > len(unbiased) cut down 100 samples from maximum set
    while True:
        # <editor-fold desc="drop samples so that groups could be divided to num_envs">
        num_envs = len(env_prob)
        dropped_samples = 0
        for l in range(num_labels):
            residual_correctly_biased, residual_incorrectly_biased = len(correctly_biased_ind[l]) % num_envs, len(
                incorrectly_biased_ind[l]) % num_envs
            if residual_correctly_biased:
                correctly_biased_ind[l] = correctly_biased_ind[l][:-residual_correctly_biased]
                dropped_samples += residual_correctly_biased
            if residual_incorrectly_biased:
                incorrectly_biased_ind[l] = incorrectly_biased_ind[l][:-residual_incorrectly_biased]
                dropped_samples += residual_incorrectly_biased
        # </editor-fold>

        # <editor-fold desc="For each environment calculate number of samples to take from each subset">
        cnt_correctly_biased, cnt_incorrectly_biased = [], []  # list of lists: num_labels x num_envs

        for l in range(num_labels):
            n, c = len(correctly_biased_ind[l]) // num_envs, len(incorrectly_biased_ind[l]) // num_envs
            cnt_correctly_biased_per_env, cnt_incorrectly_biased_per_env = [], []  # list of length: num_envs
            for p in env_prob:
                n_new, c_new, x = calc_size_envs(p, n, c)
                cnt_correctly_biased_per_env.append(n_new), cnt_incorrectly_biased_per_env.append(c_new)
            cnt_correctly_biased.append(cnt_correctly_biased_per_env)
            cnt_incorrectly_biased.append(cnt_incorrectly_biased_per_env)

        env_size = (np.sum(np.array(cnt_correctly_biased).reshape(num_labels, num_envs), axis=0) +
                    np.sum(np.array(cnt_incorrectly_biased).reshape(num_labels, num_envs), axis=0)).tolist()
        max_env_size = np.max(env_size)
        cnt_missing_per_env = [max_env_size - e_size for e_size in env_size]
        # </editor-fold>

        if not num_envs * max_env_size > len(unbiased_ind):
            break

        correctly_biased_lens = [len(correctly_biased_ind[l]) for l in range(num_labels)]
        incorrectly_biased_lens = [len(incorrectly_biased_ind[l]) for l in range(num_labels)]
        lens = np.array([correctly_biased_lens, incorrectly_biased_lens])
        trim_flag = np.argmax(lens)
        if trim_flag < len(correctly_biased_ind):
            correctly_biased_ind[trim_flag - len(correctly_biased_ind)] = correctly_biased_ind[
                                                                              trim_flag - len(correctly_biased_ind)][
                                                                          :-100]
        else:
            incorrectly_biased_ind[trim_flag - len(correctly_biased_ind)] = incorrectly_biased_ind[
                                                                                trim_flag - len(correctly_biased_ind)][
                                                                            :-100]

    cnt_extra = len(unbiased_ind) - num_envs * max_env_size

    # <editor-fold desc="create environments">
    envs = [[] for _ in range(num_envs)]
    for e in range(num_envs):
        cnts_per_label = list(zip(*cnt_correctly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += correctly_biased_ind[lbl][:cnts_per_label[lbl]]
            correctly_biased_ind[lbl] = correctly_biased_ind[lbl][cnts_per_label[lbl]:]
        cnts_per_label = list(zip(*cnt_incorrectly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += incorrectly_biased_ind[lbl][:cnts_per_label[lbl]]
            incorrectly_biased_ind[lbl] = incorrectly_biased_ind[lbl][cnts_per_label[lbl]:]
        envs[e] += unbiased_ind[:cnt_missing_per_env[e]]
        unbiased_ind = unbiased_ind[cnt_missing_per_env[e]:]
    # </editor-fold>

    # if there are unbiased samples left - divide them equally between environments, to reduce samples thrown away
    left_overs = cnt_extra // num_envs
    if left_overs > 0:
        for e in range(num_envs):
            envs[e] += unbiased_ind[:left_overs]
            unbiased_ind = unbiased_ind[left_overs:]

    envs = [NLISubset(dataset, indices) for indices in envs]

    return envs


def create_restricted_envs(dataset, scores, env_prob, threshold1, threshold2,
                                      biased_samples_ratio, total_size=None, rng=None, rel_tol=0.001):
    """
    Restricted version of create_envs. The resulting environments must hold :
    #biased_samples / #unbiased_samples = biased_samples_ratio. Also, if total_size parameter is passed, the accumulated
    size of resulting environments must be exactly total_size. This is achieved by up-sampling biased or unbiased samples,
    according to the biased_samples_ratio passed in.
    :param dataset: dataset to divide to environments
    :param scores: list of tuples (sample_index, sample_pred), generated from the file. sample_pred is a list of len
    num_labels, with the predicted probability or each class.
    :param env_prob: tuple of environment probabilities
    :param threshold1: samples for which the total variation distance between the vector of predictions and
     the uniform distribution <= threshold1 are considered unbiased, where as those whose distance > threshold1 are
     considered biased and passed to deciding whether they are correctly or incorrectly biased
    :param threshold2: after removing unbiased samples, this is the threshold used to distinguish the correctly biased
     from the incorrecty biased. For samples with predicted label = gt label and difference between two highest
     predicted probabilities >= threshold2 are considered correctly biased,
     where those with predicted label != gt label and difference between two highest
     predicted probabilities >= threshold2 are considered incorrectly biased.
    :param biased_samples_ratio: Specifies the required following ratio biased_samples / unbiased_samples.
    :param total_size: Resulting environments required accumulated size. Can be none (no size restriction). The size
    restriction (if exists) must be feasible - i.e. bigger or equal to the size of the initial unrestricted environments.
    :param rng: numpy.random.RandomState - for reproducibility of the samples' split
    :return: list (of length num_envs) of NLISubset
    """

    num_samples, num_labels = len(scores), len(scores[0][-1])
    assert len(dataset) == num_samples, 'Scores file and data file are of different length'

    # split dataset by bias, return 3 lists of (disjoint) indices
    # unbiased_ind - list,
    # correctly_biased_ind - list of len num_labels of lists,
    # incorrectly_biased_ind - list of len num_labels of lists
    unbiased_ind, correctly_biased_ind, incorrectly_biased_ind = split_dataset_by_bias(dataset, scores, threshold1,
                                                                                       threshold2)
    if rng is None:
        print("Random seed is used")
        rng = np.random.RandomState()
    # shuffle
    rng.shuffle(unbiased_ind)
    for i in range(num_labels):
        rng.shuffle(correctly_biased_ind[i])
        rng.shuffle(incorrectly_biased_ind[i])

    # assign indices to environments (without unbiased).
    # If envs size > len(unbiased) cut down 100 samples from maximum set

    # <editor-fold desc="drop samples so that groups could be divided to num_envs">
    num_envs = len(env_prob)
    assert num_envs == 2, 'Only implemented for 2 environments'
    dropped_samples = 0
    for l in range(num_labels):
        residual_correctly_biased, residual_incorrectly_biased = len(correctly_biased_ind[l]) % num_envs, len(
            incorrectly_biased_ind[l]) % num_envs
        if residual_correctly_biased:
            correctly_biased_ind[l] = correctly_biased_ind[l][:-residual_correctly_biased]
            dropped_samples += residual_correctly_biased
        if residual_incorrectly_biased:
            incorrectly_biased_ind[l] = incorrectly_biased_ind[l][:-residual_incorrectly_biased]
            dropped_samples += residual_incorrectly_biased
    # </editor-fold>

    # <editor-fold desc="For each environment calculate number of samples to take from each subset">
    cnt_correctly_biased, cnt_incorrectly_biased = [], []  # list of lists: num_labels x num_envs

    for l in range(num_labels):
        n, c = len(correctly_biased_ind[l]) // num_envs, len(incorrectly_biased_ind[l]) // num_envs
        cnt_correctly_biased_per_env, cnt_incorrectly_biased_per_env = [], []  # list of length: num_envs
        for p in env_prob:
            n_new, c_new, x = calc_size_envs(p, n, c)
            cnt_correctly_biased_per_env.append(n_new), cnt_incorrectly_biased_per_env.append(c_new)
        cnt_correctly_biased.append(cnt_correctly_biased_per_env)
        cnt_incorrectly_biased.append(cnt_incorrectly_biased_per_env)

    # </editor-fold>

    while True:
        # <editor-fold desc="update counts so that environment probability hold and environments are of same size">
        validity_restriction = True
        for i in range(num_labels):
            p1, p2 = min(env_prob), max(env_prob)
            if env_prob[0] < env_prob[1]:
                x1, y1, x2, y2 = cnt_correctly_biased[i][0], cnt_incorrectly_biased[i][0], cnt_correctly_biased[i][1], \
                                 cnt_incorrectly_biased[i][1]
            else:
                x1, y1, x2, y2 = cnt_correctly_biased[i][1], cnt_incorrectly_biased[i][1], cnt_correctly_biased[i][0], \
                                 cnt_incorrectly_biased[i][0]

            x2 = int(x1 * (p2 / p1))
            y1 = int(x1 * (1 / p1 - 1))
            y2_ideal = int(x2 * (1 / p2 - 1))
            y2 = x1 + y1 - x2

            assert math.isclose(y2, y2_ideal, abs_tol=5)
            assert all([x1 >= 0, y1 >= 0, x2 >= 0, y2 >= 0])
            validity_restriction = validity_restriction and all([x1 + x2 <= len(correctly_biased_ind[i]), y1 + y2 <= len(incorrectly_biased_ind[i])])

            if env_prob[0] < env_prob[1]:
                cnt_correctly_biased[i][0], cnt_incorrectly_biased[i][0], cnt_correctly_biased[i][1], \
                                 cnt_incorrectly_biased[i][1] = x1, y1, x2, y2
            else:
                cnt_correctly_biased[i][1], cnt_incorrectly_biased[i][1], cnt_correctly_biased[i][0], \
                                 cnt_incorrectly_biased[i][0] = x1, y1, x2, y2
        # </editor-fold>

        # calculate number of unbiased samples required to hold the biased samples ratio
        num_biased = np.sum(cnt_correctly_biased).item() + np.sum(cnt_incorrectly_biased).item()
        num_unbiased = int(num_biased * (1 / biased_samples_ratio - 1))
        validity_restriction = validity_restriction and num_unbiased <= len(unbiased_ind)

        # <editor-fold desc="If resulting dataset is too big, reduce number of samples, if resulting dataset is too small raise error">
        if total_size is None:
            size_restriction_holds = True
        else:
            size_restriction_holds = math.isclose(num_unbiased + num_biased, total_size, rel_tol=rel_tol)

        if size_restriction_holds and validity_restriction:
            break
        else:
            if total_size is not None:
                assert num_unbiased + num_biased > total_size, 'biggest environments are too small'
            if env_prob[0] < env_prob[1]:
                for i in range(num_labels):
                    cnt_correctly_biased[i][0] = max(cnt_correctly_biased[i][0] - 1, 0)
            else:
                for i in range(num_labels):
                    cnt_correctly_biased[i][1] = max(cnt_correctly_biased[i][1] - 1, 0)
        # </editor-fold>

    assert num_unbiased <= len(unbiased_ind), "Environment biased sample ratio is not feasible"
    unbiased_ind = unbiased_ind[:num_unbiased]

    # <editor-fold desc="create environments with onlu biased sample to hold environment probability">
    envs = [[] for _ in range(num_envs)]
    for e in range(num_envs):
        cnts_per_label = list(zip(*cnt_correctly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += correctly_biased_ind[lbl][:cnts_per_label[lbl]]
            correctly_biased_ind[lbl] = correctly_biased_ind[lbl][cnts_per_label[lbl]:]
        cnts_per_label = list(zip(*cnt_incorrectly_biased))[e]
        for lbl in range(num_labels):
            envs[e] += incorrectly_biased_ind[lbl][:cnts_per_label[lbl]]
            incorrectly_biased_ind[lbl] = incorrectly_biased_ind[lbl][cnts_per_label[lbl]:]
    # </editor-fold>

    # <editor-fold desc="add unbiased samples to create required biased samples ratio">
    left_overs = len(unbiased_ind) // num_envs
    if left_overs > 0:
        for e in range(num_envs):
            envs[e] += unbiased_ind[:left_overs]
            unbiased_ind = unbiased_ind[left_overs:]
    # </editor-fold>

    envs = [NLISubset(dataset, indices) for indices in envs]

    return envs


def split_dataset_by_bias(dataset, scores, threshold1, threshold2):
    """
        Split dataset to unbiased, correctly_biased and incorrectly biased indices according to scores.
        Split to unbiased and biased - consider as unbiased samples with probability close to uniform, other samples
        are considered as biased.
        Split biased samples to correctly biased and incorrectly biased:
        samples woth predicted label = gt label are correctly biased, the others are incorrectly biased.
        Further filter to get only confident biased samples.
        :param dataset: dataset to divide to environments
        :param scores: list of tuples (sample_index, sample_pred), generated from the file. sample_pred is a list of len
        num_labels, with the predicted probability or each class.
        :param env_prob: tuple of environment probabilities
        :param threshold1: samples for which the total variation distance between the vector of predictions and
         the uniform distribution <= threshold1 are considered unbiased, where as those whose distance > threshold1 are
         considered biased and passed to deciding whether they are correctly or incorrectly biased
        :param threshold2: after removing unbiased samples, this is the threshold used to distinguish the correctly biased
         from the incorrecty biased. For samples with predicted label = gt label and difference between two highest
         predicted probabilities >= threshold2 are considered correctly biased,
         where those with predicted label != gt label and difference between two highest
         predicted probabilities >= threshold2 are considered incorrectly biased.
        :return: unbiased_ind, correctly_biased_ind, incorrectly_biased_ind
        unbiased_ind - list, correctly_biased_ind - list (of len num_label) of lists,
        incorrectly_biased_ind - list (of len num_label) of lists
        """
    ind, prob = [np.array(list(x)) for x in zip(*scores)]  # shapes: ind - n , prob - n x num_labels
    n, num_labels = prob.shape

    # split indices to indices of biased samples and indices of unbiased samples,
    # according to total variation distance from uniform distribution
    dist_from_uniform = 0.5 * np.linalg.norm(prob - np.ones_like(prob) * (1 / num_labels), ord=1, axis=1)  # shape: n
    unbiased_ind, biased_ind = ind[dist_from_uniform <= threshold1], ind[dist_from_uniform > threshold1]
    biased_prob = prob[dist_from_uniform > threshold1, :]

    # split biased samples to correctly biased and incorrectly biased samples, by sorting
    # according to probability assigned to ground truth label
    labels = dataset.labels
    labels = np.array(labels)[biased_ind]  # get labels of biased samples
    sort_indices = np.argsort(biased_prob, axis=-1)
    first_largest_label, second_largest_label = sort_indices[:, -1], sort_indices[:, -2]
    first_largest_pred, second_largest_pred = biased_prob[
                                                  [i for i in range(len(first_largest_label))], first_largest_label], \
                                              biased_prob[
                                                  [i for i in range(len(second_largest_label))], second_largest_label]

    correctly_pred_ind, incorrectly_predicted_ind = first_largest_label == labels, first_largest_label != labels
    pred_difference = first_largest_pred - second_largest_pred
    correctly_biased_ind = biased_ind[np.bitwise_and(correctly_pred_ind, pred_difference > threshold2)]
    incorrectly_biased_ind = biased_ind[np.bitwise_and(incorrectly_predicted_ind, pred_difference > threshold2)]

    correctly_biased_prob = biased_prob[np.bitwise_and(correctly_pred_ind, pred_difference > threshold2), :]
    incorrectly_biased_prob = biased_prob[np.bitwise_and(incorrectly_predicted_ind, pred_difference > threshold2), :]

    # divide correctly and incorrectly biased to sub groups according to bias type. We assume that bias type
    # is indicated by prediction, so this actually means to divide to sub groups according to model prediction
    pred_correctly_biased = np.argmax(correctly_biased_prob, axis=-1)
    correctly_biased_ind = [correctly_biased_ind[pred_correctly_biased == i].tolist() for i in range(num_labels)]
    pred_incorrectly_biased = np.argmax(incorrectly_biased_prob, axis=-1)
    incorrectly_biased_ind = [incorrectly_biased_ind[pred_incorrectly_biased == i].tolist() for i in range(num_labels)]

    return unbiased_ind.tolist(), correctly_biased_ind, incorrectly_biased_ind


def balance_dataset(dataset, scores):
    """
    Balance dataset and corresponding scores - if classes are not balanced, take the number of samples corresponding to
    minority class from each class and throw away the rest.
    :param dataset: dataset
    :param scores: scores. Will be filtered to match the samples.
    :return: tuple (new_dataset, new_scores)
    new_dataset - NLIDataset constructed from the class balanced new_samples
    new_scores - scores filtered appropriately to new_samples
    """
    rng = np.random.RandomState(seed=42)
    labels, samples = dataset.labels, dataset.samples
    label_str_to_int, label_int_to_str = dataset.label_str_to_int, dataset.label_int_to_str
    num_samples, num_labels = len(scores), len(scores[0][-1])
    assert len(dataset) == num_samples

    # get number of samples in minority class
    cnt = Counter(labels)
    labels_cnt = cnt.most_common(len(cnt))
    minority_lbl, minority_cnt = labels_cnt[-1]

    # generate balanced indices - each class has equal number of samples, and filter samples appropriately
    indices = []
    for i in range(num_labels):
        bool_flag = list(map(lambda x: x == i, labels))
        indices_per_label = np.arange(num_samples)[bool_flag]
        indices_per_label = rng.choice(indices_per_label, size=minority_cnt, replace=False).tolist()
        indices += indices_per_label
    new_samples = list(map(tuple, np.array(samples)[indices]))

    # filter scores appropriately (according to indices)
    indices = sorted(indices)
    scores = sorted(scores, key=lambda x: x[0])
    new_scores = []
    for i, ind in enumerate(indices):
        _, dist = scores[ind]
        new_scores.append([i, dist])

    # shuffle new_samples and new_scores
    shuffle_ind = rng.permutation(len(indices))
    new_samples = [new_samples[ind] for ind in shuffle_ind]
    new_scores = [new_scores[ind] for ind in shuffle_ind]

    new_dataset = NLIDataset(new_samples, label_str_to_int, label_int_to_str)
    return new_dataset, new_scores


def get_bias_ratio(dataset, scores, threshold1, threshold2):
    num_samples, num_labels = len(scores), len(scores[0][-1])
    assert len(dataset) == num_samples, 'Scores file and data file are of different length'

    # split dataset by bias, return 3 lists of (disjoint) indices
    # unbiased_ind - list,
    # correctly_biased_ind - list of len num_labels of lists,
    # incorrectly_biased_ind - list of len num_labels of lists
    unbiased_ind, correctly_biased_ind, incorrectly_biased_ind = split_dataset_by_bias(dataset, scores, threshold1,
                                                                                       threshold2)
    cnt_correctly_biased = sum([len(x) for x in correctly_biased_ind])
    cnt_incorrectly_biased = sum([len(x) for x in incorrectly_biased_ind])
    num_biased = cnt_correctly_biased + cnt_incorrectly_biased
    num_unbiased = len(unbiased_ind)

    if num_biased == 0:
        return 0
    else:
        return num_biased / (num_biased + num_unbiased)


def calc_size_envs(p, n, c):
    """
    Input is required environment probability p and current number of correctly biased (n) and incorrectly biased (c)
    number of samples in the environment.
    Calculate the number of samples we need from correctly_biased and incorrectly_biased
    to get the required env probability p
    :param p: required env probability - p = p(y|bias) = correctly_biased / (correctly_biased + incorrectly_biased)
    :param n: current #correctly_biased
    :param c: current #incorrectly_biased
    :return: n, c, x
    n - required #correctly_biased, c - required #incorrectly_biased, x - #dropped_samples
    """
    if n == 0 and c == 0:
        return n, c, 0
    assert n > 0 and c > 0, 'Can not generate environments'

    if p > n / (n + c):
        x = int((n * (p - 1) + c * p) / p)
        c = c - x
    elif p < n / (n + c):
        x = int((n - p * (n + c)) / (1 - p))
        n = n - x
    else:
        x = 0

    return n, c, x


class NLIDataset(Dataset):
    def __init__(self, samples, label_str_to_int, label_int_to_str):
        """
        Initialize dataset
        :param samples: list of tuples of strings (p, h, y)
        :param label_str_to_int: function receiving string indicating the label and converts to integer indicating the label
        :param label_int_to_str: function receiving integer indicating the label and converts to string indicating the label
        """
        self.samples = samples  # list of tuples of strings (p, h, y)
        self.label_str_to_int = label_str_to_int
        self.label_int_to_str = label_int_to_str
        self.labels = self._map_labels(list(zip(*samples))[-1])  # map all labels from str to int

    def __getitem__(self, idx):
        p, h, y = self.samples[idx]
        return p, h, torch.tensor(self.label_str_to_int(
            y))  # p,h are returned ase strings. Will be tokenized in training process (in batches)

    def __len__(self):
        assert len(self.samples) == len(self.samples) == len(self.labels), 'Samples and features have different lengths'
        return len(self.samples)

    def _map_labels(self, labels_str):
        labels_int = []

        for lbl in labels_str:
            labels_int.append(self.label_str_to_int(lbl))

        return labels_int


class NLIOverlapFeaturesDataset(Dataset):
    def __init__(self, samples, label_str_to_int, label_int_to_str):
        """
        Initialize dataset
        :param samples: list of tuples of strings (p, binary_parse_p, h, binary_parse_h, y)
        :param label_str_to_int: function receiving string indicating the label and converts to integer indicating the label
        :param label_int_to_str: function receiving integer indicating the label and converts to string indicating the label
        """
        samples_phy, features = self._extract_features(samples)
        self.features = features  # list of tuples (all_in, overlap_ratio, is_subsequence, is_constituent)
        self.samples = samples_phy  # list of tuples of strings (p, h, y)
        self.label_str_to_int = label_str_to_int
        self.label_int_to_str = label_int_to_str
        self.labels = self._map_labels(list(zip(*samples_phy))[-1])  # map all labels from str to int

    def __getitem__(self, idx):
        p, h, y = self.samples[idx]
        return p, h, torch.tensor([*self.features[idx]], dtype=torch.float), torch.tensor(self.label_str_to_int(y))

    def __len__(self):
        assert len(self.features) == len(self.samples) == len(
            self.labels), 'Samples and features have different lengths'
        return len(self.features)

    def _map_labels(self, labels_str):
        labels_int = []

        for lbl in labels_str:
            labels_int.append(self.label_str_to_int(lbl))

        return labels_int

    def _extract_features(self, samples):
        features = []
        sim_features = []
        samples_phy = []

        for samp in samples:
            prem, prem_parse, hyp, hyp_parse, label = samp
            all_in, overlap_ratio = lexical_overlap(prem, hyp, norm_by_hyp=False)
            subsequence_flag = is_subsequence(prem, hyp)
            constituent_flag = is_constituent(hyp, prem_parse)

            samples_phy.append((prem, hyp, label))
            features.append((all_in, overlap_ratio, subsequence_flag, constituent_flag))

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        bert = BertModel.from_pretrained('bert-base-uncased').to(device=device)
        bert.eval()  # we don't train Bert, just use it to get similarity scores
        ds = BasicDataset(samples)
        dl = DataLoader(ds, batch_size=64)
        with torch.no_grad():
            for batch in dl:
                prem, prem_parse, hyp, hyp_parse, label = batch
                input_dict_h = tokenizer(hyp, padding=True, truncation=True, return_tensors='pt')
                input_dict_p = tokenizer(prem, padding=True, truncation=True, return_tensors='pt')
                input_dict_h = {k: v.to(device) for k, v in input_dict_h.items()}
                input_dict_p = {k: v.to(device) for k, v in input_dict_p.items()}

                repr_h = bert(**input_dict_h)[0]
                repr_p = bert(**input_dict_p)[0]
                similarity = get_word_similarity(repr_p, repr_h, input_dict_p['attention_mask'],
                                                 input_dict_h['attention_mask']).detach().to('cpu')  # tensor batch x 3
                sim_features.append(similarity)
        sim_features = torch.cat(sim_features, dim=0)  # tensor : len(samples) x 3
        features = torch.tensor(features, dtype=torch.float)  # tensor: len(samples) x 4
        extracted_features = torch.cat([features, sim_features], dim=-1)  # tensor : len(samples) x 7

        return samples_phy, extracted_features.tolist()


class NLISubset(Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        # self.samples = [samp for ind, samp in enumerate(self.dataset.samples) if ind in indices]
        # self.labels = [lbl for ind, lbl in enumerate(self.dataset.labels) if ind in indices]
        self.samples = list(map(tuple, np.array(self.dataset.samples)[indices]))
        self.labels = np.array(self.dataset.labels)[indices].tolist()
        self.label_str_to_int = self.dataset.label_str_to_int
        self.label_int_to_str = self.dataset.label_int_to_str


class BasicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def label_distribution(envs):
    for i, e in enumerate(envs):
        labels = np.array(e.labels)
        label_int_to_str = e.label_int_to_str
        ratios = {}
        for lbl in set(labels):
            ratios[label_int_to_str(lbl)] = np.round(np.mean(labels == lbl), 2)
        print(f'Environment {i}: {ratios}')
    print(f'Size of resulting dataset: {sum([len(e) for e in envs])}')


# functions to get features for overlap bias


def sentence_to_words(sent: str):
    words = []
    for word in sent.split():
        if word not in [".", "?", "!"]:
            words.append(word.lower())
    return words


def sentence_to_words_no_punct(sent: str):
    words = []
    for word in sent.split():
        if word not in [".", "?", "!"]:
            words.append(word.lower().replace(".", "").replace("?", "").replace("!", ""))
    return words


def _prem_hypothesis_to_words(premise: str, hypothesis: str):
    prem_words = sentence_to_words(premise)
    hyp_words = sentence_to_words(hypothesis)
    return prem_words, hyp_words


def lexical_overlap(premise: str, hypothesis: str, norm_by_hyp=True):
    r"""Check if a given premise and hypothesis lexically overlap.
    Parameters:
        premise (str): The premise
        hypothesis (str): The hypothesis
        norm_by_hyp (bool): If True, the returned overlap percentage is calculated w.r.t. the hypothesis.
        Otherwise, it is calculated w.r.t. the premise.
    Returns:
        all_in (bool): True if all the words in the hypothesis are also in the premise, False otherwise.
        overlap_percent (int): The percentage of overlapping words (types) in the hypothesis the are also in
        the premise.
    """
    prem_words, hyp_words = _prem_hypothesis_to_words(premise, hypothesis)

    all_in = True

    for word in hyp_words:
        if word not in prem_words:
            all_in = False
            break

    num_overlapping = len(list(set(hyp_words) & set(prem_words)))
    if norm_by_hyp:
        overlap_percent = num_overlapping / len(set(hyp_words)) if len(set(hyp_words)) > 0 else 0
    else:
        overlap_percent = num_overlapping / len(set(prem_words)) if len(set(prem_words)) > 0 else 0

    return all_in, overlap_percent


def is_subsequence(premise, hypothesis):
    prem_words, hyp_words = _prem_hypothesis_to_words(premise, hypothesis)

    prem_filtered = " ".join(prem_words)
    hyp_filtered = " ".join(hyp_words)

    return hyp_filtered in prem_filtered


def is_constituent(hypothesis, parse):
    parse_new = sentence_to_words(parse)

    all_phrases = parse_phrase_list(" ".join(parse_new), [])

    hyp_words = sentence_to_words_no_punct(hypothesis)

    hyp_filtered = " ".join(hyp_words)
    return hyp_filtered in all_phrases


def parse_phrase_list(parse, phrases):
    if parse == "":
        return phrases

    phrase_list = phrases

    words = parse.split()
    this_phrase = []
    next_level_parse = []
    for index, word in enumerate(words):
        if word == "(":
            next_level_parse += this_phrase
            this_phrase = ["("]

        elif word == ")" and len(this_phrase) > 0 and this_phrase[0] == "(":
            phrase_list.append(" ".join(this_phrase[1:]))
            next_level_parse += this_phrase[1:]
            this_phrase = []
        elif word == ")":
            next_level_parse += this_phrase
            next_level_parse.append(")")
            this_phrase = []
        else:
            this_phrase.append(word)
    return parse_phrase_list(" ".join(next_level_parse), phrase_list)


def get_word_similarity(prem_matrix, hyp_matrix, p_mask, h_mask):
    """Compute similarity metric between Bert representation of premise tokens and hypothesis tokens."""
    # prem_matrix, hyp_matrix : 1 x seq_len x hidden_dim (768)
    # p_mask, h_mask: 1 x seq_len

    prem_matrix = f.normalize(prem_matrix, p=2, dim=-1)
    hyp_matrix = f.normalize(hyp_matrix, p=2, dim=-1)

    prem_matrix = prem_matrix * p_mask.view(prem_matrix.shape[0], prem_matrix.shape[1], 1).float()
    hyp_matrix = hyp_matrix * h_mask.view(hyp_matrix.shape[0], hyp_matrix.shape[1], 1).float()

    similarity_matrix = hyp_matrix.bmm(prem_matrix.transpose(2, 1))  # batch *seqlen(h)*seqlen(p)
    similarity = torch.max(similarity_matrix, -1)[0]  # batch *seqlen(h)

    sim_score = []

    similarity_replace = similarity.clone()
    # all the similarity values are smaller than 1 so 10 is a good number
    # so that the masked elements are not selected during the top minimum computations.
    similarity_replace[h_mask == 0] = 10
    y, i = torch.topk(similarity, k=1, dim=1, largest=False, sorted=True)  # batch x seqlen(h)
    sim_score.append(y.view(-1, 1))  # batch x 1

    h_lens = torch.sum(h_mask, 1)
    sum_similarity = torch.sum(similarity, 1)
    mean_similarity = sum_similarity / h_lens.float()
    sim_score.append(mean_similarity.view(-1, 1))  # batch x 1

    max_similarity = torch.max(similarity, 1)[0]
    sim_score.append(max_similarity.view(-1, 1))  # batch x 1

    similarity_score = torch.cat(sim_score, 1)  # batch x 3
    return similarity_score

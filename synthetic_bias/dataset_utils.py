import numpy as np
from torch.utils.data import Dataset
import settings
import torch
from itertools import accumulate, compress
from collections import Counter, defaultdict
import pandas as pd


def create_datasets(filepath, num_datasets=1, biased_samples_ratio=0.0, env_prob=(0.0,), rng=None, bias_tokens=None,
                    bias_pattern='simple', size=None):
    """
    Create #(num_envs) NLIDatasets from filepath. This function reads data from filepath, splits the samples to
    disjoint subsets according to num_envs, and constructs a NLIDataset from each subset.
    :param filepath: path to file to read the data from
    :param num_datasets: number of datasets to create (disjoint)
    :param rng: numpy.random.RandomState - for reproducibility of the samples' split
    :return: list of NLIDataset initialized instances
    """
    assert len(env_prob) == num_datasets, "Specify the probability for each dataset"
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # remove line ending and split on tabs. Skip first line (headers)
    samples = []
    for line in lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        candidate = tuple(samp)
        if len(candidate[0].split()) < 80:
            samples.append(candidate)

    # samples is a list of tuples: (p, h, y)

    if rng is None:
        print("Random seed is used")
        rng = np.random.RandomState()

    if num_datasets > 1:
        rng.shuffle(samples)
        if size is not None:
            assert len(samples) >= size, "Requested dataset size is too big"
            lengths = [size // num_datasets] * num_datasets
        else:
            lengths = [len(samples) // num_datasets] * num_datasets
    else:
        if size is not None:
            assert len(samples) >= size, "Requested dataset size is too big"
            lengths = [size]
        else:
            lengths = [len(samples)]
    print(f'Dropped {len(samples) - sum(lengths)} samples\n')

    datasets = []
    for offset, length, prob in zip(accumulate(lengths), lengths, env_prob):
        if bias_pattern == 'simple':
            datasets.append(
                NLIDatasetSimpleBiasPattern(samples[offset - length:offset], biased_samples_ratio=biased_samples_ratio,
                                            prob=prob, rng=rng, bias_tokens=bias_tokens))
        else:
            datasets.append(
                NLIDatasetComplexBiasPattern(samples[offset - length:offset], biased_samples_ratio=biased_samples_ratio,
                                             prob=prob, rng=rng, bias_tokens=bias_tokens))

    return datasets


class NLIDatasetComplexBiasPattern(Dataset):
    def __init__(self, samples, biased_samples_ratio=0.0, prob=0.0, rng=None, bias_tokens=None):
        """
        Initialize dataset
        :param samples: list of tuples of strings (p, h, y)
        :param bias_function: function to add bias to sample (receive the tuple as input and should return a tuple)
        """
        self.num_samples = len(samples)
        self.bias_tokens = bias_tokens if bias_tokens is not None else settings.VOCAB_BIAS[0]
        self.samples = self._initialize_dataset(samples, biased_samples_ratio=biased_samples_ratio, prob=prob, rng=rng)
        self.stats_counts, self.stats_cond_prob = self._get_bias_stats()

    def __getitem__(self, idx):
        samp = self.samples[idx]
        p, h, y = samp
        return p, h, torch.tensor(settings.labels_string_to_int(
            y))  # p,h are returned as strings. Will be tokenized in training process (in batches)

    def __len__(self):
        return self.num_samples

    def _initialize_dataset(self, samples, biased_samples_ratio=0.0, prob=0.0, rng=None):
        if rng is None:
            print("Random seed is used to create dataset")
            rng = np.random.RandomState()
        samples = self._add_bias(samples, biased_samples_ratio, prob, rng)

        return samples

    def _add_bias(self, samples, biased_samples_ratio, prob, rng):
        """
        Generate functions to add bias to datasets.
        """
        new_samples = []
        original_indices = [i for i in range(self.num_samples)]
        shuffled_indices = []
        if biased_samples_ratio == 0:
            new_samples = samples
        elif biased_samples_ratio < 1:
            # calc number of biased samples per label
            biased_samples_count = int(biased_samples_ratio * len(samples))
            biased_samples_count = biased_samples_count - biased_samples_count % settings.NUM_LABELS
            biased_samples_count_per_label = biased_samples_count // settings.NUM_LABELS

            # if not enough label in each class -> change biased_samples_count_per_label
            cnt = min(Counter(list(zip(*samples))[-1]).values())
            if cnt < biased_samples_count_per_label:
                biased_samples_count_per_label = cnt

            # for each label add biases to get required prob
            for label in range(settings.NUM_LABELS):
                filtered_samples = list(filter(lambda x: settings.labels_string_to_int(x[-1]) == label, samples))
                filtered_indices = list(
                    compress(original_indices, [settings.labels_string_to_int(x[-1]) == label for x in samples]))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    bias_indices = [lbl for lbl in range(settings.NUM_LABELS)]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (settings.NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[1]

                    chosen_bias_token = self.bias_tokens[bias_idx]
                    if type(chosen_bias_token) == tuple or type(chosen_bias_token) == list:
                        n = len(chosen_bias_token)
                        chosen_bias_tokens = rng.choice(self.bias_tokens[bias_idx], n - 1,
                                                        replace=False).tolist()  # list of n-1 correctly correlated bias tokens
                        non_chosen_bias_idx = bias_indices[:label] + bias_indices[label + 1:]
                        foreign_bias_tokens = [list(self.bias_tokens[i]) for i in non_chosen_bias_idx]
                        foreign_bias_tokens = [x for sublst in foreign_bias_tokens for x in
                                               sublst]  # list of 2*n incorrectly correlated tokens
                        chosen_bias_tokens.append(rng.choice(foreign_bias_tokens, 1).item())  # list of n tokens
                        rng.shuffle(chosen_bias_tokens)
                        chosen_bias_token = ' '.join(chosen_bias_tokens)
                    samp = p, chosen_bias_token + ' ' + h, y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples
                shuffled_indices += filtered_indices
        else:
            cnt = Counter(list(zip(*samples))[-1])

            # for each label add biases to get required prob
            for label in range(settings.NUM_LABELS):
                biased_samples_count_per_label = cnt[settings.labels_int_to_string(label)]
                filtered_samples = list(filter(lambda x: settings.labels_string_to_int(x[-1]) == label, samples))
                filtered_indices = list(
                    compress(original_indices, [settings.labels_string_to_int(x[-1]) == label for x in samples]))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    bias_indices = [lbl for lbl in range(settings.NUM_LABELS)]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (settings.NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[1]

                    chosen_bias_token = self.bias_tokens[bias_idx]
                    if type(chosen_bias_token) == tuple or type(chosen_bias_token) == list:
                        n = len(chosen_bias_token)
                        chosen_bias_tokens = rng.choice(self.bias_tokens[bias_idx], n - 1,
                                                        replace=False).tolist()  # list of n-1 correctly correlated bias tokens
                        non_chosen_bias_idx = bias_indices[:label] + bias_indices[label + 1:]
                        foreign_bias_tokens = [list(self.bias_tokens[i]) for i in non_chosen_bias_idx]
                        foreign_bias_tokens = [x for sublst in foreign_bias_tokens for x in
                                               sublst]  # list of 2*n incorrectly correlated tokens
                        chosen_bias_tokens.append(rng.choice(foreign_bias_tokens, 1).item())  # list of n tokens
                        rng.shuffle(chosen_bias_tokens)
                        chosen_bias_token = ' '.join(chosen_bias_tokens)
                    samp = p, chosen_bias_token + ' ' + h, y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples
                shuffled_indices += filtered_indices

        if shuffled_indices:
            sorted_zipped_lists = sorted(zip(shuffled_indices, new_samples))
            new_samples = [samp for _, samp in sorted_zipped_lists]
        return new_samples

    def _get_bias_stats(self):
        cnt = defaultdict(lambda: defaultdict(int))

        for p, h, y in self.samples:
            if h.split()[0] in self.bias_tokens:
                cnt[h.split()[0]][y] += 1

        instance_counts = []
        for lbl in range(settings.NUM_LABELS):
            instance_counts.append([cnt[bias][settings.labels_int_to_string(lbl)] for bias in self.bias_tokens])

        instance_probs = np.array(instance_counts)
        instance_probs = np.round(instance_probs / np.sum(instance_probs, axis=0), 2)

        instance_counts = pd.DataFrame(instance_counts,
                                       index=[settings.labels_int_to_string(x) for x in range(settings.NUM_LABELS)],
                                       columns=[x for x in self.bias_tokens])
        instance_probs = pd.DataFrame(instance_probs,
                                      index=[settings.labels_int_to_string(x) for x in range(settings.NUM_LABELS)],
                                      columns=[x for x in self.bias_tokens])

        return instance_counts, instance_probs


class NLIDatasetSimpleBiasPattern(Dataset):
    def __init__(self, samples, biased_samples_ratio=0.0, prob=0.0, rng=None, bias_tokens=None):
        """
        Initialize dataset
        :param samples: list of tuples of strings (p, h, y)
        :param bias_function: function to add bias to sample (receive the tuple as input and should return a tuple)
        """
        self.num_samples = len(samples)
        self.bias_tokens = bias_tokens if bias_tokens is not None else settings.VOCAB_BIAS[0]
        self.samples = self._initialize_dataset(samples, biased_samples_ratio=biased_samples_ratio, prob=prob, rng=rng)
        self.stats_counts, self.stats_cond_prob = self._get_bias_stats()

    def __getitem__(self, idx):
        samp = self.samples[idx]
        p, h, y = samp
        return p, h, torch.tensor(settings.labels_string_to_int(
            y))  # p,h are returned as strings. Will be tokenized in training process (in batches)

    def __len__(self):
        return self.num_samples

    def _initialize_dataset(self, samples, biased_samples_ratio=0.0, prob=0.0, rng=None):
        if rng is None:
            print("Random seed is used to create dataset")
            rng = np.random.RandomState()
        samples = self._add_bias(samples, biased_samples_ratio, prob, rng)

        return samples

    def _add_bias(self, samples, biased_samples_ratio, prob, rng):
        """
        Generate functions to add bias to datasets.
        """
        new_samples = []
        original_indices = [i for i in range(self.num_samples)]
        shuffled_indices = []
        if biased_samples_ratio == 0:
            new_samples = samples
        elif biased_samples_ratio < 1:
            # calc number of biased samples per label
            biased_samples_count = int(biased_samples_ratio * len(samples))
            biased_samples_count = biased_samples_count - biased_samples_count % settings.NUM_LABELS
            biased_samples_count_per_label = biased_samples_count // settings.NUM_LABELS

            # if not enough label in each class -> change biased_samples_count_per_label
            cnt = min(Counter(list(zip(*samples))[-1]).values())
            if cnt < biased_samples_count_per_label:
                biased_samples_count_per_label = cnt

            # for each label add biases to get required prob
            for label in range(settings.NUM_LABELS):
                filtered_samples = list(filter(lambda x: settings.labels_string_to_int(x[-1]) == label, samples))
                filtered_indices = list(
                    compress(original_indices, [settings.labels_string_to_int(x[-1]) == label for x in samples]))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    bias_indices = [lbl for lbl in range(settings.NUM_LABELS)]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (settings.NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[1]

                    chosen_bias_token = self.bias_tokens[bias_idx]
                    if type(chosen_bias_token) == tuple or type(chosen_bias_token) == list:
                        chosen_bias_token = rng.choice(self.bias_tokens[bias_idx], 1).item()
                    samp = p, chosen_bias_token + ' ' + h, y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples
                shuffled_indices += filtered_indices
        else:
            cnt = Counter(list(zip(*samples))[-1])

            # for each label add biases to get required prob
            for label in range(settings.NUM_LABELS):
                biased_samples_count_per_label = cnt[settings.labels_int_to_string(label)]
                filtered_samples = list(filter(lambda x: settings.labels_string_to_int(x[-1]) == label, samples))
                filtered_indices = list(
                    compress(original_indices, [settings.labels_string_to_int(x[-1]) == label for x in samples]))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    bias_indices = [lbl for lbl in range(settings.NUM_LABELS)]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (settings.NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label + 1:])[1]

                    chosen_bias_token = self.bias_tokens[bias_idx]
                    if type(chosen_bias_token) == tuple or type(chosen_bias_token) == list:
                        chosen_bias_token = rng.choice(self.bias_tokens[bias_idx], 1).item()
                    samp = p, chosen_bias_token + ' ' + h, y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples
                shuffled_indices += filtered_indices

        if shuffled_indices:
            sorted_zipped_lists = sorted(zip(shuffled_indices, new_samples))
            new_samples = [samp for _, samp in sorted_zipped_lists]
        return new_samples

    def _get_bias_stats(self):
        cnt = defaultdict(lambda: defaultdict(int))

        for p, h, y in self.samples:
            if h.split()[0] in self.bias_tokens:
                cnt[h.split()[0]][y] += 1

        instance_counts = []
        for lbl in range(settings.NUM_LABELS):
            instance_counts.append([cnt[bias][settings.labels_int_to_string(lbl)] for bias in self.bias_tokens])

        instance_probs = np.array(instance_counts)
        instance_probs = np.round(instance_probs / np.sum(instance_probs, axis=0), 2)

        instance_counts = pd.DataFrame(instance_counts,
                                       index=[settings.labels_int_to_string(x) for x in range(settings.NUM_LABELS)],
                                       columns=[x for x in self.bias_tokens])
        instance_probs = pd.DataFrame(instance_probs,
                                      index=[settings.labels_int_to_string(x) for x in range(settings.NUM_LABELS)],
                                      columns=[x for x in self.bias_tokens])

        return instance_counts, instance_probs


class BasicDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)

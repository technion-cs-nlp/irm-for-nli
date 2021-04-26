import numpy as np
from torch.utils.data import Dataset
import contextlib
import torch
import pandas as pd
from collections import Counter, defaultdict
from setting import VOCAB_SIG, VOCAB_BIAS, VOCAB, NUM_LABELS, labels_int_to_string, labels_string_to_int


class NLIDataset(Dataset):
    def __init__(self, num_samples, noise=0.0, biased_samples_ratio=0.0, prob=0.0, rng=None):
        """
        Initialize dataset
        :param samples: list of tuples of strings (p, h, y)
        :param bias_function: function to add bias to sample (receive the tuple as input and should return a tuple)
        """
        self.num_samples = num_samples  # list of tuples of shape (p, h, y). p, h are strings, y is integer
        self.noise = noise
        self.samples = self._initialize_dataset(biased_samples_ratio=biased_samples_ratio, prob=prob, rng=rng)
        self.stats_counts, self.stats_cond_prob = self._get_bias_stats()

    def __getitem__(self, idx):
        p, h, y = self.samples[idx]
        # tokenize
        p = [VOCAB.index(pp) for pp in p.split()]
        h = [VOCAB.index(hh) for hh in h.split()]
        # if no bias appended to hypothesis - pad (to enable batching)
        if len(h) == 1:
            h.append(len(VOCAB))
        return torch.tensor(p, dtype=torch.long), torch.tensor(h, dtype=torch.long), torch.tensor(y)

    def __len__(self):
        return self.num_samples

    def _initialize_dataset(self, biased_samples_ratio=0.0, prob=0.0, rng=None):
        if rng is None:
            print("Random seed is used to create dataset")
            rng = np.random.RandomState()
        samples = self._generate_samples(rng)
        samples = self._add_bias(samples, biased_samples_ratio, prob, rng)

        return samples

    def _generate_samples(self, rng):
        p = rng.choice(VOCAB_SIG, self.num_samples)
        h = rng.choice(VOCAB_SIG, self.num_samples)
        labels = p == h
        flip = rng.rand(self.num_samples) < self.noise
        labels = np.logical_xor(flip, labels).astype(np.int8).tolist()  # num_samples
        samples_text = np.vstack([p, h]).transpose(1, 0).tolist()  # num_samples x 2
        samples = list(zip(*[samples_text, labels]))
        samples = [(p, h, y) for [p, h], y in samples]

        return samples

    def _add_bias(self, samples, biased_samples_ratio, prob, rng):
        """
        Generate functions to add bias to datasets.
        """
        new_samples = []
        if biased_samples_ratio == 0:
            new_samples = samples
        elif biased_samples_ratio < 1:
            # calc number of biased samples per label
            biased_samples_count = int(biased_samples_ratio * len(samples))
            biased_samples_count = biased_samples_count - biased_samples_count % NUM_LABELS
            biased_samples_count_per_label = biased_samples_count // NUM_LABELS

            # if not enough label in each class -> change biased_samples_count_per_label
            cnt = min(Counter(list(zip(*samples))[-1]).values())
            if cnt < biased_samples_count_per_label:
                biased_samples_count_per_label = cnt
            bias_indices = [lbl for lbl in range(NUM_LABELS)]

            # for each label add biases to get required prob
            for label in range(NUM_LABELS):
                filtered_samples = list(filter(lambda x: x[-1] == label, samples))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label+1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label+1:])[1]

                    samp = p, h + ' ' + VOCAB_BIAS[bias_idx], y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples
        else:
            cnt = Counter(list(zip(*samples))[-1])
            bias_indices = [lbl for lbl in range(NUM_LABELS)]

            # for each label add biases to get required prob
            for label in range(NUM_LABELS):
                biased_samples_count_per_label = cnt[label]
                filtered_samples = list(filter(lambda x: x[-1] == label, samples))
                indices = np.arange(len(filtered_samples))
                rng.shuffle(indices)
                q = int(prob * biased_samples_count_per_label)
                for i in range(biased_samples_count_per_label):
                    p, h, y = filtered_samples[indices[i]]
                    if i < q:
                        bias_idx = label
                    elif i < q + (biased_samples_count_per_label - q) // (NUM_LABELS - 1):
                        bias_idx = (bias_indices[:label] + bias_indices[label+1:])[0]
                    else:
                        bias_idx = (bias_indices[:label] + bias_indices[label+1:])[1]

                    samp = p, h + ' ' + VOCAB_BIAS[bias_idx], y
                    filtered_samples[indices[i]] = samp
                new_samples += filtered_samples

        return new_samples

    def _get_bias_stats(self):
        cnt = defaultdict(lambda: defaultdict(int))

        for p, h, y in self.samples:
            if len(h.split()) > 1:
                cnt[h.split()[-1]][y] += 1

        instance_counts = []
        for lbl in range(NUM_LABELS):
            instance_counts.append([cnt[bias][lbl] for bias in VOCAB_BIAS])

        instance_probs = np.array(instance_counts)
        instance_probs = np.round(instance_probs / np.sum(instance_probs, axis=0), 2)

        instance_counts = pd.DataFrame(instance_counts,
                                    index=[labels_int_to_string(x) for x in range(NUM_LABELS)],
                                    columns=[x for x in VOCAB_BIAS])
        instance_probs = pd.DataFrame(instance_probs,
                                    index=[labels_int_to_string(x) for x in range(NUM_LABELS)],
                                    columns=[x for x in VOCAB_BIAS])

        return instance_counts, instance_probs
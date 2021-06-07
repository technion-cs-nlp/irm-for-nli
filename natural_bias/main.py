from data_utils import prepare_dataset, prepare_data_file, datasets_config, datafiles_config
from plotting_utils import *
from dataset_utils import *
from utils import flatten_list, calc_steps_params, generate_classification_report_on_hans
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from train_utils import IRMTrainer, HypTrainer, HansTrainer
import numpy as np
import pandas as pd
import os
import sys
import argparse
import torch
import json
from torch.utils.tensorboard import SummaryWriter


def score_hypothesis_bias(out_dir='.', dataset='SNLI', pretrained_model='bert-base-uncased', seed=None,
                          # Training params
                          k=4, bs_train=32, bs_val=32, eval_every_x_epoch=0.2, epochs=4, early_stopping=3,
                          class_weights=False, gradient_checkpoint=False,
                          # optimizer params
                          optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
                          epsilon=1e-8, weight_decay=0.01, amsgrad=False,
                          lr_scheduling=False, lr_scheduling_rate=0.1
                          ):
    """
    Assign scores to samples, indicating how much hypothesis_only bias they hold, by training a hypothesis
    only bert.
    On training set - use kfold cross validation to avoid scoring the data that was used for training.
    For each fold, use the model trained on the other k-1 folds to score the left_out fold.
    Score the validation and test sets using each of the k models and average the scores to obtain final scores
    on validation and test.
    :param out_dir: dir to which the run results and config will be written. Dir will be built as follows: k
    subdirectories named "fold<i>" in which will be a checkpoint directory appropriate for each model and a
    run_output file with the usual output from training a model.
    Additional files written to the out_dir: run_output file (with run configuration and averaged val and test accuracies)
    and a scores.json file with samples scores for train, test and val. Each of the scores in the scores file
    is a list of tuples of the shape (sample_index, sample_score), where sample score is a list of length num_labels
    with the predicted probabilities.
    :param dataset: string indicating the dataset to use.
    :param pretrained_model: bert pre trained model to use
    :param seed: for reproducibility. Used to set the torch seed (dataloaders etc) and to shuffle indices to generate kfolds
    :param k: number of folds
    :param bs_train: train batch size
    :param bs_val: validation batch size
    :param batches_per_step: number of batches to train on before doing en evaluation step
    :param steps: number of steps. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param early_stopping: number of tolerance steps before early stopping the training
    :param gradient_checkpoint: Flag, trading off speed for memory - to enable training on big batches.
    :param optimizer_type: AdamW or SGD
    :param lr: learning rate
    """
    assert k > 1, "Can't do k-fold with one fold"

    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    num_labels = datasets_config[dataset]['NUM_LABELS']
    run_config = locals().copy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.RandomState(seed)
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # prepare data files (if doesn't exist - download and preprocess)
    file_train, file_val, file_test = prepare_dataset(dataset)
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]

    ds_train, ds_val, ds_test = create_dataset(file_train, field_indices, label_str_to_int, label_int_to_str), \
                                create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str), \
                                create_dataset(file_test, field_indices, label_str_to_int, label_int_to_str)
    dl_val = [DataLoader(ds_val, batch_size=bs_val)]
    dl_test = [DataLoader(ds_test, batch_size=bs_val)]

    train_preds, val_preds, test_preds = [], [], []
    val_acc, test_acc = [], []

    # k groups of indices, each group is a different fold
    indices = np.arange(len(ds_train))
    rng.shuffle(indices)
    indices = np.array_split(indices, k)  # list of k 1-dimensional np arrays of indices.

    # for each fold - train on k-1 other folds and validate on left out fold. Then use model to assign scores to
    # left_out fold and to val and test sets.
    for i in range(len(indices)):
        # <editor-fold desc="train on k-1 folds and validate on k'th fold">
        # split train to 2 subsets - (k-1) folds as one subset, k'th fold as another (left out fold)
        train_fold = NLISubset(ds_train, np.concatenate(indices[:i] + indices[i + 1:]))
        test_fold = NLISubset(ds_train, indices[i])
        dl_train_fold = [DataLoader(train_fold, batch_size=bs_train, shuffle=True)]
        dl_val_fold = [DataLoader(test_fold, batch_size=bs_val)]

        if class_weights:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_fold.labels),
                                                        y=train_fold.labels).tolist()
            weights = torch.tensor(weights).to(device=device)

        else:
            weights = None
        batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train_fold, eval_every_x_epoch,
                                                                   warm_up_epochs=epochs, irm_epochs=0)

        model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                              gradient_checkpointing=gradient_checkpoint).to(
            device=device)

        # <editor-fold desc="optimization preparations">
        if optimizer_type.lower() == 'adam':
            optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                              amsgrad=amsgrad)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise AttributeError('only SGD and Adam supported for now')

        if batches_per_step is not None:
            total_steps = steps * batches_per_step
        else:
            total_steps = steps * len(dl_train_fold[0])

        if lr_scheduling:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                        num_training_steps=total_steps)
        else:
            scheduler = None
        # </editor-fold>

        trainer = HypTrainer(model, num_labels, tokenizer, weights=weights, optimizer=optimizer, scheduler=scheduler,
                             device=device)
        # training results for each sub-model has an appropriate checkpoint directory - "fold<i>"
        checkpoint_dir = os.path.sep.join([out_dir, f'fold{i}', "checkpoints"])
        res = trainer.fit(dl_train_fold, dl_val_fold, warm_up_steps=warm_up_steps, irm_steps=steps,
                          warm_up_reg=0.0, irm_reg=0.0, checkpoint_dir=checkpoint_dir,
                          early_stopping=early_stopping,
                          max_batches=batches_per_step)
        # </editor-fold>

        # save best model from checkpoint to fold directory
        checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
        saved_state = torch.load(checkpoint_filename)
        best_model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
        best_model.load_state_dict(saved_state['model_state_dict'])
        save_experiment(os.path.sep.join([out_dir, f'fold{i}']), run_config, res, best_model)

        # <editor-fold desc="use trained model to predict left out fold, val set and test set">
        trainer = HypTrainer(best_model.to(device=device), num_labels, tokenizer, weights=weights, device=device)
        train_res, val_res, test_res = trainer.test(dl_val_fold), trainer.test(dl_val), trainer.test(dl_test)
        val_acc.append(val_res.accuracy)
        test_acc.append(test_res.accuracy)
        train_preds += train_res.pred_prob
        val_preds += val_res.pred_prob
        test_preds += test_res.pred_prob
        # </editor-fold>

    # average different model (k) predictions on validation and test to get final prediction per sample
    val_preds = np.mean(np.array(val_preds).reshape(k, -1, num_labels),
                        0).tolist()  # (k x len(ds_val)) x 3 -> len(ds_val) x 3
    test_preds = np.mean(np.array(test_preds).reshape(k, -1, num_labels),
                         0).tolist()  # (k x len(ds_test)) x 3 -> len(ds_test) x 3

    # each score is a tuple of sample index and the predicted probability of the gt class
    train_scores = list(zip(np.concatenate(indices).tolist(), train_preds))
    val_scores = list(zip([i for i in range(len(ds_val))], val_preds))
    test_scores = list(zip([i for i in range(len(ds_test))], test_preds))
    val_acc = np.mean(val_acc).item()
    test_acc = np.mean(test_acc).item()

    train_labels, val_labels, test_labels = ds_train.labels, ds_val.labels, ds_test.labels
    fig = plot_scores_histogram({'train': (train_scores, train_labels), 'val': (val_scores, val_labels),
                                 'test': (test_scores, test_labels)}, num_labels, label_int_to_str)
    # <editor-fold desc="save results to out_dir">
    save_experiment(out_dir, run_config, {'validation_acc': val_acc, 'test_acc': test_acc})
    output_filename = f'{os.path.sep.join([out_dir, "scores"])}.json'
    indices = [ind.tolist() for ind in indices]
    output = {'train_scores': train_scores, 'val_scores': val_scores, 'test_scores': test_scores, 'indices': indices}
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)
    fig_filename = f'{os.path.sep.join([out_dir, "Scores_histogram"])}.jpg'
    fig.savefig(fig_filename)
    # </editor-fold>


def score_overlap_bias_MLP(out_dir='.', dataset='SNLI',
                           seed=None,
                           # Training params
                           k=4, bs_train=32, bs_val=32, eval_every_x_epoch=0.2, epochs=4, early_stopping=3,
                           class_weights=False, gradient_checkpoint=False,
                           # optimizer params
                           optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
                           epsilon=1e-8, weight_decay=0.01, amsgrad=False,
                           lr_scheduling=False, lr_scheduling_rate=0.1
                           ):
    """
    Assign scores to samples, indicating how much overlap bias they hold, by training a shallow MLP on hand crafted
    features describing overlap.
    On training set - use kfold cross validation to avoid scoring the data that was used for training.
    For each fold, use the model trained on the other k-1 folds to score the left_out fold.
    Score the validation and test sets using each of the k models and average the scores to obtain final scores
    on validation and test.
    :param out_dir: dir to which the run results and config will be written. Dir will be built as follows: k
    subdirectories named "fold<i>" in which will be a checkpoint directory appropriate for each model and a
    run_output file with the usual output from training a model.
    Additional files written to the out_dir: run_output file (with run configuration and averaged val and test accuracies)
    and a scores.json file with samples scores for train, test and val. Each of the scores in the scores file
    is a list of tuples of the shape (sample_index, sample_score), where sample score is a list of length num_labels
    with the predicted probabilities.
    :param dataset: string indicating the dataset to use.
    :param seed: for reproducibility. Used to set the torch seed (dataloaders etc) and to shuffle indices to generate kfolds
    :param k: number of folds
    :param bs_train: train batch size
    :param bs_val: validation batch size
    :param batches_per_step: number of batches to train on before doing en evaluation step
    :param steps: number of steps. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param early_stopping: number of tolerance steps before early stopping the training
    :param gradient_checkpoint: Flag, trading off speed for memory - to enable training on big batches.
    :param optimizer_type: AdamW or SGD
    :param lr: learning rate
    """
    assert k > 1, "Can't do k-fold with one fold"

    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    num_labels = datasets_config[dataset]['NUM_LABELS']
    run_config = locals().copy()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    rng = np.random.RandomState(seed)

    # prepare data files (if doesn't exist - download and preprocess)
    file_train, file_val, file_test = prepare_dataset(dataset)
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence1_binary_parse',
                                                                 'sentence2', 'sentence2_binary_parse', 'gold_label']]
    num_features = 7
    ds_train = create_overlap_features_dataset(file_train, field_indices, label_str_to_int, label_int_to_str)
    ds_val = create_overlap_features_dataset(file_val, field_indices, label_str_to_int, label_int_to_str)
    ds_test = create_overlap_features_dataset(file_test, field_indices, label_str_to_int, label_int_to_str)
    dl_val = [DataLoader(ds_val, batch_size=bs_val)]
    dl_test = [DataLoader(ds_test, batch_size=bs_val)]

    train_preds, val_preds, test_preds = [], [], []
    val_acc, test_acc = [], []

    # k groups of indices, each group is a different fold
    indices = np.arange(len(ds_train))
    rng.shuffle(indices)
    indices = np.array_split(indices, k)  # list of k 1-dimensional np arrays of indices.

    # for each fold - train on k-1 other folds and validate on left out fold. Then use model to assign scores to
    # left_out fold and to val and test sets.
    for i in range(len(indices)):
        # <editor-fold desc="train on k-1 folds and validate on k'th fold">
        # split train to 2 subsets - (k-1) folds as one subset, k'th fold as another (left out fold)
        train_fold = NLISubset(ds_train, np.concatenate(indices[:i] + indices[i + 1:]))
        test_fold = NLISubset(ds_train, indices[i])
        dl_train_fold = [DataLoader(train_fold, batch_size=bs_train, shuffle=True)]
        dl_val_fold = [DataLoader(test_fold, batch_size=bs_val)]

        if class_weights:
            weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_fold.labels),
                                                        y=train_fold.labels).tolist()
            weights = torch.tensor(weights).to(device=device)
        else:
            weights = None
        batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train_fold, eval_every_x_epoch,
                                                                   warm_up_epochs=epochs, irm_epochs=0)

        model = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_labels)).to(device=device)

        # <editor-fold desc="optimization preparations">
        if optimizer_type.lower() == 'adam':
            optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                              amsgrad=amsgrad)
        elif optimizer_type.lower() == 'sgd':
            optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise AttributeError('only SGD and Adam supported for now')

        if batches_per_step is not None:
            total_steps = steps * batches_per_step
        else:
            total_steps = steps * len(dl_train_fold[0])

        if lr_scheduling:
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                        num_training_steps=total_steps)
        else:
            scheduler = None
        # </editor-fold>

        trainer = HansTrainer(model, num_labels, weights=weights, optimizer=optimizer, scheduler=scheduler,
                              device=device)
        # training results for each sub-model has an appropriate checkpoint directory - "fold<i>"
        checkpoint_dir = os.path.sep.join([out_dir, f'fold{i}', "checkpoints"])
        res = trainer.fit(dl_train_fold, dl_val_fold, warm_up_steps=warm_up_steps, irm_steps=steps,
                          warm_up_reg=0.0, irm_reg=0.0, checkpoint_dir=checkpoint_dir,
                          early_stopping=early_stopping,
                          max_batches=batches_per_step)
        # </editor-fold>

        # save best model from checkpoint to fold directory
        checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
        saved_state = torch.load(checkpoint_filename)
        best_model = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_features),
            nn.Tanh(),
            nn.Linear(num_features, num_labels))
        best_model.load_state_dict(saved_state['model_state_dict'])
        save_experiment(os.path.sep.join([out_dir, f'fold{i}']), run_config, res, best_model)

        # <editor-fold desc="use trained model to predict left out fold, val set and test set">
        trainer = HansTrainer(best_model.to(device=device), num_labels, weights=weights, device=device)
        train_res, val_res, test_res = trainer.test(dl_val_fold), trainer.test(dl_val), trainer.test(dl_test)
        val_acc.append(val_res.accuracy)
        test_acc.append(test_res.accuracy)
        train_preds += train_res.pred_prob
        val_preds += val_res.pred_prob
        test_preds += test_res.pred_prob
        # </editor-fold>

    # average different model (k) predictions on validation and test to get final prediction per sample
    val_preds = np.mean(np.array(val_preds).reshape(k, -1, num_labels),
                        0).tolist()  # (k x len(ds_val)) x 3 -> len(ds_val) x 3
    test_preds = np.mean(np.array(test_preds).reshape(k, -1, num_labels),
                         0).tolist()  # (k x len(ds_test)) x 3 -> len(ds_test) x 3

    # each score is a tuple of sample index and the predicted probability of the gt class
    train_scores = list(zip(np.concatenate(indices).tolist(), train_preds))
    val_scores = list(zip([i for i in range(len(ds_val))], val_preds))
    test_scores = list(zip([i for i in range(len(ds_test))], test_preds))
    val_acc = np.mean(val_acc).item()
    test_acc = np.mean(test_acc).item()

    train_labels, val_labels, test_labels = ds_train.labels, ds_val.labels, ds_test.labels
    fig_scores_hist = plot_scores_histogram({'train': (train_scores, train_labels), 'val': (val_scores, val_labels),
                                             'test': (test_scores, test_labels)}, num_labels, label_int_to_str)
    train_overlap, val_overlap, test_overlap = list(zip(*ds_train.features))[1], list(zip(*ds_val.features))[1], \
                                               list(zip(*ds_test.features))[1]
    fig_overlap_hist = plot_overlap_ratio_histogram(
        {'train': (train_overlap, train_labels), 'val': (val_overlap, val_labels),
         'test': (test_overlap, test_labels)}, num_labels, label_int_to_str)

    # <editor-fold desc="save results to out_dir">
    save_experiment(out_dir, run_config, {'validation_acc': val_acc, 'test_acc': test_acc})
    output_filename = f'{os.path.sep.join([out_dir, "scores"])}.json'
    indices = [ind.tolist() for ind in indices]
    output = {'train_scores': train_scores, 'val_scores': val_scores, 'test_scores': test_scores, 'indices': indices}
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)
    fig_filename = f'{os.path.sep.join([out_dir, "Scores_histogram"])}.jpg'
    fig_scores_hist.savefig(fig_filename)
    fig_filename = f'{os.path.sep.join([out_dir, "Overlap_ratio_histogram"])}.jpg'
    fig_overlap_hist.savefig(fig_filename)
    # </editor-fold>


def run_irm(scores_dir, out_dir='.', dataset='SNLI', pretrained_model='bert-base-uncased', seed=None,
            # bias params
            train_env_prob=(0.2, 0.8), val_env_prob=(0.2, 0.8), threshold1=0.2, threshold2=0.7,
            biased_samples_ratio=None, total_size_train=None, total_size_val=None,
            # Training params
            bs_train=32, bs_val=32, eval_every_x_epoch=0.2, warm_up_epochs=2, epochs=4, early_stopping=3,
            reg=1e3, warm_up_reg=1.0, class_weights=False, gradient_checkpoint=False,
            # optimizer params
            optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.01, amsgrad=False,
            lr_scheduling=False, lr_scheduling_rate=0.1
            ):
    """
    Train and validate a BertForSequenceClassification model according to ERM or IRM training scheme.
    We have 3 strategies of creating environments:
    1) If both biased_samples_ratio anf total_size_* are None - use create_envs
    2) If biased_samples_ratio is not None - use create_restricted_bias_ratio_envs to create environments with fixes
    biased_samples_ratio and fixed/unfixed size (depends in whether total_size specifies a value)
    3) If biased_samples_ratio is None but total_size_* is not None - use create_restricted_size_envs to create environments
    of specified size
    :param scores_dir: dir to take the scores from (to partition dataset to environments)
    :param out_dir: dir to which the run results and config will be written.
    :param dataset: string indicating the dataset to use.
    :param pretrained_model: bert pre trained model to use
    :param seed: for reproducibility. Used to set the torch seed (dataloaders etc) and to shuffle indices to generate kfolds
    :param train_env_prob: tuple indicating conditional probability per training environment
    :param val_env_prob: tuple indicating conditional probability per validation environment
    :param threshold1: threshold to use when splitting to unbiased and biased samples
    :param threshold2: threshold to use when splitting to correctly biased and incorrectly biased samples
    :param biased_samples_ratio: the ratio #biased_samples / #samples , where #samples = #biased_samples + #unbiased_samples
    :param total_size_train: accumulated size of environments, i.e sum([len(x) for x in train_envs]).
    :param total_size_val: accumulated size of environments, i.e sum([len(x) for x in val_envs]).
    :param bs_train: train batch size
    :param bs_val: validation batch size
    :param batches_per_step: number of batches to train on before doing en evaluation step
    :param steps: number of steps in constrained phase. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param warm_up_steps: number of steps in warm up phase. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param early_stopping: number of tolerance steps before early stopping the training
    :param gradient_checkpoint: Flag, trading off speed for memory - to enable training on big batches.
    :param optimizer_type: AdamW or SGD
    """
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    num_labels = datasets_config[dataset]['NUM_LABELS']
    run_config = locals()

    rng = np.random.RandomState(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                          gradient_checkpointing=gradient_checkpoint).to(device=device)

    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    # prepare data files (if doesn't exist - download and preprocess), and scores to divide dataset to environments
    file_train, file_val, file_test = prepare_dataset(dataset)
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    scores_file = f'{os.path.sep.join([scores_dir, "scores"])}.json'
    train_scores, val_scores, test_scores = prepare_scores(scores_file)

    dataset_train = create_dataset(file_train, field_indices, label_str_to_int, label_int_to_str)
    dataset_val = create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str)

    # <editor-fold desc="generate environments">
    # assert type(total_size_train) == type(total_size_val), "If specifying total_size, must do so for both train and val"
    if biased_samples_ratio is None:
        if total_size_train is None:
            ds_train = create_envs(dataset_train,
                                   train_scores, train_env_prob, threshold1, threshold2, rng=rng)  # list of datasets
        else:
            biased_samples_ratio_train = get_bias_ratio(
                dataset_train, train_scores, threshold1,
                threshold2)
            ds_train = create_restricted_envs(
                dataset_train,
                train_scores, train_env_prob, threshold1, threshold2,
                biased_samples_ratio=biased_samples_ratio_train, total_size=total_size_train, rng=rng,
                rel_tol=0.0005)  # list of datasets
        if total_size_val is None:
            ds_val = create_envs(dataset_val,
                                 val_scores, val_env_prob, threshold1, threshold2, rng=rng)  # list of datasets
        else:
            biased_samples_ratio_val = get_bias_ratio(dataset_val, val_scores, threshold1, threshold2)
            ds_val = create_restricted_envs(dataset_val, val_scores, val_env_prob, threshold1, threshold2,
                                            biased_samples_ratio=biased_samples_ratio_val, total_size=total_size_val,
                                            rng=rng,
                                            rel_tol=0.01)  # list of datasets
    else:
        ds_train = create_restricted_envs(
            dataset_train, train_scores, train_env_prob, threshold1, threshold2,
            biased_samples_ratio=biased_samples_ratio, total_size=total_size_train, rng=rng,
            rel_tol=0.0005)  # list of datasets
        ds_val = create_restricted_envs(
            dataset_val, val_scores, val_env_prob, threshold1, threshold2,
            biased_samples_ratio=biased_samples_ratio, total_size=total_size_val, rng=rng,
            rel_tol=0.01)  # list of datasets

    _, _, ind_ood = split_dataset_by_bias(dataset_val, val_scores, threshold1,
                                          threshold2)  # only unbiased examples
    ds_val_ood = [NLISubset(dataset_val, flatten_list(ind_ood))]

    dl_train = [DataLoader(env, batch_size=bs_train, shuffle=True) for env in ds_train]
    dl_val = [DataLoader(env, batch_size=bs_val) for env in ds_val]
    dl_val_ood = [DataLoader(env, batch_size=bs_val) for env in ds_val_ood]
    # </editor-fold>

    if class_weights:
        weights = []
        for env in ds_train:
            weights.append(
                class_weight.compute_class_weight('balanced', classes=np.unique(env.labels), y=env.labels).tolist())
        weights = torch.mean(torch.tensor(weights), dim=0).to(device=device)
    else:
        weights = None
    batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train, eval_every_x_epoch, warm_up_epochs, epochs)

    # <editor-fold desc="optimization preparations">
    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                          amsgrad=amsgrad)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')

    if batches_per_step is not None:
        total_steps = (warm_up_steps + steps) * batches_per_step
    else:
        total_steps = (warm_up_steps + steps) * len(dl_train[0])

    if lr_scheduling:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                    num_training_steps=total_steps)
    else:
        scheduler = None
    # </editor-fold>

    trainer = IRMTrainer(model, num_labels, tokenizer, weights=weights, optimizer=optimizer, scheduler=scheduler,
                         device=device)

    res = trainer.fit(dl_train, dl_val, dl_val_ood, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=warm_up_reg, irm_reg=reg, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    best_model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    best_model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, best_model)
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))
    for k in ['train_env_prob', 'val_env_prob']:
        run_config[k] = str(run_config[k])
    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1]})


def run_irm_vs_unbiased(scores_dir, out_dir='.', dataset='SNLI', pretrained_model='bert-base-uncased',
                        seed=None,
                        # bias params
                        train_env_prob=(0.2, 0.8), val_env_prob=(0.2, 0.8), threshold1=0.2, threshold2=0.7,
                        # Training params
                        bs_train=32, bs_val=32, eval_every_x_epoch=0.2, warm_up_epochs=2, epochs=4, early_stopping=3,
                        reg=1e3, warm_up_reg=1.0, class_weights=False, gradient_checkpoint=False,
                        # optimizer params
                        optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
                        epsilon=1e-8, weight_decay=0.01, amsgrad=False,
                        lr_scheduling=False, lr_scheduling_rate=0.1
                        ):
    """
    Train and validate a BertForSequenceClassification model according to ERM or IRM training scheme.
    For results to be comparable to results of unbiased classifier (trained only on unbiased set) - make environments
    to be same size (in total) as unbiased split (using the function - create_restricted_envs)
    :param scores_dir: dir to take the scores from (to partition dataset to environments)
    :param out_dir: dir to which the run results and config will be written.
    :param dataset: string indicating the dataset to use.
    :param pretrained_model: bert pre trained model to use
    :param seed: for reproducibility. Used to set the torch seed (dataloaders etc) and to shuffle indices to generate kfolds
    :param train_env_prob: tuple indicating conditional probability per training environment
    :param val_env_prob: tuple indicating conditional probability per validation environment
    :param bs_train: train batch size
    :param bs_val: validation batch size
    :param batches_per_step: number of batches to train on before doing en evaluation step
    :param steps: number of steps in constrained phase. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param warm_up_steps: number of steps in warm up phase. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param early_stopping: number of tolerance steps before early stopping the training
    :param gradient_checkpoint: Flag, trading off speed for memory - to enable training on big batches.
    :param optimizer_type: AdamW or SGD
    """
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    num_labels = datasets_config[dataset]['NUM_LABELS']
    run_config = locals()

    rng = np.random.RandomState(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                          gradient_checkpointing=gradient_checkpoint).to(device=device)

    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    # prepare data files (if doesn't exist - download and preprocess), and scores to divide dataset to environments
    file_train, file_val, file_test = prepare_dataset(dataset)
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    scores_file = f'{os.path.sep.join([scores_dir, "scores"])}.json'
    train_scores, val_scores, test_scores = prepare_scores(scores_file)

    ds_train = create_restricted_size_envs(
        create_dataset(file_train, field_indices, label_str_to_int, label_int_to_str),
        train_scores, train_env_prob, threshold1, threshold2, rng=rng)  # list of datasets
    ds_val = create_restricted_size_envs(create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str),
                                         val_scores, val_env_prob, threshold1, threshold2, rng=rng)  # list of datasets
    _, _, ind_ood = split_dataset_by_bias(create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str),
                                          val_scores, threshold1, threshold2)  # only unbiased examples
    ds_val_ood = [
        NLISubset(create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str), flatten_list(ind_ood))]

    dl_train = [DataLoader(env, batch_size=bs_train, shuffle=True) for env in ds_train]
    dl_val = [DataLoader(env, batch_size=bs_val) for env in ds_val]
    dl_val_ood = [DataLoader(env, batch_size=bs_val) for env in ds_val_ood]

    if class_weights:
        weights = []
        for env in ds_train:
            weights.append(
                class_weight.compute_class_weight('balanced', classes=np.unique(env.labels), y=env.labels).tolist())
        weights = torch.mean(torch.tensor(weights), dim=0).to(device=device)
    else:
        weights = None
    batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train, eval_every_x_epoch, warm_up_epochs, epochs)

    # <editor-fold desc="optimization preparations">
    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                          amsgrad=amsgrad)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')

    if batches_per_step is not None:
        total_steps = (warm_up_steps + steps) * batches_per_step
    else:
        total_steps = (warm_up_steps + steps) * len(dl_train[0])

    if lr_scheduling:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                    num_training_steps=total_steps)
    else:
        scheduler = None
    # </editor-fold>

    trainer = IRMTrainer(model, num_labels, tokenizer, weights=weights, optimizer=optimizer, scheduler=scheduler,
                         device=device)

    res = trainer.fit(dl_train, dl_val, dl_val_ood, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=warm_up_reg, irm_reg=reg, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    best_model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    best_model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, best_model)
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))
    for k in ['train_env_prob', 'val_env_prob']:
        run_config[k] = str(run_config[k])
    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1]})


def train_unbiased_baseline(scores_dir, out_dir='.', dataset='SNLI', pretrained_model='bert-base-uncased', seed=None,
                            # bias params
                            threshold1=0.2, threshold2=0.7,
                            # Training params
                            bs_train=32, bs_val=32, eval_every_x_epoch=0.2, epochs=4, early_stopping=3,
                            class_weights=False, gradient_checkpoint=False,
                            # optimizer params
                            optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
                            epsilon=1e-8, weight_decay=0.01, amsgrad=False,
                            lr_scheduling=False, lr_scheduling_rate=0.1
                            ):
    """
    Train and validate a BertForSequenceClassification model on unbiased split according to ERM training scheme.
    :param scores_dir: dir to take the scores from (to partition dataset to environments)
    :param out_dir: dir to which the run results and config will be written.
    :param dataset: string indicating the dataset to use.
    :param pretrained_model: bert pre trained model to use
    :param seed: for reproducibility.
    :param threshold1: threshold to use when splitting to unbiased and biased samples
    :param threshold2: threshold to use when splitting to correctly biased and incorrectly biased samples
    :param bs_train: train batch size
    :param bs_val: validation batch size
    :param batches_per_step: number of batches to train on before doing en evaluation step
    :param steps: number of steps. Note that each step could be only part of the dataset (less than an epoch).
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    If entire dataset was iterated but more steps to go, start iterating again through the dataset.
    :param early_stopping: number of tolerance steps before early stopping the training
    :param gradient_checkpoint: Flag, trading off speed for memory - to enable training on big batches.
    :param optimizer_type: AdamW or SGD
    """
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    num_labels = datasets_config[dataset]['NUM_LABELS']
    run_config = locals()

    rng = np.random.RandomState(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                          gradient_checkpointing=gradient_checkpoint).to(device=device)

    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    # prepare data files (if doesn't exist - download and preprocess), and scores to divide dataset to environments
    file_train, file_val, file_test = prepare_dataset(dataset)
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    scores_file = f'{os.path.sep.join([scores_dir, "scores"])}.json'
    train_scores, val_scores, test_scores = prepare_scores(scores_file)

    dataset_train = create_dataset(file_train, field_indices, label_str_to_int, label_int_to_str)
    dataset_val = create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str)

    unbiased_ind_train, _, _ = split_dataset_by_bias(dataset_train, train_scores, threshold1,
                                                     threshold2)  # only unbiased examples
    ds_train = NLISubset(dataset_train, flatten_list(unbiased_ind_train))

    unbiased_ind_val, _, _ = split_dataset_by_bias(dataset_val, val_scores, threshold1,
                                                   threshold2)  # only unbiased examples
    ds_val = NLISubset(dataset_val, flatten_list(unbiased_ind_val))

    dl_train = [DataLoader(ds_train, batch_size=bs_train, shuffle=True)]
    dl_val = [DataLoader(ds_val, batch_size=bs_val)]

    if class_weights:
        weights = class_weight.compute_class_weight('balanced', classes=np.unique(ds_train.labels),
                                                    y=ds_train.labels).tolist()
        weights = torch.tensor(weights).to(device=device)
    else:
        weights = None
    batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train, eval_every_x_epoch, warm_up_epochs=epochs,
                                                               irm_epochs=0)

    # <editor-fold desc="optimization preparations">
    if optimizer_type.lower() == 'adam':
        optimizer = AdamW(model.parameters(), lr=lr, betas=(beta1, beta2), eps=epsilon, weight_decay=weight_decay,
                          amsgrad=amsgrad)
    elif optimizer_type.lower() == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise AttributeError('only SGD and Adam supported for now')

    if batches_per_step is not None:
        total_steps = steps * batches_per_step
    else:
        total_steps = steps * len(dl_train[0])

    if lr_scheduling:
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(lr_scheduling_rate * total_steps),
                                                    num_training_steps=total_steps)
    else:
        scheduler = None
    # </editor-fold>

    trainer = IRMTrainer(model, num_labels, tokenizer, optimizer=optimizer, scheduler=scheduler, device=device)

    res = trainer.fit(dl_train, dl_val, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=0.0, irm_reg=0.0, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    best_model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels)
    best_model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, best_model)
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))

    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1]})


def test_on_splits(test_file, test_dir, subset='val', out_dir='.', seed=None,
                   threshold1=None, threshold2=None,
                   bs_test=32
                   ):
    """Test a fine-tuned model on splits of a dataset - the unbiased, bias aligned and bias unaligned splits"""
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']
    reg = pretrained_cfg.get('reg', 0.0)
    dataset = pretrained_cfg['dataset']
    scores_dir = pretrained_cfg['scores_dir']
    if threshold1 is None or threshold2 is None:
        threshold1 = pretrained_cfg['threshold1']
        threshold2 = pretrained_cfg['threshold2']

    scores_file = f'{os.path.sep.join([scores_dir, "scores"])}.json'
    train_scores, val_scores, test_scores = prepare_scores(scores_file)
    if subset == 'train':
        scores = train_scores
    elif subset == 'val':
        scores = val_scores
    else:
        scores = test_scores

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    ds = create_dataset(test_file, field_indices, label_str_to_int, label_int_to_str)
    ds_unbiased_ind, ds_correctly_biased_ind, ds_incorrectly_biased_ind = split_dataset_by_bias(ds, scores,
                                                                                                threshold1, threshold2)

    dl_test_unbiased = [DataLoader(NLISubset(ds, flatten_list(ds_unbiased_ind)), batch_size=bs_test)]
    dl_test_correctly_biased = [DataLoader(NLISubset(ds, flatten_list(ds_correctly_biased_ind)), batch_size=bs_test)]
    dl_test_incorrectly_biased = [
        DataLoader(NLISubset(ds, flatten_list(ds_incorrectly_biased_ind)), batch_size=bs_test)]

    tester = IRMTrainer(model, num_labels, tokenizer, device=device)

    res_incorrectly_biased = tester.test(dl_test_incorrectly_biased, reg=reg)
    res_correctly_biased = tester.test(dl_test_correctly_biased, reg=reg)
    res_unbiased = tester.test(dl_test_unbiased, reg=reg)

    save_experiment(out_dir, run_config, {'bias unaligned split': res_incorrectly_biased._asdict(),
                                          'bias aligned split': res_correctly_biased._asdict(),
                                          'unbiased split': res_unbiased._asdict()})


def test_on_envs(test_file, test_dir, subset='val', out_dir='.', seed=None,
                 threshold1=None, threshold2=None, env_prob=None,
                 bs_test=32
                 ):
    """Test a fine-tuned model on environments. Intended to allow testing final model on validation set, and
    perform model selection."""
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    if subset == 'test':
        assert env_prob is not None, "If testing on test subset need to pass in desired environments probability"

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']
    reg = pretrained_cfg.get('reg', 0.0)
    dataset = pretrained_cfg['dataset']
    scores_dir = pretrained_cfg['scores_dir']
    if threshold1 is None or threshold2 is None:
        threshold1 = pretrained_cfg['threshold1']
        threshold2 = pretrained_cfg['threshold2']
    if env_prob is None:
        if subset == 'train':
            env_prob = pretrained_cfg['train_env_prob']
        else:
            env_prob = pretrained_cfg['val_env_prob']

    scores_file = f'{os.path.sep.join([scores_dir, "scores"])}.json'
    train_scores, val_scores, test_scores = prepare_scores(scores_file)
    if subset == 'train':
        scores = train_scores
    elif subset == 'val':
        scores = val_scores
    else:
        scores = test_scores

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]

    ds = create_envs(create_dataset(test_file, field_indices, label_str_to_int, label_int_to_str),
                     scores, env_prob, threshold1, threshold2, rng=rng)  # list of datasets

    dl = [DataLoader(env, batch_size=bs_test) for env in ds]

    tester = IRMTrainer(model, num_labels, tokenizer, device=device)

    res = tester.test(dl, reg=reg)

    save_experiment(out_dir, run_config, res)


def test_on_file(test_file, test_dir, out_dir='.', seed=None,
                 bs_test=32
                 ):
    """Test a fine-tuned model on dataset generated from passed in test_file without
    any interventions (splits, environments etc.)"""
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']
    reg = pretrained_cfg.get('reg', 0.0)
    dataset = pretrained_cfg['dataset']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']
    assert os.path.isfile(test_file), f'{test_file} - No such file'
    with open(test_file, 'r') as f:
        header = f.readline()
    fields = header.split()
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    ds = create_dataset(test_file, field_indices, label_str_to_int, label_int_to_str)

    dl = [DataLoader(ds, batch_size=bs_test)]

    tester = IRMTrainer(model, num_labels, tokenizer, device=device)

    res = tester.test(dl, reg=reg)

    test_file_name = os.path.splitext(os.path.split(test_file)[-1])[0]
    save_experiment(out_dir, run_config, {test_file_name: res._asdict()})


def test_on_hans(test_dir, out_dir='.', seed=None,
                 bs_test=32
                 ):
    """Test a fine-tuned model on Hans.
    Write predictions to hans_pres.txt file in the following format:
    tsv file with two columns - pairID (indicating the Hans sample pairID) and gold_label (indicating the predicted
    label by the model).
    Use predictions to generate run_output.json folder with classification report for each heuristic."""
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']
    reg = pretrained_cfg.get('reg', 0.0)
    dataset = pretrained_cfg['dataset']
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # <editor-fold desc="prepare hans samples">
    urlpath = datafiles_config['HANS']['urlpath']
    fields, filters = datafiles_config['HANS']['fields'], datafiles_config['HANS']['filters']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'pairID']]
    hans_filepath = prepare_data_file('HANS')

    with open(hans_filepath, 'r') as f:
        lines = f.readlines()

    # remove line ending and split on tabs. Skip first line (headers)
    samples = []
    for line in lines[1:]:
        samp = line.splitlines()[0].split(sep='\t')
        samp = tuple([samp[field_idx] for field_idx in field_indices])
        samples.append(samp)
    # </editor-fold>

    ds = BasicDataset(samples)
    dl = DataLoader(ds, batch_size=bs_test)

    preds, ids = [], []
    model.eval()
    with torch.no_grad():
        for batch in dl:
            p, h, id = batch
            input_dict = tokenizer(p, h, padding=True, truncation=True, return_tensors='pt')
            input_dict = {k: v.to(device) for k, v in input_dict.items()}

            logits = model(**input_dict)[0]
            preds += torch.argmax(logits.to('cpu'), dim=1).tolist()
            ids += list(id)

    # map predictions to string, and then map to 2 - way (neutral and contradiction = non-entailment) for hans
    preds = list(map(label_int_to_str, preds))
    preds = list(map(lambda x: 'non-entailment' if x in ['neutral', 'contradiction'] else x, preds))
    df = pd.DataFrame(list(zip(*[ids, preds])), columns=['pairID', 'gold_label'])

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.sep.join([out_dir, 'hans_preds.txt'])
    with open(out_file, 'w') as f:
        f.seek(0)
        df.to_csv(f, sep='\t', index=False, line_terminator='\n')
        f.truncate()

    res = generate_classification_report_on_hans(hans_filepath, out_file)
    save_experiment(out_dir, run_config, res)


def analyze_splits_difficulty(dataset, scores_file, subset, threshold1, threshold2):
    """
    Use the validation labels in SNLI and MNLI (5 labels, 1 by the original author, 4 by other annotators) to
    determine the difficulty of each of the splits (unbiased, bias aligned and bias unaligned splits).
    Since only samples with majority vote are kept in the dataset, we look at the distributions of samples recieving
    3, 4 or 5 votes for the gold label, and this is done for each split.
    :param dataset: dataset analyzed
    :param scores_file: file containing scores to split the test file
    :param subset: val or test. Subset to evaluate the difficulty of the samples in.
    :param threshold1: used for split_dataset_by_bias
    :param threshold2: split_dataset_by_bias
    :return: 3 dictionaries (1 per split) indicating the distributions of samples according to majority vote 3, 4, 5
    """
    label_str_to_int = datasets_config[dataset]['label_str_to_int']
    label_int_to_str = datasets_config[dataset]['label_int_to_str']

    _, val_scores, test_scores = prepare_scores(scores_file)
    _, file_val, file_test = prepare_dataset(dataset)
    if subset == 'val':
        scores = val_scores
        file = file_val
    else:
        scores = test_scores
        file = file_test

    fields = datasets_config[dataset]['fields']
    field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    unbiased, correctly_biased, incorrectly_biased = split_dataset_by_bias(
        create_dataset(file, field_indices, label_str_to_int, label_int_to_str), scores, threshold1, threshold2)

    with open(file, 'r') as f:
        lines = f.readlines()

    # remove line ending and split on tabs. Skip first line (headers)
    samples = [tuple(line.splitlines()[0].split(sep='\t')) for line in
               lines[1:]]  # samples is a list of tuples of the shape (..., label1-5)

    confidence_dict = {3: [], 4: [], 5: []}
    for ind, samp in enumerate(samples):
        gold_label, other_labels = samp[-6], samp[-5:]
        conf = other_labels.count(gold_label)
        confidence_dict[conf].append(ind)

    unbiased, correctly_biased, incorrectly_biased = flatten_list(unbiased), flatten_list(
        correctly_biased), flatten_list(incorrectly_biased)

    conf_dict_unbiased, conf_dict_correctly_biased, conf_dict_incorrectly_biased = {}, {}, {}
    for k in list(confidence_dict.keys()):
        conf_dict_unbiased[k] = round(len(set(unbiased) & set(confidence_dict[k])) / len(set(unbiased)), 2)
        conf_dict_correctly_biased[k] = round(
            len(set(correctly_biased) & set(confidence_dict[k])) / len(set(correctly_biased)), 2)
        conf_dict_incorrectly_biased[k] = round(
            len(set(incorrectly_biased) & set(confidence_dict[k])) / len(set(incorrectly_biased)), 2)

    return conf_dict_unbiased, conf_dict_correctly_biased, conf_dict_incorrectly_biased


def save_experiment(out_dir, config, res, model=None):
    if not isinstance(res, dict):
        res = res._asdict()
    output = dict(
        config=config,
        results=res
    )
    output_filename = f'{os.path.sep.join([out_dir, "run_output"])}.json'
    os.makedirs(out_dir, exist_ok=True)
    with open(output_filename, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'*** Output file {output_filename} written')

    if model is not None:
        if 'save_pretrained' in dir(model):
            model.save_pretrained(out_dir)
        else:
            model_filename = f'{os.path.sep.join([out_dir, "pytorch_model"])}.bin'
            torch.save(model.state_dict(), model_filename)
        print('*** Model saved')


def parse_cli():
    p = argparse.ArgumentParser(description='Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_score = sp.add_parser('score-hypothesis-bias', help='Score samples according to amount of hypothesis only bias')
    sp_score.set_defaults(subcmd_fn=score_hypothesis_bias)

    # <editor-fold desc="score_hypothesis_bias params">
    # data and model params
    sp_score.add_argument('--out-dir', type=str,
                          help='Name dir to save results',
                          default='.')
    sp_score.add_argument('--dataset', type=str,
                          help='Dataset to train on',
                          default='SNLI')
    sp_score.add_argument('--pretrained-model', type=str,
                          help='Name of the huggingface model', default='bert-base-uncased')
    sp_score.add_argument('--seed', '-s', type=int, help='Random seed',
                          required=False)

    # training params
    sp_score.add_argument('--k', type=int, help='Number of folds',
                          default=4)
    sp_score.add_argument('--bs-train', type=int, help='Train batch size',
                          default=32, metavar='BATCH_SIZE')
    sp_score.add_argument('--bs-val', type=int, help='Val batch size',
                          default=32, metavar='BATCH_SIZE')
    sp_score.add_argument('--eval-every-x-epoch', type=float,
                          help='Evaluate on validation every x fraction of an epoch',
                          default=0.2)
    sp_score.add_argument('--epochs', type=int,
                          help='Number of epochs', default=4)
    sp_score.add_argument('--early-stopping', type=int,
                          help='Stop after this many epochs without improvement', default=3)
    sp_score.add_argument('--class-weights',
                          help='Use class weights in loss function', action='store_true')
    sp_score.add_argument('--gradient-checkpoint',
                          help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_score.add_argument('--optimizer-type', '-ot', type=str,
                          help='Which type of optimizer to use', default="Adam")
    sp_score.add_argument('--lr', '-lr', type=float,
                          help='Learning rate', default=1e-5)
    sp_score.add_argument('--momentum', '-m', type=float,
                          help='Momentum for SGD', default=0.9)
    sp_score.add_argument('--beta1', '-b1', type=float,
                          default=0.9)
    sp_score.add_argument('--beta2', '-b2', type=float,
                          default=0.999)
    sp_score.add_argument('--epsilon', '-eps', type=float,
                          default=1e-6)
    sp_score.add_argument('--weight-decay', '-wd', type=float,
                          default=0.01)
    sp_score.add_argument('--amsgrad', action='store_true')
    sp_score.add_argument('--lr-scheduling', action='store_true')
    sp_score.add_argument('--lr-scheduling-rate', type=float,
                          default=0.1)
    # </editor-fold>

    sp_score_overlap = sp.add_parser('score-overlap-bias-MLP', help='Score samples according to overlap bias features')
    sp_score_overlap.set_defaults(subcmd_fn=score_overlap_bias_MLP)

    # <editor-fold desc="score_overlap_bias_MLP params">
    # data and model params
    sp_score_overlap.add_argument('--out-dir', type=str,
                                  help='Name dir to save results',
                                  default='.')
    sp_score_overlap.add_argument('--dataset', type=str,
                                  help='Dataset to train on',
                                  default='SNLI')
    sp_score_overlap.add_argument('--seed', '-s', type=int, help='Random seed',
                                  required=False)

    # training params
    sp_score_overlap.add_argument('--k', type=int, help='Number of folds',
                                  default=4)
    sp_score_overlap.add_argument('--bs-train', type=int, help='Train batch size',
                                  default=32, metavar='BATCH_SIZE')
    sp_score_overlap.add_argument('--bs-val', type=int, help='Val batch size',
                                  default=32, metavar='BATCH_SIZE')
    sp_score_overlap.add_argument('--eval-every-x-epoch', type=float,
                                  help='Evaluate on validation every x fraction of an epoch',
                                  default=0.2)
    sp_score_overlap.add_argument('--epochs', type=int,
                                  help='Number of epochs', default=4)
    sp_score_overlap.add_argument('--early-stopping', type=int,
                                  help='Stop after this many epochs without improvement', default=3)
    sp_score_overlap.add_argument('--class-weights',
                                  help='Use class weights in loss function', action='store_true')
    sp_score_overlap.add_argument('--gradient-checkpoint',
                                  help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_score_overlap.add_argument('--optimizer-type', '-ot', type=str,
                                  help='Which type of optimizer to use', default="Adam")
    sp_score_overlap.add_argument('--lr', '-lr', type=float,
                                  help='Learning rate', default=1e-5)
    sp_score_overlap.add_argument('--momentum', '-m', type=float,
                                  help='Momentum for SGD', default=0.9)
    sp_score_overlap.add_argument('--beta1', '-b1', type=float,
                                  default=0.9)
    sp_score_overlap.add_argument('--beta2', '-b2', type=float,
                                  default=0.999)
    sp_score_overlap.add_argument('--epsilon', '-eps', type=float,
                                  default=1e-6)
    sp_score_overlap.add_argument('--weight-decay', '-wd', type=float,
                                  default=0.01)
    sp_score_overlap.add_argument('--amsgrad', action='store_true')
    sp_score_overlap.add_argument('--lr-scheduling', action='store_true')
    sp_score_overlap.add_argument('--lr-scheduling-rate', type=float,
                                  default=0.1)
    # </editor-fold>

    # Experiment config
    sp_exp = sp.add_parser('run-irm', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_irm)

    # <editor-fold desc="run-irm params">
    # data and model params
    sp_exp.add_argument('scores_dir', type=str,
                        help='Name of dir with bias scores of the samples')
    sp_exp.add_argument('--out-dir', type=str,
                        help='Name dir to save results',
                        default='.')
    sp_exp.add_argument('--dataset', type=str,
                        help='Dataset to train on',
                        default='SNLI')
    sp_exp.add_argument('--pretrained-model', type=str,
                        help='Name of the huggingface model', default='bert-base-uncased')
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        required=False)
    sp_exp.add_argument('--train-env-prob', nargs='*', type=float,
                        help='Bias probabilities per training environment',
                        default=[0.2, 0.8])
    sp_exp.add_argument('--val-env-prob', nargs='*', type=float,
                        help='Bias probability for validation environment',
                        default=[0.2, 0.8])
    sp_exp.add_argument('--threshold1', type=float,
                        help='Sample with score above this threshold is considered biased ',
                        default=0.2)
    sp_exp.add_argument('--threshold2', type=float,
                        help='Sample with score above this threshold is considered biased ',
                        default=0.7)
    sp_exp.add_argument('--biased-samples-ratio', type=float,
                        help='Required ratio of biased samples from total samples in training set')
    sp_exp.add_argument('--total-size-train', type=float,
                        help='Required accumulated size of train environments')
    sp_exp.add_argument('--total-size-val', type=float,
                        help='Required accumulated size of val environments')

    # training params
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-val', type=int, help='Val batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--eval-every-x-epoch', type=float, help='Evaluate on validation every x fraction of an epoch',
                        default=0.2)
    sp_exp.add_argument('--epochs', type=int,
                        help='Maximal number of constrained epochs', default=4)
    sp_exp.add_argument('--warm-up-epochs', type=int,
                        help='Maximal number of warm up epochs', default=2)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without improvement', default=3)
    sp_exp.add_argument('--reg', type=float,
                        help='IRM regularization weight', default=1e3)
    sp_exp.add_argument('--warm-up-reg', type=float,
                        help='IRM regularization weight for warm up', default=1.0)
    sp_exp.add_argument('--class-weights',
                        help='Use class weights in loss function', action='store_true')
    sp_exp.add_argument('--gradient-checkpoint',
                        help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_exp.add_argument('--optimizer-type', '-ot', type=str,
                        help='Which type of optimizer to use', default="Adam")
    sp_exp.add_argument('--lr', '-lr', type=float,
                        help='Learning rate', default=1e-5)
    sp_exp.add_argument('--momentum', '-m', type=float,
                        help='Momentum for SGD', default=0.9)
    sp_exp.add_argument('--beta1', '-b1', type=float,
                        default=0.9)
    sp_exp.add_argument('--beta2', '-b2', type=float,
                        default=0.999)
    sp_exp.add_argument('--epsilon', '-eps', type=float,
                        default=1e-6)
    sp_exp.add_argument('--weight-decay', '-wd', type=float,
                        default=0.01)
    sp_exp.add_argument('--amsgrad', action='store_true')
    sp_exp.add_argument('--lr-scheduling', action='store_true')
    sp_exp.add_argument('--lr-scheduling-rate', type=float,
                        default=0.1)
    # </editor-fold>

    # Experiment config
    sp_exp_vs_unbiased = sp.add_parser('run-irm-vs-unbiased', help='Run experiment with a single '
                                                                   'configuration')
    sp_exp_vs_unbiased.set_defaults(subcmd_fn=run_irm_vs_unbiased)

    # <editor-fold desc="run-irm-vs-unbiased params">
    # data and model params
    sp_exp_vs_unbiased.add_argument('scores_dir', type=str,
                                    help='Name of dir with bias scores of the samples')
    sp_exp_vs_unbiased.add_argument('--out-dir', type=str,
                                    help='Name dir to save results',
                                    default='.')
    sp_exp_vs_unbiased.add_argument('--dataset', type=str,
                                    help='Dataset to train on',
                                    default='SNLI')
    sp_exp_vs_unbiased.add_argument('--pretrained-model', type=str,
                                    help='Name of the huggingface model', default='bert-base-uncased')
    sp_exp_vs_unbiased.add_argument('--seed', '-s', type=int, help='Random seed',
                                    required=False)
    sp_exp_vs_unbiased.add_argument('--train-env-prob', nargs='*', type=float,
                                    help='Bias probabilities per training environment',
                                    default=[0.2, 0.8])
    sp_exp_vs_unbiased.add_argument('--val-env-prob', nargs='*', type=float,
                                    help='Bias probability for validation environment',
                                    default=[0.2, 0.8])
    sp_exp_vs_unbiased.add_argument('--threshold1', type=float,
                                    help='Sample with score above this threshold is considered biased ',
                                    default=0.2)
    sp_exp_vs_unbiased.add_argument('--threshold2', type=float,
                                    help='Sample with score above this threshold is considered biased ',
                                    default=0.7)

    # training params
    sp_exp_vs_unbiased.add_argument('--bs-train', type=int, help='Train batch size',
                                    default=32, metavar='BATCH_SIZE')
    sp_exp_vs_unbiased.add_argument('--bs-val', type=int, help='Val batch size',
                                    default=32, metavar='BATCH_SIZE')
    sp_exp_vs_unbiased.add_argument('--eval-every-x-epoch', type=float,
                                    help='Evaluate on validation every x fraction of an epoch',
                                    default=0.2)
    sp_exp_vs_unbiased.add_argument('--epochs', type=int,
                                    help='Maximal number of constrained epochs', default=4)
    sp_exp_vs_unbiased.add_argument('--warm-up-epochs', type=int,
                                    help='Maximal number of warm up steps', default=2)
    sp_exp_vs_unbiased.add_argument('--early-stopping', type=int,
                                    help='Stop after this many epochs without improvement', default=3)
    sp_exp_vs_unbiased.add_argument('--reg', type=float,
                                    help='IRM regularization weight', default=1e3)
    sp_exp_vs_unbiased.add_argument('--warm-up-reg', type=float,
                                    help='IRM regularization weight for warm up', default=1.0)
    sp_exp_vs_unbiased.add_argument('--class-weights',
                                    help='Use class weights in loss function', action='store_true')
    sp_exp_vs_unbiased.add_argument('--gradient-checkpoint',
                                    help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_exp_vs_unbiased.add_argument('--optimizer-type', '-ot', type=str,
                                    help='Which type of optimizer to use', default="Adam")
    sp_exp_vs_unbiased.add_argument('--lr', '-lr', type=float,
                                    help='Learning rate', default=1e-5)
    sp_exp_vs_unbiased.add_argument('--momentum', '-m', type=float,
                                    help='Momentum for SGD', default=0.9)
    sp_exp_vs_unbiased.add_argument('--beta1', '-b1', type=float,
                                    default=0.9)
    sp_exp_vs_unbiased.add_argument('--beta2', '-b2', type=float,
                                    default=0.999)
    sp_exp_vs_unbiased.add_argument('--epsilon', '-eps', type=float,
                                    default=1e-6)
    sp_exp_vs_unbiased.add_argument('--weight-decay', '-wd', type=float,
                                    default=0.01)
    sp_exp_vs_unbiased.add_argument('--amsgrad', action='store_true')
    sp_exp_vs_unbiased.add_argument('--lr-scheduling', action='store_true')
    sp_exp_vs_unbiased.add_argument('--lr-scheduling-rate', type=float,
                                    default=0.1)
    # </editor-fold>

    # Experiment config
    sp_unbiased_baseline = sp.add_parser('train-unbiased-baseline',
                                         help='Train only on unbiased subset to get upper bound on '
                                              'unbiased model performance for in-distribution')
    sp_unbiased_baseline.set_defaults(subcmd_fn=train_unbiased_baseline)

    # <editor-fold desc="train-unbiased-baseline params">
    # data and model params
    sp_unbiased_baseline.add_argument('scores_dir', type=str,
                                      help='Name of dir with bias scores of the samples')
    sp_unbiased_baseline.add_argument('--out-dir', type=str,
                                      help='Name dir to save results',
                                      default='.')
    sp_unbiased_baseline.add_argument('--dataset', type=str,
                                      help='Dataset to train on',
                                      default='SNLI')
    sp_unbiased_baseline.add_argument('--pretrained-model', type=str,
                                      help='Name of the huggingface model', default='bert-base-uncased')
    sp_unbiased_baseline.add_argument('--seed', '-s', type=int, help='Random seed',
                                      required=False)
    sp_unbiased_baseline.add_argument('--threshold1', type=float,
                                      help='Sample with score above this threshold is considered biased ',
                                      default=0.2)
    sp_unbiased_baseline.add_argument('--threshold2', type=float,
                                      help='Sample with score above this threshold is considered biased ',
                                      default=0.7)

    # training params
    sp_unbiased_baseline.add_argument('--bs-train', type=int, help='Train batch size',
                                      default=32, metavar='BATCH_SIZE')
    sp_unbiased_baseline.add_argument('--bs-val', type=int, help='Val batch size',
                                      default=32, metavar='BATCH_SIZE')
    sp_unbiased_baseline.add_argument('--eval-every-x-epoch', type=float,
                                      help='Evaluate on validation every x fraction of an epoch',
                                      default=0.2)
    sp_unbiased_baseline.add_argument('--epochs', type=int,
                                      help='Maximal number of constrained epochs', default=4)
    sp_unbiased_baseline.add_argument('--early-stopping', type=int,
                                      help='Stop after this many epochs without improvement', default=3)
    sp_unbiased_baseline.add_argument('--class-weights',
                                      help='Use class weights in loss function', action='store_true')
    sp_unbiased_baseline.add_argument('--gradient-checkpoint',
                                      help='Checkpoint gradient to fit big batches in memory', action='store_true')

    # optimization params
    sp_unbiased_baseline.add_argument('--optimizer-type', '-ot', type=str,
                                      help='Which type of optimizer to use', default="Adam")
    sp_unbiased_baseline.add_argument('--lr', '-lr', type=float,
                                      help='Learning rate', default=1e-5)
    sp_unbiased_baseline.add_argument('--momentum', '-m', type=float,
                                      help='Momentum for SGD', default=0.9)
    sp_unbiased_baseline.add_argument('--beta1', '-b1', type=float,
                                      default=0.9)
    sp_unbiased_baseline.add_argument('--beta2', '-b2', type=float,
                                      default=0.999)
    sp_unbiased_baseline.add_argument('--epsilon', '-eps', type=float,
                                      default=1e-6)
    sp_unbiased_baseline.add_argument('--weight-decay', '-wd', type=float,
                                      default=0.01)
    sp_unbiased_baseline.add_argument('--amsgrad', action='store_true')
    sp_unbiased_baseline.add_argument('--lr-scheduling', action='store_true')
    sp_unbiased_baseline.add_argument('--lr-scheduling-rate', type=float,
                                      default=0.1)
    # </editor-fold>

    sp_test_splits = sp.add_parser('test-on-splits', help='Evaluate model on test or validation')
    sp_test_splits.set_defaults(subcmd_fn=test_on_splits)

    # <editor-fold desc="test-on-splits params">
    sp_test_splits.add_argument('test_file', type=str,
                                help='File to evaluate model on')
    sp_test_splits.add_argument('test_dir', type=str,
                                help='Name dir to load fine-tuned model')

    sp_test_splits.add_argument('--out-dir', type=str,
                                help='Name dir to save results',
                                default='.')
    sp_test_splits.add_argument('--subset', type=str,
                                help='The subset according to which we load scores',
                                choices=['train', 'val', 'test'], default='val')

    sp_test_splits.add_argument('--seed', '-s', type=int, help='Random seed',
                                required=False)
    sp_test_splits.add_argument('--threshold1', type=float,
                                help='Scores whose total variation distance from uniform distribution is'
                                     'less then this threshold are considered unbiased')
    sp_test_splits.add_argument('--threshold2', type=float,
                                help='Scores for which the probability of the ground truth label is above this '
                                     'threshold are considered correctly biased.')
    sp_test_splits.add_argument('--bs-test', type=int, help='Batch size',
                                default=32, metavar='BATCH_SIZE')

    # </editor-fold>

    sp_test_envs = sp.add_parser('test-on-envs', help='Evaluate model on environments')
    sp_test_envs.set_defaults(subcmd_fn=test_on_envs)

    # <editor-fold desc="test-on-envs params">
    sp_test_envs.add_argument('test_file', type=str,
                              help='File to evaluate model on')
    sp_test_envs.add_argument('test_dir', type=str,
                              help='Name dir to load fine-tuned model')

    sp_test_envs.add_argument('--out-dir', type=str,
                              help='Name dir to save results',
                              default='.')
    sp_test_envs.add_argument('--subset', type=str,
                              help='The subset according to which we load scores',
                              choices=['train', 'val', 'test'], default='val')

    sp_test_envs.add_argument('--seed', '-s', type=int, help='Random seed',
                              required=False)
    sp_test_envs.add_argument('--threshold1', type=float,
                              help='Scores whose total variation distance from uniform distribution is'
                                   'less then this threshold are considered unbiased')
    sp_test_envs.add_argument('--threshold2', type=float,
                              help='Scores for which the probability of the ground truth label is above this '
                                   'threshold are considered correctly biased.')
    sp_test_envs.add_argument('--env-prob', nargs='*', type=float,
                              help='Bias probabilities for environments')
    sp_test_envs.add_argument('--bs-test', type=int, help='Batch size',
                              default=32, metavar='BATCH_SIZE')

    # </editor-fold>

    sp_test_file = sp.add_parser('test-on-file', help='Evaluate model on test or validation')
    sp_test_file.set_defaults(subcmd_fn=test_on_file)

    # <editor-fold desc="test-on-file params">
    sp_test_file.add_argument('test_file', type=str,
                              help='File to evaluate model on')
    sp_test_file.add_argument('test_dir', type=str,
                              help='Name dir to load fine-tuned model')

    sp_test_file.add_argument('--out-dir', type=str,
                              help='Name dir to save results',
                              default='.')

    sp_test_file.add_argument('--seed', '-s', type=int, help='Random seed',
                              required=False)
    sp_test_file.add_argument('--bs-test', type=int, help='Batch size',
                              default=32, metavar='BATCH_SIZE')

    # </editor-fold>

    sp_test_hans = sp.add_parser('test-on-hans', help='Evaluate model on test or validation')
    sp_test_hans.set_defaults(subcmd_fn=test_on_hans)

    # <editor-fold desc="test-on-hans params">
    sp_test_hans.add_argument('test_dir', type=str,
                              help='Name dir to load fine-tuned model')

    sp_test_hans.add_argument('--out-dir', type=str,
                              help='Name dir to save results',
                              default='.')

    sp_test_hans.add_argument('--seed', '-s', type=int, help='Random seed',
                              required=False)
    sp_test_hans.add_argument('--bs-test', type=int, help='Batch size',
                              default=32, metavar='BATCH_SIZE')

    # </editor-fold>

    sp_analyze_difficulty = sp.add_parser('analyze-splits-difficulty', help='Analyze splits difficulty '
                                                                            'according to annotator agreement')
    sp_analyze_difficulty.set_defaults(subcmd_fn=analyze_splits_difficulty)

    # <editor-fold desc="analyze-splits-difficulty params">
    sp_analyze_difficulty.add_argument('dataset', type=str,
                                       help='Name of dataset to analyze')
    sp_analyze_difficulty.add_argument('scores_file', type=str,
                                       help='Scores file')
    sp_analyze_difficulty.add_argument('subset', type=str,
                                       help='The subset according to which we load scores',
                                       choices=['val', 'test'])
    sp_analyze_difficulty.add_argument('threshold1', type=float,
                                       help='Scores whose total variation distance from uniform distribution is'
                                            'less then this threshold are considered unbiased')
    sp_analyze_difficulty.add_argument('threshold2', type=float,
                                       help='Scores for which the probability of the ground truth label is above this '
                                            'threshold are considered correctly biased.')

    # </editor-fold>

    parsed = p.parse_args()

    if 'subcmd_fn' not in parsed:
        p.print_help()
        sys.exit()
    return parsed


if __name__ == "__main__":
    torch.cuda.empty_cache()

    parsed_args = parse_cli()
    subcmd_fn = parsed_args.subcmd_fn
    del parsed_args.subcmd_fn
    print(f'*** Starting {subcmd_fn.__name__} with config:\n{parsed_args}')
    subcmd_fn(**vars(parsed_args))
    # score_hypothesis_bias(out_dir='scores/hypothesis_bias/snli_bin', dataset='SNLI',
    #                       seed=None,
    #                       # Training params
    #                       k=2, bs_train=512, bs_val=512, eval_every_x_epoch=0.2, epochs=12, early_stopping=5,
    #                       gradient_checkpoint=False,
    #                       )
    # score_overlap_bias_MLP(out_dir='scores/overlap_bias/snli_bin', dataset='SNLI_Binary',
    #                        seed=None,
    #                        # Training params
    #                        k=2, bs_train=512, bs_val=512, eval_every_x_epoch=0.2, epochs=1, early_stopping=5,
    #                        gradient_checkpoint=False,
    #                        )
    # from data_utils import datasets_config
    #
    # dataset = 'SNLI'
    # label_str_to_int, label_int_to_str = datasets_config[dataset]['label_str_to_int'], datasets_config[dataset][
    #     'label_int_to_str']
    # fields = datasets_config[dataset]['fields']
    # field_indices = [fields.index(field_name) for field_name in ['sentence1', 'sentence2', 'gold_label']]
    # file_train, file_val, file_test = prepare_dataset(dataset)
    # ds_train = create_dataset(file_train, field_indices, label_str_to_int, label_int_to_str)
    # ds_val = create_dataset(file_val, field_indices, label_str_to_int, label_int_to_str)
    # ds_test = create_dataset(file_test, field_indices, label_str_to_int, label_int_to_str)
    #
    # with open('scores/hypothesis_bias/snli/scores.json', 'r') as f:
    #     scores = json.load(f)
    # train_scores, val_scores, test_scores = scores['train_scores'], scores['val_scores'], scores['test_scores']
    # train_labels, val_labels, test_labels = ds_train.labels, ds_val.labels, ds_test.labels
    # fig = plot_scores_histogram({'Train': (train_scores, train_labels),
    #                              'Val': (val_scores, val_labels),
    #                              'Test': (test_scores, test_labels)}, 2, label_int_to_str)
    # fig.show()
    # fig = plot_splits_vs_thresholds(ds_train, train_scores, set='train')
    # fig.show()
    # run_irm('scores/overlap_bias/mnli/binary', out_dir='temp', dataset='MNLI_Binary',
    #         pretrained_model='bert-base-uncased',
    #         seed=None,
    #         # bias params
    #         biased_samples_ratio=0.52, total_size_train=100000, total_size_val=5000,
    #         train_env_prob=(0.7, 0.9), val_env_prob=(0.7, 0.9), threshold1=0.11, threshold2=0.4,
    #         class_weights=True,
    #         # Training params
    #         bs_train=4, bs_val=4, eval_every_x_epoch=0.2, epochs=4, warm_up_epochs=1, early_stopping=5,
    #         reg=1e3, warm_up_reg=1.0, gradient_checkpoint=False
    #         )
    # run_irm_vs_unbiased('scores/hypothesis_bias/snli', 'unbiased_exp', out_dir='temp', dataset='SNLI', pretrained_model='bert-base-uncased',
    #                     seed=None,
    #                     # bias params
    #                     train_env_prob=(0.7, 0.9), val_env_prob=(0.7, 0.9),
    #                     # Training params
    #                     bs_train=32, bs_val=32, eval_every_x_epoch=0.2, epochs=1, warm_up_epochs=1, early_stopping=3,
    #                     reg=1e3, warm_up_reg=1.0, gradient_checkpoint=False,
    #                     # optimizer params
    #                     optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
    #                     epsilon=1e-8, weight_decay=0.01, amsgrad=False,
    #                     lr_scheduling=False, lr_scheduling_rate=0.1
    #                     )
    # train_unbiased_baseline('scores/hypothesis_bias/snli', out_dir='temp', dataset='SNLI', pretrained_model='bert-base-uncased',
    #         seed=None,
    #         # bias params
    #       threshold1=0.2, threshold2=0.5,
    #         # Training params
    #         bs_train=256, bs_val=256, batches_per_step=1, steps=1,  early_stopping=3,
    #         gradient_checkpoint=False
    #         )
    # from utils import flatten_list
    #
    # def analyze_splits_difficulty(subset='val', threshold1=0.2, threshold2=0.5):
    #     dataset = 'SNLI'
    #
    #     file_train, file_val, file_test = prepare_dataset(dataset)
    #     label_str_to_int = datasets_config[dataset]['label_str_to_int']
    #     label_int_to_str = datasets_config[dataset]['label_int_to_str']
    #
    #     train_scores, val_scores, test_scores = prepare_scores('scores/hypothesis_bias/snli/scores.json')
    #     if subset == 'val':
    #         unbiased, correctly_biased, incorrectly_biased = split_dataset_by_bias(
    #             create_dataset(file_val, label_str_to_int, label_int_to_str), val_scores, threshold1, threshold2)
    #         conf_file = 'data/SNLI_Validation_Labels/snli_1.0_dev.txt'
    #     elif subset == 'test':
    #         unbiased, correctly_biased, incorrectly_biased = split_dataset_by_bias(
    #             create_dataset(file_test, label_str_to_int, label_int_to_str), test_scores, threshold1, threshold2)
    #         conf_file = 'data/SNLI_Validation_Labels/snli_1.0_test.txt'
    #
    #     assert os.path.isfile(conf_file), f'{conf_file} - No such file'
    #     with open(conf_file, 'r') as f:
    #         lines = f.readlines()
    #
    #     # remove line ending and split on tabs. Skip first line (headers)
    #     samples = [tuple(line.splitlines()[0].split(sep='\t')) for line in
    #                lines[1:]]  # samples is a list of (p, h, y, label1-5)
    #
    #     confidence_dict = {3: [], 4: [], 5: []}
    #     for ind, samp in enumerate(samples):
    #         gold_label, other_labels = samp[-6], samp[-5:]
    #         conf = other_labels.count(gold_label)
    #         confidence_dict[conf].append(ind)
    #
    #     unbiased, correctly_biased, incorrectly_biased = flatten_list(unbiased), flatten_list(
    #         correctly_biased), flatten_list(incorrectly_biased)
    #
    #     conf_dict_unbiased, conf_dict_correctly_biased, conf_dict_incorrectly_biased = {}, {}, {}
    #     for k in list(confidence_dict.keys()):
    #         conf_dict_unbiased[k] = round(len(set(unbiased) & set(confidence_dict[k])) / len(set(unbiased)), 2)
    #         conf_dict_correctly_biased[k] = round(len(set(correctly_biased) & set(confidence_dict[k])) / len(set(correctly_biased)), 2)
    #         conf_dict_incorrectly_biased[k] = round(len(set(incorrectly_biased) & set(confidence_dict[k])) / len(set(incorrectly_biased)), 2)
    #
    #     print(f'for {subset} set:')
    #     print(f'unbiased: \n {conf_dict_unbiased}')
    #     print(f'correctly_biased: \n {conf_dict_correctly_biased}')
    #     print(f'incorrectly_biased: \n {conf_dict_incorrectly_biased}')
    #
    #
    # analyze_splits_difficulty(subset='val', threshold1=0.2, threshold2=0.5)
    #
    # analyze_splits_difficulty(subset='test', threshold1=0.2, threshold2=0.5)
    # test_on_hans('temp', 'temp', bs_test=4)
    # test_on_splits('data/MNLI/multinli_1.0_dev_mismatched.txt', '.', subset='test')
    # from utils import print_metrics_on_hans
    # print_metrics_on_hans('data/data_files/hans.txt', 'hans_preds.txt')

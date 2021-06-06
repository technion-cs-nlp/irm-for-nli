from data_utils import prepare_dataset
from dataset_utils import create_datasets, BasicDataset
import settings
from utils import calc_steps_params, general_cm_to_fig, cm_to_fig, to_tuple
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW, SGD
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, get_linear_schedule_with_warmup, \
    AutoModelForSequenceClassification, AutoTokenizer
from attribution_utils import *
from lit_utils import *
from train_utils import IRMTrainer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import argparse
import re
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from lit_nlp import server_flags


def run_irm(out_dir='.', dataset='SNLI', num_labels=3, pretrained_model='bert-base-uncased', seed=None,
            # bias params
            train_env_prob=(0.8, 0.9), val_env_prob=(0.8, 0.9), val_ood_env_prob=(1 / 3,), biased_samples_ratio=1.0,
            bias_tokens_per_label=1, bias_pattern='simple',
            # Training params
            bs_train=32, bs_val=32, train_size=None, val_size=None,
            eval_every_x_epoch=0.2, epochs=4, warm_up_epochs=2, early_stopping=3,
            reg=1e3, warm_up_reg=1.0, gradient_checkpoint=False,
            # optimizer params
            optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.01, amsgrad=False,
            lr_scheduling=False, lr_scheduling_rate=0.1
            ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()
    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    # prepare data files (if doesn't exist - download and preprocess)
    file_train, file_val, file_test = prepare_dataset(dataset)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=num_labels,
                                                          gradient_checkpointing=gradient_checkpoint).to(device=device)

    rng = np.random.RandomState(seed)

    bias_tokens = settings.VOCAB_BIAS[bias_tokens_per_label - 1]
    run_config['bias_tokens'] = bias_tokens  # add bias tokens to saved configuration

    # create biased datasets by appending unused tokens to hypothesis
    assert bias_pattern == 'simple' or bias_tokens_per_label > 1, "Can't run with complex bias pattern and 1 bias token per label"
    ds_train = create_datasets(file_train, num_datasets=len(train_env_prob), rng=rng, bias_tokens=bias_tokens,
                               biased_samples_ratio=biased_samples_ratio, env_prob=train_env_prob,
                               bias_pattern=bias_pattern, size=train_size)
    ds_val = create_datasets(file_val, num_datasets=len(val_env_prob), rng=rng, bias_tokens=bias_tokens,
                             biased_samples_ratio=biased_samples_ratio, env_prob=val_env_prob,
                             bias_pattern=bias_pattern, size=val_size)
    ds_val_ood = create_datasets(file_val, num_datasets=1, rng=rng, bias_tokens=bias_tokens,
                                 biased_samples_ratio=biased_samples_ratio, env_prob=val_ood_env_prob,
                                 bias_pattern=bias_pattern)
    dl_train = [DataLoader(env, batch_size=bs_train, shuffle=True) for env in ds_train]
    dl_val = [DataLoader(env, batch_size=bs_val) for env in ds_val]
    dl_val_ood = [DataLoader(env, batch_size=bs_val) for env in ds_val_ood]

    batches_per_step, warm_up_steps, steps = calc_steps_params(dl_train, eval_every_x_epoch, warm_up_epochs, epochs)

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

    trainer = IRMTrainer(model, num_labels, tokenizer, optimizer, scheduler, device=device)

    res = trainer.fit(dl_train, dl_val, dl_val_ood, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=warm_up_reg, irm_reg=reg, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, model)
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))
    for k in ['train_env_prob', 'val_env_prob', 'val_ood_env_prob', 'bias_tokens']:
        run_config[k] = str(run_config[k])
    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1],
                        'hparam/val_ood_accuracy': res.test_acc_ood[-1]})

    return res


def test_irm(test_file, test_dir, out_dir='.', seed=None,
             env_prob=(0.0,), biased_samples_ratio=None, bias_tokens=None,
             bs_test=32, reg=1e3
             ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    rng = np.random.RandomState(seed)

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = pretrained_cfg['pretrained_model'], pretrained_cfg['num_labels']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir).to(device=device)
    rng = np.random.RandomState(seed)

    # generate functions to add bias to datasets
    assert len(env_prob) == 1, "Only one test environment is supported"
    if bias_tokens is None or len(bias_tokens) != num_labels:
        print("Using same bias tokens as used when training")
        bias_tokens = pretrained_cfg['bias_tokens']
        bias_tokens = to_tuple(bias_tokens)
    if biased_samples_ratio is None:
        biased_samples_ratio = pretrained_cfg['biased_samples_ratio']

    # create biased datasets by appending unused tokens to hypothesis
    ds_test = create_datasets(test_file, num_datasets=1, rng=rng, biased_samples_ratio=biased_samples_ratio,
                              env_prob=env_prob, bias_tokens=bias_tokens, bias_pattern='simple')
    dl_test = [DataLoader(env, batch_size=bs_test) for env in ds_test]

    tester = IRMTrainer(model, num_labels, tokenizer, device=device)

    res = tester.test(dl_test, reg=reg)

    loss = res.loss
    acc = res.accuracy
    cm_fig = cm_to_fig(res.cm)
    pred_prob = res.pred_prob
    test_res = {'loss': loss, 'accuracy': acc, 'predicted probabilities': pred_prob}

    save_experiment(out_dir, run_config, test_res)
    output_filename = f'{os.path.sep.join([out_dir, "confusion_matrix"])}.png'
    cm_fig.savefig(output_filename)

    return res


def analyze_irm(model_dir, ref_dir, model_test_dir='test', model_attribution_dir=None,
                ref_test_dir='test', scores_file=None, scores_set='test', scores_bias='', out_dir='.', bs=8):
    """Analyze a model. The hypothesis - a model tends to use bias when the sample is difficult.
    Assumption - confidence of unbiased model implies difficulty of sample.
    """
    analyze_config = locals()

    # <editor-fold desc="Prepare model test predictions and test file ground truth labels">
    assert os.path.isdir(model_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([model_dir, model_test_dir, "run_output"])}.json') as f:
        model_run_output_dict = json.load(f)
    pred_prob = model_run_output_dict['results']['predicted probabilities']
    model_config = model_run_output_dict['config']
    pred_prob = np.array(pred_prob)
    y_pred = np.argmax(pred_prob, axis=-1)  # np array
    test_file = model_config['test_file']

    ds_test = create_datasets(test_file, num_datasets=1)[0]
    num_samples = ds_test.num_samples
    y_gt = list(zip(*ds_test.samples))[-1]
    num_labels = len(set(y_gt))
    y_gt = np.array(list(map(settings.labels_string_to_int, y_gt)))  # np array

    assert pred_prob.shape[0] == num_samples, 'Model num samples is different from test file'
    assert pred_prob.shape[1] == num_labels, 'Model num labels is different from test file'
    # </editor-fold>

    # <editor-fold desc="Prepare weights from reference model predictions">
    # get all ref runs
    assert os.path.isdir(ref_dir), "Ref directory doesn't exist"
    test_dirs = list(
        filter(lambda x: re.match('.*run[0-9]+.*/' + ref_test_dir, x[0]) is not None and 'run_output.json' in x[2],
               os.walk(ref_dir)))
    assert len(test_dirs) > 0, 'No reference predictions'

    ref_pred_prob = []
    for d in test_dirs:
        with open(os.sep.join([d[0], 'run_output.json'])) as f:
            ref_run_output_dict = json.load(f)
        probs = ref_run_output_dict['results']['predicted probabilities']
        ref_config = ref_run_output_dict['config']
        assert len(probs) == num_samples, 'Lengths of datasets do not match between reference runs'
        assert len(probs[0]) == num_labels, 'Different number of labels between ref and model'
        assert ref_config['test_file'] == test_file, 'Ref and model tested on different test files'
        ref_pred_prob.append(probs)  # num runs x num samples x num labels

    ref_pred_prob_std = np.std(np.array(ref_pred_prob), axis=0)  # num samples x num labels
    ref_stability_mu, ref_stability_std = np.mean(ref_pred_prob_std), np.std(ref_pred_prob_std)
    ref_pred_prob = np.mean(np.array(ref_pred_prob), axis=0)  # np array: num samples x num labels

    # </editor-fold>

    # <editor-fold desc="confusion matrix with reference model probabiliites as weights">
    weights = torch.tensor(
        ref_pred_prob[np.arange(num_samples), y_pred].tolist())  # take only probabilities of class predicted by model
    with wrap_confusion_matrix(num_labels, torch.tensor(y_gt), torch.tensor(y_pred), weights) as padded_inputs:
        cm = confusion_matrix(y_true=padded_inputs[0], y_pred=padded_inputs[1], sample_weight=padded_inputs[2])
        cm_count = confusion_matrix(y_true=padded_inputs[0], y_pred=padded_inputs[1])
    cm = np.around(cm / (cm_count + 1e-6), 2)
    # </editor-fold>

    # calculate total variation mean and std of reference model predictions according to correctly and incorrectly
    # predicted by model
    total_variation_dict = {}
    weights = ref_pred_prob[y_gt == y_pred, :]
    dist_from_uniform = 0.5 * np.linalg.norm(weights - np.ones_like(weights) * (1 / num_labels), ord=1, axis=1)
    total_variation_dict['correctly predicted'] = dist_from_uniform
    weights = ref_pred_prob[y_gt != y_pred, :]
    dist_from_uniform = 0.5 * np.linalg.norm(weights - np.ones_like(weights) * (1 / num_labels), ord=1, axis=1)
    total_variation_dict['incorrectly predicted'] = dist_from_uniform

    # save results
    save_experiment(out_dir, analyze_config,
                    {'SampleConfidence': cm.tolist(),
                     'ref_stability_mu': ref_stability_mu.item(), 'ref_stability_std': ref_stability_std.item(),
                     'correctly predicted total variation mean': np.mean(
                         total_variation_dict['correctly predicted']).item(),
                     'correctly predicted total variation std': np.std(
                         total_variation_dict['correctly predicted']).item(),
                     'incorrectly predicted total variation mean': np.mean(
                         total_variation_dict['incorrectly predicted']).item(),
                     'incorrectly predicted total variation std': np.std(
                         total_variation_dict['incorrectly predicted']).item()})
    cm_fig = general_cm_to_fig(cm, colormap_name='Wistia')
    cm_fig.savefig(f'{os.path.sep.join([out_dir, "SampleConfidence"])}.png')

    # calculate histograms
    y_pred_ref = np.argmax(ref_pred_prob, axis=-1)
    # <editor-fold desc="Predicted the same and correctly">
    fig, axes = plt.subplots(nrows=num_labels, ncols=2, sharex='all', sharey='all', figsize=(20, 15), dpi=125)
    num_rows, num_cols = num_labels, 2
    for lbl, ax in zip([x for x in range(num_labels)], axes):
        for probabilities, a in zip([ref_pred_prob, pred_prob], ax):
            indices = np.logical_and(y_pred == y_pred_ref, y_pred == y_gt)
            indices = np.logical_and(indices, y_gt == lbl)
            probabilities = probabilities[indices, lbl]
            n, bins, patches = a.hist(probabilities, bins=np.arange(0.0, 1.1, 0.1),
                                      edgecolor='black', rwidth=0.7)
            a.set_xticks(bins)
        ax[0].set_ylabel(f'Count of samples, label {settings.labels_int_to_string(lbl)}')
    for j in range(num_cols):
        axes[-1, j].set_xlabel('Score')
    fig.savefig(f'{os.path.sep.join([out_dir, "PredictedSameCorrectly"])}.png')
    # </editor-fold>

    # <editor-fold desc="Predicted the same and incorrectly">
    fig, axes = plt.subplots(nrows=num_labels, ncols=2, sharex='all', sharey='all', figsize=(20, 15), dpi=125)
    num_rows, num_cols = num_labels, 2
    for lbl, ax in zip([x for x in range(num_labels)], axes):
        for probabilities, a in zip([ref_pred_prob, np.array(pred_prob)], ax):
            indices = np.logical_and(y_pred == y_pred_ref, y_pred != y_gt)
            indices = np.logical_and(indices, y_pred == lbl)
            probabilities = probabilities[indices, lbl]
            n, bins, patches = a.hist(probabilities, bins=np.arange(0.0, 1.1, 0.1),
                                      edgecolor='black', rwidth=0.7)
            a.set_xticks(bins)
        ax[0].set_ylabel(f'Count of samples, label {settings.labels_int_to_string(lbl)}')
    for j in range(num_cols):
        axes[-1, j].set_xlabel('Score')
    fig.savefig(f'{os.path.sep.join([out_dir, "PredictedSameInCorrectly"])}.png')
    # </editor-fold>

    # <editor-fold desc="Predicted Differently">
    for lbl_ref in [x for x in range(num_labels)]:
        fig, axes = plt.subplots(nrows=num_labels, ncols=2, sharex='all', sharey='all', figsize=(20, 15), dpi=125)
        num_rows, num_cols = num_labels, 2
        for lbl, ax in zip([x for x in range(num_labels)], axes):
            indices = np.logical_and(y_pred != y_pred_ref, y_pred_ref == lbl_ref)
            indices = np.logical_and(indices, y_pred == lbl)
            n, bins, patches = ax[0].hist(ref_pred_prob[indices, lbl_ref], bins=np.arange(0.0, 1.1, 0.1),
                                          edgecolor='black', rwidth=0.7)
            ax[0].set_xticks(bins)
            ax[0].set_ylabel(f'Count of samples, label {settings.labels_int_to_string(lbl)}')
            n, bins, patches = ax[1].hist(pred_prob[indices, lbl], bins=np.arange(0.0, 1.1, 0.1),
                                          edgecolor='black', rwidth=0.7)
            ax[1].set_xticks(bins)

        for j in range(num_cols):
            axes[-1, j].set_xlabel('Score')
        save_name = f'PredictedDifferentlyRefLabel{settings.labels_int_to_string(lbl_ref)}'
        fig.savefig(f'{os.path.sep.join([out_dir, save_name])}.png')
    # </editor-fold>

    if model_attribution_dir is not None:
        with open(f'{os.path.sep.join([model_dir, model_attribution_dir, "run_output"])}.json') as f:
            model_attribution_run_output_dict = json.load(f)
        attributions = model_attribution_run_output_dict['results']['PredictedLabelAttribution']
        attributions_config = model_attribution_run_output_dict['config']
        attributions = np.array(attributions)
        assert attributions.shape[
                   0] == num_samples, 'Lengths of datasets do not match between attribution and test file'
        assert attributions_config['test_file'] == test_file, 'Attributions and model tested on different test files'

        # uniformity_measure = 0.5 * np.linalg.norm(ref_pred_prob - np.ones_like(ref_pred_prob) * (1 / num_labels), ord=1, axis=1)
        uniformity_measure = np.max(ref_pred_prob, axis=-1)

        indices_predicted_same_correctly = np.logical_and(y_pred == y_pred_ref, y_pred == y_gt)
        indices_predicted_same_incorrectly = np.logical_and(y_pred == y_pred_ref, y_pred != y_gt)
        indices_predicted_differently = y_pred != y_pred_ref
        assert all(indices_predicted_same_correctly.astype(int) + indices_predicted_same_incorrectly.astype(int) + \
                   indices_predicted_differently.astype(int) == np.ones(num_samples))
        for i, (plt_name, inds) in enumerate(
                zip(['predicted same correctly', 'predicted same incorrectly', 'predicted differently'],
                    [indices_predicted_same_correctly, indices_predicted_same_incorrectly,
                     indices_predicted_differently])):
            fig, ax = plt.subplots(figsize=(20, 15))

            marker_color = [np.array(['g', 'r', 'b'])[[x, y, z]].item() for x, y, z in
                            zip(indices_predicted_same_correctly, indices_predicted_same_incorrectly,
                                indices_predicted_differently)]
            for lbl in range(num_labels):
                inds_lbl = np.logical_and(inds, y_pred == lbl)
                marker_color = ['g', 'r', 'b'][lbl]
                ax.scatter(uniformity_measure[inds_lbl], attributions[inds_lbl], c=marker_color,
                           alpha=0.7, label=settings.labels_int_to_string(lbl))
            # ax.set_xlim((0.9, 1))
            ax.set_xlabel('Max reference probability')
            ax.set_ylabel('Bias token attribution')
            ax.set_title(f'Samples {plt_name}')
            ax.legend()
            fig.savefig(f'{os.path.sep.join([out_dir, "AttributionVsDist" + str(i)])}.png')

        # check for natural bias in high confidence low attribution samples (if scores are provided)
        if scores_file is not None:
            assert os.path.isfile(scores_file), f'{scores_file} - No such file'
            with open(scores_file, 'r') as f:
                scores = json.load(f)
            if scores_set == 'test':
                scores = scores['test_scores']
            elif scores_set == 'val':
                scores = scores['val_scores']
            else:
                scores = scores['train_scores']
            scores = sorted(scores, key=lambda x: x[0])  # sort scores by index of sample
            scores = np.array(list(zip(*scores))[-1])  # after sorting dump indices and only leave scores as 2d np array
            assert scores.shape[0] == len(y_pred), "Different num samples in scores file"
            indices_predicted_the_same = y_pred == y_pred_ref
            scores = scores[indices_predicted_the_same, y_pred[indices_predicted_the_same]]

            fig, ax = plt.subplots(figsize=(20, 15))
            ax.scatter(uniformity_measure[indices_predicted_the_same], attributions[indices_predicted_the_same],
                       c=scores, cmap='Wistia', alpha=0.7)
            ax.set_xlabel('Max reference probability')
            ax.set_ylabel('Bias token attribution')
            plot_name = f'{scores_bias}Bias'
            fig.savefig(f'{os.path.sep.join([out_dir, plot_name])}.png')


def attribute_irm(test_file, test_dir, out_dir='.',
                  env_prob=(0.0,), biased_samples_ratio=None, bias_tokens=None,
                  seed=None, bs=8):
    """Use Captums IntegratedGradients to calculate attributions with respect to the bias token.
     The results are summarized in a "confusion matrix" style. Two confusion matrices are created - one attributing
     the gold label and one attributing the predicted label. Both confusion matrices are saved as png images in
     the out_dir and as lists in the results section of the run_output.json. Attributions are saved as lists in the
     run_output.json file."""
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        run_cfg = json.load(config_file)['config']
    pretrained_model, num_labels = run_cfg['pretrained_model'], run_cfg['num_labels']
    if biased_samples_ratio is None:
        biased_samples_ratio = run_cfg['biased_samples_ratio']
    if bias_tokens is None or len(bias_tokens) != num_labels:
        bias_tokens = run_cfg['bias_tokens']
        bias_tokens = to_tuple(bias_tokens)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
    model = BertForSequenceClassification.from_pretrained(test_dir, gradient_checkpointing=False).to(device=device)
    rng = np.random.RandomState(seed)

    ds_test = create_datasets(test_file, num_datasets=1, rng=rng, biased_samples_ratio=biased_samples_ratio,
                              env_prob=env_prob, bias_tokens=bias_tokens, bias_pattern='simple')[0]

    dl_test = DataLoader(ds_test, batch_size=bs, sampler=SequentialSampler(ds_test))

    # quantitative analysis of gradient attribution of the bias token embedding
    cm_pred, cm_gold, attr_pred, attr_gold = confusion_matrix_from_embedding_attribution(model, tokenizer, dl_test,
                                                                                         device=device,
                                                                                         internal_bs=None)
    cm_pred_fig, cm_gold_fig = general_cm_to_fig(cm_pred, colormap_name='coolwarm'), general_cm_to_fig(cm_gold,
                                                                                                       colormap_name='coolwarm')

    # save results
    save_experiment(out_dir, run_config,
                    {'PredictedLabelAttributionCM': cm_pred.tolist(), 'GoldLabelAttributionCM': cm_gold.tolist(),
                     'PredictedLabelAttribution': attr_pred, 'GoldLabelAttribution': attr_gold})
    cm_pred_fig.savefig(f'{os.path.sep.join([out_dir, "PredictedLabelAttribution"])}.png')
    cm_gold_fig.savefig(f'{os.path.sep.join([out_dir, "GoldLabelAttribution"])}.png')


def run_lit(test_file, test_dir, port=6006, warm_start=0.0, bs=32):
    if not isinstance(test_dir, list):
        test_dir = [test_dir]
    run_cfg = []
    for diri in test_dir:
        assert os.path.isdir(diri), "Model directory doesn't exist"
        with open(f'{os.path.sep.join([diri, "run_output"])}.json') as config_file:
            run_cfg.append(json.load(config_file)['config'])
        for k in ['pretrained_model', 'num_labels', 'bias_tokens', 'bias_loc']:
            assert all([cfg[k] == run_cfg[0][k] for cfg in run_cfg]), "Models can't be compared"

    run_cfg = run_cfg[0]  # if all models relevant config is the same, arbitrarily use the first config
    pretrained_model, num_labels = run_cfg['pretrained_model'], run_cfg['num_labels']
    # create biased datasets by appending tokens to hypothesis
    bias_tokens, bias_loc = run_cfg['bias_tokens'], run_cfg['bias_loc']
    bias_function = [get_bias_function([tok] * num_labels, prob=1.0, num_labels=num_labels, flag=3, bias_loc=bias_loc)
                     for tok in bias_tokens]
    # bias_function = []
    bias_function.append(get_bias_function(bias_tokens, prob=1.0, num_labels=num_labels, flag=0))

    ds_test = []
    for bias_f in bias_function:
        ds_test += create_datasets(test_file, num_datasets=1, biased_samples_ratio=0.0, prob=0.0)

    # Create wrapped datasets
    datasets = {}
    bias_types = ['entailment', 'neutral', 'contradiction']
    bias_types.sort()
    bias_types += ['none']
    for ds, bias_type, bias_tok in zip(ds_test, bias_types, bias_tokens + ['none']):
        dataset_name = os.path.splitext(os.path.split(test_file)[-1])[0] + '_' + bias_type
        datasets[dataset_name] = NLIDatasetWrapper(ds, bias_type, bias_tok)

    # Create wrapped models
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    models = {}
    for diri in test_dir:
        model_config = BertConfig.from_pretrained(diri, output_hidden_states=True,
                                                  output_attentions=True,
                                                  gradient_checkpointing=False)
        # model_config = BertConfig.from_pretrained(pretrained_model, num_labels=3,
        #                                           output_hidden_states=True,
        #                                           output_attentions=True,
        #                                           gradient_checkpointing=False)
        model = BertForSequenceClassification.from_pretrained(diri, config=model_config)
        # model = BertForSequenceClassification.from_pretrained(pretrained_model, config=model_config)
        models[os.path.split(diri)[-2] + '_' + os.path.split(diri)[-1]] = NLIModelWrapper(tokenizer, model, bs)

    # Start the LIT server. See server_flags.py for server options.
    flags = server_flags.get_flags()
    flags['port'] = port
    flags['warm_start'] = warm_start
    lit_demo = dev_server.Server(models, datasets, **flags)
    lit_demo.serve()


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
        model.save_pretrained(out_dir)
        print('*** Model saved')


def parse_cli():
    p = argparse.ArgumentParser(description='Experiments')
    sp = p.add_subparsers(help='Sub-commands')

    # Experiment config
    sp_exp = sp.add_parser('run-irm', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_irm)

    # <editor-fold desc="run-irm params">
    # data and model params
    sp_exp.add_argument('--out-dir', type=str,
                        help='Name dir to save results',
                        default='.')
    sp_exp.add_argument('--dataset', type=str,
                        help='Dataset to train on',
                        default='SNLI')
    sp_exp.add_argument('--num-labels', type=int,
                        help='Number of label types',
                        default=3)
    sp_exp.add_argument('--pretrained-model', type=str,
                        help='Name of the huggingface model', default='bert-base-uncased')
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        required=False)
    sp_exp.add_argument('--train-env-prob', nargs='*', type=float,
                        help='Bias probabilities per training environment',
                        default=[0.8, 0.9])
    sp_exp.add_argument('--val-env-prob', nargs='*', type=float,
                        help='Bias probability for validation environment',
                        default=[0.8, 0.9])
    sp_exp.add_argument('--val-ood-env-prob', nargs=1, type=float,
                        help='Bias probability for OOD validation environment',
                        default=[0.333])
    sp_exp.add_argument('--biased-samples-ratio', type=float,
                        help='Ratio of samples to be biased from entire dataset',
                        default=1.0)
    sp_exp.add_argument('--bias-tokens-per-label', type=int,
                        help='Ratio of samples to be biased from entire datasetNUmber of bias tokens used per label',
                        default=1, choices=[1, 2, 3, 4, 5])
    sp_exp.add_argument('--bias-pattern', type=str,
                        help='Bias pattern to inject - for simple pattern only one bias token will be appended to the '
                             'hypothesis, for complex pattern multiple bias tokens will be appended',
                        default='simple', choices=['simple', 'complex'])

    # training params
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-val', type=int, help='Val batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--train-size', type=int, help='Train data size')
    sp_exp.add_argument('--val-size', type=int, help='Val data size')
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


    sp_test = sp.add_parser('test-irm', help='Evaluate model on test or validation')
    sp_test.set_defaults(subcmd_fn=test_irm)

    # <editor-fold desc="test-irm params">
    sp_test.add_argument('test_file', type=str,
                         help='File to evaluate model on')
    sp_test.add_argument('test_dir', type=str,
                         help='Name dir to load fine-tuned model')

    sp_test.add_argument('--out-dir', type=str,
                         help='Name dir to save results',
                         default='.')

    sp_test.add_argument('--seed', '-s', type=int, help='Random seed',
                         required=False)
    sp_test.add_argument('--env-prob', type=float, nargs=1,
                         help='Bias probability for validation environment', default=[0.0])
    sp_test.add_argument('--biased-samples-ratio', type=float,
                         help='Ratio of samples to be biased from entire dataset')
    sp_test.add_argument('--bias-tokens', nargs='*', type=str,
                         help='Bias tokens')
    sp_test.add_argument('--bs-test', type=int, help='Batch size',
                         default=32, metavar='BATCH_SIZE')
    sp_test.add_argument('--reg', type=float,
                         help='IRM regularization weight', default=1e3)

    # </editor-fold>

    sp_analyze = sp.add_parser('analyze-irm', help='Analyze model')
    sp_analyze.set_defaults(subcmd_fn=analyze_irm)

    # <editor-fold desc="analyze-irm params">
    sp_analyze.add_argument('model_dir', type=str,
                            help='Directory of model to analyze')
    sp_analyze.add_argument('ref_dir', type=str,
                            help='Parent directory containing reference model runs')
    sp_analyze.add_argument('--model-test-dir', type=str,
                            help='name of subdirectory of the model test runs', default='test')
    sp_analyze.add_argument('--model-attribution-dir', type=str,
                            help='name of subdirectory of the model attribution runs')
    sp_analyze.add_argument('--ref-test-dir', type=str,
                            help='Name test dir to look for in the ref dir to get predictions for test file',
                            default='test')
    sp_analyze.add_argument('--scores-file', type=str,
                            help='Scores file')
    sp_analyze.add_argument('--scores-set', type=str,
                            help='Scores file', choices=['train', 'val', 'test'],
                            default='test')
    sp_analyze.add_argument('--scores-bias', type=str,
                            help='Name of bias reflected by scores',
                            default='')
    sp_analyze.add_argument('--out-dir', type=str,
                            help='Name dir to save results',
                            default='.')

    sp_analyze.add_argument('--bs', type=int, help='Batch size',
                            default=8, metavar='BATCH_SIZE')
    # </editor-fold>

    sp_attribute = sp.add_parser('attribute-irm', help='Attribute predictions to model embeddings')
    sp_attribute.set_defaults(subcmd_fn=attribute_irm)

    # <editor-fold desc="attribute-irm params">
    sp_attribute.add_argument('test_file', type=str,
                              help='File to analyze model on')
    sp_attribute.add_argument('test_dir', type=str,
                              help='Name dir to load fine-tuned model')
    sp_attribute.add_argument('--out-dir', type=str,
                              help='Name dir to save results',
                              default='.')

    sp_attribute.add_argument('--env-prob', type=float, nargs=1,
                              help='Bias probability for validation environment', default=[0.0])
    sp_attribute.add_argument('--biased-samples-ratio', type=float,
                              help='Ratio of samples to be biased from entire dataset')
    sp_attribute.add_argument('--bias-tokens', nargs='*', type=str,
                              help='Bias tokens')
    sp_attribute.add_argument('--seed', '-s', type=int, help='Random seed',
                              required=False)
    sp_attribute.add_argument('--bs', type=int, help='Batch size',
                              default=32, metavar='BATCH_SIZE')
    # </editor-fold>

    sp_lit = sp.add_parser('run-lit', help='Language Interpretability Tool')
    sp_lit.set_defaults(subcmd_fn=run_lit)

    # <editor-fold desc="run-lit params">
    sp_lit.add_argument('test_file', type=str,
                        help='File to analyze model on')
    sp_lit.add_argument('test_dir', type=str, nargs='*',
                        help='Name dir to load fine-tuned model')
    sp_lit.add_argument('--port', type=int, help='port',
                        default=5432)
    sp_lit.add_argument('--warm-start', type=float, help='fraction of dataset to run model when initializing',
                        default=0.0)
    sp_lit.add_argument('--bs', type=int, help='batch size',
                        default=32)
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
    # run_irm(out_dir='results',
    #         # bias params
    #         train_env_prob=(0.7, 0.9), val_env_prob=(0.7, 0.9), val_ood_env_prob=(0.33,), biased_samples_ratio=1.0,
    #         # Training params
    #         bs_train=2, bs_val=2, train_size=1000,
    #         eval_every_x_epoch=0.2, epochs=0, warm_up_epochs=1, early_stopping=3,
    #         reg=1e3, warm_up_reg=1.0, gradient_checkpoint=False,
    #         # optimizer params
    #         optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
    #         epsilon=1e-8, weight_decay=0.01, amsgrad=False,
    #         lr_scheduling=False, lr_scheduling_rate=0.1
    #         )
    # run_irm_LBTW(out_dir='results',
    #              # bias params
    #              train_env_prob=(0.8, 0.9), val_env_prob=(0.8, 0.9), val_ood_env_prob=(1 / 3,),
    #              biased_samples_ratio=0.0,
    #              # Training params
    #              bs_train=2, bs_val=2, eval_every_x_epoch=0.2, epochs=4, warm_up_epochs=2, early_stopping=3,
    #              reg=1e3, warm_up_reg=1.0, alpha=0.1, gradient_checkpoint=False,
    #              # optimizer params
    #              optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
    #              epsilon=1e-8, weight_decay=0.01, amsgrad=False,
    #              lr_scheduling=False, lr_scheduling_rate=0.1
    #              )
    # test_irm('data/SNLI/snli_1.0_train_temp.txt', 'results', out_dir='results/test', seed=None,
    #          env_prob=(0.8,), biased_samples_ratio=0.5, bias_tokens=('<c>', '<e>', '<n>'),
    #          bs_test=32, reg=1e3
    #          )
    # attribute_irm('data/SNLI/snli_1.0_dev.txt', 'results', out_dir='results/attribution',
    #               seed=None, bs=8, manual=False, samp_ind=None)
    # analyze_irm('results/erm', model_test_dir='test', model_attribution_dir='attribution_contradiction',
    #             ref_dir='results/ref2', ref_test_dir='test', out_dir='temp', bs=8)

    # from bias_utils import list_vocab_unused_tokens
    # import numpy as np
    #
    # dataset = 'SNLI'
    # file_train, file_val, file_test = prepare_dataset(dataset)
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #
    # ds_train = create_datasets(file_train, num_datasets=1, biased_samples_ratio=0.0, env_prob=(1.0,))[0]
    # ds_val = create_datasets(file_val, num_datasets=1, biased_samples_ratio=0.0, env_prob=(1.0,))[0]
    # ds_test = create_datasets(file_test, num_datasets=1, biased_samples_ratio=0.0, env_prob=(1.0,))[0]
    # temp = []
    # # unused_tokens = set(list_vocab_unused_tokens(ds_train, tokenizer)) & \
    # #                 set(list_vocab_unused_tokens(ds_val, tokenizer)) & \
    # #                 set(list_vocab_unused_tokens(ds_test, tokenizer))
    # a = set(list_vocab_unused_tokens(ds_train, tokenizer))
    # b = set(list_vocab_unused_tokens(ds_val, tokenizer))
    # c = set(list_vocab_unused_tokens(ds_test, tokenizer))
    # # unused_tokens = list(unused_tokens)
    # # bias_tokens = np.random.choice(unused_tokens, 12)

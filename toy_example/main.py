from dataset_utils import NLIDataset
from torch.utils.data import DataLoader, SequentialSampler
from torch.optim import AdamW, SGD
from transformers import get_linear_schedule_with_warmup
from model_utils import NLINet
from train_utils import IRMTrainer
import torch.nn as nn
import numpy as np
import os
import sys
import argparse
import torch
import json
from torch.utils.tensorboard import SummaryWriter
from setting import VOCAB_SIG, VOCAB_BIAS, VOCAB, NUM_LABELS


def run_irm(out_dir='.', seed=None, embedd_dim=10, hidden_dim=10, num_layers=1, multi_class=True,
            # bias params
            noise=0.0, biased_samples_ratio=1.0, train_env_prob=(0.8, 0.9), val_env_prob=(0.8, 0.9), val_ood_env_prob=1/3,
            # Training params
            bs_train=32, bs_val=32, batches_per_step=10, steps=20, warm_up_steps=5, early_stopping=3,
            reg=1e3, warm_up_reg=1.0,
            # optimizer params
            optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
            epsilon=1e-8, weight_decay=0.01, amsgrad=False,
            lr_scheduling=False, lr_scheduling_rate=0.1
            ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals().copy()
    checkpoint_dir = os.path.sep.join([out_dir, "checkpoints"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed = nn.Embedding(len(VOCAB)+1, embedd_dim, padding_idx=len(VOCAB))
    model = NLINet(embed, num_layers=num_layers, hidden_dim=hidden_dim, multi_class=multi_class).to(device=device)
    if multi_class:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()

    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    ds_train = [NLIDataset(10000, noise=noise, biased_samples_ratio=biased_samples_ratio, prob=q, rng=rng)
                for q in train_env_prob]
    ds_val = [NLIDataset(1000, noise=noise, biased_samples_ratio=biased_samples_ratio, prob=q, rng=rng)
              for q in val_env_prob]
    ds_val_ood = [NLIDataset(1000, noise=0.0, biased_samples_ratio=1.0, prob=val_ood_env_prob, rng=rng)]
    dl_train = [DataLoader(env, batch_size=bs_train, shuffle=True) for env in ds_train]
    dl_val = [DataLoader(env, batch_size=bs_val) for env in ds_val]
    dl_val_ood = [DataLoader(env, batch_size=bs_val) for env in ds_val_ood]

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

    trainer = IRMTrainer(model, loss_fn, optimizer, scheduler, device=device)

    res = trainer.fit(dl_train, dl_val, dl_val_ood, warm_up_steps=warm_up_steps, irm_steps=steps,
                      warm_up_reg=warm_up_reg, irm_reg=reg, checkpoint_dir=checkpoint_dir,
                      early_stopping=early_stopping,
                      max_batches=batches_per_step)

    # save last checkpointed model
    checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
    saved_state = torch.load(checkpoint_filename)
    model.load_state_dict(saved_state['model_state_dict'])
    save_experiment(out_dir, run_config, res, model)
    os.rename(f'{os.path.sep.join([checkpoint_dir, "model_predictions"])}.png',
              f'{os.path.sep.join([out_dir, "model_predictions"])}.png')
    writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]), flush_secs=10)
    for k in ['train_env_prob', 'val_env_prob']:
        run_config[k] = str(run_config[k])
    writer.add_hparams(run_config,
                       {'hparam/train_accuracy': res.train_acc[-1], 'hparam/val_accuracy': res.test_acc[-1],
                        'hparam/val_ood_accuracy': res.test_acc_ood[-1]})

    return res


def test_irm(test_dir, out_dir='.', seed=None,
             noise=0.0, biased_samples_ratio=1.0, env_prob=0.0,
             bs_test=32, reg=1e3
             ):
    if not seed:
        seed = np.random.randint(0, 2 ** 31)
    torch.manual_seed(seed)
    run_config = locals()

    assert os.path.isdir(test_dir), "Model directory doesn't exist"
    with open(f'{os.path.sep.join([test_dir, "run_output"])}.json') as config_file:
        pretrained_cfg = json.load(config_file)['config']
    embedd_dim, hidden_dim, num_layers, multi_class = pretrained_cfg['embedd_dim'], pretrained_cfg['hidden_dim'], \
                                                      pretrained_cfg['num_layers'], pretrained_cfg['multi_class'], \

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embed = nn.Embedding(len(VOCAB) + 1, embedd_dim, padding_idx=len(VOCAB))
    model = NLINet(embed, num_layers=num_layers, hidden_dim=hidden_dim, multi_class=multi_class)
    model_filename = f'{os.path.sep.join([test_dir, "pytorch_model"])}.bin'
    model.load_state_dict(torch.load(model_filename))
    model = model.to(device=device)
    if multi_class:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.BCEWithLogitsLoss()
    rng = np.random.RandomState(seed)

    # create biased datasets by appending unused tokens to hypothesis
    ds_test = NLIDataset(1000, noise=noise, biased_samples_ratio=biased_samples_ratio, prob=env_prob, rng=rng)
    dl_test = [DataLoader(ds_test, batch_size=bs_test)]

    tester = IRMTrainer(model, loss_fn, device=device)

    res = tester.test(dl_test, reg=reg)

    save_experiment(out_dir, run_config, res)

    return res


def save_experiment(out_dir, config, res, model=None):
    output = dict(
        config=config,
        results=res._asdict()
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
    sp_exp = sp.add_parser('run-irm', help='Run experiment with a single '
                                           'configuration')
    sp_exp.set_defaults(subcmd_fn=run_irm)

    # <editor-fold desc="run-irm params">
    # data and model params
    sp_exp.add_argument('--out-dir', type=str,
                        help='Name dir to save results',
                        default='.')
    sp_exp.add_argument('--seed', '-s', type=int, help='Random seed',
                        required=False)
    sp_exp.add_argument('--embedd-dim', type=int,
                        help='Embedding dimension of the model (both premise and hypothesis)',
                        default=10)
    sp_exp.add_argument('--hidden-dim', type=int,
                        help='Bias probability for OOD validation environment',
                        default=10)
    sp_exp.add_argument('--num-layers', type=int,
                        help='Number of hidden layers in model',
                        default=1)
    sp_exp.add_argument('--multi-class',
                        help='Multi-class classification - output dim is 2', action='store_true')
    sp_exp.add_argument('--noise', type=float,
                         help='Noise - float in [0.0, 1.0] expressing probability of flipping label', default=0.0)
    sp_exp.add_argument('--biased-samples-ratio', type=float,
                         help='Float in [0.0, 1.0] expressing ratio of biased samples from entire dataset', default=1.0)
    sp_exp.add_argument('--train-env-prob', nargs='*', type=float,
                        help='Bias probabilities per training environment',
                        default=[0.8, 0.9])
    sp_exp.add_argument('--val-env-prob', nargs='*', type=float,
                        help='Bias probability for validation environment',
                        default=[0.8, 0.9])
    sp_exp.add_argument('--val-ood-env-prob', type=float,
                        help='Bias probability for OOD validation environment',
                        default=1/3)

    # training params
    sp_exp.add_argument('--bs-train', type=int, help='Train batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--bs-val', type=int, help='Val batch size',
                        default=32, metavar='BATCH_SIZE')
    sp_exp.add_argument('--batches-per-step', type=int, help='Batches per training step',
                        default=None)
    sp_exp.add_argument('--steps', type=int,
                        help='Maximal number of constrained steps', default=20)
    sp_exp.add_argument('--warm-up-steps', type=int,
                        help='Maximal number of warm up steps', default=5)
    sp_exp.add_argument('--early-stopping', type=int,
                        help='Stop after this many epochs without improvement', default=3)
    sp_exp.add_argument('--reg', type=float,
                        help='IRM regularization weight', default=1e3)
    sp_exp.add_argument('--warm-up-reg', type=float,
                        help='IRM regularization weight for warm up', default=1.0)

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
    sp_test.add_argument('test_dir', type=str,
                         help='Name dir to load fine-tuned model')

    sp_test.add_argument('--out-dir', type=str,
                         help='Name dir to save results',
                         default='.')

    sp_test.add_argument('--seed', '-s', type=int, help='Random seed',
                         required=False)
    sp_test.add_argument('--noise', type=float,
                         help='Noise - float in [0.0, 1.0] expressing probability of flipping label', default=0.0)
    sp_test.add_argument('--biased-samples-ratio', type=float,
                         help='Float in [0.0, 1.0] expressing ratio of biased samples from entire dataset', default=1.0)
    sp_test.add_argument('--env-prob', type=float,
                         help='Bias probability for validation environment', default=0.0)
    sp_test.add_argument('--bs-test', type=int, help='Batch size',
                         default=32, metavar='BATCH_SIZE')
    sp_test.add_argument('--reg', type=float,
                         help='IRM regularization weight', default=1e3)

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
    # run_irm(out_dir='temp', seed=None, embedd_dim=2, hidden_dim=2, num_layers=1, multi_class=False,
    #         # bias params
    #         noise=0.25, biased_samples_ratio=0.5, train_env_prob=(0.8, 0.9), val_env_prob=(0.8, 0.9), val_ood_env_prob=1/3,
    #         # Training params
    #         bs_train=32, bs_val=32, batches_per_step=2, steps=1, warm_up_steps=2, early_stopping=3,
    #         reg=1e3, warm_up_reg=1.0,
    #         # optimizer params
    #         optimizer_type='Adam', lr=1e-5, momentum=0.9, beta1=0.9, beta2=0.999,
    #         epsilon=1e-8, weight_decay=0.01, amsgrad=False,
    #         lr_scheduling=False, lr_scheduling_rate=0.1
    #         )
    # test_irm('temp', out_dir='temp/test2', seed=None,
    #          noise=0.0, biased_samples_ratio=1.0, env_prob=0.1,
    #          bs_test=32, reg=1e3
    #          )
    # from utils import plot_predictions
    #
    # preds_dict = {('a', 'a c', 0): 60 * [0.2] + 60 * [0.3], ('a', 'a c', 1): 60 * [0.2] + 60 * [0.3],
    #               ('a', 'a d', 0): 60 * [0.7] + 60 * [0.8], ('a', 'a d', 1): 60 * [0.7] + 60 * [0.8],
    #               ('a', 'b c', 0): 60 * [0.12] + 60 * [0.27], ('a', 'b c', 1): 60 * [0.12] + 60 * [0.27],
    #               ('a', 'b d', 0): 60 * [0.74] + 60 * [0.84], ('a', 'b d', 1): 60 * [0.74] + 60 * [0.84]
    #     , ('b', 'a c', 0): 60 * [0.08] + 60 * [0.14], ('b', 'a c', 1): 60 * [0.08] + 60 * [0.14],
    #               ('b', 'a d', 0): 60 * [0.78] + 60 * [0.9], ('b', 'a d', 1): 60 * [0.78] + 60 * [0.9]
    #     , ('b', 'b c', 0): 60 * [0.25] + 60 * [0.32], ('b', 'b c', 1): 60 * [0.25] + 60 * [0.32],
    #               ('b', 'b d', 0): 60 * [0.82] + 60 * [0.95], ('b', 'b d', 1): 60 * [0.82] + 60 * [0.95]}
    # fig = plot_predictions(preds_dict)
    # fig.show()

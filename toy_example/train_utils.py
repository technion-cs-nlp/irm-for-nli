import abc
import os
import sys
import tqdm
import torch
import math
from torch.utils.data import DataLoader
from typing import Callable, Any, Iterator
import numpy as np
from pathlib import Path
from transformers import BertForSequenceClassification
from torch.nn import CrossEntropyLoss
from torch.autograd import grad
from utils import *
from typing import List
from torch.utils.tensorboard import SummaryWriter
# from pytorch_memlab import MemReporter
from sklearn.metrics import precision_score, recall_score, confusion_matrix


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training/testing models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit/test)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, optimizer, scheduler, device):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param optimizer: The optimizer to train with.
        :param scheduler: The lr scheduler to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_labels = 2

    def fit(self, dl_train: List[DataLoader], dl_test: List[DataLoader], dl_test_ood: List[DataLoader],
            warm_up_steps=5, irm_steps=20, checkpoint_dir: str = None, early_stopping: int = None,
            warm_up_reg=1.0, irm_reg=1000.0,
            print_every=10, post_epoch_fn=None, max_batches=None) -> FitResult:
        """
        Trains the model for multiple steps with a given training set,
        and calculates validation loss over two given validation sets: in-domain and out-of-domain.
        The training is divided into two phases - warm up phase in which the regularization weight should be
        relatively small (allowing the model to converge to a good place in parameter space) and a constrained phase
        enforcing the regularization constraint.
        Early stopping is implemented for both phases. Early stopping during the warm up phase will skip to the
        constrained phase, early stopping the constrained phase terminates training.
        For each step - the model trains on at most max_batches of the training set, and then evaluates on the entire
        test and test_ood sets. Lower max_batches allows for higher evaluation granularity.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the in-distribution test set.
        :param dl_test_ood: Dataloader for the out-of-distribution test set.
        :param warm_up_steps: Number of warm up steps to train for.
        :param irm_steps: Number of constrained steps to train for.
        :param checkpoint_dir: Whether to save model to file every time the
                                test set accuracy improves. Should be a string containing a dir without extension.
        :param early_stopping: Whether to stop training early if there is no
                                improvement for this number of steps.
        :param warm_up_reg: Weight of constraint for warm up phase.
        :param irm_reg: Weight of constraint for constrained phase.
        :param print_every: Print progress every this number of epochs.
        :param post_epoch_fn: A function to call after each epoch completes.
        :param max_batches: Maximal number of batches to train on each step. If None - each step trains on entire
                            train set.
        :return: A FitResult object containing train and test losses per epoch.
        """
        assert self.optimizer is not None, "Only testing possible if no optimizer passed"

        actual_num_steps = 0
        actual_num_warm_up_steps = None
        train_loss, train_error, train_penalty, train_acc = [], [], [], []
        test_loss, test_error, test_penalty, test_acc = [], [], [], []
        test_loss_ood, test_error_ood, test_penalty_ood, test_acc_ood = [], [], [], []

        best_loss, best_train_error, best_test_error = None, None, None
        steps_without_improvement = 0
        dl_train_iter_len = 0

        unique_samples_predictions = None

        checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Load state from checkpoint file if it exists
        if os.path.isfile(checkpoint_filename):
            print(f'*** Loading checkpoint file {checkpoint_filename}')
            saved_state = torch.load(checkpoint_filename,
                                     map_location=self.device)
            unique_samples_predictions = saved_state.get('unique_samples_predictions', unique_samples_predictions)
            best_loss = saved_state.get('best_loss', best_loss)
            best_train_error = saved_state.get('best_train_error', best_train_error)
            best_test_error = saved_state.get('best_test_error', best_test_error)
            steps_without_improvement = \
                saved_state.get('swi', steps_without_improvement)
            train_loss = saved_state.get('train_loss', train_loss)
            train_acc = saved_state.get('train_acc', train_acc)
            train_error = saved_state.get('train_loss', train_error)
            train_penalty = saved_state.get('train_acc', train_penalty)
            test_loss = saved_state.get('test_loss', test_loss)
            test_acc = saved_state.get('test_acc', test_acc)
            test_error = saved_state.get('test_loss', test_error)
            test_penalty = saved_state.get('test_acc', test_penalty)
            test_loss_ood = saved_state.get('test_loss', test_loss_ood)
            test_acc_ood = saved_state.get('test_acc', test_acc_ood)
            test_error_ood = saved_state.get('test_loss', test_error_ood)
            test_penalty_ood = saved_state.get('test_acc', test_penalty_ood)
            actual_num_steps = saved_state.get('ans', actual_num_steps)
            actual_num_warm_up_steps = saved_state.get('ans_warm_up', actual_num_warm_up_steps)
            writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]), purge_step=actual_num_steps,
                                   flush_secs=10)
            warm_up_steps = saved_state.get('warm_up_steps', warm_up_steps)
            irm_steps = saved_state.get('irm_steps', irm_steps)
            warm_up_reg = saved_state.get('warm_up_reg', warm_up_reg)
            irm_reg = saved_state.get('irm_reg', irm_reg)
            self.optimizer.load_state_dict(saved_state['optimizer_state_dict'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(saved_state['scheduler_state_dict'])
            print(f"Loading model from {checkpoint_dir}")
            self.model.load_state_dict(saved_state['model_state_dict'])
            self.model.to(self.device)
        else:
            writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]), flush_secs=10)

        num_steps = warm_up_steps + irm_steps

        while actual_num_steps < num_steps:
            step = actual_num_steps
            save_checkpoint = False
            verbose = False  # pass this to train/test_epoch.
            if step % print_every == 0 or step == num_steps - 1:
                verbose = True
            self._print(f'--- STEP {step + 1}/{num_steps} ---', verbose)

            best_loss, best_train_error, best_test_error = list(map(lambda x: x if x is not None else math.inf,
                                                                    [best_loss, best_train_error, best_test_error]))
            if actual_num_steps < warm_up_steps:
                reg = warm_up_reg
            else:
                reg = irm_reg

            # <editor-fold desc="train and evaluate">
            # train
            if dl_train_iter_len <= 0:
                dl_train_iter = iter(zip(*dl_train))
                dl_train_iter_len = len(dl_train[0])
            train_result = self.train_epoch(dl_train_iter, reg, max_batches=max_batches, verbose=verbose)
            dl_train_iter_len = dl_train_iter_len - max_batches if max_batches is not None else 0
            update_results([(train_loss, 'losses'), (train_error, 'errors'), (train_penalty, 'penalties'),
                            (train_acc, 'accuracy')], train_result)
            tensorboard_add_result(writer, train_result, actual_num_steps, 'train')
            writer.add_scalar('warm_up/train', 1 if actual_num_steps < warm_up_steps else 0, actual_num_steps)
            writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], actual_num_steps)

            # val
            dl_test_iter = iter(zip(*dl_test))
            test_result = self.test_epoch(dl_test_iter, reg, verbose=verbose)
            update_results([(test_loss, 'losses'), (test_error, 'errors'), (test_penalty, 'penalties'),
                            (test_acc, 'accuracy')], test_result)
            tensorboard_add_result(writer, test_result, actual_num_steps, 'val')

            # val OOD
            dl_test_ood_iter = iter(zip(*dl_test_ood))
            test_result_ood = self.test_epoch(dl_test_ood_iter, reg, verbose=verbose)
            update_results([(test_loss_ood, 'losses'), (test_error_ood, 'errors'), (test_penalty_ood, 'penalties'),
                            (test_acc_ood, 'accuracy')], test_result_ood)
            tensorboard_add_result(writer, test_result_ood, actual_num_steps, 'val_ood')
            # </editor-fold>

            unique_samples_predictions = self.analyze(dl_train[0].dataset, unique_samples_predictions)

            # <editor-fold desc="early stopping">
            loss_increase = test_loss[-1] >= best_loss
            error_overfit = train_error[-1] < best_train_error and test_error[-1] >= best_test_error
            if early_stopping and (loss_increase or error_overfit):
                steps_without_improvement += 1
                if steps_without_improvement >= early_stopping:
                    if actual_num_steps < warm_up_steps:
                        actual_num_warm_up_steps = actual_num_steps + 1
                        actual_num_steps = warm_up_steps - 1  # if early stopping for warm up - skip to post warm up
                        steps_without_improvement = 0
                    else:
                        actual_num_steps += 1  # update before breaking out of loop
                        break
            else:
                steps_without_improvement = 0
                save_checkpoint = True

            best_loss, best_train_error, best_test_error = list(map(lambda x: min(x),
                                                                    [(test_loss[-1], best_loss),
                                                                     (train_error[-1], best_train_error),
                                                                     (test_error[-1], best_test_error)]))

            # </editor-fold>

            actual_num_steps += 1
            # ========================

            # Save model checkpoint if requested
            if save_checkpoint:
                saved_state = dict(unique_samples_predictions=unique_samples_predictions,
                                   best_loss=best_loss,
                                   best_train_error=best_train_error,
                                   best_test_error=best_test_error,
                                   swi=steps_without_improvement,
                                   train_loss=train_loss,
                                   train_acc=train_acc,
                                   train_error=train_error,
                                   train_penalty=train_penalty,
                                   test_loss=test_loss,
                                   test_acc=test_acc,
                                   test_error=test_error,
                                   test_penalty=test_penalty,
                                   test_loss_ood=test_loss_ood,
                                   test_acc_ood=test_acc_ood,
                                   test_error_ood=test_error_ood,
                                   test_penalty_ood=test_penalty_ood,
                                   ans=actual_num_steps,
                                   ans_warm_up=actual_num_warm_up_steps,
                                   warm_up_steps=warm_up_steps,
                                   irm_steps=irm_steps,
                                   warm_up_reg=warm_up_reg,
                                   irm_reg=irm_reg,
                                   optimizer_state_dict=self.optimizer.state_dict(),
                                   scheduler_state_dict=self.scheduler.state_dict() if self.scheduler is not None else None,
                                   model_state_dict=self.model.state_dict()
                                   )
                torch.save(saved_state, checkpoint_filename)
                print(f'*** Saved checkpoint {checkpoint_filename} '
                      f'at step {step + 1}')

            if post_epoch_fn:
                post_epoch_fn(step, train_result, test_result, verbose)

        ans_warm_up = actual_num_warm_up_steps if actual_num_warm_up_steps is not None else warm_up_steps
        if irm_steps and irm_reg:
            fig = plot_predictions(unique_samples_predictions, ans_warm_up)
        else:
            fig = plot_predictions(unique_samples_predictions)
        writer.add_figure('Model Predictions', fig)
        fig.savefig(f'{os.path.sep.join([checkpoint_dir, "model_predictions"])}.png')
        return FitResult(num_warm_up_steps=ans_warm_up, num_steps=actual_num_steps,
                         train_loss=train_loss, train_error=train_error, train_penalty=train_penalty,
                         train_acc=train_acc,
                         test_loss=test_loss, test_error=test_error, test_penalty=test_penalty, test_acc=test_acc,
                         test_loss_ood=test_loss_ood, test_error_ood=test_error_ood, test_penalty_ood=test_penalty_ood,
                         test_acc_ood=test_acc_ood)

    def test(self, dl_test: List[DataLoader], reg=1000.0) -> TestResult:

        dl_test_iter = iter(zip(*dl_test))
        test_epoch_result = self.test_epoch(dl_test_iter, reg, verbose=False)
        test_epoch_loss = torch.tensor(test_epoch_result.losses).mean().item()
        test_epoch_acc = test_epoch_result.accuracy

        return TestResult(test_epoch_loss, test_epoch_acc)

    def analyze(self, ds, unique_samples_predictions=None):
        """
        Add documentation
        """
        unique_samples = list(set(ds.samples))
        unique_samples_ind = [ds.samples.index(samp) for samp in unique_samples]
        if unique_samples_predictions is None:
            unique_samples_predictions = dict([(samp, []) for samp in unique_samples])

        for ind, samp in zip(unique_samples_ind, unique_samples):
            p, h, y = ds[ind]
            with torch.no_grad():
                logits, _ = self.model((p.unsqueeze(0).to(self.device), h.unsqueeze(0).to(self.device)))
                if self.model.output_dim == 2:
                    pred = torch.nn.functional.softmax(logits.detach().to('cpu'), dim=-1)[:, 1].item()
                else:
                    pred = torch.nn.functional.sigmoid(logits.detach().to('cpu')).item()
            unique_samples_predictions[samp].append(pred)
        return unique_samples_predictions

    def train_epoch(self, dl_train: Iterator, reg, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, reg, self.num_labels, **kw)

    def test_epoch(self, dl_test: Iterator, reg, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        # self.model.eval()  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, reg, self.num_labels, **kw)

    @abc.abstractmethod
    def train_batch(self, batch, reg) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch, reg) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult], reg, num_labels,
                       max_batches=None, verbose=True) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        # global return_acc
        losses, errors, penalties = [], [], []
        num_samples, num_correct = 0, 0
        cm = None

        if max_batches is None:
            max_batches = int(1e6)
            verbose = False

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=max_batches,
                       file=pbar_file) as pbar:

            for batch_idx in range(max_batches):
                data = next(dl, None)  # data is tuple of length num_envs (len(dl)) of batches
                if data is not None:
                    num_envs = len(data)
                    num_samples += sum([len(data[i][0]) for i in range(num_envs)])
                    batch_res = forward_fn(data, reg)
                    torch.cuda.empty_cache()

                    pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                    pbar.update()

                    losses.append(batch_res.loss / num_envs)
                    errors.append(batch_res.error / num_envs)
                    penalties.append(batch_res.penalty / num_envs)

                    y_gt, y_pred = torch.tensor(batch_res.gt), torch.tensor(batch_res.pred)
                    if cm is None:
                        with wrap_confusion_matrix(num_labels, y_gt, y_pred) as padded_inputs:
                            cm = confusion_matrix(padded_inputs[0], padded_inputs[1], sample_weight=padded_inputs[2])
                    else:
                        with wrap_confusion_matrix(num_labels, y_gt, y_pred) as padded_inputs:
                            cm += confusion_matrix(padded_inputs[0], padded_inputs[1], sample_weight=padded_inputs[2])

                    num_correct += batch_res.num_correct
                    num_batches = batch_idx + 1
                else:
                    break

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        return EpochResult(losses=losses, errors=errors, penalties=penalties, accuracy=accuracy,
                           cm=np.around(100.0 * cm / cm.sum(), 2))


class IRMTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer=None, scheduler=None, device='cpu'):
        super().__init__(model, optimizer, scheduler, device)
        self.loss_fn = loss_fn
        self.dummy_classifier = torch.nn.Parameter(torch.ones(1, self.model.output_dim)).to(self.device)

    def train_batch(self, batch, reg) -> BatchResult:
        self.optimizer.zero_grad()
        error_item, penalty_item, loss_item, num_correct = 0.0, 0.0, 0.0, 0
        pred_list, gt_list = [], []
        loss_func = self.loss_fn

        for batch_e in batch:
            p, h, y = batch_e  # p: tensor of shape batch x 1, h: tensor of shape batch x 2, y: tensor of shape batch
            y = y.to(self.device)

            logits, _ = self.model((p.to(self.device), h.to(self.device)))

            if self.model.output_dim == 2:
                error = loss_func(logits, y)
                error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            else:
                error = loss_func(logits.squeeze(), y.to(dtype=torch.float))
                error_for_penalty = loss_func((logits * self.dummy_classifier).squeeze(), y.to(dtype=torch.float))
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss.backward()
            loss_item += loss.detach().item()
            error_item += error.detach().item()
            penalty_item += penalty.detach().item()
            # validate
            if self.model.output_dim == 2:
                pred = torch.argmax(logits.detach().to('cpu'), dim=1)
            else:
                pred = (logits.detach().to('cpu') > 0).to(dtype=torch.float).squeeze()
            gt = y.detach().to('cpu')
            num_correct += torch.sum(gt == pred).item()
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, error, penalty, error_for_penalty, logits, y
            torch.cuda.empty_cache()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list)

    def test_batch(self, batch, reg) -> BatchResult:
        error_item, penalty_item, loss_item, num_correct = 0.0, 0.0, 0.0, 0
        pred_list, gt_list = [], []
        loss_func = self.loss_fn

        for batch_e in batch:
            with torch.no_grad():
                p, h, y = batch_e  # p: tensor of shape batch x 1, h: tensor of shape batch x 2, y: tensor of shape batch
                y = y.to(self.device)

                logits, _ = self.model((p.to(self.device), h.to(self.device)))

            if self.model.output_dim == 2:
                error = loss_func(logits, y)
                error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            else:
                error = loss_func(logits.squeeze(), y.to(dtype=torch.float))
                error_for_penalty = loss_func((logits * self.dummy_classifier).squeeze(), y.to(dtype=torch.float))
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss_item += loss.detach().item()
            error_item += error.detach().item()
            penalty_item += penalty.detach().item()
            # validate
            if self.model.output_dim == 2:
                pred = torch.argmax(logits.detach().to('cpu'), dim=1)
            else:
                pred = (logits.detach().to('cpu') > 0).to(dtype=torch.float).squeeze()
            gt = y.detach().to('cpu')
            num_correct += torch.sum(gt == pred).item()
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, error, penalty, error_for_penalty, logits, y

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list)

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
from transformers import AutoModelForSequenceClassification, BertModel
from torch.nn import CrossEntropyLoss
from torch.autograd import grad
from utils import *
from typing import List
from torch.utils.tensorboard import SummaryWriter
#from pytorch_memlab import MemReporter
from sklearn.metrics import precision_score, recall_score, confusion_matrix, f1_score


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training/testing models.
    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit/test)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """

    def __init__(self, model, num_labels, optimizer, scheduler, device):
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
        self.num_labels = num_labels

    def fit(self, dl_train: List[DataLoader], dl_test: List[DataLoader], dl_test_ood: List[DataLoader] = None,
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

        num_envs = len(dl_train)
        actual_num_steps = 0
        actual_num_warm_up_steps = None
        train_loss, train_error, train_penalty, train_acc = [], [], [], []
        train_error_per_env, train_penalty_per_env = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]
        train_accuracy_per_env = [[] for _ in range(num_envs)]
        test_loss, test_error, test_penalty, test_acc = [], [], [], []
        test_error_per_env, test_penalty_per_env = [[] for _ in range(num_envs)], [[] for _ in range(num_envs)]
        test_accuracy_per_env = [[] for _ in range(num_envs)]
        test_loss_ood, test_error_ood, test_penalty_ood, test_acc_ood = [], [], [], []
        test_error_per_env_ood, test_penalty_per_env_ood = [[]], [[]]
        test_accuracy_per_env_ood = [[]]

        best_loss, best_train_error, best_test_error = None, None, None
        steps_without_improvement = 0
        dl_train_iter_len = 0

        checkpoint_filename = f'{os.path.sep.join([checkpoint_dir, "checkpoint_cfg"])}.pt'
        os.makedirs(checkpoint_dir, exist_ok=True)
        # Load state from checkpoint file if it exists
        if os.path.isfile(checkpoint_filename):
            print(f'*** Loading checkpoint file {checkpoint_filename}')
            saved_state = torch.load(checkpoint_filename,
                                     map_location=self.device)
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
            writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]), purge_step=actual_num_steps)
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
            writer = SummaryWriter(os.path.sep.join([checkpoint_dir, "tensorboard"]))

        num_steps = warm_up_steps + irm_steps

        label_int_to_str = dl_train[0].dataset.label_int_to_str

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

            train_loss.append(np.mean(train_result.losses).item())
            train_error.append(np.mean(train_result.errors).item())
            train_penalty.append(np.mean(train_result.penalties).item())
            train_acc.append(train_result.accuracy)
            for i, train_error_per_env_i in enumerate(train_error_per_env):
                train_error_per_env_i.append(np.mean(train_result.errors_per_env[i]).item())
            for i, train_penalty_per_env_i in enumerate(train_penalty_per_env):
                train_penalty_per_env_i.append(np.mean(train_result.penalties_per_env[i]).item())
            for i, train_accuracy_per_env_i in enumerate(train_accuracy_per_env):
                train_accuracy_per_env_i.append(np.mean(train_result.accuracies_per_env[i]).item())

            tensorboard_add_result(writer, train_result, label_int_to_str, actual_num_steps, 'train')
            writer.add_scalar('warm_up/train', 1 if actual_num_steps < warm_up_steps else 0, actual_num_steps)
            writer.add_scalar('lr/train', self.optimizer.param_groups[0]['lr'], actual_num_steps)

            # val
            dl_test_iter = iter(zip(*dl_test))
            test_result = self.test_epoch(dl_test_iter, reg, verbose=verbose)

            test_loss.append(np.mean(test_result.losses).item())
            test_error.append(np.mean(test_result.errors).item())
            test_penalty.append(np.mean(test_result.penalties).item())
            test_acc.append(test_result.accuracy)
            for i, test_error_per_env_i in enumerate(test_error_per_env):
                test_error_per_env_i.append(np.mean(test_result.errors_per_env[i]).item())
            for i, test_penalty_per_env_i in enumerate(test_penalty_per_env):
                test_penalty_per_env_i.append(np.mean(test_result.penalties_per_env[i]).item())
            for i, test_accuracy_per_env_i in enumerate(test_accuracy_per_env):
                test_accuracy_per_env_i.append(np.mean(test_result.accuracies_per_env[i]).item())

            tensorboard_add_result(writer, test_result, label_int_to_str, actual_num_steps, 'val')

            # val OOD
            if dl_test_ood is not None:
                dl_test_ood_iter = iter(zip(*dl_test_ood))
                test_result_ood = self.test_epoch(dl_test_ood_iter, reg, verbose=verbose)

                test_loss_ood.append(np.mean(test_result_ood.losses).item())
                test_error_ood.append(np.mean(test_result_ood.errors).item())
                test_penalty_ood.append(np.mean(test_result_ood.penalties).item())
                test_acc_ood.append(test_result_ood.accuracy)
                for i, test_error_per_env_i_ood in enumerate(test_error_per_env_ood):
                    test_error_per_env_i_ood.append(np.mean(test_result_ood.errors_per_env[i]).item())
                for i, test_penalty_per_env_i_ood in enumerate(test_penalty_per_env_ood):
                    test_penalty_per_env_i_ood.append(np.mean(test_result_ood.penalties_per_env[i]).item())
                for i, ood_test_accuracy_per_env_i in enumerate(test_accuracy_per_env_ood):
                    ood_test_accuracy_per_env_i.append(np.mean(test_result_ood.accuracies_per_env[i]).item())

                tensorboard_add_result(writer, test_result_ood, label_int_to_str, actual_num_steps, 'val_ood')
            # </editor-fold>

            # <editor-fold desc="early stopping">
            if max([warm_up_reg, irm_reg]) > 0.0 and actual_num_steps >= (warm_up_steps - 1):
                loss_increase = (test_loss[-1] - best_loss) / best_loss >= 0.2
                error_overfit = train_error[-1] < best_train_error and test_error[-1] >= best_test_error
                early_stopping_flag = loss_increase or error_overfit
            else:
                loss_increase = (test_loss[-1] - best_loss) / best_loss >= 0.1
                early_stopping_flag = loss_increase
            if early_stopping and early_stopping_flag:
                steps_without_improvement += 1
                if steps_without_improvement >= early_stopping:
                    if actual_num_steps < warm_up_steps:
                        actual_num_warm_up_steps = actual_num_steps + 1
                        actual_num_steps = warm_up_steps - 1  # if early stopping for warm up - skip to post warm up
                        steps_without_improvement = 0  # reset count
                        # load saved model - in case we continue to constrained phase
                        saved_state = torch.load(checkpoint_filename,
                                                 map_location=self.device)
                        self.model.load_state_dict(saved_state['model_state_dict'])
                        self.model.to(self.device)
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
                saved_state = dict(best_loss=best_loss,
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
        return FitResult(num_warm_up_steps=ans_warm_up, num_steps=actual_num_steps-(warm_up_steps - ans_warm_up),
                         train_loss=train_loss, train_error=train_error, train_penalty=train_penalty,
                         train_acc=train_acc,
                         test_loss=test_loss, test_error=test_error, test_penalty=test_penalty, test_acc=test_acc,
                         test_loss_ood=test_loss_ood, test_error_ood=test_error_ood, test_penalty_ood=test_penalty_ood,
                         test_acc_ood=test_acc_ood,
                         #per env
                         train_error_per_env=train_error_per_env, train_penalty_per_env=train_penalty_per_env,
                         train_acc_per_env=train_accuracy_per_env,
                         test_error_per_env=test_error_per_env, test_penalty_per_env=test_penalty_per_env, test_acc_per_env=test_accuracy_per_env,
                         test_error_ood_per_env=test_error_per_env_ood, test_penalty_ood_per_env=test_penalty_per_env_ood,
                         test_acc_ood_per_env=test_accuracy_per_env_ood
                         )

    def test(self, dl_test: List[DataLoader], reg=1000.0) -> TestResult:

        dl_test_iter = iter(zip(*dl_test))
        test_epoch_result = self.test_epoch(dl_test_iter, reg, verbose=False)
        test_epoch_loss = torch.tensor(test_epoch_result.losses).mean().item()
        test_epoch_acc = test_epoch_result.accuracy
        y_true, y_pred = test_epoch_result.y_true, test_epoch_result.y_pred
        test_f1_macro = f1_score(y_true, y_pred, average='macro')
        rep = classification_report(y_true, y_pred, output_dict=True,
                                    target_names=[dl_test[0].dataset.label_int_to_str(x) for x in range(self.num_labels)])
        if test_epoch_result.pred_prob:
            return TestResult(test_epoch_loss, test_epoch_acc,
                              f1_macro=test_f1_macro, classification_report=rep,
                              pred_prob=test_epoch_result.pred_prob)
        else:
            return TestResult(test_epoch_loss, test_epoch_acc,
                              f1_macro=test_f1_macro, classification_report=rep)

    def train_epoch(self, dl_train: Iterator, reg, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train()  # set train mode
        return self._foreach_batch(dl_train, self.train_batch, reg, self.num_labels, **kw)

    def test_epoch(self, dl_test: Iterator, reg, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.eval()  # set evaluation (test) mode
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
        errors_per_env, penalties_per_env = [], []
        pred_prob = []
        accuracies, accuracies_per_env = [], []
        y_gt_epoch, y_pred_epoch = [], []
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
                    num_samples = [len(data[i][0]) for i in range(num_envs)]  # list of len #envs
                    batch_res = forward_fn(data, reg)
                    torch.cuda.empty_cache()

                    pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                    pbar.update()

                    losses.append(batch_res.loss / num_envs)  # list of len #batches
                    errors.append(sum(batch_res.error) / num_envs)  # list of len #batches
                    errors_per_env.append(batch_res.error)  # list of lists: #batches x #envs
                    penalties.append(sum(batch_res.penalty) / num_envs)  # list of len #batches
                    penalties_per_env.append(batch_res.penalty)  # list of lists: #batches x #envs
                    accuracies.append(np.mean(100. * np.array(batch_res.num_correct) / np.array(num_samples)).tolist())  # list of len #batches
                    accuracies_per_env.append((100. * np.array(batch_res.num_correct) / np.array(num_samples)).tolist())  # list of lists: #batches x #envs

                    if batch_res.pred_prob is not None:
                        pred_prob += batch_res.pred_prob

                    y_gt, y_pred = torch.tensor(batch_res.gt), torch.tensor(batch_res.pred)
                    y_gt_epoch += y_gt
                    y_pred_epoch += y_pred
                    if cm is None:
                        with wrap_confusion_matrix(num_labels, y_gt, y_pred) as padded_inputs:
                            cm = confusion_matrix(padded_inputs[0], padded_inputs[1], sample_weight=padded_inputs[2])
                    else:
                        with wrap_confusion_matrix(num_labels, y_gt, y_pred) as padded_inputs:
                            cm += confusion_matrix(padded_inputs[0], padded_inputs[1], sample_weight=padded_inputs[2])

                    num_batches = batch_idx + 1
                else:
                    break

            errors_per_env = list(zip(*errors_per_env))  # reshape list of lists : #batches x #envs -> #envs x #batches
            penalties_per_env = list(
                zip(*penalties_per_env))  # reshape list of lists : #batches x #envs -> #envs x #batches
            accuracies_per_env = list(
                zip(*accuracies_per_env))  # reshape list of lists : #batches x #envs -> #envs x #batches
            avg_loss = sum(losses) / num_batches
            accuracy = np.mean(accuracies).item()
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'Accuracy {accuracy:.1f})')

        if pred_prob:
            return EpochResult(losses=losses, errors=errors, penalties=penalties,
                               errors_per_env=errors_per_env, penalties_per_env=penalties_per_env,
                               accuracy=accuracy, accuracies_per_env=accuracies_per_env,
                               y_true=y_gt_epoch, y_pred=y_pred_epoch,
                               cm=np.around(100.0 * cm / cm.sum(), 2), pred_prob=pred_prob)
        else:
            return EpochResult(losses=losses, errors=errors, penalties=penalties,
                               errors_per_env=errors_per_env, penalties_per_env=penalties_per_env,
                               accuracy=accuracy, accuracies_per_env=accuracies_per_env,
                               y_true=y_gt_epoch, y_pred=y_pred_epoch,
                               cm=np.around(100.0 * cm / cm.sum(), 2))


class HypTrainer(Trainer):
    def __init__(self, model, num_labels, tokenizer, weights=None, optimizer=None, scheduler=None, device='cpu'):
        super().__init__(model, num_labels, optimizer, scheduler, device)
        self.tokenizer = tokenizer
        self.dummy_classifier = torch.nn.Parameter(torch.ones(1, self.num_labels)).to(self.device)
        self.loss_func = CrossEntropyLoss(weight=weights)

    def train_batch(self, batch, reg) -> BatchResult:
        self.optimizer.zero_grad()
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list = [], []
        loss_func = self.loss_func

        for batch_e in batch:
            p, h, y = batch_e
            input_dict = self.tokenizer(h, padding=True, truncation=True, return_tensors='pt')
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            y = y.to(self.device)

            outputs = self.model(**input_dict, labels=y)

            error_e, logits = outputs[:2]
            error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()
            error = error_e

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss.backward()
            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(logits.detach().to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error_e, error, penalty, error_for_penalty, logits, y
            torch.cuda.empty_cache()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list)

    def test_batch(self, batch, reg) -> BatchResult:
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list, predicted_prob = [], [], []
        loss_func = self.loss_func

        for batch_e in batch:
            with torch.no_grad():
                p, h, y = batch_e
                input_dict = self.tokenizer(h, padding=True, truncation=True, return_tensors='pt')
                input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
                y = y.to(self.device)

                outputs = self.model(**input_dict, labels=y)

                error_e, logits = outputs[:2]
            error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()
            error = error_e

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(logits.to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_softmax = torch.nn.functional.softmax(logits.detach().to('cpu'), -1)
            predicted_prob += pred_softmax.tolist()  # list of lists,each sublist is the predicted probs for that sample
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error_e, error, penalty, error_for_penalty, logits, y

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list, pred_prob=predicted_prob)


class HansTrainer(Trainer):
    def __init__(self, model, num_labels, weights=None, optimizer=None, scheduler=None, device='cpu'):
        super().__init__(model, num_labels, optimizer, scheduler, device)
        self.dummy_classifier = torch.nn.Parameter(torch.ones(1, self.num_labels)).to(self.device)
        self.loss_func = CrossEntropyLoss(weight=weights)

    def train_batch(self, batch, reg) -> BatchResult:
        self.optimizer.zero_grad()
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list = [], []
        loss_func = self.loss_func

        for batch_e in batch:
            p, h, features, y = batch_e

            y = y.to(self.device)
            outputs = self.model(features.to(self.device))

            error = loss_func(outputs, y)
            error_for_penalty = loss_func(outputs * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss.backward()
            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(outputs.detach().to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error, penalty, error_for_penalty, y
            torch.cuda.empty_cache()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list)

    def test_batch(self, batch, reg) -> BatchResult:
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list, predicted_prob = [], [], []
        loss_func = self.loss_func

        for batch_e in batch:
            with torch.no_grad():
                p, h, features, y = batch_e

                y = y.to(self.device)
                outputs = self.model(features.to(self.device))

                error = loss_func(outputs, y)
            error_for_penalty = loss_func(outputs * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(outputs.to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_softmax = torch.nn.functional.softmax(outputs.detach().to('cpu'), -1)
            predicted_prob += pred_softmax.tolist()  # list of lists,each sublist is the predicted probs for that sample
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error, penalty, error_for_penalty, y

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct,
                           pred=pred_list, gt=gt_list, pred_prob=predicted_prob)


class IRMTrainer(Trainer):
    def __init__(self, model, num_labels, tokenizer, weights=None, optimizer=None, scheduler=None, device='cpu'):
        super().__init__(model, num_labels, optimizer, scheduler, device)
        self.tokenizer = tokenizer
        self.dummy_classifier = torch.nn.Parameter(torch.ones(1, self.num_labels)).to(self.device)
        self.loss_func = CrossEntropyLoss(weight=weights)

    def train_batch(self, batch, reg) -> BatchResult:
        self.optimizer.zero_grad()
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list = [], []
        loss_func = self.loss_func

        for batch_e in batch:
            p, h, y = batch_e
            input_dict = self.tokenizer(p, h, padding=True, truncation=True, return_tensors='pt')
            input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
            y = y.to(self.device)

            outputs = self.model(**input_dict, labels=y)

            error_e, logits = outputs[:2]
            error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()
            error = error_e

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss.backward()
            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(logits.detach().to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error_e, error, penalty, error_for_penalty, logits, y
            torch.cuda.empty_cache()

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct, pred=pred_list, gt=gt_list)

    def test_batch(self, batch, reg) -> BatchResult:
        error_item, penalty_item, loss_item, num_correct = [], [], 0.0, []
        pred_list, gt_list = [], []
        loss_func = self.loss_func

        for batch_e in batch:
            with torch.no_grad():
                p, h, y = batch_e
                input_dict = self.tokenizer(p, h, padding=True, truncation=True, return_tensors='pt')
                input_dict = {k: v.to(self.device) for k, v in input_dict.items()}
                y = y.to(self.device)

                outputs = self.model(**input_dict, labels=y)

                error_e, logits = outputs[:2]
            error_for_penalty = loss_func(logits * self.dummy_classifier, y)
            penalty = ((grad(error_for_penalty, self.dummy_classifier, create_graph=True)[0]) ** 2).sum()
            error = error_e

            if reg <= 1.0:
                loss = (error + reg * penalty)
            else:
                loss = (error + reg * penalty) / reg

            loss_item += loss.detach().item()
            error_item.append(error.detach().item())
            penalty_item.append(penalty.detach().item())
            # validate
            pred = torch.argmax(logits.to('cpu'), dim=1)
            gt = y.detach().to('cpu')
            num_correct.append(torch.sum(gt == pred).item())
            pred_list += pred.tolist()
            gt_list += gt.tolist()
            del loss, outputs, error_e, error, penalty, error_for_penalty, logits, y

        return BatchResult(loss=loss_item, error=error_item, penalty=penalty_item, num_correct=num_correct, pred=pred_list, gt=gt_list)

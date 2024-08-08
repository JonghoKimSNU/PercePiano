import torch
import numpy as np
from skorch import NeuralNetRegressor
from modAL.models import ActiveLearner

from pathlib import Path
import shutil
import random
import math
import time
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import torch as th
import wandb
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import ScorePerformDataset
import utils

from train_m2pf import LABEL_LIST, LABEL_LIST19, prepare_directories_and_logger, load_model


def uniform(learner, X, n_instances=1):
    query_idx = np.random.choice(
        range(len(X)), size=n_instances, replace=False)
    return query_idx, X[query_idx]


def max_entropy(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([learner.forward(X[random_subset], training=True).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis=0)
    acquisition = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]


def bald(learner, X, n_instances=1, T=100):
    random_subset = np.random.choice(range(len(X)), size=2000, replace=False)
    with torch.no_grad():
        outputs = np.stack([learner.forward(X[random_subset], training=True).cpu().numpy()
                            for t in range(100)])
    pc = outputs.mean(axis=0)
    H = (-pc*np.log(pc + 1e-10)).sum(axis=-1)
    E_H = - np.mean(np.sum(outputs * np.log(outputs + 1e-10),
                    axis=-1), axis=0)  # [batch size]
    acquisition = H - E_H
    idx = (-acquisition).argsort()[:n_instances]
    query_idx = random_subset[idx]
    return query_idx, X[query_idx]


def active_learning_procedure(query_strategy,
                              X_test,
                              y_test,
                              X_pool,
                              y_pool,
                              X_initial,
                              y_initial,
                              estimator,
                              n_queries=30, # 100
                              n_instances=10):
    learner = ActiveLearner(estimator=estimator,
                            X_training=X_initial,
                            y_training=y_initial,
                            query_strategy=query_strategy,
                            )
    perf_hist = [learner.score(X_test, y_test)]
    for index in range(n_queries):
        # Finds the n_instances most informative point in the data provided by calling the query_strategy function.
        query_idx, query_instance = learner.query(X_pool, n_instances) 
        learner.teach(X_pool[query_idx], y_pool[query_idx])
        X_pool = np.delete(X_pool, query_idx, axis=0)
        y_pool = np.delete(y_pool, query_idx, axis=0)
        model_accuracy = learner.score(X_test, y_test)
        print('Accuracy after query {n}: {acc:0.4f}'.format(
            n=index + 1, acc=model_accuracy))
        perf_hist.append(model_accuracy)
    return perf_hist


def get_batch_result(model, batch, loss_calculator, device, is_valid=False, args=None):
  # if meas_note:
  batch_x, batch_y, note_locations, targets = utils.batch_to_device( # len(batch) = 9
      batch, device)
  logits = model(
        batch_x, batch_y, note_location = note_locations)
  mask = targets == -100
  if args.pc_grad:
    return logits, targets
  loss = loss_calculator(logits[~mask], targets[~mask])
  if is_valid:
    return logits, loss, targets
  else:
    return logits, loss

class FeatureCollate:
    def __call__(self, batch):
        batch_x = pack_sequence([sample[0]['x'] for sample in batch], enforce_sorted=False)
        batch_y = pad_sequence([sample[0]['y'] for sample in batch], batch_first=True)
        note_locations = {'beat': pad_sequence([sample[0]['note_locations']['beat'] for sample in batch], True).long(),
                            'measure': pad_sequence([sample[0]['note_locations']['measure'] for sample in batch], True).long(),
                            'section': pad_sequence([sample[0]['note_locations']['section'] for sample in batch], True).long(),
                            'voice': pad_sequence([sample[0]['note_locations']['voice'] for sample in batch], True).long()
                            }
        #   if batch[0][7] is not None:
        #     edges = pad_sequence([sample[7] for sample in batch], batch_first=True) # TODO:
        #   else:
        labels = torch.stack([sample[1] for sample in batch])
        return dict(x = batch_x,
                y = batch_y,
                note_locations = note_locations
                ), labels

# https://skorch.readthedocs.io/en/stable/user/FAQ.html#how-do-i-use-a-pytorch-dataset-with-skorch
def prepare_dataloader(args):
    hier_type = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas', 'meas_note']
    curr_type = [x for x in hier_type if getattr(args, x)]
    train_set = ScorePerformDataset(args.data_path,
                                    type="train",
                                    len_slice=args.len_slice,
                                    len_graph_slice=args.len_graph_slice,
                                    graph_keys=args.graph_keys,
                                    hier_type=curr_type,
                                    num_labels=args.num_labels,
                                    no_augment=args.no_augment,
                                    label_file_path=args.label_file_path,
                                    selected_labels=args.selected_labels,
                                    bayesian=args.bayesian)
    valid_set = ScorePerformDataset(args.data_path,
                                    type="valid",
                                    len_slice=args.len_valid_slice,
                                    len_graph_slice=args.len_graph_slice,
                                    graph_keys=args.graph_keys,
                                    hier_type=curr_type,
                                    num_labels=args.num_labels,
                                    label_file_path=args.label_file_path,
                                    selected_labels=args.selected_labels,
                                    bayesian=args.bayesian)

    # def _tensor_to_numpy(data):
    #     new_data = []
    #     for x in data:
    #         if isinstance(x, th.Tensor):
    #             x = x.numpy()
    #         elif isinstance(x, dict):
    #             for k, v in x.items():
    #                 if isinstance(v, th.Tensor):
    #                     x[k] = v.numpy()
    #                 else:
    #                     x[k] = v
    #         else:
    #             x = x
    #         new_data.append(x)
    #     return new_data
    # trainset_np = [_tensor_to_numpy(data) for data in train_set]
    # validset_np = [_tensor_to_numpy(data) for data in valid_set]
    trainset_np = train_set
    validset_np = valid_set
    X_train = [x for x in trainset_np]
    y_train = [torch.Tensor(x[8]) for x in trainset_np]
    X_valid = [x for x in validset_np]
    y_valid = torch.stack([torch.Tensor(x[8]) for x in validset_np])

    # dataset split for active learning
    # balanced sampling
    # initial_idx = np.array([],dtype=np.int)
    # for i in range(10):
    #     idx = np.random.choice(np.where(y_train==i)[0], size=2, replace=False)
    #     initial_idx = np.concatenate((initial_idx, idx))
    #initial_idx = np.array([],dtype=np.int)
    idxs = np.random.choice(
        range(len(X_train)), size=200, replace=False)
    #initial_idx = np.concatenate((initial_idx, idx))
    X_initial = [X_train[idx] for idx in idxs]
    y_initial = torch.stack([y_train[idx] for idx in idxs])

    X_pool = [X_train[idx] for idx in range(len(X_train)) if idx not in idxs]
    y_pool = torch.stack([y_train[idx] for idx in range(len(X_train)) if idx not in idxs])

    # X_pool = np.delete(X_train, idxs, axis=0)
    # y_pool = torch.stack(np.delete(y_train, idxs, axis=0))

    # x, batch_y, edges, note_locations
    # collator = FeatureCollate()
    # X_initial = collator(X_initial)
    X_initial = [dict(x=x[0],
                     y=x[1],
                     edges=x[7],
                     note_locations=x[4]
                     ) for x in X_initial]
    # X_pool = collator(X_pool)
    X_pool = [dict(x=x[0],
                     y=x[1],
                     edges=x[7],
                     note_locations=x[4]
                     ) for x in X_pool]
    # X_pool = collator(X_pool)
    X_valid = [dict(x=x[0],
                     y=x[1],
                     edges=x[7],
                     note_locations=x[4]
                     ) for x in X_valid]


    return X_valid, y_valid, X_initial, y_initial, X_pool, y_pool


def train_active_learner(
    args,
        model,
        device,
        num_epochs,
        criterion,
        exp_name,
):

    logger, out_dir = prepare_directories_and_logger(
        args.checkpoints_dir, args.logs, exp_name, args.make_log)
    shutil.copy(args.yml_path, args.checkpoints_dir/exp_name)

    best_valid_loss = float("inf")
    best_valid_R2 = - float("inf")
    # best_trill_loss = float("inf")
    start_epoch = 0
    iteration = 0
    multi_perf_iter = 0

    if args.resume_training:
        model = load_model(
            model, optimizer, device, args)
    # model.stats = train_loader.dataset.stats
    #TODO: how to add scheduler?
    optimizer = th.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay)

    scheduler = th.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

    if args.selected_labels:
        # get index
        args.selected_labels = sorted(
            [LABEL_LIST.index(x) for x in args.selected_labels])
    else:
        args.selected_labels = None

    X_test, y_test, X_initial, y_initial, X_pool, y_pool = \
        prepare_dataloader(args)

    # https://skorch.readthedocs.io/en/stable/user/neuralnet.html#multiple-input-arguments
    estimator = NeuralNetRegressor(model,
                                   max_epochs=args.num_epochs,
                                   batch_size=args.batch_size,
                                   lr=args.lr,
                                   optimizer=torch.optim.Adam,
                                   optimizer__weight_decay=args.weight_decay,
                                   criterion=criterion,
                                   iterator_train__shuffle=True,
                                   iterator_train__collate_fn=FeatureCollate(),
                                   iterator_valid__collate_fn=FeatureCollate(),
                                   iterator_valid__shuffle=False,
                                   train_split=None,
                                   verbose=0,
                                   device=device)
    
    entropy_perf_hist = active_learning_procedure(max_entropy,
                                                  X_test,
                                                  y_test,
                                                  X_pool,
                                                  y_pool,
                                                  X_initial,
                                                  y_initial,
                                                  estimator,)

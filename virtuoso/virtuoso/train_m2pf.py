from pathlib import Path
import shutil
import random
import math
import time

import numpy as np
import torch as th
import torch
#import wandb
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from itertools import count
from utils import evaluate_std_score

from dataset import (ScorePerformDataset, FeatureCollate)
from logger import Logger
import utils

# selected 19
LABEL_LIST19 = [
    "Timing_Stable_Unstable",
    "Articulation_Long_Short",
    "Articulation_Soft_cushioned_Hard_solid",
    "Pedal_Sparse/dry_Saturated/wet",
    "Pedal_Clean_Blurred",
    "Timbre_Even_Colorful",
    "Timbre_Shallow_Rich",
    "Timbre_Bright_Dark",
    "Timbre_Soft_Loud",
    "Dynamic_Sophisticated/mellow_Raw/crude",
    "Dynamic_Little_dynamic_range_Large_dynamic_range",
    "Music_Making_Fast_paced_Slow_paced",
    "Music_Making_Flat_Spacious",
    "Music_Making_Disproportioned_Balanced",
    "Music_Making_Pure_Dramatic/expressive",
    "Emotion_&_Mood_Optimistic/pleasant_Dark",
    "Emotion_&_Mood_Low_Energy_High_Energy",
    "Emotion_&_Mood_Honest_Imaginative",
    "Interpretation_Unsatisfactory/doubtful_Convincing"
]

def load_model(model, optimizer, device, args):
    # if args.resumeTraining and not args.trainTrill:
    print("=> loading checkpoint '{}'".format(args.checkpoint))
    # model_codes = ['prime', 'trill']
    # filename = 'prime_' + args.modelCode + args.resume
    checkpoint = th.load(args.checkpoint,  map_location='cpu')
    # best_valid_loss = checkpoint['best_valid_loss']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.device = device
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # iteration = checkpoint['training_step']
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.checkpoint, checkpoint['epoch']))
    # start_epoch = checkpoint['epoch'] - 1
    # best_prime_loss = checkpoint['best_valid_loss']
    # print(f'Best valid loss was {best_prime_loss}')
    return model #, optimizer, start_epoch, iteration, best_valid_loss


def prepare_dataloader(args):
    hier_type = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas', 'meas_note']
    curr_type = [x for x in hier_type if getattr(args, x)]
    train_set_stable = None

    train_set = ScorePerformDataset(args.data_path,
                                        type="train",
                                        len_slice=args.len_slice,
                                        len_graph_slice=args.len_graph_slice,
                                        graph_keys=args.graph_keys,
                                        hier_type=curr_type,
                                        num_labels=args.num_labels,
                                        no_augment = args.no_augment,
                                        label_file_path=args.label_file_path,
                                        selected_labels=args.selected_labels,
                                        bayesian = args.bayesian)
    valid_set = ScorePerformDataset(args.data_path,
                                    type="valid",
                                    len_slice=args.len_valid_slice,
                                    len_graph_slice=args.len_graph_slice,
                                    graph_keys=args.graph_keys,
                                    hier_type=curr_type,
                                    num_labels=args.num_labels,
                                    label_file_path=args.label_file_path,
                                    selected_labels=args.selected_labels,
                                    bayesian = args.bayesian)
    test_set = ScorePerformDataset(args.data_path,
                                    type="test",
                                    len_slice=args.len_valid_slice,
                                    len_graph_slice=args.len_graph_slice,
                                    graph_keys=args.graph_keys,
                                    hier_type=curr_type,
                                    num_labels=args.num_labels,
                                    label_file_path=args.label_file_path,
                                    selected_labels=args.selected_labels,
                                    bayesian = args.bayesian)                                
    train_loader = DataLoader(train_set, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    test_loader = DataLoader(test_set, args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    print("train set: ", len(train_set), "valid set: ", len(valid_set), "test set: ", len(test_set))
    
    return train_loader, valid_loader, test_loader 


def prepare_directories_and_logger(output_dir, log_dir, exp_name, make_log=True):
    out_dir = output_dir / exp_name
    print(out_dir)
    (out_dir / log_dir).mkdir(exist_ok=True, parents=True)
    if make_log:
        logger = Logger(out_dir / log_dir)
        return logger, out_dir
    else:
        return None, out_dir

def get_batch_result(model, batch, loss_calculator, device, is_valid=False, args=None):
  '''
  '''
  sigmoid = torch.nn.Sigmoid()
  # if meas_note:
  batch_x, batch_y, beat_y, meas_y, note_locations, align_matched, pedal_status, edges, targets, label_std = utils.batch_to_device( # len(batch) = 9
      batch, device)

  if args.multi_level:
    logitss = model(
            batch_x, batch_y, edges, note_locations)  
    total_loss = 0

    loss = loss_calculator(sigmoid(logitss[-1]), targets)
    if args.cont_loss:
        cont_logits = torch.nn.Softmax(dim=0)(logitss[-1])
        loss += args.cont_loss * torch.nn.MSELoss()(cont_logits, targets)
    total_loss += loss
    loss = total_loss

    # output logits
    logits = sigmoid(logitss[-1])
  else:
    logits = model(
            batch_x, batch_y, edges, note_locations)
    loss = 0
    mask = targets == -100
    loss = loss + loss_calculator(sigmoid(logits[~mask]), targets[~mask])

    logits = sigmoid(logits)
  if is_valid:
    return logits, loss, targets, label_std
  else:
    return logits, loss

def inf_norm(x):
    return torch.norm(x, p=float("inf"), dim=-1, keepdim=True)

def kl_loss(input, target, reduction="batchmean"):
    return F.kl_div(
        F.log_softmax(input, dim=-1),
        F.softmax(target, dim=-1),
        reduction=reduction,
    )

def train_step(model, batch, optimizer, scheduler, loss_calculator, logger, device, args, iteration):
    start = time.perf_counter()
    logits, total_loss = get_batch_result(
        model, batch, loss_calculator, device, args = args)
    optimizer.zero_grad()
    total_loss.backward()
    grad_norm = th.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    scheduler.step()

    loss_dict = dict()
    duration = time.perf_counter() - start
    if args.make_log:
        logger.log_training(total_loss.item(), loss_dict, grad_norm,
                            optimizer.param_groups[0]['lr'], duration, iteration)
        logger.log_training(total_loss.item(), loss_dict, grad_norm,
                            optimizer.param_groups[0]['lr'], duration, iteration)

def get_validation_loss(model, valid_loader, loss_calculator, device, num_labels=25, args=None, selected_labels = None, return_each = False):
    valid_loss = []
    y_true = []
    y_pred = []
    label_std_list = []

    with th.no_grad():
        for _, valid_batch in enumerate(valid_loader):
            logits, total_loss, targets, label_std = get_batch_result(
                model, valid_batch, loss_calculator, device, is_valid=True, args = args)
            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(logits.detach().cpu().numpy())
            label_std_list.append(label_std)
            if args.pc_grad:
                valid_loss.append(total_loss.mean().item())
            else:
               valid_loss.append(total_loss.item())
                
    
    valid_loss = sum(valid_loss) / len(valid_loader.dataset)
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    label_std = np.vstack(label_std_list)
    valid_metric_dict = dict()
    std_score_1 = evaluate_std_score(y_pred, y_true, label_std, acceptable_range=1)
    std_score_05 = evaluate_std_score(y_pred, y_true, label_std, acceptable_range=0.5)
    std_score_01 = evaluate_std_score(y_pred, y_true, label_std, acceptable_range=0.1)
    valid_metric_dict["std_score_1"] = np.array(std_score_1).mean()
    valid_metric_dict["std_score_05"] = np.array(std_score_05).mean()
    valid_metric_dict["std_score_01"] = np.array(std_score_01).mean()
    
    for i in range(num_labels):
        valid_metric_dict[f"std_score_1-{i}"] = std_score_1[i]
        valid_metric_dict[f"std_score_05-{i}"] = std_score_05[i]
        valid_metric_dict[f"std_score_01-{i}"] = std_score_01[i]


    r2 = r2_score(y_true, y_pred)
    r2_flatten = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    mse = mean_squared_error(y_true, y_pred)
    #result = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
    valid_metric_dict ["R2"] = r2
    valid_metric_dict ["R2flat"] = r2_flatten
    valid_metric_dict ["MSE"] = mse  

    for i, label_name in enumerate(LABEL_LIST19[:num_labels]):
        valid_metric_dict [f"R2-{i}"] = r2_score(y_true[:,i], y_pred[:,i])
        valid_metric_dict [f"MSE-{i}"] = mean_squared_error(y_true[:,i], y_pred[:,i])
        
    print('Valid loss: {}'.format(valid_loss))
    if return_each:
        return valid_loss, valid_metric_dict, y_true, y_pred
    else:
        return valid_loss, valid_metric_dict


def train(args,
          model,
          device,
          num_epochs,
          criterion,
          exp_name,
          ):


  logger, out_dir = prepare_directories_and_logger(
      args.checkpoints_dir, args.logs, exp_name, args.make_log)
  shutil.copy(args.yml_path, args.checkpoints_dir/exp_name)

  args.selected_labels = None
  
  train_loader, valid_loader, test_loader = prepare_dataloader(
      args)

  optimizer = th.optim.Adam(
        model.parameters(), weight_decay=args.weight_decay, lr=args.lr)
  loss_calculator = criterion

  model_parameters = filter(lambda p: p.requires_grad, model.parameters())
  params = sum([np.prod(p.size()) for p in model_parameters])
  print('Number of Network Parameters is ', params)

  best_valid_loss = float("inf")
  best_valid_R2 = - float("inf")
  start_epoch = 0
  iteration = 0
  multi_perf_iter = 0

  if args.resume_training:
      model = load_model(
          model, optimizer, device, args)
  scheduler = th.optim.lr_scheduler.StepLR(
      optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate)

  # load data
  args.iters_per_checkpoint = len(train_loader)
  print('Loading the training data...')
  model.train()
  args.iters_per_checkpoint = len(train_loader)      
  for epoch in tqdm(range(start_epoch, num_epochs), desc="overall"):
    if args.curriculum and epoch < int(num_epochs*0.2):
      continue
    print('current training step is ', iteration)
    train_loader.dataset.update_slice_info()
    for _, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
      train_step(model, batch, optimizer, scheduler,
                 loss_calculator, logger, device, args, iteration)
      iteration += 1

      if iteration % args.iters_per_checkpoint == 0:
        model.eval()
        if args.bayesian or args.bayesian_v2:
            model.score_encoder.voice_net.train()
            model.score_encoder.lstm.train()
            model.score_encoder.beat_rnn.train()
            model.out_fc[0].train()
            model.out_fc[2].train()
        
        selected_labels_eval = LABEL_LIST19
        valid_loss, valid_metric_dict = get_validation_loss(
            model, valid_loader, loss_calculator, device, args.num_labels, args,
            selected_labels = selected_labels_eval)

        if args.make_log:
            logger.log_validation(valid_loss, valid_metric_dict, model, iteration)
            #valid_metric_dict = pack_validation_log(valid_metric_dict, valid_loss)
            keys = list(valid_metric_dict.keys())
            for key in keys:
                new_key = f"validation/{key}"
                valid_metric_dict[new_key] = valid_metric_dict.pop(key)
            valid_metric_dict['validation/total_loss'] = valid_loss

        test_loss, test_metric_dict = get_validation_loss(
            model, test_loader, loss_calculator, device, args.num_labels, args,
            selected_labels = selected_labels_eval)
        if args.make_log:
            logger.log_validation(test_loss, test_metric_dict, model, iteration, is_test=True)
            keys = list(test_metric_dict.keys())
            for key in keys:
                new_key = f"test/{key}"
                test_metric_dict[new_key] = test_metric_dict.pop(key)
            test_metric_dict['test/total_loss'] = test_loss


        is_best = valid_loss < best_valid_loss
        if is_best:
            best_valid_loss = min(best_valid_loss, valid_loss)
            utils.save_checkpoint(args.checkpoints_dir / exp_name,
                                {'epoch': epoch + 1,
                                'state_dict': model.state_dict(),
                                'best_valid_loss': best_valid_loss,
                                'optimizer': optimizer.state_dict(),
                                'training_step': iteration,
                                'stats': model.stats,
                                'network_params': model.network_params,
                                'model_code': args.model_code,
                                "args": args,
                                }, is_best)
        model.train()

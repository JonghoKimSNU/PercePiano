from pathlib import Path
import shutil
import random
import math
import time

import numpy as np
import torch as th
from sklearn.metrics import r2_score, mean_squared_error
from tqdm import tqdm
from torch.utils.data import DataLoader
# from .dataset import ScorePerformDataset, FeatureCollate, MultiplePerformSet, multi_collate, EmotionDataset
# from .logger import Logger, pack_emotion_log, pack_train_log, pack_validation_log
# from .loss import LossCalculator, cal_multiple_perf_style_loss
# from . import utils
# from .inference import generate_midi_from_xml, get_input_from_xml, save_model_output_as_midi
# from .model_constants import valid_piece_list
# from .emotion import validate_style_with_emotion_data
# from . import style_analysis as sty

from dataset import ScorePerformDataset, FeatureCollate, MultiplePerformSet, multi_collate, EmotionDataset
from logger import Logger, pack_emotion_log, pack_train_log, pack_validation_log
#from loss import LossCalculator, cal_multiple_perf_style_loss
import utils
from train_m2pf import LABEL_LIST, LABEL_LIST19, get_validation_loss
#from inference import generate_midi_from_xml, get_input_from_xml, save_model_output_as_midi
#from model_constants import valid_piece_list
#from emotion import validate_style_with_emotion_data
#import style_analysis as sty

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


def prepare_dataloader(args, return_each=False):
    hier_type = ['is_hier', 'in_hier', 'hier_beat', 'hier_meas', 'meas_note']
    curr_type = [x for x in hier_type if getattr(args, x)]
    # train_set = ScorePerformDataset(args.data_path,
    #                                 type="train",
    #                                 len_slice=args.len_slice,
    #                                 len_graph_slice=args.len_graph_slice,
    #                                 graph_keys=args.graph_keys,
    #                                 hier_type=curr_type,
    #                                 num_labels=args.num_labels)
    valid_set = ScorePerformDataset(args.data_path,
                                    #type="valid",
                                    type="test", # TODO: fix this when you need
                                    len_slice=args.len_valid_slice,
                                    len_graph_slice=args.len_graph_slice,
                                    graph_keys=args.graph_keys,
                                    hier_type=curr_type,
                                    num_labels=args.num_labels,
                                    label_file_path=args.label_file_path,
                                    selected_labels=args.selected_labels)
    valid_loader = DataLoader(valid_set, args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=args.pin_memory, collate_fn=FeatureCollate())
    if return_each:
        return valid_loader, [d['perform_path'] for d in valid_set.data]
    else:
        return valid_loader, None #, emotion_loader, multi_perf_loader


def prepare_directories_and_logger(output_dir, log_dir, exp_name, make_log=True):
    out_dir = output_dir / exp_name
    print(out_dir)
    (out_dir / log_dir).mkdir(exist_ok=True, parents=True)
    if make_log:
        logger = Logger(out_dir / log_dir)
        return logger, out_dir
    else:
        return None, out_dir

# def get_validation_loss(model, valid_loader, loss_calculator, device, num_labels=25, selected_labels = LABEL_LIST):
#     valid_loss = []
#     y_true = []
#     y_pred = []

#     with th.no_grad():
#         for _, valid_batch in tqdm(enumerate(valid_loader)):
#             t0=time.time()
#             logits, total_loss, targets = get_batch_result(
#                 model, valid_batch, loss_calculator, device, is_valid=True)
#             th.cuda.current_stream().synchronize()
#             #print("1 batch time", time.time() - t0)
#             y_true.append(targets.detach().cpu().numpy())
#             y_pred.append(logits.detach().cpu().numpy())
#             valid_loss.append(total_loss.item())
    
#     valid_loss = sum(valid_loss) / len(valid_loader.dataset)
#     y_true = np.vstack(y_true)
#     y_pred = np.vstack(y_pred)
#     valid_metric_dict = dict()
#     print(y_true.shape, y_pred.shape)
#     r2 = r2_score(y_true, y_pred)
#     r2_flatten = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
#     mse = mean_squared_error(y_true, y_pred)
#     #result = r2_score(y_true.reshape(-1), y_pred.reshape(-1))
#     valid_metric_dict ["R2"] = r2
#     valid_metric_dict ["R2flat"] = r2_flatten
#     valid_metric_dict ["MSE"] = mse  

#     for i, label_name in enumerate(selected_labels):
#         valid_metric_dict ["R2" + label_name] = r2_score(y_true[:,i], y_pred[:,i])
    
#     print('Valid_loss: {}'.format(valid_loss))
#     return valid_loss, valid_metric_dict


def evaluate(args,
          model,
          device,
          num_epochs,
          criterion,
          return_each = False):
    if args.selected_labels:
        # get index
        args.selected_labels = sorted([LABEL_LIST.index(x) for x in args.selected_labels])
    else:
        args.selected_labels = None
    valid_loader, perform_paths = prepare_dataloader(
        args, return_each = return_each)
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_calculator = criterion
    # print("data load time", time.time() - t0)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Number of Network Parameters is ', params)

    if args.resume_training:
        model = load_model(
            model, optimizer, device, args)

    model.eval()

    if args.selected_labels:
        selected_labels_eval = [LABEL_LIST[selected] for selected in args.selected_labels]
    elif args.num_labels == 19:
        selected_labels_eval = LABEL_LIST19
    else:
        selected_labels_eval = LABEL_LIST

    if return_each:
        valid_loss, valid_metric_dict, y_true, y_pred = get_validation_loss(
                model, valid_loader, loss_calculator, device, args.num_labels, args,
                selected_labels = selected_labels_eval, return_each = return_each)
    else:
            valid_loss, valid_metric_dict = get_validation_loss(
                model, valid_loader, loss_calculator, device, args.num_labels, args,
                selected_labels = selected_labels_eval, return_each = return_each)
    #if logger:
    #    logger.log_validation(valid_loss, valid_metric_dict, model, iteration)
    #valid_metric_dict = pack_validation_log(valid_metric_dict, valid_loss)
    keys = list(valid_metric_dict.keys())
    for key in keys:
        new_key = f"validation/{key}"
        valid_metric_dict[new_key] = valid_metric_dict.pop(key)
    valid_metric_dict['validation/total_loss'] = valid_loss
    if return_each:
        return valid_metric_dict, perform_paths, y_true, y_pred
    else:
        return valid_metric_dict
    #end of epoch

def eval_case_study(args,
          model,
          device,
          num_epochs,
          criterion,
          ):
    if args.selected_labels:
        # get index
        args.selected_labels = sorted([LABEL_LIST.index(x) for x in args.selected_labels])
    else:
        args.selected_labels = None
    valid_loader, perform_paths = prepare_dataloader(
        args, return_each = True)
    optimizer = th.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_calculator = criterion
    # print("data load time", time.time() - t0)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    # print('Number of Network Parameters is ', params)

    if args.resume_training:
        model = load_model(
            model, optimizer, device, args)

    model.eval()

    if args.selected_labels:
        selected_labels_eval = [LABEL_LIST[selected] for selected in args.selected_labels]
    elif args.num_labels == 19:
        selected_labels_eval = LABEL_LIST19
    else:
        selected_labels_eval = LABEL_LIST

    valid_loss, valid_metric_dict, y_true, y_pred = get_validation_loss(
            model, valid_loader, loss_calculator, device, args.num_labels, args,
            selected_labels = selected_labels_eval, return_each = True)
    #if logger:
    #    logger.log_validation(valid_loss, valid_metric_dict, model, iteration)
    #valid_metric_dict = pack_validation_log(valid_metric_dict, valid_loss)
    keys = list(valid_metric_dict.keys())
    for key in keys:
        new_key = f"validation/{key}"
        valid_metric_dict[new_key] = valid_metric_dict.pop(key)
    valid_metric_dict['validation/total_loss'] = valid_loss
    # for k,v in valid_metric_dict.items():
    #     print(k, v)

    each_metric_dict = dict()
    # print(y_true.shape)
    for i, (y_t, y_p) in enumerate(zip(y_true, y_pred)):
        # print(y_t.shape, y_p.shape)
        each_metric_dict[perform_paths[i]] = [y_t, y_p, mean_squared_error(y_t, y_p)] #true, pred, error


    return valid_metric_dict, each_metric_dict
    #end of epoch


# elif args.sessMode in ['test', 'testAll', 'testAllzero', 'encode', 'encodeAll', 'evaluate', 'correlation']:
# ### test session

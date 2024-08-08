import sys
from datetime import datetime
from pathlib import Path, PosixPath
import time
from torch import distributed, nn
from torch.nn.parallel.distributed import DistributedDataParallel
import torch
# from .parser import get_parser, get_name
# from . import utils
# from .train import train
# from .model_m2pf import VirtuosoNet
# from .inference import inference, inference_with_emotion
from parser import get_parser, get_name
import utils
from eval_m2pf import evaluate
from model_m2pf import VirtuosoNet, VirtuosoNetMultiLevel
import numpy as np
import random
from collections import defaultdict
import os
import json
#from inference import inference, inference_with_emotion

def find_file_with_string_in_name(directory, substring):
    for root, dirs, files in os.walk(directory):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if substring in file_path and "checkpoint_best.pt" in file_path:
                return_file_path = file_path
    return return_file_path


def main():
    scores = defaultdict(list)
    parser = get_parser()
    args = parser.parse_args()
    random.seed(args.th_seed)
    print(args)
    np.random.seed(args.th_seed)
    torch.manual_seed(args.th_seed)

    if args.return_each:
        perform_path_all = []
        y_pred_all = []


    for fold in range(args.n_folds):
        parser = get_parser()
        args = parser.parse_args()
        if args.n_folds > 1:
            args.checkpoint = str(args.checkpoint)
            print("checkpoint input", args.checkpoint)
            checkpoint_prefix, checkpoint_suffix = args.checkpoint.split("foldx")
            checkpoint_fold = f"fold{fold}"
            checkpoint_prefix = os.path.join(checkpoint_prefix, checkpoint_fold)
            args.checkpoint = find_file_with_string_in_name(checkpoint_prefix, checkpoint_suffix)
            args.data_path = str(args.data_path)
            args.data_path = args.data_path.replace("foldx", str(fold))
            print("checkpoint output", args.checkpoint)
            print(args.data_path)
            args.checkpoint = PosixPath(args.checkpoint)
            args.data_path = PosixPath(args.data_path)
        else:
            args.checkpoint = PosixPath(str(os.path.join(args.checkpoint, "checkpoint_best.pt")))
            args.data_path = PosixPath(args.data_path)
        print(args.checkpoint, args.data_path)

        args, net_param, data_stats = utils.handle_args(args)
        print(net_param)

        device = utils.get_device(args)

        criterion = nn.MSELoss()

        if args.world_size > 1:
            if device != "cuda" and args.rank == 0:
                print("Error: distributed training is only available with cuda device", file=sys.stderr)
                sys.exit(1)
            torch.cuda.set_device(args.rank % torch.cuda.device_count())
            distributed.init_process_group(backend="nccl",
                                        init_method="tcp://" + args.master,
                                        rank=args.rank,
                                        world_size=args.world_size)
        net_param.num_label = args.num_labels
        t0 = time.time()
        print("checkpoint:", args.checkpoint)
        if args.multi_level:
            model = VirtuosoNetMultiLevel(net_param, data_stats, multi_level=args.multi_level)
        
        model = model.to(device)
        if args.return_each:
            metric_dict, perform_paths, _, y_pred = evaluate(args,
                    model,
                    device,
                    args.num_epochs, 
                    criterion,
                    return_each=True
                    )
        else:
            metric_dict = evaluate(args,
                    model,
                    device,
                    args.num_epochs, 
                    criterion
                    )
        for k, v in metric_dict.items():
            scores[k].append(v)
        print("std_score_1, std_score_05, std_score_01, r2, mse")
        print(metric_dict["validation/std_score_1"], metric_dict["validation/std_score_05"], metric_dict["validation/std_score_01"],
                    metric_dict["validation/R2"], metric_dict["validation/MSE"])
        for i in range(args.num_labels):
            print(metric_dict[f"validation/std_score_1-{i}"], metric_dict[f"validation/std_score_05-{i}"],
                 metric_dict[f"validation/std_score_01-{i}"],
                 metric_dict[f"validation/R2-{i}"], metric_dict[f"validation/MSE-{i}"])
        print()
        if args.return_each:
            perform_path_all.extend(perform_paths)
            y_pred_all.extend(y_pred.tolist())
            break
    print('final scores')
    # print(scores)
    # print tsv format. group by metric. std_score_1, std_score_05, std_score_01, r2, mse
    print("std_score_1, std_score_05, std_score_01, r2, mse")
    print(np.mean(scores["validation/std_score_1"]), np.mean(scores["validation/std_score_05"]), np.mean(scores["validation/std_score_01"]), np.mean(scores["validation/R2"]), np.mean(scores["validation/MSE"]))
    for i in range(args.num_labels):
        print(np.mean(scores[f"validation/std_score_1-{i}"]), np.mean(scores[f"validation/std_score_05-{i}"]), np.mean(scores[f"validation/std_score_01-{i}"]), np.mean(scores[f"validation/R2-{i}"]), np.mean(scores[f"validation/MSE-{i}"]))

    if args.return_each:
        args.checkpoint = str(args.checkpoint)
        all_results_save = defaultdict(list)
        for i in range(len(perform_path_all)):
            all_results_save[perform_path_all[i]].append(y_pred_all[i])
         # TODO: save all_results_save

if __name__ == '__main__':
    main()
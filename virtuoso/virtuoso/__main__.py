import sys
from datetime import datetime
from pathlib import Path

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
from train_m2pf import train
from train_m2pf_cls import train_cls
# from virtuoso.legacy_al_train import train_active_learner
from al_train_m2pf import al_train
from train_m2pf_compare import train_comparison
from model_m2pf import (VirtuosoNet, VirtuosoNetMultiLevel, VirtuosonetAL, VirtuosoNetSingle,
                        VirtuosoNet_classification, DeepComparator, ShallowComparator)
import random
import numpy as np
#from inference import inference, inference_with_emotion


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    random.seed(args.th_seed)
    np.random.seed(args.th_seed)
    torch.manual_seed(args.th_seed)
    # random.seed(0)

    if args.selected_labels:
        args.selected_labels = args.selected_labels.split("-")
        #args.selected_labels = [label.strip("\"").strip("'") for label in args.selected_labels]
        args.num_labels = len(args.selected_labels)
        print(args.selected_labels)
    if args.session_mode == "active_learning":
        assert args.query_strategy is not None
    args, net_param, data_stats = utils.handle_args(args)
    print(net_param)
    name = get_name(parser, args) + "_" + datetime.now().strftime('%y%m%d%H%M')
        # datetime.now().strftime('%y%m%d')
        # datetime.now().strftime('%y%m%d-%H%M%S')
    print(f"Experiment {name}")

    device = utils.get_device(args)
    # criterion = utils.make_criterion_func(args.loss_type)
    # TODO: my criterion
    if args.pc_grad:
        criterion = nn.MSELoss(reduction="none")
    elif args.classification:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    if args.noise:
        assert args.noise_version in [0, 1, 2]

    if args.bayesian:
        criterion = nn.KLDivLoss(reduction="batchmean")
        args.label_file_path = "/root/v2/muzic/virtuosonet/data/midi_label_map_distribution.json"
        args.batch_size = 1

    if args.world_size > 1:
        if device != "cuda" and args.rank == 0:
            print(
                "Error: distributed training is only available with cuda device", file=sys.stderr)
            sys.exit(1)
        torch.cuda.set_device(args.rank % torch.cuda.device_count())
        distributed.init_process_group(backend="nccl",
                                       init_method="tcp://" + args.master,
                                       rank=args.rank,
                                       world_size=args.world_size)
    net_param.num_label = args.num_labels

    if args.multi_level:
        model = VirtuosoNetMultiLevel(net_param, data_stats, multi_level=args.multi_level)
    elif args.classification:
        model = VirtuosoNet_classification(net_param, data_stats, num_labels = args.classification)
    elif args.comparison:
        model = DeepComparator(net_param, data_stats)
    elif args.ablation:
        model = VirtuosoNetSingle(net_param, data_stats)
    else:
        model = VirtuosoNet(net_param, data_stats)
    model = model.to(device)

    # if not (args.session_mode =="train" and args.resume_training):
    #     checkpoint = torch.load(args.checkpoint)
    # checkpoint = args.checkpoints / f"{name}.pt"
    # checkpoint_tmp = args.checkpoints / f"{name}.pt.tmp"
    # if args.resume_training and checkpoint.exists():
    #     checkpoint.unlink()

    if args.session_mode == "train":
        if args.classification:
            train_cls(args,
                  model,
                  device,
                  args.num_epochs,
                  criterion,
                  name,
                  )
        elif args.comparison:
            train_comparison(args,
                    model,
                    device,
                    args.num_epochs,
                    criterion,
                    name,
                    )
        else:
            train(args,
                model,
                device,
                args.num_epochs,
                criterion,
                name,
                )
    elif args.session_mode == "active_learning":
        al_train(args,
                 model,
                 device,
                 args.num_epochs,
                 criterion,
                 name)
    # elif args.session_mode == "inference":
    #     # stats= utils.load_dat(args.data_path / 'stat.dat')
    #     inference(args, model, device)

    # elif args.session_mode == "inference_with_emotion":
    #     inference_with_emotion(args, model, device)


if __name__ == '__main__':
    main()

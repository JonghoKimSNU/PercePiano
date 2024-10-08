import argparse
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser("virtuosonet")
    parser.add_argument("-mode", "--session_mode", type=str,
                        default="train", help="train or inference")
    parser.add_argument("--query_strategy", type=str, default="")
    parser.add_argument("-yml", "--yml_path", type=str,
                        #default="isgn_param.yml",
                         help="yml file path")
    parser.add_argument("-data", "--data_path", type=Path,
                        default=Path("datasets/main_pkl"), help="data dir name")
    parser.add_argument("--emotion_data_path", type=Path,
                    default=Path("datasets/emotion_pkl"), help="data dir name")
    parser.add_argument("--resume", type=str,
                        default="_best.pth.tar", help="best model path")
    parser.add_argument("--xml_path", type=Path,
                        default=Path('/home/svcapp/userdata/dev/virtuosoNet/test_pieces/bps_5_1/musicxml_cleaned.musicxml'))
    parser.add_argument("--output_path", type=Path,
                        default=Path('test_result/'))
    parser.add_argument("--valid_xml_dir", type=Path,
                        default=Path('/home/teo/userdata/datasets/chopin_cleaned/'))
    # model model options
    parser.add_argument("-trill", "--is_trill", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="train trill")
    # parser.add_argument("-slur", "--slurEdge", default=False,
    #                     type=lambda x: (str(x).lower() == 'true'), help="slur edge in graph")
    # parser.add_argument("-voice", "--voiceEdge", default=True,
    #                     type=lambda x: (str(x).lower() == 'true'), help="network in voice level")
    # TODO: no redundancy?
    parser.add_argument("--is_hier", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--in_hier", default=False, type=lambda x: (str(x).lower() == 'true'))
    parser.add_argument("--hier_beat", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--hier_model", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--hier_meas", default=False, type=lambda x: (str(x).lower() == 'true'))   
    parser.add_argument("--meas_note", default=False, type=lambda x: (str(x).lower() == 'true'))   
    # model
    # parser.add_argument("--model_name", type=str,
    #                     default="model name", help="vir decode classification or 1 level classification")    
    # training parameters
    parser.add_argument("--th_seed", default=7,
                        type=int, help="torch random seed")
    parser.add_argument("--pc_grad", default=0,
                        type=int, help="if use gradient surgery, set the number of tasks")
    parser.add_argument("--selected_labels", required=False, default="",
                        #nargs='+', 
                        type=str,
                        help="use selected labels")
    parser.add_argument("--bayesian", action="store_true", help = "use bayesian inference(kldloss)")
    parser.add_argument("--bayesian_v2", action="store_true", help = "use bayesian inference(mseloss)")
    # for virtual adversarial training
    parser.add_argument("--noise", type=int, default=0, help = "1: virtual loss + actual loss 2: virtual loss only 3: actual loss only")
    parser.add_argument("--noise_version", type = int, default = 0, help = "version of noise. v0: to score, v1: to perform, v2: to both")
    # classification or regression
    parser.add_argument("--classification", type = int, default = 0, help = "0: regression, x > 1: classification with x classes")
    parser.add_argument("--comparison", type = int, default = 0, help = "1: comparison")
    parser.add_argument("--ablation", action="store_true", help = "ablation study")
    # # contrastive loss
    parser.add_argument("--cont_loss", type = float, default = 0, help = "alpha value for contrastive loss")
    parser.add_argument("--multi_level", type = str, default = "", help = "note, beat, measure, total_note_cat seperated by comma")
    parser.add_argument("--ml_loss", type = float, default = 0, help = "weight for multi level loss")
    parser.add_argument("--hn", action="store_true", help = "use hard negative")

    # dist parallel options
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--master")

    # save options
    parser.add_argument("--checkpoints_dir", 
                        type=Path,
                        default=Path('/root/v2/muzic/virtuosonet/checkpoints_m2pf/'),
                        help='folder to store checkpoints')
    parser.add_argument("--checkpoint", 
                    type=Path,
                    default=Path('/home/svcapp/userdata/dev/virtuosoNet/isgn_best.pt'),
                    help='path to load checkpoint')    
    parser.add_argument("--evals",
                        type=Path,
                        default=Path('evals')
                        )
    parser.add_argument("--save",
                        action="store_true",)
    parser.add_argument("--logs",
                        type=Path,
                        default=Path("logs")
                        )
    
    # training option
    parser.add_argument("--no_augment",
                        action="store_true",)
    parser.add_argument("--num_epochs",
                    type=int,
                    default=100
                    )
    parser.add_argument("--num_labels",
                    type=int,
                    default=19
                    )
    parser.add_argument("--label_file_path",
                    type=str,
                    default="data/midi_label_map_mean_reg_cls_19_with0.json"
                    )
    parser.add_argument("--batch_size",
                    type=int,
                    default=32
                    )
    parser.add_argument("--iters_per_checkpoint",
                    type=int,
                    default=700
                    )
    parser.add_argument("--iters_per_multi_perf",
                    type=int,
                    default=50
                    )
    parser.add_argument("--lr",
                        type=float,
                        default=1e-4
                        )
    parser.add_argument("--len_slice",
                        type=int,
                        default=5000 # TODO: i fixed this
                        )
    parser.add_argument("--len_graph_slice",
                        type=int,
                        default=400
                        )
    parser.add_argument("--len_valid_slice",
                        type=int,
                        default=5000 # TODO: i fixed this
                        )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5
                        )
    parser.add_argument("--lr_decay_step",
                        type=float,
                        default=3000
                        )
    parser.add_argument("--lr_decay_rate",
                        type=float,
                        default=0.98
                        )
    parser.add_argument("--delta_weight",
                        type=float,
                        default=1
                        )
    parser.add_argument("--meas_loss_weight",
                        type=float,
                        default=1
                        )
    parser.add_argument("--multi_perf_dist_loss_margin",
                        type=float,
                        default=1
                        )
    parser.add_argument("--grad_clip",
                        type=float,
                        default=2
                        ) 
    parser.add_argument("--kld_max",
                        type=float,
                        default=0.02
                        ) 
    parser.add_argument("--kld_sig",
                        type=float,
                        default=15e3
                        ) 
    parser.add_argument("-loss", "--loss_type", type=str,
                        default='MSE', help='type of training loss')
    parser.add_argument("-delta", "--delta_loss", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="apply delta value as loss during training")
    parser.add_argument("--vel_balance_loss", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="apply velocity balance as loss during training")

    # environment options
    parser.add_argument("-dev", "--device", type=str,
                        default='cpu', help="cuda device number")
    parser.add_argument("--num_workers", type=int,
                        default=0, help="num workers for dataloader")
    parser.add_argument("--pin_memory", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="pin memory for loader")
    parser.add_argument("--make_log", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="make log for training")
    parser.add_argument("--multi_perf_compensation", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="train style vector to be zero with multiple performances")
    parser.add_argument("-code", "--model_code", type=str,
                        default='isgn', help="code name for saving the model")
    parser.add_argument("-tCode", "--trillCode", type=str,
                        default='trill_default', help="code name for loading trill model")
    parser.add_argument("-comp", "--composer", type=str,
                        default='Beethoven', help="composer name of the input piece")
    parser.add_argument("--qpm_primo", type=int, help="Tempo at the beginning of the input piece in quarter notes per minute")
    parser.add_argument("--latent", type=float, default=0, help='initial_z value')
    parser.add_argument("-bp", "--boolPedal", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='make pedal value zero under threshold')
    parser.add_argument("-reTrain", "--resume_training", default=False, type=lambda x: (
        str(x).lower() == 'true'), help='resume training after loading model')
    # parser.add_argument("-perf", "--perfName", default='Anger_sub1',
    #                     type=str)
    parser.add_argument("-hCode", "--hierCode", type=str,
                        default='han_measure', help="code name for loading hierarchy model")
    # parser.add_argument("-intermd", "--intermediate_loss", default=True,
    #                     type=lambda x: (str(x).lower() == 'true'), help="intermediate loss in ISGN")
    parser.add_argument("--tempo_loss_in_note", default=False,
                        action='store_true', help="calculate tempo loss in note-level instead of beat-level")

    # inference options
    parser.add_argument("-dskl", "--disklavier", default=True,
                        type=lambda x: (str(x).lower() == 'true'), help="save midi for disklavier")
    parser.add_argument("--multi_instruments", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="save multi instruments as separate track")
    parser.add_argument("--tempo_clock", default=False,
                        type=lambda x: (str(x).lower() == 'true'), help="add tempo clock track in output MIDI")
    parser.add_argument("--velocity_multiplier", type=float,
                    default=1, help="multiplier that broaden diff between mean velocity and each note's velocity")
    parser.add_argument("-clock", "--clock_interval_in_16th", default=4,
                        type=int, help="midi clock interval")
    parser.add_argument("-csv", "--save_csv", default=False,
                        action='store_true', help="save midi note in csv")
    parser.add_argument("-cluster", "--save_cluster", default=False,
                        action='store_true', help="save cluster of note embedding")
    parser.add_argument( "--mod_midi_path", default=None, type=str, help="path of modified midi path")
    # random seed

    # for active learning
    parser.add_argument("--n_queries", type=int, default=0, help='number of queries for active learning')
    parser.add_argument("--reset_each_query", action = 'store_true', help='reset model each query')

    # kfold
    parser.add_argument("--n_folds", type=int, default=0, help='number of folds for kfold')
    parser.add_argument("--curriculum", type=int, default=0, help='curriculum learning (ratio of data to use for initial 20% of epochs)')
    
    # case study
    parser.add_argument("--return_each", action = 'store_true', help='return each fold result')

    return parser


def get_name(parser, args):
    """
    Return the name of an experiment given the args. Some parameters are ignored,
    for instance --workers, as they do not impact the final result.
    """
    ignore_args = set([
        "checkpoints_dir",
        "deterministic",
        "eval",
        "evals",
        "eval_cpu",
        "eval_workers",
        "logs",
        "master",
        "rank",
        "restart",
        "save",
        "save_model",
        "show",
        "valid",
        "workers",
        "world_size",
        "device",
        "num_workers",
        "pin_memory",
        "make_log",
        "graph_keys",
        "data_path",
        "emotion_data_path",
        "net_param",
        "iters_per_checkpoint",
        "selected_labels",
        "meas_note"
    ])
    parts = []
    name_args = dict(args.__dict__)
    for name, value in name_args.items():
        if name in ignore_args:
            continue
        if value != parser.get_default(name):
            if isinstance(value, Path):
                parts.append(f"{name}_{value.name}")
            else:
                parts.append(f"{name}_{value}")
    if parts:
        name = "_".join(parts)
    else:
        name = "default"
    if args.selected_labels:
        name += f"/{'-'.join([l[:5] for l in args.selected_labels])}/"
    return name

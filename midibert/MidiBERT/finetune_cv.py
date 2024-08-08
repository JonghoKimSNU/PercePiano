import argparse
import numpy as np
import pickle
import os
import random
import json
from torch.utils.data import DataLoader
import torch
from transformers import BertConfig
from torch.nn.utils.rnn import pad_sequence
from MidiBERT.model import MidiBert
from MidiBERT.finetune_trainer import FinetuneTrainer
from MidiBERT.finetune_dataset import FinetuneDataset, FinetuneDatasetAddon, FinetuneDatasetAlign
from MidiBERT.virtuosonet_model import VirtuosoNet, AlignLSTM
import pathlib
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from collections import defaultdict

def evaluate_std_score(predictions, true_values, label_stds, acceptable_range=0.05):
    """
    Evaluate multi-label regression predictions.
    Args:
    - predictions (numpy array): Predicted values for each label.
    - true_values (numpy array): True values for each label.
    - acceptable_range (float): Acceptable range as a percentage (0.05 for 5%).
    Returns:
    - within_range_pct (list of floats): Percentage of predictions within acceptable range for each label.
    """
    num_labels = predictions.shape[1]  # Get the number of labels
    within_range_pct = []

    for label in range(num_labels):
        label_preds = predictions[:, label]
        label_true = true_values[:, label]
        label_std = label_stds[:, label]

        # Calculate percentage of predictions within acceptable range
        lower_bound = label_true - acceptable_range * label_std
        upper_bound = label_true + acceptable_range * label_std
        within_range = np.logical_and(label_preds >= lower_bound, label_preds <= upper_bound)
        within_range_pct.append(np.mean(within_range) * 100.0)

    return within_range_pct


def get_args():
    parser = argparse.ArgumentParser(description='')

    ### mode ###
    parser.add_argument('--task', choices=['melody', 'velocity', 'composer', 'emotion', 'percepiano'], required=True)
    ### path setup ###
    parser.add_argument('--dict_file', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--name', type=str, default='')
    parser.add_argument('--ckpt', default='result/pretrain/default/model_best.ckpt')
    parser.add_argument("--addons_path", type=str, default="", help="addon information to be loaded")
    parser.add_argument("--align_path", type=str, default="", help="align information to be loaded")
    parser.add_argument("--net_params_path", type=str, default="", help="path to virtuosonet parameters")

    ### parameter setting ###
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--class_num', type=int)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_seq_len', type=int, default=512, help='all sequences are padded to `max_seq_len`')
    parser.add_argument('--hs', type=int, default=768)
    parser.add_argument("--index_layer", type=int, default=12, help="number of layers")
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-5, help='initial learning rate')
    parser.add_argument('--nopretrain', action="store_true")  # default: false
    
    # cross validation
    parser.add_argument('--start_cv_idx', type=int, default=0)
    parser.add_argument('--end_cv_idx', type=int, default=8)
    
    parser.add_argument('--head_type', type=str, default='attentionhead', choices=['attentionhead', 'linearhead'])
    parser.add_argument('--output_type', type=str, default='note', choices=['note', 'beat', 'measure', 'voice', 'total_note_cat'])

    ### cuda ###
    parser.add_argument("--cpu", action="store_true") # default=False
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=[0,1], help="CUDA device ids")
    parser.add_argument("--do_eval", action = "store_true", help="whether to do evaluation")
    args = parser.parse_args()

    if args.task == 'melody':
        args.class_num = 4
    elif args.task == 'velocity':
        args.class_num = 7
    elif args.task == 'composer':
        args.class_num = 8
    elif args.task == 'emotion':
        args.class_num = 4
    elif args.task == 'percepiano':
        args.class_num = 19

    return args


def load_data(data_root, dataset, task, cv_idx=0):
    # data_root = f'Data/CP_data/{dataset}'

    if dataset == 'emotion':
        dataset = 'emopia'

    if dataset not in ['pop909', 'composer', 'emopia', 'percepiano']:
        print(f'Dataset {dataset} not supported')
        exit(1)


    X_train = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_train.npy'), allow_pickle=True)
    X_val = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_valid.npy'), allow_pickle=True)
    X_test = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_test.npy'), allow_pickle=True)

    print('X_train: {}, X_valid: {}, X_test: {}'.format(X_train.shape, X_val.shape, X_test.shape))

    if dataset == 'pop909':
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_{task[:3]}ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_{task[:3]}ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_{task[:3]}ans.npy'), allow_pickle=True)
    elif dataset == 'percepiano':
        y_train = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_{cv_idx}_test_ans.npy'), allow_pickle=True)
    else:
        y_train = np.load(os.path.join(data_root, f'{dataset}_train_ans.npy'), allow_pickle=True)
        y_val = np.load(os.path.join(data_root, f'{dataset}_valid_ans.npy'), allow_pickle=True)
        y_test = np.load(os.path.join(data_root, f'{dataset}_test_ans.npy'), allow_pickle=True)

    print('y_train: {}, y_valid: {}, y_test: {}'.format(y_train.shape, y_val.shape, y_test.shape))

    return X_train, X_val, X_test, y_train, y_val, y_test

def custom_collate_fn(batch):
    note_location_batch = {'beat': pad_sequence([sample[3]['beat'] for sample in batch], batch_first=True).long(),
                        'measure': pad_sequence([sample[3]['measure'] for sample in batch], True).long(),
                        'section': pad_sequence([sample[3]['section'] for sample in batch], True).long(),
                        'voice': pad_sequence([sample[3]['voice'] for sample in batch], True).long()
                        }
    data_len_batch = [item[4] for item in batch]
    # Convert input_ids, labels, addons to tensors and stack
    input_ids_tensor = torch.stack([item[0] for item in batch], dim=0)
    labels_tensor = torch.stack([item[1] for item in batch], dim=0)
    addons_tensor = torch.stack([item[2] for item in batch], dim=0)
    return_tuple = (
        input_ids_tensor,
        labels_tensor,
        addons_tensor,
        note_location_batch,
        data_len_batch
    )
    if len(batch[0]) > 5:
        found_addon_idxs = [item[5] for item in batch]
        return_tuple += (found_addon_idxs,)
    return return_tuple

def main():
    # set seed
    seed = 2021
    torch.manual_seed(seed)             # cpu
    torch.cuda.manual_seed(seed)        # current gpu
    torch.cuda.manual_seed_all(seed)    # all gpu
    np.random.seed(seed)
    random.seed(seed)

    # argument
    args = get_args()

    print("Loading Dictionary")
    with open(args.dict_file, 'rb') as f:
        e2w, w2e = pickle.load(f)

    print("\nLoading Dataset") 
    if args.task == 'melody' or args.task == 'velocity':
        dataset = 'pop909' 
        seq_class = False
    elif args.task == 'composer':
        dataset = 'composer'
        seq_class = True
    elif args.task == 'emotion':
        dataset = 'emopia'
        seq_class = True
    elif args.task == "percepiano":
        dataset = 'percepiano'
        seq_class = True
    
    cv_results = defaultdict(list)
    for cv_idx in range(args.start_cv_idx, args.end_cv_idx):
        print()
        print(cv_idx)
        print()
        if args.addons_path:
            data_root = args.addons_path
        elif args.align_path:
            data_root = args.align_path
        elif "composition4" in args.name:
            data_root = f'Data/CP_data/{dataset}_composition4split_addon'
        elif "performer" in args.name:
            data_root = f'Data/CP_data/{dataset}_performersplit_addon'
        else:
            data_root = f'Data/CP_data/{dataset}'
        # data_root = args.addons_path if args.addons_path else f'Data/CP_data/{dataset}'
        X_train, X_val, X_test, y_train, y_val, y_test = load_data(data_root, dataset, f"{cv_idx}_" + args.task, cv_idx)
        
        if args.addons_path:
            #args.addons_path = os.path.join(args.addons_path, f"{args.task}")
            trainset = FinetuneDatasetAddon(X=X_train, y=y_train, addons_path=os.path.join(args.addons_path,f"{args.task}_{cv_idx}_train"))
            validset = FinetuneDatasetAddon(X=X_val, y=y_val, addons_path=os.path.join(args.addons_path,f"{args.task}_{cv_idx}_valid"))
            testset = FinetuneDatasetAddon(X=X_test, y=y_test, addons_path=os.path.join(args.addons_path,f"{args.task}_{cv_idx}_test"))
            train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, 
                                      collate_fn = custom_collate_fn, shuffle=True)
            print("   len of train_loader",len(train_loader))
            valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers,
                                      collate_fn = custom_collate_fn,)
            print("   len of valid_loader",len(valid_loader))
            test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers,
                                     collate_fn = custom_collate_fn,)
            print("   len of valid_loader",len(test_loader))
        elif args.align_path:
            trainset = FinetuneDatasetAlign(X=X_train, y=y_train, addons_path=os.path.join(args.align_path,f"{args.task}_{cv_idx}_train"))
            validset = FinetuneDatasetAlign(X=X_val, y=y_val, addons_path=os.path.join(args.align_path,f"{args.task}_{cv_idx}_valid"))
            testset = FinetuneDatasetAlign(X=X_test, y=y_test, addons_path=os.path.join(args.align_path,f"{args.task}_{cv_idx}_test"))
            train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            print("   len of train_loader",len(train_loader))
            valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
            print("   len of valid_loader",len(valid_loader))
            test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
            print("   len of valid_loader",len(test_loader))

        else:
            trainset = FinetuneDataset(X=X_train, y=y_train)
            validset = FinetuneDataset(X=X_val, y=y_val) 
            testset = FinetuneDataset(X=X_test, y=y_test) 

            train_loader = DataLoader(trainset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
            print("   len of train_loader",len(train_loader))
            valid_loader = DataLoader(validset, batch_size=args.batch_size, num_workers=args.num_workers)
            print("   len of valid_loader",len(valid_loader))
            test_loader = DataLoader(testset, batch_size=args.batch_size, num_workers=args.num_workers)
            print("   len of valid_loader",len(test_loader))


        print("\nBuilding BERT model")
        configuration = BertConfig(max_position_embeddings=args.max_seq_len,
                                    position_embedding_type='relative_key_query',
                                    hidden_size=args.hs)

        midibert = MidiBert(bertConfig=configuration, e2w=e2w, w2e=w2e)
        best_mdl = ''
        if not args.nopretrain and not args.do_eval:
            best_mdl = args.ckpt
            print("   Loading pre-trained model from", best_mdl.split('/')[-1])
            checkpoint = torch.load(best_mdl, map_location='cpu')
            midibert.load_state_dict(checkpoint['state_dict'])

        if args.addons_path:
            assert args.net_params_path, "net_params_path must be specified when using addons"
            net_params = json.load(open(args.net_params_path, 'r'))
            virtuosonet = VirtuosoNet(input_size = 60, bert_hidden_size=args.hs,
                                        net_params=net_params)
        elif args.align_path:
            virtuosonet = AlignLSTM(input_size = 2, bert_hidden_size=args.hs)
        else:
            virtuosonet = None
        
        index_layer = int(args.index_layer)-13
        print("\nCreating Finetune Trainer using index layer", index_layer)
        trainer = FinetuneTrainer(midibert, train_loader, valid_loader, test_loader, index_layer, args.lr, args.class_num,
                                    args.hs, y_test.shape, args.cpu, args.cuda_devices, None, bool(args.addons_path), virtuosonet,
                                    head_type=args.head_type, output_type = args.output_type, align = bool(args.align_path))

        if args.do_eval: # load fine-tuned model
            print("\nEvaluation Start")
            best_mdl = args.ckpt.replace("xx", str(cv_idx))
            print('\nLoad ckpt from', best_mdl)  
            checkpoint = torch.load(best_mdl, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['state_dict'])                
            save_dir = best_mdl.replace("model_best.ckpt", "")
            label_stds = json.load(open("/root/v2/muzic/virtuosonet/label_2round_std_reg_19_with0_rm_highstd0.json"))
            label_names = json.load(open(os.path.join(data_root, f"{dataset}_{cv_idx}_test_names.json")))
            label_stds = np.array([label_stds[label] for label in label_names])
            
        else:
            print("\nTraining Start")
            if args.addons_path:
                save_dir = os.path.join('result/finetune/', args.task + '_' + args.name, 
                                        f"{pathlib.Path(args.net_params_path).stem}_lr{args.lr}_bs{args.batch_size}_{args.head_type}_{args.output_type}", 
                                        f'{cv_idx}')
            else:
                if args.nopretrain:
                    save_dir = os.path.join('result/finetune/', args.task + '_' + args.name, "nopretrain",
                                        f"lr{args.lr}_bs{args.batch_size}_{args.head_type}", 
                                        f'{cv_idx}')
                else:
                    save_dir = os.path.join('result/finetune/', args.task + '_' + args.name, 
                                        f"lr{args.lr}_bs{args.batch_size}_{args.head_type}", 
                                        f'{cv_idx}')
            os.makedirs(save_dir, exist_ok=True)
            filename = os.path.join(save_dir, 'model.ckpt')
            print("   save model at {}".format(filename))

        best_r2, best_epoch = 0, 0  
        bad_cnt = 0

        if args.do_eval:
            # cv_result = []
            test_loss, test_r2, test_all_output = trainer.test()
            total_yhat, total_y = test_all_output
            # for each 19 classes, calculate r2, mse
            print('test loss: {}, test_r2: {}'.format(test_loss, test_r2))
            with open(os.path.join(save_dir, 'prediction.tsv'), 'w') as outfile:
                outfile.write("Loading fine-tuned model from " + best_mdl.split('/')[-1] + '\n')
                outfile.write('class_name,std_score_1,std_score_05,std_score_01,r2,mse\n')
                mse_total = mean_squared_error(total_y, total_yhat)
                std_score_1_totals = evaluate_std_score(total_yhat, total_y, label_stds, 1)
                std_score_05_totals = evaluate_std_score(total_yhat, total_y, label_stds, 0.5)
                std_score_01_totals = evaluate_std_score(total_yhat, total_y, label_stds, 0.1)
                std_score_1_total = np.mean(std_score_1_totals)
                std_score_05_total = np.mean(std_score_05_totals)
                std_score_01_total = np.mean(std_score_01_totals)

                outfile.write("total, {}, {}, {}, {}, {}\n".format(std_score_1_total, std_score_05_total, std_score_01_total, test_r2, mse_total))
                cv_results['total'].append((std_score_1_total, std_score_05_total, std_score_01_total, test_r2, mse_total))
                for i in range(args.class_num):
                    r2 = r2_score(total_y[:,i], total_yhat[:,i])
                    mse = mean_squared_error(total_y[:,i], total_yhat[:,i])
                    std_score_1 = std_score_1_totals[i]
                    std_score_05 = std_score_05_totals[i]
                    std_score_01 = std_score_01_totals[i]
                    outfile.write("{}, {}, {}, {}, {}, {}\n".format(i, std_score_1, std_score_05, std_score_01, r2, mse))
                    cv_results[i].append((std_score_1, std_score_05, std_score_01, r2, mse))
                # r2 of each section. section1: y is between 0~1/7, section2: y is between 1/7~2/7, ...
                r2_per_section = defaultdict(list)
                for i in range(len(total_y)):
                    for j in range(args.class_num):
                        section = total_y[i][j] * 7 // 1
                        r2_per_section[section].append(mean_squared_error([total_y[i][j]], [total_yhat[i][j]]))
                # sort r2_per_section by key
                r2_per_section = dict(sorted(r2_per_section.items()))
                for section, r2s in r2_per_section.items():
                    outfile.write("section{}, {}\n".format(section, np.mean(r2s)))

                
        else:
            with open(os.path.join(save_dir, 'log'), 'a') as outfile:
                outfile.write("Loading pre-trained model from " + best_mdl.split('/')[-1] + '\n')
                for epoch in range(args.epochs):
                    train_loss, train_r2 = trainer.train()
                    valid_loss, valid_r2 = trainer.valid()
                    test_loss, test_r2, test_all_output = trainer.test()

                    is_best = valid_r2 >= best_r2
                    best_r2 = max(valid_r2, best_r2)
                    
                    if is_best:
                        print('best epoch: {}'.format(epoch+1))
                        outfile.write('best epoch: {}\n'.format(epoch+1))
                        bad_cnt, best_epoch = 0, epoch
                    else:
                        bad_cnt += 1
                    
                    print('epoch: {}/{} | Train Loss: {} | Train r2: {} | Valid Loss: {} | Valid r2: {} | Test loss: {} | Test r2: {}'.format(
                        epoch+1, args.epochs, train_loss, train_r2, valid_loss, valid_r2, test_loss, test_r2))

                    trainer.save_checkpoint(epoch, train_r2, valid_r2, 
                                            valid_loss, train_loss, is_best, filename)


                    outfile.write('Epoch {}: train_loss={}, valid_loss={}, test_loss={}, train_r2={}, valid_r2={}, test_r2={}\n'.format(
                        epoch+1, train_loss, valid_loss, test_loss, train_r2, valid_r2, test_r2))
                
                    if bad_cnt > 20:
                        print('valid acc not improving for 20 epochs')
                        outfile.write('valid acc not improving for 20 epochs\n')
                        break
    if args.do_eval:
        final_save_dir = args.ckpt.replace("xx/model_best.ckpt", "prediction_all.tsv")
        with open(final_save_dir, 'w') as fw:
            fw.write("class_name,std_score_1,std_score_05,std_score_01,r2,mse\n")
            std_score_1s, std_score_05s, std_score_01s, r2s, mses = zip(*cv_results['total'])
            fw.write(f"total,{np.mean(std_score_1s)},{np.mean(std_score_05s)},{np.mean(std_score_01s)},{np.mean(r2s)},{np.mean(mses)}\n")
            for i in range(args.class_num):
                std_score_1s, std_score_05s, std_score_01s, r2s, mses = zip(*cv_results[i])
                fw.write(f"{i},{np.mean(std_score_1s)},{np.mean(std_score_05s)},{np.mean(std_score_01s)},{np.mean(r2s)},{np.mean(mses)}\n")

if __name__ == '__main__':
    main()

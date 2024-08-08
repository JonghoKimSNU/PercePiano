import os
from pathlib import Path
import glob
import pickle
import pathlib
import argparse
import numpy as np
from data_creation.prepare_data.model import *

# args.input_dir = 

def get_args():
    parser = argparse.ArgumentParser(description='')
    ### mode ###
    parser.add_argument('-t', '--task', default='', choices=['melody', 'velocity', 'composer', 'emotion', 'percepiano'])

    ### path ###
    parser.add_argument('--dict', type=str, default='data_creation/prepare_data/dict/CP.pkl')
    parser.add_argument('--dataset', type=str, choices=["pop909", "pop1k7", "ASAP", "pianist8", "emopia", 'percepiano'])
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--input_file', type=str, default='')
    parser.add_argument('--split', type=str, default='randomfold', 
                    choices=['randomfold', 'performerfold', "composition4fold"])
    ### parameter ###
    parser.add_argument('--max_len', type=int, default=512)
    
    ### output ###    
    parser.add_argument('--output_dir', default="Data/CP_data/tmp")
    parser.add_argument('--name', default="")   # will be saved as "{output_dir}/{name}.npy"
    parser.add_argument("--addons_path", type=str, default="", help="addon information to be loaded")
    parser.add_argument("--align_path", type=str, default="", help="align information to be loaded")

    args = parser.parse_args()

    if args.task == 'melody' and args.dataset != 'pop909':
        print('[error] melody task is only supported for pop909 dataset')
        exit(1)
    elif args.task == 'composer' and args.dataset != 'pianist8':
        print('[error] composer task is only supported for pianist8 dataset')
        exit(1)
    elif args.task == 'emotion' and args.dataset != 'emopia':
        print('[error] emotion task is only supported for emopia dataset')
        exit(1)
    elif args.dataset == None and args.input_dir == None and args.input_file == None:
        print('[error] Please specify the input directory or dataset')
        exit(1)

    return args


def extract(files, args, model, mode=''):
    '''
    files: list of midi path
    mode: 'train', 'valid', 'test', ''
    args.input_dir: '' or the directory to your custom data
    args.output_dir: the directory to store the data (and answer data) in CP representation
    '''
    assert len(files)

    print(f'Number of {mode} files: {len(files)}') 

    segments, ans, other_info = model.prepare_data(
        files, args.task, int(args.max_len), addons_path=args.addons_path)
    (addons, note_location, data_len) = other_info[0], other_info[1], other_info[2]
    found_addon_idxs = other_info[3] if len(other_info) > 3 else None

    dataset = args.dataset if args.dataset != 'pianist8' else 'composer'

    if dataset == 'percepiano':
        output_file = os.path.join(args.output_dir, f'{dataset}_{mode}.npy')

    np.save(output_file, segments)
    print(f'Data shape: {segments.shape}, saved at {output_file}')

    if args.task != '':
        if args.task == 'percepiano':
            ans_file = os.path.join(args.output_dir, f'{dataset}_{mode}_ans.npy')
        np.save(ans_file, ans)
        print(f'Answer shape: {ans.shape}, saved at {ans_file}')

    if args.addons_path != "":
        addons_file = os.path.join(args.output_dir, f'{dataset}_{mode}_addons.npy')
        np.save(addons_file, addons)
        print(f'Addons shape: {addons.shape}, saved at {addons_file}')
        note_location_file = os.path.join(args.output_dir, f'{dataset}_{mode}_note_location.pkl')
        with open(note_location_file, 'wb') as f:
            pickle.dump(note_location, f)
        data_len_file = os.path.join(args.output_dir, f'{dataset}_{mode}_data_len.npy')
        np.save(data_len_file, data_len)
        print(f'Data len shape: {data_len.shape}, saved at {data_len_file}')
            

def main(): 
    args = get_args()
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # initialize model
    model = CP(dict=args.dict)

    if args.dataset == 'percepiano':
        files = glob.glob(f'Data/Dataset/{args.dataset}/*.mid')
        # performerfold
        if args.split == 'performerfold':
            for i in range(0, 4):
                train_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/train.id").read().splitlines()
                valid_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/dev.id").read().splitlines()
                test_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/test.id").read().splitlines()
                train_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in train_list]
                valid_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in valid_list]
                test_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in test_list]
                extract(train_files, args, model, f'{i}_train')
                extract(valid_files, args, model, f'{i}_valid')
                extract(test_files, args, model, f'{i}_test')
        elif args.split == "composition4fold":
            for i in range(0, 4):
                train_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/train")
                train_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in train_list if ".mid" in f]
                valid_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/valid")
                valid_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in valid_list if ".mid" in f]
                test_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/test")
                test_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in test_list if ".mid" in f]
                train_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in train_list]
                valid_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in valid_list]
                test_files = [f'Data/Dataset/{args.dataset}/{f}.mid' for f in test_list]
                extract(train_files, args, model, f'{i}_train')
                extract(valid_files, args, model, f'{i}_valid')
                extract(test_files, args, model, f'{i}_test')

    elif args.input_dir:
        files = glob.glob(f'{args.input_dir}/*.mid')
        extract(files, args, model)
    elif args.input_file:
        files = [args.input_file]
        extract(files, args, model)
    else:
        print('not supported')
        exit(1)
    

if __name__ == '__main__':
    main()

import os
import json


def main(output_dir, split, dataset): 
    

    # performerfold
    if split == 'performerfold':
        for i in range(0, 4):
            train_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/train.id").read().splitlines()
            valid_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/dev.id").read().splitlines()
            test_list = open(f"/root/v2/muzic/music-xai/processed/xaiperformersplit_data_2round_mean_reg_19/{i}/test.id").read().splitlines()
            train_files = [f for f in train_list]
            valid_files = [f for f in valid_list]
            test_files = [f for f in test_list]
            # save in json
            with open(f'{output_dir}/{dataset}_{i}_test_names.json', 'w') as f:
                json.dump(test_files, f)
    elif split == "composition4fold":
        for i in range(0, 4):
            train_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/train")
            train_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in train_list if ".mid" in f]
            valid_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/valid")
            valid_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in valid_list if ".mid" in f]
            test_list = os.listdir(f"/root/v2/muzic/virtuosonet/m2pf_allround/composition4fold/{i}/test")
            test_list = [f.lstrip("all_2rounds_").rstrip(".mid.pkl") for f in test_list if ".mid" in f]
            train_files = [f for f in train_list]
            valid_files = [f for f in valid_list]
            test_files = [f for f in test_list]
            # save in json
            with open(f'{output_dir}/{dataset}_{i}_test_names.json', 'w') as f:
                json.dump(test_files, f)


if __name__ == '__main__':
    main(output_dir = "Data/CP_data/percepiano_performersplit_addon", dataset = 'percepiano', split = 'performerfold')
    main(output_dir = "Data/CP_data/percepiano_composition4split_addon", dataset = 'percepiano', split = 'composition4fold')



import csv
from collections import OrderedDict, defaultdict
import numpy as np
import json
from tqdm import tqdm

PIANIST_MAP = OrderedDict()

def get_segment(file_name):
    return int(file_name.split("_")[-1])

def get_music_label_map_new_with0():
    file = open('/root/v2/muzic/music-xai/processed/total_2rounds.csv', encoding="utf-8")
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)

    rows = []
    for row in csvreader:
        rows.append(row)

    # sort by each segments
    music_label_map = defaultdict(list)
    for row in rows:
        user = row[0]
        file_name = ".".join(row[2].split(".")[:-1])
        if "_score" in file_name: 
            file_name = file_name.replace("_score", "_Score")
        if ("Beethoven_WoO80" in file_name and "Score" in file_name and \
            get_segment(file_name) in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]) \
            or ("Schubert_" in file_name and \
                "_no.3_4bars" in file_name and \
                "_Score_" in file_name and get_segment(file_name) == 1):
            file_name = file_name.replace("_Score_", "_Score2_")
        label_row = row[3:-2]
        assert len(label_row) == 19
        for idx, elem in enumerate(label_row):
            if elem == "":
                label_row[idx] = 0.0
            else:
                label_row[idx] = float(elem)
        flag_wrong_label = any(value > 7.1 for value in label_row)
        if flag_wrong_label:
            continue
        music_label_map[file_name].append(label_row)
    return music_label_map


def midi_label_map_mean_19_with0_rm_high_std_low2(version=2, scale = 7): 
    music_label_map = get_music_label_map_new_with0()
    music_label_map_mean = dict()
    music_label_map_std = dict()
    # mean value
    for key, annot_list in tqdm(music_label_map.items()):
        # annot_list = [num_annotators * num_labels].
        # 1. if 0 in annotators, remove it.
        # 2. if all the annotators make the label as 0, remove the music.
        # 3. calculate mean value for each label.
        annot_list = np.array(annot_list)
        annot_list = np.transpose(annot_list)
        new_annot_list = []
        for annot in annot_list:
            annot = annot[annot != 0]
            if len(annot) > 0:
                annot = np.average(annot) / 7
                new_annot_list.append(annot)
            else:
                break
        if len(new_annot_list) == 19:
            music_label_map_mean[key] = new_annot_list
        else:
            print(f"skip {key}")

    # std value
    for key, annot_list in tqdm(music_label_map.items()):
        if len(annot_list) == 1:
            music_label_map_std[key] = [0] * 19 # 이게 원래 std 였음..
        stdevs = np.std(np.array(annot_list)/7, axis=0)
        stdevs = stdevs.tolist()
        music_label_map_std[key] = stdevs
        assert len(stdevs) == 19
    # add pianist info
    for key, annot_list in tqdm(music_label_map_mean.items()):
        if key.split("_")[-2] not in PIANIST_MAP:
            PIANIST_MAP[key.split("_")[-2]] = len(PIANIST_MAP)
    print(PIANIST_MAP)
    
    for key, annot_list in tqdm(music_label_map_mean.items()):
        music_label_map_mean[key].append(PIANIST_MAP[key.split("_")[-2]])

    json.dump(music_label_map_mean, open(f"label_2round_mean_reg_19_with0_rm_highstd{version}.json", 'w'))
    json.dump(music_label_map_std, open(f"label_2round_std_reg_19_with0_rm_highstd{version}.json", 'w'))
    

if __name__ == "__main__":
    midi_label_map_mean_19_with0_rm_high_std_low2(version=0, scale = 7)
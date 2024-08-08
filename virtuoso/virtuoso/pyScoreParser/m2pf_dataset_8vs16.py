
import argparse
# from .data_class import DataSet, DEFAULT_SCORE_FEATURES, DEFAULT_PERFORM_FEATURES, PieceData
# from pathlib import Path
# from .data_for_training import PairDataset
# from .feature_extraction import ScoreExtractor, PerformExtractor
from data_class import DataSet, DEFAULT_SCORE_FEATURES, \
      DEFAULT_PERFORM_FEATURES, PieceData, DEFAULT_PURE_PERFORM_FEATURES
from pathlib import Path
from data_for_training import PairDataset
from feature_extraction import ScoreExtractor, PerformExtractor
import glob
import os
import json
from tqdm import tqdm
import time
import random
import pickle
from sklearn.model_selection import KFold
random.seed(34)

def get_id(file_name):
    return ".".join(file_name.split('/')[-1].split('.')[:-1])

def get_fold(fold_map, file_name):
    return fold_map.get(get_id(file_name), -1)

def get_performer_id(file_name):
    return str(".".join(file_name.split('/')[-1].split('.')[:-1]).split("_")[-2])

def get_performer_fold_map(file_name):
    performer_id_map = {"0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5", 
                        "6": "6", "7": "7", "8": "8", "9": "9", "10": "10", "Score": "11", "12": "12"}
    performer_id = get_performer_id(file_name)
    return int(performer_id_map[performer_id])

class M2PFSet(DataSet):
    def __init__(self, path="", split = "valid", save = True):
        super().__init__(path, save=save, split=split)
        #self.split = split

    def load_data_list(self):
        perform_lists = glob.glob(os.path.join(self.path, f"{self.split}/*.mid")) 
        if not "interval" in self.split:
            perform_lists = [p for p in perform_lists if ".".join(os.path.basename(p).split(".")[:-1]) in json.load(open("/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json")).keys()]
        print('perform list length' , len(perform_lists))
        xml_list = []
        score_midis = []
        for file in perform_lists:
            
            if "AUDIO" in file: # 240502: add new files (Beethoven)
                score_midi_name = "/root/v2/muzic/virtuosonet/data/interval/score/beethoven_woo80_score2.mid"
                xml_name = "/root/v2/muzic/virtuosonet/data/interval/score/beethoven_woo80_score2.musicxml"
            else: 
                score_midi_name =  "_".join(file.split("_")[:-2]) + "_Score_" + "_".join(file.split("_")[-1:])
                score_midi_name = "/".join(score_midi_name.split("/")[:-2])+"/score_midi/" + os.path.basename(score_midi_name)
                # if os.path.exists(score_midi_name.replace("_Score", "_Score2")):
                #     score_midi_name = score_midi_name.replace("_Score", "_Score2")
                assert os.path.exists(score_midi_name)

                xml_basename = ".".join(os.path.basename(score_midi_name).split(".")[:-1])
                # 1 -> 01, 2-> 02 ...
                #if xml_basename.split("_")[-1].startswith("0"):
                #    xml_basename = "_".join(xml_basename.split("_")[:-1] + [xml_basename.split("_")[-1][1:]])
                xml_name = "/".join(score_midi_name.split("/")[:-2])+"/score_xml/" + xml_basename + ".musicxml"
                try:
                    assert os.path.exists(xml_name), print("error", score_midi_name, xml_name)
                except:
                    if "Schubert_D935_no.3" in score_midi_name:
                        print("error", score_midi_name, xml_name)
                    else:
                        raise AssertionError("error", score_midi_name, xml_name)
            
            xml_list.append(xml_name)
            score_midis.append(score_midi_name)
            # print(score_midi_name, xml_name, file)
            # id = os.path.basename(file_prefix)
            #label = label_map.get(id, None)
            #labels.append(label)
        composers = ["Schubert"] * len(score_midis)
        perform_lists = [[perform] for perform in perform_lists]

        return xml_list, score_midis, perform_lists, composers #, labels

    def load_all_piece(self, scores, perform_midis, score_midis, composers, save):
        for n in range(len(scores)):
            try:
                piece = PieceData(scores[n], perform_midis[n], score_midis[n], composers[n], save=save) # 여기로 들어간 다음에
                self.pieces.append(piece)
                for perf in piece.performances:
                    self.performances.append(perf)
            except Exception as ex:
                # TODO: TGK: this is ambiguous. Can we specify which file 
                # (score? performance? matching?) and in which function the error occur?
                print(f'Error while processing {scores[n]}. Error type :{ex}')
        self.num_performances = len(self.performances)


if __name__ == '__main__':


############################## alignment version 190813, 8 fold split
    dataset = M2PFSet(path = "/root/v2/muzic/virtuosonet/data", split = "all_2rounds", save=False)
    for piece in dataset.pieces:
        try:
            piece.extract_perform_features(DEFAULT_PERFORM_FEATURES)
            piece.extract_perform_features(DEFAULT_PURE_PERFORM_FEATURES)
            piece.extract_score_features(DEFAULT_SCORE_FEATURES)
        except Exception as e: # 이거 alignment 돌아가는 순서에 따라서 어떨 때는 잘 되고 어떨 땐 안되고... 원인을 모르겠음.
            print(e)
            print(piece.meta.perform_lists[0])
            piece.performances = []
    pair_data = PairDataset(dataset)

    ##### 2rounds. 
    # (1) random 8fold
    # (2) cross-composer: D960, Beet: train&valid, D935: test
    # (3) performer based fold: Schubert.  

    # (1) random 8fold
    all_data = [filename for filename in os.listdir("/root/v2/muzic/virtuosonet/data/all_2rounds") if ".mid" in filename]
    domain = json.load(open("/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json")).keys()
    domain = [d.lower() for d in domain]
    all_data = [p for p in all_data if ".".join(os.path.basename(p).split(".")[:-1]).lower() in domain]
    all_data_16bars = [filename for filename in all_data if "16bars" in filename]
    random.shuffle(all_data_16bars)

    # make 2 datset: one contain only 16 bars, the other contain 8 bars (first half of 16 bars)

    test_list = all_data_16bars[:int(len(all_data_16bars)*0.15)]
    train_list = all_data_16bars[int(len(all_data_16bars)*0.15):int(len(all_data_16bars)*0.85)]
    valid_list = all_data_16bars[int(len(all_data_16bars)*0.85):]
    # perf feat
    VNET_INPUT_KEYS =  ('mmidi_pitch', 'mmidi_velocity', 'mmidi_start_time', 'mmidi_end_time')
    pair_data.update_dataset_split_type_m2pf(train=train_list, valid=valid_list, test = test_list)
    pair_data.update_mean_stds_of_entire_dataset() 
    pair_data.save_features_for_virtuosoNet(f"m2pf_allround/1round/16bars_perffeatonly", ["train", "valid", "test"], input_key_list=VNET_INPUT_KEYS, std_tempo=False)

    # find the correspondin 8bars. 
    # e.g. Schubert_D960_mv2_16bars_1_01.mid -> Schubert_D960_mv2_8bars_1_01.mid,
    # Schubert_D960_mv2_16bars_1_02.mid -> Schubert_D960_mv2_8bars_1_03.mid, 
    # Schubert_D960_mv2_16bars_1_03.mid -> Schubert_D960_mv2_8bars_1_05.mid, 
    
    def get_8bars (data_list_16bars):
        data_list_8bars = []
        for filename in data_list_16bars:
            filename = filename.replace("16bars", "8bars")
            segment = filename.split("_")[-1].split(".")[0]
            segment_8bars = str(int(segment)*2-1).zfill(2)
            filename = filename.split("_")[:-1] + [segment_8bars] + [".mid"]
            filename = "_".join(filename)
            data_list_8bars.append(filename)
        return data_list_8bars

    train_list_8bars = get_8bars(train_list)
    valid_list_8bars = get_8bars(valid_list)
    test_list_8bars = get_8bars(test_list)
    pair_data.update_dataset_split_type_m2pf(train=train_list_8bars, valid=valid_list_8bars, test = test_list_8bars)
    pair_data.update_mean_stds_of_entire_dataset()
    pair_data.save_features_for_virtuosoNet(f"m2pf_allround/1round/8bars_perffeatonly", ["train", "valid", "test"], input_key_list=VNET_INPUT_KEYS, std_tempo=False)
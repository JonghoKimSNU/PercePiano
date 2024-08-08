
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
random.seed(7)

BEETHOVEN_PERFORMER_IDS=[1,2,3,4,7,11,14,18,19,22,24,26]
#target = 'refactory/test_examples'
#xml_name = 'musicxml_cleaned.musicxml'
#target = '/root/v2/muzic/virtuosonet/data/'
# error_set = ("/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_8_13.mid",
#              "/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_0_01.mid",
#              "/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_0_32.mid",
#              "/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_12_29.mid",
#              "/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_7_23.mid",
#              "/root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_0_17.mid",
# )

#print(f'save and load {target}')

def get_id(file_name):
    return ".".join(file_name.split('/')[-1].split('.')[:-1])

def get_fold(fold_map, file_name):
    return fold_map.get(get_id(file_name), -1)

def get_performer_id(file_name):
    return str(".".join(file_name.split('/')[-1].split('.')[:-1]).split("_")[-2])

def get_performer_fold_map(file_name):
    # 베토벤은 26까지 슈베르트D935는 14까지
    beethoven_id_map = {elem: idx for idx, elem in enumerate(BEETHOVEN_PERFORMER_IDS)}
    performer_id = get_performer_id(file_name)
    if performer_id == "Score":
        return 0
    elif performer_id == "Score2":
        return -1
    if "Beethoven" in file_name:
        return beethoven_id_map[int(performer_id)]
    else:
        return int(performer_id)

class M2PFSet(DataSet):
    def __init__(self, path="", split = "valid", save = True):
        super().__init__(path, save=save, split=split)
        #self.split = split

    def load_data_list(self):
        #all_xml_list = sorted(glob.glob("/root/v2/muzic/virtuosonet/data/xml/*.musicxml"))
        #all_score_midis = ["/root/v2/muzic/virtuosonet/data/segmented_midi/" + Path(xml_name).stem + ".mid" for xml_name in xml_list]
        perform_lists = glob.glob(os.path.join(self.path, f"{self.split}/*.mid")) 
        #perform_lists = [p for p in perform_lists if os.path.basename(p).split(".")[0] in json.load(open("/root/v2/muzic/virtuosonet/midi_label_map_mean_reg_cls_19_with0_rm_highstd_2.json")).keys()]
        perform_lists = [p for p in perform_lists if ".".join(os.path.basename(p).split(".")[:-1]) in json.load(open("/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json")).keys()]
        print('perform list length' , len(perform_lists))
        #perform_lists = [p for p in perform_lists if not ("mv3" in p and "16bars" in p)]
        # /root/v2/muzic/virtuosonet/data/valid/Schubert_D960_mv3_8bars_8_13.mid
        xml_list = []
        #label_map = json.load(open("/root/v2/muzic/music-xai/processed/midi_label_map_mean_reg_cls_19_with0.json"))
        score_midis = []
        for file in perform_lists:
            # file_prefix = file.split(".")[0]
            score_midi_name =  "_".join(file.split("_")[:-2]) + "_Score_" + "_".join(file.split("_")[-1:])
            score_midi_name = "/".join(score_midi_name.split("/")[:-2])+"/score_midi/" + os.path.basename(score_midi_name)
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

    """  2rounds. """


    all_data = [filename for filename in os.listdir("/root/v2/muzic/virtuosonet/data/all_2rounds") if ".mid" in filename]
    domain = json.load(open("/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json")).keys()
    domain = [d.lower() for d in domain]
    all_data = [p for p in all_data if ".".join(os.path.basename(p).split(".")[:-1]).lower() in domain]
    print("all_data", len(all_data))
    random.shuffle(all_data)

    """ 4 fold """

    # (4) performer based fold:
    test_list = [data for data in all_data if get_performer_fold_map(data) in [1, 2]]
    print("test_list", len(test_list))
    all_data_without_test = [filename for filename in all_data if filename not in test_list]
    for fold in range(0,4): # 4 fold
        valid_performers = [fold*2+3, fold*2+4]
        valid_list = [data for data in all_data_without_test if get_performer_fold_map(data) in valid_performers]
        train_list = [data for data in all_data_without_test if data not in valid_list]
        pair_data.update_dataset_split_type_m2pf(train=train_list, valid=valid_list, test = test_list)
        pair_data.update_mean_stds_of_entire_dataset()

        VNET_INPUT_KEYS =  ('duration', 'beat_importance', 'measure_length', 
                    'qpm_primo',
                    'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                    'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                    'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                    'slur_beam_vec',  'composer_vec', 'notation', 
                    'tempo_primo',
                    'beat_tempo', 'measure_tempo', 'section_tempo', 'velocity', 'onset_deviation', 'beat_dynamics', 'measure_dynamics',
                            'articulation', 'pedal_refresh_time', # articulation, beat_tempo에 start time, end time이 들어감
                                'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                                'pedal_refresh', 'pedal_cut',
                                'mmidi_pitch', 'mmidi_velocity', 'mmidi_start_time', 'mmidi_end_time'
                    )
        pair_data.update_dataset_split_type_m2pf(train=train_list, valid=valid_list, test = test_list)
        pair_data.update_mean_stds_of_entire_dataset() 
        pair_data.save_features_for_virtuosoNet(f"m2pf_allround/performer4fold/{fold}", ["train", "valid", "test"], input_key_list=VNET_INPUT_KEYS, std_tempo=False)
        
        """perffeatonly"""
        VNET_INPUT_KEYS =  ('mmidi_pitch', 'mmidi_velocity', 'mmidi_start_time', 'mmidi_end_time')

        pair_data.update_dataset_split_type_m2pf(train=train_list, valid=valid_list, test = test_list)
        pair_data.update_mean_stds_of_entire_dataset() 
        pair_data.save_features_for_virtuosoNet(f"m2pf_allround/performer4fold_perffeatonly_wopedal/{fold}", ["train", "valid", "test"], input_key_list=VNET_INPUT_KEYS, std_tempo=False)
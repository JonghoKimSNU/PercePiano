import argparse

# from .data_class import DataSet, DEFAULT_SCORE_FEATURES, DEFAULT_PERFORM_FEATURES, PieceData
# from pathlib import Path
# from .data_for_training import PairDataset
# from .feature_extraction import ScoreExtractor, PerformExtractor
from data_class import (
    DataSet,
    DEFAULT_SCORE_FEATURES,
    DEFAULT_PERFORM_FEATURES,
    PieceData,
    DEFAULT_PURE_PERFORM_FEATURES,
)
from pathlib import Path
from data_for_training import PairDataset
from feature_extraction import ScoreExtractor, PerformExtractor
from m2pf_dataset_performerfold import M2PFSet
import glob
import os
import json
from tqdm import tqdm
import time
import random
import pickle

random.seed(7)


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)


def main(save_path="/root/muzic/MIDI-BERT/Data/Dataset/percepiano/virtuosonet/", path="/root/v2/muzic/virtuosonet/data",
         split="all_2rounds", save=False):
    dataset = M2PFSet(
        path=path,
        split=split,
        save=save,
    )
    for piece in dataset.pieces:
        try:
            piece.extract_perform_features(DEFAULT_PERFORM_FEATURES)
            piece.extract_perform_features(DEFAULT_PURE_PERFORM_FEATURES)
            piece.extract_score_features(DEFAULT_SCORE_FEATURES)
        except (
            Exception
        ) as e:  # 이거 alignment 돌아가는 순서에 따라서 어떨 때는 잘 되고 어떨 땐 안되고... 원인을 모르겠음.
            print(e)
            print(piece.meta.perform_lists[0])
            piece.performances = []
    pair_data = PairDataset(dataset)
    print()
    os.makedirs(save_path, exist_ok=True)
    for pair in tqdm(pair_data.data_pairs):
        perform_path = pair.perform_path
        feature_data = pair.features
        note_location = feature_data.pop("note_location")
        feature_data.pop("labels")
        num_notes = feature_data.pop("num_notes")
        index_to_be_deleted = []
        for i, mmidi in enumerate(
            zip(feature_data["mmidi_pitch"], feature_data["mmidi_velocity"])
        ):
            if mmidi[0] == 0 and mmidi[1] == 0:
                index_to_be_deleted.append(i)
        for key, value in feature_data.items():
            if (
                not isinstance(value, list) or len(value) != num_notes
            ):  # global features like qpm_primo, tempo_primo, composer_vec
                value = [value] * num_notes
                feature_data[key] = value
            filtered_value = [
                v for i, v in enumerate(value) if i not in index_to_be_deleted
            ]
            feature_data[key] = filtered_value
        # filter note_location according to index_to_be_deleted
        note_location = {
            'beat': [elem for i, elem in enumerate(note_location['beat']) if i not in index_to_be_deleted],
            'voice': [elem for i, elem in enumerate(note_location['voice']) if i not in index_to_be_deleted],
            'measure': [elem for i, elem in enumerate(note_location['measure']) if i not in index_to_be_deleted],
            'section': [elem for i, elem in enumerate(note_location['section']) if i not in index_to_be_deleted],
        }
        feature_data["note_location"] = note_location
        # assert same length of note_location and other features
        assert len(feature_data["mmidi_pitch"]) == len(feature_data["note_location"]["beat"])
        pickle.dump(
            feature_data,
            open(os.path.join(save_path, perform_path.split("/")[-1].replace(".mid", ".pkl")),"wb"),
        )

def test():
    check_loaded = open("/root/muzic/MIDI-BERT/Data/Dataset/percepiano/virtuosonet/Beethoven_WoO80_thema_8bars_1_1.pkl","rb")
    check_loaded = pickle.load(check_loaded)
    print(check_loaded.keys())
    print(check_loaded["note_location"])
    print(check_loaded["mmidi_pitch"])

if __name__ == "__main__":
    # save_path = "/root/muzic/MIDI-BERT/Data/Dataset/percepiano/virtuosonet/" # save only if aligned with performance
    main(save_path="/root/muzic/MIDI-BERT/Data/Dataset/percepiano/virtuosonet/")
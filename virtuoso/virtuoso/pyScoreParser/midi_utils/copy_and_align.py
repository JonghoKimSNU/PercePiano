from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import subprocess
import utils
import argparse
import pretty_midi
import glob
import time
from tqdm import tqdm
import json

INPUT_DIR = 'data/all_2rounds'
LABELS= json.load(open('/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json')).keys()
parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=INPUT_DIR,
                    help="Abs path to midi folder")
parser.add_argument("--wholescore", action='store_true',
                    help="use whole score file")
parser.add_argument("--align_dir", default='/root/v2/muzic/virtuosonet/data/AlignmentTool/',
                    help="Abs path to Nakamura's Alignment tool")
args = parser.parse_args()

os.chdir(args.align_dir)
t0 = time.time()
midi_files = glob.glob(f"{INPUT_DIR}/*.mid")

# use only files that have label
midi_files = [el for el in midi_files if ".".join(os.path.basename(el).split(".")[:-1]) in LABELS]
print(midi_files, len(midi_files))

n_match = 0
n_unmatch = 0
for midi_file in tqdm(midi_files):
    if 'midi.mid' in midi_file or 'XP.mid' in midi_file or 'midi_cleaned.mid' in midi_file:
        continue

    if os.path.isfile(midi_file.replace('.mid', '_infer_corresp.txt')):
        n_match += 1
        continue

    score_midi = "_".join(midi_file.split("_")[:-2]) + "_Score_" + "_".join(midi_file.split("_")[-1:])
    score_midi = os.path.join("/root/v2/muzic/virtuosonet/data/score_midi", os.path.basename(score_midi))
    perform_midi = midi_file
    assert os.path.exists(perform_midi)
    if not os.path.exists(score_midi):
        print("score midi not exists", score_midi, perform_midi)
        n_unmatch += 1
        continue

    shutil.copy(perform_midi, os.path.join(args.align_dir, 'infer.mid'))
    shutil.copy(score_midi, os.path.join(args.align_dir, 'score.mid'))

    try:
        subprocess.check_call(["bash", f"{args.align_dir}MIDIToMIDIAlign.sh", "score", "infer"])
        shutil.move('infer_corresp.txt', midi_file.replace('.mid', '_infer_corresp.txt'))
        shutil.move('infer_match.txt', midi_file.replace('.mid', '_infer_match.txt'))
        shutil.move('infer_spr.txt', midi_file.replace('.mid', '_infer_spr.txt'))
        shutil.move('score_spr.txt', os.path.join(args.align_dir, '_score_spr.txt'))
        n_match += 1
    except:
        print('Error to process {}'.format(midi_file))
        n_unmatch += 1
        pass

print('match:{:d}, unmatch:{:d}'.format(n_match, n_unmatch))

print("time passed", time.time() - t0)
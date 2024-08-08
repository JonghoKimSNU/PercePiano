# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import os
# import shutil
# import subprocess
# from. import utils as utils
# import argparse
# import pretty_midi

# # find midi files in INPUT_DIR, and align it using Nakamura's alignment tool.
# # (https://midialignment.github.io/demo.html)
# # midi.mid in same subdirectory will be regarded as score file.
# # make alignment result files in same directory. read Nakamura's manual for detail.

# INPUT_DIR = '/home/ilcobo2/chopin'

# parser = argparse.ArgumentParser()
# parser.add_argument("--input_dir", default=INPUT_DIR,
#                     help="Abs path to midi folder")
# parser.add_argument("--align_dir", default='/home/ilcobo2/AlignmentTool_v2',
#                     help="Abs path to Nakamura's Alignment tool")
# args = parser.parse_args()
# INPUT_DIR = args.input_dir

# os.chdir(args.align_dir)

# '''
# # read from text list
# f = open('temp_fix.txt', 'rb')
# lines = f.readlines()
# f.close()
# midi_files = [el.strip() for el in lines]
# '''

# # read from folder
# midi_files = utils.find_files_in_subdir(INPUT_DIR, '*.mid')

# n_match = 0
# n_unmatch = 0
# for midi_file in midi_files:
    
#     if 'midi.mid' in midi_file or 'XP.mid' in midi_file or 'midi_cleaned.mid' in midi_file:
#         continue

#     if 'Chopin_Sonata' in midi_file:
#         continue

#     if os.path.isfile(midi_file.replace('.mid', '_infer_corresp.txt')):
#         n_match += 1
#         continue

#     file_folder, file_name = utils.split_head_and_tail(midi_file)
#     perform_midi = midi_file
#     score_midi = os.path.join(file_folder, 'midi_cleaned.mid')
#     if not os.path.isfile(score_midi):
#         score_midi = os.path.join(file_folder, 'midi.mid')
#     print(perform_midi)
#     print(score_midi)

#     mid = pretty_midi.PrettyMIDI(score_midi)

#     n_notes = len(mid.instruments[0].notes)

#     '''
#     if n_notes >= 8000:
#         n_unmatch +=1
#         continue
#     '''

#     shutil.copy(perform_midi, os.path.join(args.align_dir, 'infer.mid'))
#     shutil.copy(score_midi, os.path.join(args.align_dir, 'score.mid'))

#     try:
#         subprocess.check_call(["sudo", "sh", "MIDIToMIDIAlign.sh", "score", "infer"])
#     except:
#         print('Error to process {}'.format(midi_file))
#         pass
#     else:
#         shutil.move('infer_corresp.txt', midi_file.replace('.mid', '_infer_corresp.txt'))
#         shutil.move('infer_match.txt', midi_file.replace('.mid', '_infer_match.txt'))
#         shutil.move('infer_spr.txt', midi_file.replace('.mid', '_infer_spr.txt'))
#         shutil.move('score_spr.txt', os.path.join(args.align_dir, '_score_spr.txt'))
# print('match:{:d}, unmatch:{:d}'.format(n_match, n_unmatch))


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
# find midi files in INPUT_DIR, and align it using Nakamura's alignment tool.
# (https://midialignment.github.io/demo.html)
# midi.mid in same subdirectory will be regarded as score file.
# make alignment result files in same directory. read Nakamura's manual for detail.

INPUT_DIR = '/root/v2/muzic/virtuosonet/data/all_2rounds'
LABELS= json.load(open('/root/v2/muzic/virtuosonet/label_2round_mean_reg_19_with0_rm_highstd0.json')).keys()

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default=INPUT_DIR,
                    help="Abs path to midi folder")
parser.add_argument("--wholescore", action='store_true',
                    help="use whole score file")
parser.add_argument("--from8to16", type=str, default="", choices=["8to16", "8to8"],
                    help="match 8 bar with 16 bars")
parser.add_argument("--align_dir", default='/root/v2/muzic/virtuosonet/data/AlignmentTool_v190813/',
                    help="Abs path to Nakamura's Alignment tool")
parser.add_argument("--align_dir2", default='/root/v2/muzic/virtuosonet/data/AlignmentTool/',
                    help="Abs path to Nakamura's Alignment tool for supplementray")
args = parser.parse_args()
print(args)
os.chdir(args.align_dir)
t0 = time.time()


# read from folder
#midi_files = utils.find_files_in_subdir(INPUT_DIR, '*.mid')
if args.from8to16:
    midi_files = glob.glob(f"{INPUT_DIR}/*8bars*.mid")

else:
    midi_files = glob.glob(f"{INPUT_DIR}/*.mid")

# use only files that have label
midi_files = [el for el in midi_files if ".".join(os.path.basename(el).split(".")[:-1]) in LABELS]
print(midi_files, len(midi_files))
n_match = 0
n_unmatch = 0
unmatch_list = []
for midi_file in tqdm(midi_files):
    if 'midi.mid' in midi_file or 'XP.mid' in midi_file or 'midi_cleaned.mid' in midi_file:
        continue

    if os.path.isfile(midi_file.replace('.mid', '_infer_corresp.txt')):
        n_match += 1
        continue
    else:
        score_midi = "_".join(midi_file.split("_")[:-2]) + "_Score_" + "_".join(midi_file.split("_")[-1:])
        score_midi = os.path.join(os.path.dirname(midi_file),".." , "score_midi", os.path.basename(score_midi))
    perform_midi = midi_file
    assert os.path.exists(perform_midi)
    if not os.path.exists(score_midi):
        print("score midi not exists", score_midi)
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
        unmatch_list += [midi_file]
        pass

print('match:{:d}, unmatch:{:d}'.format(n_match, n_unmatch))
print(unmatch_list)
print("time passed", time.time() - t0)


import numpy as np
import pickle
from tqdm import tqdm
import data_creation.prepare_data.utils as utils
import json
import os
import csv

def read_corresp(txtpath):
    file = open(txtpath, 'r')
    reader = csv.reader(file, dialect='excel', delimiter='\t')
    corresp_list = []
    for row in reader:
        if len(row) == 1:
            continue
        temp_dic = {'alignID': row[0], 'alignOntime': row[1], 'alignSitch': row[2], 'alignPitch': row[3], 'alignOnvel': row[4], 'refID':row[5], 'refOntime':row[6], 'refSitch':row[7], 'refPitch':row[8], 'refOnvel':row[9] }
        corresp_list.append(temp_dic)

    return corresp_list

def find_by_key(alist, key1, value1, key2, value2):
    for i, dic in enumerate(alist):
        if abs(float(dic[key1]) - value1) < 0.01 and int(dic[key2]) == value2: 
            return i
    return -1

Composer = {
    "Bethel": 0,
    "Clayderman": 1,
    "Einaudi": 2,
    "Hancock": 3,
    "Hillsong": 4,
    "Hisaishi": 5,
    "Ryuichi": 6,
    "Yiruma": 7,
    "Padding": 8,
}

Emotion = {
    "Q1": 0,
    "Q2": 1,
    "Q3": 2,
    "Q4": 3,
}

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

# percepiano
percepiano_label_map = json.load(open('Data/Dataset/label_2round_mean_reg_19_with0_rm_highstd0.json'))
ADDON_FEATURE_LIST=['beat_importance', 'measure_length', 
                    'qpm_primo', 'following_rest', 'distance_from_abs_dynamic', 
                    'distance_from_recent_tempo', 'beat_position', 'xml_position', 
                    'grace_order', 'preceded_by_grace_note', 'followed_by_fermata_rest', 
                    'tempo', 'dynamic', 'time_sig_vec', 'slur_beam_vec', 
                    'notation', 'tempo_primo', 'beat_tempo', 'onset_deviation', 'articulation', 
                    'pedal_refresh_time', 'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal', 'pedal_refresh', 'pedal_cut', 
                    'beat_dynamics', 'measure_tempo', 'measure_dynamics', 'section_tempo', 'note_location']

def find_addons(addons_path, input_path, note_items):
    filename = ".".join(os.path.basename(input_path).split('.')[:-1])+'.pkl'
    addons = pickle.load(open(os.path.join(addons_path, filename), 'rb'))
    if not len(addons['mmidi_pitch']) == len(note_items):
        print("different length for file ", input_path)
        assert len(addons['mmidi_pitch']) > len(note_items)
    found_addon_idxs = []
    # addon file format: dictionary. each value is list with same length. match with note_items. remove redundant keys, and make it a list
    for elem in note_items:
        # find elem in addons that by start time, pitch, velocity all match
        # save according to the found order
        found = False
        for i, (pitch, velocity, start) in enumerate(zip(addons['mmidi_pitch'], addons['mmidi_velocity'], addons['mmidi_start_time'])):
            if elem.pitch == pitch and elem.velocity == velocity and (elem.start == round(start,4) or elem.start == round(start,4)-0.0001 or elem.start == round(start,4)+0.0001):
                found = True
                found_addon_idxs.append(i)
                break
        if not found:
            print(elem)
            print([(a,b,c) for (a,b,c) in zip(addons['mmidi_pitch'], addons['mmidi_velocity'], addons['mmidi_start_time'])])
            assert found
    note_location = addons.pop('note_location')
    # remove keys that are not in ADDON_FEATURE_LIST
    # for each value in addons,
    #  1) remove the ones that are not in found_addon_idxs, 2) sort by found_addon_idxs
    for key in list(addons.keys()):
        if key not in ADDON_FEATURE_LIST:
            addons.pop(key)
            continue
        addons[key] = [addons[key][i] for i in found_addon_idxs]
    # 'addons' is a dictionary where the values are either lists of lists or lists of floats, and want to concatenate these values along the keys. The result should be a NumPy array with the shape (#length of list, #keys + a).
    # Find the length of the lists
    length_of_list = len(next(iter(addons.values())))

    # select note location according to found_addon_idxs, too (note location: dic of lists)
    note_location = {key: [note_location[key][i] for i in found_addon_idxs] for key in note_location}

    for _, v in note_location.items():
        assert len(v) == len(addons[ADDON_FEATURE_LIST[0]])

    # Concatenate values along keys
    result = []
    for i in range(length_of_list):
        row = []
        for key, value in addons.items():
            if isinstance(value[i], list):
                row.extend(value[i])
            else:
                row.append(value[i])
        result.append(row)
    # Convert the result to a NumPy array
    addons = np.array(result)
    return addons, note_location

class CP(object):
    def __init__(self, dict):
        # load dictionary
        self.event2word, self.word2event = pickle.load(open(dict, 'rb'))
        # pad word: ['Bar <PAD>', 'Position <PAD>', 'Pitch <PAD>', 'Duration <PAD>']
        self.pad_word = [self.event2word[etype]['%s <PAD>' % etype] for etype in self.event2word]

    def extract_events(self, input_path, task, addons_path=""):
        note_items, tempo_items, note_in_seconds = utils.read_items(input_path, return_note_in_seconds=bool(addons_path))
        if len(note_items) == 0:   # if the midi contains nothing
            return None
        if addons_path:
            addons, note_location = find_addons(addons_path, input_path, note_in_seconds)
            found_addon_idxs = None
        else:
            addons, note_location = None
            found_addon_idxs = None
        note_items = utils.quantize_items(note_items)
        max_time = note_items[-1].end
        items = tempo_items + note_items
        
        groups = utils.group_items(items, max_time)
        events = utils.item2event(groups, task)
        if addons_path:
            assert len(events) == len(addons)
        return events, (addons, note_location, found_addon_idxs)

    def padding(self, data, max_len, ans):
        pad_len = max_len - len(data)
        for _ in range(pad_len):
            if not ans:
                data.append(self.pad_word)
            else:
                data.append(0)

        return data

    def prepare_data(self, midi_paths, task, max_len, addons_path=""):
        all_words, all_ys = [], []
        all_data_lens = []
        all_addons = []
        all_note_location = []
        all_found_addon_idxs = []
        for path in tqdm(midi_paths):
            # extract events
            events, other_info = self.extract_events(path, task, addons_path)
            addons, note_location = other_info[0], other_info[1]
            # assert note_location is always non-decreasing
            # if note_location is not None:
            #     for v in range(1, len(note_location['measure'])):
            #         assert note_location['measure'][v] >= note_location['measure'][v-1] -> fail
            if len(other_info) == 3:
                found_addon_idxs = other_info[2]
            if not events:  # if midi contains nothing
                print(f'skip {path} because it is empty')
                continue
            # events to words
            words, ys = [], []
            for note_tuple in events:
                nts, to_class = [], -1
                for e in note_tuple:
                    e_text = '{} {}'.format(e.name, e.value)
                    nts.append(self.event2word[e.name][e_text])
                    if e.name == 'Pitch':
                        to_class = e.Type
                words.append(nts)
                if task == 'melody' or task == 'velocity':
                    ys.append(to_class+1)
                
            # slice to chunks so that max length = max_len (default: 512)
            slice_words, slice_ys = [], []
            slice_addons, slice_note_location = [], []
            slice_found_addon_idxs = []
            data_lens = []
            for i in range(0, len(words), max_len):
                slice_words.append(words[i:i+max_len])
                data_lens.append(len(slice_words[-1]))
                if addons is not None:
                    slice_addons.append(addons[i:i+max_len])
                if task == "percepiano":
                    name = ".".join(path.split('/')[-1].split('.')[:-1])
                    slice_ys.append(percepiano_label_map[name][:-1])
                    if note_location is not None:
                        slice_note_location.append(note_location)
                else:
                    slice_ys.append(ys[i:i+max_len])
            # padding or drop
            # drop only when the task is 'composer' and the data length < max_len//2
            if len(slice_words[-1]) < max_len:
                slice_words[-1] = self.padding(slice_words[-1], max_len, ans=False)
                if addons is not None:
                    slice_addons[-1] = np.pad(slice_addons[-1], ((0, max_len-len(slice_addons[-1])), (0,0)), 'constant', constant_values=0)
                    

            if (task == 'melody' or task == 'velocity') and len(slice_ys[-1]) < max_len:
                slice_ys[-1] = self.padding(slice_ys[-1], max_len, ans=True)
            all_words = all_words + slice_words
            all_ys = all_ys + slice_ys
            if addons is not None:
                all_addons = all_addons + slice_addons
                all_data_lens = all_data_lens + data_lens
                if note_location is not None:
                    all_note_location = all_note_location + slice_note_location
                if found_addon_idxs is not None:
                    all_found_addon_idxs = all_found_addon_idxs + slice_found_addon_idxs
    
        all_words = np.array(all_words)
        all_ys = np.array(all_ys)
        if addons is not None:
            all_addons = np.array(all_addons)
            all_data_lens = np.array(all_data_lens)
        

        return all_words, all_ys, (all_addons, all_note_location, all_data_lens, all_found_addon_idxs)

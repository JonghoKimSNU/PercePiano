import os
import random
import numpy as np
import pickle

from numpy.lib.npyio import save
#from . import dataset_split
import dataset_split
from pathlib import Path
from tqdm import tqdm

NORM_FEAT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 
                  'qpm_primo',
                  "section_tempo",
                  'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                  'beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time',
                  'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                  'pedal_refresh', 'pedal_cut', 
                  'beat_tempo', 'beat_dynamics', 'measure_tempo', 'measure_dynamics')

PRESERVE_FEAT_KEYS = ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 'following_rest')


VNET_COPY_DATA_KEYS = ('note_location', 'align_matched', 'articulation_loss_weight')
VNET_INPUT_KEYS =  ('midi_pitch', 'duration', 'beat_importance', 'measure_length', 
                    'qpm_primo',
                    "section_tempo",
                    'following_rest', 'distance_from_abs_dynamic', 'distance_from_recent_tempo',
                    'beat_position', 'xml_position', 'grace_order', 'preceded_by_grace_note',
                    'followed_by_fermata_rest', 'pitch', 'tempo', 'dynamic', 'time_sig_vec',
                    'slur_beam_vec',  'composer_vec', 'notation', 
                    'tempo_primo'
                    )

VNET_OUTPUT_KEYS = ('beat_tempo', 'velocity', 'onset_deviation', 'articulation', 'pedal_refresh_time', # articulation, beat_tempo에 start time, end time이 들어감 #TODO: 230530 pitch가 여기 들어가야 할 것 같은데
                            'pedal_cut_time', 'pedal_at_start', 'pedal_at_end', 'soft_pedal',
                            'pedal_refresh', 'pedal_cut')
VNET_BEAT_KEYS = ('beat_tempo', 'beat_dynamics')
VNET_MEAS_KEYS = ('measure_tempo', 'measure_dynamics')
                            # , 'beat_tempo', 'beat_dynamics', 'measure_tempo', 'measure_dynamics')

STD_KEYS = ('beat_tempo', 'measure_tempo', 'section_tempo')

class ScorePerformPairData:
  def __init__(self, piece, perform, exclude_long_graces=False):
    self.piece_path = piece.meta.xml_path
    self.perform_path = perform.midi_path
    self.graph_edges = piece.notes_graph
    if 'mmidi_pitch' in perform.perform_features:
      pair_idx_to_insert = self.align_different_length(perform)
      self.insert_unaligned_midi_features(piece, perform, pair_idx_to_insert)
      assert len(perform.perform_features['mmidi_pitch']) == len(piece.score_features['midi_pitch']), \
        print("two not matched", len(perform.perform_features['mmidi_pitch']), len(piece.score_features['midi_pitch']))
      assert len(perform.perform_features['mmidi_pitch']) == len(piece.score_features['note_location']['beat']), \
        print("two not matched", len(perform.perform_features['mmidi_pitch']), len(piece.score_features['note_location']['beat']))
      self.features = {**piece.score_features, **perform.perform_features}
      self.features['num_notes'] = len(perform.perform_features['mmidi_pitch'])
    else:
       self.features = {**piece.score_features, **perform.perform_features}
       self.features['num_notes'] = piece.num_notes
    
    self.split_type = None
    
    # self.features['num_notes'] = len(perform.midi_notes)
    self.features['labels'] = piece.labels
    if exclude_long_graces:
      self._exclude_long_graces()

  def _exclude_long_graces(self, max_grace_order=5):
    for i, order in enumerate(self.features['grace_order']):
      if order < -max_grace_order:
        self.features['align_matched'][i] = 0
        self.features['onset_deviation'][i] = 0.0
    for i, dev in enumerate(self.features['onset_deviation']):
      if abs(dev) > 4:
        self.features['align_matched'][i] = 0
        self.features['onset_deviation'][i] = 0.0

  def _find_insert_index(self, sorted_list, value):
    for index, element in enumerate(sorted_list):
        if value < element:
            return index
    return len(sorted_list) 

  def align_different_length(self, perform):
    matched_midi_note_idx = [pair['midi_idx'] for pair in perform.pairs if pair != []]
    unmatched_midi_note_idx = [i for i, note in reversed(list(enumerate(perform.midi_notes))) if i not in matched_midi_note_idx]
    pair_idx_to_insert = [self._find_insert_index(matched_midi_note_idx, midi_idx) for midi_idx in unmatched_midi_note_idx]
    # 1. insert mmidi_xx in matched midi note idx 2. insert mmidi_xx in unmatched midi note idx (according to pair_idx_to_insert)
    for key in perform.perform_features:
       if key.startswith('mmidi_'):
        new_list = []
        for i, pair in enumerate(perform.pairs):
          if pair != []:
            new_list.append(perform.perform_features[key][pair['midi_idx']])
          else:
            new_list.append(0)
        for target_idx, soruce_idx in zip(pair_idx_to_insert, unmatched_midi_note_idx):
          new_list.insert(target_idx, perform.perform_features[key][soruce_idx])
        
        perform.perform_features[key] = new_list
    # for i, pair in reversed(list(enumerate(perform.pairs))):
    #   if pair['midi_idx'] in unmatched_midi_note_idx:
    #     pair_idx_to_insert.append(i)
    return pair_idx_to_insert
  
  def insert_unaligned_midi_features(self, piece, perform, idx_to_insert):
    aligned_len = piece.num_notes
    for key in piece.score_features:
      if key in ['beat_position', 'xml_position', 'measure_length']:
        piece.score_features[key] = self.insert_value_to_list(piece.score_features[key], -1, idx_to_insert)
      elif key == 'note_location':
        for note_loc_key in piece.score_features[key]:
          if note_loc_key == 'voice':
            piece.score_features[key][note_loc_key] = self.insert_value_to_list(piece.score_features[key][note_loc_key], 0, idx_to_insert)
          else:
            piece.score_features[key][note_loc_key] = self.insert_value_to_list(piece.score_features[key][note_loc_key], -1, idx_to_insert)
      elif not isinstance(piece.score_features[key], list) or len(piece.score_features[key]) != aligned_len:
        continue
      else:
        piece.score_features[key] = self.insert_value_to_list(piece.score_features[key], 0, idx_to_insert)
    
    for key in perform.perform_features:
      if not isinstance(perform.perform_features[key], list) or len(perform.perform_features[key]) != aligned_len:
        # mmidi features
        continue
      else:
        perform.perform_features[key] = self.insert_value_to_list(perform.perform_features[key], 0, idx_to_insert)
        
  def insert_value_to_list(self, alist, value, idx_to_insert):
    if isinstance(alist[0], list):
      list_len = len(alist[0])
      for idx in idx_to_insert:
        if value == -1:
          if idx >= len(alist):
            insert_value = alist[-1]
          else:
            insert_value = alist[idx]
        else:
          insert_value = [value] * list_len
        alist.insert(idx, insert_value)
    else:
      for idx in idx_to_insert:
        if value == -1:
          if idx >= len(alist):
            insert_value = alist[-1]
          else:
            insert_value = alist[idx]
        else:
          insert_value = value
        alist.insert(idx, insert_value)
    return alist
     


class PairDataset:
  def __init__(self, dataset, exclude_long_graces=False):
    self.dataset_path = dataset.path
    self.data_pairs = []
    self.feature_stats = None
    for piece in dataset.pieces:
      for performance in piece.performances:
        if performance is None:
          continue
        if 'align_matched' not in performance.perform_features:
          continue
        len_notes =  len(performance.perform_features['align_matched'])
        num_aligned_notes = sum(performance.perform_features['align_matched'])
        # if len_notes - num_aligned_notes > 800:
        #   continue
        # if len_notes > num_aligned_notes * 1.5:
        #   continue
        self.data_pairs.append(ScorePerformPairData(piece, performance, exclude_long_graces))
        # if performance is not None \
        #         and 'align_matched' in performance.perform_features\
        #         and len(performance.perform_features['align_matched']) - sum(performance.perform_features['align_matched']) < 800:
        #     self.data_pairs.append(ScorePerformPairData(piece, performance, exclude_long_graces))
  
  def __len__(self):
    return len(self.data_pairs)

  def __getitem__(self, idx):
    return self.data_pairs[idx]

  def get_squeezed_features(self, target_feat_keys):
    squeezed_values = dict()
    for feat_type in target_feat_keys:
      squeezed_values[feat_type] = []
    for pair in self.data_pairs:
      for feat_type in target_feat_keys:
        if isinstance(pair.features[feat_type], list):
          squeezed_values[feat_type] += pair.features[feat_type]
        else:
          squeezed_values[feat_type].append(pair.features[feat_type]) # 전체 데이터를 고려
    return squeezed_values

  def update_mean_stds_of_entire_dataset(self, target_feat_keys=NORM_FEAT_KEYS):
      squeezed_values = self.get_squeezed_features(target_feat_keys)
      self.feature_stats = cal_mean_stds(squeezed_values, target_feat_keys)

  def update_dataset_split_type(self, valid_set_list=dataset_split.VALID_LIST, test_set_list=dataset_split.TEST_LIST):
      for pair in self.data_pairs:
        path = pair.piece_path
        for valid_name in valid_set_list:
          if valid_name in path:
            pair.split_type = 'valid'
            break
        if pair.split_type is None:
            pair.split_type = 'train'

  def update_dataset_split_type_m2pf(self, train = [], valid = [], test = []):
        for pair in self.data_pairs:
            name = os.path.split(pair.perform_path)[1]
            #print(name)
            if name in train:
                pair.split_type = 'train'
            elif name in valid:
                pair.split_type = 'valid'
            elif name in test:
                pair.split_type = 'test'
            # else:
            #     pair.split_type = 'test'

  def shuffle_data(self):
      random.shuffle(self.data_pairs)

  def save_features_for_virtuosoNet(self, 
                                    save_folder, 
                                    split_types = ["train","valid","test"],
                                    update_stats=True, 
                                    #valid_set_list=dataset_split.VALID_LIST, 
                                    #test_set_list=dataset_split.TEST_LIST, 
                                    input_key_list=VNET_INPUT_KEYS,
                                    output_key_list=VNET_OUTPUT_KEYS,
                                    std_tempo=True):
      print("save features for virtuosoNet", save_folder)
      '''
      Convert features into format of VirtuosoNet training data
      :return: None (save file)
      '''
      train = 0
      test = 0
      valid = 0
      def _flatten_path(file_path):
          return '_'.join(file_path.parts)

      save_folder = Path(save_folder)

      save_folder.mkdir(parents=True, exist_ok=True)
      for split in split_types:
        (save_folder / split).mkdir(exist_ok=True)
  
      #if update_stats:
      #    self.update_mean_stds_of_entire_dataset()
      #self.update_dataset_split_type(valid_set_list=valid_set_list, test_set_list=test_set_list)
      feature_converter = FeatureConverter(self.feature_stats, self.data_pairs[0].features, input_key_list, output_key_list, std_tempo=std_tempo) # len(self.data_pairs[1].features['midi_pitch']) # it has different length

      for pair_data in self.data_pairs:
          if pair_data.split_type is None:
              #print(pair_data.perform_path.split("/")[-1] + " is not assigned to any split type")
              continue
          if pair_data.split_type == 'train':
              train += 1
          elif pair_data.split_type == 'valid':
              valid += 1
          elif pair_data.split_type == 'test':
              test += 1

          #if pair_data.split_type == 'test':
             #print(pair_data.perform_path.split("/")[-1] + " is test")
          # formatted_data = dict()
          formatted_data = feature_converter(pair_data.features, std_tempo=std_tempo)
          # formatted_data['input_data'], formatted_data['output_data'], formatted_data['meas_level_data'], formatted_data['beat_level_data'] = \
          #       convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats, input_keys=input_key_list, output_keys=output_key_list)
          for key in VNET_COPY_DATA_KEYS:
              formatted_data[key] = pair_data.features[key]
          formatted_data['graph'] = pair_data.graph_edges
          formatted_data['score_path'] = pair_data.piece_path
          formatted_data['perform_path'] = pair_data.perform_path

          save_name = _flatten_path(
              Path(pair_data.perform_path).relative_to(Path(self.dataset_path))) + '.pkl'
        
          with open(save_folder / pair_data.split_type / save_name, "wb") as f:
              pickle.dump(formatted_data, f, protocol=2)
      print("train, valid, test : ", train, valid, test)
      for split_type in split_types:
        with open(save_folder / split_type / "stat.pkl", "wb") as f:
            pickle.dump({'stats': self.feature_stats, 
                            'input_keys': input_key_list, 
                            'output_keys': output_key_list, 
                            'measure_keys': VNET_MEAS_KEYS,
                            'key_to_dim': feature_converter.key_to_dim_idx
                            }, f, protocol=2)


  def save_features_for_analysis(self, 
                                    save_folder, 
                                    split_type = "train",
                                    update_stats=True, 
                                    #valid_set_list=dataset_split.VALID_LIST, 
                                    #test_set_list=dataset_split.TEST_LIST, 
                                    input_key_list=VNET_INPUT_KEYS,
                                    output_key_list=VNET_OUTPUT_KEYS):
      '''
      Convert features into format of VirtuosoNet training data
      :return: None (save file)
      '''
      def _flatten_path(file_path):
          return '_'.join(file_path.parts)

      save_folder = Path(save_folder)
      split_types = [split_type]

      save_folder.mkdir(parents=True, exist_ok=True)
      for split in split_types:
        (save_folder / split).mkdir(exist_ok=True)
  
      #if update_stats:
      #    self.update_mean_stds_of_entire_dataset()
      #self.update_dataset_split_type(valid_set_list=valid_set_list, test_set_list=test_set_list)
      self.feature_stats = dict()
      #samplefeaturedata
      feature_converter = FeatureConverter(self.feature_stats, self.data_pairs[0].features, input_key_list, output_key_list, std_tempo=self.std_tempo) # len(self.data_pairs[1].features['midi_pitch']) # it has different length
      # 여기 이미 Note location, beat tempo, onset_deviation, articulation

      for pair_data in tqdm(self.data_pairs):
          # formatted_data = dict()
          formatted_data = feature_converter(pair_data.features)
          # formatted_data['input_data'], formatted_data['output_data'], formatted_data['meas_level_data'], formatted_data['beat_level_data'] = \
          #       convert_feature_to_VirtuosoNet_format(pair_data.features, self.feature_stats, input_keys=input_key_list, output_keys=output_key_list)
          for key in VNET_COPY_DATA_KEYS:
              formatted_data[key] = pair_data.features[key]
          formatted_data['graph'] = pair_data.graph_edges
          formatted_data['score_path'] = pair_data.piece_path
          formatted_data['perform_path'] = pair_data.perform_path

          save_name = _flatten_path(
              Path(pair_data.perform_path).relative_to(Path(self.dataset_path))) + '.pkl'

          with open(save_folder / pair_data.split_type / save_name, "wb") as f:
              pickle.dump(formatted_data, f, protocol=2)

# def combine_dict_to_array():




def cal_mean_stds(feat_datas, target_features):
    stats = dict()
    for feat_type in target_features:
        mean = sum(feat_datas[feat_type]) / len(feat_datas[feat_type])
        var = sum((x-mean)**2 for x in feat_datas[feat_type]) / len(feat_datas[feat_type])
        stds = var ** 0.5
        if stds == 0:
            stds = 1
        stats[feat_type] = {'mean': mean, 'stds':stds}
    return stats


class FeatureConverter:
  def __init__(self, stats, sample_feature_data=None, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS, beat_keys=VNET_BEAT_KEYS, meas_keys=VNET_MEAS_KEYS, std_tempo=False):
    self.stats = stats
    self.keys = {'input': input_keys, 'output': output_keys, 'beat': beat_keys, 'meas': meas_keys}

    self.preserve_keys = PRESERVE_FEAT_KEYS
    if std_tempo:
       self.std_keys = STD_KEYS

    if sample_feature_data is not None:
      self._init_with_sample_data(sample_feature_data, std_tempo=std_tempo)
  
  def _init_with_sample_data(self, sample_feature_data, std_tempo=False):
    if 'num_notes' not in sample_feature_data.keys():
      sample_feature_data['num_notes'] = len(sample_feature_data[self.keys['input'][0]])
    self._preserve_feature_before_normalization(sample_feature_data)
    if std_tempo:
        self._calculate_feature_after_normalization2(sample_feature_data)
    self.dim = {}
    self.key_to_dim_idx = {}
    if sample_feature_data is not None:
      for key_type in self.keys:
        selected_type_features = []
        for key in self.keys[key_type]:
          value = self._check_if_global_and_normalize(sample_feature_data, key)
          selected_type_features.append(value)
        dimension, key_to_dim_idx = self._cal_dimension(selected_type_features, self.keys[key_type])
        self.dim[key_type] = dimension
        self.key_to_dim_idx[key_type] = key_to_dim_idx

  def _check_if_global_and_normalize(self, feature_data, key):
    value = feature_data[key]
    if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
        value = [value] * feature_data['num_notes']
    if key in self.stats:  # if key needs normalization,
        value = [(x - self.stats[key]['mean']) / self.stats[key]['stds'] for x in value]
    return value

  def _cal_dimension(self, data_with_all_features, keys): #여기서 84가 나오는디..
    assert len(data_with_all_features) == len(keys)
    total_length = 0
    key_to_dim_idx = {}
    for feat_data, key in zip(data_with_all_features, keys):
      if isinstance(feat_data[0], list):
        length = len(feat_data[0])
      else:
        length = 1
      key_to_dim_idx[key] = (total_length, total_length+length)
      total_length += length
    return total_length, key_to_dim_idx
  
  def _preserve_feature_before_normalization(self, feature_data):
    for key in self.preserve_keys:
      if key in feature_data:
        new_key_name = key + '_unnorm'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = feature_data[key][:]
        if new_key_name not in self.keys['input']:
          self.keys['input'] = tuple(list(self.keys['input']) + [new_key_name])

  def _calculate_feature_after_normalization(self, feature_data):
    for key in self.std_keys:
      if key in feature_data:
        value = [(x - self.stats[key]['mean']) / self.stats[key]['stds'] for x in feature_data[key]]

        new_key_name = key + '_std'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = np.std(value)
        if new_key_name not in self.keys['output']:
          self.keys['output'] = tuple(list(self.keys['output']) + [new_key_name])

        new_key_name = key + '_mean'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = np.mean(value)
        if new_key_name not in self.keys['output']:
          self.keys['output'] = tuple(list(self.keys['output']) + [new_key_name])

        new_key_name = key + '_max'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = np.max(value)
        if new_key_name not in self.keys['output']:
          self.keys['output'] = tuple(list(self.keys['output']) + [new_key_name])

        new_key_name = key + '_min'
        if new_key_name not in feature_data:
          feature_data[new_key_name] = np.min(value)
        if new_key_name not in self.keys['output']:
          self.keys['output'] = tuple(list(self.keys['output']) + [new_key_name])      
    return feature_data 

  def _calculate_feature_after_normalization2(self, feature_data):
    """
    calculate beat_tempo statistics at each measure level
    """
    key = "beat_tempo"
    if key in feature_data:
        value = [(x - self.stats[key]['mean']) / self.stats[key]['stds'] for x in feature_data[key]]
        measure_locs = feature_data['note_location']['measure']
        prev_meas_loc = 0
        tmp_inputs = [[],[],[],[]]
        tmp_meas_infos = []
        for i, meas_loc in enumerate(measure_locs):
            if meas_loc == prev_meas_loc:
                tmp_meas_infos.append(value[i])
            else:
                tmp_inputs[0].extend([float(np.std(tmp_meas_infos))] * len(tmp_meas_infos))
                tmp_inputs[1].extend([float(np.mean(tmp_meas_infos))] * len(tmp_meas_infos))
                tmp_inputs[2].extend([float(np.max(tmp_meas_infos))] * len(tmp_meas_infos))
                tmp_inputs[3].extend([float(np.min(tmp_meas_infos))] * len(tmp_meas_infos))
                tmp_meas_infos = []
                tmp_meas_infos.append(value[i])
            prev_meas_loc = meas_loc
        # last one
        tmp_inputs[0].extend([float(np.std(tmp_meas_infos))] * len(tmp_meas_infos))
        tmp_inputs[1].extend([float(np.mean(tmp_meas_infos))] * len(tmp_meas_infos))
        tmp_inputs[2].extend([float(np.max(tmp_meas_infos))] * len(tmp_meas_infos))
        tmp_inputs[3].extend([float(np.min(tmp_meas_infos))] * len(tmp_meas_infos))

        assert len(tmp_inputs[0]) == len(measure_locs)

        for i, new_key_name in enumerate(["_std", "_mean", "_max", "_min"]):
            new_key_name = key + new_key_name
            if new_key_name not in feature_data:
                feature_data[new_key_name] = tmp_inputs[i]
            if new_key_name not in self.keys['output']:
                self.keys['output'] = tuple(list(self.keys['output']) + [new_key_name])


    return feature_data 

  def make_feat_to_array(self, feature_data, key_type, std_tempo=False):
    if key_type == 'input':
      self._preserve_feature_before_normalization(feature_data)
    if std_tempo and key_type=='output':
      feature_data = self._calculate_feature_after_normalization2(feature_data)
    datas = [self._check_if_global_and_normalize(feature_data, key) for key in self.keys[key_type]]
    if hasattr(self, 'dim'): # 27개 (unnorm 5개 추가돼서)
      dimension = self.dim[key_type] # input = 84
    else:
      dimension, _ = self._cal_dimension(datas, self.keys[key_type]) 
    array = np.zeros((feature_data['num_notes'], dimension))
    current_idx = 0
    for value in datas:
      if isinstance(value[0], list):
        length = len(value[0])
        array[:, current_idx:current_idx + length] = value
      else:
        length = 1
        array[:,current_idx] = value
      current_idx += length
    return array

  def __call__(self, feature_data, std_tempo=False):
    '''
    feature_data (dict): score or perform features in dict
    '''
    if 'num_notes' not in feature_data.keys():
      feature_data['num_notes'] = len(feature_data[self.keys['input'][0]])
    self._preserve_feature_before_normalization(feature_data)
    if std_tempo:
      self._calculate_feature_after_normalization2(feature_data)    
    output = {}
    for key_type in self.keys:
      output[key_type] = self.make_feat_to_array(feature_data, key_type, std_tempo=std_tempo)
    return output

    

'''
def convert_feature_to_VirtuosoNet_format(feature_data, stats, input_keys=VNET_INPUT_KEYS, output_keys=VNET_OUTPUT_KEYS, meas_keys=VNET_MEAS_KEYS, beat_keys=VNET_BEAT_KEYS):
    if 'num_notes' not in feature_data.keys():
        feature_data['num_notes'] = len(feature_data[input_keys[0]])

    def check_if_global_and_normalize(key):
        value = feature_data[key]
        if not isinstance(value, list) or len(value) != feature_data['num_notes']:  # global features like qpm_primo, tempo_primo, composer_vec
            value = [value] * feature_data['num_notes']
        if key in stats:  # if key needs normalization,
            value = [(x - stats[key]['mean']) / stats[key]['stds'] for x in value]
        return value

    def add_to_list(alist, item):
        if isinstance(item, list):
            alist += item
        else:
            alist.append(item)
        return alist

    def cal_dimension(data_with_all_features, keys):
        assert len(data_with_all_features) == len(keys)
        total_length = 0
        key_to_dim_idx = {}
        for feat_data, key in zip(data_with_all_features, keys):
            if isinstance(feat_data[0], list):
                length = len(feat_data[0])
            else:
                length = 1
            key_to_dim_idx[key] = (total_length, total_length+length)
            total_length += length
        return total_length, key_to_dim_idx

    def make_feat_to_array(keys):
        datas = [] 
        for key in keys:
            value = check_if_global_and_normalize(key)
            datas.append(value)
        dimension, key_to_dim_idx = cal_dimension(datas, keys)
        array = np.zeros((feature_data['num_notes'], dimension))
        current_idx = 0
        for value in datas:
            if isinstance(value[0], list):
                length = len(value[0])
                array[:, current_idx:current_idx + length] = value
            else:
                length = 1
                array[:,current_idx] = value
            current_idx += length
        return array

    input_array = make_feat_to_array(input_keys)
    output_array = make_feat_to_array(output_keys)
    meas_array = make_feat_to_array(meas_keys)
    beat_array = make_feat_to_array(beat_keys)
    return input_array, output_array, meas_array, beat_array


def get_feature_from_entire_dataset(dataset, target_score_features, target_perform_features):
    # e.g. feature_type = ['score', 'duration'] or ['perform', 'beat_tempo']
    output_values = dict()
    for feat_type in (target_score_features + target_perform_features):
        output_values[feat_type] = []
    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_score_features:
                # output_values[feat_type] += piece.score_features[feat_type]
                output_values[feat_type].append(piece.score_features[feat_type])
            for feat_type in target_perform_features:
                output_values[feat_type].append(performance.perform_features[feat_type])
    return output_values



def cal_mean_stds_of_entire_dataset(dataset, target_features):
    # :param dataset: DataSet class
    # :param target_features: list of dictionary keys of features
    # :return: dictionary of mean and stds
    output_values = dict()
    for feat_type in (target_features):
        output_values[feat_type] = []

    for piece in dataset.pieces:
        for performance in piece.performances:
            for feat_type in target_features:
                if feat_type in piece.score_features:
                    output_values[feat_type] += piece.score_features[feat_type]
                elif feat_type in performance.perform_features:
                    output_values[feat_type] += performance.perform_features[feat_type]
                else:
                    print('Selected feature {} is not in the data'.format(feat_type))

    stats = cal_mean_stds(output_values, target_features)

    return stats

def normalize_feature(data_values, target_feat_keys):
    for feat in target_feat_keys:
        concatenated_data = [note for perf in data_values[feat] for note in perf]
        mean = sum(concatenated_data) / len(concatenated_data)
        var = sum(pow(x-mean,2) for x in concatenated_data) / len(concatenated_data)
        # data_values[feat] = [(x-mean) / (var ** 0.5) for x in data_values[feat]]
        for i, perf in enumerate(data_values[feat]):
            data_values[feat][i] = [(x-mean) / (var ** 0.5) for x in perf]
    return data_values


def normalize_pedal_value(pedal_value):
    return (pedal_value - 64)/128

'''
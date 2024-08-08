import torch
import torch.nn as nn
from torch.autograd import Variable

import model_constants as cons
from model_utils import make_higher_node, reparameterize, span_beat_to_note_num
import model_utils as utils
from module import GatedGraph, ContextAttention
from model_constants import QPM_INDEX, QPM_PRIMO_IDX, TEMPO_IDX, PITCH_IDX
from pyScoreParser.xml_utils import xml_notes_to_midi
from pyScoreParser.feature_to_performance import apply_tempo_perform_features

import encoder_score as encs
import encoder_perf as encp
import decoder as dec
import residual_selector as res
import note_embedder as nemb

from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence, pack_padded_sequence

# VOICE_IDX = 11
# PITCH_IDX = 13
# TEMPO_IDX = PITCH_IDX + 13
DYNAMICS_IDX = TEMPO_IDX + 5
LEN_DYNAMICS_VEC = 4
TEMPO_PRIMO_IDX = -2
NUM_VOICE_FEED_PARAM = 2

class VirtuosoNet(nn.Module):
    def __init__(self, net_param, data_stats):
        super(VirtuosoNet, self).__init__() 
        self.note_embedder = getattr(nemb, net_param.note_embedder_name)(net_param, data_stats)
        self.score_encoder = getattr(encs, net_param.score_encoder_name)(net_param) # HanEncoder
        self.performance_encoder = getattr(encp, net_param.performance_encoder_name)(net_param) # HanPerfEncoder
        self.network_params = net_param
        self.stats = data_stats
        self.stats['graph_keys'] = net_param.graph_keys
        self.out_fc = nn.Sequential(
            nn.Dropout(net_param.drop_out), # added
            nn.Linear(net_param.encoder.size * 2, net_param.encoder.size * 2),
            nn.GELU(),
            nn.Dropout(net_param.drop_out),
            nn.Linear(net_param.encoder.size * 2, net_param.num_label),
        )

    def forward(self, x, y = None, edges=None, note_locations=None, initial_z=None, expanded_y = None):
        x_embedded = self.note_embedder(x)
        score_embedding = self.score_encoder(x_embedded, edges, note_locations) # keys: dict_keys(['note', 'beat', 'measure', 'beat_spanned', 'measure_spanned', 'total_note_cat'])
        if initial_z is None:
            performance_embedding = self.performance_encoder(score_embedding, y, edges, note_locations, return_z=False, expanded_y = expanded_y)
            logits = self.out_fc(performance_embedding)
            return logits
        else:
            raise ValueError("initial z should be None")

class VirtuosoNetSingle(nn.Module):
    def __init__(self, net_param, data_stats):
        super(VirtuosoNetSingle, self).__init__()
        self.note_embedder = getattr(nemb, net_param.note_embedder_name)(net_param, data_stats)
        self.encoder_size = net_param.encoder.size
        self.num_attention_head = net_param.num_attention_head
        self.lstm = nn.LSTM(net_param.note.size, net_param.note.size,
                             net_param.note.layer + net_param.voice.layer + net_param.beat.layer + net_param.measure.layer, 
                             batch_first=True, bidirectional=True, dropout=net_param.drop_out)
        self.note_contractor = nn.Linear(self.encoder_size * 2, self.encoder_size * 2)
        self.note_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.network_params = net_param
        self.stats = data_stats
        self.stats['graph_keys'] = net_param.graph_keys
        self.out_fc = nn.Sequential(
            nn.Dropout(net_param.drop_out), # added
            nn.Linear(net_param.encoder.size * 2, net_param.encoder.size * 2),
            nn.GELU(),
            nn.Dropout(net_param.drop_out),
            nn.Linear(net_param.encoder.size * 2, net_param.num_label),
        )

    def forward(self, x, y = None, edges=None, note_locations=None, initial_z=None, expanded_y = None):
        x_embedded = self.note_embedder(x)
        score_embedding, _ = self.lstm(x_embedded)
        score_embedding, _ = pad_packed_sequence(score_embedding, True)
        note = self.note_contractor(score_embedding)
        note_output = self.note_attention(note)
        outputs = self.out_fc(note_output)
        return outputs

class VirtuosoNetMultiLevel(nn.Module):
    def __init__(self, net_param, data_stats, multi_level = "note,beat,measure"):
        super(VirtuosoNetMultiLevel, self).__init__() 
        self.note_embedder = getattr(nemb, net_param.note_embedder_name)(net_param, data_stats)
        self.score_encoder = getattr(encs, net_param.score_encoder_name)(net_param) # HanEncoder
        self.performance_encoder = getattr(encp, net_param.performance_encoder_name)(net_param) # HanPerfEncoder
        #self.residual_info_selector = getattr(res, net_param.residual_info_selector_name)(data_stats) # TempoVecMeasSelector
        #self.performance_decoder = getattr(dec, net_param.performance_decoder_name)(net_param) # HanMeasNoteDecoder
        self.network_params = net_param
        self.stats = data_stats
        self.stats['graph_keys'] = net_param.graph_keys

        self.encoder_size = net_param.encoder.size
        self.encoded_vector_size = net_param.encoded_vector_size
        self.encoder_input_size = net_param.encoder.input
        self.num_attention_head = net_param.num_attention_head
        self.multi_level = multi_level.split(",")
        print("levels: ", self.multi_level)
        if "measure" in self.multi_level:
            self.measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        if "beat" in self.multi_level:
            self.beat_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        if "voice" in self.multi_level:
            self.voice_contractor = nn.Linear(self.encoder_size * 4, self.encoder_size * 2)
            self.voice_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        if "note" in self.multi_level:
            self.note_contractor = nn.Linear(self.encoder_size * 2, self.encoder_size * 2)
            self.note_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        if "total_note_cat" in self.multi_level:
            self.performance_contractor = nn.Linear(self.encoder_size * 8, self.encoder_size * 2)
            self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.out_fc = nn.Sequential(
            nn.Dropout(net_param.drop_out), # added
            nn.Linear(net_param.encoder.size * 2, net_param.encoder.size * 2),
            nn.GELU(),
            nn.Dropout(net_param.drop_out),
            nn.Linear(net_param.encoder.size * 2, net_param.num_label),
        )

    def forward(self, x, y, edges= None, note_locations=None, initial_z=None):
        x_embedded = self.note_embedder(x)
        score_embedding = self.score_encoder(x_embedded, edges, note_locations)
        outputs = ()

        if "note" in self.multi_level:
            note = self.note_contractor(score_embedding['note'])
            note_output = self.note_attention(note)
            outputs += (self.out_fc(note_output),)
        if "voice" in self.multi_level:
            voice = self.voice_contractor(score_embedding['voice'])
            voice_output = self.voice_attention(voice)
            outputs += (self.out_fc(voice_output),)
        if "beat" in self.multi_level:
            beat_output = self.beat_attention(score_embedding['beat'])
            outputs += (self.out_fc(beat_output),)
        if "measure" in self.multi_level:
            measure_output = self.measure_attention(score_embedding['measure'])
            outputs += (self.out_fc(measure_output),)
        if "total_note_cat" in self.multi_level:
            total_note_cat = self.performance_contractor(score_embedding['total_note_cat'])
            total_note_cat_output = self.performance_final_attention(total_note_cat)
            outputs += (self.out_fc(total_note_cat_output),)

        return outputs

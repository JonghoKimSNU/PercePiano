import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from .model_utils import make_higher_node, reparameterize, masking_half, encode_with_net, run_hierarchy_lstm_with_pack
# from .module import GatedGraph, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack
from model_utils import make_higher_node, reparameterize, masking_half, encode_with_net, run_hierarchy_lstm_with_pack
from module import GatedGraph, SimpleAttention, ContextAttention, GatedGraphX, GatedGraphXBias, GraphConvStack


class BaselineEncoder(nn.Module):
  def __init__(self, net_params):
    super().__init__()
    self.performance_embedding_size = net_params.performance.size
    self.encoder_size = net_params.encoder.size
    self.encoded_vector_size = net_params.encoded_vector_size
    self.encoder_input_size = net_params.encoder.input
    self.encoder_layer_num = net_params.encoder.layer
    self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True, batch_first=True)
    
    self.performance_embedding_layer = nn.Linear(net_params.output_size, self.performance_embedding_size)
    self.performance_contractor =  nn.Linear(self.encoder_input_size, self.encoder_size)

    self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
    self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)

  def _expand_perf_feature(self, y):
    '''
    Simply expand performance features to larger dimension
    y (torch.Tensor): performance features (N x T x C)
    '''
    is_padded = (y==0).all(dim=-1)

    expanded_y = self.performance_embedding_layer(y)

    mask = torch.ones_like(expanded_y)
    mask[is_padded] = 0
    expanded_y *= mask
    # expanded_y[is_padded] = 0
    return expanded_y

  def _get_perform_style_from_input(self, perform_concat):
    perform_style_contracted = self.performance_contractor(perform_concat)
    mask = torch.ones_like(perform_style_contracted)
    mask[(perform_concat==0).all(dim=-1)] = 0
    perform_style_contracted *= mask

    perform_style_encoded = run_hierarchy_lstm_with_pack(perform_style_contracted, self.performance_note_encoder)
    perform_style_vector = perform_style_encoded.mean(dim=1)
    perform_z, perform_mu, perform_var = \
        encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
    return perform_z, perform_mu, perform_var
  
  def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
    total_note_cat = score_embedding['total_note_cat']

    expanded_y = self._expand_perf_feature(y)
    perform_concat = torch.cat((total_note_cat, expanded_y), 2)
    
    perform_z, perform_mu, perform_var = self._get_perform_style_from_input(perform_concat)
    if return_z:
        return sample_multiple_z(perform_mu, perform_var, num_samples)
    return perform_z, perform_mu, perform_var    

class BaselineGRUEncoder(BaselineEncoder):
  def __init__(self, net_params):
    super().__init__(net_params)
    self.performance_note_encoder = nn.GRU(self.encoder_size, self.encoder_size, bidirectional=True, batch_first=True)

class PerformanceEncoder(nn.Module):
  def __init__(self, net_params):
    super().__init__()
    self.performance_embedding_size = net_params.performance.size
    self.encoder_size = net_params.encoder.size
    self.encoded_vector_size = net_params.encoded_vector_size
    self.encoder_input_size = net_params.encoder.input
    self.encoder_layer_num = net_params.encoder.layer
    self.num_attention_head = net_params.num_attention_head

    self.performance_measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

    self.performance_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
    self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
    self.performance_encoder_mean = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)
    self.performance_encoder_var = nn.Linear(self.encoder_size * 2, self.encoded_vector_size)


  def _get_note_hidden_states(self, perform_style_contracted, edges):
    raise NotImplementedError

  def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
    perform_style_contracted = self.performance_contractor(perform_concat)
    mask = torch.ones_like(perform_style_contracted)
    mask[(perform_concat==0).all(dim=-1)] = 0
    perform_style_contracted *= mask
    perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
    performance_measure_nodes = make_higher_node(perform_style_note_hidden, self.performance_measure_attention, measure_numbers,
                                            measure_numbers, lower_is_note=True) # FIXME:
    perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
    # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
    perform_style_vector = self.performance_final_attention(perform_style_encoded)
    perform_z, perform_mu, perform_var = \
        encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
    return perform_z, perform_mu, perform_var

  def _expand_perf_feature(self, y):
    '''
    Simply expand performance features to larger dimension

    y (torch.Tensor): performance features (N x T x C)
    '''
    is_padded = (y==0).all(dim=-1)

    expanded_y = self.performance_embedding_layer(y)

    mask = torch.ones_like(expanded_y)
    mask[is_padded] = 0
    expanded_y *= mask
    # expanded_y[is_padded] = 0
    return expanded_y

  def _masking_notes(self, perform_concat):
    '''
    perform_concat (torch.Tensor): N x T x C
    out (torch.Tensor): N x T//2 x C
    '''
    return masking_half(perform_concat)

  def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
    perform_style_contracted = self.performance_contractor(perform_concat)
    mask = torch.ones_like(perform_style_contracted)
    mask[(perform_concat==0).all(dim=-1)] = 0
    perform_style_contracted *= mask
    perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
    # performance_beat_nodes = make_higher_node(perform_style_note_hidden, self.performance_beat_attention, beat_numbers,
    #                                         beat_numbers, lower_is_note=True)
    
    performance_measure_nodes = make_higher_node(perform_style_note_hidden, self.performance_measure_attention, measure_numbers,
                                            measure_numbers, lower_is_note=True) # FIXME:
    perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
    # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
    perform_style_vector = self.performance_final_attention(perform_style_encoded)
    perform_z, perform_mu, perform_var = \
        encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
    return perform_z, perform_mu, perform_var

  def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
    measure_numbers = note_locations['measure']
    total_note_cat = score_embedding['total_note_cat']

    expanded_y = self._expand_perf_feature(y)
    perform_concat = torch.cat((total_note_cat, expanded_y), 2)
    perform_concat = self._masking_notes(perform_concat)
    
    perform_z, perform_mu, perform_var = self._get_perform_style_from_input(perform_concat, edges, measure_numbers)
    if return_z:
        return sample_multiple_z(perform_mu, perform_var, num_samples)
    return perform_z, perform_mu, perform_var


class HanPerfEncoder(PerformanceEncoder):
    def __init__(self, net_params) -> None:
      super(HanPerfEncoder, self).__init__(net_params)
      self.performance_note_encoder = nn.LSTM(self.encoder_size, self.encoder_size, bidirectional=True, batch_first=True)
      self.performance_embedding_layer = nn.Sequential(
          nn.Linear(net_params.output_size, self.performance_embedding_size),
          # nn.Dropout(net_params.drop_out),
          # nn.ReLU(),
      )
      self.performance_contractor = nn.Sequential(
          nn.Linear(self.encoder_input_size, self.encoder_size),
          # nn.Dropout(net_params.drop_out),
          # # nn.BatchNorm1d(self.encoder_size),
          # nn.ReLU()
      )
    #   if net_params.encoder.mid_encoder:
    #       self.mid_encoder=True
    #       self.performance_encoder_mid = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
    #       self.performance_mid_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

    def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers):
        perform_style_contracted = self.performance_contractor(perform_concat)
        mask = torch.ones_like(perform_style_contracted)
        mask[(perform_concat==0).all(dim=-1)] = 0
        perform_style_contracted *= mask
        # lstm 첫번째 통과
        perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
        performance_measure_nodes = make_higher_node(perform_style_note_hidden, self.performance_measure_attention, measure_numbers,
                                                measure_numbers, lower_is_note=True) # FIXME:
        # lstm 두번째 통과
        perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
        # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        # perform_z, perform_mu, perform_var = \
        #     encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        return perform_style_vector

    def _get_note_hidden_states(self, perform_style_contracted, edges):
        perform_note_encoded = run_hierarchy_lstm_with_pack(perform_style_contracted, self.performance_note_encoder)
        # perform_note_encoded, _ = self.performance_note_encoder(perform_style_contracted)
        return perform_note_encoded

    def _get_perform_style_and_mid_from_input(self, perform_concat, edges, measure_numbers):
        perform_style_contracted = self.performance_contractor(perform_concat)
        mask = torch.ones_like(perform_style_contracted)
        mask[(perform_concat==0).all(dim=-1)] = 0
        perform_style_contracted *= mask
        # lstm 첫번째 통과
        perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
        performance_measure_nodes = make_higher_node(perform_style_note_hidden, self.performance_measure_attention, measure_numbers,
                                                measure_numbers, lower_is_note=True) # FIXME:
        # lstm 중간통과
        perform_style_encoded_mid = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder_mid)
        # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        # https://github.com/hotpotqa/hotpot/blob/3635853403a8735609ee997664e1528f4480762a/model.py#LL85C26-L85C26
        perform_style_mid = self.performance_mid_attention(perform_style_encoded_mid)
        
        # lstm 두번째 통과
        perform_style_encoded = run_hierarchy_lstm_with_pack(perform_style_encoded_mid, self.performance_encoder)
        # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)

        # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        # perform_style_vector = self.performance_final_attention(perform_style_encoded)
        # perform_z, perform_mu, perform_var = \
        #     encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        return perform_style_vector, perform_style_mid

    def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
        measure_numbers = note_locations['measure']
        total_note_cat = score_embedding['total_note_cat']
        if expanded_y is None:
            expanded_y = self._expand_perf_feature(y)
        
        perform_concat = torch.cat((total_note_cat, expanded_y), 2)
        perform_concat = self._masking_notes(perform_concat)
        
        # if self.mid_encoder:
        #     perform_style_vector, mid_output = self._get_perform_style_and_mid_from_input(perform_concat, edges, measure_numbers)
        #     return perform_style_vector, mid_output
        # else:
        perform_style_vector = self._get_perform_style_from_input(perform_concat, edges, measure_numbers)
        #if return_z:
        #    return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_style_vector

class NonMaskingHanPerfEncoder(HanPerfEncoder):
  def __init__(self, net_params) -> None:
    super(NonMaskingHanPerfEncoder, self).__init__(net_params)

  def _masking_notes(self, perform_concat):
    '''
    This Encoder does not mask notes
    '''
    return perform_concat 

class HanSkipEncoder(nn.Module):
    def __init__(self, net_params) -> None:
        super().__init__()
        self.encoder_size = net_params.encoder.size
        self.encoded_vector_size = net_params.encoded_vector_size
        self.encoder_input_size = net_params.encoder.input
        self.num_attention_head = net_params.num_attention_head
        self.performance_contractor = nn.Linear(self.encoder_size * 8, self.encoder_size * 2)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
    def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
       contracted = self.performance_contractor(score_embedding['total_note_cat'])
       # output = self.performance_final_attention(score_embedding['measure'])
       output = self.performance_final_attention(contracted)
    # TODO: output을 뭘로 써야할까??
       return output

class HanMultiLevelEncoder(nn.Module):
    def __init__(self, net_params) -> None:
        super().__init__()
        self.encoder_size = net_params.encoder.size
        self.encoded_vector_size = net_params.encoded_vector_size
        self.encoder_input_size = net_params.encoder.input
        self.num_attention_head = net_params.num_attention_head
        
        self.measure_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.beat_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.note_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)
        self.performance_contractor = nn.Linear(self.encoder_size * 8, self.encoder_size * 2)
        self.performance_final_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

    def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
        #contracted = self.performance_contractor(score_embedding['total_note_cat'])
        measure_output = self.measure_attention(score_embedding['measure'])
        beat_output = self.beat_attention(score_embedding['beat'])
        note_output = self.note_attention(score_embedding['note'])
        total_note_cat = self.performance_contractor(score_embedding['total_note_cat'])
        output = self.performance_final_attention(total_note_cat)

        # TODO: output을 뭘로 써야할까??
        return output, measure_output, beat_output, note_output


class NonMaskingBeatHanPerfEncoder(HanPerfEncoder):
    def __init__(self, net_params) -> None:
        super(NonMaskingBeatHanPerfEncoder, self).__init__(net_params)
        self.performance_beat_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

        self.performance_beat_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
    def _masking_notes(self, perform_concat):
       return perform_concat

    def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers, beat_numbers):
        perform_style_contracted = self.performance_contractor(perform_concat)
        mask = torch.ones_like(perform_style_contracted)
        mask[(perform_concat==0).all(dim=-1)] = 0
        perform_style_contracted *= mask #[bs, length, 64]
        # lstm 첫번째 통과
        perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
        performance_beat_nodes = make_higher_node(perform_style_note_hidden, self.performance_beat_attention, beat_numbers,
                                                beat_numbers, lower_is_note=True)
        performance_style_beat_hidden = run_hierarchy_lstm_with_pack(performance_beat_nodes, self.performance_beat_encoder)
        performance_measure_nodes = make_higher_node(performance_style_beat_hidden, self.performance_measure_attention, beat_numbers,
                                                measure_numbers, lower_is_note=False)
        
        # lstm 두번째 통과
        perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
        # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
        perform_style_vector = self.performance_final_attention(perform_style_encoded)
        # perform_z, perform_mu, perform_var = \
        #     encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
        return perform_style_vector
    
    def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
        measure_numbers = note_locations['measure']
        beat_numbers = note_locations['beat']
        total_note_cat = score_embedding['total_note_cat']
        if expanded_y is None:
            expanded_y = self._expand_perf_feature(y)
        
        perform_concat = torch.cat((total_note_cat, expanded_y), 2)
        perform_concat = self._masking_notes(perform_concat)
        
        # if self.mid_encoder:
        #     perform_style_vector, mid_output = self._get_perform_style_and_mid_from_input(perform_concat, edges, measure_numbers)
        #     return perform_style_vector, mid_output
        # else:
        perform_style_vector = self._get_perform_style_from_input(perform_concat, edges, measure_numbers, beat_numbers)
        #if return_z:
        #    return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_style_vector

# without score!
# class NonMaskingOnlyHanPerfEncoder(HanPerfEncoder):
#     def __init__(self, net_params) -> None:
#         super(NonMaskingBeatHanPerfEncoder, self).__init__(net_params)
#         self.performance_beat_attention = ContextAttention(self.encoder_size * 2, self.num_attention_head)

#         self.performance_beat_encoder = nn.LSTM(self.encoder_size * 2, self.encoder_size,  num_layers=self.encoder_layer_num, batch_first=True, bidirectional=True)
#     def _masking_notes(self, perform_concat):
#        return perform_concat

#     def _get_perform_style_from_input(self, perform_concat, edges, measure_numbers, beat_numbers):
#         perform_style_contracted = self.performance_contractor(perform_concat)
#         mask = torch.ones_like(perform_style_contracted)
#         mask[(perform_concat==0).all(dim=-1)] = 0
#         perform_style_contracted *= mask #[bs, length, 64]
#         # lstm 첫번째 통과
#         perform_style_note_hidden = self._get_note_hidden_states(perform_style_contracted, edges)
#         performance_beat_nodes = make_higher_node(perform_style_note_hidden, self.performance_beat_attention, beat_numbers,
#                                                 beat_numbers, lower_is_note=True)
#         performance_style_beat_hidden = run_hierarchy_lstm_with_pack(performance_beat_nodes, self.performance_beat_encoder)
#         performance_measure_nodes = make_higher_node(performance_style_beat_hidden, self.performance_measure_attention, beat_numbers,
#                                                 measure_numbers, lower_is_note=False)
        
#         # lstm 두번째 통과
#         perform_style_encoded = run_hierarchy_lstm_with_pack(performance_measure_nodes, self.performance_encoder)
#         # perform_style_encoded, _ = self.performance_encoder(performance_measure_nodes)
#         perform_style_vector = self.performance_final_attention(perform_style_encoded)
#         # perform_z, perform_mu, perform_var = \
#         #     encode_with_net(perform_style_vector, self.performance_encoder_mean, self.performance_encoder_var)
#         return perform_style_vector

#     def run_voice_net(self, y, voice_numbers, max_voice):
#         if isinstance(y, torch.nn.utils.rnn.PackedSequence):
#           y, _ = nn.utils.rnn.pad_packed_sequence(y, True)
#         num_notes = y.size(1)
#         output = torch.zeros(y.shape[0], y.shape[1], self.voice_net.hidden_size * 2).to(y.device)
#         # voice_numbers = torch.Tensor(voice_numbers)
#         for i in range(1,max_voice+1):
#           voice_x_bool = voice_numbers == i
#           num_voice_notes = torch.sum(voice_x_bool)
#           num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)

#           if num_voice_notes > 0:
#             voice_notes = [y[i, voice_x_bool[i]] if torch.sum(voice_x_bool[i])>0 else torch.zeros(1,y.shape[-1]).to(y.device) for i in range(len(y)) ]
#             voice_x = pad_sequence(voice_notes, True)
#             pack_voice_x = pack_padded_sequence(voice_x, [len(x) for x in voice_notes], True, False)
#             ith_voice_out, _ = self.voice_net(pack_voice_x)
#             ith_voice_out, _ = pad_packed_sequence(ith_voice_out, True)
            
#             span_mat = torch.zeros(y.shape[0], num_notes, voice_x.shape[1]).to(y.device)
#             voice_where = torch.nonzero(voice_x_bool)
#             span_mat[voice_where[:,0], voice_where[:,1], torch.cat([torch.arange(num_batch_voice_notes[i]) for i in range(len(y))])] = 1

#             output += torch.bmm(span_mat, ith_voice_out)
#         return output

#     def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
#         voice_numbers = note_locations['voice']
#         measure_numbers = note_locations['measure']
#         beat_numbers = note_locations['beat']
#         total_note_cat = score_embedding['total_note_cat']
#         if expanded_y is None:
#             expanded_y = self._expand_perf_feature(y)
        
#         perform_concat = torch.cat((total_note_cat, expanded_y), 2)
#         perform_concat = self._masking_notes(perform_concat)
        
#         # if self.mid_encoder:
#         #     perform_style_vector, mid_output = self._get_perform_style_and_mid_from_input(perform_concat, edges, measure_numbers)
#         #     return perform_style_vector, mid_output
#         # else:
#         perform_style_vector = self._get_perform_style_from_input(perform_concat, edges, measure_numbers, beat_numbers)
#         #if return_z:
#         #    return sample_multiple_z(perform_mu, perform_var, num_samples)
#         return perform_style_vector

class NonMaskingHanPerfEncoderWithoutScore(NonMaskingHanPerfEncoder):
    def __init__(self, net_params) -> None:
        super(NonMaskingHanPerfEncoderWithoutScore, self).__init__(net_params)

    def forward(self, score_embedding, y=None, edges=None, note_locations=None, return_z=False, num_samples=10, return_mid=False, expanded_y=None):
        measure_numbers = note_locations['measure']
        total_note_cat = score_embedding['total_note_cat']
        if expanded_y is None:
            expanded_y = self._expand_perf_feature(y)
        
        total_note_cat = torch.zeros_like(total_note_cat)
        perform_concat = torch.cat((total_note_cat, expanded_y), 2)
        perform_concat = self._masking_notes(perform_concat)
        
        # if self.mid_encoder:
        #     perform_style_vector, mid_output = self._get_perform_style_and_mid_from_input(perform_concat, edges, measure_numbers)
        #     return perform_style_vector, mid_output
        # else:
        perform_style_vector = self._get_perform_style_from_input(perform_concat, edges, measure_numbers)
        #if return_z:
        #    return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_style_vector

# class DropoutHanPerfEncoder(HanPerfEncoder):
#   def __init__(self, net_params) -> None:
#     super(DropoutHanPerfEncoder, self).__init__(net_params)
#     self.performance_embedding_layer = nn.Sequential(
#         nn.Dropout(net_params.drop_out),
#         nn.Linear(net_params.output_size, self.performance_embedding_size),
        
#         # nn.ReLU(),
#     )
#     self.performance_contractor = nn.Sequential(
#         nn.Dropout(net_params.drop_out),
#         nn.Linear(self.encoder_input_size, self.encoder_size),
#         # # nn.BatchNorm1d(self.encoder_size),
#         # nn.ReLU()
#     )    

class IsgnPerfEncoder(PerformanceEncoder):
  def __init__(self, net_params):
    super(IsgnPerfEncoder, self).__init__(net_params)
    self.performance_contractor = nn.Sequential(
        nn.Linear(net_params.encoder.input, net_params.encoder.size * 2),
        # nn.Dropout(net_params.drop_out),
        # nn.ReLU(),
    )
    self.performance_embedding_layer = nn.Sequential(
        nn.Linear(net_params.output_size, net_params.performance.size),
        # nn.Dropout(net_params.drop_out),
        # nn.ReLU(),
    )
    self.performance_graph_encoder = GatedGraph(net_params.encoder.size * 2, net_params.num_edge_types)

  def _get_note_hidden_states(self, perform_style_contracted, edges):
    return self.performance_graph_encoder(perform_style_contracted, edges)

  def _masking_notes(self, perform_concat):
    return perform_concat # TODO: Implement it with sliced graph

class IsgnPerfEncoderX(IsgnPerfEncoder):
  def __init__(self, net_params):
      super(IsgnPerfEncoderX, self).__init__(net_params)
      self.performance_contractor = nn.Sequential(
        nn.Linear(net_params.encoder.input, net_params.encoder.size),
        nn.Dropout(net_params.drop_out),
        nn.ReLU(),
      )
      self.performance_graph_encoder = GatedGraphX(net_params.encoder.size, net_params.encoder.size * 2, net_params.num_edge_types)

  def _get_note_hidden_states(self, perf_sty, edges):
    zero_hidden = torch.zeros(perf_sty.shape[0], perf_sty.shape[1], perf_sty.shape[2]*2).to(perf_sty).device
    return self.performance_graph_encoder(perf_sty, zero_hidden, edges)


class IsgnPerfEncoderXBias(IsgnPerfEncoderX):
  def __init__(self, net_params):
    super(IsgnPerfEncoderXBias, self).__init__(net_params)
    self.performance_graph_encoder = GatedGraphXBias(net_params.encoder.size, net_params.encoder.size, net_params.num_edge_types)



class IsgnPerfEncoderMasking(IsgnPerfEncoder):
    def __init__(self, net_params):
        super(IsgnPerfEncoderMasking, self).__init__(net_params)

    def forward(self, score_embedding, y, edges, note_locations, return_z=False, num_samples=10):
        measure_numbers = note_locations['measure']
        # note_out, _, = score_embedding
        note_out = score_embedding['total_note_cat']

        expanded_y = self.performance_embedding_layer(y)
        perform_concat = torch.cat((note_out.repeat(y.shape[0], 1, 1), expanded_y), 2)

        if self.training():
          perform_concat = masking_half(perform_concat)

        perform_z, perform_mu, perform_var = self.get_perform_style_from_input(perform_concat, edges, measure_numbers)
        if return_z:
            return sample_multiple_z(perform_mu, perform_var, num_samples)
        return perform_z, perform_mu, perform_var


class GcnPerfEncoderMasking(IsgnPerfEncoder):
    def __init__(self, net_params):
        super().__init__(net_params)
        self.performance_graph_encoder = GraphConvStack(net_params.encoder.size, 
                                                        net_params.encoder.size, 
                                                        net_params.num_edge_types, 
                                                        num_layers=net_params.encoder.layer,
                                                        drop_out=net_params.drop_out)

    
def sample_multiple_z(perform_mu, perform_var, num=10):
    assert perform_mu.dim() == 2
    total_perform_z = []
    for i in range(num):
      temp_z = reparameterize(perform_mu, perform_var)
      total_perform_z.append(temp_z)
    return torch.stack(total_perform_z, dim=1)
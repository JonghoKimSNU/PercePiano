import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from MidiBERT.virtuosonet_model_utils import make_higher_node, run_hierarchy_lstm_with_pack, span_beat_to_note_num

class ContextAttention(nn.Module):
    def __init__(self, size, num_head):
        super(ContextAttention, self).__init__()
        self.attention_net = nn.Linear(size, size)
        self.num_head = num_head

        if size % num_head != 0:
            raise ValueError("size must be dividable by num_head", size, num_head)
        self.head_size = int(size/num_head)
        self.context_vector = torch.nn.Parameter(torch.Tensor(num_head, self.head_size, 1))
        nn.init.uniform_(self.context_vector, a=-1, b=1)

    def get_attention(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        # attention_split = torch.cat(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
        similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
        similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
        return similarity

    def forward(self, x):
        attention = self.attention_net(x)
        attention_tanh = torch.tanh(attention)
        if self.head_size != 1:
            attention_split = torch.stack(attention_tanh.split(split_size=self.head_size, dim=2), dim=0)
            similarity = torch.bmm(attention_split.view(self.num_head, -1, self.head_size), self.context_vector)
            similarity = similarity.view(self.num_head, x.shape[0], -1).permute(1,2,0)
            similarity[x.sum(-1)==0] = -1e10 # mask out zero padded_ones
            softmax_weight = torch.softmax(similarity, dim=1)

            x_split = torch.stack(x.split(split_size=self.head_size, dim=2), dim=2)
            weighted_x = x_split * softmax_weight.unsqueeze(-1).repeat(1,1,1, x_split.shape[-1])
            attention = weighted_x.view(x_split.shape[0], x_split.shape[1], x.shape[-1])
            
            # weighted_mul = torch.bmm(softmax_weight.transpose(1,2), x_split)
            # restore_size = int(weighted_mul.size(0) / self.num_head)
            # attention = torch.cat(weighted_mul.split(split_size=restore_size, dim=0), dim=2)
        else:
            softmax_weight = torch.softmax(attention, dim=1)
            attention = softmax_weight * x

        sum_attention = torch.sum(attention, dim=1)
        return sum_attention

class HanEncoderNoteLSTM(nn.Module):
    def __init__(self, 
                 note_size, note_layer,
                  voice_size, voice_layer,
                   beat_size, beat_layer,
                    measure_size, measure_layer,
                     num_attention_head, drop_out=0.2):
        super(HanEncoderNoteLSTM, self).__init__()
        self.lstm = nn.LSTM(note_size, note_size, note_layer, batch_first=True, bidirectional=True, dropout=drop_out)

        self.voice_net = nn.LSTM(note_size, voice_size, voice_layer,
                                    batch_first=True, bidirectional=True, dropout=drop_out)
        self.beat_attention = ContextAttention((note_size + voice_size) * 2,
                                                num_attention_head)
        self.beat_rnn = nn.LSTM((note_size + voice_size) * 2, beat_size, beat_layer, batch_first=True, bidirectional=True, dropout=drop_out)
        self.measure_attention = ContextAttention(beat_size*2, num_attention_head)
        self.measure_rnn = nn.LSTM(beat_size * 2, measure_size, measure_layer, batch_first=True, bidirectional=True)

    def _cat_highway(self, lstm_out, embed_out):
      return lstm_out

    def forward(self, x, note_locations, lengths):
        voice_numbers = note_locations['voice']

        if not isinstance(x, nn.utils.rnn.PackedSequence):
          x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        max_voice = torch.max(voice_numbers)
        note_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        note_out, _ = pad_packed_sequence(note_out, True)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out = torch.cat((note_out,voice_out), 2)

        beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self.run_beat_and_measure(hidden_out, note_locations)
        hidden_out = self._cat_highway(hidden_out, x)
        # return hidden_out
        return {'note': note_out, 'voice': hidden_out, 'beat': beat_hidden_out, 'measure': measure_hidden_out, 'beat_spanned':beat_out_spanned, 'measure_spanned':measure_out_spanned,
                'total_note_cat': torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1) }

    def run_voice_net(self, batch_x, voice_numbers, max_voice):
        if isinstance(batch_x, torch.nn.utils.rnn.PackedSequence):
          batch_x, _ = nn.utils.rnn.pad_packed_sequence(batch_x, True)
        num_notes = batch_x.size(1)
        output = torch.zeros(batch_x.shape[0], batch_x.shape[1], self.voice_net.hidden_size * 2).to(batch_x.device)
        # voice_numbers = torch.Tensor(voice_numbers)
        for i in range(1,max_voice+1):
          voice_x_bool = voice_numbers == i
          num_voice_notes = torch.sum(voice_x_bool)
          num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)

          if num_voice_notes > 0:
            voice_notes = [batch_x[i, voice_x_bool[i]] if torch.sum(voice_x_bool[i])>0 else torch.zeros(1,batch_x.shape[-1]).to(batch_x.device) for i in range(len(batch_x)) ]
            # IndexError: The shape of the mask [120] at index 0 does not match the shape of the indexed tensor [105, 256] at index 0
            voice_x = pad_sequence(voice_notes, True)
            pack_voice_x = pack_padded_sequence(voice_x, [len(x) for x in voice_notes], True, False)
            ith_voice_out, _ = self.voice_net(pack_voice_x)
            ith_voice_out, _ = pad_packed_sequence(ith_voice_out, True)
            
            span_mat = torch.zeros(batch_x.shape[0], num_notes, voice_x.shape[1]).to(batch_x.device)
            voice_where = torch.nonzero(voice_x_bool)
            span_mat[voice_where[:,0], voice_where[:,1], torch.cat([torch.arange(num_batch_voice_notes[i]) for i in range(len(batch_x))])] = 1

            output += torch.bmm(span_mat, ith_voice_out)
        return output

    def run_beat_and_measure(self, hidden_out, note_locations):
        beat_numbers = note_locations['beat']
        measure_numbers = note_locations['measure']
        beat_nodes = make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, lower_is_note=True)
        # beat_hidden_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_rnn)

        beat_hidden_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_rnn, note_numbers=beat_numbers)
        measure_nodes = make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers)
        # measure_hidden_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_rnn)
        measure_hidden_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_rnn, note_numbers=measure_numbers)
        beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers)
        measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers)

        return beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned

    def get_attention_weights(self, x, edges, note_locations):
        voice_numbers = note_locations['voice']
        x = self.note_fc(x)
        max_voice = max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out = torch.cat((hidden_out,voice_out), 2)
        beat_hidden_out, _, _, _ = self.run_beat_and_measure(hidden_out, note_locations)

        weights = self.beat_attention.get_attention(hidden_out).squeeze()
        weights_mean = torch.mean(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])
        weights_std = torch.std(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])

        beat_weights = self.measure_attention.get_attention(beat_hidden_out).squeeze()
        beat_weights_mean = torch.mean(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])
        beat_weights_std = torch.std(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])

        norm_weights =  (weights-weights_mean)/weights_std
        norm_beat_weights = (beat_weights-beat_weights_mean)/beat_weights_std
        return {'note':norm_weights.permute(1,0).cpu().numpy(), 'beat':norm_beat_weights.permute(1,0).cpu().numpy()}


class HanEncoder(nn.Module):
    def __init__(self, 
                 note_size, note_layer,
                  voice_size, voice_layer,
                   beat_size, beat_layer,
                    measure_size, measure_layer,
                     num_attention_head, drop_out=0.2):
        super(HanEncoder, self).__init__()
        self.lstm = nn.LSTM(note_size, note_size, note_layer, batch_first=True, bidirectional=True, dropout=drop_out)

        # self.voice_net = nn.LSTM(note_size, voice_size, voice_layer,
        #                             batch_first=True, bidirectional=True, dropout=drop_out)
        # self.beat_attention = ContextAttention((note_size + voice_size) * 2,
        #                                         num_attention_head)
        # self.beat_rnn = nn.LSTM((note_size + voice_size) * 2, beat_size, beat_layer, batch_first=True, bidirectional=True, dropout=drop_out)
        # self.measure_attention = ContextAttention(beat_size*2, num_attention_head)
        # self.measure_rnn = nn.LSTM(beat_size * 2, measure_size, measure_layer, batch_first=True, bidirectional=True)

    def _cat_highway(self, lstm_out, embed_out):
      return lstm_out

    def forward(self, x, note_locations, lengths):
        voice_numbers = note_locations['voice']
        # x = self.note_fc(x)

        if not isinstance(x, nn.utils.rnn.PackedSequence):
          x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        max_voice = torch.max(voice_numbers)
        voice_out = self.run_voice_net(x, voice_numbers, max_voice)
        hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden_out, _ = pad_packed_sequence(hidden_out, True)
        hidden_out = torch.cat((hidden_out,voice_out), 2)

        # beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned = self.run_beat_and_measure(hidden_out, note_locations)
        hidden_out = self._cat_highway(hidden_out, x)
        return hidden_out
        # return torch.cat([hidden_out, beat_out_spanned, measure_out_spanned], dim=-1)

    # def run_voice_net(self, batch_x, voice_numbers, max_voice):
    #     if isinstance(batch_x, torch.nn.utils.rnn.PackedSequence):
    #       batch_x, _ = nn.utils.rnn.pad_packed_sequence(batch_x, True)
    #     num_notes = batch_x.size(1)
    #     output = torch.zeros(batch_x.shape[0], batch_x.shape[1], self.voice_net.hidden_size * 2).to(batch_x.device)
    #     # voice_numbers = torch.Tensor(voice_numbers)
    #     for i in range(1,max_voice+1):
    #       voice_x_bool = voice_numbers == i
    #       num_voice_notes = torch.sum(voice_x_bool)
    #       num_batch_voice_notes = torch.sum(voice_x_bool, dim=1)

    #       if num_voice_notes > 0:
    #         voice_notes = [batch_x[i, voice_x_bool[i]] if torch.sum(voice_x_bool[i])>0 else torch.zeros(1,batch_x.shape[-1]).to(batch_x.device) for i in range(len(batch_x)) ]
    #         # IndexError: The shape of the mask [120] at index 0 does not match the shape of the indexed tensor [105, 256] at index 0
    #         voice_x = pad_sequence(voice_notes, True)
    #         pack_voice_x = pack_padded_sequence(voice_x, [len(x) for x in voice_notes], True, False)
    #         ith_voice_out, _ = self.voice_net(pack_voice_x)
    #         ith_voice_out, _ = pad_packed_sequence(ith_voice_out, True)
            
    #         span_mat = torch.zeros(batch_x.shape[0], num_notes, voice_x.shape[1]).to(batch_x.device)
    #         voice_where = torch.nonzero(voice_x_bool)
    #         span_mat[voice_where[:,0], voice_where[:,1], torch.cat([torch.arange(num_batch_voice_notes[i]) for i in range(len(batch_x))])] = 1

    #         output += torch.bmm(span_mat, ith_voice_out)
    #     return output

    # def run_beat_and_measure(self, hidden_out, note_locations):
    #     beat_numbers = note_locations['beat']
    #     measure_numbers = note_locations['measure']
    #     beat_nodes = make_higher_node(hidden_out, self.beat_attention, beat_numbers, beat_numbers, lower_is_note=True)
    #     # beat_hidden_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_rnn)

    #     beat_hidden_out = run_hierarchy_lstm_with_pack(beat_nodes, self.beat_rnn, note_numbers=beat_numbers)
    #     measure_nodes = make_higher_node(beat_hidden_out, self.measure_attention, beat_numbers, measure_numbers)
    #     # measure_hidden_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_rnn)
    #     measure_hidden_out = run_hierarchy_lstm_with_pack(measure_nodes, self.measure_rnn, note_numbers=measure_numbers)
    #     beat_out_spanned = span_beat_to_note_num(beat_hidden_out, beat_numbers)
    #     measure_out_spanned = span_beat_to_note_num(measure_hidden_out, measure_numbers)

    #     return beat_hidden_out, measure_hidden_out, beat_out_spanned, measure_out_spanned

    # def get_attention_weights(self, x, edges, note_locations):
    #     voice_numbers = note_locations['voice']
    #     x = self.note_fc(x)
    #     max_voice = max(voice_numbers)
    #     voice_out = self.run_voice_net(x, voice_numbers, max_voice)
    #     hidden_out,_ = self.lstm(x)  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
    #     hidden_out = torch.cat((hidden_out,voice_out), 2)
    #     beat_hidden_out, _, _, _ = self.run_beat_and_measure(hidden_out, note_locations)

    #     weights = self.beat_attention.get_attention(hidden_out).squeeze()
    #     weights_mean = torch.mean(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])
    #     weights_std = torch.std(weights, axis=1).unsqueeze(1).repeat(1,weights.shape[1])

    #     beat_weights = self.measure_attention.get_attention(beat_hidden_out).squeeze()
    #     beat_weights_mean = torch.mean(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])
    #     beat_weights_std = torch.std(beat_weights, axis=1).unsqueeze(1).repeat(1,beat_weights.shape[1])

    #     norm_weights =  (weights-weights_mean)/weights_std
    #     norm_beat_weights = (beat_weights-beat_weights_mean)/beat_weights_std
    #     return {'note':norm_weights.permute(1,0).cpu().numpy(), 'beat':norm_beat_weights.permute(1,0).cpu().numpy()}

class VirtuosoNet(nn.Module):
    def __init__(self, input_size, bert_hidden_size = 768, embedding_size=128, net_params = None):
        """
        get input from bert, concat with virtuoso embedding, then hierarchical attention layer
        """
        super(VirtuosoNet, self).__init__() 
        # virtuosonet_embedder
        self.net_params = net_params
        self.linear = nn.Linear(input_size, embedding_size)
        self.virtuoso_projector = nn.Linear(embedding_size + bert_hidden_size, net_params['note_size'])
        self.score_encoder = HanEncoderNoteLSTM(net_params['note_size'], net_params['note_layer'],
                                        net_params['voice_size'], net_params['voice_layer'],
                                        net_params['beat_size'], net_params['beat_layer'],
                                        net_params['measure_size'], net_params['measure_layer'],
                                        net_params['num_attention_head'],
                                        net_params['drop_out'])
        
    def forward(self, bert_output, virtuoso_input, note_locations=None, lengths = None, found_addon_idxs = None):
        # concat two, then project
        virtuoso_input = self.linear(virtuoso_input)
        virtuoso_input = torch.sigmoid(virtuoso_input)
        # gelu activation
        # virtuoso_input = torch.nn.functional.gelu(virtuoso_input)
        if found_addon_idxs is not None:
            """concat bert output with virtuoso input.  if the value is -1, fill with zero.
            e.g. for each element in batch,
            found_addon_idx = [0, 2, -1, 1]
              [cat(bert_output[0], virtuoso_input[0]), cat(bert_output[2], virtuoso_input[1]), 
                 cat(zeros_like(bert_output[0]), virtuoso_input[2]), cat(bert_output[1], virtuoso_input[3])]
             """
            x = torch.cat((torch.zeros((bert_output.shape[0], virtuoso_input.shape[1], bert_output.shape[2])).to(bert_output.device), virtuoso_input), dim=-1)
            for batch_idx, _ in enumerate(found_addon_idxs):
                found_addon_idx = found_addon_idxs[batch_idx]
                # fill x with bert_output according to found_addon_idx
                for i, idx in enumerate(found_addon_idx):
                    if idx != -1:
                        x[batch_idx, i] = torch.cat((bert_output[batch_idx, idx], virtuoso_input[batch_idx, i]), dim=-1)
        else:
            x = torch.cat((bert_output, virtuoso_input), dim=-1)
        x_embedded = self.virtuoso_projector(x)

        performance_embedding = self.score_encoder(x_embedded, note_locations, lengths) # keys: dict_keys(['note', 'beat', 'measure', 'beat_spanned', 'measure_spanned', 'total_note_cat'])

        return performance_embedding
    
class AlignLSTM(nn.Module):
    # lstm only
    def __init__(self, input_size, bert_hidden_size = 768, embedding_size=32, note_size = 256):
        """
        get input from bert, concat with virtuoso embedding, then hierarchical attention layer
        """
        super(AlignLSTM, self).__init__() 
        self.linear = nn.Linear(input_size, embedding_size)
        self.virtuoso_projector = nn.Linear(embedding_size + bert_hidden_size, note_size)
        self.score_encoder = nn.LSTM(note_size, note_size, 1, batch_first=True, bidirectional=True, dropout=0.2)
        
    def forward(self, bert_output, virtuoso_input, lengths = None):
        # concat two, then project
        virtuoso_input = self.linear(virtuoso_input)
        virtuoso_input = torch.sigmoid(virtuoso_input)
        x = torch.cat((bert_output, virtuoso_input), dim=-1)
        x_embedded = self.virtuoso_projector(x)

        if not isinstance(x_embedded, nn.utils.rnn.PackedSequence):
            x_embedded = nn.utils.rnn.pack_padded_sequence(x_embedded, lengths, batch_first=True, enforce_sorted=False)

        performance_embedding, _ = self.score_encoder(x_embedded)
        performance_embedding, _ = pad_packed_sequence(performance_embedding, True)

        return performance_embedding
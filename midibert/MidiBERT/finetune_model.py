import math
import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from MidiBERT.model import MidiBert


class TokenClassification(nn.Module):
    def __init__(self, midibert, class_num, hs):
        super().__init__()
        
        self.midibert = midibert
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hs, 256),
            nn.ReLU(),
            nn.Linear(256, class_num)
        )
    
    def forward(self, y, attn, layer):
        # feed to bert 
        y = self.midibert(y, attn, output_hidden_states=True)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        y = y.hidden_states[layer]
        return self.classifier(y)


class SequenceClassification(nn.Module):
    def __init__(self, midibert, class_num, hs, da=128, r=4, head_type = 'attentionhead'):
        super(SequenceClassification, self).__init__()
        self.midibert = midibert
        self.head_type = head_type
        if self.head_type == 'attentionhead':
            self.attention = SelfAttention(hs, da, r)
            self.classifier = nn.Sequential(
                nn.Linear(hs*r, 256),
                nn.ReLU(),
                nn.Linear(256, class_num)
            )        
        elif self.head_type == 'linearhead':
            # self.attention = SelfAttention(hs, da, r)
            self.classifier = nn.Sequential(
                nn.Linear(hs, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, class_num)
            )

    def forward(self, x, attn, layer):             # x: (batch, 512, 4)
        x = self.midibert(x, attn, output_hidden_states=True)   # (batch, 512, 768)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        if self.head_type == 'attentionhead':
            attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
            m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
            flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
            res = self.classifier(flatten)      # res: (batch, class_num)            
        # average
        elif self.head_type == 'linearhead':
            flatten = torch.mean(x, dim=1)
            res = self.classifier(flatten)  # res: (batch, class_num)
        return res

class SequenceClassificationAlign(nn.Module):
    def __init__(self, midibert, virtuosonet, class_num, hs, da=128, r=4, head_type = 'attentionhead', output_type = 'note'):
        super(SequenceClassificationAlign, self).__init__()
        self.midibert = midibert
        self.head_type = head_type
        self.virtuosonet = virtuosonet
        self.projector = nn.Linear(512, hs)
        if self.head_type == 'attentionhead':
            self.attention = SelfAttention(hs, da, r)
            self.classifier = nn.Sequential(
                nn.Linear(hs*r, 256),
                nn.ReLU(),
                nn.Linear(256, class_num)
            )        
        elif self.head_type == 'linearhead':
            # self.attention = SelfAttention(hs, da, r)
            self.classifier = nn.Sequential(
                nn.Linear(hs, 256),
                nn.Dropout(0.2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, class_num)
            )

    def forward(self, x, attn, layer, addons = None, data_len = None):             # x: (batch, 512, 4)
        x = self.midibert(x, attn, output_hidden_states=True)   # (batch, 512, 768)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        x = x.hidden_states[layer]
        x = self.virtuosonet(x, addons, data_len)
        x = self.projector(x)
        if self.head_type == 'attentionhead':
            attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
            m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
            flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
            res = self.classifier(flatten)      # res: (batch, class_num)            
        # average
        elif self.head_type == 'linearhead':
            flatten = torch.mean(x, dim=1)
            res = self.classifier(flatten)  # res: (batch, class_num)
        return res

class SequenceClassificationHierarchical(nn.Module):
    def __init__(self, midibert, virtuosonet, class_num, hs, da=128, r=4, head_type = 'attentionhead', output_type = 'note'):
        super(SequenceClassificationHierarchical, self).__init__()
        self.midibert = midibert
        self.virtuosonet = virtuosonet
        self.head_type = head_type
        self.output_type = output_type

        if self.output_type == 'note' or self.output_type == 'beat' or self.output_type == 'measure':
            self.projector = nn.Linear(virtuosonet.net_params['note_size'] * 2, hs)
        elif self.output_type == 'voice':
            self.projector = nn.Linear(virtuosonet.net_params['note_size'] * 4 , hs)
        elif self.output_type == 'total_note_cat':
            self.projector = nn.Linear(virtuosonet.net_params['note_size'] * 8, hs)
        if self.head_type == 'attentionhead':
            self.attention = SelfAttention(hs, da, r)
            self.classifier = nn.Sequential(
                nn.Linear(hs*r, 256),
                nn.ReLU(),
                nn.Linear(256, class_num)
            )
        else:
            self.classifier = nn.Sequential(
                nn.Linear(hs, 256),
                nn.GELU(),
                nn.Linear(256, class_num)
            )
    def forward(self, x, attn, layer, addons=None, note_location = None, data_len = None, found_addon_idxs = None):             # x: (batch, 512, 4)
        midibert_out = self.midibert(x, attn, output_hidden_states=True)   # (batch, 512, 768)
        #y = y.last_hidden_state         # (batch_size, seq_len, 768)
        midibert_out = midibert_out.hidden_states[layer]
        x = self.virtuosonet(midibert_out, addons, note_location, data_len, found_addon_idxs)
        x = self.projector(x[self.output_type])
        # print(x.shape)
        if self.head_type == 'attentionhead':
            attn_mat = self.attention(x)        # attn_mat: (batch, r, 512)
            m = torch.bmm(attn_mat, x)          # m: (batch, r, 768)
            flatten = m.view(m.size()[0], -1)   # flatten: (batch, r*768)
            res = self.classifier(flatten)      # res: (batch, class_num)
        elif self.head_type == 'linearhead':
            # average
            # residual connection
            x = x + midibert_out[:, :x.shape[1], :]
            flatten = torch.mean(x, dim=1)
            res = self.classifier(flatten)      # res: (batch, class_num)
        return res


class SelfAttention(nn.Module):
    def __init__(self, input_dim, da, r):
        '''
        Args:
            input_dim (int): batch, seq, input_dim
            da (int): number of features in hidden layer from self-attn
            r (int): number of aspects of self-attn
        '''
        super(SelfAttention, self).__init__()
        self.ws1 = nn.Linear(input_dim, da, bias=False)
        self.ws2 = nn.Linear(da, r, bias=False)

    def forward(self, h):
        attn_mat = F.softmax(self.ws2(torch.tanh(self.ws1(h))), dim=1)
        attn_mat = attn_mat.permute(0,2,1)
        return attn_mat


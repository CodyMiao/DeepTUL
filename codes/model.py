# coding: utf-8
from __future__ import print_function
from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import utils as nn_utils
import time
from collections import Counter
import numpy as np

class Attn(nn.Module):
    def __init__(self, method, hidden_size,loc_size,use_cuda):
        super(Attn, self).__init__()

        self.method = method 
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.loc_size= loc_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, self.hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, self.hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(self.hidden_size))

    def forward(self, out_state, history):
        seq_len = history.size()[0]
        state_len = out_state.size()[0]

        attn_energies = Variable(torch.zeros(state_len, seq_len)).cuda() if self.use_cuda else Variable(torch.zeros(state_len, seq_len))
        for i in range(state_len):
            for j in range(seq_len):
                if self.strategies_type == 'AVE-sdot' or self.strategies_type== 'MAX-sdot':
                    attn_energies[i, j] = self.score(out_state[i], history[j])/sqrt(self.loc_size)
                else:
                    attn_energies[i, j] = self.score(out_state[i], history[j])
        
        return F.softmax(attn_energies, dim = 1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output) 
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output) 
            energy = hidden.dot(energy)
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output)))
            energy = self.other.dot(energy)
            return energy


class AttnRnnModel(nn.Module):
    """rnn model with long-term history attention"""

    def __init__(self, parameters):
        super(AttnRnnModel, self).__init__()
        self.loc_size = parameters.loc_size
        self.loc_emb_size = parameters.loc_emb_size
        self.tim_size = parameters.tim_size
        self.tim_emb_size = parameters.tim_emb_size
        self.uid_size = parameters.uid_size
        self.uid_emb_size = parameters.uid_emb_size
        self.hidden_size = parameters.hidden_size
        self.attn_type = parameters.attn_type
        self.rnn_type = parameters.rnn_type 
        self.use_cuda = parameters.use_cuda
        self.strategies_type = parameters.strategies_type

        self.emb_loc = nn.Embedding(self.loc_size, self.loc_emb_size)
        self.emb_tim = nn.Embedding(self.tim_size, self.tim_emb_size)
        self.emb_uid = nn.Embedding(self.uid_size, self.uid_emb_size)

        input_attn_size = self.loc_emb_size + self.tim_emb_size + self.uid_emb_size

        if self.rnn_type == 'BILSTM':
            self.fc_attn = nn.Linear(input_attn_size, 2*self.hidden_size)
            self.attn = Attn(self.attn_type, 2*self.hidden_size, 2*self.loc_size,self.use_cuda) 
        else:
            self.fc_attn = nn.Linear(input_attn_size, self.hidden_size)
            self.attn = Attn(self.attn_type, self.hidden_size, 2*self.loc_size,self.use_cuda) 
        input_size = self.loc_emb_size + self.tim_emb_size

        if self.rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, self.hidden_size, 1, batch_first = True)
        elif self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1, batch_first = True)
        elif self.rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, self.hidden_size, 1, batch_first = True)
        elif self.rnn_type == 'BILSTM':
            self.rnn = nn.LSTM(input_size, self.hidden_size, 1, batch_first = True,bidirectional=True)


        if self.rnn_type == 'BILSTM':
            self.fc_final = nn.Linear(6* self.hidden_size, self.uid_size)
        else:
            self.fc_final = nn.Linear(3* self.hidden_size, self.uid_size)
        self.dropout = nn.Dropout(p=parameters.dropout_p)
        self.init_weights()

    def init_weights(self):
        """
        Here we reproduce Keras default initialization weights for consistency with Keras version
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)

        for t in ih:
            nn.init.xavier_uniform(t)
        for t in hh:
            nn.init.orthogonal(t)
        for t in b:
            nn.init.constant(t, 0)

    def forward(self, loc, tim, input_lengths, history_loc, history_tim, history_uid, history_count):
        batch_size = loc.size(0)
        max_length = loc.size(1)
        loc = loc.transpose(0, 1)
        tim = tim.transpose(0, 1)
        if self.rnn_type == 'BILSTM':
            h1 = Variable(torch.zeros(1 * 2, batch_size, self.hidden_size))
            c1 = Variable(torch.zeros(1 * 2, batch_size, self.hidden_size))
            score = Variable(torch.zeros(batch_size, max_length, self.uid_size))
            temp_out_state = Variable(torch.zeros(max_length, batch_size, self.hidden_size * 2))
        else:
            h1 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            c1 = Variable(torch.zeros(1, batch_size, self.hidden_size))
            score = Variable(torch.zeros(batch_size, max_length, self.uid_size))
            temp_out_state = Variable(torch.zeros(max_length, batch_size, self.hidden_size))
        
        if self.use_cuda:
            h1 = h1.cuda()
            c1 = c1.cuda()
            score =  score.cuda()
            temp_out_state = temp_out_state.cuda()
        for t in range(max_length):

            loc_emb = self.emb_loc(loc[t].unsqueeze(1))
            tim_emb = self.emb_tim(tim[t].unsqueeze(1))

    
            x = torch.cat((loc_emb, tim_emb), 2)
            x = self.dropout(x)

            if self.rnn_type == 'GRU' or self.rnn_type == 'RNN':
                out_state, h1 = self.rnn(x, h1)
            elif self.rnn_type == 'LSTM' or self.rnn_type == 'BILSTM':
                out_state, (h1, c1) = self.rnn(x, (h1, c1))
            temp_out_state[t] =  out_state.squeeze(1)
        temp_out_state = temp_out_state.transpose(0, 1)
        
        for t in range(batch_size):
            loc_emb_history = self.emb_loc(history_loc[t]).squeeze(1)
            tim_emb_history = self.emb_tim(history_tim[t]).squeeze(1)
            uid_emb_history = self.emb_uid(history_uid[t]).squeeze(1)
            count = 0
            loc_emb_history2 = Variable(torch.zeros(len(history_count[t]), loc_emb_history.size()[-1])) 
            tim_emb_history2 = Variable(torch.zeros(len(history_count[t]), tim_emb_history.size()[-1]))
            uid_emb_history2 = Variable(torch.zeros(len(history_count[t]), uid_emb_history.size()[-1]))
            
            if self.use_cuda:
                loc_emb_history2 = loc_emb_history2.cuda()
                tim_emb_history2 = tim_emb_history2.cuda()
                uid_emb_history2 = uid_emb_history2.cuda()

            for i, c in enumerate(history_count[t]):
                if c == 1:
                    tmp = uid_emb_history[count].unsqueeze(0)
                else:
                    if self.strategies_type == 'AVE-sdot' or self.strategies_type== 'AVE-dot':
                        data = uid_emb_history[count:count + c, :].detach().numpy()
                        uniques = np.unique(data,axis=0)
                        uid_max = Variable(torch.FloatTensor(uniques))
                        tmp = torch.mean(uid_emb_history[count:count + c, :], dim=0, keepdim=True)
                    else:
                        uid_number = Counter(history_uid)
                        uid_max =sorted(uid_number.items(), key=lambda x: x[1], reverse=True)[0][0]
                        uid_max = Variable(torch.cuda.LongTensor(uid_max))
                        tmp = self.emb_uid(uid_max).squeeze(1)
                uid_emb_history2[i, :] = tmp  
                loc_emb_history2[i, :] = loc_emb_history[count, :].unsqueeze(0)
                tim_emb_history2[i, :] = tim_emb_history[count, :].unsqueeze(0)
                count += c

            history = torch.cat((loc_emb_history2, tim_emb_history2, uid_emb_history2), 1)
            history = self.dropout(history)
            history = F.tanh(self.fc_attn(history))
            out_state = temp_out_state[t][input_lengths[t]-1].unsqueeze(0)
            attn_weights = self.attn(out_state, history).unsqueeze(0) 
            context = attn_weights.bmm(history.unsqueeze(0)).squeeze(0)
            out = torch.cat((out_state, context, out_state.squeeze(1)), 1)
            out = self.dropout(out)
            y = self.fc_final(out) 
            score[t] = y
        return score

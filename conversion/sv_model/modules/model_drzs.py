import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class MultiLSTM(nn.Module):
    '''
    This network is used to compute each row of the affinity matrix for speaker diarization.
    e.g:
        input: 
        111111111111111111
        223414411114442211
        output:
        000010011110000011
    '''
    def __init__(self, input_dim, hidden_dim, depth, dropout, fc_dim):
        super(MultiLSTM, self).__init__()
        if(depth >= 2):
            self.lstm = nn.LSTM(input_dim, hidden_dim, depth, batch_first=True, bidirectional=True, dropout=dropout)
        else:
            self.lstm = nn.LSTM(input_dim, hidden_dim, depth, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(hidden_dim * 2, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def forward(self, x):
        self.lstm.flatten_parameters()
        x = self.lstm(x)[0]
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.squeeze(2)
        return x

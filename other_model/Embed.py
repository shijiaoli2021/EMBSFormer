import torch.nn as nn
import torch
import math


'''
for sequence about input's shape: (batch, seq_len, d_model) or (batch, N, seq_len, d_model)
'''


'''
Initial data embedding
'''
class DataTokenEmbedding(nn.Module):
    def __init__(self, d_model, embeded_dim):
        super(DataTokenEmbedding, self).__init__()
        self.dataTokenEmbeding = nn.Linear(d_model, embeded_dim)

    def forward(self, x):
        out = self.dataTokenEmbeding(x)
        return out

'''
position_encoding just use Trigonometric function here
'''
class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len=30, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i/d_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** (i/d_model)))
        self.register_buffer("pe",pe)


    def forward(self, x):
        inputShape = x.shape
        seq_len = inputShape(-2)
        pe = 0
        if(len(inputShape) == 3):
            pe = self.pe[:,:seq_len].unsqueeze(0)
        elif(len(inputShape) == 4):
            pe = self.pe[:,:seq_len].unsqueeze(0)
            pe = pe.unsqueeze(1)
        return self.dropout(x + pe)


'''
Laplace graph matrix feature embedding
'''
class LaplacianPE(nn.Module):
    def __init__(self, laplacian_dim, embedded_dim):
        super(LaplacianPE, self).__init__()
        self.laplacianEmbedded = nn.Linear(laplacian_dim, embedded_dim)

    def forward(self, lap_matrix, dim_num):
        out = self.laplacianEmbedded(lap_matrix).unsqueeze(0)
        if dim_num == 4:
            out = out.unsqueeze(0)
        return out


'''
data embedding
'''
class DataEmbedding(nn.Module):
    def __init__(self, embeded_dim, feature_dim, max_seq_len, dropout= 0.1, add_time_in_day= False, add_day_in_week= False):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.add_time_in_day = add_time_in_day
        self.add_time_in_week = add_day_in_week
        self.feature_dim = feature_dim
        self.posEncoding = PositionalEncoder(embeded_dim, max_seq_len, dropout)
        self.dataTokenEmbedding = DataTokenEmbedding(feature_dim, embeded_dim)
        if add_time_in_day:
            self.minutes_size = 1440
            self.addInDayEmbedding = nn.Embedding(self.minutes_size, embeded_dim)
        if add_day_in_week:
            self.day_size = 7
            self.addInWeekEmbedding = nn.Embedding(self.day_size, embeded_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # dataToken
        cashOut = self.dataTokenEmbedding(x[:, :, :, :self.feature_dim])
        # in day
        if self.add_time_in_day:
            cashOut+= self.addInDayEmbedding(x[:, :, :, self.feature_dim].round().long())
        # in week
        if self.add_time_in_week:
            cashOut+= self.addInWeekEmbedding(x[:, :, :, self.feature_dim+1].round().long())

        return self.dropout(cashOut)



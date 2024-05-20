import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def attention(q, k, v, d_k, mask= None, dropout = None):

    # attention = softMax(q*k'/d_k) => mask() => * v
    # (batch, N, l, d_model)
    A = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k))
    if mask is not None:
        mask = mask.unsqueeze(1)
        A = A.masked_fill(mask == 0, -1e9)
    A = F.softmax(A, dim=-1)
    output = torch.matmul(A, v)
    if dropout is not None:
        output = dropout(output)
    return output

class FeedForward(nn.Module):
    def __init__(self, d_model, hidden_dim=None, d_ff=64, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 64
        if hidden_dim is None:
            hidden_dim = d_model
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, hidden_dim)


    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

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
data embedding
'''
class DataEmbedding(nn.Module):
    def __init__(self, embeded_dim, feature_dim, dropout= 0.1, add_time_in_day= False, add_day_in_week= False):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.add_time_in_day = add_time_in_day
        self.add_time_in_week = add_day_in_week
        self.feature_dim = feature_dim
        self.posEncoding = PositionalEncoder(embeded_dim, dropout=dropout)
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
            cashOut += self.addInDayEmbedding(x[:, :, :, self.feature_dim].round().long())
        # in week
        if self.add_time_in_week:
            cashOut += self.addInWeekEmbedding(x[:, :, :, self.feature_dim+1].round().long())

        return self.dropout(cashOut)

'''
take chebyshev ploynomials as kernal for graph conv
'''
class Cheb_conv(nn.Module):
    def __init__(self, chebyshev_ploynomials, K, in_channel, out_channel):
        super(Cheb_conv, self).__init__()
        self.K = K
        self.out_channels = out_channel
        self.chebyshev_ploynomials = chebyshev_ploynomials
        self.DEVICE = chebyshev_ploynomials[0].device
        self.thetaList = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channel, out_channel).to(self.DEVICE)) for _ in range(K)])
        self.relu = nn.ReLU()
        self.initParameter()

    def initParameter(self):
        for i in range(self.K):
            nn.init.xavier_uniform_(self.thetaList[i])

    def forward(self, x):
        '''

        :param x: (batch, seq_len, nodes_num, features(in_channel))
        :return:
        '''
        batch_size, seq_len, nodes_num, in_channel = x.shape
        output = torch.zeros(batch_size, seq_len, nodes_num, self.out_channels).to(self.DEVICE)
        for i in range(self.K):
            cheb = self.chebyshev_ploynomials[i]
            #(batch, seq_len, nodes_num, in_channel)
            cash = cheb.matmul(x)
            # (batch, seq_len, nodes_num, out_channel)
            cash = cash.matmul(self.thetaList[i])
            #(batch, nodes_num, out_channel, seq_len)
            output += cash

        return self.relu(output)

'''
g-lstm
'''
class Glstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, bs=1e-9):
        super(Glstm, self).__init__()
        self.bs = bs
        self.hidden_dim = hidden_dim
        self.U_x = nn.Parameter(torch.FloatTensor(input_dim, 4*hidden_dim))
        self.V_x = nn.Parameter(torch.FloatTensor(hidden_dim, 4*hidden_dim))
        self.b_x = nn.Parameter(torch.FloatTensor(4*hidden_dim))

        self.U_g = nn.Parameter(torch.FloatTensor(input_dim, 2*hidden_dim))
        self.V_g = nn.Parameter(torch.FloatTensor(hidden_dim, 2*hidden_dim))
        self.b_g = nn.Parameter(torch.FloatTensor(2*hidden_dim))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.initParameter()

    def initParameter(self):
        nn.init.xavier_uniform_(self.U_x)
        nn.init.xavier_uniform_(self.V_x)
        nn.init.xavier_uniform_(self.U_g)
        nn.init.xavier_uniform_(self.V_g)
        nn.init.constant_(self.b_x, self.bs)
        nn.init.constant_(self.b_g, self.bs)

    def forward(self, x, o_s, init_stat=None):
        # x(batch, seq_len, nodes_num, input_dim)
        # os(batch, seq_len, nodes_num, input_dim)
        # print("os:{}, ug:{}, vg:{}".format(o_s.shape, self.U_g.shape, self.V_g.shape))
        batch_size, seq_len, nodes_num, input_dim = x.shape
        device = x.device
        hidden_seq = []
        if init_stat is None:
            ht, ct = (torch.zeros(batch_size, nodes_num, self.hidden_dim).to(device),
             torch.zeros(batch_size, nodes_num, self.hidden_dim).to(device))
        else:
            ht, ct = init_stat

        for i in range(seq_len):
            xt = x[:, i, :, :]
            o_st = o_s[:, i, :, :]
            gates = xt @ self.U_x + ht @ self.V_x + self.b_x
            governor = o_st @ self.U_g + ht @ self.V_g + self.b_g
            ft, it, ot, ut = (self.sigmoid(gates[:, :, :self.hidden_dim]), self.sigmoid(gates[:, :, self.hidden_dim:2*self.hidden_dim]),
                              self.sigmoid(gates[:, :, 2*self.hidden_dim:3*self.hidden_dim]), self.sigmoid(gates[:, :, 3*self.hidden_dim:]))
            gov_f, gov_u = (self.sigmoid(governor[:, :, :self.hidden_dim]), self.sigmoid(governor[:, :, self.hidden_dim:]))
            ct = gov_f * ft * ct + gov_u * it*ut
            ht = ot * self.tanh(ct)
            hidden_seq.append(ht.unsqueeze(0))

        hidden_seq = torch.cat(hidden_seq, dim=0)
        hidden_seq = hidden_seq.transpose(0, 1).contiguous()
        # (batch, seq_len, nodes_num, input_dim)
        return hidden_seq, ht, ct



class Muti_glstm(nn.Module):
    def __init__(self, dim_blocks, output_dim):
        super(Muti_glstm, self).__init__()
        self.lstmModuleList = nn.ModuleList()
        for block in dim_blocks:
            self.lstmModuleList.append(Glstm(block["input_dim"], block["output_dim"]))
        self.feedforward = FeedForward(dim_blocks[-1]["output_dim"], hidden_dim=output_dim)


    def forward(self, x, o_s):
        hidden_seq = x
        for i in range(len(self.lstmModuleList)):
            hidden_seq, ht, ct = self.lstmModuleList[i](hidden_seq, o_s)
        return self.feedforward(hidden_seq)


class self_attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(self_attention, self).__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.muti_d_model = d_model // heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size, seq_len, nodes_num, d_model = x.shape
        Q = self.q_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model)
        K = self.k_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model)
        V = self.v_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model)
        spacial_attention = attention(Q, K, V, self.muti_d_model, dropout=self.dropout).view(batch_size, seq_len, -1,
                                                                                             d_model)
        self_attention = self.final_linear(spacial_attention)
        return self_attention


class GALSTM(nn.Module):
    def __init__(self,
                 seq_len,
                 pre_len,
                 feature_dim,
                 embedding_dim,
                 add_time_in_day,
                 add_day_in_week,
                 blocks,
                 hidden_dim,
                 num_of_time_filters,
                 heads,
                 dropout=0.0):
        super(GALSTM, self).__init__()
        self.seq_len = seq_len
        self.add_time_in_day = add_time_in_day
        self.add_day_in_week = add_day_in_week
        self.embedding_for_x = DataEmbedding(embedding_dim, 1, dropout, add_time_in_day, add_day_in_week)
        self.embedding_for_os = DataEmbedding(embedding_dim, feature_dim-1, dropout, add_time_in_day, add_day_in_week)
        self.gcn = Cheb_conv(blocks["cheb"]["cheb_ploynomials"],
                             blocks["cheb"]["K"], blocks["cheb"]["in_channel"],
                             blocks["cheb"]["out_channel"])
        self.muti_glstm = Muti_glstm(blocks["dim_blocks"], hidden_dim)
        self.time_conv = nn.Conv2d(hidden_dim, num_of_time_filters, kernel_size=(1, 3), stride=(1,1), padding=(0,1))
        self.residual_conv = nn.Conv2d(hidden_dim, num_of_time_filters, kernel_size=(1,1), stride=(1,1))
        self.decode_self_attention = nn.MultiheadAttention(num_of_time_filters, heads)
        self.feedforward = FeedForward(num_of_time_filters)
        self.final_conv = nn.Conv2d(seq_len, pre_len, kernel_size=(1, num_of_time_filters))

    def forward(self, x):
        #(batch_size, seq_len. nodes_num. channel)
        embedding_x = self.embedding_for_x(torch.cat((x[:, :, :, 0].unsqueeze(3), x[:, :, :, 3:]), dim=3)
                                           if self.add_time_in_day or self.add_day_in_week else x[:, :, :, 0].unsqueeze(3))
        gcn_x = self.gcn(embedding_x)
        batch_size, seq_len, nodes_num, embedding_dim = embedding_x.shape
        embedding_os = self.gcn(self.embedding_for_os(x[:, :, :, 1:]))
        hidden_seq = self.muti_glstm(gcn_x, embedding_os)
        time_conv = self.time_conv(hidden_seq.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        # print(embedding_x.transpose(2, 3).contiguous().view(batch_size*nodes_num, seq_len, -1).shape)
        query = time_conv[:, -1, :, :].unsqueeze(1).repeat(1, self.seq_len, 1, 1)
        decode_self_attention, _ = self.decode_self_attention(query.transpose(2, 3).contiguous().view(batch_size*nodes_num, seq_len, -1)
                                                           , embedding_x.transpose(2, 3).contiguous().view(batch_size*nodes_num, seq_len, -1)
                                                           , embedding_x.transpose(2, 3).contiguous().view(batch_size*nodes_num, seq_len, -1))
        out = self.final_conv(self.residual_conv(hidden_seq.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
                              + self.feedforward(decode_self_attention.contiguous().view(batch_size, nodes_num, seq_len, -1).transpose(1, 2)))[:, :, :, -1]
        return out



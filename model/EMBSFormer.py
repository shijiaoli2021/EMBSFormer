import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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
    def __init__(self,
                 embeded_dim,
                 feature_dim,
                 dropout= 0.1,
                 add_time_in_day=False,
                 add_day_in_week=False,
                 add_holiday=False):
        super().__init__()
        self.embeded_dim = embeded_dim
        self.add_time_in_day = add_time_in_day
        self.add_time_in_week = add_day_in_week
        self.add_holiday = add_holiday
        self.feature_dim = feature_dim
        self.posEncoding = PositionalEncoder(embeded_dim, dropout=dropout)
        self.dataTokenEmbedding = DataTokenEmbedding(feature_dim, embeded_dim)
        if add_time_in_day:
            self.minutes_size = 1440
            self.addInDayEmbedding = nn.Embedding(self.minutes_size, embeded_dim)
        if add_day_in_week:
            self.day_size = 7
            self.addInWeekEmbedding = nn.Embedding(self.day_size, embeded_dim)
        if add_holiday:
            self.holiday_size = 2
            self.addInHoliday = nn.Embedding(self.holiday_size, embeded_dim)


        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # dataToken
        cashOut = self.dataTokenEmbedding(x[:, :, :, :self.feature_dim])
        # in day
        if self.add_time_in_day:
            cashOut += self.addInDayEmbedding(x[:, :, :, self.feature_dim].round().long())
        # in week
        if self.add_time_in_week:
            cashOut += self.addInWeekEmbedding(x[:, :, :, self.feature_dim + 1].round().long())

        if self.add_holiday:
            cashOut += self.addInWeekEmbedding(x[:, :, :, self.feature_dim + 2].round().long())

        return self.dropout(cashOut)

'''layer_norm'''
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps= 1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, input):
        output = self.alpha * (input - input.mean(dim=-1, keepdim=True))\
        /(input.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return output


'''carculate the attention'''
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


'''Feedforward'''
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=64, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

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
        self.thetaList = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channel, out_channel)) for _ in range(K)])
        self.relu = nn.ReLU()
        self.initParameter()

    def initParameter(self):
        for i in range(self.K):
            nn.init.xavier_uniform_(self.thetaList[i])
    
    def cheb_ploynomials_to_device(self, device):
        for i in range(self.K):
            self.chebyshev_ploynomials[i] = self.chebyshev_ploynomials[i].to(device)

    def forward(self, x):
        '''

        :param x: (batch, seq_len, nodes_num, features(in_channel))
        :return:
        '''
        device = x.device
        if self.chebyshev_ploynomials[-1].device != device:
            self.cheb_ploynomials_to_device(device)
        batch_size, seq_len, nodes_num, in_channel = x.shape
        output = torch.zeros((batch_size, seq_len, nodes_num, self.out_channels), device=device)
        for i in range(self.K):
            # cheb = self.chebyshev_ploynomials[i]
            #(batch, seq_len, nodes_num, in_channel)
            cash = self.chebyshev_ploynomials[i].matmul(x)
            # (batch, seq_len, nodes_num, out_channel)
            output += cash.matmul(self.thetaList[i])
            #(batch, nodes_num, out_channel, seq_len)
            # output += cash

        return self.relu(output)


'''spacial self-attention'''
class Spacial_self_attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(Spacial_self_attention, self).__init__()
        assert d_model%heads==0
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.muti_d_model = d_model//heads
        self.qs_linear = nn.Linear(d_model, d_model)
        self.ks_linear = nn.Linear(d_model, d_model)
        self.vs_linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        #(batch_size, seq_len, nodes_num, d_model)
        batch_size, seq_len, nodes_num, d_model = x.shape
        Q = self.qs_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model).transpose(2, 3)
        K = self.ks_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model).transpose(2, 3)
        V = self.vs_linear(x).view(batch_size, seq_len, -1, self.heads, self.muti_d_model).transpose(2, 3)
        return self.final_linear(attention(Q, K, V, self.muti_d_model, dropout=self.dropout).transpose(2, 3).contiguous().view(batch_size, seq_len, -1, d_model))


'''Temporal self-attention'''
class Temporal_self_attention(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super(Temporal_self_attention, self).__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.muti_d_model = d_model // heads
        self.qt_linear = nn.Linear(d_model, d_model)
        self.kt_linear = nn.Linear(d_model, d_model)
        self.vt_linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        #(batch_size, seq_len, nodes_num, d_model)
        batch_size, seq_len, nodes_num, d_model = x.shape
        Q = self.qt_linear(x).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        K = self.kt_linear(x).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        V = self.vt_linear(x).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        return self.final_linear(attention(Q, K, V, self.muti_d_model, dropout=self.dropout).transpose(2, 3).contiguous().view(batch_size, nodes_num, -1, d_model).transpose(1, 2))



'''spcial_attention + temporal_attention + gcn + time_conv + (residual)'''
class STAG_block(nn.Module):
    def __init__(self, backbone, heads, seq_len, droupout=0.1, **kwargs):
        super(STAG_block, self).__init__()
        self.gcn = Cheb_conv(backbone["cheb_ploynomials"],
                             backbone["K"],
                             backbone["in_channel"],
                             backbone["out_channel"])
        self.spacial_self_attention = Spacial_self_attention(backbone["in_channel"], heads, droupout)
        self.temporal_self_attention = Temporal_self_attention(backbone["in_channel"], heads, droupout)
        self.feedforward_spacial = FeedForward(backbone["in_channel"])
        # self.deta_attention = Deta_self_attention(backbone["in_channel"], seq_len, heads, droupout)
        self.feedforward_temporal = FeedForward(backbone["in_channel"])
        # self.final_linear = nn.Linear(backbone["hidden_dim"], backbone["hidden_dim"])
        self.gcn_conv = nn.Conv2d(backbone["out_channel"], backbone["hidden_dim"], kernel_size=(1, 3),
                                  padding=(0, 1))
        self.residual_conv = nn.Conv2d(backbone["in_channel"], backbone["hidden_dim"], kernel_size=(1, 1))
        self.layernorm = LayerNorm(backbone["in_channel"])
        self.layernorm_out = LayerNorm(backbone["hidden_dim"])
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.feedforward_spacial(self.layernorm(self.spacial_self_attention(x)))
        out = self.feedforward_temporal(self.layernorm(self.temporal_self_attention(out)))
        gcn = self.gcn_conv(self.gcn(out).permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return self.relu(self.layernorm_out(self.residual_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1) + gcn))

class STAG_subblock(nn.Module):
    def __init__(self, backbone_list, heads, seq_len, droupout=0.1, **kwargs):
        super(STAG_subblock, self).__init__()
        self.subblock = nn.Sequential()
        for backbone in backbone_list:
            self.subblock.append(STAG_block(backbone, heads, seq_len, droupout))

    def forward(self, x):
        return self.subblock(x)


class Similarity_attention(nn.Module):
    def __init__(self, seq_len, pre_len, d_model, heads, dropout=0.1, **kwargs):
        super(Similarity_attention, self).__init__()
        assert d_model % heads == 0
        self.heads = heads
        self.dropout = nn.Dropout(dropout)
        self.muti_d_model = d_model // heads
        self.qt_linear = nn.Linear(d_model, d_model)
        self.kt_linear = nn.Linear(d_model, d_model)
        self.vt_linear = nn.Linear(d_model, d_model)
        self.final_linear = nn.Linear(d_model, d_model)
        self.atention_conv = nn.Conv2d(seq_len, pre_len, kernel_size=(1, 1))

    def forward(self, k_x, q_x, v_x):
        batch_size, seq_len, nodes_num, d_model = k_x.shape
        k = self.kt_linear(self.atention_conv(k_x)).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        q = self.qt_linear(self.atention_conv(q_x)).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        v = self.vt_linear(v_x).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model).transpose(2, 3)
        # qs_x = self.qt_linear(self.atention_conv(k_x)).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model)
        # vs_x = self.vt_linear(self.atention_conv(k_x)).transpose(1, 2).view(batch_size, nodes_num, -1, self.heads, self.muti_d_model)
        day_or_week_attention = attention(q, k, v, self.muti_d_model, dropout=self.dropout)\
            .transpose(2, 3).contiguous().view(batch_size, nodes_num, -1, d_model).transpose(1, 2)
        # sday_or_week_attention = attention(qs_x, k, vs_x, self.muti_d_model, dropout=self.dropout) \
        #     .view(batch_size, nodes_num, -1, d_model).transpose(1, 2)
        # day_or_week_attention = self.final_linear(day_or_week_attention + sday_or_week_attention)
        return self.final_linear(day_or_week_attention)

class Similarity_module(nn.Module):
    def __init__(self, seq_len, pre_len, d_model, heads, cycle_num, dropout=0.1, **kwargs):
        super(Similarity_module, self).__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.cycle_num = cycle_num
        self.similarity_attention = Similarity_attention(seq_len, pre_len, d_model, heads)
        self.layer_norm = LayerNorm(d_model)
        self.feedforward = FeedForward(d_model)
        self.time_conv = nn.Conv2d(pre_len, pre_len, kernel_size=(1, d_model))

    def forward(self, recent, cycle_term):
        batch_size, seq_len, nodes_num, embedding_dim = recent.shape
        out = self.feedforward(self.layer_norm(self.similarity_attention(recent, cycle_term[:, :seq_len, :, :],
                                             cycle_term[:, seq_len:seq_len + self.pre_len, :, :])))
        if (self.cycle_num>1):
            for i in range(self.cycle_num):
                cash_index = i*(seq_len + self.pre_len)
                out += self.feedforward(self.layer_norm(self.similarity_attention(recent, cycle_term[:, cash_index:cash_index+seq_len, :, :],
                                                 cycle_term[:, cash_index+seq_len : cash_index + seq_len + self.pre_len, :, :])))
        return self.time_conv(self.layer_norm((out)))



class ASTGFormer(nn.Module):
    def __init__(self,
                 seq_len,
                 pre_len,
                 nodes_num,
                 feature_dim,
                 embedded_dim,
                 backbone_list,
                 heads,
                 cycle_matrix_config,
                 add_time_in_day=False,
                 add_day_in_week=False,
                 add_holiday=False,
                 droupout=0.1):
        super(ASTGFormer, self).__init__()
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.data_embedding = DataEmbedding(embedded_dim, feature_dim, droupout, add_time_in_day, add_day_in_week, add_holiday)
        self.sub_blocks = STAG_subblock(backbone_list, heads, seq_len, droupout)
        self.final_conv = nn.Conv2d(seq_len, pre_len, kernel_size=(1, backbone_list[-1]["hidden_dim"]))
        self.cycle_matrix_config = cycle_matrix_config
        # similarity module list
        self.cycle_module_list = nn.ModuleList([Similarity_module(seq_len, pre_len, embedded_dim, heads, int(cycle_config['num']), droupout) for cycle_config in cycle_matrix_config])
        # W for simirity
        self.W_list_similarity = nn.ParameterList([nn.Parameter(torch.FloatTensor(pre_len, nodes_num)) for _ in range(len(cycle_matrix_config))])
        self.W_hour = nn.Parameter(torch.FloatTensor(pre_len, nodes_num))
        self.init_parameter()

    def init_parameter(self):
        nn.init.xavier_uniform_(self.W_hour)
        for i in range(len(self.cycle_matrix_config)):
            nn.init.xavier_uniform_(self.W_list_similarity[i])

    def forward(self, x_list):
        # for x_hour
        x_hour = self.data_embedding(x_list[0].permute(0, 3, 1, 2))
        cycle_data = x_list[1].transpose(0, 1)
        # (batch_size, seq_len, nodes_num, embedding_dim)
        # -> (batch_size, seq_len, nodes_num, hidden_dim)
        # -> (batch_size, pre_len, nodes_num)
        hour_out = self.final_conv(self.sub_blocks(x_hour))[:, :, :, -1]
        pre = self.W_hour * hour_out
        for i in range(len(self.cycle_matrix_config)):
            pre += self.W_list_similarity[i] * self.cycle_module_list[i](x_hour, self.data_embedding(cycle_data[i].permute(0, 3, 1, 2)))[:, :, :, -1]
        return pre.transpose(1, 2)


# class EMBSFormer(nn.Module):
#     def __init__(self,
#                  seq_len,
#                  pre_len,
#                  nodes_num,
#                  feature_dim,
#                  embedded_dim,
#                  backbone_list,
#                  heads,
#                  num_of_days,
#                  num_of_weeks,
#                  add_time_in_day=False,
#                  add_day_in_week=False,
#                  add_holiday=False,
#                  droupout=0.1):
#         super(EMBSFormer, self).__init__()
#         self.seq_len = seq_len
#         self.pre_len = pre_len
#         self.num_of_days = num_of_days
#         self.num_of_weeks = num_of_weeks
#         self.data_embedding = DataEmbedding(embedded_dim, feature_dim, droupout, add_time_in_day, add_day_in_week, add_holiday)
#         self.sub_blocks = STAG_subblock(backbone_list, heads, seq_len, droupout)
#         self.day_attention = Similarity_attention(seq_len, pre_len, embedded_dim, heads, droupout)
#         self.week_attention = Similarity_attention(seq_len, pre_len, embedded_dim, heads, droupout)
#         self.final_conv = nn.Conv2d(seq_len,
#                                     pre_len,
#                                     kernel_size=(1, backbone_list[-1]["hidden_dim"]))
#         self.W_hour = nn.Parameter(torch.FloatTensor(pre_len, nodes_num))
#         self.W_day = nn.Parameter(torch.FloatTensor(pre_len, nodes_num))
#         self.W_week = nn.Parameter(torch.FloatTensor(pre_len, nodes_num))
#         # self.W_day_sum = nn.Parameter(torch.FloatTensor(num_of_days, pre_len, nodes_num))
#         # self.W_week_sum = nn.Parameter(torch.FloatTensor(num_of_weeks, pre_len, nodes_num))
#         self.layer_norm = LayerNorm(embedded_dim)
#         self.feedforward = FeedForward(embedded_dim)
#         self.day_final_conv = nn.Conv2d(pre_len, pre_len, kernel_size=(1, embedded_dim))
#         self.week_final_conv = nn.Conv2d(pre_len, pre_len, kernel_size=(1, embedded_dim))
#         self.init_parameter()
#
#     def init_parameter(self):
#         nn.init.xavier_uniform_(self.W_day)
#         nn.init.xavier_uniform_(self.W_hour)
#         nn.init.xavier_uniform_(self.W_week)
#         # nn.init.xavier_uniform_(self.W_day_sum)
#         # nn.init.xavier_uniform_(self.W_week_sum)
#
#     def forward(self, x_list):
#         # for x_hour
#         x_hour = self.data_embedding(x_list[0].permute(0, 3, 1, 2))
#         device = x_hour.device
#         batch_size, seq_len, nodes_num, embedding_dim = x_hour.shape
#         x_day = x_list[1].permute(0, 3, 1, 2)
#         x_week = x_list[2].permute(0, 3, 1, 2)
#         # (batch_size, seq_len, nodes_num, embedding_dim)
#         # -> (batch_size, seq_len, nodes_num, hidden_dim)
#         # -> (batch_size, pre_len, nodes_num)
#         hour_out = self.final_conv(self.sub_blocks(x_hour))[:, :, :, -1]
#         x_day_out = torch.zeros(batch_size, self.pre_len, nodes_num, embedding_dim).to(device)
#         for i in range(self.num_of_days):
#             x_day_cash = self.data_embedding(
#                 x_day[:, i * (self.pre_len + self.seq_len):(i + 1) * (self.pre_len + self.seq_len), :, :])
#             day_out_attention = self.feedforward(
#                 self.layer_norm(self.day_attention(x_hour, x_day_cash[:, :self.seq_len, :, :], x_day_cash[:, self.seq_len:, :, :])))
#             # x_day_out += self.W_day_sum[i, :, :] * self.day_final_conv(day_out_attention)[:, :, :, -1]
#             x_day_out += day_out_attention
#         x_week_out = torch.zeros(batch_size, self.pre_len, nodes_num, embedding_dim).to(device)
#         for i in range(self.num_of_weeks):
#             x_week_cash = self.data_embedding(
#                 x_week[:, i * (self.pre_len + self.seq_len):(i + 1) * (self.pre_len + self.seq_len), :, :])
#             week_out_attention = self.feedforward(self.layer_norm(
#                     self.week_attention(x_hour, x_week_cash[:, :self.seq_len, :, :], x_week_cash[:, self.seq_len:, :, :])))
#             # x_week_out += self.W_week_sum[i, :, :] * self.week_final_conv(week_out_attention)[:, :, :, -1]
#             x_week_out += week_out_attention
#         pre = self.W_hour * hour_out \
#               + self.W_day * self.day_final_conv(self.layer_norm(x_day_out))[:, :, :, -1] \
#               + self.W_week * self.week_final_conv(self.layer_norm(x_week_out))[:, :, :, -1]
#         return pre.transpose(1, 2)

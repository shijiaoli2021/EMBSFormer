import torch.nn as nn
import torch
import math
import torch.nn.functional as F


'''
what? muti-head just do 1 for Q,K,V
'''

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

class MutiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads, dropout = 0.1):
        super(MutiHeadSelfAttention, self).__init__()
        assert d_model % heads == 0
        self.h = heads
        self.d_model = d_model
        self.s_h = d_model // heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.outLinear = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask = None):
        # (batch, l, N, d_model)  (batch, l1, h, N, s_h) * (batch, l2, h, s_h, N)*(batch, l2, h, )
        batchSize = q.size(0)
        posOneDim_q = q.size(1)
        posOneDim_k = k.size(1)
        posOneDim_v = v.size(1)
        q = self.q_linear(q).view(batchSize, posOneDim_q, -1, self.h, self.s_h)
        k = self.k_linear(k).view(batchSize, posOneDim_k, -1, self.h, self.s_h)
        v = self.v_linear(v).view(batchSize, posOneDim_v, -1, self.h, self.s_h)

        q = q.transpose(1, 3)
        k = k.transpose(1, 3)
        v = v.transpose(1, 3)

        att = attention(q, k, v, self.s_h, mask, self.dropout)
        concat = att.transpose(1, 3).contiguous().view(batchSize, posOneDim_q, -1, self.d_model)
        out = self.outLinear(concat)
        return out


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=0.1):
        super().__init__()

        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x



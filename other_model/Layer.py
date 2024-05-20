import torch.nn as nn
from other_model import SubLayers
import copy


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

'''
for spatialAttention
'''
class SpatialAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, norm_eps=1e-6, ff_middle_dim=2048):
        super(SpatialAttentionLayer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.layerNorm = SubLayers.LayerNorm(d_model, norm_eps)
        self.spatialAttention = SubLayers.MutiHeadSelfAttention(d_model, heads, dropout)
        self.feedForward = SubLayers.FeedForward(d_model, ff_middle_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x(batch, l, N, d_model) laplacian(N,N)  laplacian * x =>(batch, l, N, d_model)
        # mutiheads-SpatialAttentionNx
        x = x + self.dropout1(self.spatialAttention(x, x, x, mask))
        x = self.layerNorm(x)
        # (batch, l, N, d_model)
        x = x + self.dropout2(self.feedForward(x))
        return self.layerNorm(x)


'''
for temporal-Attention
'''
class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, norm_eps=1e-6, ff_middle_dim=2048):
        super(TemporalAttentionLayer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.layerNorm = SubLayers.LayerNorm(d_model, norm_eps)
        self.temporalAttention = SubLayers.MutiHeadSelfAttention(d_model, heads, dropout)
        self.feedForward = SubLayers.FeedForward(d_model, ff_middle_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # mutiheads-SpatialAttentionNx
        x = x + self.dropout1(self.temporalAttention(x, x, x, mask))
        x = self.layerNorm(x)
        # (batch, l, N, d_model)
        x = x + self.dropout2(self.feedForward(x))
        return self.layerNorm(x)


class DecodeLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1, norm_eps=1e-6, ff_middle_dim=2048):
        super(DecodeLayer, self).__init__()
        self.d_model = d_model
        self.heads = heads
        self.layerNorm = SubLayers.LayerNorm(d_model, norm_eps)
        self.selfAttention = SubLayers.MutiHeadSelfAttention(d_model, heads, dropout)
        self.feedForward = SubLayers.FeedForward(d_model, ff_middle_dim, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, e_spacial, e_temporal, src_mask, trg_mask):
        x = x + self.dropout1(self.selfAttention(x, x, x, trg_mask))
        x = self.layerNorm(x)
        src = e_spacial + e_temporal.transpose(-3, -2)
        x = x + self.dropout2(self.selfAttention(x, src, src, src_mask))
        x = self.layerNorm(x)
        x = x + self.dropout3(self.feedForward(x))
        x = self.layerNorm(x)
        return x




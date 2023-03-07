import torch, torch.nn as nn, torch.nn.functional as F

class StatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=2), x.std(dim=2)], dim=1)
        return out

    
class AvgPool(nn.Module):
    
    def __init__(self):
        super(AvgPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        return x.mean(dim=2)
    
class TemporalStatsPool(nn.Module):
    
    def __init__(self):
        super(StatsPool, self).__init__()

    def forward(self, x):
        x = x.view(x.shape[0], x.shape[1], -1)
        out = torch.cat([x.mean(dim=1), x.std(dim=1)], dim=1)
        return out


class ScaleDotProductAttention(nn.Module):
    
    def __init__(self, embed_dim):
        super(ScaleDotProductAttention, self).__init__()
        self.scaling = float(embed_dim) ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        if not self.training:
            return x.mean(dim=1)
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        attn_output_weights = F.softmax(torch.bmm(q, k.transpose(1, 2)), dim=-1)
        return torch.bmm(attn_output_weights, x)


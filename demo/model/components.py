# model/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        return self.model(x)

class GatedFeatureMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.4):
        super().__init__()
        layers = []
        dims = [input_dim, 256, hidden_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(dims)-2 else 0.2))
        self.mlp = nn.Sequential(*layers)

        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)
        return x * self.gate(x)  # feature-wise gating

class GatedFeatureMLPDeepOmics(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.4):
        super().__init__()
        # 1. MLP for progressive dimension reduction
        layers = []
        dims = [input_dim, 4096, 1024, hidden_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(dims)-2 else 0.2))
        self.mlp = nn.Sequential(*layers)

        # 2. Gating layer (like SE-block)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)      # (B, hidden_dim)
        gate = self.gate(x)  # (B, hidden_dim)
        return x * gate      # feature-wise gating

class GFSA(nn.Module):
    def __init__(self, embed_dim=128, heads=4, K=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dim_head = embed_dim // heads
        self.K = K

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.w0 = nn.Parameter(torch.zeros(heads))
        self.w1 = nn.Parameter(torch.ones(heads))
        self.wK = nn.Parameter(torch.zeros(heads))

    def forward(self, x):
        B, N, D = x.size()
        H = self.heads
        d = self.dim_head

        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        att = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        att = F.softmax(att, dim=-1)

        att_squared = torch.matmul(att, att)
        att_K = att + (self.K-1)*(att_squared - att)

        I = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)

        w0 = self.w0.view(1, H, 1, 1)
        w1 = self.w1.view(1, H, 1, 1)
        wK = self.wK.view(1, H, 1, 1)

        gf_att = w0 * I + w1 * att + wK * att_K

        out = torch.matmul(gf_att, v)
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        return self.out_proj(out)

class GFSAEncoderLayer(nn.Module):
    def __init__(self, embed_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.gfsa = GFSA(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x2 = self.gfsa(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        x2 = self.ffn(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x

class FeatureFusionGFSA(nn.Module):
    def __init__(self, hidden_dim=128, heads=4, num_layers=8, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            GFSAEncoderLayer(embed_dim=hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
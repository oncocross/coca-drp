# model/cdr_model.py

import torch
import torch.nn as nn
from .components import *

class CDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        '''
        morgan torch.Size([1024])
        gin torch.Size([300])
        desc torch.Size([200])
        dti torch.Size([1572])
        crispr torch.Size([17931])
        pro torch.Size([4922])
        exp torch.Size([15278])
        meth torch.Size([14608])
        '''
        
        # Fingerprint projectors
        self.morgan_proj = GatedFeatureMLP(1024)
        # self.desc_proj = GatedFeatureMLP(200)
        self.gin_proj = GatedFeatureMLP(300)
        self.dti_proj = GatedFeatureMLP(1572)
        
        self.crispr_proj = GatedFeatureMLPDeepOmics(17931)
        self.pro_proj = GatedFeatureMLPDeepOmics(4922)
        self.exp_proj = GatedFeatureMLPDeepOmics(15278)
        self.meth_proj = GatedFeatureMLPDeepOmics(14608)

        self.fusion_attn_drug = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic1 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic2 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic3 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)

        # ----- cross-modal -----
        self.cross_attn_dq_o = CrossAttention(embed_dim=128, num_heads=4)  # drug(query) ← omics(key/value)
        self.cross_attn_oq_d = CrossAttention(embed_dim=128, num_heads=4)  # omics(query) ← drug(key/value)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )

    def forward(self, batch):
        morgan = self.morgan_proj(batch['morgan'])
        gin    = self.gin_proj(batch['gin'])
        dti    = self.dti_proj(batch['dti'])
        exp    = self.exp_proj(batch['exp'])
        meth   = self.meth_proj(batch['meth'])
        pro    = self.pro_proj(batch['pro'])
        crispr = self.crispr_proj(batch['crispr'])

        # drug fusion
        d_tokens = torch.stack([morgan, gin, dti], dim=1)
        d_repr   = self.fusion_attn_drug(d_tokens)  # [B,3,128]

        # omics fusion (hierarchical)
        o_tokens_1 = torch.stack([exp, meth], dim=1)
        o_repr1    = self.fusion_attn_omic1(o_tokens_1)
        o_tokens_2 = torch.cat([o_repr1, pro.unsqueeze(1)], dim=1)
        o_repr2    = self.fusion_attn_omic2(o_tokens_2)
        o_tokens_3 = torch.cat([o_repr2, crispr.unsqueeze(1)], dim=1)
        o_repr     = self.fusion_attn_omic3(o_tokens_3)  # [B,4,128]

        # ===== 3) Cross-modal =====
        # drug-as-query attends to omics
        d_interact = self.cross_attn_dq_o(d_repr, o_repr)   # [B,3,128]
        # omics-as-query attends to drug
        o_interact = self.cross_attn_oq_d(o_repr, d_repr)   # [B,4,128]

        # ===== 4) Classifier =====
        tokens = torch.cat([d_interact, o_interact], dim=1)  # [B, 3+4, 128] = [B,7,128]
        out    = self.classifier(tokens)                     # [B,1]

        return out.squeeze(-1), d_interact, o_interact
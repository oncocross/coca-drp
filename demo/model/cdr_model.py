# model/architecture.py

import torch
import torch.nn as nn
from .components import GatedFeatureMLP, GatedFeatureMLPDeepOmics, FeatureFusionGFSA

class CDRModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Fingerprint projectors
        self.morgan_proj = GatedFeatureMLP(1024)
        self.gin_proj = GatedFeatureMLP(300)
        self.dti_proj = GatedFeatureMLP(1572)
        
        # Omics projectors
        self.crispr_proj = GatedFeatureMLPDeepOmics(17931)
        self.pro_proj = GatedFeatureMLPDeepOmics(4922)
        self.exp_proj = GatedFeatureMLPDeepOmics(15278)
        self.meth_proj = GatedFeatureMLPDeepOmics(14608)

        # Fusion modules
        self.fusion_attn_drug = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)

        # Classifier
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
        d_vecs = [
            self.morgan_proj(batch['morgan']),
            self.gin_proj(batch['gin']),
            self.dti_proj(batch['dti']),
        ]
        o_vecs = [
            self.crispr_proj(batch['crispr']),
            self.pro_proj(batch['pro']),
            self.exp_proj(batch['exp']),
            self.meth_proj(batch['meth']),
        ]
        
        d_repr = self.fusion_attn_drug(torch.stack(d_vecs, dim=1))
        o_repr = self.fusion_attn_omic(torch.stack(o_vecs, dim=1))
        
        # Concatenate fused representations
        combined_repr = torch.cat([d_repr, o_repr], dim=1)
        
        out = self.classifier(combined_repr)
        
        return out.squeeze(-1), d_repr, o_repr
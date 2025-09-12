# model/cdr_model.py
# Contains the main Cancer Drug Response (CDR) PyTorch model definition.

import torch
import torch.nn as nn
# Import all custom layer modules from the sibling 'layers.py' file.
from model.layers import *

class CDRModel(nn.Module):
    """
    The main model for predicting cancer drug response (lnIC50).
    It processes multi-modal drug and omics features, fuses them using
    self-attention and cross-attention, and predicts a final response value.
    """
    def __init__(self):
        super().__init__()
        
        # --- 1. Feature Projectors ---
        # Individual MLPs to project each raw feature vector into a common embedding space (128 dimensions).
        
        # Drug feature projectors
        self.morgan_proj = GatedFeatureMLP(1024)
        self.gin_proj = GatedFeatureMLP(300)
        self.dti_proj = GatedFeatureMLP(1572)
        
        # Omics feature projectors (using a deeper MLP for higher dimensionality)
        self.crispr_proj = GatedFeatureMLPDeepOmics(17931)
        self.pro_proj = GatedFeatureMLPDeepOmics(4922)
        self.exp_proj = GatedFeatureMLPDeepOmics(15278)
        self.meth_proj = GatedFeatureMLPDeepOmics(14608)

        # --- 2. Intra-modal Fusion (Self-Attention) ---
        # GFSA encoders to fuse features within the same modality (drug or omics).
        self.fusion_attn_drug = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic1 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic2 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)
        self.fusion_attn_omic3 = FeatureFusionGFSA(hidden_dim=128, heads=4, num_layers=8)

        # --- 3. Cross-modal Fusion (Cross-Attention) ---
        # Cross-attention layers to model interactions between drug and omics modalities.
        self.cross_attn_dq_o = CrossAttention(embed_dim=128, num_heads=4) # drug(query) attends to omics(key/value)
        self.cross_attn_oq_d = CrossAttention(embed_dim=128, num_heads=4) # omics(query) attends to drug(key/value)

        # --- 4. Final Classifier ---
        # An MLP that takes the fused representations and predicts the final lnIC50 value.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*7, 512), # Input size is 128 * (3 drug tokens + 4 omics tokens)
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
        # --- 1. Projection ---
        # Pass each raw feature through its dedicated projector MLP.
        morgan = self.morgan_proj(batch['morgan'])
        gin    = self.gin_proj(batch['gin'])
        dti    = self.dti_proj(batch['dti'])
        exp    = self.exp_proj(batch['exp'])
        meth   = self.meth_proj(batch['meth'])
        pro    = self.pro_proj(batch['pro'])
        crispr = self.crispr_proj(batch['crispr'])

        # --- 2. Intra-modal Fusion ---
        # Fuse drug features into a single representation.
        d_tokens = torch.stack([morgan, gin, dti], dim=1)
        d_repr   = self.fusion_attn_drug(d_tokens)  # Shape: [Batch, 3, 128]

        # Fuse omics features hierarchically.
        o_tokens_1 = torch.stack([exp, meth], dim=1)
        o_repr1    = self.fusion_attn_omic1(o_tokens_1)
        o_tokens_2 = torch.cat([o_repr1, pro.unsqueeze(1)], dim=1)
        o_repr2    = self.fusion_attn_omic2(o_tokens_2)
        o_tokens_3 = torch.cat([o_repr2, crispr.unsqueeze(1)], dim=1)
        o_repr     = self.fusion_attn_omic3(o_tokens_3)  # Shape: [Batch, 4, 128]

        # --- 3. Cross-modal Fusion ---
        # Model interactions between the two modalities.
        d_interact = self.cross_attn_dq_o(d_repr, o_repr) # Drug representation refined by omics info.
        o_interact = self.cross_attn_oq_d(o_repr, d_repr) # Omics representation refined by drug info.

        # --- 4. Prediction ---
        # Concatenate all refined tokens and pass them to the classifier.
        tokens = torch.cat([d_interact, o_interact], dim=1) # Shape: [Batch, 7, 128]
        out    = self.classifier(tokens)                    # Shape: [Batch, 1]

        # Return the final prediction and the intermediate interaction representations.
        return out.squeeze(-1), d_interact, o_interact
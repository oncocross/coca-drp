# model/components.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureMLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) block for feature processing."""
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # Linear layer to project features.
            nn.LayerNorm(hidden_dim),         # Layer normalization for stable training.
            nn.ReLU(),                        # ReLU activation function.
            nn.Dropout(0.2)                   # Dropout for regularization.
        )

    def forward(self, x):
        return self.model(x)

class GatedFeatureMLP(nn.Module):
    """An MLP with a feature-wise gating mechanism."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.4):
        super().__init__()
        # 1. A deeper MLP to process the features.
        layers = []
        dims = [input_dim, 256, hidden_dim] # Define layer dimensions.
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            # Apply higher dropout to intermediate layers.
            layers.append(nn.Dropout(dropout if i < len(dims)-2 else 0.2))
        self.mlp = nn.Sequential(*layers)

        # 2. A gate layer that learns a weight for each feature.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid() # Sigmoid squashes the output to a range of [0, 1].
        )

    def forward(self, x):
        # Process input through the main MLP.
        x = self.mlp(x)
        # Apply the gate by element-wise multiplication. This allows the model
        # to learn to scale features up or down based on their importance.
        return x * self.gate(x)

class GatedFeatureMLPDeepOmics(nn.Module):
    """A deeper Gated MLP specifically designed for high-dimensional omics data."""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.4):
        super().__init__()
        # 1. A deep MLP for progressive dimensionality reduction.
        layers = []
        dims = [input_dim, 4096, 1024, hidden_dim]
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.LayerNorm(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout if i < len(dims)-2 else 0.2))
        self.mlp = nn.Sequential(*layers)

        # 2. A gating layer inspired by Squeeze-and-Excitation (SE) blocks.
        # It uses a bottleneck to learn more complex inter-feature relationships.
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), # Squeeze
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), # Excite
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.mlp(x)       # (Batch, hidden_dim)
        gate = self.gate(x)   # (Batch, hidden_dim)
        return x * gate       # Apply feature-wise gating.

class GFSA(nn.Module):
    """
    Gated Feature Self-Attention.
    A modified self-attention mechanism that incorporates higher-order attention
    relationships through a weighted combination of identity, standard attention,
    and a polynomial attention term.
    """
    def __init__(self, embed_dim=128, heads=4, K=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.dim_head = embed_dim // heads
        self.K = K

        # Standard linear projections for Query, Key, Value, and Output.
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Trainable weights for combining the different attention components.
        self.w0 = nn.Parameter(torch.zeros(heads)) # Weight for identity matrix
        self.w1 = nn.Parameter(torch.ones(heads))  # Weight for standard attention
        self.wK = nn.Parameter(torch.zeros(heads)) # Weight for higher-order attention

    def forward(self, x):
        B, N, D = x.size() # Batch, Sequence Length, Dimension
        H = self.heads
        d = self.dim_head

        # 1. Project to Q, K, V and reshape for multi-head attention.
        q = self.q_proj(x).view(B, N, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, N, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, N, H, d).transpose(1, 2)

        # 2. Calculate standard attention scores.
        att = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
        att = F.softmax(att, dim=-1)

        # 3. Calculate a higher-order attention matrix.
        # This can be interpreted as considering multi-step relationships (like friends of friends).
        att_squared = torch.matmul(att, att)
        att_K = att + (self.K-1)*(att_squared - att)

        # 4. Create an identity matrix for the residual connection component.
        I = torch.eye(N, device=x.device).unsqueeze(0).unsqueeze(0)

        # Reshape trainable weights to match attention matrix dimensions.
        w0 = self.w0.view(1, H, 1, 1)
        w1 = self.w1.view(1, H, 1, 1)
        wK = self.wK.view(1, H, 1, 1)

        # 5. Combine the components into the final Gated Feature Attention matrix.
        gf_att = w0 * I + w1 * att + wK * att_K

        # 6. Apply the final attention to the Value vectors.
        out = torch.matmul(gf_att, v)
        # Reshape back to the original input shape.
        out = out.transpose(1, 2).contiguous().view(B, N, D)

        # 7. Final linear projection.
        return self.out_proj(out)

class GFSAEncoderLayer(nn.Module):
    """
    A single Transformer encoder layer that uses the custom GFSA module for self-attention.
    Follows the standard "Add & Norm" structure.
    """
    def __init__(self, embed_dim=128, heads=4, dropout=0.1):
        super().__init__()
        self.gfsa = GFSA(embed_dim, heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        # Standard Feed-Forward Network (FFN).
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        # First sub-layer: GFSA followed by dropout, residual connection, and layer norm.
        x2 = self.gfsa(x)
        x = x + self.dropout1(x2)
        x = self.norm1(x)

        # Second sub-layer: FFN followed by dropout, residual connection, and layer norm.
        x2 = self.ffn(x)
        x = x + self.dropout2(x2)
        x = self.norm2(x)

        return x

class FeatureFusionGFSA(nn.Module):
    """A stack of GFSAEncoderLayer modules to create a full encoder."""
    def __init__(self, hidden_dim=128, heads=4, num_layers=8, dropout=0.1):
        super().__init__()
        # Create a list of encoder layers.
        self.layers = nn.ModuleList([
            GFSAEncoderLayer(embed_dim=hidden_dim, heads=heads, dropout=dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # Pass the input sequentially through all the layers.
        for layer in self.layers:
            x = layer(x)
        return x


class CrossAttention(nn.Module):
    """
    A standard cross-attention module.
    The 'query' sequence attends to the 'key_value' sequence.
    """
    def __init__(self, embed_dim=128, num_heads=4, dropout=0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=dropout)
        self.ln   = nn.LayerNorm(embed_dim) # Layer norm for stability.

    def forward(self, query, key_value):
        # query shape: [Batch, Query_Seq_Len, Dim]
        # key_value shape: [Batch, Key_Seq_Len, Dim]
        # In cross-attention, key and value are the same.
        out, _ = self.attn(query, key_value, key_value)
        # Apply a residual connection and layer normalization.
        return self.ln(out + query)
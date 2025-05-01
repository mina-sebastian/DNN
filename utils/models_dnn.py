from typing import Callable, List
import torch
import torch.nn as nn


class DualInputMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=1):
        """
        Example:
        - input_dim = 2560 (dimension of LLMic embeddings)
        - hidden_dim = 512  (feel free to adjust)
        - num_classes = 1   (binary classification)
        """
        super().__init__()

        # Separate branches for title and content
        self.title_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.content_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combine both embeddings
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, title_emb, content_emb):
        """
        :param title_emb: shape (batch_size, 2560)
        :param content_emb: shape (batch_size, 2560)
        :return: shape (batch_size, num_classes)
        """
        title_features = self.title_branch(title_emb)
        content_features = self.content_branch(content_emb)

        # Concatenate along dimension=1
        combined_features = torch.cat((title_features, content_features), dim=1)

        # Final classification
        out = self.combined(combined_features)
        return out
    
# class RoARCMLP(nn.Module):
#     def __init__(self,
#                  input_size: int,
#                  hidden_sizes: List[int],
#                  device: torch.device,
#                  activation_fn: Callable,
#                  output_activation_fn: Callable):
#         super().__init__()
#         layers = []
#         prev_size = input_size
#         self.device = device
#         self.activation_fn = activation_fn
#         self.output_activation_fn = output_activation_fn

#         for hs in hidden_sizes:
#             layers.append(nn.Linear(prev_size, hs))
#             # layers.append(nn.BatchNorm1d(hs, momentum=0.1))
#             layers.append(activation_fn)
#             prev_size = hs
#         layers.append(nn.Linear(prev_size, 1))
#         self.network = nn.Sequential(*layers)

#     def forward(self, x):
#         x = x.to(self.device)
#         x = self.network(x)
#         x = self.output_activation_fn(x)
#         return x    
    

class CincualInputMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        """
        Example:
        - input_dim = 2560 (dimension of LLMic embeddings)
        - hidden_dim = 512  (feel free to adjust)
        - num_classes = 4   (q classification)
        """
        super().__init__()

        # Separate branches for question and content
        self.q_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.option_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(4)
        ])

        # Combine both embeddings
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, q_emb, option_embs):
        """
        :param q_emb: Tensor of shape (batch_size, emb_dim)
        :param option_embs: Tensor of shape (batch_size, num_options, emb_dim)
        :return: Tensor of shape (batch_size, num_classes)
        """
        q_features = self.q_branch(q_emb)  # [batch_size, hidden_dim]

        # Unpack options from the second dimension
        # option_embs[:, 0] = option A, shape: [batch_size, emb_dim]
        option_features = [
            branch(option_embs[:, i])  # apply each branch to the i-th option
            for i, branch in enumerate(self.option_branches)
        ]  # each = [batch_size, hidden_dim]

        # Concatenate along the feature dimension
        combined_features = torch.cat([q_features] + option_features, dim=1)  # [batch_size, hidden_dim * 5]

        out = self.combined(combined_features)  # [batch_size, num_classes]
        return out


class CrossAttentionMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        super().__init__()
        
        # Linear projections for question and options into shared hidden space
        self.q_proj = nn.Linear(input_dim, hidden_dim)
        self.o_proj = nn.Linear(input_dim, hidden_dim)

        # Attention mechanism: question attends to options
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=0.4,
            batch_first=True  # Important: enables (batch, seq, dim) format
        )

        # Final classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, q_emb, option_embs):
        """
        q_emb:        (batch, input_dim)
        option_embs:  (batch, 4, input_dim)
        """
        # Project into hidden dimension
        q_proj = self.q_proj(q_emb).unsqueeze(1)         # (batch, 1, hidden)
        o_proj = self.o_proj(option_embs)                # (batch, 4, hidden)

        # Apply attention: question attends to options
        attn_output, attn_weights = self.attention(q_proj, o_proj, o_proj)  # (batch, 1, hidden)
        attn_output = attn_output.squeeze(1)             # (batch, hidden)

        # Flatten the options to preserve their individual information
        options_flat = o_proj.reshape(o_proj.size(0), -1)  # (batch, hidden * 4)

        # Fuse attended summary + all options
        combined = torch.cat([attn_output, options_flat], dim=1)  # (batch, hidden * 5)

        # Classify
        return self.classifier(combined)  # (batch, num_classes)


class PairwiseQAScorer(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512):
        super().__init__()
        self.pair_encoder = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1)  # output: score
        )

    def forward(self, q_emb, option_embs):
        # q_emb: (batch, input_dim)
        # option_embs: (batch, 4, input_dim)

        q_rep = q_emb.unsqueeze(1).expand(-1, 4, -1)  # (batch, 4, input_dim)
        fused = torch.cat([q_rep, option_embs], dim=2)  # (batch, 4, input_dim*2)

        logits = self.pair_encoder(fused).squeeze(-1)  # (batch, 4)
        return logits  # Use CrossEntropyLoss with target = correct option index

import torch.nn.functional as F
class MultiOptionMLP(nn.Module):
    """
    Vectorised version of QAOptionMLP.
    Accepts (batch, 1, d) + (batch, 4, d)   →   logits (batch, 4)
    """
    def __init__(self, input_dim: int, hidden_dim: int = 512, dropout=0.3):
        super().__init__()
        self.norm   = nn.LayerNorm(input_dim)
        in_dim      = input_dim * 4          # q, a, |q-a|, q*a
        self.fc1    = nn.Linear(in_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, 1)
        self.drop   = nn.Dropout(dropout)

    def _pair_features(self, q, a):
        """q, a: [batch, 4, d]  (q already broadcast)"""
        diff  = torch.abs(q - a)
        prod  = q * a
        return torch.cat([q, a, diff, prod], dim=-1)   # [batch, 4, 4d]

    def forward(self, q_emb, option_embs):
        """
        q_emb       : [batch, embed_dim]          (question)
        option_embs : [batch, 4, embed_dim]       (four options)
        returns     : [batch, 4]                  (logits)
        """
        # normalise
        q = self.norm(q_emb)                      # [batch, d]
        a = self.norm(option_embs)                # [batch, 4, d]

        # broadcast q -> [batch, 4, d]
        q = q.unsqueeze(1).expand_as(a)

        x = self._pair_features(q, a)             # [batch, 4, 4d]
        x = F.gelu(self.fc1(x))
        x = self.drop(x)
        logits = self.fc2(x).squeeze(-1)          # [batch, 4]
        return logits

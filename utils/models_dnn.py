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


class OneOptionMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 4096, dropout=0.3):
        super().__init__()
        in_dim      = input_dim          # q, a, |q-a|, q*a
        self.fc1    = nn.Linear(in_dim, hidden_dim)
        self.fc2    = nn.Linear(hidden_dim, 1)
        self.fc2    = nn.Linear(hidden_dim, 2048)
        self.fc3    = nn.Linear(2048, 512)
        self.fc4    = nn.Linear(512, 1)
        self.drop   = nn.Dropout(dropout)


    def forward(self, emb):
        x = self.norm(emb)                      # [batch, d]
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.drop(x)
        logits = self.fc4(x).squeeze(-1)          # [batch, 1]
        return logits
    
class QuadricInputMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        """
        Example:
        - input_dim = 2560 (dimension of LLMic embeddings)
        - hidden_dim = 512  (feel free to adjust)
        - num_classes = 4   (q classification)
        """
        super().__init__()

        self.option_branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ) for _ in range(4)
        ])

        # Combine both embeddings
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, option_embs):
        """
        :param q_emb: Tensor of shape (batch_size, emb_dim)
        :param option_embs: Tensor of shape (batch_size, num_options, emb_dim)
        :return: Tensor of shape (batch_size, num_classes)
        """
        
        # Unpack options from the second dimension
        # option_embs[:, 0] = option A, shape: [batch_size, emb_dim]
        option_features = [
            branch(option_embs[:, i])  # apply each branch to the i-th option
            for i, branch in enumerate(self.option_branches)
        ]  # each = [batch_size, hidden_dim]

        # Concatenate along the feature dimension
        combined_features = torch.cat(option_features, dim=1)  # [batch_size, hidden_dim * 4]

        out = self.combined(combined_features)  # [batch_size, num_classes]
        return out


class SharedInputMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        super().__init__()

        # One shared branch applied to all options
        self.shared_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Combined classifier head
        self.combined = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Linear(hidden_dim * 2, hidden_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, option_embs):
        # Apply the same branch to each option
        option_features = [self.shared_branch(option_embs[:, i]) for i in range(4)]
        combined_features = torch.cat(option_features, dim=1)  # [batch_size, hidden_dim * 4]
        out = self.combined(combined_features)  # [batch_size, num_classes]
        return out

class PairwiseQuadricMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        super().__init__()

        # Shared or independent encoders for each option (your choice)
        self.option_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Pairwise comparison MLP (same for all pairs)
        self.pairwise_comparator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1)  # scalar score for this pair
        )

        # Final scoring layer (per option)
        self.final_classifier = nn.Linear(6, num_classes)  # 6 comparisons

    def forward(self, option_embs):
        """
        :param option_embs: Tensor of shape (batch_size, 4, input_dim)
        :return: Tensor of shape (batch_size, num_classes)
        """
        batch_size = option_embs.size(0)

        # Step 1: Encode each option
        encoded = [self.option_branch(option_embs[:, i]) for i in range(4)]  # list of 4 tensors [batch, hidden_dim]

        # Step 2: Pairwise differences
        diffs = []
        indices = []
        for i in range(4):
            for j in range(i + 1, 4):
                diff = torch.abs(encoded[i] - encoded[j])  # [batch, hidden_dim]
                diffs.append(self.pairwise_comparator(diff))  # [batch, 1]
                indices.append((i, j))

        # Step 3: Build pairwise matrix [batch, 4, 4] and count "wins"
        scores = torch.zeros(batch_size, 4, device=option_embs.device)

        for idx, (i, j) in enumerate(indices):
            s_ij = diffs[idx].squeeze(1)  # [batch]
            scores[:, i] += s_ij
            scores[:, j] += -s_ij  # opposite sign: i wins → j loses

        return scores  # logits for 4 options





import torch
import torch.nn as nn

class PairwiseQuadricWithQueryMLP(nn.Module):
    def __init__(self, input_dim=2560, hidden_dim=512, num_classes=4):
        super().__init__()

        # Shared encoder for each option
        self.option_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Shared comparator for each (option - query) pair
        self.scoring_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)  # score for how well option matches query
        )

    def forward(self, option_embs):
        """
        :param option_embs: Tensor of shape (batch_size, 4, input_dim)
        :return: Tensor of shape (batch_size, 4)
        """
        batch_size = option_embs.size(0)

        # Step 1: Encode options individually
        encoded_options = [self.option_encoder(option_embs[:, i]) for i in range(4)]  # list of 4 x [batch, hidden_dim]

        # Step 2: Compute pseudo-query (mean of all options)
        stacked_options = torch.stack(encoded_options, dim=1)  # [batch, 4, hidden_dim]
        query = torch.mean(stacked_options, dim=1)  # [batch, hidden_dim]

        # Step 3: Score each option against query
        scores = []
        for i in range(4):
            diff = torch.abs(encoded_options[i] - query)  # [batch, hidden_dim]
            score = self.scoring_mlp(diff).squeeze(1)  # [batch]
            scores.append(score)

        # Stack final logits
        logits = torch.stack(scores, dim=1)  # [batch, 4]
        return logits

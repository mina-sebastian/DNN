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




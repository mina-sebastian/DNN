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
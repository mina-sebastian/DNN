import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseRankingLoss(nn.Module):
    """
    Margin-based pairwise loss for multiple-choice QA.

    For each sample:
        - The correct option should be scored higher than each incorrect one
        - Enforces: score_correct - score_wrong >= margin
        - Aggregates the loss over all incorrect options

    Parameters:
    ----------
    margin : float
        Minimum desired difference between correct and incorrect scores.
    reduction : str
        'mean' or 'sum' over the batch losses.
    """

    def __init__(self, margin: float = 0.5, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        scores : torch.Tensor
            Tensor of shape (B, num_options) — predicted scores for each option.
        labels : torch.Tensor
            Tensor of shape (B,) — correct answer index (0–num_options-1) for each example.

        Returns
        -------
        torch.Tensor
            Scalar loss value.
        """
        B, num_options = scores.shape
        device = scores.device

        total_loss = []
        for i in range(B):
            correct_idx = labels[i]
            correct_score = scores[i, correct_idx]

            for j in range(num_options):
                if j == correct_idx:
                    continue
                wrong_score = scores[i, j]
                # Margin loss: max(0, margin - (correct - wrong))
                loss = F.relu(self.margin - (correct_score - wrong_score))
                total_loss.append(loss)

        loss_tensor = torch.stack(total_loss)

        if self.reduction == "mean":
            return loss_tensor.mean()
        elif self.reduction == "sum":
            return loss_tensor.sum()
        else:
            return loss_tensor  # no reduction

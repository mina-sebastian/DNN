# from utils.base_model import ARC_TEST, ARC_TRAIN
# from utils.datasets_class import MultipleChoiceCombinedDataset
# from torch.utils.data import DataLoader, random_split
# from collections import Counter
# import torch

# from utils.models_dnn import ArcMLP
# from utils.pairwise_loss import PairwiseRankingLoss
# from utils.train_util import test_model, train_and_evaluate
# generator = torch.Generator().manual_seed(42)
# import torch.optim as optim


# models = (
#     ("llmic", 2560),
#     ("mgpt/mGPT-1.3B-romanian", 2048),
#     ("rollama/RoLlama2-7b-Instruct-2024-10-09", 4096),
#     ("llama/Llama-3.1-8B", 4096),
#     ("google/gemma-3-1b-pt", 1152),
# )
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pooling_strategies = ["mean", "last_token", "echo"]

# for model_id, emb_dim in models:
#     for strategy in pooling_strategies:
#             print(f"Using model: {model_id} with strategy: {strategy}")

#             full_dataset = MultipleChoiceCombinedDataset(
#                 csv_file=ARC_TRAIN,
#                 get_embedding=None,
#                 name=f'roarc_train_{model_id}_{strategy}',
#                 save_interval=1
#             )

#             test_dataset = MultipleChoiceCombinedDataset(
#                 csv_file=ARC_TEST,
#                 get_embedding=None,
#                 name=f'roarc_test_{model_id}_{strategy}',
#                 save_interval=1
#             )

#             print(f"Loaded {len(full_dataset)} samples.")
#             print(f"Shape of data example: {full_dataset[0]}")




import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.base_model import ARC_TRAIN
from utils.datasets_class import MultipleChoiceCombinedDataset

# ─────────────────────────────────────────────────────────────
MODELS = (
    ("llmic",                             2560),
    # ("mgpt/mGPT-1.3B-romanian",           2048),
    # ("rollama/RoLlama2-7b-Instruct-2024-10-09", 4096),
    # ("llama/Llama-3.1-8B",                4096),
    # ("google/gemma-3-1b-pt",              1152),
)
POOLING = ["last_token"]

DEVICE  = torch.device("cpu")
BATCH   = 900000

COLORS  = {
    "last_token": "#cfe2ff",
    "mean"      : "#ffeeba",
    "echo"      : "#d4edda",
}

os.makedirs("similarity_charts", exist_ok=True)


# def compute_stats(ds: MultipleChoiceCombinedDataset) -> tuple[float, float]:
#     correct_sims = []
#     wrong_sims = []

#     for idx in range(len(ds)):
#         q, opts, label = ds[idx]  # q: (d,), opts: (4, d), label: int

#         q = F.normalize(q.unsqueeze(0), dim=1)       # (1, d)
#         opts = F.normalize(opts, dim=1)              # (4, d)

#         sims = F.cosine_similarity(q, opts, dim=1)   # (4,)

#         correct_sims.append(sims[label].item())

#         wrong_indices = [i for i in range(4) if i != label]
#         wrong_avg = sims[wrong_indices].mean().item()
#         wrong_sims.append(wrong_avg)

#     return float(np.mean(correct_sims)), float(np.mean(wrong_sims))


from typing import Callable

def compute_stats(
    ds: MultipleChoiceCombinedDataset,
    similarity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
) -> tuple[float, float, list[float], list[float]]:
    correct_sims = []
    wrong_sims = []

    for idx in range(len(ds)):
        q, opts, label = ds[idx]

        q = q.unsqueeze(0)
        sims = similarity_fn(q, opts)

        correct_sims.append(sims[label].item())

        wrong_indices = [i for i in range(4) if i != label]
        wrong_avg = sims[wrong_indices].mean().item()
        wrong_sims.append(wrong_avg)

    return float(np.mean(correct_sims)), float(np.mean(wrong_sims)), correct_sims, wrong_sims




def cosine_similarity(q: torch.Tensor, opts: torch.Tensor) -> torch.Tensor:
    # q = F.normalize(q, dim=1)
    # opts = F.normalize(opts, dim=1)
    print(F.cosine_similarity(q, opts, dim=1))
    return F.cosine_similarity(q, opts, dim=1)

def dot_product(q: torch.Tensor, opts: torch.Tensor) -> torch.Tensor:
    return torch.matmul(opts, q.t()).squeeze(1)  # shape (4,)

def euclidean_similarity(q: torch.Tensor, opts: torch.Tensor) -> torch.Tensor:
    return -torch.norm(opts - q, dim=1)  # negative distance = similarity

def manhattan_similarity(q: torch.Tensor, opts: torch.Tensor) -> torch.Tensor:
    return -torch.sum(torch.abs(opts - q), dim=1)

def angular_distance(q: torch.Tensor, opts: torch.Tensor) -> torch.Tensor:
    # q = F.normalize(q, dim=1)
    # opts = F.normalize(opts, dim=1)
    cos_sim = F.cosine_similarity(q, opts, dim=1)
    return 1 - torch.acos(torch.clamp(cos_sim, -1 + 1e-5, 1 - 1e-5)) / np.pi  # normalize to [0,1]

def plot_similarity_hist(correct_sims, wrong_sims, model_id, strat, metric_name):
    plt.figure(figsize=(8, 5))
    bins = np.linspace(min(correct_sims + wrong_sims), max(correct_sims + wrong_sims), 30)

    plt.hist(correct_sims, bins=bins, alpha=0.6, label="Correct", color="green", density=True)
    plt.hist(wrong_sims, bins=bins, alpha=0.6, label="Wrong", color="red", density=True)

    plt.title(f"{model_id} | {strat} | {metric_name}", fontsize=13)
    plt.xlabel(f"{metric_name} score")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    plt.tight_layout()
    out_path = f"similarity_charts/hist_{metric_name}_{model_id.replace('/', '_')}_{strat}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    print(f"   ↳ saved histogram → {out_path}")


similarity_metrics = {
    "cos": cosine_similarity,
    # "dot": dot_product,
    # "euclidean": euclidean_similarity,
    # "manhattan": manhattan_similarity,
    # "angular": angular_distance
}

for metric_name, metric_fn in similarity_metrics.items():
    for model_id, _ in MODELS:
        bar_data = {}

        for strat in POOLING:
            print(f"{model_id}  |  {strat}")

            ds = MultipleChoiceCombinedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=None,
                name=f"roarc_train_{model_id}_{strat}",
                save_interval=1,
            )


            print(f"  Using similarity: {metric_name}")
            avg_corr, avg_wrong, correct_sims, wrong_sims = compute_stats(ds, similarity_fn=metric_fn)

            # bar data stays the same
            bar_data[strat] = [avg_corr, avg_wrong]

            # add this line:
            plot_similarity_hist(correct_sims, wrong_sims, model_id, strat, metric_name)

            

        # Plot
        fig, ax = plt.subplots(figsize=(7, 5))
        x = np.arange(2)
        width = 0.22

        for i, strat in enumerate(["last_token", "mean", "echo"]):
            vals = bar_data[strat]
            bar_positions = x + (i - 1) * width
            bars = ax.bar(bar_positions, vals, width,
                        color=COLORS[strat],
                        edgecolor="black",
                        label=strat.replace("_", " "))

            for j, val in enumerate(vals):
                ax.text(bar_positions[j], val + 0.01, f"{val:.3f}",
                        ha='center', va='bottom', fontsize=9)

        ax.set_ylabel(metric_name)
        ax.set_xticks(x)
        ax.set_xticklabels([
            "avg sim(q, correct)",
            "avg sim(q, wrong)"
        ], rotation=10, ha="right")
        ax.set_title(model_id, fontsize=13, pad=12)

        ax.legend(frameon=False, loc="upper right")

        plt.tight_layout()
        out_path = f"similarity_charts/{metric_name}_{model_id.replace('/', '_')}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
        print(f"saved {out_path}")


print("DONE")

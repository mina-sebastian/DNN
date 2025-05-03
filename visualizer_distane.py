# -------------------------------------------------------------
# Distance‑check: is question closer to its correct option?
# -------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.base_model import ARC_TRAIN
from utils.datasets_class import MultipleChoiceCombinedDataset                  # nice progress bar

USE_COSINE = False   # True → cosine distance, False → Euclidean (L2)
SAVE_FIGS  = True    # set False if you don’t want PNGs

def pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two 1‑D vectors."""
    if USE_COSINE:
        sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
        return 1.0 - sim               # cosine distance
    else:
        return np.linalg.norm(a - b)   # Euclidean (L2)

correct_dists     = []
min_wrong_dists   = []
margin_dists      = []
is_correct_closer = []

# Define or load dataset_to_visualise before using it
# Example: dataset_to_visualise = load_dataset() or dataset_to_visualise = some_predefined_list
dataset_to_visualise = MultipleChoiceCombinedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=None,
                emb_dim=2560,
                name=f'roarc_train_llmic_mean',
                save_interval=1
            )

for q_idx in tqdm(range(1000), desc="Scanning dataset"):
    q_vec, opt_vecs, correct_idx = dataset_to_visualise[q_idx]  # tensors
    q_vec      = q_vec.numpy()
    opt_vecs   = opt_vecs.numpy()          # shape: (num_options, emb_dim)

    dists      = np.array([pair_distance(q_vec, v) for v in opt_vecs])
    d_corr     = dists[correct_idx]
    d_others   = np.delete(dists, correct_idx)
    d_min_oth  = d_others.min()

    correct_dists.append(d_corr)
    min_wrong_dists.append(d_min_oth)
    margin_dists.append(d_min_oth - d_corr)
    is_correct_closer.append(d_corr < d_min_oth)

accuracy = np.mean(is_correct_closer)
print(f"\n✔  Correct option is the closest in {accuracy:.2%} of the questions.")

# -------------------------------------------------------------
# Plot 1 – scatter: d(correct) vs d(nearest wrong)
# -------------------------------------------------------------
plt.figure(figsize=(6, 6))
plt.scatter(correct_dists, min_wrong_dists, alpha=0.6)
lim = [min(correct_dists + min_wrong_dists), max(correct_dists + min_wrong_dists)]
plt.plot(lim, lim, linestyle="--")          # diagonal
plt.xlabel("‖q − (q+correct)‖")
plt.ylabel("‖q − (q+nearest wrong)‖")
plt.title("Question → Option distances")
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig("scatter_distances.png", dpi=160)
plt.show()

# -------------------------------------------------------------
# Plot 2 – histogram: Δ = min_wrong − correct  (margin)
# -------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(margin_dists, bins=30, alpha=0.8)
plt.axvline(0, linestyle="--")
plt.xlabel("Δ distance  (nearest wrong − correct)")
plt.ylabel("Number of questions")
plt.title("Margin: is the correct option closer?")
plt.tight_layout()
if SAVE_FIGS:
    plt.savefig("histogram_margins.png", dpi=160)
plt.show()

import torch
import numpy as np
from utils.base_model import ARC_TEST
from utils.datasets_class import MultipleChoiceCombinedDataset
from utils.models_dnn import ArcMLP

test_dataset = MultipleChoiceCombinedDataset(
    csv_file=ARC_TEST,
    get_embedding=None,
    name=f'roarc_test_rollama/RoLlama2-7b-Instruct-2024-10-09_last_token',
    save_interval=1
)

model = ArcMLP(
    input_dim=4096,
    hidden_dim=4096 * 2,
)

model_path = "E:\\DNN\\models\\final\\rollama\\RoLlama2-7b-Instruct-2024-10-09_last_token_arc_final\\06-05-2025_20-18_rollama\\RoLlama2-7b-Instruct-2024-10-09_last_token_arc_final_model.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

top_20 = []

with torch.no_grad():
    for i in range(len(test_dataset)):
        q, q_plus_opts, label = test_dataset[i]
        q = q.unsqueeze(0).to(device)
        q_plus_opts = q_plus_opts.unsqueeze(0).to(device)
        output = model(q, q_plus_opts).squeeze(0).cpu()  # shape: (4,)

        correct_score = output[label].item()
        wrong_scores = [output[j].item() for j in range(4) if j != label]
        best_wrong = max(wrong_scores)
        margin = correct_score - best_wrong

        top_20.append((margin, i, correct_score, best_wrong, label, output.tolist()))

# Sort by descending confidence margin
top_20.sort(reverse=True, key=lambda x: x[0])

# Show top 20 most confident predictions
print("\nTop 20 most confident correct predictions (by margin between correct and best wrong):")
for rank, (margin, idx, correct_score, best_wrong, label, logits) in enumerate(top_20[:20]):
    print(f"[{rank:02}] idx={idx} | margin={margin:.4f} | correct={correct_score:.4f} | best_wrong={best_wrong:.4f} | label={label} | outputs={logits}")
    

import matplotlib.pyplot as plt
FIG_DIR = "figures"
# ───────────────────────────────────────────────────────────────
# 2)  HISTOGRAM –  FULL MARGIN DISTRIBUTION
# ───────────────────────────────────────────────────────────────
# If you kept only the top‑20, regenerate *all* margins first:
all_margins = []

with torch.no_grad():
    for q, q_plus_opts, lbl in test_dataset:
        q = q.unsqueeze(0).to(device)
        q_plus_opts = q_plus_opts.unsqueeze(0).to(device)
        out = model(q, q_plus_opts).squeeze(0).cpu()

        correct   = out[lbl].item()
        best_wrong = max(out[j].item() for j in range(4) if j != lbl)
        all_margins.append(correct - best_wrong)

all_margins = np.array(all_margins)

# Split margins into positives and negatives
positive_margins = all_margins[all_margins > 0]
negative_margins = all_margins[all_margins < 0]

plt.figure(figsize=(6, 4))
plt.hist(negative_margins, bins=20, color="red", alpha=0.7, label="Incorrect > Correct")
plt.hist(positive_margins, bins=20, color="green", alpha=0.7, label="Correct > Incorrect")

plt.axvline(0, linestyle="--", color="black")
plt.xlabel("Δ margin  (correct - nearest wrong)")
plt.ylabel("# questions")

title_acc = (all_margins > 0).mean() * 100
plt.title(f"Margin distribution (accuracy: {title_acc:.2f}%)")
plt.legend()
plt.tight_layout()

# Save the figure
plt.savefig(FIG_DIR + "/" + "margin_histogram.png", dpi=300)
plt.close()
print("✓ saved figures/margin_histogram.png")

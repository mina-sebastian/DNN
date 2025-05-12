from utils.base_model import ARC_TEST, ARC_TRAIN
from utils.datasets_class import MultipleChoiceCombinedDataset
from torch.utils.data import DataLoader, random_split
from collections import Counter
import torch

from utils.models_dnn import ArcMLP
from utils.pairwise_loss import PairwiseRankingLoss
from utils.train_util import test_model, train_and_evaluate
generator = torch.Generator().manual_seed(42)
import torch.optim as optim


models = (
    # ("llmic", 2560),
    # ("mgpt/mGPT-1.3B-romanian", 2048),
    # ("rollama/RoLlama2-7b-Instruct-2024-10-09", 4096),
    # ("llama/Llama-3.1-8B", 4096),
    ("google/gemma-3-1b-pt", 1152),
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16

pooling_strategies = ["last_token"]

histories = {}

def add_to_histories(model_id, strategy, history):
    if model_id not in histories:
        histories[model_id] = {}
    if strategy not in histories[model_id]:
        histories[model_id][strategy] = []
    histories[model_id][strategy].append(history)

for model_id, emb_dim in models:
    for strategy in pooling_strategies:
            print(f"Using model: {model_id} with strategy: {strategy}")

            full_dataset = MultipleChoiceCombinedDataset(
                csv_file=ARC_TRAIN,
                get_embedding=None,
                name=f'laroseda_train_{model_id}_{strategy}',
                save_interval=1
            )

            test_dataset = MultipleChoiceCombinedDataset(
                csv_file=ARC_TEST,
                get_embedding=None,
                name=f'laroseda_test_{model_id}_{strategy}',
                save_interval=1
            )

            print(f"Loaded {len(full_dataset)} samples.")
            print(f"Shape of full dataset: {full_dataset[0][0].shape}")

            print(f"Number of labels: {len(full_dataset.labels)}")
            print(f"Number of unique labels: {Counter(full_dataset.labels.tolist())}")

            train_ratio = 0.8
            train_size = int(train_ratio * len(full_dataset))
            val_size = len(full_dataset) - train_size

            train_dataset, val_dataset = random_split(
                full_dataset,
                lengths=[train_size, val_size],
                generator=generator
            )

            # print(f"Full dataset size: {Counter(full_dataset.labels)}")
            # print(f"Train dataset size: {Counter(train_dataset.dataset.labels[train_dataset.indices])}")
            # print(f"Val dataset size: {Counter(val_dataset.dataset.labels[val_dataset.indices])}")

            print(f"Shape of train dataset: {train_dataset.dataset[0][0].shape}")
            print(f"Shape of val dataset: {val_dataset.dataset[0][0].shape}")

            print(f"Train samples: {len(train_dataset)}")
            print(f"Val samples: {len(val_dataset)}")

            train_dataloader = DataLoader(
                    train_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=True)

            val_dataloader = DataLoader(
                    val_dataset, 
                    batch_size=BATCH_SIZE, 
                    shuffle=False)
            
            mlp_model = ArcMLP(
                input_dim=emb_dim,
                hidden_dim=emb_dim * 2,
            ).to(device)

            history, best_model = train_and_evaluate(
                model=mlp_model,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                criterion=PairwiseRankingLoss(),
                # optimizer = optim.AdamW(mlp_model.parameters(), lr=1e-4, weight_decay=1e-4),
                # optimizer=optim.Adam(mlp_model.parameters(), lr=1e-4),
                # optimizer=optim.SGD(self.mlp_model.parameters(), lr=1e-5, momentum=0.4),
                optimizer=optim.SGD(
                    mlp_model.parameters(),
                    lr=0.0001,
                    momentum=0.9,
                ),
                # use_scheduler=True,
                num_epochs=10,
                device=device,
                name=f'{model_id}_{strategy}_arc_final',
                save=True,
                is_binary=False,
                do_all_metrics=True,
                # one_input=True
            )
            add_to_histories(model_id, strategy, history)

            test_model(mlp_model, model_id, test_dataset, device=device, roarc=True)


# import numpy as np
# import matplotlib.pyplot as plt

# # ------------ CONFIG ---------------------------------------------------------
# metric_key   = "val_f1"        # what metric decides “best”
# curve_to_plot = "val_f1"       # which metric’s history to draw
# # -----------------------------------------------------------------------------

# best_per_model = {}  # model_id → (strategy, history)

# for model_id, strategy_dict in histories.items():
#     # gather the best F1 reached by each strategy
#     best_f1_per_strategy = {}
#     for strat, runs in strategy_dict.items():
#         # if you launched the same (model,strategy) multiple times, keep the best run
#         best_run_f1 = max( max(run[metric_key]) for run in runs )
#         # optionally remember which run gave it, if you care
#         best_f1_per_strategy[strat] = best_run_f1
    
#     # select the winning strategy
#     winning_strat = max(best_f1_per_strategy, key=best_f1_per_strategy.get)
#     # take the *single* run of that strategy that actually produced the top score
#     winning_history = max(
#         strategy_dict[winning_strat],
#         key=lambda h: max(h[metric_key])
#     )
#     best_per_model[model_id] = (winning_strat, winning_history)

# #reset plot
# plt.clf()
# plt.cla()

# # --------------- PLOT --------------------------------------------------------
# plt.figure(figsize=(10, 6))

# for model_id, (strat, history) in best_per_model.items():
#     y = history[curve_to_plot]
#     x = range(1, len(y) + 1)
#     label = f"{model_id} – {strat}"
#     plt.plot(x, y, marker="o", label=label)

# plt.title(f"Chosen history for every model ({curve_to_plot})")
# plt.xlabel("Epoch")
# plt.ylabel(curve_to_plot.replace('_', ' ').title())
# plt.grid(True, linestyle="--", alpha=0.5)
# plt.legend()
# plt.tight_layout()
# plt.savefig(f"best_histories_{curve_to_plot}.png", dpi=160)
# plt.show()
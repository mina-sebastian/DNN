from typing import List
import torch
from torch.utils.data import DataLoader
from datasets_class import TitleContentDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim

from collections import Counter

from models_dnn import DualInputMLP
from train_util import evaluate_on_threshold, get_test_probs_targets, plot_metrics, train_and_evaluate
device = "cuda"
model_id = "faur-ai/LLMic"

# HYPERPARAMS
BATCH_SIZE = 32

# tokenizer = AutoTokenizer.from_pretrained(model_id)
# model_llmic = AutoModel.from_pretrained(model_id).to(device)


def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model_llmic(**inputs)

    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    # attention masks is needed to ignore padding and special tokens
    attention_mask = inputs['attention_mask'].unsqueeze(-1)

    masked_hidden = hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    return mean_pooled.squeeze().cpu().numpy()

full_dataset = TitleContentDataset(
    csv_file="laroseda_train.csv",
    get_embedding=get_embedding,  
    type="train",
    name="mGPT-1-3B-romanian",                 
    emb_dim=2048,                
    save_interval=200            
)

test_dataset = TitleContentDataset(
    csv_file="laroseda_test.csv",
    get_embedding=get_embedding,
    type="test",
    name="mGPT-1-3B-romanian",
    emb_dim=2048,
    save_interval=200
)

train_ratio = 0.8
train_size = int(train_ratio * len(full_dataset))
val_size = len(full_dataset) - train_size

# Optional: set a manual seed for reproducible splits
generator = torch.Generator().manual_seed(42)

train_dataset, val_dataset = random_split(
    full_dataset,
    lengths=[train_size, val_size],
    generator=generator
)

#Couners for the dataset
print(f"Full dataset size: {Counter(full_dataset.labels)}")
print(f"Train dataset size: {Counter(train_dataset.dataset.labels[train_dataset.indices])}")
print(f"Val dataset size: {Counter(val_dataset.dataset.labels[val_dataset.indices])}")
print(f"Test dataset size: {Counter(test_dataset.labels)}")

print(f"Train samples: {len(train_dataset)}")
print(f"Val samples: {len(val_dataset)}")

# train_data, val_data = train_test_split(
#     train_dataset, test_size=0.2, random_state=42, shuffle=True)


# print(train_data[0])
# train_data, train_labels = train_data[:, :-1], train_data[:, -1].float()
# val_data, val_labels = val_data[:, :-1], val_data[:, -1].float()

# print(len(train_data), len(val_data))

# print("Train data:", train_data[0])
# print("Validation data:", val_data[0])

# print("Train data:", train_data[0])

# train_dataset = TensorDataset(
#     torch.tensor(train_data).float(),
#     torch.tensor(train_labels).float()
# )


print(train_dataset[0])
print(val_dataset[0])

train_dataloader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True)

val_dataloader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False)




llmic_mlp = DualInputMLP(input_dim=2048, hidden_dim=512).to(device)

history, best_model = train_and_evaluate(
    model=llmic_mlp,
    train_loader=train_dataloader,
    val_loader=val_dataloader,
    criterion=nn.BCEWithLogitsLoss(),
    optimizer=optim.Adam(llmic_mlp.parameters(), lr=1e-3),
    num_epochs=10,
    device=device
)

plot_metrics(history)

# # Save the model
torch.save(best_model.state_dict(), "llmic_mlp_model.pth")
# best_model = llmic_mlp
# best_model.load_state_dict(torch.load("llmic_mlp_model.pth"))
best_model.eval()
probs_default, targets_test = get_test_probs_targets(best_model, test_dataset, device=device)

rows = []
acc, prec, rec, f1 = evaluate_on_threshold(probs_default, targets_test, 0.5)
rows.append(["llmic", 0.5, acc, prec, rec, f1])




import pandas as pd

df = pd.DataFrame(rows, columns=["Model", "Threshold", "Accuracy", "Precision", "Recall", "F1"])
styled = df.style.highlight_max(axis=0, subset=["Accuracy", "Precision", "Recall", "F1"], color='red')

# show the table in a Jupyter Notebook
# styled
# Save the table to an HTML file
styled.to_html("metrics.html")

# model = train_dual_mlp(llmic_mlp, train_dataloader, 
#                         loss_criterion=nn.BCEWithLogitsLoss(), 
#                         optimizer=optim.Adam(llmic_mlp.parameters(), lr=1e-3), 
#                         epochs=10, 
#                         device=device)

# text1 = "Ana are mere"
# text2 = "Ion are pere pe care le-a luat de la magazin"

# text1_embedding = get_embedding(text1)
# text2_embedding = get_embedding(text2)
# print(text1_embedding.shape, text2_embedding.shape)
# print(text1_embedding, text2_embedding)

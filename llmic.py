print("Loading transformer")
from typing import List
from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer, AutoModel
print("Loading torch")
import torch
print("Loading numpy and pandas")
import numpy as np
import pandas as pd
print("Loading os")
import os
print("Loading Dataset")
from torch.utils.data import Dataset, DataLoader, TensorDataset
from datasets_class import TitleContentDataset
from torch.utils.data import random_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
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
    name="llmic",                 
    emb_dim=2560,                
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



def compute_accuracy(predictions: List[int], labels:List[int]) -> float:
    """
    Compute accuracy given the predictions of a binary classifier and the
    ground truth label.
    predictions: list of model predictions (0 or 1)
    labels: list of ground truth labels (0 or 1)
  """
    epoch_accuracy = len([1 for i in range(len(predictions)) if predictions[i] == labels[i]]) / len(predictions)
    return epoch_accuracy

def train_epoch(model, train_dataloader, loss_crt, optimizer, device, threshold=0.5):
    """
    model: Model object
    train_dataloader: DataLoader over the training dataset
    loss_crt: loss function object
    optimizer: Optimizer object
    device: torch.device('cpu) or torch.device('cuda')

    The function returns:
     - the epoch training loss, which is an average over the individual batch
       losses
     - the predictions made by the model
     - the labels
    """
    model.train()
    epoch_loss = 0.0
    num_batches = len(train_dataloader)
    predictions = []
    labels = []
    for title_batch, content_batch, batch_labels in train_dataloader:
         # Move data to the appropriate device
        title_batch = title_batch.to(device)
        content_batch = content_batch.to(device)

        batch_labels = batch_labels.to(device)
        batch_labels = batch_labels.float()

        # Forward pass
        output = model(title_batch, content_batch)
        output = output.squeeze(dim = 1)
        # change the result into a probability using sigmoid
        probs = torch.sigmoid(output)
        # get batch predictions
        batch_predictions = (probs >= threshold).float()
        predictions += batch_predictions.tolist()
        labels += batch_labels.tolist()

        # compute loss for current batch
        loss = loss_crt(output, batch_labels)

        # compute gradients
        loss.backward()

        # update parameters
        optimizer.step()

        # reset gradients
        optimizer.zero_grad()

        epoch_loss += loss.item()

    epoch_loss /= num_batches

    return epoch_loss, predictions, labels

def eval_epoch(model, validation_dataloader, loss_crt, device, threshold=0.5):
    """
    model: Model object
    val_dataloader: DataLoader over the validation dataset
    loss_crt: loss function object
    device: torch.device('cpu) or torch.device('cuda')

    The function returns:
     - the epoch validation loss, which is an average over the individual batch
       losses
     - the predictions made by the model
     - the labels
    """
    model.eval()
    epoch_loss = 0.0
    num_batches = len(validation_dataloader)
    predictions = []
    labels = []
    with torch.no_grad():
        for title_batch, content_batch, batch_labels  in validation_dataloader:
            title_batch = title_batch.to(device)
            content_batch = content_batch.to(device)
            batch_labels = batch_labels.to(device)
            
            batch_labels = batch_labels.float()
            output = model(title_batch, content_batch)
            output = output.squeeze(dim = 1)
            probs = torch.sigmoid(output)
            batch_predictions = (probs >= threshold).float()
            predictions += batch_predictions.tolist()
            labels += batch_labels.tolist()

            loss = loss_crt(output, batch_labels)
            epoch_loss += loss.item()

    epoch_loss /= num_batches

    return epoch_loss, predictions, labels

def train_dual_mlp(model, train_dataloader, loss_criterion, optimizer, epochs=10, device='cuda' ):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    train_f1_scores, val_f1_scores = [], []
    train_precisions, val_precisions = [], []
    train_recalls, val_recalls = [], []

    for epoch in range(epochs):
        train_epoch_loss, train_predictions, train_labels = train_epoch(
            model,
            train_dataloader,
            loss_criterion,
            optimizer,
            device,
            0.5
        )
        val_epoch_loss, val_predictions, val_labels = eval_epoch(
            model,
            val_dataloader,
            loss_criterion,
            device
        )
        train_acc = compute_accuracy(train_predictions, train_labels)
        val_acc = compute_accuracy(val_predictions, val_labels)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        # train_precision, train_recall, train_f1_score, _ = precision_recall_fscore_support(train_labels, train_predictions, average='binary', zero_division=0)
        # train_precisions.append(train_precision)
        # train_recalls.append(train_recall)
        # train_f1_scores.append(train_f1_score)
        # val_precision, val_recall, val_f1_score, _ = precision_recall_fscore_support(val_labels, val_predictions, average='binary', zero_division=0)
        # val_precisions.append(val_precision)
        # val_recalls.append(val_recall)
        # val_f1_scores.append(val_f1_score)

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_epoch_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
            #   f"Train F1: {train_f1_score:.4f} | "
              f"Val Loss: {val_epoch_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | ")
            #   f"Val F1: {val_f1_score:.4f}")
        

    print(classification_report(val_labels, val_predictions, target_names=["Negative", "Positive"]))
    return model

llmic_mlp = DualInputMLP(input_dim=2560, hidden_dim=512, num_classes=1).to(device)

model = train_dual_mlp(llmic_mlp, train_dataloader, 
                        loss_criterion=nn.BCEWithLogitsLoss(), 
                        optimizer=optim.Adam(llmic_mlp.parameters(), lr=1e-3), 
                        epochs=10, 
                        device=device)

# text1 = "Ana are mere"
# text2 = "Ion are pere pe care le-a luat de la magazin"

# text1_embedding = get_embedding(text1)
# text2_embedding = get_embedding(text2)
# print(text1_embedding.shape, text2_embedding.shape)
# print(text1_embedding, text2_embedding)

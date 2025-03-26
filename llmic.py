from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datasets_dnn import dataset_laroseda
import pandas as pd
import os
from torch.utils.data import Dataset

device = "cuda"
model_id = "faur-ai/LLMic"

print(dataset_laroseda['train'][2]['title'], "-", dataset_laroseda['train'][2]['content'], "label:", dataset_laroseda['train'][2]['starRating'])

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)


def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_dim)
    # attention masks is needed to ignore padding and special tokens
    attention_mask = inputs['attention_mask'].unsqueeze(-1)

    masked_hidden = hidden_states * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts

    return mean_pooled.squeeze().cpu().numpy()


class TitleContentDataset(Dataset):
    def __init__(
        self,
        csv_file,
        title_col="title",
        content_col="content",
        label_col="starRating",
        title_emb_file="title_embeddings.npy",
        content_emb_file="content_embeddings.npy",
        label_file="labels.npy"
    ):
        """
        :param csv_file: Path to CSV with columns [title_col, content_col, label_col].
        :param title_col: Name of the 'title' column in CSV.
        :param content_col: Name of the 'content' column in CSV.
        :param label_col: Name of the label column in CSV (e.g., starRating).
        :param title_emb_file: Where to store/load title embeddings (.npy).
        :param content_emb_file: Where to store/load content embeddings (.npy).
        :param label_file: Where to store/load labels (.npy).
        """

        self.df = pd.read_csv(csv_file)
        
        # If you already computed embeddings before, just load them from disk
        if os.path.exists(title_emb_file) and os.path.exists(content_emb_file) and os.path.exists(label_file):
            print("Loading precomputed embeddings from disk...")
            self.title_embs = np.load(title_emb_file)
            self.content_embs = np.load(content_emb_file)
            self.labels = np.load(label_file)
        else:
            print("No precomputed embeddings found. Computing and saving them now...")
            title_embeddings = []
            content_embeddings = []
            labels = []

            # Iterate over each row in the CSV
            for idx, row in self.df.iterrows():
                title_text = str(row[title_col])
                content_text = str(row[content_col])
                label_value = row[label_col]   # e.g., 1,2,4,5 or 0/1 depending on data

                # Compute embeddings
                t_emb = get_embedding(title_text)
                c_emb = get_embedding(content_text)

                title_embeddings.append(t_emb)
                content_embeddings.append(c_emb)
                labels.append(label_value)

            # Convert to numpy arrays
            self.title_embs = np.array(title_embeddings)
            self.content_embs = np.array(content_embeddings)
            self.labels = np.array(labels)

            # Save for future
            np.save(title_emb_file, self.title_embs)
            np.save(content_emb_file, self.content_embs)
            np.save(label_file, self.labels)

        # Optional: convert labels to numeric classes if needed:
        # e.g., map {1,2} -> 0 (negative), {4,5} -> 1 (positive)
        # Or just store them as-is for a multiclass approach

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Title & content embedding are numpy arrays
        title_emb = self.title_embs[idx]
        content_emb = self.content_embs[idx]
        label = self.labels[idx]
        
        # Optionally convert to torch tensors here (many do it in collate_fn)
        title_emb = torch.tensor(title_emb, dtype=torch.float32)
        content_emb = torch.tensor(content_emb, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)  # or float, depending on your criterion

        return (title_emb, content_emb), label

text1 = "Ana are mere"
text2 = "Ion are pere pe care le-a luat de la magazin"

text1_embedding = get_embedding(text1)
text2_embedding = get_embedding(text2)
print(text1_embedding.shape, text2_embedding.shape)
print(text1_embedding, text2_embedding)

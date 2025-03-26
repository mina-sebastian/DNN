from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import os
from torch.utils.data import Dataset
from tqdm import tqdm

# Load the dataset
# dataset_laroseda = load_dataset("universityofbucharest/laroseda")


class TitleContentDataset(Dataset):
    def __init__(
        self,
        csv_file,
        get_embedding,
        name,
        title_col="title",
        content_col="content",
        label_col="starRating"
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

        title_emb_file = f"embeddings/{name}_title_embeddings.npy"
        content_emb_file = f"embeddings/{name}_content_embeddings.npy"
        label_file = f"embeddings/{name}_labels.npy"
        
        # If you already computed embeddings before, just load them from disk
        if os.path.exists(title_emb_file) and os.path.exists(content_emb_file) and os.path.exists(label_file):
            print("Loading precomputed embeddings from disk...")
            self.title_embs = np.load(title_emb_file)
            self.content_embs = np.load(content_emb_file)
            self.labels = np.load(label_file)
        else:
            #make dirs
            os.makedirs("embeddings", exist_ok=True)

            print("No precomputed embeddings found. Computing and saving them now...")
            title_embeddings = []
            content_embeddings = []
            labels = []

            # Iterate over each row in the CSV with progress bar
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Computing embeddings"):
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


# train_df = dataset_laroseda["train"].to_pandas()
# test_df = dataset_laroseda["test"].to_pandas()

# # Save them separately
# train_df.to_csv("laroseda_train.csv", index=False)
# test_df.to_csv("laroseda_test.csv", index=False)

# # Or combine into one CSV if you want everything in a single file
# all_df = pd.concat([train_df, test_df], ignore_index=True)
# all_df.to_csv("laroseda_all.csv", index=False)

# # Load the combined CSV
# all_df = pd.read_csv("laroseda_all.csv")


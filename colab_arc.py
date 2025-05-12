from collections import deque
import os
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from utils.base_model import ARC_TEST, ARC_TRAIN, LAROSEDA_TEST, LAROSEDA_TRAIN

HUGGINGFACE_TOKEN = "TOKEN"

class TitleContentDataset(Dataset):
    def __init__(
        self,
        csv_file,
        get_embedding,
        name,
        type="train",
        title_col="title",
        content_col="content",
        label_col="starRating",
        emb_dim=2560,
        save_interval=100
    ):
        """
        :param csv_file: Path to CSV with columns [title_col, content_col, label_col].
        :param get_embedding: A function(text) -> vector, returning e.g. a 2560-d numpy array.
        :param name: String prefix for the embedding files
        :param emb_dim: Dimension of embeddings (2560 by default)
        :param save_interval: How many rows to process before caching partial progress
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.num_samples = len(self.df)
        self.emb_dim = emb_dim
        self.title_col = title_col
        self.content_col = content_col
        self.label_col = label_col
        self.save_interval = save_interval

        # Paths for saving
        self.title_emb_file = f"embeddings/{name}_{type}_title_embeddings.npy"
        self.content_emb_file = f"embeddings/{name}_{type}_content_embeddings.npy"
        self.label_file = f"embeddings/{name}_{type}_labels.npy"

        if not os.path.exists(os.path.dirname(self.title_emb_file)):
            os.makedirs(os.path.dirname(self.title_emb_file), exist_ok=True)
        
        if not os.path.exists(os.path.dirname(self.content_emb_file)):
            os.makedirs(os.path.dirname(self.content_emb_file), exist_ok=True)
        
        if not os.path.exists(os.path.dirname(self.label_file)):
            os.makedirs(os.path.dirname(self.label_file), exist_ok=True)

        # Make sure directory exists
        os.makedirs("embeddings", exist_ok=True)

        # How many rows have already been written to disk?
        self.current_count = self._read_existing_count()

        # We'll temporarily hold new rows in memory until we have `save_interval` or we finish.
        self.title_buffer = []
        self.content_buffer = []
        self.labels_buffer = []

        if self.current_count < self.num_samples:
            # Create TQDM bars
            self.main_pbar = tqdm(
                total=self.num_samples,
                desc=f'Processing {self.csv_file}',
                initial=self.current_count,
                position=1,
                leave=True
            )
            self.log_pbar = tqdm(
                total=5,
                position=0,  # below main bar
                bar_format="{desc}",
                leave=False
            )
            self.log_queue = deque(maxlen=5)

            if self.current_count > 0:
                self._log_message(f"Detected {self.current_count}/{self.num_samples} already on disk. Resuming...")

            self._compute_and_append_embeddings(get_embedding, start=self.current_count)

            # Close bars
            self.main_pbar.close()
            self.log_pbar.close()

        self._load_full_arrays()

    def _log_message(self, msg: str):
        """
        Insert `msg` at the *top* of a deque and redraw the log bar:
        - Newest line is on top
        - Maximum 5 lines
        """
        self.log_queue.appendleft(msg)
        desc_str = "\n".join(self.log_queue)
        self.log_pbar.set_description_str(desc_str)
        self.log_pbar.refresh()

    def _read_existing_count(self):
        """
        Returns how many rows of embeddings are already stored on disk.
        If no files exist, or shapes mismatch, return 0.
        """
        if (os.path.exists(self.title_emb_file) and
            os.path.exists(self.content_emb_file) and
            os.path.exists(self.label_file)):
            tmp_title = np.load(self.title_emb_file)
            tmp_content = np.load(self.content_emb_file)
            tmp_labels = np.load(self.label_file)

            if (tmp_title.shape[0] == tmp_content.shape[0] == tmp_labels.shape[0]):
                return tmp_title.shape[0]
        
        return 0

    def _compute_and_append_embeddings(self, get_embedding, start=0):
        """
        From `start` to `num_samples`, compute embeddings, store in small buffers.
        Every `save_interval` items, append them to disk (extend the .npy arrays).
        """
        for idx in range(start, self.num_samples):
            row = self.df.iloc[idx]
            title_text = str(row[self.title_col])
            content_text = str(row[self.content_col])
            label_value = row[self.label_col]

            # Compute embeddings
            t_emb = get_embedding(title_text)
            c_emb = get_embedding(content_text)

            self.title_buffer.append(t_emb)
            self.content_buffer.append(c_emb)
            self.labels_buffer.append(label_value)

            # Update the main progress bar
            self.main_pbar.update(1)

            # If we hit the interval, append to disk
            if (len(self.title_buffer) % self.save_interval) == 0:
                self._append_to_disk()
                self.main_pbar.set_postfix_str(f"Appended up to row {idx+1} on disk")

        # If anything is left in the buffers, append one last time
        if len(self.title_buffer) > 0:
            self._append_to_disk()
            self._log_message("Appended final chunk to disk.")

    def _append_to_disk(self):
        """
        Append the current buffers to each .npy file.
        Then clear the buffers, and update self.current_count.
        """
        # Convert buffers to arrays
        t_array = np.array(self.title_buffer, dtype=np.float32)   # shape=(chunk_size, emb_dim)
        c_array = np.array(self.content_buffer, dtype=np.float32) # shape=(chunk_size, emb_dim)
        l_array = np.array(self.labels_buffer, dtype=np.int64)    # shape=(chunk_size,)

        # Read existing arrays (if any)
        if os.path.exists(self.title_emb_file):
            existing_t = np.load(self.title_emb_file)
            existing_c = np.load(self.content_emb_file)
            existing_l = np.load(self.label_file)

            # Concat
            new_t = np.concatenate([existing_t, t_array], axis=0)
            new_c = np.concatenate([existing_c, c_array], axis=0)
            new_l = np.concatenate([existing_l, l_array], axis=0)
        else:
            # No existing data, so the new arrays are the entire set
            new_t = t_array
            new_c = c_array
            new_l = l_array

        # Save them back
        np.save(self.title_emb_file, new_t)
        np.save(self.content_emb_file, new_c)
        np.save(self.label_file, new_l)

        # Clear buffers
        self.title_buffer.clear()
        self.content_buffer.clear()
        self.labels_buffer.clear()

        # Update count
        self.current_count = new_t.shape[0]

    def _load_full_arrays(self):
        """
        After all is done, load the final arrays from disk into memory
        so we can implement __getitem__ easily.
        """
        # It's possible no data is on disk if the CSV was empty
        if not os.path.exists(self.title_emb_file):
            # Edge case: if the CSV is empty?
            self.title_embs = np.empty((0, self.emb_dim), dtype=np.float32)
            self.content_embs = np.empty((0, self.emb_dim), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int64)
            return

        self.title_embs = np.load(self.title_emb_file)
        self.content_embs = np.load(self.content_emb_file)
        self.labels = np.load(self.label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Convert to torch Tensors if desired
        title_emb = torch.tensor(self.title_embs[idx], dtype=torch.float32)
        content_emb = torch.tensor(self.content_embs[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        label = torch.where((label == 1) | (label == 2), 0, 1)
        return [title_emb, content_emb, label]


model_name = "roarc"
strategy = "mean"

def get_embedding(model, tokenizer, text, strategy: str = "mean"):
    """
    Return a single vector for *text* using one of four pooling strategies:
        ├─ "mean"        : mean‑pool last_hidden_state
        ├─ "max"         : max‑pool last_hidden_state
        ├─ "last_token"  : hidden state of the final real token
        └─ "echo"        : mean‑pool *input* embeddings (a.k.a. “echo”)
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=False,
        max_length=512
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    hidden_states   = outputs.last_hidden_state               # (b, L, d)
    attention_mask  = inputs["attention_mask"].unsqueeze(-1)  # (b, L, 1)

    # ───────────────────────────── mean ─────────────────────────────
    if strategy == "mean":
        masked = hidden_states * attention_mask
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)

    # ───────────────────────────── max ──────────────────────────────
    elif strategy == "max":
        masked = hidden_states * attention_mask + (1.0 - attention_mask) * -1e9
        pooled, _ = masked.max(dim=1)

    # ─────────────────────────── last token ─────────────────────────
    elif strategy == "last_token":
        seq_lengths = attention_mask.squeeze(-1).sum(dim=1)          # (#tokens)
        idx = (seq_lengths - 1).long().unsqueeze(-1).unsqueeze(-1)   # (b,1,1)
        idx = idx.expand(-1, 1, hidden_states.size(-1))              # (b,1,d)
        pooled = hidden_states.gather(1, idx).squeeze(1)             # (b,d)

    # ───────────────────────────── echo ─────────────────────────────
    elif strategy == "echo":
        with torch.no_grad():  # ← safe context
            input_embs = model.get_input_embeddings()(inputs["input_ids"])
            masked = input_embs * attention_mask
            pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)
    # ───────────────────────── unknown ──────────────────────────────
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return pooled.squeeze().cpu().numpy()

from transformers import AutoTokenizer, AutoModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = None
tokenizer = None
strategy = "mean"

model_ids = [
    # "ai-forever/mGPT-1.3B-romanian",
    # "faur-ai/LLMic",
    # "OpenLLM-Ro/RoLlama2-7b-Instruct-2024-10-09",
    # "meta-llama/Meta-Llama-3.1-8B",
    "google/gemma-3-1b-pt"
]


pooling_strategies = ["last_token"]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_embeddings_model(text: str) -> np.ndarray:
    """Get the embedding for a single text using the specified model and strategy."""
    return get_embedding(model, tokenizer, text, strategy=strategy)

for model_id in model_ids:
    model = AutoModel.from_pretrained(model_id,
                           token=HUGGINGFACE_TOKEN           
                                      ).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=HUGGINGFACE_TOKEN)
    for strategy in pooling_strategies:
        print(f"Using model: {model_id} with strategy: {strategy}")

        TitleContentDataset(
            csv_file=LAROSEDA_TRAIN,
            get_embedding=get_embeddings_model,
            name=f'laroseda_train_{model_id}_{strategy}',
            save_interval=1
        )

        TitleContentDataset(
            csv_file=LAROSEDA_TEST,
            get_embedding=get_embeddings_model,
            name=f'laroseda_test_{model_id}_{strategy}',
            save_interval=1
        )


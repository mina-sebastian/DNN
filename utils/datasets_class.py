import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import deque


########## FOR LAROSEDA ##########
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
































########## FOR ROARC ##########

class MultipleChoicePointwiseCached(Dataset):
    def __init__(
        self,
        csv_file: str,
        get_embedding,          # function(text) -> numpy array
        question_col: str = "instruction",
        correct_col: str = "answer",
        options_cols = ("option_a", "option_b", "option_c", "option_d"),
        emb_dim=768,            # embedding size returned by get_embedding
        name="mchoice_cache",  # prefix for .npy files
        save_interval=100       # how many samples to buffer before appending
    ):
        """
        Multiple-choice dataset with partial caching for pointwise approach:

        - Each row in the CSV is a question with 4 options => 4 sub-examples.
        - Label = 1 if that option is correct, else 0.
        - If the cache .npy files exist but are incomplete, we resume from where we left off.
        - If they don't exist, we create them from scratch.

        :param csv_file: path to CSV with columns [question_col, correct_col, option_a, ...]
        :param get_embedding: function(text) -> np.array of shape (emb_dim,)
        :param question_col: column name for the question text
        :param correct_col: column name for the correct letter (e.g. "A")
        :param options_cols: tuple/list of columns for the answer options
        :param emb_dim: size of each embedding vector
        :param name: prefix for .npy cache files
        :param save_interval: how many sub-examples to buffer before saving to disk
        """
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.num_rows = len(self.df)

        self.get_embedding = get_embedding
        self.question_col = question_col
        self.correct_col = correct_col
        self.options_cols = options_cols
        self.num_options = len(options_cols)
        self.emb_dim = emb_dim
        self.save_interval = save_interval

        # total_subexamples = e.g. 4 * num_rows
        self.total_subexamples = self.num_rows * self.num_options

        # Filenames for embeddings and labels
        self.emb_file = f"{name}_embeddings.npy"
        self.lbl_file = f"{name}_labels.npy"

        print(f"Embeddings will be cached in {self.emb_file} and {self.lbl_file}")


        # We'll store the final arrays in memory
        self.samples = None
        self.labels = None

        # Check how many sub-examples are already cached
        self.current_count = self._read_existing_count()

        # If we haven't cached them all, continue computing
        # if self.current_count < self.total_subexamples:
        #     print(f"Detected {self.current_count}/{self.total_subexamples} already on disk. Resuming...")
        #     self._compute_and_append_embeddings()

        # Finally, load the full arrays into memory
        self._load_full_arrays()

    def _read_existing_count(self):
        """
        Returns how many sub-examples are currently cached in the .npy files.
        If the files don't exist, or the shapes mismatch, we return 0.
        """
        if os.path.exists(self.emb_file) and os.path.exists(self.lbl_file):
            emb_data = np.load(self.emb_file)
            lbl_data = np.load(self.lbl_file)
            # sub-examples must match in length
            if emb_data.shape[0] == lbl_data.shape[0] and emb_data.shape[1] == self.emb_dim:
                return emb_data.shape[0]
        return 0

    def _compute_and_append_embeddings(self):
        """
        Compute embeddings for all sub-examples from current_count..(total_subexamples-1),
        buffering them in memory and appending them to .npy files in chunks of `save_interval`.
        """
        # We'll buffer new samples/labels here
        new_embs = []
        new_labels = []

        # Loop from the sub-example we left off up to total_subexamples-1
        for sub_idx in tqdm(range(self.current_count, self.total_subexamples), desc="Processing embeddings"):
            row_idx = sub_idx // self.num_options  # which row in df
            opt_idx = sub_idx % self.num_options   # which option in that row

            row = self.df.iloc[row_idx]
            question_text = str(row[self.question_col])
            correct_letter = str(row[self.correct_col]).strip().upper()

            # e.g. 'A','B','C','D' (or however many you have)
            letter = ["A","B","C","D","E"][:self.num_options][opt_idx]
            option_text = str(row[self.options_cols[opt_idx]])

            # label = 1 if letter matches correct_letter
            label = 1 if letter == correct_letter else 0

            # combine text for embedding
            combined_text = f"Q: {question_text} A: {option_text}"
            emb = self.get_embedding(combined_text)

            new_embs.append(emb)
            new_labels.append(label)

            # If we've buffered enough, flush to disk
            if len(new_embs) >= self.save_interval:
                self._append_to_disk(new_embs, new_labels)
                new_embs.clear()
                new_labels.clear()

        # flush any remainder
        if len(new_embs) > 0:
            self._append_to_disk(new_embs, new_labels)
            new_embs.clear()
            new_labels.clear()

    def _append_to_disk(self, emb_list, lbl_list):
        """
        Takes the new embeddings and labels in emb_list/lbl_list,
        appends them to the .npy files, and updates self.current_count.
        """
        new_emb_array = np.array(emb_list, dtype=np.float32)  # shape: (chunk_size, emb_dim)
        new_lbl_array = np.array(lbl_list, dtype=np.int64)    # shape: (chunk_size,)

        if os.path.exists(self.emb_file) and os.path.exists(self.lbl_file):
            # load existing
            old_emb = np.load(self.emb_file)
            old_lbl = np.load(self.lbl_file)
            # concat
            emb_out = np.concatenate([old_emb, new_emb_array], axis=0)
            lbl_out = np.concatenate([old_lbl, new_lbl_array], axis=0)
        else:
            # no existing data
            emb_out = new_emb_array
            lbl_out = new_lbl_array

        # save them back
        np.save(self.emb_file, emb_out)
        np.save(self.lbl_file, lbl_out)

        # update current_count
        self.current_count = emb_out.shape[0]

    def _load_full_arrays(self):
        """
        After everything is computed, load the final arrays from disk into memory
        to allow __getitem__ usage.
        """
        if not os.path.exists(self.emb_file) or not os.path.exists(self.lbl_file):
            # nothing saved
            self.samples = np.empty((0, self.emb_dim), dtype=np.float32)
            self.labels = np.empty((0,), dtype=np.int64)
            return

        self.samples = np.load(self.emb_file)
        self.labels = np.load(self.lbl_file)

    def __len__(self):
        return len(self.labels)  # or self.samples.shape[0]

    def __getitem__(self, idx):
        x = self.samples[idx]  # shape = (emb_dim,)
        y = self.labels[idx]
        # convert to torch tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        return x_tensor, y_tensor

class MultipleChoiceSeparatedDataset(Dataset):
    def __init__(
        self,
        csv_file,
        get_embedding,
        name,
        question_col="instruction",
        correct_col="answer",
        options_cols=("option_a", "option_b", "option_c", "option_d"),
        emb_dim=768,
        save_interval=100
    ):
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.num_samples = len(self.df)
        self.emb_dim = emb_dim
        self.save_interval = save_interval

        self.question_col = question_col
        self.correct_col = correct_col
        self.options_cols = options_cols
        self.num_options = len(options_cols)

        self.emb_dir = "embeddings/roarc"
        os.makedirs(self.emb_dir, exist_ok=True)

        # File paths
        self.q_emb_file = f"{self.emb_dir}/{name}_questions.npy"
        self.opt_emb_files = [
            f"{self.emb_dir}/{name}_option_{chr(97+i)}.npy" for i in range(self.num_options)
        ]
        self.label_file = f"{self.emb_dir}/{name}_labels.npy"

        # Count already processed samples
        self.current_count = self._read_existing_count()

        # if self.current_count < self.num_samples:
        #     print(f"Detected {self.current_count}/{self.num_samples} already on disk. Resuming...")
        #     self._compute_and_cache(get_embedding)

        self._load_full_arrays()

    def _read_existing_count(self):
        if (os.path.exists(self.q_emb_file)
            and all(os.path.exists(f) for f in self.opt_emb_files)
            and os.path.exists(self.label_file)):
            q = np.load(self.q_emb_file)
            o = [np.load(f) for f in self.opt_emb_files]
            l = np.load(self.label_file)
            if all(arr.shape[0] == q.shape[0] == l.shape[0] for arr in o):
                return q.shape[0]
        return 0

    def _compute_and_cache(self, get_embedding):
        q_buffer = []
        opt_buffers = [[] for _ in range(self.num_options)]
        label_buffer = []

        for idx in tqdm(range(self.current_count, self.num_samples), desc=f'Embedding {self.csv_file} MCQs'):
            row = self.df.iloc[idx]

            q_emb = get_embedding(str(row[self.question_col]))
            q_buffer.append(q_emb)

            correct_letter = str(row[self.correct_col]).strip().upper()
            correct_index = ord(correct_letter) - ord('A')

            for i, opt_col in enumerate(self.options_cols):
                opt_emb = get_embedding(str(row[opt_col]))
                opt_buffers[i].append(opt_emb)

            label_buffer.append(correct_index)

            if len(q_buffer) >= self.save_interval:
                self._append_to_disk(q_buffer, opt_buffers, label_buffer)
                q_buffer.clear()
                label_buffer.clear()
                for buffer in opt_buffers:
                    buffer.clear()

        if len(q_buffer) > 0:
            self._append_to_disk(q_buffer, opt_buffers, label_buffer)

    def _append_to_disk(self, q, opts, labels):
        q_arr = np.array(q, dtype=np.float32)
        o_arrs = [np.array(opt, dtype=np.float32) for opt in opts]
        l_arr = np.array(labels, dtype=np.int64)

        if os.path.exists(self.q_emb_file):
            q_old = np.load(self.q_emb_file)
            o_old = [np.load(f) for f in self.opt_emb_files]
            l_old = np.load(self.label_file)

            q_combined = np.concatenate([q_old, q_arr])
            o_combined = [np.concatenate([o, n]) for o, n in zip(o_old, o_arrs)]
            l_combined = np.concatenate([l_old, l_arr])
        else:
            q_combined = q_arr
            o_combined = o_arrs
            l_combined = l_arr

        np.save(self.q_emb_file, q_combined)
        for i, f in enumerate(self.opt_emb_files):
            np.save(f, o_combined[i])
        np.save(self.label_file, l_combined)

        self.current_count = q_combined.shape[0]

    def _load_full_arrays(self):
        self.q_embs = np.load(self.q_emb_file)
        self.opt_embs = [np.load(f) for f in self.opt_emb_files]
        self.labels = np.load(self.label_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        q = torch.tensor(self.q_embs[idx], dtype=torch.float32)
        opts = torch.stack([torch.tensor(o[idx], dtype=torch.float32) for o in self.opt_embs])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return q, opts, label









import os
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultipleChoiceCombinedDataset(Dataset):
    """MCQ dataset that keeps **only** the embeddings of
    `(question + option_i)` for *i ∈ [0, num_options).*  
    Stand‑alone question vectors are **not** stored – exactly as requested.

    Parameters
    ----------
    csv_file : str
        Path to the CSV with the columns
        ``question_col``, ``options_cols``, and ``correct_col``.
    get_embedding : Callable[[str], np.ndarray]
        Function that returns a 1‑D NumPy array for a given text.
    name : str
        Prefix for the `.npy` files persisted under ``embeddings/roarc/combined``.
    sep : str, default " \n "
        Delimiter inserted between question and option when composing the text.
    save_interval : int, default 100
        After how many rows to flush intermediate results to disk.
    """

    def __init__(
        self,
        csv_file: str,
        get_embedding: Callable[[str], np.ndarray],
        name: str,
        question_col: str = "instruction",
        correct_col: str = "answer",
        options_cols: Sequence[str] = ("option_a", "option_b", "option_c", "option_d"),
        emb_dim: int = 768,
        save_interval: int = 100,
        sep: str = " \n ",
    ) -> None:
        super().__init__()
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file).reset_index(drop=True)
        self.num_samples = len(self.df)
        self.emb_dim = emb_dim
        self.save_interval = save_interval
        self.sep = sep

        self.question_col = question_col
        self.correct_col = correct_col
        self.options_cols = tuple(options_cols)
        self.num_options = len(self.options_cols)

        # Cache directory & file templates
        self.emb_dir = "embeddings/roarc/combined"
        os.makedirs(self.emb_dir, exist_ok=True)

        self.q_emb_file = f"{self.emb_dir}/{name}_questions.npy"
        self.opt_emb_files = [
            f"{self.emb_dir}/{name}_q_plus_option_{chr(97 + i)}.npy" for i in range(self.num_options)
        ]
        self.label_file = f"{self.emb_dir}/{name}_labels.npy"

        # Resume if partial cache exists
        self.current_count = self._read_existing_count()
        if self.current_count < self.num_samples:
            print(
                f"Embedding {self.csv_file}: {self.current_count}/{self.num_samples} rows cached. Resuming…"
            )
            self._compute_and_cache(get_embedding)

        self._load_full_arrays()

    # ------------------------------------------------------------------
    # Disk IO helpers
    # ------------------------------------------------------------------
    def _read_existing_count(self) -> int:
        """Returns rows already cached (0 if none/inconsistent)."""
        if os.path.exists(self.q_emb_file) and all(os.path.exists(p) for p in self.opt_emb_files):
            q = np.load(self.q_emb_file, mmap_mode="r")
            opts = [np.load(p, mmap_mode="r") for p in self.opt_emb_files]
            labels = np.load(self.label_file, mmap_mode="r")
            if all(o.shape[0] == q.shape[0] == labels.shape[0] for o in opts):
                return q.shape[0]
        return 0

    def _compute_and_cache(self, get_embedding: Callable[[str], np.ndarray]) -> None:
        # Buffers
        q_buffer: list[np.ndarray] = []
        opt_buffers: list[list[np.ndarray]] = [[] for _ in range(self.num_options)]
        label_buffer: list[int] = []

        for idx in tqdm(range(self.current_count, self.num_samples), desc="Embedding rows"):
            row = self.df.iloc[idx]
            q_text = str(row[self.question_col])
            q_emb = get_embedding(q_text)
            q_buffer.append(q_emb)

            # Correct label
            correct_letter = str(row[self.correct_col]).strip().upper()
            correct_index = ord(correct_letter) - ord("A")
            label_buffer.append(correct_index)

            # Embed every combined text
            for i, opt_col in enumerate(self.options_cols):
                combined_text = f"{q_text}{self.sep}{str(row[opt_col])}"
                emb = get_embedding(combined_text)
                opt_buffers[i].append(emb)

            # Flush periodically
            if len(label_buffer) >= self.save_interval:
                self._append_to_disk(q_buffer, opt_buffers, label_buffer)
                label_buffer.clear()
                for buf in opt_buffers:
                    buf.clear()
                q_buffer.clear()

        # Flush remaining
        if label_buffer:
            self._append_to_disk(q_buffer, opt_buffers, label_buffer)

    def _append_to_disk(self, questions: list[np.ndarray], opts: list[list[np.ndarray]], labels: list[int]) -> None:
        o_arrs = [np.asarray(buf, dtype=np.float32) for buf in opts]
        l_arr = np.asarray(labels, dtype=np.int64)
        q_arr = np.asarray(questions, dtype=np.float32)

        if os.path.exists(self.label_file):
            o_old = [np.load(p) for p in self.opt_emb_files]
            l_old = np.load(self.label_file)
            q_old = np.load(self.q_emb_file)

            o_combined = [np.concatenate([old, new]) for old, new in zip(o_old, o_arrs)]
            l_combined = np.concatenate([l_old, l_arr])
            q_combined = np.concatenate([q_old, q_arr])
        else:
            o_combined = o_arrs
            l_combined = l_arr
            q_combined = q_arr


        for path, arr in zip(self.opt_emb_files, o_combined):
            np.save(path, arr)
        np.save(self.label_file, l_combined)
        np.save(self.q_emb_file, q_combined)

        self.current_count = l_combined.shape[0]

    def _load_full_arrays(self) -> None:
        self.opt_embs = [torch.from_numpy(np.load(p)) for p in self.opt_emb_files]
        self.labels = torch.from_numpy(np.load(self.label_file))
        self.q_embs = torch.from_numpy(np.load(self.q_emb_file))

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # type: ignore[override]
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Returns ``(opt_embs, label)``.

        * ``opt_embs`` – tensor of shape ``(num_options, emb_dim)``.
        * ``label`` – scalar ``torch.long`` index of the correct option (0‑based).
        """
        question = self.q_embs[idx]  # (emb_dim,)
        opts = torch.stack([o[idx] for o in self.opt_embs])  # (num_options, emb_dim)
        label = self.labels[idx]
        return question, opts, label

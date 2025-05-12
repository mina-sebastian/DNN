
# device = "cuda"
# import os, gc, math, re
# from collections import Counter
# import numpy as np
# import pandas as pd
# from tqdm.auto import tqdm

# import torch
# from transformers import AutoTokenizer, AutoModel

# # Kaggle paths to the LAROSeDa CSVs
# TRAIN_CSV = "data/laroseda/laroseda_train.csv"
# TEST_CSV  = "data/laroseda/laroseda_test.csv"

# # ────────────────────────────────
# FLAG_DONE = {
# ("Meta-Llama-3-1-8B", "mean", "train"),
# ("Meta-Llama-3-1-8B", "mean", "test"),
# ("Meta-Llama-3-1-8B", "last_token", "train"),
# ("Meta-Llama-3-1-8B", "last_token", "test"),
# }

# STATUS = {}

# # ────────────────────────────────

# try:
#     from transformers import BitsAndBytesConfig          
#     _bnb_ok = True
# except Exception:
#     _bnb_ok = False                            

# # ============================================================
# # 1. Small helpers
# # ============================================================
# def _ensure_dir(path: str):
#     """mkdir -p for any (sub)folder."""
#     os.makedirs(path, exist_ok=True)

# def _infer_hidden_size(model):
#     """Tries a few common names; raises if none is found."""
#     for attr in ("hidden_size", "n_embd", "d_model"):
#         if hasattr(model.config, attr):
#             return int(getattr(model.config, attr))
#     raise RuntimeError("Could not determine hidden size for this model.")

# def _emb_files(model_name: str, split: str, strategy: str, field: str):
#     """
#     Returns e.g.  
#     embeddings/LLMic/laroseda_train_LLMic_mean_title.npy
#     """
#     fld = f"embeddings/laroseda/{model_name}"
#     _ensure_dir(fld)
#     return f"{fld}/laroseda_{split}_{model_name}_{strategy}_{field}.npy"


# def get_embedding(model, tokenizer, text, strategy: str = "mean"):
#     """
#     Return a single vector for *text* using one of four pooling strategies:
#         ├─ "mean"        : mean‑pool last_hidden_state
#         ├─ "max"         : max‑pool last_hidden_state
#         ├─ "last_token"  : hidden state of the final real token
#         └─ "echo"        : mean‑pool *input* embeddings (a.k.a. “echo”)
#     """

#     inputs = tokenizer(
#         text,
#         return_tensors="pt",
#         truncation=True,
#         padding=False,
#         max_length=512
#     ).to(model.device)

#     with torch.no_grad():
#         outputs = model(**inputs)

#     hidden_states   = outputs.last_hidden_state               # (b, L, d)
#     attention_mask  = inputs["attention_mask"].unsqueeze(-1)  # (b, L, 1)

#     # ───────────────────────────── mean ─────────────────────────────
#     if strategy == "mean":
#         masked = hidden_states * attention_mask
#         pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)

#     # ───────────────────────────── max ──────────────────────────────
#     elif strategy == "max":
#         masked = hidden_states * attention_mask + (1.0 - attention_mask) * -1e9
#         pooled, _ = masked.max(dim=1)

#     # ─────────────────────────── last token ─────────────────────────
#     elif strategy == "last_token":
#         seq_lengths = attention_mask.squeeze(-1).sum(dim=1)          # (#tokens)
#         idx = (seq_lengths - 1).long().unsqueeze(-1).unsqueeze(-1)   # (b,1,1)
#         idx = idx.expand(-1, 1, hidden_states.size(-1))              # (b,1,d)
#         pooled = hidden_states.gather(1, idx).squeeze(1)             # (b,d)

#     # ───────────────────────────── echo ─────────────────────────────
#     elif strategy == "echo":
#         with torch.no_grad():  # ← safe context
#             input_embs = model.get_input_embeddings()(inputs["input_ids"])
#             masked = input_embs * attention_mask
#             pooled = masked.sum(dim=1) / attention_mask.sum(dim=1)
#     # ───────────────────────── unknown ──────────────────────────────
#     else:
#         raise ValueError(f"Unknown strategy: {strategy}")

#     return pooled.squeeze().cpu().numpy()

# def summary():
#     """
#     Print a summary of the embeddings created.
#     """
#     print("\n======= EMBEDDING SUMMARY =======")
#     made  = sum(v=="made"  for v in STATUS.values())
#     disk  = sum(v=="disk"  for v in STATUS.values())
#     flag  = sum(v=="flag"  for v in STATUS.values())
#     print(f"made new     : {made}")
#     print(f"skipped (disk): {disk}")
#     print(f"skipped (flag): {flag}")
#     print(f"total handled: {made+disk+flag}/24")
#     for k,v in sorted(STATUS.items()):
#         print(f"{k}: {v}")
#     print("==================================\n")

# # ============================================================
# # 2. Core routine: create embeddings for ONE model & ONE strategy
# # ============================================================
# import torch, gc, os, re
# from tqdm.auto import tqdm
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
# from accelerate import init_empty_weights, load_checkpoint_and_dispatch   # auto‑offload helper

# # ------------------------------------------------------------
# # create_embeddings –– works on P100 & survives Ctrl‑C
# # ------------------------------------------------------------
# def create_embeddings(model_id: str,
#                       strategy: str,
#                       save_every: int = 100,
#                       train_csv: str = TRAIN_CSV,
#                       test_csv: str = TEST_CSV):

#     assert strategy in {"mean", "last_token", "echo"}

#     model_name = re.sub(r"[^\w\-]", "-", model_id.split("/")[-1])  # FS‑safe

#     if (model_name, strategy, "train") in FLAG_DONE:
#         print(f"⚑⚑ Fully skipped {model_name}:{strategy} (flagged)")
#         STATUS[(model_name, strategy, "train")] = "flag"
#         return

#     if (model_name, strategy, "test") in FLAG_DONE:
#         print(f"⚑⚑ Fully skipped {model_name}:{strategy} (flagged)")
#         STATUS[(model_name, strategy, "test")] = "flag"
#         return
        
#     # ────────────────────────── 1) Load model safely ──────────────────────────
#     print(f"\n🔄 Loading {model_id} …")
#     torch_dtype = torch.float16
#     needs_4bit  = any(k in model_id.lower() for k in ["llama-3", "8b", "7b"])
#     if needs_4bit and _bnb_ok:
#         # 4‑bit quantization with bitsandbytes
#         quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
#                                        bnb_4bit_compute_dtype=torch.float16,
#                                        bnb_4bit_use_double_quant=True,
#                                        bnb_4bit_quant_type="nf4")
#         model = AutoModel.from_pretrained(model_id,
#                                           device_map="auto",
#                                           torch_dtype=torch_dtype,
#                                           quantization_config=quant_cfg)
#     else:
#         # small models: FP16 on the single GPU
#         max_mem = {0: "14GiB", "cpu": "32GiB"}
#         model = AutoModel.from_pretrained(model_id,
#                                         torch_dtype=torch_dtype,
#                                         device_map="cuda",
#                                           max_memory=max_mem)
#     tokenizer = AutoTokenizer.from_pretrained(model_id)

#     model.eval()

#     # Local wrapper so we don’t repeat args
#     def embed_fn(text: str):
#         return get_embedding(model, tokenizer, text, strategy)

#     # --------------------------------------------------------
#     # split‑level worker  (defined **before** we call it!)
#     # --------------------------------------------------------
#     def _process_split(csv_path: str, split: str):
        
#         df     = pd.read_csv(csv_path)
#         n_rows = len(df)

#         title_f   = _emb_files(model_name, split, strategy, "title")
#         content_f = _emb_files(model_name, split, strategy, "content")
#         label_f   = _emb_files(model_name, split, strategy, "label")

#         done_rows = 0
#         if all(os.path.exists(p) for p in (title_f, content_f, label_f)):
#             done_rows = np.load(title_f, mmap_mode='r').shape[0]
#             if done_rows == n_rows:
#                 print(f"✓ {model_name}:{strategy}:{split} already on disk")
#                 STATUS[(model_name, strategy, split)] = "disk"       # ← new
#                 return
#             print(f"⟳ resuming at {done_rows}/{n_rows}")

#         t_buf, c_buf, l_buf = [], [], []
#         pbar = tqdm(range(done_rows, n_rows), desc=f"{model_name}|{strategy}|{split}")

#         for i in pbar:
#             row = df.iloc[i]
#             t_buf.append(embed_fn(str(row["title"])))
#             c_buf.append(embed_fn(str(row["content"])))
#             l_buf.append(int(row["starRating"]))

#             if (len(t_buf) % save_every == 0) or i == n_rows - 1:
#                 _flush(title_f, content_f, label_f, t_buf, c_buf, l_buf)
#                 pbar.set_postfix(rows=i+1)
#         STATUS[(model_name, strategy, split)] = "made"  

#     # --------------------------------------------------------
#     # helper to append and clear Python lists
#     # --------------------------------------------------------
#     def _flush(t_path, c_path, l_path, tb, cb, lb):
#         if not tb:
#             return
#         arr_t = np.asarray(tb, dtype=np.float32)
#         arr_c = np.asarray(cb, dtype=np.float32)
#         arr_l = np.asarray(lb, dtype=np.int64)
#         if os.path.exists(t_path):
#             arr_t = np.concatenate([np.load(t_path), arr_t])
#             arr_c = np.concatenate([np.load(c_path), arr_c])
#             arr_l = np.concatenate([np.load(l_path), arr_l])
#         np.save(t_path, arr_t)
#         np.save(c_path, arr_c)
#         np.save(l_path, arr_l)
#         tb.clear(); cb.clear(); lb.clear()

#     # --------------------------------------------------------
#     # try / except to handle Ctrl‑C and still clean up
#     # --------------------------------------------------------
#     try:
#         _process_split(train_csv, "train")
#         _process_split(test_csv,  "test")

#     except KeyboardInterrupt:
#         print("🛑 Interrupted by user – partial progress kept.")
#         raise

#     finally:
#         # universal cleanup
#         del model, tokenizer
#         torch.cuda.empty_cache()
#         gc.collect()
#         print(f"🧹 GPU cleared for {model_id}:{strategy}")


# # ============================================================
# # 3. Run everything
# # ============================================================
# model_ids = [
#     "ai-forever/mGPT-1.3B-romanian",
#     "faur-ai/LLMic",
#     "OpenLLM-Ro/RoLlama2-7b-Instruct-2024-10-09",
#     "meta-llama/Meta-Llama-3.1-8B",
# ]


# pooling_strategies = ["mean", "last_token", "echo"]

# try:
#     for mid in model_ids:
#         for strat in pooling_strategies:
#             create_embeddings(mid, strat)
#     summary()
# except KeyboardInterrupt:
#     summary()
#     print("🏃‍♂️ Notebook stopped by user.")

# print("🎉 All requested embeddings are done!")























device = "cuda"
import os, gc, math, re
from collections import Counter
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModel

# Kaggle paths to the LAROSeDa CSVs
TRAIN_CSV = "data/laroseda/laroseda_train.csv"
TEST_CSV  = "data/laroseda/laroseda_test.csv"

# ────────────────────────────────
FLAG_DONE = {
("Meta-Llama-3-1-8B", "mean", "train"),
("Meta-Llama-3-1-8B", "mean", "test"),
("Meta-Llama-3-1-8B", "last_token", "train"),
("Meta-Llama-3-1-8B", "last_token", "test"),
}

STATUS = {}

# ────────────────────────────────

try:
    from transformers import BitsAndBytesConfig          
    _bnb_ok = True
except Exception:
    _bnb_ok = False                            

# ============================================================
# 1. Small helpers
# ============================================================
def _ensure_dir(path: str):
    """mkdir -p for any (sub)folder."""
    os.makedirs(path, exist_ok=True)

def _infer_hidden_size(model):
    """Tries a few common names; raises if none is found."""
    for attr in ("hidden_size", "n_embd", "d_model"):
        if hasattr(model.config, attr):
            return int(getattr(model.config, attr))
    raise RuntimeError("Could not determine hidden size for this model.")

def _emb_files(model_name: str, split: str, strategy: str, field: str):
    """
    Returns e.g.  
    embeddings/LLMic/laroseda_train_LLMic_mean_title.npy
    """
    fld = f"embeddings/laroseda/{model_name}"
    _ensure_dir(fld)
    return f"{fld}/laroseda_{split}_{model_name}_{strategy}_{field}.npy"


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

def summary():
    """
    Print a summary of the embeddings created.
    """
    print("\n======= EMBEDDING SUMMARY =======")
    made  = sum(v=="made"  for v in STATUS.values())
    disk  = sum(v=="disk"  for v in STATUS.values())
    flag  = sum(v=="flag"  for v in STATUS.values())
    print(f"made new     : {made}")
    print(f"skipped (disk): {disk}")
    print(f"skipped (flag): {flag}")
    print(f"total handled: {made+disk+flag}/24")
    for k,v in sorted(STATUS.items()):
        print(f"{k}: {v}")
    print("==================================\n")

# ============================================================
# 2. Core routine: create embeddings for ONE model & ONE strategy
# ============================================================
import torch, gc, os, re
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch   # auto‑offload helper

# ------------------------------------------------------------
# create_embeddings –– works on P100 & survives Ctrl‑C
# ------------------------------------------------------------
def create_embeddings(model_id: str,
                      strategy: str,
                      save_every: int = 100,
                      train_csv: str = TRAIN_CSV,
                      test_csv: str = TEST_CSV):

    assert strategy in {"mean", "last_token", "echo"}

    model_name = re.sub(r"[^\w\-]", "-", model_id.split("/")[-1])  # FS‑safe

    if (model_name, strategy, "train") in FLAG_DONE:
        print(f"⚑⚑ Fully skipped {model_name}:{strategy} (flagged)")
        STATUS[(model_name, strategy, "train")] = "flag"
        return

    if (model_name, strategy, "test") in FLAG_DONE:
        print(f"⚑⚑ Fully skipped {model_name}:{strategy} (flagged)")
        STATUS[(model_name, strategy, "test")] = "flag"
        return
        
    # ────────────────────────── 1) Load model safely ──────────────────────────
    print(f"\n🔄 Loading {model_id} …")
    torch_dtype = torch.float16
    needs_4bit  = any(k in model_id.lower() for k in ["llama-3", "8b", "7b"])
    if needs_4bit and _bnb_ok:
        # 4‑bit quantization with bitsandbytes
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                       bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_use_double_quant=True,
                                       bnb_4bit_quant_type="nf4")
        model = AutoModel.from_pretrained(model_id,
                                          device_map="auto",
                                          torch_dtype=torch_dtype,
                                          quantization_config=quant_cfg)
    else:
        # small models: FP16 on the single GPU
        max_mem = {0: "14GiB", "cpu": "32GiB"}
        model = AutoModel.from_pretrained(model_id,
                                        torch_dtype=torch_dtype,
                                        device_map="cuda",
                                          max_memory=max_mem)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    model.eval()

    # Local wrapper so we don’t repeat args
    def embed_fn(text: str):
        return get_embedding(model, tokenizer, text, strategy)

    # --------------------------------------------------------
    # split‑level worker  (defined **before** we call it!)
    # --------------------------------------------------------
    def _process_split(csv_path: str, split: str):
        
        df     = pd.read_csv(csv_path)
        n_rows = len(df)

        questions_f = _emb_files(model_name, split, strategy, "questions")
        q_plus_option_a_f = _emb_files(model_name, split, strategy, "q_plus_option_a")
        q_plus_option_b_f = _emb_files(model_name, split, strategy, "q_plus_option_b")
        q_plus_option_c_f = _emb_files(model_name, split, strategy, "q_plus_option_c")
        q_plus_option_d_f = _emb_files(model_name, split, strategy, "q_plus_option_d")
        label_f   = _emb_files(model_name, split, strategy, "label")

        done_rows = 0
        if all(os.path.exists(p) for p in (questions_f, q_plus_option_a_f, q_plus_option_b_f, q_plus_option_c_f, q_plus_option_d_f, label_f)):
            done_rows = np.load(questions_f, mmap_mode='r').shape[0]
            if done_rows == n_rows:
                print(f"✓ {model_name}:{strategy}:{split} already on disk")
                STATUS[(model_name, strategy, split)] = "disk"       # ← new
                return
            print(f"⟳ resuming at {done_rows}/{n_rows}")

        q_buf, q_plus_option_a_buf, q_plus_option_b_buf, q_plus_option_c_buf, q_plus_option_d_buf, l_buf = [], [], [], [], [], []
        pbar = tqdm(range(done_rows, n_rows), desc=f"{model_name}|{strategy}|{split}")

        for i in pbar:
            row = df.iloc[i]
            # t_buf.append(embed_fn(str(row["title"])))
            # c_buf.append(embed_fn(str(row["content"])))
            # l_buf.append(int(row["starRating"]))
            question = str(row["instruction"])
            label = str(row["answer"]).strip().upper()
            q_buf.append(embed_fn(question))
             

            if (len(t_buf) % save_every == 0) or i == n_rows - 1:
                _flush(title_f, content_f, label_f, t_buf, c_buf, l_buf)
                pbar.set_postfix(rows=i+1)
        STATUS[(model_name, strategy, split)] = "made"  

    # --------------------------------------------------------
    # helper to append and clear Python lists
    # --------------------------------------------------------
    def _flush(t_path, c_path, l_path, tb, cb, lb):
        if not tb:
            return
        arr_t = np.asarray(tb, dtype=np.float32)
        arr_c = np.asarray(cb, dtype=np.float32)
        arr_l = np.asarray(lb, dtype=np.int64)
        if os.path.exists(t_path):
            arr_t = np.concatenate([np.load(t_path), arr_t])
            arr_c = np.concatenate([np.load(c_path), arr_c])
            arr_l = np.concatenate([np.load(l_path), arr_l])
        np.save(t_path, arr_t)
        np.save(c_path, arr_c)
        np.save(l_path, arr_l)
        tb.clear(); cb.clear(); lb.clear()

    # --------------------------------------------------------
    # try / except to handle Ctrl‑C and still clean up
    # --------------------------------------------------------
    try:
        _process_split(train_csv, "train")
        _process_split(test_csv,  "test")

    except KeyboardInterrupt:
        print("🛑 Interrupted by user – partial progress kept.")
        raise

    finally:
        # universal cleanup
        del model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        print(f"🧹 GPU cleared for {model_id}:{strategy}")


# ============================================================
# 3. Run everything
# ============================================================
model_ids = [
    "ai-forever/mGPT-1.3B-romanian",
    "faur-ai/LLMic",
    "OpenLLM-Ro/RoLlama2-7b-Instruct-2024-10-09",
    "meta-llama/Meta-Llama-3.1-8B",
]


pooling_strategies = ["mean", "last_token", "echo"]

try:
    for mid in model_ids:
        for strat in pooling_strategies:
            create_embeddings(mid, strat)
    summary()
except KeyboardInterrupt:
    summary()
    print("🏃‍♂️ Notebook stopped by user.")

print("🎉 All requested embeddings are done!")

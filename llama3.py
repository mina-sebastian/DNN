print("Loading transformer")
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
print("Loading torch")
import torch
print("Loading numpy and pandas")
import numpy as np
import pandas as pd
print("Loading os")
import os
print("Loading Dataset")
from torch.utils.data import Dataset
from datasets_class import TitleContentDataset

device = "cuda"
model_id = "meta-llama/Meta-Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_id)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# model = AutoModelForCausalLM.from_pretrained(
#     model_id,
#     device_map="auto",         # Automatically place layers on GPU/CPU
#     torch_dtype=torch.float16, # Use half precision
# )


model = AutoModel.from_pretrained(
    model_id,
    device_map="auto",         # Automatically place layers on GPU/CPU
    torch_dtype=torch.float16, # Use half precision
    trust_remote_code = True
).to(device)

def get_embedding(text, strategy="mean"):
    """
    Return an embedding for 'text' using LLaMA's last_hidden_state + various pooling strategies.
    Available strategies: ['mean', 'last_token', 'max', 'summary']
    """

    # 1) Basic embedding from hidden states
    if strategy != "summary":
        # Standard approach: tokenize + forward pass
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs)
            # , output_hidden_states=True
        hidden_states = outputs.last_hidden_state # (batch, seq_len, hidden_dim) outputs.hidden_states[-1] 
        attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch, seq_len, 1)

        if strategy == "mean":
            # Mean Pooling
            masked_hidden = hidden_states * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            pooled = summed / counts

        elif strategy == "max":
            # Max Pooling
            # Replace padding (0) with a large negative number before max
            # This ensures we don't get zeros from padded tokens
            masked_hidden = hidden_states * attention_mask + (1.0 - attention_mask) * -1e9
            pooled, _ = masked_hidden.max(dim=1)

        elif strategy == "last_token":
            # The last token's hidden state
            seq_lengths = attention_mask.squeeze(-1).sum(dim=1)  # how many tokens per sample
            # We’ll gather the last actual token for each batch element
            idx = (seq_lengths - 1).long().unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden_states.size(-1))
            pooled = hidden_states.gather(1, idx).squeeze(1)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        return pooled.squeeze().cpu().numpy()

    # 2) Prompt-based summary approach
    else:
        # We'll prompt the model to "summarize" or "re-describe" the text, then embed that summary
        summary_prompt = f"Summarize this text in one sentence:\n\n{text}\n\nSummary:"
        inputs = tokenizer(
            summary_prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            # Generate a short summary
            summary_ids = model.generate(
                **inputs,
                max_new_tokens=40,  # limit summary length
                do_sample=False
            )

        summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Now embed the summary itself (2nd forward pass)
        summary_inputs = tokenizer(
            summary_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**summary_inputs)
        hidden_states = outputs.last_hidden_state # outputs.hidden_states[-1] 
        attention_mask = summary_inputs['attention_mask'].unsqueeze(-1)

        # Simple mean pooling for the summary
        masked_hidden = hidden_states * attention_mask
        summed = masked_hidden.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        pooled = summed / counts

        return pooled.squeeze().cpu().numpy()

# ============== EXAMPLE USAGE ==============
train_dataset = TitleContentDataset(
    csv_file="laroseda_train.csv",
    get_embedding=get_embedding,  
    type="train",
    name="llama3",                 
    emb_dim=4096,                
    save_interval=200            
)

test_dataset = TitleContentDataset(
    csv_file="laroseda_test.csv",
    get_embedding=get_embedding,  
    type="test",
    name="llama3",                 
    emb_dim=4096,                
    save_interval=200            
)
# test different strategies on sample sentences
# text1 = "Ana are mere"
# text2 = "Ion are pere pe care le-a luat de la magazin"

# for strat in ["mean", "max", "last_token"]: # , "summary"
#     print(f"\n--- Testing strategy: {strat} ---")
#     emb1 = get_embedding(text1, strategy=strat)
#     emb2 = get_embedding(text2, strategy=strat)
#     print("Shapes:", emb1.shape, emb2.shape)
#     print("Sample embedding1:", emb1[:5], "...")
#     print("Sample embedding2:", emb2[:5], "...")

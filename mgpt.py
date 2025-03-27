print("Loading transformer")
from transformers import AutoTokenizer, AutoModel
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
model_id = "ai-forever/mGPT-1.3B-romanian"

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

# train_dataset = train_dataset = TitleContentDataset(
#     csv_file="laroseda_train.csv",
#     get_embedding=get_embedding,  
#     type="train",
#     name="mGPT-1-3B-romanian",                 
#     emb_dim=2048,                
#     save_interval=200            
# )

test_dataset = TitleContentDataset(
    csv_file="laroseda_test.csv",
    get_embedding=get_embedding,  
    type="test",
    name="mGPT-1-3B-romanian",                 
    emb_dim=2048,                
    save_interval=200            
)

# text1 = "Ana are mere"
# text2 = "Ion are pere pe care le-a luat de la magazin"

# text1_embedding = get_embedding(text1)
# text2_embedding = get_embedding(text2)
# print(text1_embedding.shape, text2_embedding.shape)
# print(text1_embedding, text2_embedding)
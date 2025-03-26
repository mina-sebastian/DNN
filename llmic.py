from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from datasets_dnn import dataset_laroseda

device = "cuda"
model_id = "faur-ai/LLMic"

print(dataset_laroseda['train'][2]['content'])

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id).to(device)

print(model.encode("some text").to('cpu'))


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

def get_embeddings(dataset):
    embeddings = []
    for text in dataset:
        print(text['content'])
        embeddings.append(get_embedding(text['content']))
    return np.array(embeddings)

text1 = "Ana are mere"
text2 = "Ion are pere pe care le-a luat de la magazin"

# text1_embedding = get_embedding(text1)
# text2_embedding = get_embedding(text2)
# print(text1_embedding.shape, text2_embedding.shape)
# print(text1_embedding, text2_embedding)

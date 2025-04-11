import torch
from transformers import AutoTokenizer


def get_embedding(model, tokenizer, text, strategy="mean"):
    """
    Return an embedding for 'text' using LLaMA's last_hidden_state + various pooling strategies.
    Available strategies: ['mean', 'last_token', 'max', 'summary']
    """

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
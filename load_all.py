from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_ids = [
    #"ai-forever/mGPT-1.3B-romanian",
    #"faur-ai/LLMic",
    #"OpenLLM-Ro/RoLlama2-7b-Instruct-2024-10-09",
    "meta-llama/Meta-Llama-3.1-8B"
]

models = {}

for model_id in model_ids:
    print(f"\nLoading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        device_map="auto", # let accelerate place layers across GPU/CPU
        torch_dtype=torch.float16
        ).to(device)
    models[model_id] = {
        "tokenizer": tokenizer,
        "model": model
    }

print("\n✅ All models loaded into memory and cached.")

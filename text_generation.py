# Script de génération de texte avec top-p sampling.


import torch
import tiktoken

# Initialize tokenizer (GPT-2 encoding)
enc = tiktoken.get_encoding("gpt2")

# Function: Top-p sampling
def top_p_sampling(logits, top_p=0.9, temperature=1.0):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # Sort the probabilities
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Mask out tokens beyond top_p cumulative probability
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask right to keep at least one token
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    sorted_probs[sorted_indices_to_remove] = 0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Sample one token
    next_token = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_indices.gather(-1, next_token)
    return next_token

# Function: Text generation
def generate_text(model, prompt, device, max_new_tokens=100, top_p=0.9, temperature=1.0):
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)

        for _ in range(max_new_tokens):
            logits = model(input_ids)
            logits = logits[:, -1, :]  
            next_token = top_p_sampling(logits, top_p=top_p, temperature=temperature)
            input_ids = torch.cat((input_ids, next_token), dim=1)

        output = enc.decode(input_ids[0].tolist())
    return output
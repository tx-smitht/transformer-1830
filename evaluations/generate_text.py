import sys
import os
import torch

# Get the root directory (one level up from evaluations/)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script path
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move one level up

# Add project root to Python path
sys.path.append(project_root)

from transformers import DecoderTransformer, SimpleTokenizer
from config import *


model_path = '../models/final_model_384_6.pth'

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 
                     ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

def generate_text(prompt, max_new_tokens=100, temperature=temperature, top_k=top_k):
    # Encode the prompt
    encoded = tokenizer.encode(prompt)
    context = torch.tensor([encoded], dtype=torch.long, device=device)
    
    # Generate new tokens
    generated = model.generate(
        context,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k
    )
    
    # Decode and return the result
    return tokenizer.decode(generated[0].tolist())

# Initialize tokenizer with the same text used for training
with open('../data/book_of_mormon.txt', 'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = SimpleTokenizer()
tokenizer.fit(text)


# Initialize and load the model
model = DecoderTransformer(
    vocab_size=tokenizer.vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
).to(device)


model.load_state_dict(torch.load(model_path))
print(f"Model loaded from {model_path}")

# Test the text generation
prompt = "58"
generated_text = generate_text(
    prompt,
    max_new_tokens=50,
    temperature=0.9,
    top_k=10        
)
print(generated_text)
import os
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys


# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Import hyperparameters from config file
from config import (
    n_embd, n_head, n_layer, dropout, block_size,
    batch_size, learning_rate, max_iters, eval_interval
)
# Set the device - use MPS for M1 Mac
device = torch.device('mps' if torch.backends.mps.is_available() else 
                     ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# If using MPS, enable memory efficient attention if available
if device.type == 'mps':
    print("Using Apple M1/M2 GPU acceleration")
    # Optional: Set memory format for better performance
    torch.set_default_dtype(torch.float32)  # MPS works best with float32
    


# Data Processing
class SimpleTokenizer:
    """A basic token-level tokenizer that splits on whitespace and punctuation"""
    def __init__(self):
        self.tokens = []
        self.stoi = {}
        self.itos = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3,
        }
    
    def fit(self, text):
        # Add special tokens
        self.stoi = self.special_tokens.copy()
        self.itos = {v: k for k, v in self.special_tokens.items()}
        
        # Split text into tokens
        import re
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        unique_tokens = sorted(set(words))
        
        # Create mappings starting after special tokens
        next_idx = len(self.special_tokens)
        for token in unique_tokens:
            if token not in self.stoi:
                self.stoi[token] = next_idx
                self.itos[next_idx] = token
                next_idx += 1
        
        self.vocab_size = len(self.stoi)
        print(f"Vocabulary size (including special tokens): {self.vocab_size}")
    
    def encode(self, text):
        # Split text and convert to token indices
        import re
        words = re.findall(r'\b\w+\b|[^\w\s]', text)
        return [self.stoi.get(word, self.special_tokens['<UNK>']) for word in words]
    
    def decode(self, ids):
        # Convert token indices back to text
        return ' '.join([self.itos[id] for id in ids])

class TextDataset(Dataset):
    """
    A PyTorch Dataset for processing text data into overlapping sequences.
    
    This dataset creates training examples for a language model by:
    - Taking sequences of length block_size from the input data
    - Creating input-target pairs where the target is shifted by one position
    
    For example, if the sequence is "hello" and block_size is 3:
    - Input:  "hel"  Target: "ell"
    - Input:  "ell"  Target: "llo"
    
    This allows the model to learn to predict the next character given a sequence.
    """
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        # Length of the dataset
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a block of data and the corresponding target
        x = torch.tensor(self.data[idx:idx+self.block_size], dtype=torch.long)
        y = torch.tensor(self.data[idx+1:idx+self.block_size+1], dtype=torch.long)
        return x, y

def load_data_from_files(file_pattern):
    """Load and concatenate text from all files matching the pattern"""
    text = ""

    file_paths = glob.glob(file_pattern)
    if not file_paths:
        raise ValueError(f"No files found matching pattern: {file_pattern}")
    
    print(f"Loading {len(file_paths)} text files...")
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text += f.read()
            print(f"Loaded: {file_path}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    return text

# Model Architecture
class MultiHeadAttention(nn.Module):
    """
    Implements multi-head self-attention mechanism for the transformer architecture.
    
    Multi-head attention allows the model to jointly attend to information from 
    different representation subspaces at different positions. This module:
    - Splits the input into multiple heads
    - Computes attention scores between all positions
    - Applies causal masking to prevent attending to future positions
    - Combines the results of all heads
    
    The attention mechanism helps the model capture complex relationships and 
    dependencies between different positions in the sequence.
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        assert n_embd % n_head == 0
        self.head_size = n_embd // n_head
        
        # Linear layers for key, query, and value
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality
        
        # Calculate query, key, values for all heads in batch
        k = self.key(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)  # (B, nh, T, hs)
        
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))  # (B, nh, T, T)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))  # Causal masking
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply attention weights to values
        y = att @ v  # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, nh * hs)
        
        # Output projection
        y = self.resid_dropout(self.proj(y))
        return y

class FeedForward(nn.Module):
    """
    Implements the feed-forward neural network component of a transformer block.
    
    This network consists of two linear transformations with a GELU activation 
    in between, followed by dropout. The architecture is:
    input -> linear -> GELU -> linear -> dropout -> output
    
    The feed-forward network processes each position independently and identically,
    allowing the model to introduce non-linearity and transform the representations.
    """
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Represents a single transformer block combining attention and feed-forward layers.
    
    Each block contains:
    1. Layer normalization followed by multi-head self-attention
    2. Layer normalization followed by feed-forward network
    3. Residual connections around each of these sub-layers
    
    This structure allows the model to process sequential data while maintaining
    gradient flow through deep networks.
    """
    def __init__(self, n_embd, n_head, dropout, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout, block_size)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class DecoderTransformer(nn.Module):
    """
    A complete transformer decoder model for language modeling tasks.
    
    This model generates text by predicting the next token given a sequence of 
    previous tokens. It includes:
    - Token embeddings to convert indices to vectors
    - Positional embeddings to encode token positions
    - Multiple transformer blocks for processing
    - Final layer normalization and projection to vocabulary size
    
    Key features:
    - Autoregressive generation (each prediction depends on previous tokens)
    - Temperature-controlled sampling for generation
    - Optional top-k sampling for better quality output
    
    The model can be used for both training (with teacher forcing) and 
    generation (autoregressive).
    """
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, dropout, block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is {self.block_size}"
        
        # Token and position embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        pos_emb = self.position_embedding(pos)  # (1, T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Language model head
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            # Reshape logits to (B*T, vocab_size) and targets to (B*T) for cross_entropy
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate text using the model with optional temperature and top-k sampling
        """
        for _ in range(max_new_tokens):
            # Crop context if it's too long
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self.forward(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering if specified
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# Training and evaluation functions
@torch.no_grad()
def estimate_loss(model, train_dataloader, val_dataloader, eval_iters=20):
    model.eval()
    out = {}
    for split, dataloader in [('train', train_dataloader), ('val', val_dataloader)]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            if i >= len(dataloader):
                break
            x, y = next(iter(dataloader))
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def train_model():
    # Load and process data
    data_path = "data/*.txt"
    text = load_data_from_files(data_path)
    print(f"Total text length: {len(text)} characters")
    
    tokenizer = SimpleTokenizer()
    tokenizer.fit(text)
    data = tokenizer.encode(text)
    print(f"Total tokens: {len(data)}")
    
    # Create train/val split
    n = len(data)
    train_data = data[:int(n*0.9)]
    val_data = data[int(n*0.9):]
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_data, block_size)
    val_dataset = TextDataset(val_data, block_size)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer, and scheduler
    model = DecoderTransformer(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")
    
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_iters)
    
    # Training loop
    losses = []
    train_losses = []  # Track training losses
    val_losses = []    # Track validation losses
    iter_nums = []     # Track iteration numbers
    iter_num = 0
    best_val_loss = float('inf')
    start_time = time.time()
    
    try:
        print("Starting training...")
        progress_bar = tqdm(range(max_iters), desc="Training")
        for _ in progress_bar:
            # Sample a batch of data
            x, y = next(iter(train_dataloader))
            x, y = x.to(device), y.to(device)
            
            # Forward pass
            logits, loss = model(x, y)
            losses.append(loss.item())
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluate and print progress
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                elapsed_time = time.time() - start_time
                train_loss = np.mean(losses[-eval_interval:])
                val_loss = estimate_loss(model, train_dataloader, val_dataloader)['val']
                
                # Store losses for plotting
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                iter_nums.append(iter_num)
                
                print(f"Iter {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}, time {elapsed_time:.2f}s")
                
                # Save the best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), 'models/best_model.pth')
                    print("Saved best model")
            
            iter_num += 1
    except KeyboardInterrupt:
        print("Training interrupted")
        print("Attention mask shape:", model.blocks[0].attn.mask.shape)
    finally:
        print("Training completed")
        torch.save(model.state_dict(), 'models/final_model.pth')
        print("Saved final model")
        
        # Plot the training and validation losses
        plt.figure(figsize=(10, 6))
        plt.plot(iter_nums, train_losses, label='Training Loss')
        plt.plot(iter_nums, val_losses, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_loss.png')
        plt.close()
        print("Loss plot saved as 'training_loss.png'")

if __name__ == "__main__":
    train_model()
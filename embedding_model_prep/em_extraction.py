import os
import torch
import sys

# Add project root to path to ensure imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.append(project_root)

# Import your transformer model and tokenizer
from transformers import DecoderTransformer, SimpleTokenizer
from embedding_model import WordEmbeddingModel  # Import the new model class we created

# Import hyperparameters from config file
from config import (
    n_embd, n_head, n_layer, dropout, block_size
)



def extract_and_save_embedding_model(model_path, save_dir):
    """
    Extract the embedding layer from a trained transformer model and save it as a standalone model
    
    Args:
        model_path: Path to the saved transformer model state_dict
        save_dir: Directory to save the embedding model
        tokenizer_path: Optional path to the saved tokenizer
    """

    # Set the device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    # Create tokenizer
    # Read text file
    with open('../data/book_of_mormon.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Define the tokenizer
    tokenizer = SimpleTokenizer()
    # Fit the tokenizer on the text
    tokenizer.fit(text)
    
    # Load the transformer model
    if tokenizer:
        vocab_size = tokenizer.vocab_size
    else:
        # If no tokenizer is provided, we need to infer the vocab size
        print("No tokenizer provided, using vocab_size from model state_dict")
        state_dict = torch.load(model_path, map_location=device)
        vocab_size = state_dict['token_embedding.weight'].shape[0]
    
    # Initialize the transformer model
    transformer = DecoderTransformer(
        vocab_size=vocab_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        block_size=block_size,
        dropout=dropout
    ).to(device)
    
    # Load the saved model weights
    transformer.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Loaded transformer model from {model_path}")
    
    # Create the embedding model from the transformer
    embedding_model = WordEmbeddingModel.from_transformer(transformer)
    print(f"Created embedding model with dimensions: {embedding_model.embedding.weight.shape}")
    
    # Save the embedding model in Hugging Face format
    embedding_model.save_pretrained(save_dir)
    print(f"Saved embedding model to {save_dir}")
    
    # If we have a tokenizer, save it with the model
    if tokenizer:
        import pickle
        with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
            pickle.dump(tokenizer, f)
        
        # Also save a vocabulary file that's easier to read
        with open(os.path.join(save_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            for token, idx in sorted(tokenizer.stoi.items(), key=lambda x: x[1]):
                f.write(f"{idx}\t{token}\n")
        
        print(f"Saved tokenizer to {save_dir}")

if __name__ == "__main__":
    model_path = "../models/final_model_384_6.pth"  # Path to your trained transformer model
    tokenizer_path = "../models/tokenizer.pkl"  # Path to your saved tokenizer (if available)
    save_dir = f"../models/word_embedding_{n_embd}"  # Directory to save the embedding model
    
    extract_and_save_embedding_model(model_path, save_dir)
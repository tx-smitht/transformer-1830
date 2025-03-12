import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import json

class WordEmbeddingModel(nn.Module):
    """
    A standalone word embedding model that can be used for word representations
    and uploaded to Hugging Face Hub.
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids):
        """
        Get embeddings for the input token ids
        """
        return self.embedding(input_ids)
    
    def get_embedding_matrix(self):
        """
        Returns the full embedding matrix
        """
        return self.embedding.weight.data
        
    def get_vector(self, token_id):
        """
        Get the embedding vector for a specific token id
        """
        if token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} is out of range (vocab size: {self.vocab_size})")
        return self.embedding.weight.data[token_id].cpu().numpy()
    
    def similarity(self, token_id1, token_id2):
        """
        Calculate cosine similarity between two token embeddings
        """
        vec1 = self.embedding.weight.data[token_id1]
        vec2 = self.embedding.weight.data[token_id2]
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    @classmethod
    def from_transformer(cls, transformer_model):
        """
        Create a WordEmbeddingModel from a trained transformer model
        """
        # Extract the embedding layer and its parameters
        token_embedding = transformer_model.token_embedding
        vocab_size, embedding_dim = token_embedding.weight.shape
        
        # Create a new embedding model
        model = cls(vocab_size, embedding_dim)
        
        # Copy the weights from the transformer's embedding layer
        model.embedding.weight.data = token_embedding.weight.data.clone()
        
        return model
    
    def save_pretrained(self, save_directory):
        """
        Save the model in a format compatible with Hugging Face
        """
        os.makedirs(save_directory, exist_ok=True)
        
        # Save the model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # Create a config file
        config = {
            "vocab_size": self.vocab_size,
            "embedding_dim": self.embedding_dim,
            "model_type": "word_embedding",
        }
        
        import json
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Create a simple README
        with open(os.path.join(save_directory, "README.md"), "w") as f:
            f.write(f"# Word Embedding Model\n\n")
            f.write(f"This is a word embedding model with vocabulary size {self.vocab_size} ")
            f.write(f"and embedding dimension {self.embedding_dim}.\n\n")
            f.write("It was extracted from a transformer language model.")
        
    @classmethod
    def from_pretrained(cls, model_path):
        """
        Load a model from a Hugging Face-style directory or repository ID
        """
        # Handle both local path and Hugging Face repo_id
        if not os.path.isdir(model_path):
            # This is a Hugging Face repo ID, download the files
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
        else:
            # This is a local directory
            config_path = os.path.join(model_path, "config.json")
            weights_path = os.path.join(model_path, "pytorch_model.bin")
        
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Create model
        model = cls(config["vocab_size"], config["embedding_dim"])
        
        # Load weights
        state_dict = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model
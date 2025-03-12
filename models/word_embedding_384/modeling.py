import os
import torch
import torch.nn as nn
from torch.nn import functional as F

class WordEmbeddingModel(nn.Module):
    """
    A standalone word embedding model for word representations
    """
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
    def forward(self, input_ids):
        return self.embedding(input_ids)
    
    def get_vector(self, token_id):
        if token_id >= self.vocab_size:
            raise ValueError(f"Token ID {token_id} is out of range (vocab size: {self.vocab_size})")
        return self.embedding.weight.data[token_id].cpu().numpy()
    
    def similarity(self, token_id1, token_id2):
        vec1 = self.embedding.weight.data[token_id1]
        vec2 = self.embedding.weight.data[token_id2]
        return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
    
    @classmethod
    def from_pretrained(cls, model_path):
        """Load from Hugging Face Hub or local directory"""
        import json
        
        # Handle both local path and Hugging Face repo_id
        if not os.path.isdir(model_path):
            from huggingface_hub import hf_hub_download
            config_path = hf_hub_download(repo_id=model_path, filename="config.json")
            weights_path = hf_hub_download(repo_id=model_path, filename="pytorch_model.bin")
        else:
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
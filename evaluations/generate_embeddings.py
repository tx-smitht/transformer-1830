import os
import sys
import pickle
from huggingface_hub import hf_hub_download
from scipy.spatial.distance import cosine

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# Now you can import from embedding_model_prep
from embedding_model_prep.embedding_model import WordEmbeddingModel

# Download the model files
model_path = hf_hub_download(repo_id="tx-smitht/word-embeddings-1830", filename="pytorch_model.bin")
config_path = hf_hub_download(repo_id="tx-smitht/word-embeddings-1830", filename="config.json")
tokenizer_path = hf_hub_download(repo_id="tx-smitht/word-embeddings-1830", filename="tokenizer.pkl")

# Load the model
model = WordEmbeddingModel.from_pretrained("tx-smitht/word-embeddings-1830")

# Load the tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Function to get embedding for a word
def get_word_embedding(word):
    if word not in tokenizer.stoi:
        raise ValueError(f"Word '{word}' not found in vocabulary")
    token_id = tokenizer.stoi[word]
    return model.get_vector(token_id)

# Function to find similar words
def find_similar_words(word, n=5):
    # Get embedding for the input word
    word_embedding = get_word_embedding(word)
    
    # Calculate similarities with all words
    similarities = []
    for token, idx in tokenizer.stoi.items():
        if token == word:
            continue
        vec = model.get_vector(idx)
        similarity = 1 - cosine(word_embedding, vec)
        similarities.append((token, similarity))
    
    # Return top N similar words
    return sorted(similarities, key=lambda x: x[1], reverse=True)[:n]

# # Test with some words
# test_words = ['curious', 'strange', 'fine']

# for word in test_words:
#     print(f"\nSimilar words to '{word}':")
#     similar_words = find_similar_words(word)
#     for similar_word, similarity in similar_words:
#         print(f"{similar_word}: {similarity:.3f}")

# Define a function to calculate and print similarity
def print_similarity(word1, word2):
    embedding1 = get_word_embedding(word1)
    embedding2 = get_word_embedding(word2)
    similarity = 1 - cosine(embedding1, embedding2)
    print(f"Similarity between '{word1}' and '{word2}': {similarity:.3f}")

# Calculate similarities between word pairs
word_pairs = [
    ("curious", "strange"),
    ("curious", "fine"),
    ("awful", "bad"),
    ("awful", "wonderful")
]

for word1, word2 in word_pairs:
    print_similarity(word1, word2)

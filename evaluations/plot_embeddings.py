import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import sys

model_name = 'final_model_384_6.pth'

# Get the root directory (one level up from evaluations/)
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get current script path
project_root = os.path.abspath(os.path.join(script_dir, ".."))  # Move one level up

# Add project root to Python path
sys.path.append(project_root)

# Import configuration and transformer modules
from config import n_embd, n_head, n_layer, dropout, block_size, checkpoint_path
from transformers import DecoderTransformer, SimpleTokenizer

# Set device
device = torch.device('mps' if torch.backends.mps.is_available() else 
                     ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

with open('/Users/Tom/Documents/dev/deep-learning-edu/1828-embedding-model/transformer-1830/data/book_of_mormon.txt', 'r', encoding='utf-8') as f:
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

# Load the saved model
model_path = os.path.join(checkpoint_path, model_name)
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}. Please train the model first.")
    exit(1)

model.load_state_dict(torch.load(model_path))

# Get embedding weights
embedding_weights = model.token_embedding.weight.cpu().detach().numpy()

# Choose 10 interesting words to visualize
words_to_plot = ['laman', 'nephi', 'zarahemla', 'jerusalem', 'God', 'Lord', 'pray', 'cry', 'people', 'inheritance','commandments', 'for', 'said', 'unto', 'christ']
word_indices = [tokenizer.stoi[word] for word in words_to_plot if word in tokenizer.stoi]

if len(word_indices) < len(words_to_plot):
    print("Warning: Some words were not found in the vocabulary. Only plotting available words.")
    words_to_plot = [words_to_plot[i] for i in range(len(words_to_plot)) if words_to_plot[i] in tokenizer.stoi]

selected_embeddings = embedding_weights[word_indices]

# Reduce to 2D using PCA
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(selected_embeddings)

# Create the plot
plt.figure(figsize=(10, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)

# Add word labels
for i, word in enumerate(words_to_plot):
    plt.annotate(word, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.title('Word Embeddings Visualized in 2D')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.grid(True)
plt.savefig('evaluations/word_embeddings.png')
plt.show()

# Print explained variance ratio
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
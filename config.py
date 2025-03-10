"""
Centralized configuration file for model hyperparameters.
All model parameters should be defined here to maintain consistency across
training, inference, and visualization.
"""

# Model architecture parameters
n_embd = 384       # Embedding dimension
n_head = 6          # Number of attention heads
n_layer = 6         # Number of transformer layers
dropout = 0.2       # Dropout rate
block_size = 256 # Maximum sequence length

# Training parameters
batch_size = 64     # Batch size for training
learning_rate = 3e-4  # Learning rate
max_iters = 3000    # Maximum number of training iterations
eval_interval = 300 # Evaluate model every n iterations

# Generation parameters
temperature = 1.0   # Sampling temperature for text generation
top_k = 40         # Top-k sampling parameter

# Paths and data parameters
data_path = "data/*.txt"  # Path to training data
checkpoint_path = "models/"  # Path to save model checkpoints

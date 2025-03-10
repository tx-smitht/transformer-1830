---
language: en
tags:
- word-embeddings
- transformer
- book-of-mormon
- historical-text
license: mit
datasets:
- custom
---

# Word Embeddings Model - 1830s Text

This model provides word embeddings trained on historical text from the 1830s, specifically focusing on the Book of Mormon text. The embeddings capture semantic relationships between words in the context of early 19th-century religious literature.

## Model Details

- Embedding dimension: 384
- Training corpus: Book of Mormon (1830 edition)
- Architecture: Transformer-based embedding model
- Context window size: 256
- Vocabulary size: 5999

## Usage

```python
from huggingface_hub import from_pretrained

model = WordEmbeddingModel.from_pretrained('tx-smitht/word-embeddings-1830')
```

## Training Data

The model was trained on the 1830 edition of the Book of Mormon, representing historical English language patterns from the early 19th century.

## Limitations

- Domain-specific vocabulary focused on religious and historical text
- Limited to early 19th-century American English language patterns
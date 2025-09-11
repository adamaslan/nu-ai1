import torch
import numpy as np

# Load the file
embeddings = torch.load('passage_embeddings.pt')

# Check the type and structure
print(f"Type: {type(embeddings)}")
print(f"Shape: {embeddings.shape}")
print(f"Data type: {embeddings.dtype}")
print(f"Device: {embeddings.device}")

# Basic statistics
print(f"Min value: {embeddings.min()}")
print(f"Max value: {embeddings.max()}")
print(f"Mean: {embeddings.mean()}")
print(f"Standard deviation: {embeddings.std()}")
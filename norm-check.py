import torch

embeddings = torch.load('passage_embeddings.pt')
norms = torch.norm(embeddings, dim=1)
print(f"Embedding norms - Min: {norms.min():.4f}, Max: {norms.max():.4f}")
is_normalized = torch.allclose(norms, torch.ones_like(norms), atol=1e-3)
print(f"Are embeddings normalized? {is_normalized}")
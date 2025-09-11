from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

embeddings_np = embeddings.cpu().numpy()

# t-SNE works well with 150 points
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings_np)

plt.figure(figsize=(12, 8))
scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                     alpha=0.7, s=60, c=range(150), cmap='tab10')
plt.title('Passage Embeddings Visualization (t-SNE)')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(scatter, label='Passage Index')

# Optionally annotate some points
for i in [0, 1, 2, 3, 4]:  # Annotate first 5 points
    plt.annotate(f'P{i}', (embeddings_2d[i, 0], embeddings_2d[i, 1]))
plt.show()
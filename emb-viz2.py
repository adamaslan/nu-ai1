import pandas as pd
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

# Load your data and handle missing values
df = pd.read_csv('updated_ideas.csv')
passages = df['Ideas'].fillna('').astype(str).tolist()  # Convert NaN to empty strings, then to strings

# Load embeddings
embeddings = torch.load('passage_embeddings.pt')

print(f"Loaded {len(passages)} passages")
print(f"Embeddings shape: {embeddings.shape}")

# Check for any remaining issues
empty_passages = sum(1 for p in passages if len(p.strip()) == 0)
if empty_passages > 0:
    print(f"Warning: Found {empty_passages} empty passages")

def create_annotated_visualization(passages, max_label_length=30):
    embeddings_np = embeddings.cpu().numpy()
    
    # Create 2D representation
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    embeddings_2d = tsne.fit_transform(embeddings_np)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(15, 10))
    scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                        alpha=0.6, s=60, c=range(150), cmap='tab20')
    
    # Add text labels for some points (to avoid overcrowding)
    step = max(1, len(passages) // 20)  # Show ~20 labels
    for i in range(0, len(passages), step):
        # Handle empty or very short passages
        passage_text = passages[i].strip()
        if len(passage_text) == 0:
            label = f"[Empty passage {i}]"
        else:
            label = passage_text[:max_label_length] + "..." if len(passage_text) > max_label_length else passage_text
        
        ax.annotate(f'{i}: {label}', 
                   (embeddings_2d[i, 0], embeddings_2d[i, 1]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, alpha=0.7)
    
    plt.title('Passage Embeddings with Text Labels')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.tight_layout()
    plt.show()

# Now run the visualization
create_annotated_visualization(passages)
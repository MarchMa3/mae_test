import torch
from sklearn.decomposition import PCA

checkpoint = torch.load('checkpoints_mimic/epoch_2.pth')
token_emb = checkpoint['model_state_dict']['token_embedding.weight'].numpy()

pca = PCA()
pca.fit(token_emb)

cumsum = np.cumsum(pca.explained_variance_ratio_)
print(f"Top 10 dims explain: {cumsum[9]*100:.1f}% variance")
print(f"Top 50 dims explain: {cumsum[49]*100:.1f}% variance")
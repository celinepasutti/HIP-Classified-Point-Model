import torch
import numpy
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

checkpoints = './data/humanml3d_82k_ConvVAE_e2000.pt'

# Load the tensor from the .pt file
tensor = torch.load(checkpoints, map_location=torch.device('cpu'))

# Ensure the tensor is on the CPU
# tensor = tensor.cpu()

# Convert the tensor to a NumPy array
tensor_np = tensor.numpy()

# Initialize the TSNE model
tsne = TSNE(n_components=3, random_state=0)

# Fit and transform the data
tensor_tsne = tsne.fit_transform(tensor_np)

# Apply KMeans clustering to the 3D data
kmeans = KMeans(n_clusters=20, random_state=0)
clusters = kmeans.fit_predict(tensor_tsne)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot each point, coloring by cluster
scatter = ax.scatter(tensor_tsne[:, 0], tensor_tsne[:, 1], tensor_tsne[:, 2], c=clusters, cmap='viridis', marker='o')

# Create a legend with the cluster labels
legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
ax.add_artist(legend1)

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()

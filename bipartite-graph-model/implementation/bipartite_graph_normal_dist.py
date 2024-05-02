import os
import numpy as np
import matplotlib.pyplot as plt
from constants import BASE_PATH

# Creating a bipartite graph
input_n = np.zeros((30, 30))  # Initialize matrix with zeros

# Get the dimensions of the graph
n_node_n, n_edge_n = input_n.shape

# Loop for every node in the input nodes
for a in range(n_node_n):
    # Create 30 random numbers for each edge of that node using normal distribution
    w_n = 0.3 * np.random.randn(n_edge_n) + 1
    
    # Normalize to sum to 1
    w_norm_n = w_n / np.sum(w_n)
    
    # Put normalized values into matrix created before
    input_n[a, :] = w_norm_n

# Print the resulting matrix (optional)
print(input_n)

# Calculate the sum of the signal arriving at each output node (column-wise sum)
output_n = np.sum(input_n, axis=0)

# Print the result
print(output_n)

# Calculate variance
variance_n = np.var(output_n)

# Calculate minimum
minimum_n = np.min(output_n)

# Calculate maximum
maximum_n = np.max(output_n)

# Print the results
print("Variance:", variance_n)
print("Minimum:", minimum_n)
print("Maximum:", maximum_n)


# Set up figure environment
plt.figure(figsize=(12, 8))

# Plot heatmap for output_n
plt.subplot(3, 2, 4)
h_n = plt.imshow(output_n.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar(h_n)
plt.title('Heatmap - output signal in nodes (Normal distribution)', fontsize=12)
plt.ylabel('Nodes')
plt.xlabel('Heatmap')

# Plot histogram for input_n
plt.subplot(3, 2, 2)
iprandn = plt.hist(input_n.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7)
plt.ylabel('Number of edges', fontsize=14)
plt.xlabel('Weight signal', fontsize=14)
plt.ylim([0, 300])
plt.title('Normal distribution', fontsize=14)

# Plot histogram for output_n
plt.subplot(3, 2, 6)
bars = plt.hist(output_n, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7)
plt.ylabel('Number of nodes', fontsize=14)
plt.xlabel('Signal', fontsize=14)
plt.ylim([0, 10])

plt.tight_layout()
plt.show()

# Uncomment to generate new edge weights to use in next scripts
np.savetxt(os.path.join(BASE_PATH, 'data/inputs_normal_dis.csv'), input_n, delimiter=',')

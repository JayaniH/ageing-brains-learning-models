import numpy as np
import matplotlib.pyplot as plt

# Using uniform distribution

# Create a 30x30 matrix initialized with zeros
input_u = np.zeros((30, 30))

# Get the dimensions of the graph
n_node_u, n_edge_u = input_u.shape

# Loop for every node in the input nodes
for i in range(n_node_u):
    # Create 30 random numbers for each edge of that node
    w_u = np.random.rand(n_edge_u)
    
    # Normalize to sum to 1
    w_norm_u = w_u / np.sum(w_u)
    
    # Put normalized values into matrix created before
    input_u[i, :] = w_norm_u

# Calculate the sum of the signal arriving at each output node (column-wise sum)
output_u = np.sum(input_u, axis=0)

# Print the result
print(output_u)

# Calculate variance
variance_u = np.var(output_u)

# Calculate minimum
minimum_u = np.min(output_u)

# Calculate maximum
maximum_u = np.max(output_u)

# Print the results
print("Variance:", variance_u)
print("Minimum:", minimum_u)
print("Maximum:", maximum_u)


# Set up figure environment
plt.figure(figsize=(12, 8))

# Plot histogram for input_u
plt.subplot(3, 2, 1)
iprandu = plt.hist(input_u.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7)
plt.text(0.04, 250, 'Weight assignment distribution curve for edges', fontsize=16, fontweight='bold')
plt.ylabel('Number of edges', fontsize=14)
plt.xlabel('Weight signal', fontsize=14)
plt.ylim([0, 200])
plt.title('Uniform distribution', fontsize=14)

# Plot histogram for output_u
plt.subplot(3, 2, 5)
bars = plt.hist(output_u, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7)
plt.text(1.1, 11, 'Resulting Output signal distribution', fontsize=16, fontweight='bold')
plt.ylabel('Number of nodes', fontsize=14)
plt.xlabel('Signal', fontsize=14)
plt.ylim([0, 10])

# Plot heatmap for output_u
plt.subplot(3, 2, 3)
h_u = plt.imshow(output_u.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar(h_u)
plt.title('Heatmap - output signal in nodes (Uniform distribution)', fontsize=12)
plt.ylabel('Nodes')
plt.xlabel('Heatmap')

plt.tight_layout()
plt.show()

# Uncomment to generate new edge weights to use in next scripts
# np.savetxt(os.path.join(BASE_PATH, 'data/inputs_uniform_dis.csv'), input_u, delimiter=',')

import os
import numpy as np
import matplotlib.pyplot as plt
from constants import BASE_PATH

# Using uniform distribution
# Check if the file exists
file_path = os.path.join(BASE_PATH, 'data/original_data/Inputs_uniform_dis.csv')
if os.path.isfile(file_path):
    # Read the matrix from the file
    print("Reading data from file")
    input_u = np.loadtxt(file_path, delimiter=',')
else:
    # Generate new data
    print("Generating new data")

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

# Calculate variance
variance_u = np.var(output_u)

# Calculate minimum
minimum_u = np.min(output_u)

# Calculate maximum
maximum_u = np.max(output_u)

# Print the results
print("Output statistics (Uniform distribution)")
print("Variance:", variance_u)
print("Minimum:", minimum_u)
print("Maximum:", maximum_u)


# Plot the data - subplots
# Set up figure environment
plt.figure(figsize=(5, 8))

# Plot histogram for input_u
plt.subplot(3, 1, 1)
iprandu = plt.hist(input_u.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.ylim([0, 200])
plt.title('Weight assignment distribution curve for edges\n(Uniform distribution)', fontsize=12)

# Plot histogram for output_u
plt.subplot(3, 1, 3)
bars = plt.hist(output_u, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Output signal distribution in nodes\n(Uniform distribution)', fontsize=12)
plt.ylim([0, 10])

# Plot heatmap for output_u
plt.subplot(3, 1, 2)
h_u = plt.imshow(output_u.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar(h_u)
plt.title('Heatmap - output signal in nodes\n(Uniform distribution)', fontsize=12)
plt.ylabel('Nodes')
plt.xlabel('Heatmap')

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figures/uniform_distribution.png'))
plt.show()

# Plot the data - separate plots
# Plot histogram for input_u    
plt.hist(input_u.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.title('Input Weights (Uniform Distribution)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/uniform_distribution_input.png'))
plt.show()

# Plot histogram for output_u
plt.hist(output_u, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Output Signals (Uniform Distribution)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/uniform_distribution_output.png'))
plt.show()

# Plot heatmap for output_u
plt.imshow(output_u.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar()
plt.title('Heatmap - Output Signal in Nodes (Uniform Distribution)', fontsize=12)
plt.ylabel('Nodes')
plt.xlabel('Heatmap')
plt.savefig(os.path.join(BASE_PATH, 'figures/uniform_distribution_heatmap.png'))
plt.show()

# Save the data to a file
if not os.path.isfile(file_path):
    np.savetxt(os.path.join(BASE_PATH, 'data/inputs_uniform_dis.csv'), input_u, delimiter=',')
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import BASE_PATH

# Check if the file exists
file_path = os.path.join(BASE_PATH, 'data/original_data/Inputs_normal_dis.csv')
if os.path.isfile(file_path):
    # Read the matrix from the file
    print("Reading data from file")
    input_n = np.loadtxt(file_path, delimiter=',')
else:
    # Generate new data
    print("Generating new data")

    # Creating a bipartite graph
    input_n = np.zeros((30, 30))  # Initialize matrix with zeros

    # Get the dimensions of the graph
    n_node_n, n_edge_n = input_n.shape
    for a in range(n_node_n):
        # Create 30 random numbers for each edge of that node using normal distribution
        w_n = 0.3 * np.random.randn(n_edge_n) + 1

        # Normalize to sum to 1
        w_norm_n = w_n / np.sum(w_n)

        # Put normalized values into matrix created before
        input_n[a, :] = w_norm_n

# Calculate the sum of the signal arriving at each output node (column-wise sum)
output_n = np.sum(input_n, axis=0)

# Calculate variance
variance_n = np.var(output_n)

# Calculate minimum
minimum_n = np.min(output_n)

# Calculate maximum
maximum_n = np.max(output_n)

# Print the results
print("Output statistics (Normal distribution)")
print("Variance:", variance_n)
print("Minimum:", minimum_n)
print("Maximum:", maximum_n)

# Plot the data - subplots
# Set up figure environment
plt.figure(figsize=(5, 8))

# Plot histogram for input_n
plt.subplot(3, 1, 1)
iprandn = plt.hist(input_n.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.ylim([0, 300])
plt.title('Weight assignment distribution curve for edges\n(Normal distribution)', fontsize=12)

# Plot heatmap for output_n
plt.subplot(3, 1, 2)
h_n = plt.imshow(output_n.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar(h_n)
plt.xticks([]) 
plt.title('Heatmap - output signal in nodes\n(Normal distribution)', fontsize=12)
plt.ylabel('Nodes')
# plt.xlabel('Heatmap')

# Plot histogram for output_n
plt.subplot(3, 1, 3)
bars = plt.hist(output_n, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Resulting Output signal distribution\n(Normal distribution)', fontsize=12)
plt.ylim([0, 10])

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figures/normal_distribution.png'))
plt.show()


# Plot the data - separate plots
# Plot histogram for input_n
plt.hist(input_n.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.title('Input Weights (Normal Distribution)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/normal_distribution_input.png'))
plt.show()

# Plot histogram for output_n
plt.hist(output_n, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Output Signals (Normal Distribution)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/normal_distribution_output.png'))
plt.show()

# Plot heatmap for output_n
plt.imshow(output_n.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar()
plt.xticks([])
plt.title('Heatmap - Output Signal in Nodes (Normal Distribution)', fontsize=12)
plt.ylabel('Nodes', fontsize=10)
plt.savefig(os.path.join(BASE_PATH, 'figures/normal_distribution_heatmap.png'))
plt.show()

if not os.path.isfile(file_path):
    np.savetxt(os.path.join(BASE_PATH, 'data/inputs_normal_dis.csv'), input_n, delimiter=',')

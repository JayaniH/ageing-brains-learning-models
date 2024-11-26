import os
import numpy as np
import matplotlib.pyplot as plt
from constants import BASE_PATH

# Check if the file exists
file_path = os.path.join(BASE_PATH, 'data/inputs_kaiming_1.csv')
if os.path.isfile(file_path):
    # Read the matrix from the file
    print("Reading data from file")
    input = np.loadtxt(file_path, delimiter=',')
else:
    # Generate new data
    print("Generating new data")

    # Creating a bipartite graph
    input = np.zeros((30, 30))  # Initialize matrix with zeros

    # Get the dimensions of the graph
    n_node, n_edge = input.shape

    # Set the weights using Kaiming initialization
    w = np.random.randn(n_node, n_edge) * np.sqrt(2 / n_edge) + 1

    for a in range(n_node):
        # Normalize to sum to 1
        w_norm = w[a] / np.sum(w[a])

        # Put normalized values into matrix created before
        input[a, :] = w_norm

# Calculate the sum of the signal arriving at each output node (column-wise sum)
output = np.sum(input, axis=0)

# Calculate variance
variance = np.var(output)

# Calculate minimum
minimum = np.min(output)

# Calculate maximum
maximum = np.max(output)

# Print the results
print("Output statistics (Kaiming initialization)")
print("Variance:", variance)
print("Minimum:", minimum)
print("Maximum:", maximum)

# Plot the data - subplots
# Set up figure environment
plt.figure(figsize=(5, 8))

# Plot histogram for input_n
plt.subplot(3, 1, 1)
iprandn = plt.hist(input.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.ylim([0, 300])
plt.title('Weight assignment distribution curve for edges\n(Kaiming initialization)', fontsize=12)

# Plot heatmap for output_n
plt.subplot(3, 1, 2)
h = plt.imshow(output.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar(h)
plt.xticks([]) 
plt.title('Heatmap - output signal in nodes\n(Kaiming initialization)', fontsize=12)
plt.ylabel('Nodes')
# plt.xlabel('Heatmap')

# Plot histogram for output_n
plt.subplot(3, 1, 3)
bars = plt.hist(output, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Resulting Output signal distribution\n(Kaiming initialization)', fontsize=12)
plt.ylim([0, 10])

plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, 'figures/kaiming_initialization_1.png'))
plt.show()


# Plot the data - separate plots
# Plot histogram for input_n
plt.hist(input.flatten(), bins=10, color=[0.2, 0.2, 0.9], alpha=0.7, edgecolor='black')
plt.ylabel('Number of edges', fontsize=10)
plt.xlabel('Weight signal', fontsize=10)
plt.title('Input Weights (Kaiming initialization)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/kaiming_initialization_input_1.png'))
plt.show()

# Plot histogram for output_n
plt.hist(output, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
plt.ylabel('Number of nodes', fontsize=10)
plt.xlabel('Signal', fontsize=10)
plt.title('Output Signals (Kaiming initialization)', fontsize=12)
plt.savefig(os.path.join(BASE_PATH, 'figures/kaiming_initialization_output_1.png'))
plt.show()

# Plot heatmap for output_n
plt.imshow(output.reshape(-1, 1), cmap='winter', aspect='auto')
plt.colorbar()
plt.xticks([])
plt.title('Heatmap - Output Signal in Nodes (Kaiming initialization)', fontsize=12)
plt.ylabel('Nodes', fontsize=10)
plt.savefig(os.path.join(BASE_PATH, 'figures/kaiming_initialization_heatmap_1.png'))
plt.show()

if not os.path.isfile(file_path):
    np.savetxt(os.path.join(BASE_PATH, 'data/inputs_kaiming_1.csv'), input, delimiter=',')

import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from constants import BASE_PATH, ACTIVE_NODE_COUNT, TOTAL_NODE_COUNT

def get_input_and_output_patterns(initial_weights):
    """
    Generate input patterns and the corresponding output signals for a given set of initial weights

    :param initial_weights: The initial weights of the network
    :param total_node_count: The total number of nodes in the network
    :param active_node_count: The number of active nodes in each input pattern

    :return: The matrix of input patterns, the input matrix, and the output signals
    """

    # Generate all combinations of active nodes
    active_combinations = list(combinations(range(1, TOTAL_NODE_COUNT + 1), ACTIVE_NODE_COUNT))

    print("Number of input patterns:", len(active_combinations))

    # Create a matrix of input patterns
    input_patterns = np.zeros((TOTAL_NODE_COUNT, len(active_combinations)))

    # Define the dimensions of the input matrix
    input = np.zeros((TOTAL_NODE_COUNT, TOTAL_NODE_COUNT, len(active_combinations)))
    output = np.zeros((TOTAL_NODE_COUNT, len(active_combinations)))

    # Set active nodes to 1 in each input pattern
    for i in range(len(active_combinations)):
        input_patterns[np.array(active_combinations[i]) - 1, i] = 1

    # Assign the weights to the input patterns
    for i in range(len(active_combinations)):
        for j in range(TOTAL_NODE_COUNT):
            if input_patterns[j, i] == 1:  # If the node fires for this pattern
                input[j, :, i] = initial_weights[j, :]

    # Determine the signal reaching the output nodes for each input pattern
    for i in range(len(active_combinations)):
        for j in range(TOTAL_NODE_COUNT):
            output[j, i] = np.sum(input[:, j, i])
    
    return input_patterns, input, output


def get_threshold(th_values, output, min_count = 125000, max_count = 150000):
    """
    Determine the threshold value that results in a desired number of valid input patterns

    :param th_values: The threshold values to test
    :param output: The signal reaching the output nodes for each input pattern
    :param min_count: The minimum number of valid input patterns
    :param max_count: The maximum number of valid input patterns

    :return: The threshold value that results in the desired number of valid input patterns
    """
    
    # Calculate the maximum signal for each input pattern
    max_signals = np.max(output, axis=0)

    # Initialize array to store the count of valid input patterns for each threshold
    valid_counts = np.zeros(len(th_values))

    # Determine the count of valid input patterns for each threshold
    for i, th_value in enumerate(th_values):
        valid_counts[i] = np.sum(max_signals <= th_value)

    # Find the index of the first threshold value that falls within the desired range
    valid_indices = np.where((valid_counts >= min_count) & (valid_counts <= max_count))[0]
    if valid_indices.size == 0:
        # If no threshold meets the criteria, return None
        return None
    else:
        # Choose the threshold with the most valid input patterns
        chosen_index = valid_indices[np.argmax(valid_counts[valid_indices])]
        threshold = th_values[chosen_index]
        return threshold


def get_valid_inputs_and_outputs(threshold, input, input_patterns, output, save = False):
    """
    Determine the valid input patterns based on the chosen threshold

    :param threshold: The threshold value used to determine valid input patterns
    :param input: The input matrix
    :param input_patterns: The matrix of input patterns
    :param output: The signal reaching the output nodes for each input pattern
    :param save: Whether to save the matrices to .csv files

    :return: The valid input patterns, the input matrix for valid input patterns, and the output signals for valid input patterns
    """

    # Calculate the valid input pattern indices based on the chosen threshold
    valid_input_pattern_indices = np.where(np.max(output, axis=0) <= threshold)[0]
    print("Number of valid input patterns:", len(valid_input_pattern_indices))

    valid_input_patterns = input_patterns[:, valid_input_pattern_indices]
    valid_output = output[:, valid_input_pattern_indices]
    valid_input = input[:, :, valid_input_pattern_indices]

    if save:
        np.savetxt(os.path.join(BASE_PATH, 'data/valid_input_weights(thresholded).csv'), valid_input.reshape(-1, valid_input.shape[-1]), delimiter=",")
        np.savetxt(os.path.join(BASE_PATH, 'data/valid_input_patterns(thresholded).csv'), valid_input_patterns, delimiter=",", fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'data/valid_output_signals(thresholded).csv'), valid_output, delimiter=",", fmt='%.4f')

    return valid_input_patterns, valid_input, valid_output


def get_output_statistics(valid_output):
    """
    Calculate the variance in the output patterns

    :param valid_output: The signal reaching the output nodes for each valid input pattern

    :return: The variance in the output patterns, the average variance, and the standard deviation of the variance
    """

    variance = np.max(valid_output, axis=0) - np.min(valid_output, axis=0)
    avg_variance = np.mean(variance)
    sd_variance = np.std(variance)

    return variance, avg_variance, sd_variance

def plot_signal_statistics(output, threshold, variance, avg_variance, sd_variance):
    """
    Plot the maximum signal reaching the output nodes for each input pattern and the variance in the output patterns
    :param output: The signal reaching the output nodes for each input pattern
    :param threshold: The threshold value used to determine valid input patterns
    :param variance: The variance in the output patterns
    :param avg_variance: The average variance in the output patterns
    :param sd_variance: The standard deviation of the variance in the output patterns
    """

    # Plotting the thresholds
    graph = np.max(output, axis=0)  # Maximum signal reaching each of the nodes for a given IP

    plt.figure(figsize=(12, 6))

    # Plot maximum signal reaching output nodes
    plt.subplot(1, 2, 1)
    plt.hist(graph, color=[0.3, 0.9, 0.9], bins=30, edgecolor='black')  # Plot the histogram
    plt.xlim(0.2, 0.32)
    plt.axvline(threshold, color='red', linewidth=3)  # To show the determined threshold
    plt.title("Maximum signal reaching output nodes in all input patterns", fontsize=12)
    plt.xlabel("Maximum signal", fontsize=10)
    plt.ylabel("Input Patterns", fontsize=10)
    plt.text(0.28, 22800, f'Threshold = {threshold:.4f}', fontsize=10)
    plt.gca().set_facecolor('w')

    # Plot the variance in output patterns
    plt.subplot(1, 2, 2)
    plt.hist(variance, color=[0.1, 0.9, 0.3], bins=30, edgecolor='black')
    plt.title("Output variance across each valid input pattern", fontsize=12)
    plt.xlabel("Signal variance in output nodes", fontsize=10)
    plt.ylabel("Input Patterns", fontsize=10)
    plt.text(0.1, 10000, f'Mean variance = {avg_variance:.4f}', fontsize=10)
    plt.text(0.1, 9000, f'SD variance = {sd_variance:.4f}', fontsize=10)
    plt.gca().set_facecolor('w')

    plt.tight_layout()
    # plt.savefig(os.path.join(BASE_PATH, '/figures/thresholds_and_variance.png'))
    plt.show()

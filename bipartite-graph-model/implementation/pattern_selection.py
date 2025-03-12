import os
import json
import numpy as np
import pandas as pd
from itertools import combinations
from matplotlib import pyplot as plt
from constants import ACTIVE_NODE_COUNT, BASE_PATH, TOTAL_NODE_COUNT

def get_all_input_and_output_patterns(initial_weights):
    """
    Generate input patterns and the corresponding output signals for a given set of initial weights

    :param initial_weights: The initial weights of the network
    :param total_node_count: The total number of nodes in the network
    :param active_node_count: The number of active nodes in each input pattern

    :return: The matrix of input patterns, the weights for each input pattern, and the output signals
    """

    # Generate all combinations of active nodes
    active_combinations = list(combinations(range(1, TOTAL_NODE_COUNT + 1), ACTIVE_NODE_COUNT))

    # Create a matrix of input patterns
    input_patterns = np.zeros((TOTAL_NODE_COUNT, len(active_combinations)))

    # Define the dimensions of the input matrix
    weights = np.zeros((TOTAL_NODE_COUNT, TOTAL_NODE_COUNT, len(active_combinations)))
    outputs = np.zeros((TOTAL_NODE_COUNT, len(active_combinations)))

    # Set active nodes to 1 in each input pattern
    for i in range(len(active_combinations)):
        input_patterns[np.array(active_combinations[i]) - 1, i] = 1

    # Assign the weights to the input patterns
    for i in range(len(active_combinations)):
        for j in range(TOTAL_NODE_COUNT):
            if input_patterns[j, i] == 1:  # If the node fires for this pattern
                weights[j, :, i] = initial_weights[j, :]

    # Determine the signal reaching the output nodes for each input pattern
    for i in range(len(active_combinations)):
        for j in range(TOTAL_NODE_COUNT):
            outputs[j, i] = np.sum(weights[:, j, i])
    
    return input_patterns, weights, outputs


def filter_pattern_weights(initial_weights, input_patterns):
    """
    Filter the weights corresponding output signals for a given set of input patterns

    :param initial_weights: The initial weights of the network
    :param input_patterns: The matrix of input patterns

    :return: The weights for each input pattern and the output signals
    """

    # Define the dimensions of the input matrix
    weights = np.zeros((TOTAL_NODE_COUNT, TOTAL_NODE_COUNT, input_patterns.shape[1]))
    outputs = np.zeros((TOTAL_NODE_COUNT, input_patterns.shape[1]))

    # Assign the weights to the input patterns
    for i in range(input_patterns.shape[1]):
        for j in range(TOTAL_NODE_COUNT):
            if input_patterns[j, i] == 1:  # If the node fires for this pattern
                weights[j, :, i] = initial_weights[j, :]

    # Determine the signal reaching the output nodes for each input pattern
    for i in range(input_patterns.shape[1]):
        for j in range(TOTAL_NODE_COUNT):
            outputs[j, i] = np.sum(weights[:, j, i])
    
    return weights, outputs


def select_threshold(thresholds, outputs, min_count = 125000, max_count = 150000):
    """
    Determine the threshold value that results in a desired number of valid input patterns

    :param thresholds: The threshold values to test
    :param outputs: The signal reaching the output nodes for each input pattern
    :param min_count: The minimum number of valid input patterns
    :param max_count: The maximum number of valid input patterns

    :return: The threshold value that results in the desired number of valid input patterns
    """
    
    # Calculate the maximum signal for each input pattern
    max_signals = np.max(outputs, axis=0)

    # Initialize array to store the count of valid input patterns for each threshold
    valid_counts = np.zeros(len(thresholds))

    # Determine the count of valid input patterns for each threshold
    for i, th_value in enumerate(thresholds):
        valid_counts[i] = np.sum(max_signals <= th_value)

    # Find the index of the first threshold value that falls within the desired range
    valid_indices = np.where((valid_counts >= min_count) & (valid_counts <= max_count))[0]
    if valid_indices.size == 0:
        # If no threshold meets the criteria, return None
        return None
    else:
        # Choose the threshold with the most valid input patterns
        chosen_index = valid_indices[np.argmax(valid_counts[valid_indices])]
        threshold = thresholds[chosen_index]
        return threshold
    

def find_threshold(outputs, percentage = 0.25):
    """
    Determine the threshold that results in a desired percentage of valid input patterns

    :param outputs: The signal reaching the output nodes for each input pattern
    :param percentage: The desired percentage of valid input patterns

    :return: The threshold value that results in the desired percentage of valid input patterns
    """

    # Calculate the maximum signal for each input pattern
    max_signals = np.max(outputs, axis=0)

    # Determine the threshold value that results in the desired percentage of valid input patterns
    threshold = np.percentile(max_signals, 100 * percentage)

    return threshold


def search_threshold_with_ref(output, current_threshold=0.2405, target_count=137017, tol=1e-6, count_tol=0.20, step=1e-9):
    """
    Incrementally adjust the threshold starting from current_threshold, moving in small steps 
    until the valid pattern count is within 5% of target_count, or the threshold change 
    exceeds tol. 
    
    In addition to returning the calibrated threshold, the function also returns the valid pattern count.
    
    :param output: The signal reaching the output nodes for each input pattern
    :param current_threshold: The original threshold value (default 0.2405)
    :param target_count: The expected valid pattern count (default 317017)
    :param tol: The maximum allowed change in threshold from current_threshold (default 1e-6)
    :param step: The incremental step size for adjusting the threshold (default 1e-7)
    :return: A tuple (calibrated_threshold, valid_count) if found within tol, otherwise (None, None).
    """
    
    # Compute the maximum signal for each input pattern
    max_signals = np.max(output, axis=0)
    print(f"Max signals: {max_signals}")
    
    # Define the acceptable valid count range (5% of target_count)
    lower_bound = np.floor(target_count * (1 - count_tol))
    upper_bound = np.ceil(target_count * (1 + count_tol))

    print(f"Target count: {target_count}, Lower bound: {lower_bound}, Upper bound: {upper_bound}")

    # Define the endpoints of the allowed tolerance window
    lower_threshold = current_threshold - tol
    upper_threshold = current_threshold + tol

    # Endpoints of the valid count for the tolerance window
    valid_count_lower = np.sum(max_signals <= lower_threshold)
    valid_count_upper = np.sum(max_signals <= upper_threshold)

    valid_count_current = np.sum(max_signals <= current_threshold)

    print(f"Current valid count: {valid_count_current}, Lower valid count: {valid_count_lower}, Upper valid count: {valid_count_upper}")

            # Create a dataframe with the required information
    data = {
        'threshold_tolerance': [tol],
        'lower_th': [lower_threshold],
        'upper_th': [upper_threshold],
        'valid_count_tolerance_%': [count_tol*100],
        'lower_bound': [lower_bound],
        'upper_bound': [upper_bound],
        'initial_valid_count': [valid_count_current],
        'valid_count_lower': [valid_count_lower],
        'valid_count_upper': [valid_count_upper],
        'final_threshold': [current_threshold],
        'final_valid_count': [valid_count_current]
    }
    df = pd.DataFrame(data)
    
    # Early termination: if both endpoints yield counts that are either both too low or both too high,
    # then no threshold within the tol window can bring the count into the acceptable range.
    if (valid_count_lower < lower_bound and valid_count_upper < lower_bound) or (valid_count_lower > upper_bound and valid_count_upper > upper_bound):
        print("No threshold within the tolerance window can bring the count into the acceptable range.")
        df.loc[0, 'final_threshold'],  df.loc[0, 'final_valid_count'] = np.nan, np.nan
        return None, None, df

    new_threshold = current_threshold
    valid_count = valid_count_current

    if valid_count_current == target_count:
        df.loc[0, 'final_threshold'],  df.loc[0, 'final_valid_count'] = current_threshold, valid_count
        return current_threshold, valid_count, df
    
    # If current valid count is too low increase the threshold
    elif valid_count_current < target_count:
        # Incrementally increase the threshold until it exceeds the upper endpoint.
        while valid_count < target_count and new_threshold <= upper_threshold:
            new_threshold += step
            valid_count = np.sum(max_signals <= new_threshold)
        df.loc[0, 'final_threshold'],  df.loc[0, 'final_valid_count'] = new_threshold, valid_count
        return new_threshold, valid_count, df
            
    elif valid_count_current > target_count:
        # Incrementally decrease the threshold until it falls below the lower endpoint.
        while valid_count > target_count and new_threshold >= lower_threshold:
            new_threshold -= step
            valid_count = np.sum(max_signals <= new_threshold)
        df.loc[0, 'final_threshold'],  df.loc[0, 'final_valid_count'] = new_threshold, valid_count
        return new_threshold, valid_count, df

    # If no acceptable threshold is found within the allowed tolerance, abandon the search.
    df.loc[0, 'final_threshold'],  df.loc[0, 'final_valid_count'] = np.nan, np.nan
    return None, None, df


def get_valid_inputs_and_outputs(weights, input_patterns, outputs, threshold):
    """
    Determine the valid input patterns based on the chosen threshold

    :param weights: The input matrix
    :param input_patterns: The matrix of input patterns
    :param outputs: The signal reaching the output nodes for each input pattern
    :param threshold: The threshold value used to determine valid input patterns

    :return: The valid input patterns, the input matrix for valid input patterns, and the output signals for valid input patterns
    """

    # Calculate the valid input pattern indices based on the chosen threshold
    valid_input_pattern_indices = np.where(np.max(outputs, axis=0) <= threshold)[0]

    valid_input_patterns = input_patterns[:, valid_input_pattern_indices]
    valid_outputs = outputs[:, valid_input_pattern_indices]
    valid_weights = weights[:, :, valid_input_pattern_indices]

    return valid_input_patterns, valid_weights, valid_outputs


def get_output_statistics(outputs):
    """
    Calculate the variance in the output patterns

    :param outputs: The signal reaching the output nodes for each input pattern

    :return: The variance in the output patterns, the average variance, and the standard deviation of the variance
    """

    variance = np.max(outputs, axis=0) - np.min(outputs, axis=0)
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

    max_outputs = np.max(output, axis=0)  # Maximum signal reaching each of the nodes for a given IP

    # Plot maximum signal reaching output nodes
    plt.hist(max_outputs, color=[0.3, 0.9, 0.9], bins=30, edgecolor='black')  # Plot the histogram
    # plt.xlim(0.2, 0.32)
    plt.axvline(threshold, color='red', linewidth=3)  # To show the determined threshold
    plt.title("Maximum signal reaching output nodes", fontsize=12)
    plt.xlabel("Maximum output signal", fontsize=10)
    plt.ylabel("Number of input patterns", fontsize=10)
    plt.text(0.28, 22800, f'Threshold = {threshold:.4f}', fontsize=10)
    plt.gca().set_facecolor('w')
    plt.savefig(os.path.join(BASE_PATH, f'figures/threshold.png'), dpi=300)
    plt.show()

    # Plot the variance in output patterns
    plt.hist(variance, color=[0.1, 0.9, 0.3], bins=30, edgecolor='black')
    plt.title("Output signal variance (difference between minimum and maximum)", fontsize=12)
    plt.xlabel("Signal variance", fontsize=10)
    plt.ylabel("Number of input patterns", fontsize=10)
    plt.text(0.1, 10000, f'Mean variance = {avg_variance:.4f}', fontsize=10)
    plt.text(0.1, 9000, f'SD variance = {sd_variance:.4f}', fontsize=10)
    plt.gca().set_facecolor('w')
    plt.savefig(os.path.join(BASE_PATH, f'figures/output_signal_variance.png'), dpi=300)
    plt.show()

    
def main():

    # Load config
    with open(os.path.join(BASE_PATH, 'config.json')) as f:
        config = json.load(f)


    ##### ----- SELECT PATTERNS ----- #######

    initial_weights = np.loadtxt(os.path.join(BASE_PATH, config['initial_weights_file']), delimiter=',')

    input_patterns, weights, outputs = get_all_input_and_output_patterns(initial_weights)

    # Define threshold values to attempt
    thresholds = [0.2, 0.2405, 0.28, 0.275, 0.278]
    threshold = select_threshold(thresholds, outputs)
    # threshold = find_threshold(output, percentage=0.230755)

    valid_input_patterns, valid_weights, valid_outputs = get_valid_inputs_and_outputs(threshold, weights, input_patterns, outputs)

    # Save the valid input patterns, weights, and output signals to .csv files
    np.savetxt(os.path.join(BASE_PATH, config['weights_per_pattern_file']), valid_weights.reshape(-1, valid_weights.shape[-1]), delimiter=",")
    np.savetxt(os.path.join(BASE_PATH, config['input_patterns_file']), valid_input_patterns, delimiter=",", fmt='%d')
    np.savetxt(os.path.join(BASE_PATH, config['initial_outputs_file']), valid_outputs, delimiter=",", fmt='%.4f')

    variance, avg_variance, sd_variance = get_output_statistics(valid_outputs)

    plot_signal_statistics(outputs, threshold, variance, avg_variance, sd_variance)


if __name__ == '__main__':
    main()

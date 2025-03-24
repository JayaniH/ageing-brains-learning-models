import json
import os

import numpy as np
from matplotlib import pyplot as plt

from constants import BASE_PATH, COLOURS, TOTAL_NODE_COUNT, THRESHOLD
from learning import execute_young_learning
from utils import roulette_wheel_selection


def initialise_weights(n):
    """
    Initialise the weights of the model

    :param n: Number of nodes

    :return weights: Array of weights
    """

    weights = np.zeros((n, n)) 

    for i in range(n):
        # Create 30 random numbers for each edge of that node using normal distribution
        w_i = 0.2 * np.random.randn(n) + 1

        # Scale to sum to 1
        w_i_scaled = w_i / np.sum(w_i)

        weights[i, :] = w_i_scaled

    return weights


def get_output_statistics(weights):
    """
    Get the output statistics of the model

    :param weights: Array of weights

    :return variance: Variance of the output signal
    :return minimum: Minimum of the output signal
    :return maximum: Maximum of the output signal
    """

    output = np.sum(weights, axis=0)

    variance = np.var(output)
    minimum = np.min(output)
    maximum = np.max(output)

    return variance, minimum, maximum


def plot_weight_distribution(weights):
    """
    Plot the weight distribution of the model

    :param initial_weights: Initial weights of the model
    """

    output = np.sum(weights, axis=0)

    plt.hist(initial_weights.flatten(), bins=10, color=COLOURS['LIGHT_BLUE'], alpha=0.7, edgecolor='black')
    plt.ylabel('Number of edges', fontsize=10)
    plt.xlabel('Weight', fontsize=10)
    plt.title('Initial weight distribution of the model', fontsize=12)
    plt.savefig(os.path.join(BASE_PATH, f'figures/weights.png'), dpi=300)
    plt.show()

    plt.hist(output, bins=10, color=[0.2, 0.9, 0.2], alpha=0.7, edgecolor='black')
    plt.ylabel('Number of nodes', fontsize=10)
    plt.xlabel('Output signal', fontsize=10)
    plt.title('Initial output signal distribution of the model', fontsize=12)
    plt.savefig(os.path.join(BASE_PATH, f'figures/output_distribution.png'), dpi=300)
    plt.show()

    plt.imshow(output.reshape(-1, 1), cmap='winter', aspect='auto')
    plt.colorbar()
    plt.xticks([])
    plt.title('Initial output signals of the model', fontsize=12)
    plt.ylabel('Nodes', fontsize=10)
    plt.savefig(os.path.join(BASE_PATH, f'figures/outputs.png'), dpi=300)
    plt.show()


def add_prior_knowledge_with_rw_selection(weights, node_count=6, edge_count=6, learning_rate=1.5):
    """
    Add prior knowledge to the weights based on roulette wheel selection

    :param weights: Array of weights
    :param node_count: Number of nodes to be changed
    :param edge_count: Number of edges to be changed
    :param learning_rate: Learning rate for the weights

    :return weights: Array of weights with prior knowledge added
    :return mask: Mask of the weights
    """

    mask = np.ones_like(weights)

    # Select node_count number of nodes using roulette wheel selection
    print(np.sum(weights, axis=0))
    node_indices = roulette_wheel_selection(np.sum(weights, axis=0), node_count)

    for i, node_index in enumerate(node_indices):
        edge_indices = roulette_wheel_selection(weights[:, node_index], edge_count)
        weights[edge_indices, node_index] *= learning_rate
        mask[edge_indices, node_index] = 0
    
    return weights, mask


def add_prior_knowledge_with_learning(initial_weights, input_pattern):
    """
    Add prior knowledge to the weights based on the input pattern

    :param weights: Array of weights
    :param input_pattern: Input pattern

    :return weights: Array of weights with prior knowledge added
    """

    weights, _, _, _, learnable_weights_filter = execute_young_learning(initial_weights, input_pattern, THRESHOLD)

    return weights, learnable_weights_filter


if __name__ == "__main__":

    # Load config
    with open(os.path.join(BASE_PATH, 'config.json'), 'r') as f:
        config = json.load(f)

    # Reproducibility
    np.random.seed(42)


    ######## INITIALISE WEIGHTS ########

    try:
        initial_weights = np.loadtxt(os.path.join(BASE_PATH, config['initial_weights_file']), delimiter=',')
    except Exception as e:
        initial_weights = initialise_weights(TOTAL_NODE_COUNT)
        np.savetxt(os.path.join(BASE_PATH, config['initial_weights_file']), initial_weights, delimiter=',')
    

    ######## ADD PRIOR KNOWLEDGE ########

    # Add prior knowledge to the weights based on the input pattern
    input_patterns = np.loadtxt(os.path.join(BASE_PATH, config['input_patterns_file']), delimiter=',')

    # Select a random input pattern
    random_pattern_index = np.random.randint(input_patterns.shape[1])

    # Save the random pattern index
    with open(os.path.join(BASE_PATH, config['random_input_pattern_index_file']), 'w') as f:
        f.write(str(random_pattern_index))

    input_pattern = input_patterns[:, random_pattern_index]
    weights1, filter1 = add_prior_knowledge_with_learning(initial_weights, input_pattern)


    # Add prior knowledge to the weights based on roulette wheel selection
    weights2, filter2 = add_prior_knowledge_with_rw_selection(initial_weights)

    np.savetxt(os.path.join(BASE_PATH, config['pattern_learned_weights_file']), weights1, delimiter=',')
    np.savetxt(os.path.join(BASE_PATH, config['pattern_learned_weights_mask_file']), filter1, delimiter=',')

    np.savetxt(os.path.join(BASE_PATH, config['rw_learned_weights_file']), weights2, delimiter=',')
    np.savetxt(os.path.join(BASE_PATH, config['rw_learned_weights_mask_file']), filter2, delimiter=',')


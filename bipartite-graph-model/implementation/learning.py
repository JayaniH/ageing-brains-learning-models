import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import roulette_wheel_selection
from constants import BASE_PATH, ACTIVE_NODE_COUNT, COLOURS, THRESHOLD


def execute_young_learning(initial_weights, input_pattern, threshold, learning_rate=1.5, learnable_weights_filter=None):
    """
    Execute the Young learning process.

    :param initial_weights: The initial weights of the edges
    :param input_pattern: The input pattern
    :param threshold: The chosen threshold value
    :param learning_rate: The learning rate
    :param learnable_weights_filter: The mask for learnable weights

    :return: Weights after learning, the number of iterations, the nodes that fired, and the signals that exceeded the threshold
    """

    weights = initial_weights.copy()

    if learnable_weights_filter is not None:
        learnable_weights_filter = learnable_weights_filter.copy()
    else:
        learnable_weights_filter = np.ones_like(weights)

    output_signals = np.dot(weights.T, input_pattern)
    active_output_node_count = np.where(output_signals >= threshold)[0].shape[0]

    iteration_count = 0
    active_output_nodes = np.full(ACTIVE_NODE_COUNT, -1)

    while active_output_node_count < ACTIVE_NODE_COUNT:
        selected_output_node_index = roulette_wheel_selection(output_signals)

        if selected_output_node_index in active_output_nodes:
            continue

        active_edges = np.where((input_pattern != 0) & (learnable_weights_filter[:, selected_output_node_index] != 0))[0]

        weights[active_edges, selected_output_node_index] *= learning_rate
        learnable_weights_filter[active_edges, selected_output_node_index] = 0

        output_signals[selected_output_node_index] = np.dot(weights[:, selected_output_node_index], input_pattern)

        if output_signals[selected_output_node_index] >= threshold:
            active_output_nodes[active_output_node_count] = selected_output_node_index
            active_output_node_count += 1
        
        iteration_count += 1

    active_output_node_signals = output_signals[active_output_nodes]
    
    return weights, iteration_count, active_output_nodes, active_output_node_signals, learnable_weights_filter


def execute_old_learning(initial_weights, input_pattern, threshold, learnable_weights_filter=None):
    """
    Execute the Old learning process.

    :param initial_weights: The initial weights of the edges
    :param input_pattern: The input pattern
    :param threshold: The chosen threshold value
    :param learnable_weights_filter: The mask for learnable weights

    :return: Weights after learning, the number of iterations, the nodes that fired, and the signals that exceeded the threshold
    """

    weights = initial_weights.copy()

    if learnable_weights_filter is not None:
        learnable_weights_filter = learnable_weights_filter.copy()
    else:
        learnable_weights_filter = np.ones_like(weights)        

    learnable_weights = weights * learnable_weights_filter
    
    output_signals = np.dot(weights.T, input_pattern)
    active_output_node_count = np.where(output_signals >= threshold)[0].shape[0]

    iteration_count = 0
    active_output_nodes = np.full(ACTIVE_NODE_COUNT, -1)
    excluded_nodes = []

    while active_output_node_count < ACTIVE_NODE_COUNT:
        selected_output_node_index = roulette_wheel_selection(output_signals)

        if selected_output_node_index in active_output_nodes or selected_output_node_index in excluded_nodes:
            continue

        # incoming_edges = weights[:, selected_output_node_index] * input_pattern
        incoming_edges = learnable_weights[:, selected_output_node_index] * input_pattern
        active_edge_indices = np.where(incoming_edges != 0)[0]

        chosen_edge_index = roulette_wheel_selection(weights[active_edge_indices, selected_output_node_index])

        chosen_edge_input_index = active_edge_indices[chosen_edge_index]

        coin_toss = np.random.randint(0, 2)
        offset = 0

        while output_signals[selected_output_node_index] < threshold and offset < len(weights[chosen_edge_input_index, :]):
            offset += 1

            if coin_toss == 0:
                next_edge_index = (selected_output_node_index + offset) % len(weights[chosen_edge_input_index, :])
            else:
                next_edge_index = (selected_output_node_index - offset) % len(weights[chosen_edge_input_index, :])

            # If the next edge belongs to an active output node or already re-wired, it cannot be used
            # If the weight is not learnable, it cannot be used
            if learnable_weights[chosen_edge_input_index, next_edge_index] == 0:
                continue

            weights[chosen_edge_input_index, selected_output_node_index] += weights[chosen_edge_input_index, next_edge_index]
            weights[chosen_edge_input_index, next_edge_index] = 0 # update the weight in the input matrix

            learnable_weights_filter[chosen_edge_input_index, selected_output_node_index], learnable_weights_filter[chosen_edge_input_index, next_edge_index] = 0, 0
            learnable_weights = weights * learnable_weights_filter

            output_signals = np.dot(weights.T, input_pattern)

            iteration_count += 1

        
        if offset == len(weights[chosen_edge_input_index, :]):
            excluded_nodes.append(selected_output_node_index)

        if output_signals[selected_output_node_index] >= threshold:
            active_output_nodes[active_output_node_count] = selected_output_node_index
            active_output_node_count += 1
    
    active_output_node_signals = output_signals[active_output_nodes]

    return weights, iteration_count, active_output_nodes, active_output_node_signals
        

def excute_learning_process(initial_weights, input_patterns, threshold, learning_type='young', learnable_weights_filter = None, save=False, tags=[]):
    """
    Execute the learning process.

    :param initial_weights: The initial weights of the edges
    :param input_patterns: The input patterns
    :param threshold: The chosen threshold value for output node activation
    :param learnable_weights_filter: The mask for learnable weights
    :param learning_type: The type of learning process to execute
    :param save: If True, save the results to .csv files
    :param tags: List of tags to add to the filenames

    :return: The number of iterations for each input pattern
    """

    if learning_type not in ['young', 'old']:
        raise ValueError("Invalid learning type")
    
    iteration_counts = np.zeros(input_patterns.shape[1])
    all_active_output_signals = np.zeros((ACTIVE_NODE_COUNT, input_patterns.shape[1]))
    all_active_output_nodes = np.full((ACTIVE_NODE_COUNT, input_patterns.shape[1]), -1)

    for pattern_index in range(input_patterns.shape[1]):
        input_pattern = input_patterns[:, pattern_index]

        if learning_type == 'young':
            _, iteration_count, active_output_nodes, active_output_node_signals, _ = execute_young_learning(initial_weights, input_pattern, threshold, learnable_weights_filter)
        
        elif learning_type == 'old':
            _, iteration_count, active_output_nodes, active_output_node_signals = execute_old_learning(initial_weights, input_pattern, threshold, learnable_weights_filter)

        all_active_output_signals[:, pattern_index] = active_output_node_signals
        all_active_output_nodes[:, pattern_index] = active_output_nodes
        iteration_counts[pattern_index] = iteration_count

    if save:
        tags_str = f"_{'_'.join(tags)}" if tags else ''
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_iteration_counts{tags_str}.csv'), iteration_counts, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_active_output_signals{tags_str}.csv'), all_active_output_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_active_output_nodes{tags_str}.csv'), all_active_output_nodes, delimiter=',', fmt='%d')

    return iteration_counts
    

def plot_learning_histograms(learning_data, labels, colors, grouped=True, filename='learning_iterations'):
    """
    Plot histograms of the number of iterations for Young and Old learning.

    :param learning_data: List of arrays with the number of iterations for each input pattern
    :param labels: List of labels for the datasets
    :param colors: List of colors for the datasets
    :param grouped: If True, bars are grouped by dataset; otherwise, bars are overlapped
    """

    maximum_iterations = int(max([max(data) for data in learning_data]))
    minimum_iterations = int(min([min(data) for data in learning_data]))

    histograms = [np.histogram(data, bins=np.arange(minimum_iterations, max(maximum_iterations, 10) + 2), range=(minimum_iterations, max(maximum_iterations, 10) + 1))[0] for data in learning_data]

    maximum_count = max([max(histogram) for histogram in histograms])
    
    bar_width = 0.35 if grouped else 0.7
    x_positions = np.arange(1, len(histograms[0]) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    bar_charts = []
    for i, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        if grouped:
            x_pos = x_positions + i * bar_width
        else:
            x_pos = x_positions
        bar_chart = ax.bar(x_pos, histogram, width=bar_width, color=color, label=label)
        bar_charts.append(bar_chart)

    plt.xlabel("Number of iterations", fontsize=10)
    plt.ylabel("Input pattern count", fontsize=10)
    plt.ylim(0, maximum_count + (maximum_count * 0.15))
    plt.title("Learning iteration comparison", fontsize=12)
    plt.xticks(x_positions + bar_width / 2 if grouped else x_positions, [str(i) for i in range(minimum_iterations, len(histograms[0]) + minimum_iterations)], rotation=45)
    plt.legend(fontsize=13)

    # Add text annotations
    def autolabel(bars, total_count):
        """Attach a text label above each bar displaying its height."""
        for bar in bars:
            height = bar.get_height()
            count = int(height)
            if count > 0:
                percentage = height / total_count * 100
                ax.annotate(f'{count} ({percentage:.2f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 5),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=5, rotation='vertical')

    for histogram, bar_chart in zip(histograms, bar_charts):
        autolabel(bar_chart, sum(histogram))

    plt.gca().set_facecolor('w')
    plt.tight_layout()
    filename = f'{filename}_grouped.png' if grouped else f'{filename}.png'
    plt.savefig(os.path.join(BASE_PATH, 'figures/', filename), dpi=300)
    plt.show()


if __name__ == "__main__":

    with open(os.path.join(BASE_PATH, 'config.json'), 'r') as f:
        config = json.load(f)

    input_patterns = np.loadtxt(os.path.join(BASE_PATH, config['input_patterns_file']), delimiter=',').astype(int)
    initial_weights = np.loadtxt(os.path.join(BASE_PATH, config['initial_weights_file']), delimiter=',')

    iteration_counts_young = excute_learning_process(initial_weights, input_patterns, THRESHOLD, learning_type='young', save=True)
    iteration_counts_old = excute_learning_process(initial_weights, input_patterns, THRESHOLD, learning_type='old', save=True)

    plot_learning_histograms(
        [iteration_counts_young, iteration_counts_old], 
        ['Young', 'Old'], 
        [COLOURS['YELLOW'], COLOURS['DARK_BLUE']],
        grouped=True
    )


    ### OLD LEARNING WITH PRIOR KNOWLEDGE (PRE-LOADED WEIGHTS) ###

    # Weights learned from random pattern
    learned_weights = np.loadtxt(os.path.join(BASE_PATH, config['pattern_learned_weights_file']), delimiter=',')
    learnable_weights_filter = np.loadtxt(os.path.join(BASE_PATH, config['pattern_learned_weights_mask_file']), delimiter=',').astype(int)

    # Remove the pattern used to pre-load the weights
    with open(os.path.join(BASE_PATH, config['random_input_pattern_index_file']), 'w') as f:
        input_pattern_index = f.readline()
    input_patterns = np.delete(input_patterns, input_pattern_index, axis=1)

    iteration_counts_old_learned = excute_learning_process(learned_weights, input_patterns, THRESHOLD, learning_type='old', save=True, tags=['pattern_learned'])

    plot_learning_histograms(
        [iteration_counts_old, iteration_counts_old_learned], 
        ['Old', 'Old with prior knowledge'], 
        [COLOURS['DARK_BLUE'], COLOURS['LIGHT_BLUE']],
        grouped=True,
        filename='old_learning_comparison_pattern_learned'
    )


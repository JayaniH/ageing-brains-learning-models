import json
import os
import numpy as np
import matplotlib.pyplot as plt

from utils import roulette_wheel_selection
from constants import BASE_PATH, ACTIVE_NODE_COUNT, COLOURS


def execute_young_learning(initial_weights, input_pattern, threshold, learning_rate=1.5):
    """
    Execute the Young learning process.

    :param initial_weights: The initial weights of the edges
    :param input_pattern: The input pattern
    :param threshold: The chosen threshold value
    :param learning_rate: The learning rate

    :return: Weights after learning, the number of iterations, the nodes that fired, and the signals that exceeded the threshold
    """

    # np.random.seed(42)

    weights = initial_weights.copy()

    output_signals = np.dot(weights.T, input_pattern)
    active_output_node_count = np.where(output_signals >= threshold)[0].shape[0]

    iteration_count = 0
    active_output_nodes = np.full(ACTIVE_NODE_COUNT, -1)

    while active_output_node_count < ACTIVE_NODE_COUNT:
        selected_output_node_index = roulette_wheel_selection(output_signals)

        if selected_output_node_index in active_output_nodes:
            continue

        weights[:, selected_output_node_index] = learning_rate * weights[:, selected_output_node_index]
        output_signals[selected_output_node_index] = np.dot(weights[:, selected_output_node_index], input_pattern)

        if output_signals[selected_output_node_index] >= threshold:
            active_output_nodes[active_output_node_count] = selected_output_node_index
            active_output_node_count += 1
        
        iteration_count += 1

    active_output_node_signals = output_signals[active_output_nodes]
    
    return weights, iteration_count, active_output_nodes, active_output_node_signals


def execute_old_learning(initial_weights, input_pattern, threshold):
    """
    Execute the Old learning process.

    :param initial_weights: The initial weights of the edges
    :param input_pattern: The input pattern
    :param threshold: The chosen threshold value

    :return: Weights after learning, the number of iterations, the nodes that fired, and the signals that exceeded the threshold
    """

    weights = initial_weights.copy()

    output_signals = np.dot(weights.T, input_pattern)
    active_output_node_count = np.where(output_signals >= threshold)[0].shape[0]

    iteration_count = 0
    active_output_nodes = np.full(ACTIVE_NODE_COUNT, -1)
    excluded_nodes = []

    while active_output_node_count < ACTIVE_NODE_COUNT:
        selected_output_node_index = roulette_wheel_selection(output_signals)

        if selected_output_node_index in active_output_nodes or selected_output_node_index in excluded_nodes:
            continue

        incoming_edges = weights[:, selected_output_node_index] * input_pattern
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

            weights[chosen_edge_input_index, selected_output_node_index] += weights[chosen_edge_input_index, next_edge_index]
            weights[chosen_edge_input_index, next_edge_index] = 0 # update the weight in the input matrix

            output_signals = np.dot(weights.T, input_pattern)

            iteration_count += 1

        
        if offset == len(weights[chosen_edge_input_index, :]):
            excluded_nodes.append(selected_output_node_index)

        if output_signals[selected_output_node_index] >= threshold:
            active_output_nodes[active_output_node_count] = selected_output_node_index
            active_output_node_count += 1
    
    active_output_node_signals = output_signals[active_output_nodes]

    return weights, iteration_count, active_output_nodes, active_output_node_signals
        

def excute_learning_process(initial_weights, input_patterns, threshold, learning_type='young', save=False):
    """
    Execute the learning process.

    :param initial_weights: The initial weights of the edges
    :param input_patterns: The input patterns
    :param threshold: The chosen threshold value for output node activation
    :param learning_type: The type of learning process to execute
    :param save: If True, save the results to .csv files

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
            _, iteration_count, active_output_nodes, active_output_node_signals = execute_young_learning(initial_weights, input_pattern, threshold)
        
        elif learning_type == 'old':
            _, iteration_count, active_output_nodes, active_output_node_signals = execute_old_learning(initial_weights, input_pattern, threshold)

        all_active_output_signals[:, pattern_index] = active_output_node_signals
        all_active_output_nodes[:, pattern_index] = active_output_nodes
        iteration_counts[pattern_index] = iteration_count

    if save:
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_iteration_counts.csv'), iteration_counts, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_active_output_signals.csv'), all_active_output_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, f'results/{learning_type}_active_output_nodes.csv'), all_active_output_nodes, delimiter=',', fmt='%d')

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
    plt.title("Learning iterations", fontsize=12)
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
    plt.savefig(os.path.join(BASE_PATH, 'figures', filename), dpi=300)
    plt.show()

if __name__ == "__main__":

    with open(os.path.join(BASE_PATH, 'config.json'), 'r') as f:
        config = json.load(f)

    input_patterns = np.loadtxt(os.path.join(BASE_PATH, config['input_patterns_file']), delimiter=',').astype(int)
    initial_weights = np.loadtxt(os.path.join(BASE_PATH, config['initial_weights_file']), delimiter=',')

    threshold = 0.2405

    iteration_counts_young = excute_learning_process(initial_weights, input_patterns, threshold, learning_type='young', save=True)
    iteration_counts_old = excute_learning_process(initial_weights, input_patterns, threshold, learning_type='old', save=True)

    plot_learning_histograms(
        [iteration_counts_young, iteration_counts_old], 
        ['Young', 'Old'], 
        [COLOURS['young'], COLOURS['old']],
        grouped=True
    )


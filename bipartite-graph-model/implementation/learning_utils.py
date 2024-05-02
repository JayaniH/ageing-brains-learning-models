import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from roulette_wheel_selection import roulette_wheel_selection
from constants import BASE_PATH, ACTIVE_NODE_COUNT


def execute_young_learning(valid_input, valid_output, threshold, save = False):
    """
    Execute the Young learning process.

    :param valid_input: The valid input signals
    :param valid_output: The valid output signals
    :param active_node_count: The number of active nodes
    :param threshold: The chosen threshold value
    :param save: If True, save the results to .csv files

    :return: The number of iterations for each input pattern, the signals that exceeded the threshold, and the nodes that fired
    """

    # Initialize arrays to store results
    young_iteration_count = np.zeros(valid_output.shape[1])
    young_over_th_signals = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))
    young_nodes_fired = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))

    # Loop over each input pattern
    for i in range(valid_output.shape[1]):
        loopcount = 0  # Initialize iteration count
        j = 0
        while True:
            chosen_output_node = roulette_wheel_selection(valid_output[:, i])
            if chosen_output_node in young_nodes_fired[:, i]:
                continue

            activated_input_nodes = np.where(valid_input[:, chosen_output_node, i])[0] # The value is non-zero only in activated nodes as set previously
            active_edges = valid_input[activated_input_nodes, chosen_output_node, i]

            chosen_edge_index = roulette_wheel_selection(active_edges)
            chosen_edge = active_edges[chosen_edge_index]

            ### OLD APPROACH ###
            # new_output = valid_output[chosen_output_node, i] + (1 - chosen_edge) / 2
            ### OR ###
            # new_weight = chosen_edge + (1 - chosen_edge) / 2
            # new_output = valid_output[chosen_output_node, i] + new_weight  # TODO: Check if this is correct. Does it double count the chosen edge?
            ### END OLD APPROACH ###

            ### NEW APPROACH ###
            learning_rate = 3
            c = 0
            new_weight = (learning_rate * chosen_edge) + c
            valid_input[activated_input_nodes[chosen_edge_index], chosen_output_node, i] = new_weight  # update the weight in the input matrix
            weight_diff = new_weight - chosen_edge
            new_output = valid_output[chosen_output_node, i] + weight_diff
            valid_output[chosen_output_node, i] = new_output  # update the output value in the output matrix
            ### END NEW APPROACH ###

            loopcount += 1

            if new_output >= threshold:
                young_over_th_signals[j, i] = new_output
                young_nodes_fired[j, i] = chosen_output_node
                j += 1

            if np.sum(young_over_th_signals[:, i] != 0) == 6:
                break

        young_iteration_count[i] = loopcount
    
    if save:
        young_output_patterns = np.zeros((valid_output.shape[0], valid_output.shape[1]))  # output patterns
        for n in range(young_output_patterns.shape[1]):
            young_output_patterns[young_nodes_fired[:, n].astype(int), n] = 1  # transform them into 1

        np.savetxt(os.path.join(BASE_PATH, 'data/young_output_patterns.csv'), young_output_patterns.astype(int), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'data/young_output_nodes_fired.csv'), young_nodes_fired.astype(int), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'data/young_over_th_signals.csv'), young_over_th_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, 'data/young_iteration_count.csv'), young_iteration_count.astype(int), delimiter=',', fmt='%d')

    return young_iteration_count, young_over_th_signals, young_nodes_fired


def execute_old_learning(valid_input, valid_output, threshold, save = False):
    """
    Execute the Old learning process.

    :param valid_input: The valid input signals
    :param valid_output: The valid output signals
    :param active_node_count: The number of active nodes
    :param threshold: The chosen threshold value
    :param save: If True, save the results to .csv files

    :return: The number of iterations for each input pattern, the signals that exceeded the threshold, the nodes that fired, and the excluded nodes
    """

    old_iteration_count = np.zeros(valid_output.shape[1])  # to store how many iterations learning took
    excluded_nodes = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))  # to store which nodes were excluded
    old_over_th_signals = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))  # to store the nodes that fired
    old_nodes_fired = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))  # matrix with the index of the nodes fired

    for i in range(valid_output.shape[1]):  # for each input pattern
        loopcount = 0  # amount of iterations
        j = 0  # counter for the nodes that fired

        while True:  # for all the nodes
            chosen_output_node = roulette_wheel_selection(valid_output[:, i])
            if chosen_output_node in old_nodes_fired[:, i] or chosen_output_node in excluded_nodes[:, i]:  # if node has been previously selected, choose another one
                continue

            activated_input_nodes = np.where(valid_input[:, chosen_output_node, i])[0]  # get the active nodes for the IP
            active_edges = valid_input[activated_input_nodes, chosen_output_node, i]  # get the values for the active node

            # Choose an edge based on the active edges
            chosen_edge_index = roulette_wheel_selection(active_edges)

            chosen_edge = active_edges[chosen_edge_index]
            ###
            new_output = valid_output[chosen_output_node, i] 
            ### OR ###
            # new_output = valid_output[chosen_output_node, i] + chosen_edge # TODO: Check if this is correct. Why do we add the chosen edge to the output value?
            ###

            coin_toss = np.random.randint(0, 2)
            next_edge_position = 0
            # next_edges = []

            while True:
                # Increment the counter
                next_edge_position += 1

                # If the maximum number of iterations is reached, exclude the node and break the loop
                if next_edge_position == 6:
                    loopcount -= 5 # TODO: Check if this is correct. Why do we need to decrement the loop count?
                    excluded_nodes[j, i] = chosen_output_node
                    break

                # Choose the direction based on the random value x
                if coin_toss == 0:
                    search_array = active_edges
                    active_edges_circular = np.hstack((search_array, search_array))
                else:
                    search_array = np.flipud(active_edges)
                    active_edges_circular = np.hstack((search_array, search_array))

                chosen_edge_position = np.where(chosen_edge == search_array)[0][0]

                # Select the second edge and update the sum of edges
                next_edge = active_edges_circular[chosen_edge_position + next_edge_position]

                ###
                # next_edges.append(next_edge)
                # update = sum(next_edges) # TODO: Check if this is correct. Doesn't this count the same edge multiple times? Shouldn't we only add the second edge from the current iteration?
                # new_output += update
                ### OR ###
                valid_input[activated_input_nodes[chosen_edge_index], chosen_output_node, i] += next_edge  # update the weight in the input matrix
                new_output += next_edge
                valid_output[chosen_output_node, i] = new_output    # update the output value in the output matrix
                ###

                # Increment the loop count
                loopcount += 1

                # If the threshold is exceeded, record the node
                if new_output >= threshold:
                    old_over_th_signals[j, i] = new_output
                    old_nodes_fired[j, i] = chosen_output_node
                    j+=1
                    break
            
            # If all nodes have exceeded the threshold, break the loop
            if np.count_nonzero(old_over_th_signals[:, i]) == 6:
                break
        
        old_iteration_count[i] = loopcount

    if save:
        # To later measure old output pattern similarity analysis
        old_output_patterns = np.zeros((valid_output.shape[0], valid_output.shape[1])) # Creating a matrix to see the nodes that fire
        # over_th_old = np.zeros((active_node_count, len(old_iteration_count))) # Putting in the index of the nodes

        for n in range(old_output_patterns.shape[1]):
            old_output_patterns[old_nodes_fired[:, n].astype(int), n] = 1  # transform them into 1

        # Store values in .csv files
        np.savetxt(os.path.join(BASE_PATH, 'data/old_output_patterns.csv'), old_output_patterns, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'data/old_output_nodes_fired.csv'), old_nodes_fired, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'data/old_over_th_signals.csv'), old_over_th_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, 'data/old_iteration_count.csv'), old_iteration_count, delimiter=',', fmt='%d')

    return old_iteration_count, old_over_th_signals, old_nodes_fired, excluded_nodes


def plot_learning_histograms(learning_data, labels, colors, grouped=True):
    """
    Plot histograms of the number of iterations for Young and Old learning.

    :param learning_data: List of arrays with the number of iterations for each input pattern
    :param labels: List of labels for the datasets
    :param colors: List of colors for the datasets
    :param grouped: If True, bars are grouped by dataset; otherwise, bars are overlapped
    """

    # Calculate histogram for each dataset
    histograms = [np.histogram(data, bins=np.arange(6, 15), range=(6, 15))[0] for data in learning_data]
    
    # Define width of bars
    bar_width = 0.35 if grouped else 0.7
    
    # Define positions for the bars
    x_positions = np.arange(1, len(histograms[0]) + 1)
    
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot bars for each dataset
    bar_charts = []
    for i, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        if grouped:
            x_pos = x_positions + i * bar_width
        else:
            x_pos = x_positions
        bar_chart = ax.bar(x_pos, histogram, width=bar_width, color=color, label=label)
        bar_charts.append(bar_chart)

    # Add labels, title, legend, etc.
    plt.xlabel("Number of iterations", fontsize=10)
    plt.ylabel("Valid input patterns", fontsize=10)
    plt.ylim(0, 150000)
    plt.title("Number of Iterations for Young and Old learning", fontsize=12)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(6, len(histograms[0]) + 6)])
    plt.legend(fontsize=13)

    # Add text annotations
    def autolabel(bars):
        """Attach a text label above each bar displaying its height."""

        for bar in bars:
            height = bar.get_height()
            count = int(height)
            percentage = height / sum(histograms[0]) * 100
            if grouped:
                ax.annotate(f'{count} ({percentage:.2f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', rotation='vertical')  # Rotate text vertically
            else:
                ax.annotate(f'{count}\n({percentage:.2f}%)',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),  # 3 points vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom')

    for bar_chart in bar_charts:
        autolabel(bar_chart)

    # Show plot
    plt.gca().set_facecolor('w')
    plt.tight_layout()
    if grouped:
        plt.savefig(os.path.join(BASE_PATH, 'figures/iterations_young_old_grouped.png'))
    else:
        plt.savefig(os.path.join(BASE_PATH, 'figures/iterations_young_old.png'))
    plt.show()

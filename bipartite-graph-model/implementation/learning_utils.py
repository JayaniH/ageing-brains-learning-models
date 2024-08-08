import logging
import os
import numpy as np
from itertools import combinations
import matplotlib.pyplot as plt
from roulette_wheel_selection import roulette_wheel_selection
from constants import BASE_PATH, ACTIVE_NODE_COUNT

logging.basicConfig(filename=os.path.join(BASE_PATH, 'logs/learning.log'), filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def execute_young_learning(valid_input, valid_output, threshold, version = 2, output_node_selection = 'Roulette', save = False, verbose = False):
    """
    Execute the Young learning process.

    :param valid_input: The valid input signals
    :param valid_output: The valid output signals
    :param active_node_count: The number of active nodes
    :param threshold: The chosen threshold value
    :param version: The version of the learning process
        1: Update the weight of the edges one at a time based on legacy update rule
        2: Update the weight of the edges one at a time based on learning rate 
        3: Update the weight of all incoming edges for one output node at once
        4: Update the weight of all incoming edges for the 6 selected output nodes at once
    :param output_node_selection: The method used to select the output node
    :param save: If True, save the results to .csv files
    :param verbose: If True, log information and write to file

    :return: The number of iterations for each input pattern, the signals that exceeded the threshold, and the nodes that fired
    """

    # Initialize arrays to store results
    young_iteration_count = np.zeros(valid_output.shape[1])
    young_over_th_signals = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))
    young_nodes_fired = np.full((ACTIVE_NODE_COUNT, valid_output.shape[1]), -1)

    if verbose:
        logging.info("Starting Young learning process")
        print("Starting Young learning process...")

    # Loop over each input pattern
    for pattern_index in range(valid_output.shape[1]):
        loopcount = 0  # Initialize iteration count
        i = 0

        if verbose:
            logging.info(f"Pattern {pattern_index + 1}")
            print(f"Pattern {pattern_index + 1} of {valid_output.shape[1]}")
            
        while np.count_nonzero(young_over_th_signals[:, pattern_index]) < 6:  # While not all nodes have fired

            # Choose the output node with the highest signal that has not fired yet
            # Mask the fired nodes in the valid_output array
            masked_output = valid_output[:, pattern_index].copy()
            masked_output[young_nodes_fired[np.where(young_nodes_fired[:, pattern_index] != -1), pattern_index]] = -np.inf

            if output_node_selection == 'Roulette':
                if version == 4:
                    # choose 6 output nodes
                    chosen_output_node_indices = roulette_wheel_selection(masked_output, 6)
                else:
                    chosen_output_node_index = roulette_wheel_selection(valid_output[:, pattern_index])
            elif output_node_selection == 'Max':
                if version == 4:
                    chosen_output_node_indices = np.argsort(masked_output)[-6:]
                else:
                    # Choose the output node with the highest signal that has not fired yet
                    chosen_output_node_index = np.argmax(masked_output)
            else:
                raise ValueError("Invalid output node selection method")
            
            if verbose:
                if version == 4:
                    logging.info(f'Output nodes (sorted):\n{np.sort(valid_output[:, pattern_index])[::-1]},\nSelected output signals: {valid_output[chosen_output_node_indices, pattern_index]},\nSelected output indices (sorted): {np.where([np.sort(valid_output[:, pattern_index])[::-1] == valid_output[chosen_output_node_index, pattern_index] for chosen_output_node_index in chosen_output_node_indices])[0][0]},\nloopcount: {loopcount}\n')
                else:
                    logging.info(f'Output nodes (sorted):\n{np.sort(valid_output[:, pattern_index])[::-1]},\nSelected output signal: {valid_output[chosen_output_node_index, pattern_index]},\nSelected output index (sorted): {np.where(np.sort(valid_output[:, pattern_index])[::-1] == valid_output[chosen_output_node_index, pattern_index])[0][0]},\nloopcount: {loopcount}\n')

            # if chosen_output_node_index in young_nodes_fired[:, pattern_index]:
            #     continue

            if version in [1, 2, 3]:
                output_signal = valid_output[chosen_output_node_index, pattern_index]
                activated_input_node_indices = np.where(valid_input[:, chosen_output_node_index, pattern_index])[0] # The value is non-zero only in activated nodes as set previously
                
            # choose and update an edge based on the active edges
            if version in [1, 2]:
                candidate_edges = valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index]

                chosen_edge_index = roulette_wheel_selection(candidate_edges)
                chosen_edge = candidate_edges[chosen_edge_index]

                if version == 1:
                    new_weight = chosen_edge + (1 - chosen_edge) / 2

                elif version == 2:
                    learning_rate = 1.5
                    c = 0
                    new_weight = (learning_rate * chosen_edge) + c
                
                valid_input[activated_input_node_indices[chosen_edge_index], chosen_output_node_index, pattern_index] = new_weight  # update the weight in the input matrix    
                weight_diff = new_weight - chosen_edge
                output_signal += weight_diff
                valid_output[chosen_output_node_index, pattern_index] = output_signal  # update the output value in the output matrix

            # update the weight of all incoming edges
            elif version == 3:
                learning_rate = 1.5
                # update the weight of all edges#
                valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index] *= learning_rate
                output_signal = np.sum(valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index])
                valid_output[chosen_output_node_index, pattern_index] = output_signal

            # upadate the weight of all incoming edges for the 6 selected output nodes
            elif version == 4:
                learning_rate = 1.5
                output_signals = valid_output[chosen_output_node_indices, pattern_index]
                logging.info(f'Output signals: {output_signals}')
                for i, chosen_output_node_index in enumerate(chosen_output_node_indices):
                    activated_input_node_indices = np.where(valid_input[:, chosen_output_node_index, pattern_index])[0]
                    valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index] *= learning_rate
                    output_signals[i] = np.sum(valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index])
                    logging.info(f'Output signal {i}: {output_signals[i]}')
                    valid_output[chosen_output_node_index, pattern_index] = output_signals[i]
            
            else:
                raise ValueError("Invalid version number")

            loopcount += 1

            if version in [1,2,3] and output_signal >= threshold:
                young_over_th_signals[i, pattern_index] = output_signal
                young_nodes_fired[i, pattern_index] = chosen_output_node_index
                i += 1
                if verbose:
                    logging.info(f'Node fired with signal {output_signal}\n')
            
            elif version == 4:
                for i, chosen_output_node_index in enumerate(chosen_output_node_indices):
                    if output_signals[i] >= threshold:
                        young_over_th_signals[i, pattern_index] = output_signals[i]
                        young_nodes_fired[i, pattern_index] = chosen_output_node_index
                        if verbose:
                            logging.info(f'Node {i} fired with signal {output_signals[i]}')

        young_iteration_count[pattern_index] = loopcount
    
    if save:
        young_output_patterns = np.zeros((valid_output.shape[0], valid_output.shape[1]))  # output patterns
        for n in range(young_output_patterns.shape[1]):
            young_output_patterns[young_nodes_fired[:, n].astype(int), n] = 1  # transform them into 1

        np.savetxt(os.path.join(BASE_PATH, 'results/young_output_patterns.csv'), young_output_patterns.astype(int), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'results/young_output_nodes_fired.csv'), young_nodes_fired.astype(int), delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'results/young_over_th_signals.csv'), young_over_th_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, 'results/young_iteration_count.csv'), young_iteration_count.astype(int), delimiter=',', fmt='%d')

    return young_iteration_count, young_over_th_signals, young_nodes_fired


def execute_old_learning(valid_input, valid_output, threshold, version = 2, output_node_selection = 'Roulette', save = False, verbose = True):
    """
    Execute the Old learning process.

    :param valid_input: The valid input signals
    :param valid_output: The valid output signals
    :param active_node_count: The number of active nodes
    :param threshold: The chosen threshold value
    :param version: The version of the learning process
        1: Update the weight of the edges one at a time using another incoming edge of the same output node
        2: Update the weight of the edges one at a time using another outgoing edge of the same input node
        3: _
        4: Update the weight of all incoming edges for the 6 selected output nodes at once
    :param output_node_selection: The method used to select the output node
    :param save: If True, save the results to .csv files
    :param verbose: If True, log information and write to file

    :return: The number of iterations for each input pattern, the signals that exceeded the threshold, the nodes that fired, and the excluded nodes
    """

    old_iteration_count = np.zeros(valid_output.shape[1])  # to store how many iterations learning took
    excluded_nodes = np.full((ACTIVE_NODE_COUNT, valid_output.shape[1]), -1)  # to store which nodes were excluded
    old_over_th_signals = np.zeros((ACTIVE_NODE_COUNT, valid_output.shape[1]))  # to store the nodes that fired
    old_nodes_fired = np.full((ACTIVE_NODE_COUNT, valid_output.shape[1]), -1)  # matrix with the index of the nodes fired

    if verbose:
        logging.info("Starting Old learning process")
        print("Starting Old learning process...")
    
    for pattern_index in range(valid_output.shape[1]):  # for each input pattern
        loopcount = 0  # amount of iterations
        i = 0  # counter for the nodes that fired

        if verbose:
            logging.info(f"Pattern {pattern_index + 1}")
            print(f"Pattern {pattern_index + 1} of {valid_output.shape[1]}")

        while np.count_nonzero(old_over_th_signals[:, pattern_index]) < 6:  # while not all nodes have fired
            # Mask the fired nodes in the valid_output array
            masked_output = valid_output[:, pattern_index].copy()
            masked_output[old_nodes_fired[np.where(old_nodes_fired[:, pattern_index] != -1), pattern_index]] = -np.inf
            masked_output[excluded_nodes[np.where(excluded_nodes[:, pattern_index] != -1), pattern_index]] = -np.inf

            if output_node_selection == 'Roulette':
                if version == 4:
                    # choose 6 output nodes
                    chosen_output_node_indices = roulette_wheel_selection(masked_output, 6)
                else:
                    chosen_output_node_index = roulette_wheel_selection(valid_output[:, pattern_index])
            elif output_node_selection == 'Max':
                if version == 4:
                    chosen_output_node_indices = np.argsort(masked_output)[-6:]
                else:
                    # Choose the output node with the highest signal that has not fired yet
                    chosen_output_node_index = np.argmax(masked_output)
            else:
                raise ValueError("Invalid output node selection method")

            if version == 4:
                logging.info(f'Selected output nodes: {chosen_output_node_indices}')

                chosen_edge_indices = []
                next_edge_indices = []
                conflict = False
                edge_position_counter = 1

                for j, chosen_output_node_index in enumerate(chosen_output_node_indices):
                    activated_input_node_indices = np.where(valid_input[:, chosen_output_node_index, pattern_index])[0]  # get the active nodes for the IP
                    candidate_edges = valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index]

                    chosen_edge_index = roulette_wheel_selection(candidate_edges)
                    
                    logging.info(f'Chosen edge index: {chosen_edge_index}')
                   
                    chosen_input_node_index = activated_input_node_indices[chosen_edge_index]

                    if (chosen_input_node_index, chosen_output_node_index) in chosen_edge_indices or (chosen_input_node_index, chosen_output_node_index) in next_edge_indices:
                        logging.info(f'Conflict in chosen edge: {chosen_input_node_index}, {chosen_output_node_index}. Current chosen edge indices: {chosen_edge_indices}, Current next edge indices: {next_edge_indices}')
                        conflict = True
                        loopcount += 1
                        break
                    
                    chosen_edge_indices.append((chosen_input_node_index, chosen_output_node_index))
                    
                    logging.info(f'Chosen edge indices: {chosen_edge_indices}')

                    coin_toss = np.random.randint(0, 2)

                    candidate_edges = valid_input[chosen_input_node_index, :, pattern_index]
                    chosen_edge_index = chosen_output_node_index

                    if coin_toss == 0:
                        next_edge_position = (chosen_edge_index + edge_position_counter) % len(candidate_edges)
                    else:
                        next_edge_position = (chosen_edge_index - edge_position_counter) % len(candidate_edges)
                    
                    logging.info(f'Next edge position: {next_edge_position}')

                    if next_edge_position in old_nodes_fired[:, pattern_index] or next_edge_position in excluded_nodes[:, pattern_index]:
                        logging.info(f'Next edge belongs to a node that has been previously selected or excluded: {next_edge_position}')
                        conflict = True
                        break

                    if (chosen_input_node_index, next_edge_position) in chosen_edge_indices or (chosen_input_node_index, next_edge_position) in next_edge_indices:
                        logging.info(f'Conflict in next edge: {chosen_input_node_index}, {next_edge_position}. Current chosen edge indices: {chosen_edge_indices}, Current next edge indices: {next_edge_indices}')
                        conflict = True
                        loopcount += 1
                        break

                    next_edge_indices.append((chosen_input_node_index, next_edge_position))
                    logging.info(f'Next edge indices: {next_edge_indices}')
                
                if not conflict:
                    for j, chosen_output_node_index in enumerate(chosen_output_node_indices):
                        logging.info(f'Updating edge for output node {chosen_output_node_index} ({j + 1} of 6)')
                        output_signal = valid_output[chosen_output_node_index, pattern_index]
                        chosen_input_node_index, chosen_output_node_index = chosen_edge_indices[j]
                        _, next_output_node_index = next_edge_indices[j]

                        output_signal += valid_input[chosen_input_node_index, next_output_node_index, pattern_index]
                        valid_output[chosen_output_node_index, pattern_index] = output_signal
                        
                        valid_input[chosen_input_node_index, chosen_output_node_index, pattern_index] += valid_input[chosen_input_node_index, next_output_node_index, pattern_index]
                        valid_input[chosen_input_node_index, next_output_node_index, pattern_index] = 0

                        if output_signal >= threshold and np.count_nonzero(old_over_th_signals[:, pattern_index]) < 6:
                            logging.info(f'Node {j} fired with signal {output_signal}')
                            old_over_th_signals[i, pattern_index] = output_signal
                            old_nodes_fired[i, pattern_index] = chosen_output_node_index
                            i += 1

                    loopcount += 1
                    edge_position_counter += 1

            elif version in [1, 2]:                                                                                                           
                if chosen_output_node_index in old_nodes_fired[:, pattern_index] or chosen_output_node_index in excluded_nodes[:, pattern_index]:  # if node has been previously selected, choose another one
                    continue

                activated_input_node_indices = np.where(valid_input[:, chosen_output_node_index, pattern_index])[0]  # get the active nodes for the IP
                candidate_edges = valid_input[activated_input_node_indices, chosen_output_node_index, pattern_index]  # get the values for the active node

                # Choose an edge based on the active edges
                chosen_edge_index = roulette_wheel_selection(candidate_edges)

                output_signal = valid_output[chosen_output_node_index, pattern_index] 

                chosen_input_node_index = activated_input_node_indices[chosen_edge_index]

                if version == 2:
                    candidate_edges = valid_input[chosen_input_node_index, :, pattern_index]
                    chosen_edge_index = chosen_output_node_index # the index of the chosen edge relative to the input node is the same as the output node

                coin_toss = np.random.randint(0, 2)
                edge_position_counter = 0

                while True:
                    # Increment the counter
                    edge_position_counter += 1

                    # If the maximum number of iterations is reached, exclude the node and break the loop
                    if edge_position_counter == len(candidate_edges):
                        loopcount -= (len(candidate_edges) - 1) # TODO: Check if this is correct. Why do we need to decrement the loop count?
                        excluded_nodes[i, pattern_index] = chosen_output_node_index
                        break

                    # Choose the direction based on the random value x
                    if coin_toss == 0:
                        # search_array = candidate_edges
                        # candidate_edges_circular = np.hstack((search_array, search_array))
                        next_edge_position = (chosen_edge_index + edge_position_counter) % len(candidate_edges)
                    else:
                        # search_array = np.flipud(candidate_edges)
                        # candidate_edges_circular = np.hstack((search_array, search_array))
                        next_edge_position = (chosen_edge_index - edge_position_counter) % len(candidate_edges)

                    # chosen_edge_position = np.where(chosen_edge == search_array)[0][0]

                    # Select the second edge and update the sum of edges
                    # next_edge = candidate_edges_circular[chosen_edge_position + edge_position_counter]
                    next_edge = candidate_edges[next_edge_position]

                    valid_input[chosen_input_node_index, chosen_output_node_index, pattern_index] += next_edge  # update the weight in the input matrix

                    if version == 2:
                        valid_input[chosen_input_node_index, next_edge_position, pattern_index] = 0  # update the weight in the input matrix

                    output_signal += next_edge
                    valid_output[chosen_output_node_index, pattern_index] = output_signal    # update the output value in the output matrix

                    # Increment the loop count
                    loopcount += 1

                    # If the threshold is exceeded, record the node
                    if output_signal >= threshold:
                        old_over_th_signals[i, pattern_index] = output_signal
                        old_nodes_fired[i, pattern_index] = chosen_output_node_index
                        i+=1
                        break

            else:
                raise ValueError("Invalid version number")

        old_iteration_count[pattern_index] = loopcount

    if save:
        # To later measure old output pattern similarity analysis
        old_output_patterns = np.zeros((valid_output.shape[0], valid_output.shape[1])) # Creating a matrix to see the nodes that fire
        # over_th_old = np.zeros((active_node_count, len(old_iteration_count))) # Putting in the index of the nodes

        for n in range(old_output_patterns.shape[1]):
            old_output_patterns[old_nodes_fired[:, n].astype(int), n] = 1  # transform them into 1

        # Store values in .csv files
        np.savetxt(os.path.join(BASE_PATH, 'results/old_output_patterns.csv'), old_output_patterns, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'results/old_output_nodes_fired.csv'), old_nodes_fired, delimiter=',', fmt='%d')
        np.savetxt(os.path.join(BASE_PATH, 'results/old_over_th_signals.csv'), old_over_th_signals, delimiter=',', fmt='%.4f')
        np.savetxt(os.path.join(BASE_PATH, 'results/old_iteration_count.csv'), old_iteration_count, delimiter=',', fmt='%d')

    return old_iteration_count, old_over_th_signals, old_nodes_fired, excluded_nodes


def plot_learning_histograms(learning_data, labels, colors, grouped=True):
    """
    Plot histograms of the number of iterations for Young and Old learning.

    :param learning_data: List of arrays with the number of iterations for each input pattern
    :param labels: List of labels for the datasets
    :param colors: List of colors for the datasets
    :param grouped: If True, bars are grouped by dataset; otherwise, bars are overlapped
    """
    maximum_iterations = max([max(data) for data in learning_data])

    # Calculate histogram for each dataset
    histograms = [np.histogram(data, bins=np.arange(1, max(maximum_iterations, 10) + 2), range=(1, max(maximum_iterations, 10) + 1))[0] for data in learning_data]

    maximum_count = max([max(histogram) for histogram in histograms])
    
    # Define width of bars
    bar_width = 0.35 if grouped else 0.7
    
    # Define positions for the bars
    x_positions = np.arange(1, len(histograms[0]) + 1)
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

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
    plt.ylim(0, maximum_count + (maximum_count * 0.1))
    plt.title("Number of Iterations for Young and Old learning", fontsize=12)
    plt.xticks(x_positions + bar_width / 2 if grouped else x_positions, [str(i) for i in range(1, len(histograms[0]) + 1)], rotation=45)
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
                            xytext=(0, 5),  # Increased vertical offset
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=5, rotation='vertical')  # Smaller font size and vertical

    for histogram, bar_chart in zip(histograms, bar_charts):
        autolabel(bar_chart, sum(histogram))

    # Show plot
    plt.gca().set_facecolor('w')
    plt.tight_layout()
    filename = 'iterations_young_old_grouped.png' if grouped else 'iterations_young_old.png'
    plt.savefig(os.path.join(BASE_PATH, 'figures', filename))
    plt.show()

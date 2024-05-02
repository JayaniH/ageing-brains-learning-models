import os
import numpy as np
from learning_utils import execute_old_learning, execute_young_learning, plot_learning_histograms
from constants import BASE_PATH, ACTIVE_NODE_COUNT, TOTAL_NODE_COUNT


def main():

    threshold = 0.2405

    valid_input = np.loadtxt(os.path.join(BASE_PATH, 'data/valid_input_weights(thresholded).csv'), delimiter=',').reshape(TOTAL_NODE_COUNT, TOTAL_NODE_COUNT, -1)
    valid_output = np.loadtxt(os.path.join(BASE_PATH, 'data/valid_output_signals(thresholded).csv'), delimiter=',')

    # Young learning
    young_iteration_count, young_over_th_signals, young_nodes_fired = execute_young_learning(valid_input, valid_output, threshold)
    print("Young learning iterations:", young_iteration_count)

    # Old learning
    old_n_iterations, old_over_th_signals, old_nodes_fired, excluded_nodes = execute_old_learning(valid_input, valid_output, threshold)
    print("Old learning iterations:", old_n_iterations)

    learning_iterations = [young_iteration_count, old_n_iterations]
    labels = ["Young", "Old"]
    colors = ["#EDB120", "#4DBEEE"]
    plot_learning_histograms(learning_iterations, labels, colors, grouped=True)  # Grouped bars
    plot_learning_histograms(learning_iterations, labels, colors, grouped=False)  # Overlapping bars


if __name__ == "__main__":
    main()

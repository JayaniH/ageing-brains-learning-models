from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from constants import BASE_PATH


def compare_iteration_counts_per_pattern(young_iteration_count, old_iteration_count):
    """
    Compare the number of iterations for young and old learning for each pattern

    :param young_iteration_count: Array of the number of iterations for young learning for each pattern
    :param old_iteration_count: Array of the number of iterations for old learning for each pattern

    :return lower_iterations: List of strings indicating which learning has lower number of iterations for each pattern
    :return lower_counts: Dictionary containing the count of patterns where young/old learning has lower number of iterations or equal number of iterations
    """
    
    lower_iterations = []
    lower_counts = {'Young': 0, 'Old': 0, 'Equal': 0}

    for young, old in zip(young_iteration_count, old_iteration_count):
        if young < old:
            lower_iterations.append("Young")
        elif young > old:
            lower_iterations.append("Old")
        else:
            lower_iterations.append("Equal")

    for key in lower_counts.keys():
        lower_counts[key] = lower_iterations.count(key)
    
    return lower_iterations, lower_counts


def count_outputs_within_margin(threshold, outputs, margin, count_threshold=6, save=True):
    """
    Count the number of signals that are within a given margin from the threshold for each pattern

    :param threshold: Threshold value for the signals
    :param outputs: Matrix of output signals
    :param margin: Margin value for the signals
    :param count_threshold: Minimum number of signals that should be within the margin for a pattern to be considered
    :param save: Boolean to save the patterns that are within the margin

    :return counts: Array of the number of signals that are within the margin for each pattern
    :return patterns: List of patterns that have signals within the margin
    """

    # Array to store the number of signals that are within the margin
    counts = np.zeros(outputs.shape[1])
    patterns = []

    for pattern_index in range(outputs.shape[1]):
        output_signals = outputs[:, pattern_index]

        # print output signals in descending order
        print(np.sort(output_signals)[::-1])

        difference = np.abs(threshold - output_signals)
        # Count the number of signals that are within the margin
        counts[pattern_index] = np.sum(difference < margin)

        # Store the pattern index
        if counts[pattern_index] >= count_threshold:
            patterns.append(pattern_index)

        # print the count and the signals that are within the margin
        print(f"Count: {counts[pattern_index]}")
        print(f"Signals within the margin: {output_signals[difference < margin]}\n")
    
    # print min, max, and average number of signals within the margin
    print(f"Min: {np.min(counts)}, Max: {np.max(counts)}, Average: {np.mean(counts)}")

    if save:
        np.savetxt(os.path.join(BASE_PATH, 'results/patterns_within_margin.csv'), patterns, fmt='%s', delimiter=',')

    return counts, patterns


def plot_outputs_within_margin(counts, margin):
    """
    Plot the number of signals that are within the margin for each pattern

    :param counts: Array of the number of signals that are within the margin for each pattern
    :param margin: Margin value for the signals
    """

    # Plot the number of signals that are within the margin for each pattern
    histogram = np.histogram(counts, bins=range(0, max(counts.astype(int)) + 2))
    
    # Truncate the histogram to remove leading zeros
    i = 0
    while histogram[0][0] == 0:
        i += 1
        histogram = (histogram[0][1:], histogram[1][1:])

    bar_chart = plt.bar(histogram[1][:-1], histogram[0], width=0.8, align='center', color='skyblue')
    for bar in bar_chart:
        height = bar.get_height()
        percentage = height / counts.size * 100
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{percentage: .2f}%', fontsize=7, ha='center', va='bottom')
    
    plt.ylim(0, max(histogram[0]) + 10000)
    plt.xticks(range(i, max(counts.astype(int)) + 1))
    plt.xlabel("Number of signals within the margin")
    plt.ylabel("Frequency (number of patterns)")
    plt.title("Number of signals within the margin for each pattern")

    plt.savefig(os.path.join(BASE_PATH, f'figures/outputs_within_margin({margin}).png'))
    plt.show()


def iteration_count_analysis(experiment_no):
    """
    Analyze the number of iterations for young and old learning for each pattern

    :param experiment_no: Experiment number
    """

    # Read iteration counts from files
    young_iteration_count = np.loadtxt(os.path.join(BASE_PATH, f'results/{experiment_no}/young_iteration_count.csv'), delimiter=',')
    old_iteration_count = np.loadtxt(os.path.join(BASE_PATH, f'results/{experiment_no}/old_iteration_count.csv'), delimiter=',')

    lower_iterations, lower_counts = compare_iteration_counts_per_pattern(young_iteration_count, old_iteration_count)
    np.savetxt(os.path.join(BASE_PATH, f'results/{experiment_no}/lower_iterations_per_pattern.csv'), lower_iterations, fmt='%s', delimiter=',')
    print(lower_iterations, lower_counts)


def count_common_nodes(young_nodes_fired, old_nodes_fired):
    """
    Count the number of common nodes in the output patterns of young and old learning for the same input pattern

    :param young_nodes_fired: Array of nodes fired for young learning
    :param old_nodes_fired: Array of nodes fired for old learning

    :return common_nodes_count: List of the number of common nodes for each pattern
    """

    common_nodes_count = []

    for i in range(young_nodes_fired.shape[1]):
        common_nodes = np.intersect1d(young_nodes_fired[:, i], old_nodes_fired[:, i])
        common_nodes_count.append(common_nodes.size)

    return common_nodes_count


def plot_common_nodes(common_nodes_count):
    """
    Plot the frequency of patterns with n common nodes in the output patterns for young and old learning

    :param common_nodes_count: List of the number of common nodes for each pattern
    """

    histogram = np.histogram(common_nodes_count, bins=range(0, max(common_nodes_count) + 2))

    bar_chart = plt.bar(histogram[1][:-1], histogram[0], width=0.8, align='center', color='skyblue')
    for bar in bar_chart:
        height = bar.get_height()
        percentage = height / len(common_nodes_count) * 100
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{percentage: .2f}%', fontsize=7, ha='center', va='bottom')
    
    plt.ylim(0, max(histogram[0]) + 10000)
    plt.xticks(range(0, max(common_nodes_count) + 1))
    plt.xlabel("Number of common nodes")
    plt.ylabel("Frequency (number of patterns)")
    plt.title("Frequency of patterns with n common nodes in young and old \noutput patterns for the same input pattern")

    plt.savefig(os.path.join(BASE_PATH, f'figures/common_output_nodes_count.png'))
    plt.show()


def get_iteration_count_for_similar_output_patterns(common_nodes_count, young_iteration_count, old_iteration_count, threshold=6):
    """
    Get the number of iterations for similar output patterns where the number of common nodes is greater than or equal to a threshold

    :param common_nodes_count: List of the number of common nodes for each pattern
    :param young_iteration_count: Array of the number of iterations for young learning for each pattern
    :param old_iteration_count: Array of the number of iterations for old learning for each pattern
    :param threshold: Minimum number of common nodes for the output patterns to be considered similar
    """

    young = young_iteration_count[np.where(np.array(common_nodes_count) >= threshold)]
    old = old_iteration_count[np.where(np.array(common_nodes_count) >= threshold)]

    maximum_iterations = max(max(young.astype(int)), max(old.astype(int)))

    # plot histogram
    histograms = [np.histogram(data, bins=np.arange(6, max(maximum_iterations, 10) + 2), range=(6, max(maximum_iterations, 10) + 1))[0] for data in [young, old]]
    labels = ["Young", "Old"]
    colors = ["#EDB120", "#4DBEEE"]

    bar_width = 0.35
    x_positions = np.arange(1, len(histograms[0]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        x_pos = x_positions + i * bar_width
        bar_chart = ax.bar(x_pos, histogram, width=bar_width, color=color, label=label)

        for bar in bar_chart:
            height = bar.get_height()
            percentage = height / sum(histogram) * 100
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n({percentage:.2f}%)', fontsize=7, ha='center', va='bottom')

        
    plt.ylim(0, max(max(histograms[0]), max(histograms[1]))*1.1)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(6, len(histograms[0]) + 6)])
    plt.xlabel("Number of iterations")
    plt.ylabel("Frequency (number of patterns)")
    plt.legend(fontsize=13)
    plt.title(f"Learning iteration analysis for similar output patterns (common nodes >= {threshold})")
    plt.savefig(os.path.join(BASE_PATH, f'figures/iteration_count_for_similar_output_patterns(more_than_{threshold}_common_nodes).png'))
    plt.show()

    # plot young and old iteration count for each pattern
    fig, ax = plt.subplots(figsize=(10, 6))

    x_positions = np.arange(1, len(young) + 1)
    bar_width = 0.35

    young_bar = ax.bar(x_positions, young, width=bar_width, color=colors[0], label=labels[0])
    old_bar = ax.bar(x_positions + bar_width, old, width=bar_width, color=colors[1], label=labels[1])

    for bar in young_bar:
        height = int(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', fontsize=7, ha='center', va='bottom')

    for bar in old_bar:
        height = int(bar.get_height())
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', fontsize=7, ha='center', va='bottom')

    plt.ylim(0, 10)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(0, len(young))])
    plt.xlabel("Pattern number")
    plt.ylabel("Number of iterations")
    plt.legend(fontsize=13)
    plt.title(f'Learning iteration count for each similar output pattern (common nodes >= {threshold})')
    plt.savefig(os.path.join(BASE_PATH, f'figures/iteration_count_for_similar_output_patterns(more_than_{threshold}_common_nodes)_per_pattern.png'))
    plt.show()
        

def plot_similar_output_patterns(common_nodes_count, input_patterns, young_output_patterns, old_output_patterns, threshold=6):
    """
    Plot the input and output patterns where the output pattern is similar (common nodes >= threshold)

    :param common_nodes_count: List of the number of common nodes for each pattern
    :param input_patterns: Array of input patterns
    :param young_output_patterns: Array of output patterns for young learning
    :param old_output_patterns: Array of output patterns for old learning
    :param threshold: Minimum number of common nodes for the output patterns to be considered similar
    """

    input_patterns = input_patterns[:, np.where(np.array(common_nodes_count) >= threshold)[0]]
    young_output_patterns = young_output_patterns[:, np.where(np.array(common_nodes_count) >= threshold)[0]]
    old_output_patterns = old_output_patterns[:, np.where(np.array(common_nodes_count) >= threshold)[0]]

    # plot young and old output patterns as binary matrices
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    ax[0].imshow(input_patterns, cmap='binary', aspect='auto')
    ax[0].set_title("Input patterns")
    ax[0].set_xlabel("Pattern number")
    ax[0].set_ylabel("Input node index")
    
    ax[1].imshow(young_output_patterns, cmap='Blues', aspect='auto', alpha=0.5)
    ax[1].imshow(old_output_patterns, cmap='Oranges', aspect='auto', alpha=0.5)
    ax[1].set_title("Output patterns")
    ax[1].set_xlabel("Pattern number")
    ax[1].set_ylabel("Output node index")

    ax[1].legend(['Young Output Patterns', 'Old Output Patterns'])

    # title for the figure
    fig.suptitle(f"Input and output patterns where the output pattern is similar (common nodes >= {threshold})")

    plt.savefig(os.path.join(BASE_PATH, f'figures/similar_output_patterns(more_than_{threshold}_common_nodes).png'))
    plt.show()


def check_nodes_fired_corresponds_to_output_patterns(nodes_fired, output_patterns):
    """
    Check if the nodes fired corresponds to the output patterns (sanity check)

    :param nodes_fired: Array of nodes fired for each pattern
    :param output_patterns: Array of output patterns for each pattern
    """

    # check if the nodes fired corresponds to the output patterns
    for i in range(nodes_fired.shape[1]):
        active_nodes = np.where(output_patterns[:, i] == 1)[0]
        if not np.array_equal(np.sort(nodes_fired[:, i].astype(int)), active_nodes):
            print(f"Pattern {i + 1} does not match the output pattern")
            print(f"Output pattern: {active_nodes}")
            print(f"Nodes fired: {nodes_fired[:, i].astype(int)}\n")


def analyze_input_nodes(weights):
    """
    Get statistics for the initial input weights and plot box plots

    :param weights: Array of input weights
    """

    weights = weights.T

    # box plot of input weights
    plt.boxplot(weights, patch_artist=True, showmeans=True)
    plt.xlabel("Input node")
    plt.ylabel("Weight")
    plt.title("Box plot of input weights in the model")

    # plt.savefig(os.path.join(BASE_PATH, 'figures/input_nodes_boxplot.png'))
    plt.show()

    # create data frame for summary statistics
    summary_statistics = pd.DataFrame({
        'Node': np.arange(1, weights.shape[1] + 1),
        'Mean': np.mean(weights, axis=0),
        'Std': np.std(weights, axis=0),
        'Min': np.min(weights, axis=0),
        'Max': np.max(weights, axis=0),
        'Median': np.median(weights, axis=0),
        'Sum': np.sum(weights, axis=0)
    })
    print("Summary statistics for input nodes:")
    print(summary_statistics)
    summary_statistics.to_csv(os.path.join(BASE_PATH, 'data/input_nodes_summary_statistics.csv'), index=False)
    

    # box plot for all input weights
    plt.boxplot(weights.flatten())
    plt.ylabel("Weight")
    plt.title("Box plot of all input weights in the model")

    # plt.savefig(os.path.join(BASE_PATH, 'figures/input_weights_boxplot_all.png'))
    plt.show()

    # create data frame for overall summary statistics
    overall_summary = pd.DataFrame({
        'Mean': np.mean(weights.flatten()),
        'Std': np.std(weights.flatten()),
        'Min': np.min(weights.flatten()),
        'Max': np.max(weights.flatten()),
        'Median': np.median(weights.flatten()),
        'Sum': np.sum(weights.flatten())
    }, index=[0])
    print("Overall summary statistics for weights:")
    print(overall_summary)
    overall_summary.to_csv(os.path.join(BASE_PATH, 'data/weights_overall_summary.csv'), index=False)


def analyze_output_nodes(weights, mode="initial"):
    """
    Get statistics for the output nodes and plot box plots

    :param weights: Array of output weights
    :param mode: Mode of analysis (initial or patterns) - initial for initial output weights and patterns for output weights for each pattern
    """

    if mode == "patterns":
        # when analysing output nodes for each pattern 24/30 weights are zero which biases the box plot
        # remove zeores
        weights = weights[weights != 0].reshape(6, 30)
        print(weights.shape)

    # box plot of output nodes
    plt.boxplot(weights, patch_artist=True, showmeans=True)
    plt.xlabel("Output node")
    plt.ylabel("Weight")
    plt.title("Box plot of output nodes in the model")

    # plt.savefig(os.path.join(BASE_PATH, 'figures/output_nodes_boxplot.png'))
    plt.show()

    # summary statistics for each output node
    mean_per_node = np.mean(weights, axis=0)
    std_per_node = np.std(weights, axis=0)
    min_per_node = np.min(weights, axis=0)
    max_per_node = np.max(weights, axis=0)
    sum_per_node = np.sum(weights, axis=0)

    # create data frame for summary statistics
    summary_statistics = pd.DataFrame({
        'Node': np.arange(1, weights.shape[1] + 1),
        'Mean': mean_per_node,
        'Std': std_per_node,
        'Min': min_per_node,
        'Max': max_per_node,
        'Sum': sum_per_node
    })
    print("Summary statistics for output nodes")
    print(summary_statistics)
    summary_statistics.to_csv(os.path.join(BASE_PATH, 'data/output_nodes_summary_statistics.csv'), index=False)

    # box plot for all output nodes
    plt.boxplot(weights.flatten())
    plt.ylabel("Weight")
    plt.title("Box plot of all output nodes in the model")

    # plt.savefig(os.path.join(BASE_PATH, 'figures/output_nodes_boxplot_all.png'))
    plt.show()


def get_uniqiue_and_duplicate_output_patterns(output_nodes_fired):
    """
    Get the unique and duplicate output pattern counts

    :param output_nodes_fired: Array of output nodes fired for each pattern

    :return column_counts: Dictionary containing the count of each output pattern
    :return unique_columns: Dictionary containing the index of unique output patterns
    :return duplicate_columns: Dictionary containing the index of duplicate output patterns
    """

    columns_as_tuples = [tuple(np.sort(row)) for row in output_nodes_fired.T]
    column_counts = Counter(columns_as_tuples)
    unique_columns = {columns_as_tuples.index(key): value for key, value in column_counts.items() if value == 1}
    duplicate_columns = {columns_as_tuples.index(key): value for key, value in column_counts.items() if value > 1}
    print(f'{len(unique_columns)} out of {output_nodes_fired.shape[1]} patterns are unique. Percentage: {len(unique_columns) / output_nodes_fired.shape[1] * 100:.2f}%')

    return column_counts, unique_columns, duplicate_columns


def plot_output_pattern_commonality(duplicate_patterns_young, duplicate_patterns_old):
    """
    Plot the commonality of output patterns for young and old learning

    :param duplicate_patterns_young: Dictionary containing the count of each output pattern for young learning
    :param duplicate_patterns_old: Dictionary containing the count of each output pattern for old learning
    """

    # get the commonality for young and old learning as lists
    commonality_young = list(duplicate_patterns_young.values())
    commonality_old = list(duplicate_patterns_old.values())

    # plot histogram of commonality
    histograms = [np.histogram(data, bins=range(1, max(max(commonality_young), max(commonality_old)) + 2), range=(1, max(max(commonality_young), max(commonality_old)) + 2))[0] for data in [commonality_young, commonality_old]]
    labels = ["Young", "Old"]
    colors = ["#EDB120", "#4DBEEE"]

    bar_width = 0.35
    x_positions = np.arange(1, len(histograms[0]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (histogram, label, color) in enumerate(zip(histograms, labels, colors)):
        print(histogram)
        x_pos = x_positions + i * bar_width
        bar_chart = ax.bar(x_pos, histogram, width=bar_width, color=color, label=label)

        for bar in bar_chart:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}\n{height/sum(commonality_young) * 100:.2f}%', fontsize=7, ha='center', va='bottom')


    plt.ylim(0, max(max(histograms[0]), max(histograms[1])) * 1.1)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(1, len(histograms[0]) + 1)])
    plt.xlabel("Number of copies of the output pattern")
    plt.ylabel("Frequency (number of occurrences)")
    plt.legend(fontsize=13)
    plt.title("Commonality of output patterns for young and old learning")
    plt.savefig(os.path.join(BASE_PATH, 'figures/output_pattern_commonality.png'))
    plt.show()


def get_individual_node_activation_counts(output_patterns):
    """
    Get the number patterns where each output node is active

    :param output_patterns: Matrix of output patterns

    :return activation_counts: List of the number of patterns where each output node is active
    """

    activation_counts = np.sum(output_patterns, axis=1)
    return activation_counts


def plot_individual_node_activation_counts(node_activation_counts_young, node_activation_counts_old):
    """
    Plot the number of patterns for where each output node is active for young and old learning

    :param node_activation_counts_young: List of the number of patterns where each output node is active after young learning
    :param node_activation_counts_old: List of the number of patterns where each output node is active after old learning
    """
    
    labels = ["Young", "Old"]
    colors = ["#EDB120", "#4DBEEE"]

    bar_width = 0.35
    x_positions = np.arange(1, 31)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (histogram, label, color) in enumerate(zip([node_activation_counts_young, node_activation_counts_old], labels, colors)):
        x_pos = x_positions + i * bar_width
        bar_chart = ax.bar(x_pos, histogram, width=bar_width, color=color, label=label)

        for bar in bar_chart:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)} ({height/(sum(histogram)/6) * 100:.2f}%)', fontsize=5, ha='center', va='bottom', rotation=90)

    plt.ylim(0, max(max(node_activation_counts_young), max(node_activation_counts_old)) * 1.2)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(1, 31)])
    plt.xlabel("Node index")
    plt.ylabel("Number of active patterns")
    plt.legend(fontsize=13)
    plt.title("Output node activation counts for young and old learning")
    plt.savefig(os.path.join(BASE_PATH, 'figures/output_node_activation_counts.png'))
    plt.show()


def main():

    ## --- ITERATION COMPARISON --- ##

    iteration_count_analysis(3)


    ## --- OUTPUT SIGNAL ANALYSIS --- ##

    counts, patterns = count_outputs_within_margin(0.2405, np.loadtxt(os.path.join(BASE_PATH, 'data/valid_output_signals(thresholded).csv'), delimiter=','), margin=0.09, count_threshold=6, save=True)
    plot_outputs_within_margin(counts, margin=0.09)


    ## --- COMMON OUTPUT NODES IN YOUNG AND OLD LEARNING --- ##

    common_nodes_count = count_common_nodes(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_nodes_fired.csv'), delimiter=','))
    plot_common_nodes(common_nodes_count)
    get_iteration_count_for_similar_output_patterns(common_nodes_count, np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_iteration_count.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_iteration_count.csv'), delimiter=','), threshold=5)
    plot_similar_output_patterns(common_nodes_count, np.loadtxt(os.path.join(BASE_PATH, 'data/valid_input_patterns(thresholded).csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_patterns.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_patterns.csv'), delimiter=','), threshold=5)

    check_nodes_fired_corresponds_to_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_patterns.csv'), delimiter=','))


    ## --- INITIAL WEIGHT ANALYSIS (INIITAL MODEL STATISTICS) --- ##
    
    input_weights = np.loadtxt(os.path.join(BASE_PATH, 'data/inputs_normal_dis.csv'), delimiter=',')
    analyze_input_nodes(input_weights)
    analyze_output_nodes(input_weights)

    valid_input = np.loadtxt(os.path.join(BASE_PATH, 'data/valid_input_weights(thresholded).csv'), delimiter=',').reshape(30, 30, -1)
    # get analysis for each pattern
    for i in range(valid_input.shape[2]):
        print(f"Pattern {i + 1}")
        analyze_input_nodes(valid_input[:, :, i])
        analyze_output_nodes(valid_input[:, :, i], mode="patterns")


    ## --- OUTPUT PATTERN COMMONALITY --- ##

    pattern_counts_young, unique_patterns_young, duplicate_patterns_young = get_uniqiue_and_duplicate_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','))
    pattern_counts_old, unique_patterns_old, duplicate_patterns_old = get_uniqiue_and_duplicate_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_nodes_fired.csv'), delimiter=','))

    plot_output_pattern_commonality(pattern_counts_young, pattern_counts_old)


    ## --- NODE ACTIVATION COUNTS --- ##

    node_activation_counts_young = get_individual_node_activation_counts(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_patterns.csv'), delimiter=','))
    node_activation_counts_old = get_individual_node_activation_counts(np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_patterns.csv'), delimiter=','))

    plot_individual_node_activation_counts(node_activation_counts_young, node_activation_counts_old)

    
if __name__ == "__main__":
    main()
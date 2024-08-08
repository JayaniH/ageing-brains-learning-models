from collections import Counter
import os
import numpy as np
import matplotlib.pyplot as plt

from constants import BASE_PATH

def compare_iteration_counts_per_pattern(young_iteration_count, old_iteration_count):
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
    # Read iteration counts from files
    young_iteration_count = np.loadtxt(os.path.join(BASE_PATH, f'results/{experiment_no}/young_iteration_count.csv'), delimiter=',')
    old_iteration_count = np.loadtxt(os.path.join(BASE_PATH, f'results/{experiment_no}/old_iteration_count.csv'), delimiter=',')

    lower_iterations, lower_counts = compare_iteration_counts_per_pattern(young_iteration_count, old_iteration_count)
    np.savetxt(os.path.join(BASE_PATH, f'results/{experiment_no}/lower_iterations_per_pattern.csv'), lower_iterations, fmt='%s', delimiter=',')
    print(lower_iterations, lower_counts)


def count_common_nodes(young_nodes_fired, old_nodes_fired):
    common_nodes_count = []

    for i in range(young_nodes_fired.shape[1]):
        common_nodes = np.intersect1d(young_nodes_fired[:, i], old_nodes_fired[:, i])
        common_nodes_count.append(common_nodes.size)

    return common_nodes_count

def plot_common_nodes(common_nodes_count):
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


def count_identical_output_patterns(output_nodes_fired, tag):
    columns_as_tuples = [tuple(np.sort(row)) for row in output_nodes_fired.T]

    column_counts = Counter(columns_as_tuples)
    duplicate_columns = {columns_as_tuples.index(key): value for key, value in column_counts.items() if value > 1}
    
    # # plot the count of identical patterns against the pattern index
    # fig, ax = plt.subplots(figsize=(10, 6))

    # x_positions = np.arange(1, len(duplicate_columns.values()) + 1)
    # bar_width = 0.35

    # bar_chart = ax.bar(x_positions, duplicate_columns.values(), width=bar_width, color='skyblue')

    # for bar in bar_chart:
    #     height = int(bar.get_height())
    #     plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', fontsize=7, ha='center', va='bottom')

    # plt.ylim(0, max(duplicate_columns.values()) + 1)
    # plt.xticks(x_positions)
    # plt.xlabel("Pattern index")
    # plt.ylabel("Number of identical patterns")
    # plt.title(f"Number of identical output patterns for {tag} learning")
    # plt.savefig(os.path.join(BASE_PATH, f'figures/identical_output_patterns({tag}).png'))
    # plt.show()

    return duplicate_columns

def plot_output_pattern_commonality(identical_patterns_young, identical_patterns_old):
    # get the commonality for young and old learning as lists
    commonality_young = list(identical_patterns_young.values())
    commonality_old = list(identical_patterns_old.values())

    # plot histogram of commonality
    histograms = [np.histogram(data, bins=range(2, max(max(commonality_young), max(commonality_old)) + 2), range=(2, max(max(commonality_young), max(commonality_old)) + 1))[0] for data in [commonality_young, commonality_old]]
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
            plt.text(bar.get_x() + bar.get_width() / 2, height, f'{height}', fontsize=7, ha='center', va='bottom')


    plt.ylim(0, max(max(histograms[0]), max(histograms[1])) * 1.1)
    plt.xticks(x_positions + bar_width / 2, [str(i) for i in range(2, len(histograms[0]) + 2)])
    plt.xlabel("Number of duplicate patterns")
    plt.ylabel("Frequency (number of occurrences)")
    plt.legend(fontsize=13)
    plt.title("Commonality of output patterns for young and old learning")
    plt.savefig(os.path.join(BASE_PATH, 'figures/output_pattern_commonality.png'))
    plt.show()

def check_nodes_fired_corresponds_to_output_patterns(nodes_fired, output_patterns):

    # check if the nodes fired corresponds to the output patterns
    for i in range(nodes_fired.shape[1]):
        active_nodes = np.where(output_patterns[:, i] == 1)[0]
        if not np.array_equal(np.sort(nodes_fired[:, i].astype(int)), active_nodes):
            print(f"Pattern {i + 1} does not match the output pattern")
            print(f"Output pattern: {active_nodes}")
            print(f"Nodes fired: {nodes_fired[:, i].astype(int)}\n")


def main():
    iteration_count_analysis(3)
    counts, patterns = count_outputs_within_margin(0.2405, np.loadtxt(os.path.join(BASE_PATH, 'data/valid_output_signals(thresholded).csv'), delimiter=','), margin=0.09, count_threshold=6, save=True)
    plot_outputs_within_margin(counts, margin=0.09)

    common_nodes_count = count_common_nodes(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_nodes_fired.csv'), delimiter=','))
    plot_common_nodes(common_nodes_count)
    get_iteration_count_for_similar_output_patterns(common_nodes_count, np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_iteration_count.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_iteration_count.csv'), delimiter=','), threshold=5)
    plot_similar_output_patterns(common_nodes_count, np.loadtxt(os.path.join(BASE_PATH, 'data/valid_input_patterns(thresholded).csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_patterns.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_patterns.csv'), delimiter=','), threshold=5)

    identical_patterns_young = count_identical_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','), tag="young")
    identical_patterns_old = count_identical_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/old_output_nodes_fired.csv'), delimiter=','), tag="old")
    plot_output_pattern_commonality(identical_patterns_young, identical_patterns_old)

    check_nodes_fired_corresponds_to_output_patterns(np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_nodes_fired.csv'), delimiter=','), np.loadtxt(os.path.join(BASE_PATH, 'results/8/young_output_patterns.csv'), delimiter=','))
    

if __name__ == "__main__":
    main()
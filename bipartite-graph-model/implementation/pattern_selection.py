import os
import numpy as np
from pattern_selection_utils import get_input_and_output_patterns, get_threshold, get_valid_inputs_and_outputs, get_output_statistics, plot_signal_statistics, get_threshold_open
from constants import BASE_PATH

def main():

    data_file_path = 'original_data\Inputs_normal_dis'

    # Read the input matrix from CSV file
    initial_weights = np.loadtxt(os.path.join(BASE_PATH, f'data\{data_file_path}.csv'), delimiter=',')

    input_patterns, input, output = get_input_and_output_patterns(initial_weights)

    # Display the dimensions of the output matrix
    print("Shape of output matrix:", output.shape)

    # Define threshold values to attempt
    th_values = [0.2, 0.2405, 0.28, 0.275, 0.278]
    threshold = get_threshold(th_values, output)
    # threshold = get_threshold_open(output, percentage=0.25)

    valid_input_patterns, valid_input, valid_output = get_valid_inputs_and_outputs(threshold, input, input_patterns, output)

    # Display the chosen threshold and valid input pattern indices
    print("Chosen threshold:", threshold)
    print("Number of valid input patterns:", valid_input_patterns.shape[1])

    variance, avg_variance, sd_variance = get_output_statistics(valid_output)

    plot_signal_statistics(output, threshold, variance, avg_variance, sd_variance)


if __name__ == '__main__':
    main()

import numpy as np

def roulette_wheel_selection(probabilities, n=1, inverse=False):
    """
    Selects an index based on the given probabilities using the roulette wheel selection method.

    :param probabilities: The probabilities of each index.
    :param n: The number of indices to select.
    :param inverse: Whether to use inverse probabilities for selection.
    :return: The selected index.
    """
    # Replace negative probabilities with zeros
    # probabilities = np.maximum(probabilities, 0)

    # Scale negative probabilities to positive
    if np.min(probabilities) < 0:
        probabilities = probabilities - np.min(probabilities) + 1e-6 # Add a small value to prevent zero probabilities

    # Invert probabilities if needed
    if inverse:
        probabilities = 1 / probabilities

    # Normalize probabilities (if not already normalized)
    total_sum = np.sum(probabilities)

    if total_sum == 0:
        raise ValueError("Sum of probabilities is zero. Cannot perform selection.")
    
    probabilities = probabilities / total_sum
    
    selected_indices = []
    
    for _ in range(n):
        # Compute cumulative probabilities
        cumulative_probabilities = np.cumsum(probabilities)
        
        # Generate a random number
        random_number = np.random.rand()
        
        # Select based on cumulative probabilities
        for i, cumulative_probability in enumerate(cumulative_probabilities):
            if random_number < cumulative_probability:
                selected_indices.append(i)

                # Set the probability of the selected index to zero to prevent it from being selected again
                probabilities[i] = 0
                # Normalize probabilities again
                total_sum = np.sum(probabilities)
                if total_sum > 0:
                    probabilities = probabilities / total_sum
                break

    return selected_indices if n > 1 else selected_indices[0]
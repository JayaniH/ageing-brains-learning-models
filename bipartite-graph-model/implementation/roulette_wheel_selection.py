import numpy as np

def roulette_wheel_selection(probabilities):
    # Normalize probabilities (if not already normalized)
    total_sum = np.sum(probabilities)
    probabilities = probabilities / total_sum
    
    # Compute cumulative probabilities
    cumulative_probabilities = np.cumsum(probabilities)
    
    # Generate a random number
    random_number = np.random.rand()
    
    # Select based on cumulative probabilities
    for i, cumulative_probability in enumerate(cumulative_probabilities):
        if random_number < cumulative_probability:
            return i

import numpy as np

def roulette_wheel_selection(arrayInput):
    len_array = len(arrayInput)

    # If input is one element, return it right away
    if len_array == 1:
        return 0

    temp = 0
    cumProb = np.zeros(len_array)

    # Normalize inputs to be a well-defined distribution
    arrayInput = arrayInput / np.sum(arrayInput)

    for i in range(len_array):
        cumProb[i] = temp + arrayInput[i]
        temp = cumProb[i]

    i_rand = np.random.rand()

    for i in range(len_array):
        if i_rand < cumProb[i]:
            return i

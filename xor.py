import random
import numpy as np


random.seed(1701)


def get_random_boolean():
    # The fastest way according to:
    # http://stackoverflow.com/questions/6824681/get-a-random-boolean-in-python
    return bool(random.getrandbits(1))


def generate_xor_row(width):
    array = np.empty(width, dtype=bool)
    for n in xrange(width-1, 3):
        array[n] = get_random_boolean()
        array[n+1] = get_random_boolean()
        array[n+2] = array[n] ^ array[n+1]
    return array.astype(float)

def generate_xor_array(length, width):
    """Generate an array of booleans of length that satisfies XOR function
    """
    return np.stack([generate_xor_row(width)]*length)

# print generate_xor_array(10, 10)
import numpy as np

def salt_pepper_noise(x, p=0.1, extreme_values=(-1, 1)):
    shape = x.shape
    x = np.copy(x)

    flipping_cond = np.random.random_sample(x.shape) <= p
    x[flipping_cond] = np.random.choice(extreme_values, np.sum(flipping_cond))
    
    return x.reshape(shape)
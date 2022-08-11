import numpy as np

def adam_gradient(parameters, count, m, v, g):
    '''
    This function implements the Adam optimizer.
    :param parameters: numpy array of real numbers, list of parameters
    :param count: integer, keeps track of the optimization step
    :param m: numpy array of real numbers, needed by Adam
    :param v: numpy array of real numbers, needed by Adam
    :param g: numpy array of real numbers, gradient of the loss function
    :return: numpy array of real numbers, optimal shift of parameters
    '''
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    lr = 1e-3
    alpha = [lr for i in range(len(parameters))]
    if count == 0:
        count = 1
    new_shift = [0 for i in range(len(parameters))]
    for i in range(len(parameters)):
        m[i] = beta1 * m[i] + (1 - beta1) * g[i]
        v[i] = beta2 * v[i] + (1 - beta2) * np.power(g[i], 2)
        alpha[i] = alpha[i] * np.sqrt(1 - np.power(beta2, count)) / (1 - np.power(beta1, count))
        new_shift[i] = alpha[i] * (m[i] / (np.sqrt(v[i]) + eps))
    return new_shift
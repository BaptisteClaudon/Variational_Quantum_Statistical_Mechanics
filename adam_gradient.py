import numpy as np

def adam_gradient(parameters, count, m, v, g, lr=0.001):
    ## This function implements adam optimizer
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
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
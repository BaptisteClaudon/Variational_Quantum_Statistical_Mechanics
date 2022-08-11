import numpy as np

def ei(i,n):
    vi = np.zeros(n)
    vi[i] = 1.0
    return vi[:]

def extend_parameters(vector, nspins):
    extended = np.zeros(len(vector)*nspins)
    for k in range(int(len(vector)/2)):
        for n in range(nspins):
            extended[k*2*nspins+n] = vector[2*k]
            extended[k*2*nspins+nspins+n] = vector[2*k+1]
    return extended


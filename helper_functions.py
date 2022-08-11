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

def are_nearest_neighbours_rectangle(X, Y, Lx, Ly):
    a, b = X
    c, d = Y
    neighbours = ((((a-c)%Lx == 1) or ((a-c)%Lx == Lx-1)) and ((b-d)%Ly==0)) or ((((b-d)%Ly == 1) or ((b-d)%Ly == Ly-1)) and ((a-c)%Lx==0))
    return neighbours

def nearest_neighbours_rectangle(Lx, Ly):
    neighbours = []
    for a in range(Lx):
        for b in range(Ly):
            if a*Ly + b < a*Ly + ((b+1)%Ly):
                neighbours.append((a*Ly + b, a*Ly + ((b+1)%Ly)))
            if a*Ly + b < a*Ly + ((b-1)%Ly):
                neighbours.append((a*Ly + b, a*Ly + ((b-1)%Ly)))
            if a*Ly + b < ((a+1)%Lx)*Ly + b:
                neighbours.append((a*Ly + b, ((a+1)%Lx)*Ly + b))
            if a*Ly + b < ((a-1)%Lx)*Ly + b:
                neighbours.append((a*Ly + b, ((a-1)%Lx)*Ly + b))
    neighbours = list(dict.fromkeys(neighbours))
    return neighbours

def neirest_neighbours_line(Lx):
    neighbours = [(i, (i+1)%Lx) for i in range(Lx)]
    return neighbours

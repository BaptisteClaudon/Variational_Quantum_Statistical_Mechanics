import numpy as np
import scipy
from scipy.linalg import expm
from hamiltonian import generate_XYZ, ising_2D_hamiltonian, heisenberg_2D_hamiltonian
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.sparse.linalg import norm
from operator import itemgetter

def partition_function(J,h,N,T):
    '''
    Computes the partition function of a classical 1D pbc Ising model.
    :param J: real number, coupling constant
    :param h: real number, magnetic field
    :param N: positive integer, number of spins
    :param T: real number, temperature
    :return: real number, partition function
    '''
    J = J/T
    h = h/T
    epsi = np.exp(-J)*np.cosh(h)
    lon = np.sqrt(np.exp(-2.*J)*np.sinh(h)**2+np.exp(2.*J))
    epsi1 = epsi + lon
    epsi2 = epsi - lon
    return epsi1**N + epsi2**N

def free_energy(J,h,N,T):
    '''
    Computes the free energy of a classical 1D pbc Ising model.
    :param J: real number, coupling constant
    :param h: real number, magnetic field
    :param N: positive integer, number of spins
    :param T: real number, temperature
    :return: real number, free energy
    '''
    return -np.log(partition_function(J,h,N,T))*T

def ana_mean_energy(J,h,N,T):
    '''
    Computes the mean energy of a classical 1D pbc Ising model.
    :param J: real number, coupling constant
    :param h: real number, magnetic field
    :param N: positive integer, number of spins
    :param T: real number, temperature
    :return: real number, mean energy
    '''
    delta = 1e-5
    return (T**2)*(np.log(partition_function(J,h,N,T+.5*delta))-np.log(partition_function(J,h,N,T-.5*delta)))/delta

def ana_entropy(J,h,N,T):
    '''
    Computes the entropy of a classical 1D pbc Ising model.
    :param J: real number, coupling constant
    :param h: real number, magnetic field
    :param N: positive integer, number of spins
    :param T: real number, temperature
    :return: real number, entropy
    '''
    return (ana_mean_energy(J,h,N,T)-free_energy(J,h,N,T))/T

def ana_c(J,h,N,T):
    '''
    Computes the heat capacity of a classical 1D pbc Ising model.
    :param J: real number, coupling constant
    :param h: real number, magnetic field
    :param N: positive integer, number of spins
    :param T: real number, temperature
    :return: real number, heat capacity
    '''
    delta = 1e-5
    return -(ana_mean_energy(J,h,N,T+.5*delta)-ana_mean_energy(J,h,N,T-.5*delta))/delta - 1/T/T*2*ana_mean_energy(J,h,N,T)**2

def best_approx(beta, H0, H=None):
    '''
    Concerns the 1D quantum Ising model with periodic boundary conditions. Computes the energy of the quasi-Gibbs state.
    :param beta: real number, initial inverse temperature
    :param H: PauliOp, final hamiltonian
    :param H0: PauliOp, initial hamiltonian
    :return: real number, quasi-Gibbs energy
    '''
    #rho0 = thermal_state(H0, 1./beta)
    ps = eigsh(H0, return_eigenvectors=False)
    ps.sort()
    if H != None:
        energies = eigsh(H, return_eigenvectors=False)
        energies.sort()
    else:
        energies = ps
    ps = np.exp(-beta*ps)
    ps *= 1./sum(ps)
    return np.dot(ps, energies)

def thermal_state(hamiltonian, temperature):
    '''
    Return a thermal state of the hamiltonian at a given temperature.
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :param temperature: positive real number, temperatur
    :return: numpy (density) matrix, thermal state
    '''
    beta = 1./temperature
    e = scipy.linalg.expm(-beta*hamiltonian)
    t = np.matrix.trace(e)
    return e/t

def find_beta(hamiltonian, entropy, beta0, hamiltonian0):
    '''
    Finds inverse temperature at which the hamiltonian thermal state has a given entropy.
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :param entropy: real number, entropy
    :param beta0: real number, initial inverse temperature
    :return: real number, final inverse temperature
    '''
    try:
        #b = np.real(scipy.optimize.newton(func=beta_times_fe, x0=beta0, args=(hamiltonian, entropy)))
        left = 1e-5
        right = 10
        b = scipy.optimize.toms748(f=beta_times_fe, a=left, b=right, args=(hamiltonian, entropy, hamiltonian0, beta0))
    except RuntimeError:
        print("scipy.optimize failed to find the final beta.")
        b = beta0
    except ValueError:
        print("scipy.optimize failed to find the final beta.")
        betarange = np.linspace(left, right, 500)
        vals = [np.abs(beta_times_fe(beta, hamiltonian, entropy, hamiltonian0, beta0)) for beta in betarange]
        from matplotlib import pyplot as plt
        plt.plot(betarange, vals)
        b = betarange[min(enumerate(vals), key=itemgetter(1))[0]]
    '''    
    except ValueError:
        print("Value error occured.")
        L = np.abs(beta_times_fe(left, hamiltonian, entropy))
        print("Left free energy error: ", L)
        R = np.abs(beta_times_fe(right, hamiltonian, entropy))
        print("Left free energy error: ", R)
        if L < R:
            b = left
        else:
            b = right
    '''
    print("Final beta: ", b)
    return b

def exact_energy(beta, hamiltonian):
    '''
    Coomputes exact energy from hermitian matrix.
    :param beta: real positive number, inverse temperature
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :return: real number, energy
    '''
    exp = scipy.linalg.expm(-beta * hamiltonian)
    Z = np.trace(exp)
    rho = exp / Z
    return np.trace(rho@hamiltonian)

def beta_times_fe(beta, hamiltonian, entropy, hamiltonian0, beta0):
    '''
    Product between beta and free energy
    :param beta: real positive number, inverse temperature
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :param entropy: real number, entropy
    :return: real number, product
    '''
    try:
        #exp = scipy.linalg.expm(-beta * hamiltonian)
        ps = eigsh(hamiltonian, return_eigenvectors=False)
    except ValueError:
        ps = [1.]
        print("Value error.")
        print("beta: ",beta)
        print("hamiltonian: ",hamiltonian)
    try:
        ps = np.exp(-beta * ps)
        Z = sum(ps)
        E = best_approx(beta0, H0=hamiltonian0, H=hamiltonian)
        f = - np.log(Z) - beta * E + entropy
    except RuntimeWarning:
        print("Z exploded")
        f = 0
    return f

def get_exact_and_best_for_ising(beta, nspins, initial_parameters=(0., -1.), final_parameters=(-1., -1.), connectivity=None):
    '''
    Gets quasi-Gibbs and Gibbs energy for the J=h=1 quantum 1D pbc Ising model.
    :param beta: real positive number, inverse temperature
    :param nspins: positive integer, number of spins
    :param initial_parameters: couple of real numbers (J0,h0), H0 = J0\sum ZZ+h0\sumX
    :param final_parameters: couple of real numbers (J,h), H = J\sum ZZ+h\sumX
    :return: (real number, real number), (Gibbs energy, quasi-Gibbs energy)
    '''
    J_0, h_0 = initial_parameters
    J, h = final_parameters
    anaent = ana_entropy(J=J_0,h=h_0,N=nspins,T=1./beta)
    if connectivity != None:
        H0 = ising_sparse_hamiltonian(J_0, h_0, connectivity, nspins)
        H = ising_sparse_hamiltonian(J, h, connectivity, nspins)
    else:
        H0 = generate_XYZ(0.,0.,J_0,h_0,n_spins=nspins,pbc=True)
        H0 = H0.to_matrix()
        H = generate_XYZ(0.,0.,J,h,n_spins=nspins,pbc=True)
        H = H.to_matrix()
    betaf = find_beta(H, anaent, beta, H0)
    exact = best_approx(betaf, H)
    best = best_approx(beta, H0, H)
    return np.real(exact), np.real(best)

def heisenberg_initial_entropy(nspins, beta):
    '''
    Get entropy of staggered magnetic field model.
    :param nspins: positive integer, number of spins
    :param beta: positive real number, inverse temperature
    :return: entropy for which to find the corresponding beta, in the Heisenberg case
    '''
    Z = 2**nspins*np.cosh(beta)**nspins
    delta = 1e-5
    Zplus = 2**nspins*np.cosh(beta+.5*delta)**nspins
    Zminus = 2**nspins*np.cosh(beta-.5*delta)**nspins
    E = (1./beta ** 2) * (np.log(Zplus) - np.log(Zminus)) / delta
    f = - np.log(Z)/beta
    s = (E-f)*beta
    return s

def get_exact_and_best_for_heisenberg(beta, nspins, final_parameters=(1., 1.), connectivity=None):
    '''
    Computes the Gibbbs and quasi-Gibbs energies in the Heisenberg case.
    :param beta: positive real number, inverse temperature
    :param nspins: positive integer, number of spins
    :param final_parameters: R^2, J1 and J2 parameters
    :param connectivity: two list of couple of positive integers, representing the first and second nearest neighbours
    :return: two real numbers, Gibbs and qGibbs energies
    '''
    J1, J2 = final_parameters
    anaent = heisenberg_initial_entropy(nspins, beta)
    hfields = np.ones(nspins)
    for k in range(int(nspins/2)):
        hfields[2*k+1] = -1
    H0 = heisenberg_2D_hamiltonian(0., 0., hfields, connectivity, nspins)
    H = heisenberg_2D_hamiltonian(J1, J2, hfields, connectivity, nspins)
    if beta < 3:
        betaf = find_beta(H.to_matrix(), anaent, beta)
    else:
        betaf = beta
    exact = exact_energy(betaf, H.to_matrix())
    best = best_approx(beta, H, H0)
    return np.real(exact), np.real(best)

def list_paulis(N, X=csr_matrix([[0,1],[1,0]])):
    l = []
    I = csr_matrix(scipy.sparse.identity(2))
    for p in range(N):
        op = X
        for q in range(p):
            op = scipy.sparse.kron(I, op)
        for q in range(p+1,N):
            op = scipy.sparse.kron(op, I)
        l.append(op)
    return l

def list_xis(N):
    return list_paulis(N)

def list_yis(N):
    return list_paulis(N, csr_matrix([[0,-1.j],[1.j,0]]))

def list_zis(N):
    return list_paulis(N, csr_matrix([[1,0],[0,-1]]))

def ising_sparse_hamiltonian(J=1., Gamma=1., neighbours=[], N=3):
    lx = list_xis(N)
    lz = list_zis(N)
    H = Gamma*sum(lx)
    for nei in neighbours:
        i, j = nei
        H += J*lz[i]@lz[j]
    return H


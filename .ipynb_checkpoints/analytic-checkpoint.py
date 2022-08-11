import numpy as np
import scipy
from scipy.linalg import expm
from hamiltonian import generate_XYZ

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

def best_approx(beta, H, H0):
    '''
    Concerns the 1D quantum Ising model with periodic boundary conditions. Computes the energy of the quasi-Gibbs state.
    :param beta: real number, initial inverse temperature
    :param H: PauliOp, final hamiltonian
    :param H0: PauliOp, initial hamiltonian
    :return: real number, quasi-Gibbs energy
    '''
    rho0 = thermal_state(H0.to_matrix(), 1./beta)
    ps = scipy.linalg.eigh(rho0, eigvals_only=True)
    ps.sort()
    energies = scipy.linalg.eigh(H.to_matrix(), eigvals_only=True)
    energies.sort()
    ps = np.flip(ps)
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

def find_beta(hamiltonian, entropy, beta0):
    '''
    Finds inverse temperature at which the hamiltonian thermal state has a given entropy.
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :param entropy: real number, entropy
    :param beta0: real number, initial inverse temperature
    :return: real number, final inverse temperature
    '''
    try:
        b = np.real(scipy.optimize.newton(func=beta_times_fe, x0=beta0, args=(hamiltonian, entropy)))
    except RuntimeError:
        print("scipy.optimize failed to find the final beta.")
        b = beta0
    except ValueError:
        print("Value error occured.")
        b = beta0
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

def beta_times_fe(beta, hamiltonian, entropy):
    '''
    Product between beta and free energy
    :param beta: real positive number, inverse temperature
    :param hamiltonian: numpy hermitian matrix, hamiltonian
    :param entropy: real number, entropy
    :return: real number, product
    '''
    try:
        exp = scipy.linalg.expm(-beta * hamiltonian)
    except ValueError:
        exp = 1.
        print("Value error.")
        print("beta: ",beta)
        print("hamiltonian: ",hamiltonian)
    Z = np.trace(exp)
    rho = exp / Z
    f = - np.log(Z) - beta * np.trace(rho @ hamiltonian) + entropy
    return f

def get_exact_and_best_for_ising(beta, nspins, initial_parameters=(0., -1.), final_parameters=(-1., -1.)):
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
    H0 = generate_XYZ(0.,0.,J_0,h_0,n_spins=nspins,pbc=True)
    H = generate_XYZ(0.,0.,J,h,n_spins=nspins,pbc=True)
    betaf = find_beta(H.to_matrix(), anaent, beta)
    exact = exact_energy(betaf, H.to_matrix())
    best = best_approx(beta, H, H0)
    return np.real(exact), np.real(best)
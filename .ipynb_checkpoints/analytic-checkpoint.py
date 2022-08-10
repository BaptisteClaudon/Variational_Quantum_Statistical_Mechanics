import numpy as np
import scipy
from scipy.linalg import expm
from hamiltonian import generate_XYZ

def partition_function(J,h,N,T):
    J = J/T
    h = h/T
    epsi = np.exp(-J)*np.cosh(h)
    lon = np.sqrt(np.exp(-2.*J)*np.sinh(h)**2+np.exp(2.*J))
    epsi1 = epsi + lon
    epsi2 = epsi - lon
    return epsi1**N + epsi2**N

def free_energy(J,h,N,T):
    return -np.log(partition_function(J,h,N,T))*T

def ana_mean_energy(J,h,N,T):
    delta = 1e-5
    return (T**2)*(np.log(partition_function(J,h,N,T+.5*delta))-np.log(partition_function(J,h,N,T-.5*delta)))/delta

def ana_entropy(J,h,N,T):
    return (ana_mean_energy(J,h,N,T)-free_energy(J,h,N,T))/T

def ana_c(J,h,N,T):
    delta = 1e-5
    return -(ana_mean_energy(J,h,N,T+.5*delta)-ana_mean_energy(J,h,N,T-.5*delta))/delta - 1/T/T*2*ana_mean_energy(J,h,N,T)**2

def best_approx(beta, H, H0):
    rho0 = thermal_state(H0.to_matrix(), 1./beta)
    ps = scipy.linalg.eigh(rho0, eigvals_only=True)
    ps.sort()
    energies = scipy.linalg.eigh(H.to_matrix(), eigvals_only=True)
    energies.sort()
    ps = np.flip(ps)
    return np.dot(ps, energies)

def thermal_state(hamiltonian, temperature):
    beta = 1./temperature
    e = scipy.linalg.expm(-beta*hamiltonian)
    t = np.matrix.trace(e)
    return e/t

def find_beta(hamiltonian, entropy, beta0):
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
    exp = scipy.linalg.expm(-beta * hamiltonian)
    Z = np.trace(exp)
    rho = exp / Z
    return np.trace(rho@hamiltonian)

def beta_times_fe(beta, hamiltonian, entropy):
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

def get_exact_and_best_for_ising(beta, nspins):
    anaent = ana_entropy(J=0.,h=-1.,N=nspins,T=1./beta)
    H0 = generate_XYZ(0.,0.,0.,-1.,n_spins=nspins,pbc=True)
    H = generate_XYZ(0.,0.,-1.,-1.,n_spins=nspins,pbc=True)
    betaf = find_beta(H.to_matrix(), anaent, beta)
    exact = exact_energy(betaf, H.to_matrix())
    best = best_approx(beta, H, H0)
    return np.real(exact), np.real(best)
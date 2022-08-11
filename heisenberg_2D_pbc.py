import numpy as np
from hamiltonian import heisenberg_2D_hamiltonian_for_ancilla
from qiskit import Aer
from qiskit.utils import QuantumInstance
from initialise_circuit import ansatz_adia_connectivity_heisenberg
from vqsm import *
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from helper_functions import heisenberg_rectangle_connectivity

Lx = 2
Ly = 4
spins = Lx*Ly
ths = 1e-5
depth = 13
J1 = 1.
J2 = .5
nparams = 3*depth
initial_params = np.zeros(nparams)
for lay in range(depth):
    initial_params[3*lay] = 1. - (lay+1)/depth
    initial_params[3*lay+1] = (lay+1)/depth*J1
    initial_params[3*lay+2] = (lay+1)/depth*J2
lr = 1e-4
shots = 2048*4
backend = Aer.get_backend('statevector_simulator')
instance = QuantumInstance(backend=backend,shots=shots)
expectator = AerPauliExpectation()
opt = 'sgd'
grad = 'spsa'
beta = 5.
additional_data = {}
additional_data['final_coupling'] = J1, J2
additional_data['ansatz'], ans = 'Adiabatic_with_connectivity', ansatz_adia_connectivity_heisenberg
additional_data['model'] = '2D pbc Heisenberg model'
connectivity = heisenberg_rectangle_connectivity(Lx, Ly)
hamiltonian = heisenberg_2D_hamiltonian_for_ancilla(J1, J2, [0]*spins, connectivity, spins)

algo = vqsm(hamiltonian=hamiltonian, ansatz=ans, ansatz_reps=depth, parameters=initial_params, lr=lr, instance=instance, shots=shots, beta=beta, expectator=expectator, connectivity=connectivity)
algo.run(ths, obs_dict={}, filename='test_heisenberg2D.dat', max_iter=300, opt=opt, grad=grad, initial_point=None, additional_data=additional_data)
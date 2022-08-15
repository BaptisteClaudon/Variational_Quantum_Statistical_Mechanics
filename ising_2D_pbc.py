import numpy as np
from hamiltonian import ising_2D_hamiltonian_for_ancilla
from qiskit import Aer
from qiskit.utils import QuantumInstance
from initialise_circuit import ansatz_adia_connectivity
from vqsm import *
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from helper_functions import nearest_neighbours_rectangle

Lx = 2
Ly = 5
spins = Lx*Ly
ths = 1e-6
depth = 10
nparams = 2*depth
final_J = 1./3
initial_params = np.zeros(nparams)
for lay in range(depth):
    initial_params[2*lay] = 1.
    initial_params[2*lay+1] = (lay+1)/depth*final_J
lr = 1e-3
shots = 2048*4
backend = Aer.get_backend('statevector_simulator')
instance = QuantumInstance(backend=backend,shots=shots)
expectator = AerPauliExpectation()
opt = 'sgd'
grad = 'spsa'
beta = 3.
additional_data = {}
additional_data['final_coupling'] = final_J
additional_data['ansatz'], ans = 'Adiabatic_with_connectivity', ansatz_adia_connectivity
additional_data['model'] = '2D pbc Ising model'
connectivity = nearest_neighbours_rectangle(Lx=Lx, Ly=Ly)
hamiltonian = ising_2D_hamiltonian_for_ancilla(J=-final_J, h=-1., neighbours=connectivity, nspins=spins)
filename = 'ising2D_' + str(Lx) + 'X' + str(Ly) + '_spins' + '_beta_' + str(beta) + '.dat'

algo = vqsm(hamiltonian=hamiltonian, ansatz=ans, ansatz_reps=depth, parameters=initial_params, lr=lr, instance=instance, shots=shots, beta=beta, expectator=expectator, connectivity=connectivity)
algo.run(ths, obs_dict={}, filename=filename, max_iter=1000, opt=opt, grad=grad, initial_point=None, additional_data=additional_data)
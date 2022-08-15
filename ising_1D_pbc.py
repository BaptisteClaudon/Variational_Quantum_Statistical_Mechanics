import numpy as np
from hamiltonian import hamiltonian_for_ancilla
from qiskit import Aer
from qiskit.utils import QuantumInstance
from initialise_circuit import ansatz_adia_connectivity
from vqsm import *
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from helper_functions import neirest_neighbours_line

spins = 13
ths = 1e-4
depth = 14
nparams = 2*depth
final_J = 1.5
initial_params = np.zeros(nparams)
for lay in range(depth):
    initial_params[2*lay] = 1.
    initial_params[2*lay+1] = (lay+1)/depth*final_J
lr = 1e-3
hamiltonian = hamiltonian_for_ancilla(J_x=0, J_y=0., J_z=-final_J, field=-1., n_spins=spins, pbc=True)
shots = 2048*4
backend = Aer.get_backend('statevector_simulator')
instance = QuantumInstance(backend=backend,shots=shots)
expectator = AerPauliExpectation()
opt = 'sgd'
grad = 'spsa'
beta = 5.
additional_data = {}
additional_data['final_coupling'] = final_J
additional_data['ansatz'], ans = 'Adiabatic_with_connectivity', ansatz_adia_connectivity
connectivity = neirest_neighbours_line(Lx=spins)
additional_data['model'] = '1D pbc Ising model'

algo = vqsm(hamiltonian=hamiltonian, ansatz=ans, ansatz_reps=depth, parameters=initial_params, lr=lr, instance=instance, shots=shots, beta=beta, expectator=expectator, connectivity=connectivity)
algo.run(ths, obs_dict={}, filename='test_ising.dat', max_iter=300, opt=opt, grad=grad, initial_point=None, additional_data=additional_data)
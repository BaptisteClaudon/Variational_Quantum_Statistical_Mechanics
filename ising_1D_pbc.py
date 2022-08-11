import numpy as np
from hamiltonian import hamiltonian_for_ancilla
from qiskit import Aer
from qiskit.utils import QuantumInstance
from initialise_circuit import ansatz_adia
from vqsm import *
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation

spins = 3
ths = 1e-6
depth = 3
nparams = 2*depth
final_J = 1.5
initial_params = np.zeros(nparams)
for lay in range(depth):
    initial_params[2*lay] = 1.
    initial_params[2*lay+1] = (lay+1)/depth*final_J
lr = 1e-2
hamiltonian = hamiltonian_for_ancilla(J_x=0, J_y=0., J_z=-final_J, field=-1., n_spins=spins, pbc=True)
shots = 2048*4
backend = Aer.get_backend('qasm_simulator')
instance = QuantumInstance(backend=backend,shots=shots)
expectator = AerPauliExpectation()
opt = 'sgd'
grad = 'spsa'
beta = 5.
additional_data = {}
additional_data['final_coupling'] = final_J

algo = vqsm(hamiltonian=hamiltonian, ansatz=ansatz_adia, ansatz_reps=depth, parameters=initial_params, lr=lr, instance=instance, shots=shots, beta=beta, expectator=expectator)
algo.run(ths, obs_dict={}, filename='test_ising.dat', max_iter=300, opt=opt, grad=grad, initial_point=None, additional_data=additional_data)
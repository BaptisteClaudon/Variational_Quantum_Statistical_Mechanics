import numpy as np
import json
import qiskit
from qiskit.circuit import ParameterVector
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from numpy import linalg as LA
from helper_functions import ei, extend_parameters
from hamiltonian import hamiltonian_for_ancilla
from expectation_value import expectation_value, variance_of_operator
from adam_gradient import adam_gradient
from initialise_circuit import ansatz_more_params

class vqsm:
    def __init__(self, hamiltonian, ansatz, ansatz_reps, parameters, lr, instance, shots, beta, expectator, connectivity):
        self.hamiltonian = hamiltonian
        self.instance = instance
        self.parameters = parameters
        self.num_parameters = len(parameters)
        self.shift = lr
        self.shots = shots
        self.expectator = expectator
        self.depth = ansatz_reps
        self.num_qubits = int(hamiltonian.num_qubits/2)
        self.beta = beta
        self.params_vec = ParameterVector('p', self.num_parameters)
        self.connectivity = connectivity
        self.ansatz = ansatz(self.params_vec, (self.num_qubits, self.depth, self.connectivity), self.beta)

    def energy(self, p):
        circuit = self.ansatz.assign_parameters({self.params_vec: p})
        return expectation_value(self.hamiltonian, circuit, self.instance, self.shots, self.expectator)

    def energy_and_gradient_spsa(self):
        Delta = np.random.randint(low=0, high=2, size=(self.num_parameters,))
        Delta = 2.*Delta - 1.
        pplus = self.parameters + self.shift*Delta
        pminus = self.parameters - self.shift*Delta
        fplus = self.energy(pplus)
        fminus = self.energy(pminus)
        grad = (fplus-fminus)/2./self.shift*Delta
        energy_approx = (fplus+fminus)/2.
        return energy_approx, grad

    def energy_and_gradient(self):
        eps = 1e-4
        g = np.zeros((self.num_parameters,))
        for i in range(self.num_parameters):
            e = ei(i, self.num_parameters)
            pplus = self.parameters + eps*e
            pminus = self.parameters - eps*e
            g[i] = 1./2./eps*(self.energy(pplus)-self.energy(pminus))
        return self.energy(self.parameters), g

    def energy_and_gradient_paramshift(self, ansatz_gradient, ext_param_vec):
        eps = np.pi/2
        g = np.zeros((self.num_parameters,))
        ext_params = extend_parameters(self.parameters, self.num_qubits)
        for i in range(self.num_parameters):
            e = ei(i*self.num_qubits, len(ext_params))
            pplus = ext_params + eps * e
            pminus = ext_params - eps * e
            circuit_plus = ansatz_gradient.assign_parameters({ext_param_vec: pplus})
            circuit_minus = ansatz_gradient.assign_parameters({ext_param_vec: pminus})
            g[i] = .5 * self.num_qubits * (expectation_value(self.hamiltonian, circuit_plus, self.instance, self.shots, self.expectator) - expectation_value(self.hamiltonian, circuit_minus, self.instance, self.shots, self.expectator))
        return self.energy(self.parameters), g

    def energy_and_qng(self, X, Z):
        E, saved_g = self.energy_and_gradient_spsa()
        g = saved_g
        tem_params = np.zeros((self.num_parameters,))
        circuit = self.ansatz.assign_parameters({self.params_vec: tem_params})
        for layer in range(self.depth):
            if layer>0:
                var = variance_of_operator(X, circuit, self.instance, self.shots)
                if np.abs(var)>1e-2:
                    g[2*layer] *= 1./var
                else:
                    print("Zero variance at layer "+str(layer)+" and operator X. Using standard gradient.")
                    return E, saved_g
            tem_params[2*layer] = self.parameters[2*layer]
            tem_params[2*layer+1] = self.parameters[2*layer+1]
            circuit = self.ansatz.assign_parameters({self.params_vec: tem_params})
            var = variance_of_operator(Z, circuit, self.instance, self.shots)
            if np.abs(var) > 1e-2:
                g[2*layer+1] *= 1. / var
            else:
                print("Zero variance at layer "+str(layer)+" and operator Z. Using standard gradient.")
                return E, saved_g
        return E, g

    def run(self, ths, obs_dict={}, filename='algo_result.dat', max_iter=100, opt='sgd', grad='spsa', initial_point=None, additional_data=None):
        if self.instance.is_statevector:
            expectation = AerPauliExpectation()
        else:
            expectation = PauliExpectation()
        if initial_point!= None:
            if len(initial_point) != len(self.parameters):
                print("TypeError: Initial parameters are not of the same size of circuit parameters")
                return
            print("\nRestart from: ")
            print(initial_point)
            self.parameters = initial_point
        if len(obs_dict) > 0:
            obs_measure = {}
            obs_error = {}
            for (obs_name, obs_pauli) in obs_dict.items():
                first_measure = expectation_value(obs_pauli, self.ansatz.assign_parameters({self.params_vec: self.parameters}), self.instance, shots=self.shots, expectation=expectation)
                obs_measure[str(obs_name)] = [first_measure[0]]
                obs_error['err_' + str(obs_name)] = [first_measure[1]]
        energies = []
        params = []
        params.append(list(self.parameters))
        count = 0
        norm_grad = ths+1.
        if opt == 'adam':
            m = np.zeros(len(self.parameters))
            v = np.zeros(len(self.parameters))
        if grad == 'qng':
            print('Warning: QNG, check current ansatz compatibility.')
            X_op = hamiltonian_for_ancilla(0., 0., 0., -.5, n_spins=self.num_qubits, pbc=True)
            Z_op =hamiltonian_for_ancilla(0., 0., -.5, 0., n_spins=self.num_qubits, pbc=True)
        if grad == 'pshift':
            print('Warning: Parameter-shift rule, check current ansatz compatibility.')
            ext_param_vec = ParameterVector('p', self.num_parameters*self.num_qubits)
            ansatz_gradient = ansatz_more_params(ext_param_vec, (self.num_qubits, self.depth), self.beta)
        while (norm_grad > ths) and (count < max_iter):
            if len(energies) > 2:
                norm_grad = np.abs(energies[-2] - energies[-1])
            print("Optimizing step:", count + 1)
            count = count + 1
            if grad == 'spsa':
                E, g = self.energy_and_gradient_spsa()
            if grad == 'eps_grad':
                E, g = self.energy_and_gradient()
            if grad == 'qng':
                E, g = self.energy_and_qng(X_op, Z_op)
            if grad == 'pshift':
                E, g = self.energy_and_gradient_paramshift(ansatz_gradient, ext_param_vec)
            print("Energy", E)
            print("Gradient", g)
            if opt=='sgd':
                self.parameters = self.parameters - self.shift*g
            if opt == 'adam':
                meas_grad = np.asarray(g)
                learning_vector = np.asarray(adam_gradient(self.parameters, count, m, v, meas_grad))
                self.parameters = self.parameters - learning_vector
            energies.append(E)
            params.append(list(self.parameters))
            if len(obs_dict) > 0:
                obs_measure = {}
                obs_error = {}
                for (obs_name, obs_pauli) in obs_dict.items():
                    first_measure = expectation_value(obs_pauli, self.ansatz.assign_parameters({self.params_vec: self.parameters}), self.instance, shots=self.shots, expectation=expectation)
                    obs_measure[str(obs_name)] = [first_measure[0]]
                    obs_error['err_' + str(obs_name)] = [first_measure[1]]
        energies.append(self.energy(self.parameters))
        params.append(list(self.parameters))
        log_data = {}
        if len(obs_dict) > 0:
            for (obs_name, obs_pauli) in obs_dict.items():
                log_data[str(obs_name)] = obs_measure[str(obs_name)]
                log_data['err_' + str(obs_name)] = obs_error['err_' + str(obs_name)]
        log_data['variational_energy'] = energies
        log_data['variational_parameters'] = params
        log_data['nspins'] = self.num_qubits
        log_data['beta'] = self.beta
        log_data['optimizer'] = opt
        log_data['gradient'] = grad
        log_data['backend'] = self.instance.backend_name
        log_data['shots'] = self.shots
        log_data['learning_rate'] = self.shift
        log_data['nlayer'] = self.depth
        log_data['threshold'] = ths
        log_data['connectivity'] = self.connectivity
        if additional_data != None:
            for el in additional_data:
                log_data[el] = additional_data[el]
        json.dump(log_data, open(filename, 'w+'))

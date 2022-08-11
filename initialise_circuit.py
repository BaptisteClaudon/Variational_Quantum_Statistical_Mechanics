import numpy as np
from qiskit import QuantumCircuit

def get_correct_proba(beta):
    prob_excited = .5*np.exp(beta)/np.cosh(beta)
    theta = 2.*np.arccos(np.sqrt(prob_excited))
    return theta

def initialise_q_circuit(nspins, beta):
    theta = get_correct_proba(beta)
    qc = QuantumCircuit(2*nspins)
    for k in range(nspins):
        qc.rx(theta=theta, qubit=nspins+k)
        qc.cx(control_qubit=nspins+k, target_qubit=k)
        qc.h(k)
    return qc

def ansatz_more_params(parameters, circ_params, beta):
    nspins, n_layer = circ_params
    circuit = initialise_q_circuit(nspins, beta)
    circuit.barrier()
    count = 0
    for j in range(n_layer):
        for i in range(nspins):
            circuit.rx(parameters[count], i)
            count = count + 1
        for i in range(nspins):
            circuit.rzz(parameters[count],i,(i+1)%nspins)
            count = count + 1
        circuit.barrier()
    return circuit

def ansatz_adia(parameters, circ_params, beta):
    nspins, n_layer =  circ_params
    circuit = initialise_q_circuit(nspins, beta)
    circuit.barrier()
    count = 0
    for j in range(n_layer):
        for i in range(nspins):
            circuit.rx(parameters[count], i)
        count = count + 1
        for i in range(nspins):
            circuit.rzz(parameters[count],i,(i+1)%nspins)
        count = count + 1
        circuit.barrier()
    return circuit

def ansatz_adia_connectivity(parameters, circ_params, beta):
    nspins, nlayer, neighbours = circ_params
    circuit = initialise_q_circuit(nspins, beta)
    circuit.barrier()
    count = 0
    for j in range(nlayer):
        for i in range(nspins):
            circuit.rx(parameters[count], i)
        count = count + 1
        for nei in neighbours:
            i, j = nei
            circuit.rzz(parameters[count],i,j)
        count = count + 1
        circuit.barrier()
    return circuit
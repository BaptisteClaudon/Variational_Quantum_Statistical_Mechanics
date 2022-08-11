from qiskit.opflow.state_fns import CircuitStateFn, StateFn
from qiskit.opflow.expectations import PauliExpectation, AerPauliExpectation
from qiskit.opflow.converters import CircuitSampler

def expectation_value(op, circuit, instance, shots = -1, expectation=AerPauliExpectation()):
    '''
    Expectation value of observable on the circuit.
    :param op: PauliOp, observable
    :param circuit: QuantumCircuit, circuit
    :param instance: QuantumInstance, instance
    :param shots: integer, number of shots
    :param expectation: qiskit.opflow.expectations, method for computing expectation value
    :return: real number, <op>
    '''
    if shots > 0:
        instance.set_config(shots=shots)
    op = StateFn(op, is_measurement=True)
    wfn = CircuitStateFn(circuit)
    opwfn = op @ wfn
    grouped = expectation.convert(opwfn)
    sampled_op = CircuitSampler(instance).convert(grouped)
    mean_value = sampled_op.eval().real
    return mean_value

def variance_of_operator(op, circuit, instance, shots=-1, expectation=PauliExpectation()):
    '''
    Variance of observable on the circuit.
    :param op: PauliOp, observable
    :param circuit: QuantumCircuit, circuit
    :param instance: QuantumInstance, instance
    :param shots: integer, number of shots
    :param expectation: qiskit.opflow.expectations, method for computing expectation value
    :return: real number, <op^2>-<op>^2
    '''
    if shots > 0:
        instance.set_config(shots=shots)
    op = StateFn(op, is_measurement=True)
    wfn = CircuitStateFn(circuit)
    opwfn = op @ wfn
    grouped = expectation.convert(opwfn)
    sampled_op = CircuitSampler(instance).convert(grouped)
    variance = expectation.compute_variance(sampled_op).real
    return variance
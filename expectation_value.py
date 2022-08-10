from qiskit.opflow.state_fns import CircuitStateFn, StateFn
from qiskit.opflow.expectations import PauliExpectation
from qiskit.opflow.converters import CircuitSampler

def expectation_value(op, circuit, instance, shots = -1, expectation=PauliExpectation()):
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
    if shots > 0:
        instance.set_config(shots=shots)
    op = StateFn(op, is_measurement=True)
    wfn = CircuitStateFn(circuit)
    opwfn = op @ wfn
    grouped = expectation.convert(opwfn)
    sampled_op = CircuitSampler(instance).convert(grouped)
    variance = expectation.compute_variance(sampled_op).real
    return variance
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, assemble
import numpy as np
from scipy.optimize import minimize
import random

def apply_fixed_ansatz(qubits, parameters, circ):
    for iz in range(len(qubits)):
        circ.ry(parameters[0][iz], qubits[iz])
    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])
    for iz in range(len(qubits)):
        circ.ry(parameters[1][iz], qubits[iz])
    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])
    for iz in range(len(qubits)):
        circ.ry(parameters[2][iz], qubits[iz])

def run_quantum_simulation(parameters, coefficient_set):
    circ = QuantumCircuit(3, 3)
    apply_fixed_ansatz([0, 1, 2], parameters, circ)
    circ.save_statevector()
    backend = Aer.get_backend('aer_simulator')
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()
    return result.get_statevector(circ, decimals=10)

def run_classical_simulation(coefficient_set):
    a1 = coefficient_set[1] * np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
                                         [0,0,0,0,-1,0,0,0], [0,0,0,0,0,-1,0,0],
                                         [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,-1]])
    a2 = coefficient_set[0] * np.array([[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0],
                                         [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
                                         [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0],
                                         [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,0,1]])
    a3 = np.add(a1, a2)
    b = np.array([float(1/np.sqrt(8))] * 8)
    return (b.dot(a3.dot(b)) / np.linalg.norm(a3.dot(b))) ** 2

def compare_quantum_vs_classical(opt_parameters, coefficient_set):
    quantum_result = run_quantum_simulation(opt_parameters, coefficient_set)
    classical_result = run_classical_simulation(coefficient_set)
    print("Quantum Computation Result:", np.linalg.norm(quantum_result))
    print("Classical Computation Result:", classical_result)
    print("Difference:", abs(np.linalg.norm(quantum_result) - classical_result))

# Optimization setup
coefficient_set = [0.55, 0.45]
out = minimize(lambda params: run_quantum_simulation([params[:3], params[3:6], params[6:9]], coefficient_set),
               x0=[random.uniform(0, 3) for _ in range(9)], method="COBYLA", options={'maxiter': 200})

opt_parameters = [out['x'][0:3], out['x'][3:6], out['x'][6:9]]
compare_quantum_vs_classical(opt_parameters, coefficient_set)

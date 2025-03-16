import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from qiskit.quantum_info import Statevector
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

def control_fixed_ansatz(qubits, parameters, auxiliary, circ):
    for i in range(len(qubits)):
        circ.cry(parameters[0][i], auxiliary, qubits[i])
    
    circ.ccx(auxiliary, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(auxiliary, qubits[1], 4)
    
    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    
    for i in range(len(qubits)):
        circ.cry(parameters[1][i], auxiliary, qubits[i])
    
    circ.ccx(auxiliary, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(auxiliary, qubits[2], 4)
    
    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    
    for i in range(len(qubits)):
        circ.cry(parameters[2][i], auxiliary, qubits[i])

def prepare_b(circ, qubits):
    for q in qubits:
        circ.h(q)

def calculate_overlap(parameters, coefficient_set, gate_set, qubits, auxiliary, backend):
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    
    # Compute ⟨x|A†A|x⟩
    qr = QuantumRegister(5)
    cr = ClassicalRegister(1)
    circ = QuantumCircuit(qr, cr)
    apply_fixed_ansatz(qubits, parameters, circ)
    
    # Apply A to |x⟩
    circ2 = QuantumCircuit(qr)
    control_fixed_ansatz(qubits, parameters, auxiliary, circ2)
    circ.compose(circ2, inplace=True)
    
    circ.save_statevector()
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    result = job.result()
    state = result.get_statevector()
    A_x = Statevector(state)
    norm_Ax = np.linalg.norm(A_x.data)**2
    
    # Compute |⟨b|A|x⟩|^2
    qr = QuantumRegister(5)
    circ = QuantumCircuit(qr)
    apply_fixed_ansatz(qubits, parameters, circ)
    control_fixed_ansatz(qubits, parameters, auxiliary, circ)
    prepare_b(circ, qubits)
    circ.h(auxiliary)
    circ.measure_all()
    
    # FIX: Reduced optimization level to avoid bug in older Qiskit versions
    t_circ = transpile(circ, backend, optimization_level=1)  # Changed from 3 to 1
    counts = backend.run(t_circ).result().get_counts()
    overlap_prob = counts.get('0' * 5, 0) / sum(counts.values())
    
    cost = 1 - overlap_prob / norm_Ax
    print(f"Cost: {cost}")
    return cost

if __name__ == "__main__":
    coefficient_set = [0.55, 0.45]
    gate_set = [[0,0,0], [0,0,1]]
    initial_params = [random.uniform(0, 2*np.pi) for _ in range(9)]
    
    backend = Aer.get_backend('aer_simulator')
    
    result = minimize(calculate_overlap, initial_params,
                     args=(coefficient_set, gate_set, [1,2,3], 0, backend),
                     method="COBYLA", options={'maxiter': 200})
    
    opt_params = result.x
    opt_params_reshaped = [opt_params[0:3], opt_params[3:6], opt_params[6:9]]
    
    # Generate solution state
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr)
    apply_fixed_ansatz([0,1,2], opt_params_reshaped, circ)
    
    backend = Aer.get_backend('statevector_simulator')
    t_circ = transpile(circ, backend)
    solution_state = backend.run(t_circ).result().get_statevector()
    print("\nSolution state vector:")
    print(solution_state)
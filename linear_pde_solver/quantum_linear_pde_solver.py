import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import numpy as np
from scipy.optimize import minimize
import random

# 1. Define quantum subroutines with explicit circuit passing
def apply_fixed_ansatz(circ, qubits, parameters):
    """Parameterized quantum circuit ansatz"""
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

def had_test(circ, gate_type, qubits, auxiliary_index, parameters):
    """Hadamard test for expectation values"""
    circ.h(auxiliary_index)
    apply_fixed_ansatz(circ, qubits, parameters)
    
    for ie in range(len(gate_type[0])):
        if gate_type[0][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    
    for ie in range(len(gate_type[1])):
        if gate_type[1][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    
    circ.h(auxiliary_index)

# 2. Define problem-specific parameters
coefficient_set = [2.0, -1.0, -1.0]  # A = 2I - XX - XX
gate_set = [
    [[0, 0, 0], [0, 0, 0]],  # Identity term
    [[1, 1, 0], [0, 0, 0]],  # XX term (first two qubits)
    [[0, 0, 0], [0, 1, 1]]   # XX term (last two qubits)
]

# Normalized source term vector (dp/dx = 1)
b = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=float)
b /= np.linalg.norm(b)

# 3. Modified cost function calculation
def calculate_cost_function(parameters):
    """Calculate VQLS cost function for Stokes flow"""
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    overall_sum_1 = 0.0
    
    # Compute Tr[A†Aψ]
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            qctl = QuantumRegister(5)
            qc = ClassicalRegister(1)
            circ = QuantumCircuit(qctl, qc)
            
            multiply = coefficient_set[i] * coefficient_set[j]
            had_test(circ, gate_set[i], [1, 2, 3], 0, parameters)
            
            # Simulate quantum circuit
            backend = Aer.get_backend('aer_simulator')
            circ.save_statevector()
            t_circ = transpile(circ, backend)
            qobj = assemble(t_circ)
            job = backend.run(qobj)
            result = job.result()
            
            # Calculate probabilities
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
            overall_sum_1 += multiply * (1 - 2*m_sum)

    # Compute |<b|A|ψ>|^2
    overall_sum_2 = 0.0
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            multiply = coefficient_set[i] * coefficient_set[j]
            mult = 1.0
            
            for extra in [0, 1]:
                qctl = QuantumRegister(5)
                qc = ClassicalRegister(1)
                circ = QuantumCircuit(qctl, qc)
                
                if extra == 0:
                    # Control-A operator
                    for ty in range(len(gate_set[i][0])):
                        if gate_set[i][0][ty] == 1:
                            circ.cz(0, ty+1)
                else:
                    # Control-b preparation
                    for ty in range(len(gate_set[j][1])):
                        if gate_set[j][1][ty] == 1:
                            circ.cz(0, ty+1)
                
                # Simulate circuit
                backend = Aer.get_backend('aer_simulator')
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                result = job.result()
                
                # Calculate probabilities
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
                mult *= (1 - 2*m_sum)
            
            overall_sum_2 += multiply * mult
    
    cost = 1 - float(overall_sum_2 / overall_sum_1)
    print(f"Cost: {cost:.4f}")
    return cost

# 4. Run optimization
np.random.seed(42)
initial_params = [random.uniform(0, 2*np.pi) for _ in range(9)]

result = minimize(calculate_cost_function,
                 x0=initial_params,
                 method="COBYLA",
                 options={'maxiter': 50})

print("\nOptimization results:")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Final cost: {result.fun:.4f}")
print(f"Evaluations: {result.nfev}")

# 5. Post-process results
if result.success:
    optimal_params = [result.x[0:3], result.x[3:6], result.x[6:9]]
    
    # Get optimal statevector
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr)
    apply_fixed_ansatz(circ, [0, 1, 2], optimal_params)
    circ.save_statevector()
    
    backend = Aer.get_backend('aer_simulator')
    t_qc = transpile(circ, backend)
    qobj = assemble(t_qc)
    job = backend.run(qobj)
    optimal_state = job.result().get_statevector(circ)
    
    # Construct Stokes matrix
    I = np.eye(2)
    X = np.array([[0, 1], [1, 0]])
    A = 2.0 * np.kron(np.kron(I, I), I) \
        -1.0 * np.kron(np.kron(X, X), I) \
        -1.0 * np.kron(I, np.kron(X, X))
    
    # Calculate solution fidelity
    solution = A @ optimal_state
    solution /= np.linalg.norm(solution)
    fidelity = np.abs(b @ solution)**2
    print(f"\nSolution fidelity: {fidelity:.4f}")
else:
    print("\nOptimization failed to converge")
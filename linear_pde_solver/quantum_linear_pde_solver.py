import qiskit
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import Aer, transpile, assemble
import numpy as np
from scipy.optimize import minimize
import random

# Numerical stability parameters
EPSILON = 1e-8
MAX_COST = 1e3

def apply_fixed_ansatz(circ, qubits, parameters):
    """Parameterized quantum circuit ansatz with input validation"""
    assert len(parameters) == 3, "Parameters need 3 layers"
    assert all(len(layer) == 3 for layer in parameters), "3 parameters per layer needed"
    
    for layer in parameters:
        for q, angle in zip(qubits, layer):
            circ.ry(angle, q)
        
        circ.cz(qubits[0], qubits[1])
        circ.cz(qubits[2], qubits[0])

def had_test(circ, gate_type, qubits, auxiliary_index, parameters):
    """Stabilized Hadamard test implementation"""
    circ.h(auxiliary_index)
    apply_fixed_ansatz(circ, qubits, parameters)
    
    # Apply controlled operations with validation
    for part in gate_type:
        if sum(part) == 0:  # Identity operation
            circ.id(auxiliary_index)
        else:
            for q, active in enumerate(part):
                if active:
                    circ.cz(auxiliary_index, qubits[q])
    
    circ.h(auxiliary_index)

# Stabilized problem parameters
coefficient_set = [2.0, -1.0, -1.0]  # A = 2I - XXI - IXX
gate_set = [
    [[0,0,0], [0,0,0]],  # Identity
    [[1,1,0], [0,0,0]],  # XX on first two qubits
    [[0,0,0], [0,1,1]]   # XX on last two qubits
]

# Regularized source term
b = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.float64)
b_norm = np.linalg.norm(b)
b = b / (b_norm + EPSILON if b_norm < EPSILON else b_norm)

def calculate_cost_function(parameters):
    """Numerically stabilized cost function"""
    try:
        # Parameter validation
        parameters = np.array(parameters).astype(np.float64)
        if np.any(np.isnan(parameters)):
            return MAX_COST
            
        params_reshaped = [
            parameters[0:3],
            parameters[3:6],
            parameters[6:9]
        ]
        
        overall_sum_1 = EPSILON  # Initialize with epsilon
        overall_sum_2 = 0.0
        
        # Compute Tr[A†Aψ]
        for i in range(len(gate_set)):
            for j in range(len(gate_set)):
                qr = QuantumRegister(5)
                circ = QuantumCircuit(qr)
                
                had_test(circ, gate_set[i], [1,2,3], 0, params_reshaped)
                
                # Stabilized simulation
                backend = Aer.get_backend('aer_simulator')
                circ.save_statevector()
                result = backend.run(transpile(circ, backend)).result()
                outputstate = np.real(result.get_statevector(circ))
                
                # Probability calculation with clipping
                prob_1 = np.clip(np.sum(outputstate[1::2]**2), 0, 1)
                term = coefficient_set[i] * coefficient_set[j] * (1 - 2*prob_1)
                overall_sum_1 += abs(term)  # Use absolute value for stability

        # Compute |<b|A|ψ>|^2 with regularization
        for i in range(len(gate_set)):
            for j in range(len(gate_set)):
                product_term = coefficient_set[i] * coefficient_set[j]
                if abs(product_term) < EPSILON:
                    continue
                
                # Double Hadamard test
                for extra in [0, 1]:
                    qr = QuantumRegister(5)
                    circ = QuantumCircuit(qr)
                    
                    if extra == 0:
                        had_test(circ, gate_set[i], [1,2,3], 0, params_reshaped)
                    else:
                        had_test(circ, gate_set[j], [1,2,3], 0, params_reshaped)
                    
                    result = Aer.get_backend('aer_simulator').run(
                        transpile(circ, Aer.get_backend('aer_simulator'))
                    ).result()
                    outputstate = np.real(result.get_statevector(circ))
                    
                    prob_1 = np.clip(np.sum(outputstate[1::2]**2), 0, 1)
                    product_term *= (1 - 2*prob_1)
                
                overall_sum_2 += product_term

        # Final stabilized calculation
        denominator = overall_sum_1 if overall_sum_1 > EPSILON else EPSILON
        cost = 1 - np.clip(overall_sum_2 / denominator, -1, 1)
        return float(np.nan_to_num(cost, nan=MAX_COST))
    
    except Exception as e:
        print(f"Error in cost calculation: {str(e)}")
        return MAX_COST

# Robust optimization setup
np.random.seed(42)
initial_params = np.random.uniform(0, 2*np.pi, 9)

result = minimize(
    calculate_cost_function,
    x0=initial_params,
    method="SLSQP",
    bounds=[(0, 2*np.pi)]*9,
    options={'maxiter': 100, 'ftol': 1e-6}
)

print("\nOptimization results:")
print(f"Success: {result.success}")
print(f"Message: {result.message}")
print(f"Final cost: {result.fun:.4f}")

# Post-processing with validation
if result.success and not np.isnan(result.fun):
    optimal_params = [
        result.x[0:3],
        result.x[3:6],
        result.x[6:9]
    ]
    
    # Quantum state reconstruction
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr)
    apply_fixed_ansatz(circ, [0,1,2], optimal_params)
    circ.save_statevector()
    
    optimal_state = Aer.get_backend('aer_simulator').run(
        transpile(circ, Aer.get_backend('aer_simulator'))
    ).result().get_statevector(circ)
    
    # Classical verification
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    A = 2.0 * np.kron(np.kron(I,I), I) \
        -1.0 * np.kron(np.kron(X,X), I) \
        -1.0 * np.kron(I, np.kron(X,X))
    
    solution = A @ optimal_state
    solution /= np.linalg.norm(solution) + EPSILON
    fidelity = np.abs(b @ solution)**2
    print(f"\nSolution fidelity: {fidelity:.4f}")
else:
    print("Optimization failed - try adjusting initial parameters or increasing iterations")
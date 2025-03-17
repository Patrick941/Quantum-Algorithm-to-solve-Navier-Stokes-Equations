import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms import VQLS
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import QuantumInstance
from qiskit.visualization import plot_histogram

# Step 1: Discretize the Poisson equation (1D example)
# Define the matrix A and vector b for the linear system Aϕ = b
A = np.array([[2, -1], [-1, 2]])  # Tridiagonal matrix
b = np.array([3, 0])  # Right-hand side vector

# Classical solution
classical_solution = np.linalg.solve(A, b)
print("Classical solution:", classical_solution)

# Step 2: Quantum solution using VQLS
# Encode A as a sum of Pauli matrices
pauli_A = SparsePauliOp.from_list([('II', 2), ('XX', -1), ('YY', -1), ('ZZ', -1)])

# Encode normalized b as a quantum state
qc_b = QuantumCircuit(2)
qc_b.initialize([1, 0], 0)  # |b> = |0>

# Configure VQLS
ansatz = EfficientSU2(2, reps=1)  # 2-qubit ansatz
optimizer = COBYLA(maxiter=100)  # Optimizer
backend = Aer.get_backend('statevector_simulator')  # Simulator backend
quantum_instance = QuantumInstance(backend)

# Solve using VQLS
vqls = VQLS(ansatz, optimizer, quantum_instance=quantum_instance)
result = vqls.solve(pauli_A, qc_b)

# Extract the quantum solution
solution_circuit = result.solution
tqc = transpile(solution_circuit, backend)
tqc.save_statevector()
statevector = backend.run(tqc).result().get_statevector()

# Normalize the classical solution for comparison
classical_normalized = classical_solution / np.linalg.norm(classical_solution)

# Step 3: Compare quantum and classical results
# Method 1: Fidelity check
fidelity = np.abs(np.dot(statevector, classical_normalized))**2
print(f"Fidelity between quantum and classical solutions: {fidelity:.4f}")

# Method 2: Error norm
scaled_quantum = statevector * np.linalg.norm(classical_solution)
error = np.linalg.norm(scaled_quantum - classical_solution)
print(f"Error norm between quantum and classical solutions: {error:.4f}")

# Method 3: Measurement sampling
# Add measurement to the solution circuit
meas_circuit = solution_circuit.copy()
meas_circuit.measure_all()

# Simulate measurements
backend = Aer.get_backend('qasm_simulator')
counts = backend.run(meas_circuit, shots=1000).result().get_counts()
print("Measurement counts:", counts)
plot_histogram(counts)  # Visualize the measurement results
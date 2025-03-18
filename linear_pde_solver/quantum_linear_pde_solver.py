import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, execute
from qiskit.quantum_info import Statevector
from qiskit.algorithms.linear_solvers import NumPyMatrix, HHL
from qiskit.utils import QuantumInstance

# Define the matrix A (Hermitian) and vector b
A = np.array([[1, 0], [0, 2]], dtype=float)
b = np.array([1, 0], dtype=float)

# Convert A into a format suitable for HHL
matrix = NumPyMatrix(A)

# Set up the quantum instance (simulator)
backend = Aer.get_backend('statevector_simulator')
quantum_instance = QuantumInstance(backend)

# Initialize the HHL solver
hhl_solver = HHL(quantum_instance=quantum_instance)

# Solve the linear system
result = hhl_solver.solve(matrix, b)

# Extract the quantum state solution
solution_circuit = result.state

# Transpile and run the circuit to get the statevector
transpiled_circuit = transpile(solution_circuit, backend)
job = backend.run(transpiled_circuit)
statevector = job.result().get_statevector()

# The solution vector is embedded in the statevector; extract and normalize
# Note: Indices depend on qubit ordering and ancilla qubits
# This step is heuristic and problem-specific
solution = np.array([statevector[0], statevector[2]])  # Adjust indices as needed
solution_normalized = solution / np.linalg.norm(solution)

print("Quantum Solution (Normalized):", solution_normalized)
print("Classical Solution:", np.linalg.solve(A, b))
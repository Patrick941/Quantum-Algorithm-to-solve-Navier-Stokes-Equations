# Install Qiskit (if not installed)
# !pip install qiskit qiskit-aer

import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms.linear_solvers import HHL
from qiskit.algorithms.linear_solvers.matrices import NumPyMatrix
from qiskit.algorithms.linear_solvers.observables import MatrixFunctional

# Step 1: Define the linear system (e.g., 1D Poisson equation: A * x = b)
A = np.array([[4, -1], [-1, 4]])  # Discretized Laplacian matrix (small example)
b = np.array([1, 0])              # Source term/boundary conditions

# Step 2: Solve classically (for comparison)
x_classical = np.linalg.solve(A, b)
print(f"Classical solution: {x_classical}")

# Step 3: Quantum solution using HHL (simulator-based)
# Encode the matrix and vector
matrix = NumPyMatrix(A)
observable = MatrixFunctional()
hhl = HHL(quantum_instance=Aer.get_backend('aer_simulator'))

# Solve the system
result = hhl.solve(matrix, b, observable)
solution_state = result.state

# Step 4: Post-process results (measure probabilities)
qc = QuantumCircuit(2)
qc.initialize(solution_state, [0, 1])
qc.measure_all()

# Simulate measurements
simulator = Aer.get_backend('aer_simulator')
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1000).result()
counts = result.get_counts()

# Extract probabilities (approximate solution)
probabilities = {k: v / 1000 for k, v in counts.items()}
print("\nQuantum solution (probabilities):", probabilities)
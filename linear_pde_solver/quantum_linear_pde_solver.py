import numpy as np
from qiskit import BasicAer, execute
from qiskit.circuit.library import HHL
from qiskit.algorithms.linear_solvers.matrices.linear_system_matrix import LinearSystemMatrix
from qiskit.opflow import Z, I
from qiskit.utils import QuantumInstance

# Example: Discretize a simple 1D PDE into A·x = b
# Here, A is a small matrix representing the discretized system, and b is the RHS.

# Discretized matrix (adjust as needed for your PDE)
A = np.array([[2, -1,  0],
              [-1, 2, -1],
              [0, -1,  2]], dtype=float)
# Corresponding b vector
b = np.array([1, 0, 1], dtype=float)

# Convert A and b to operator form for the HHL algorithm
matrix = LinearSystemMatrix(A)
hhl = HHL()

# Execute HHL circuit on a simulator
backend = BasicAer.get_backend("statevector_simulator")
quantum_instance = QuantumInstance(backend)
solution = hhl.solve(matrix, b, quantum_instance)

print("Solution from HHL:", solution.solution)
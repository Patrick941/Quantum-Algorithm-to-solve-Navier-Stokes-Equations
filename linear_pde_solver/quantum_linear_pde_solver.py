import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms.linear_solvers import HHL
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import QuantumInstance
from qiskit.circuit.library import LinearAmplitudeFunction

# Step 1: Define the linear system (1D Poisson equation)
A = np.array([[4, -1], [-1, 4]])  # Discretized Laplacian
b = np.array([1, 0])              # Source term

# Step 2: Classical solution (for validation)
x_classical = np.linalg.solve(A, b)
print(f"Classical solution: {x_classical}")

# Step 3: Quantum setup with updated classes
# -----------------------------------------
# (a) Encode the matrix using current methods
def matrix_to_circuit(A):
    """Convert matrix to a quantum circuit (for demonstration)"""
    return LinearAmplitudeFunction(A, approximation_degree=1)

# (b) Define observable (now required to be Hermitian)
observable = SparsePauliOp.from_list([("IZ", 1), ("ZI", 1)])  # Custom observable

# (c) Configure HHL with current interface
hhl = HHL()
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'))

# (d) Solve the system
result = hhl.solve(
    matrix=matrix_to_circuit(A),
    vector=b,
    observable=observable,
    quantum_instance=quantum_instance
)

# Step 4: Post-process results
# ----------------------------
# Get normalized solution state
solution_state = result.state

# Create measurement circuit
qc = QuantumCircuit(2)
qc.initialize(solution_state, [0, 1])
qc.measure_all()

# Simulate measurements
simulator = Aer.get_backend('aer_simulator')
transpiled_qc = transpile(qc, simulator)
result = simulator.run(transpiled_qc, shots=1000).result()
counts = result.get_counts()

# Calculate probabilities
probabilities = {k: v / 1000 for k, v in counts.items()}
print("\nQuantum solution (probabilities):", probabilities)
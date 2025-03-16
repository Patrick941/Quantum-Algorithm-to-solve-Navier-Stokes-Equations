import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit.utils import QuantumInstance
from qiskit.algorithms import HHL

# Step 1: Define the linear system (1D Poisson equation)
A = np.array([[4, -1], [-1, 4]])  # Discretized Laplacian
b = np.array([1, 0])              # Source term

# Step 2: Classical solution (for validation)
x_classical = np.linalg.solve(A, b)
print(f"Classical solution: {x_classical}")

# Step 3: Quantum setup with updated matrix encoding
# -----------------------------------------
# (a) Encode the matrix using LinearAmplitudeFunction
def matrix_to_circuit(A):
    """Convert matrix to a quantum circuit"""
    # Define the domain and image of the matrix
    domain = (0, 1)  # Example domain (adjust based on your problem)
    image = (0, 1)   # Example image (adjust based on your problem)
    slope = 1.0      # Slope for linear encoding
    offset = 0.0     # Offset for linear encoding
    return LinearAmplitudeFunction(
        slope=slope,
        offset=offset,
        domain=domain,
        image=image,
        function=lambda x: np.dot(A, x)  # Matrix multiplication
    )

# (b) Define the quantum solver (HHL is deprecated, but used here for demonstration)
quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator'))
hhl = HHL(quantum_instance=quantum_instance)

# (c) Solve the system
result = hhl.solve(
    matrix=matrix_to_circuit(A),
    vector=b
)

# Step 4: Post-process results
# ----------------------------
# Get the solution state
solution_state = result.statevector

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
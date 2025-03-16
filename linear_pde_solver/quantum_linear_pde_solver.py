import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms.linear_solvers import HHL
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import PauliEvolutionGate

# Step 1: Discretize PDE into Ax = b (1D Poisson example)
N = 4  # Use a power of 2 (e.g., 4, 8, 16)
h = 1.0 / (N + 1)  # Step size

# Construct tridiagonal matrix A (Hermitian)
diag = 2 / h**2 * np.ones(N)
off_diag = -1 / h**2 * np.ones(N-1)
A = np.diag(off_diag, -1) + np.diag(diag, 0) + np.diag(off_diag, 1)

# Step 2: Encode A as a sparse Pauli operator
pauli_terms = []
for i in range(N):
    for j in range(N):
        if i == j:  # Diagonal terms (Z-based encoding)
            pauli_str = 'I' * i + 'Z' + 'I' * (N - i - 1)
            pauli_terms.append((pauli_str, A[i, j]))
        elif abs(i - j) == 1:  # Off-diagonal terms (X-based encoding)
            pauli_str = 'I' * min(i, j) + 'X' + 'I' * abs(i - j - 1) + 'X' + 'I' * (N - max(i, j) - 1)
            pauli_terms.append((pauli_str, A[i, j] / 2))
hamiltonian = SparsePauliOp.from_list(pauli_terms)

# Step 3: Prepare |b> state (length must be 2^num_qubits)
num_qubits = int(np.log2(N))  # N must be a power of 2
b = np.array([1, 0, 0, 0])  # Example source term (length = N)
b_norm = b / np.linalg.norm(b)
qc_b = QuantumCircuit(num_qubits)
qc_b.initialize(b_norm, qc_b.qubits)

# Step 4: Configure HHL with PauliEvolutionGate
evolution_gate = PauliEvolutionGate(hamiltonian, time=1.0)
hhl = HHL()
qc_hhl = hhl.solve(evolution_gate, qc_b)

# Step 5: Simulate on a backend
backend = Aer.get_backend('aer_simulator')
qc_hhl = transpile(qc_hhl, backend)
result = backend.run(qc_hhl).result()

print("Solution probabilities:", result.get_counts())
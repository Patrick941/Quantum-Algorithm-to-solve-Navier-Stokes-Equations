import numpy as np
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms.linear_solvers import HHL
from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
from qiskit.opflow import MatrixOp
from qiskit.quantum_info import Operator
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp

# Step 1: Discretize PDE into Ax = b (1D Poisson example)
N = 3  # System size (small example)
h = 1.0 / (N + 1)  # Step size

# Create tridiagonal matrix A (Hermitian by construction)
diag = 2 / h**2 * np.ones(N)
off_diag = -1 / h**2 * np.ones(N-1)
A = np.diag(off_diag, -1) + np.diag(diag, 0) + np.diag(off_diag, 1)

# Step 2: Encode A using SparsePauliOp (for Hamiltonian simulation)
pauli_terms = []
for i in range(N):
    for j in range(N):
        if A[i, j] != 0:
            pauli_str = 'I'*i + 'X' + 'I'*(N-i-1)  # Simplified placeholder
            pauli_terms.append((pauli_str, A[i, j]))
hamiltonian = SparsePauliOp.from_list(pauli_terms)

# Step 3: Prepare |b> state
b = np.array([1, 0, 0])  # Example source term
b_norm = b / np.linalg.norm(b)
qc_b = QuantumCircuit(N)
qc_b.initialize(b_norm, qc_b.qubits)

# Step 4: Configure HHL with evolution gate
evolution_gate = PauliEvolutionGate(hamiltonian, time=1.0)
hhl = HHL()
qc_hhl = hhl.solve(evolution_gate, qc_b)

# Step 5: Simulate
backend = Aer.get_backend('aer_simulator')
qc_hhl = transpile(qc_hhl, backend)
result = backend.run(qc_hhl).result()

# Step 6: Post-process (probabilities ≈ solution squared magnitudes)
solution_probs = result.get_counts()
print("Solution probabilities:", solution_probs)
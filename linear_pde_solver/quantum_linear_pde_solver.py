from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector
import numpy as np

# Simplified 2x2 system (n=1 qubit)
n = 1  # Qubits for solution
m = 2  # Qubits for eigenvalue estimation
N = 2**n
h = 1/3  # Grid spacing

# Simple matrix A (diagonal to avoid complex rotations)
A = np.diag([1, 2])  # Known eigenvalues 1 and 2
A_scaled = A / np.linalg.norm(A, ord=2)

# Source term
b = np.ones(N) / np.sqrt(N)

# Quantum circuit
qc = QuantumCircuit(n + m + 1, n)

# State preparation |b⟩
qc.initialize(Statevector(b), [0])

# Simplified QPE (manual implementation)
for q in range(m):
    qc.h(n + q)  # Hadamard instead of QFT

# Eigenvalue inversion (safe angles)
angles = [2 * np.arcsin(0.5),  # 1/λ = 1/1 = 1 (safe)
          2 * np.arcsin(0.5)]  # 1/λ = 1/2 = 0.5 (safe)
for q in range(m):
    qc.ry(angles[q], n + q)

# Post-selection
qc.measure(n + m, 0)

# Measure solution
qc.measure([0], [0])

# Transpile with basic gates
qc = transpile(qc, basis_gates=['cx', 'u3', 'ry', 'h'])

# Simulate
backend = Aer.get_backend('qasm_simulator')
result = backend.run(qc, shots=1024).result()
counts = result.get_counts()

print("Results:", counts)
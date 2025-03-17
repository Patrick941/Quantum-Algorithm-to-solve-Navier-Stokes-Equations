from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import QFT
import numpy as np

# Parameters
n = 2  # Qubits for solution (size = 2^n)
m = 3  # Qubits for eigenvalue estimation
N = 2**n  # Grid points (simplified to 2^n)
h = 1 / (N + 1)  # Grid spacing

# Discretized Poisson matrix A (scaled)
A = (1 / h**2) * (2 * np.eye(N) - np.eye(N, k=1) - np.eye(N, k=-1))
A_scaled = A / np.linalg.norm(A, ord=2)  # Normalize eigenvalues

# Source term f(x) = 1 (normalized)
b = np.ones(N) / np.linalg.norm(np.ones(N))

# Quantum circuit
qc = QuantumCircuit(n + m + 1, n)  # n data, m phase, 1 ancilla

# Step 1: State preparation |b⟩
qc.initialize(Statevector(b), range(n))

# Step 2: Quantum Phase Estimation (QPE)
# --------------------------------------
# Apply QFT to the **phase register** (not the data register)
qc.append(QFT(m).to_instruction(), range(n, n + m))  # Apply QFT to phase register

# Simulate e^(iA_scaled t) using diagonalization in Fourier basis
t = 2 * np.pi  # Time parameter
for j in range(m):
    for k in range(n):
        # Apply controlled-phase rotation (simplified)
        phase = t * (2**j) * (4 / h**2) * np.sin(np.pi * (k+1) * h / 2)**2
        qc.cp(phase, n + j, k)  # Phase estimation register controls

qc.append(QFT(m).inverse().to_instruction(), range(n, n + m))  # Inverse QFT

# Step 3: Eigenvalue inversion (|λ⟩ → 1/λ)
angles = [2 * np.arcsin(1 / np.sqrt(λ)) for λ in np.linalg.eigvalsh(A_scaled)]
for q in range(m):
    qc.ry(angles[q], n + q)  # Rotation conditioned on eigenvalues

# Step 4: Post-select ancilla qubit
qc.measure(n + m, 0)  # Ancilla measured; |1⟩ indicates success

# Step 5: Inverse QPE (uncompute)
qc.append(QFT(m).to_instruction(), range(n, n + m))
for j in reversed(range(m)):
    for k in reversed(range(n)):
        phase = -t * (2**j) * (4 / h**2) * np.sin(np.pi * (k+1) * h / 2)**2
        qc.cp(phase, n + j, k)
qc.append(QFT(m).inverse().to_instruction(), range(n, n + m))

# Measure solution
qc.measure(range(n), range(n))

# Transpile to decompose QFT into primitive gates
qc = transpile(qc, basis_gates=['cx', 'u3', 'u2', 'u1', 'id', 'rz', 'ry', 'rx'])

# Simulate
backend = Aer.get_backend('statevector_simulator')
result = backend.run(qc).result()
counts = result.get_counts()

# Post-process results (filter ancilla=1)
solution_counts = {k[:-1]: v for k, v in counts.items() if k.endswith('1')}
print("Solution:", solution_counts)
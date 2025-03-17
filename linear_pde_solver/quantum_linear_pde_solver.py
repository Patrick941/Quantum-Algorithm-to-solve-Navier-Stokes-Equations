from qiskit import QuantumCircuit, Aer, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector
import numpy as np

def solve_1d_poisson(n_qubits=2):
    # Discretize 1D Poisson equation
    N = 2**n_qubits
    h = 1/(N + 1)
    main_diag = 2/h**2 * np.ones(N)
    off_diag = -1/h**2 * np.ones(N-1)
    
    # Construct tridiagonal matrix
    A = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    A /= np.linalg.norm(A, ord=2)  # Normalize matrix
    
    # Prepare state |b>
    b = np.ones(N)
    b /= np.linalg.norm(b)
    
    # Manual HHL implementation
    n = n_qubits  # System qubits
    m = 3         # Eigenvalue estimation qubits
    qc = QuantumCircuit(n + m + 1, n)
    
    # State preparation
    qc.initialize(Statevector(b), range(n))
    
    # QPE
    qc.append(QFT(m), range(n, n+m))
    for j in range(m):
        angle = 2*np.pi*(2**j)*np.linalg.eigvalsh(A)[0]
        qc.cp(angle, n+j, 0)
    qc.append(QFT(m).inverse(), range(n, n+m))
    
    # Eigenvalue inversion (simplified)
    qc.ry(2*np.arcsin(1/np.linalg.eigvalsh(A)[0]), n+m-1)
    
    # Post-selection
    qc.measure(n+m, 0)
    
    # Simulate
    backend = Aer.get_backend('statevector_simulator')
    result = backend.run(qc).result()
    statevector = result.get_statevector()
    
    # Extract solution
    solution = {}
    for i, amp in enumerate(statevector):
        if np.abs(amp) > 1e-3:
            solution[bin(i)[2:].zfill(n+m+1)] = amp
            
    return solution

if __name__ == "__main__":
    solution = solve_1d_poisson()
    print("Quantum Solution Components:")
    for state, amp in solution.items():
        print(f"{state}: {np.abs(amp):.4f}")
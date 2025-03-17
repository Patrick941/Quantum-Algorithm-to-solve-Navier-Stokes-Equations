from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms import HHL
from qiskit.quantum_info import Statevector
from qiskit.algorithms.linear_solvers import NumPyLinearSolver
import numpy as np

def solve_1d_poisson(n_qubits=3):
    # Create tridiagonal matrix
    N = 2**n_qubits
    h = 1/(N + 1)
    main_diag = 2/h**2 * np.ones(N)
    off_diag = -1/h**2 * np.ones(N-1)
    
    # Create matrix using standard numpy array
    matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
    
    # Normalize matrix and vector
    matrix /= np.linalg.norm(matrix, ord=2)
    b = np.ones(N)
    b /= np.linalg.norm(b)

    # Use basic HHL solver
    hhl = HHL()
    result = hhl.solve(matrix, b)
    
    # Extract solution components
    solution_vector = Statevector(result.state).data.real
    solution = {}
    for i, val in enumerate(solution_vector):
        if abs(val) > 1e-5:
            solution[i] = val * result.euclidean_norm
            
    return solution

if __name__ == "__main__":
    solution = solve_1d_poisson(n_qubits=2)
    print("Quantum Solution:")
    for state, value in solution.items():
        print(f"State {bin(state)[2:].zfill(2)}: {value:.4f}")
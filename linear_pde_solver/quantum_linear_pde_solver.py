from qiskit import QuantumCircuit, Aer, transpile
from qiskit.quantum_info import Statevector
from qiskit.algorithms.linear_solvers import HHL
from qiskit.algorithms.linear_solvers.matrices import TridiagonalToeplitz
from qiskit.algorithms.linear_solvers.observables import AbsoluteAverage
import numpy as np

def solve_poisson_1d(n_qubits=3, h=0.25):
    # Discretize 1D Poisson equation
    N = 2**n_qubits
    main_diag = 2/h**2 * np.ones(N)
    off_diag = -1/h**2 * np.ones(N-1)
    
    # Create tridiagonal Toeplitz matrix
    matrix = TridiagonalToeplitz(n_qubits, main_diag[0], off_diag[0])
    
    # Source term (normalized)
    b = np.ones(N)
    b /= np.linalg.norm(b)

    # Configure HHL solver
    hhl_solver = HHL(
        quantum_instance=Aer.get_backend('statevector_simulator'),
        epsilon=1e-2
    )

    # Solve system
    result = hhl_solver.solve(matrix, b, AbsoluteAverage())

    # Process results
    solution_vector = Statevector(result.state).data.real
    norm = result.euclidean_norm
    
    # Extract meaningful state components
    solution = {}
    for i, val in enumerate(solution_vector):
        if np.abs(val) > 1e-5:  # Filter significant components
            solution[f'|{i:0{n_quplets}b}>'] = norm * val
            
    return solution

if __name__ == "__main__":
    # Example usage for 3-qubit system (8x8 matrix)
    solution = solve_poisson_1d(n_qubits=3, h=0.25)
    print("Quantum Solution Components:")
    for state, amplitude in solution.items():
        print(f"{state}: {amplitude:.4f}")
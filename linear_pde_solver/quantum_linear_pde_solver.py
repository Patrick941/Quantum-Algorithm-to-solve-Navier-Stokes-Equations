import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
from scipy.sparse import diags
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
import matplotlib.pyplot as plt

class PoissonSolverClass:
    def __init__(self, N=3, K=2, reps=2):
        self.N = N  # Number of discretization points
        self.K = K  # Number of bits to represent each u_i
        self.reps = reps  # Number of QAOA repetitions
        self.h = 1 / N  # Spacing
        self.x = np.linspace(0, 1, N+1)  # Discretized domain
        self.f = np.sin(np.pi * self.x)  # Source term f(x)
        self.A = diags([-1, 2, -1], [-1, 0, 1], shape=(N-1, N-1)).toarray() / self.h**2
        self.b = self.f[1:-1]
        self.qp = QuadraticProgram()
        self._formulate_optimization_problem()

    def _formulate_optimization_problem(self):
        # Add binary variables
        for i in range(self.N-1):
            for k in range(self.K):
                self.qp.binary_var(name=f'x_{i+1}_{k}')

        # Define the objective function: ||A u - b||^2
        # Add quadratic terms
        for i in range(self.N-1):
            for j in range(self.N-1):
                for k1 in range(self.K):
                    for k2 in range(self.K):
                        quadratic_coeff = (self.A.T @ self.A)[i, j] * (2**k1) * (2**k2)
                        self.qp.minimize(quadratic={(f'x_{i+1}_{k1}', f'x_{j+1}_{k2}'): quadratic_coeff})

        # Add linear terms
        for i in range(self.N-1):
            for k in range(self.K):
                linear_coeff = -2 * (self.b @ self.A)[i] * (2**k)
                self.qp.minimize(linear={f'x_{i+1}_{k}': linear_coeff})

        # Add linear terms
        for i in range(self.N-1):
            for k in range(self.K):
                self.qp.minimize(linear={f'x_{i+1}_{k}': -2 * self.b[i] * (2**k)})

    def solve_with_qaoa(self):
        sampler = Sampler()
        optimizer = COBYLA()
        qaoa = QAOA(sampler, optimizer, reps=self.reps)
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)

        # Solve the problem using QAOA
        qaoa_result = qaoa_optimizer.solve(self.qp)

        # Reconstruct the continuous solution
        u_qaoa = np.zeros(self.N-1)
        for i in range(self.N-1):
            for k in range(self.K):
                u_qaoa[i] += qaoa_result.x[i * self.K + k] * (2**k)

        return u_qaoa, qaoa_result.fval

    def solve_classically(self):
        classical_solver = NumPyMinimumEigensolver()
        classical_optimizer = MinimumEigenOptimizer(classical_solver)

        # Solve the problem using the classical solver
        classical_result = classical_optimizer.solve(self.qp)

        # Reconstruct the continuous solution
        u_classical = np.zeros(self.N-1)
        for i in range(self.N-1):
            for k in range(self.K):
                u_classical[i] += classical_result.x[i * self.K + k] * (2**k)

        return u_classical, classical_result.fval

    def compare_solutions(self):
        u_qaoa, qaoa_fval = self.solve_with_qaoa()
        u_classical, classical_fval = self.solve_classically()

        print("\nComparison:")
        print(f"QAOA found solution {u_qaoa} with value {qaoa_fval}")
        print(f"Classical solver found solution {u_classical} with value {classical_fval}")
        if np.allclose(u_qaoa, u_classical, atol=1e-3):
            print("QAOA found the correct optimal solution!")
        else:
            print("QAOA did not find the correct optimal solution.")
    
    def get_results(self):
        u_qaoa, qaoa_fval = self.solve_with_qaoa()
        u_classical, classical_fval = self.solve_classically()

        results = {
            "QAOA_solution": u_qaoa,
            "QAOA_value": qaoa_fval,
            "Classical_solution": u_classical,
            "Classical_value": classical_fval,
            "Difference": np.abs(u_qaoa - u_classical),
        }
        return results

# Example usage
if __name__ == "__main__":
    solver = PoissonSolverClass(N=3, K=2, reps=2)
    solver.compare_solutions()
    
    results = solver.get_results()

    # Plot the solutions
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x[1:-1], results["QAOA_solution"], 'o-', label="QAOA Solution")
    plt.plot(solver.x[1:-1], results["Classical_solution"], 's-', label="Classical Solution")
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title("Comparison of QAOA and Classical Solutions")
    plt.legend()
    plt.grid()
    plt.savefig("Images/qaoa_vs_classical_solution.png")

    # Plot the difference
    plt.figure(figsize=(10, 6))
    plt.plot(solver.x[1:-1], results["Difference"], 'x-', label="Difference (|QAOA - Classical|)")
    plt.xlabel("x")
    plt.ylabel("Difference")
    plt.title("Difference Between QAOA and Classical Solutions")
    plt.legend()
    plt.grid()
    plt.savefig("Images/qaoa_vs_classical_difference.png")
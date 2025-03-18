from qiskit_algorithms.minimum_eigensolvers import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
import numpy as np

class QAOASolver:
    def __init__(self, linear_terms, quadratic_terms, reps=2, optimizer=COBYLA(), num_bits=2):
        """
        Initialize the QAOA solver.

        Args:
            linear_terms (dict): Linear terms in the objective function (e.g., {'x1': -1, 'x2': -1}).
            quadratic_terms (dict): Quadratic terms in the objective function (e.g., {('x1', 'x2'): 2}).
            reps (int): Number of QAOA layers.
            optimizer: Classical optimizer for QAOA (e.g., COBYLA).
            num_bits (int): Number of bits to represent each continuous variable.
        """
        self.linear_terms = linear_terms
        self.quadratic_terms = quadratic_terms
        self.reps = reps
        self.optimizer = optimizer
        self.num_bits = num_bits
        self.sampler = Sampler()

    def define_problem(self):
        """Define the Quadratic Program (QP) with binary variables."""
        self.qp = QuadraticProgram()

        # Add binary variables for each continuous variable
        for var in self.linear_terms.keys():
            for k in range(self.num_bits):
                self.qp.binary_var(name=f'{var}_{k}')

        # Define the objective function using binary encoding
        for i, var1 in enumerate(self.linear_terms.keys()):
            for j, var2 in enumerate(self.linear_terms.keys()):
                for k1 in range(self.num_bits):
                    for k2 in range(self.num_bits):
                        coeff = self.quadratic_terms.get((var1, var2), 0) * (2**k1) * (2**k2)
                        if coeff != 0:
                            self.qp.minimize(quadratic={(f'{var1}_{k1}', f'{var2}_{k2}'): coeff})

        for var in self.linear_terms.keys():
            for k in range(self.num_bits):
                coeff = self.linear_terms[var] * (2**k)
                if coeff != 0:
                    self.qp.minimize(linear={f'{var}_{k}': coeff})

        return self.qp

    def solve_with_qaoa(self):
        """Solve the problem using QAOA."""
        # Convert the QP to an operator (Hamiltonian)
        operator, _ = self.qp.to_ising()

        # Set up the QAOA solver
        qaoa = QAOA(self.sampler, self.optimizer, reps=self.reps)

        # Solve the problem using QAOA
        qaoa_result = qaoa.compute_minimum_eigenvalue(operator)

        # Convert the QAOA result back to the optimization problem's solution
        qaoa_optimizer = MinimumEigenOptimizer(qaoa)
        self.qaoa_solution = qaoa_optimizer.solve(self.qp)
        return self.qaoa_solution

    def solve_classically(self):
        """Solve the problem using a classical solver."""
        classical_solver = NumPyMinimumEigensolver()
        classical_optimizer = MinimumEigenOptimizer(classical_solver)
        self.classical_solution = classical_optimizer.solve(self.qp)
        return self.classical_solution

    def reconstruct_continuous_solution(self, binary_solution):
        """Reconstruct the continuous solution from binary variables."""
        continuous_solution = {}
        for var in self.linear_terms.keys():
            value = 0
            for k in range(self.num_bits):
                value += binary_solution[f'{var}_{k}'] * (2**k)
            continuous_solution[var] = value
        return continuous_solution

    def compare_results(self):
        """Compare QAOA results with classical results."""
        # Reconstruct continuous solutions
        qaoa_continuous = self.reconstruct_continuous_solution(dict(zip(self.qp.variables, self.qaoa_solution.x)))
        classical_continuous = self.reconstruct_continuous_solution(dict(zip(self.qp.variables, self.classical_solution.x)))

        print("\nComparison:")
        print(f"QAOA found solution {qaoa_continuous} with value {self.qaoa_solution.fval}")
        print(f"Classical solver found solution {classical_continuous} with value {self.classical_solution.fval}")
        if np.allclose(list(qaoa_continuous.values()), list(classical_continuous.values()), atol=1e-3):
            print("QAOA found the correct optimal solution!")
        else:
            print("QAOA did not find the correct optimal solution.")

if __name__ == "__main__":
    # Define the linear and quadratic terms for the objective function
    linear_terms = {'x1': -1, 'x2': -1}
    quadratic_terms = {('x1', 'x2'): 2}

    # Initialize the QAOA solver
    solver = QAOASolver(linear_terms, quadratic_terms, reps=2, optimizer=COBYLA(), num_bits=2)

    # Define the problem
    solver.define_problem()

    # Solve the problem using QAOA
    qaoa_solution = solver.solve_with_qaoa()
    print("QAOA Solution:")
    print(qaoa_solution)

    # Solve the problem classically
    classical_solution = solver.solve_classically()
    print("Classical Solution:")
    print(classical_solution)

    # Compare the results
    solver.compare_results()
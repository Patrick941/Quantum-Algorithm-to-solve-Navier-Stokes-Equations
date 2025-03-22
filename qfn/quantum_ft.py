import warnings
import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
from qiskit.algorithms import VarQITE, TimeEvolutionProblem, SciPyImaginaryEvolver
from qiskit.primitives import Estimator

warnings.filterwarnings("ignore", category=DeprecationWarning)

# =============================================
# 1. Define the Poisson Equation Hamiltonian
# =============================================
# Discretize ∇²u = -f on a 4-point grid (2 qubits) with f(x) = 1
# Laplacian matrix (periodic BCs for simplicity):
# H = -∇² + f(x)
# Encoded as Pauli terms (simplified for demonstration)
hamiltonian = SparsePauliOp(
    ["II", "IZ", "ZI", "ZZ", "XX"], 
    coeffs=[2.25, -0.25, -0.25, -0.25, -0.25]
)
# Note: This Hamiltonian is derived from a 4-point discretized Laplacian + source term f=1.

# =============================================
# 2. Prepare the Ansatz Circuit
# =============================================
num_qubits = 2
ansatz = EfficientSU2(num_qubits, reps=1)
circuit = ansatz.decompose()
circuit.draw("mpl", style="iqp")
plt.title("Ansatz for Poisson Equation")
plt.savefig("images/ansatz_poisson.png", dpi=300, bbox_inches="tight")

# =============================================
# 3. Initialize Parameters
# =============================================
init_param_values = {param: np.pi / 2 for param in ansatz.parameters}

# =============================================
# 4. Set Up VarQITE
# =============================================
var_principle = ImaginaryMcLachlanPrinciple()
time = 5.0
evolution_problem = TimeEvolutionProblem(hamiltonian, time, aux_operators=[hamiltonian])

var_qite = VarQITE(ansatz, init_param_values, var_principle, Estimator())
evolution_result = var_qite.evolve(evolution_problem)

# =============================================
# 5. Exact Classical Solution
# =============================================
# Solve Hψ = Eψ classically
init_state = Statevector(ansatz.assign_parameters(init_param_values))
exact_evol = SciPyImaginaryEvolver(num_timesteps=501)
exact_solution = exact_evol.evolve(TimeEvolutionProblem(hamiltonian, time, initial_state=init_state, aux_operators=[hamiltonian]))

# =============================================
# 6. Plot Results
# =============================================
plt.figure(figsize=(10, 6))
times = evolution_result.times
h_exp_val = np.array([ele[0][0] for ele in evolution_result.observables])
exact_h_exp_val = exact_solution.observables[0][0].real

plt.plot(times, h_exp_val, label="VarQITE")
plt.plot(times, exact_h_exp_val, "--", label="Exact")
plt.xlabel("Imaginary Time (τ)")
plt.ylabel("Energy (Hamiltonian Expectation)")
plt.title("Poisson Equation: VarQITE vs Exact Imaginary Time Evolution")
plt.legend()
plt.grid(True)
plt.savefig("images/poisson_varqite.png", dpi=300, bbox_inches="tight")


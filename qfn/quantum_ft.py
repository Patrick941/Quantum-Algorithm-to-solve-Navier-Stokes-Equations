import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp

# Create Hamiltonians
hamiltonian = SparsePauliOp(['ZZ', 'IX', 'XI'], coeffs=[-0.2, -1, -1])
magnetization = SparsePauliOp(['IZ', 'ZI'], coeffs=[1, 1])

# Create and draw ansatz circuit
from qiskit.circuit.library import EfficientSU2
ansatz = EfficientSU2(hamiltonian.num_qubits, reps=1)
circuit = ansatz.decompose()
circuit.draw('mpl', style='iqp')
plt.title('Ansatz Circuit Diagram')
plt.savefig("images/ansatz.png", dpi=300, bbox_inches='tight')

# Initialize parameters
import numpy as np
init_param_values = {}
for i in range(len(ansatz.parameters)):
    init_param_values[ansatz.parameters[i]] = np.pi / 2

# Set up imaginary time evolution
from qiskit.algorithms.time_evolvers.variational import ImaginaryMcLachlanPrinciple
var_principle = ImaginaryMcLachlanPrinciple()

from qiskit.algorithms import TimeEvolutionProblem
time = 5.0
aux_ops = [hamiltonian]
evolution_problem = TimeEvolutionProblem(hamiltonian, time, aux_operators=aux_ops)

# Run VarQITE
from qiskit.algorithms import VarQITE
from qiskit.primitives import Estimator
var_qite = VarQITE(ansatz, init_param_values, var_principle, Estimator())
evolution_result = var_qite.evolve(evolution_problem)

# Prepare exact evolution comparison
from qiskit.quantum_info import Statevector
from qiskit.algorithms import SciPyImaginaryEvolver
init_state = Statevector(ansatz.assign_parameters(init_param_values))
evolution_problem = TimeEvolutionProblem(hamiltonian, time, initial_state=init_state, aux_operators=aux_ops)
exact_evol = SciPyImaginaryEvolver(num_timesteps=501)
sol = exact_evol.evolve(evolution_problem)

# Plot energy comparison
plt.figure(figsize=(10, 6))
h_exp_val = np.array([ele[0][0] for ele in evolution_result.observables])
exact_h_exp_val = sol.observables[0][0].real
times = evolution_result.times

plt.plot(times, h_exp_val, label="VarQITE")
plt.plot(times, exact_h_exp_val, label="Exact Solution", linestyle='--')
plt.xlabel("Imaginary Time (τ)", fontsize=12)
plt.ylabel("Energy", fontsize=12)
plt.title("Energy Evolution Comparison", fontsize=14)
plt.legend()
plt.grid(True)
plt.savefig("images/VARQITE.png", dpi=300, bbox_inches='tight')
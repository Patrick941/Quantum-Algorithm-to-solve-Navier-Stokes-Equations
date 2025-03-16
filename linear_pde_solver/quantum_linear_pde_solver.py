import numpy as np
from qiskit import Aer, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp, CircuitStateFn, StateFn

# 1. Discretize the 1D Poisson equation d²u/dx² = f(x) with boundary conditions u(0)=u(1)=0
# Using finite differences, we get the linear system Au = b
A = np.array([[2, -1], 
              [-1, 2]], dtype=float)
b = np.array([1, 1], dtype=float)
b_norm = np.linalg.norm(b)
b_normalized = b / b_norm  # Normalize b for state preparation

# 2. Encode matrix A as a PauliSumOp (Decompose A into Pauli matrices)
# A = 2I - X (for 2x2 case)
pauli_A = PauliSumOp.from_list([("I", 2.0), ("X", -1.0)])

# 3. Quantum circuit to prepare the state |b>
qc_b = QuantumCircuit(1)  # Using 1 qubit for 2-dimensional vector
qc_b.initialize(b_normalized, 0)

# 4. Define the variational ansatz for the solution |u(θ)〉
ansatz = RealAmplitudes(1, reps=1)  # 1 qubit, 2 parameters

# 5. Define the cost function: 〈u(θ)|A|u(θ)〉 - 2〈u(θ)|b〉
def cost_function(params):
    # Prepare |u(θ)〉
    u_circ = ansatz.assign_parameters(params)
    u_state = CircuitStateFn(u_circ)
    
    # Compute 〈u|A|u〉
    Au = pauli_A @ u_state
    expectation_A = StateFn(Au, is_measurement=True).adjoint() @ u_state
    value_A = expectation_A.eval().real
    
    # Compute 〈u|b〉
    b_state = CircuitStateFn(qc_b)
    overlap = StateFn(b_state, is_measurement=True).adjoint() @ u_state
    value_b = overlap.eval().real
    
    # Calculate cost: 〈A〉 - 2*Re(〈b|u〉) * ||b||
    return value_A - 2 * value_b * b_norm

# 6. Configure and run VQE
optimizer = COBYLA(maxiter=100)
quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))

vqe = VQE(ansatz, optimizer, quantum_instance=quantum_instance)
result = vqe.compute_minimum_eigenvalue(operator=None, cost_fn=cost_function)

# 7. Retrieve and process results
optimal_params = result.optimal_parameters
print("Optimal parameters:", optimal_params)

# Get the solution statevector
u_circ = ansatz.assign_parameters(optimal_params)
backend = Aer.get_backend('statevector_simulator')
statevector = backend.run(u_circ).result().get_statevector()

# Scale the solution by the norm of b and extract real components
solution = statevector.data.real * b_norm
print("Numerical solution:", solution)
print("Exact solution:", np.linalg.solve(A, b))
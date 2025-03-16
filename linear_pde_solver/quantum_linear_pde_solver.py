from linear_pde_solver import LinearPDESolver
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator
from scipy.optimize import minimize

class QuantumLinearPDESolver:
    def __init__(self, N, max_iter=100):
        self.N = N                    # Number of grid points
        self.max_iter = max_iter      # Optimization steps
        self.A, self.b = self.discretize()  # Discretized system

    def discretize(self):
        # Discretization using a simple finite difference scheme
        A = 2 * np.eye(self.N) - np.eye(self.N, k=1) - np.eye(self.N, k=-1)
        b = np.ones(self.N)
        return A, b

    def construct_ansatz(self):
        # Two-qubit parameterized ansatz for N=2
        theta1 = Parameter('θ1')
        theta2 = Parameter('θ2')
        qc = QuantumCircuit(2)
        qc.ry(theta1, 0)  # Rotation on qubit 0
        qc.ry(theta2, 1)  # Rotation on qubit 1
        qc.cx(0, 1)       # Entanglement between qubits
        qc.save_statevector()
        return qc

    def encode_b(self):
        # Encode |b> as |+>|+> state (H|0> H|0>)
        qc = QuantumCircuit(2)
        qc.h(0)  # Apply Hadamard gate to qubit 0
        qc.h(1)  # Apply Hadamard gate to qubit 1
        qc.save_statevector()
        return qc

    def cost_function(self, theta_values):
        sim = AerSimulator(method="statevector")
        
        # Construct the ansatz and bind parameters
        ansatz = self.construct_ansatz()
        param_dict = {param: theta_values[i] for i, param in enumerate(ansatz.parameters)}
        ansatz.assign_parameters(param_dict, inplace=True)
        
        # Compute |x(θ)>
        job_x = sim.run(ansatz)
        x_state = np.asarray(job_x.result().get_statevector(ansatz))
        
        # Compute A|x(θ)>
        Ax = self.A @ x_state

        # Compute |b>
        b_circuit = self.encode_b()
        job_b = sim.run(b_circuit)
        b_state = np.asarray(job_b.result().get_statevector(b_circuit))

        # Compute the squared norm ||Ax - b||^2
        cost = np.linalg.norm(Ax - b_state) ** 2
        return cost

    def solve(self):
        # Initial guess for parameters
        initial_theta = np.array([0.1, 0.1])  # Two parameters for the two-qubit ansatz
        
        # Optimize using a classical optimizer
        result = minimize(self.cost_function, initial_theta, method='COBYLA')
        
        # Get the optimized parameters
        optimized_theta = result.x
        
        # Get final solution |x(θ)> using the optimized theta
        ansatz = self.construct_ansatz()
        param_dict = {param: optimized_theta[i] for i, param in enumerate(ansatz.parameters)}
        ansatz.assign_parameters(param_dict, inplace=True)
        sim = AerSimulator(method="statevector")
        job_final = sim.run(ansatz)
        x_quantum = np.asarray(job_final.result().get_statevector(ansatz))
        return x_quantum

if __name__ == "__main__":
    classical_solver = LinearPDESolver(N=2)
    x_classical = classical_solver.solve()
    print("Classical solution:", x_classical)
    
    quantum_solver = QuantumLinearPDESolver(N=4)
    x_quantum = quantum_solver.solve()
    print("Quantum solution:", x_quantum)

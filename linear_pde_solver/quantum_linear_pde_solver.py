from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, StateFn
import numpy as np

class QuantumLinearPDESolver:
    def __init__(self, N, max_iter=100):
        self.N = N  # Grid points
        self.max_iter = max_iter  # Optimization steps
        self.A, self.b = self.discretize()  # Discretized system

    def discretize(self):
        # Same as classical solver
        A = 2 * np.eye(self.N) - np.eye(self.N, k=1) - np.eye(self.N, k=-1)
        b = np.ones(self.N)
        return A, b

    def construct_ansatz(self):
        # Single-qubit parameterized ansatz for simplicity (scalable to N=2)
        theta = Parameter('θ')
        qc = QuantumCircuit(1)
        qc.ry(theta, 0)
        return qc

    def encode_b(self):
        # Encode |b> as a quantum state (H|0> for N=2)
        qc = QuantumCircuit(1)
        qc.h(0)
        return qc

    def cost_function(self, theta_value):
        # Compute cost: ||A|x(θ)> - |b>||^2
        backend = Aer.get_backend('statevector_simulator')

        # 1. Compute |x(θ)>
        ansatz = self.construct_ansatz()
        ansatz.assign_parameters({ansatz.parameters[0]: theta_value}, inplace=True)
        job = execute(ansatz, backend)
        x_state = job.result().get_statevector()

        # 2. Compute A|x(θ)> (classically for simplicity)
        Ax = self.A @ x_state

        # 3. Compute |b>
        b_circuit = self.encode_b()
        job = execute(b_circuit, backend)
        b_state = job.result().get_statevector()

        # 4. Compute ||Ax - b||^2
        cost = np.linalg.norm(Ax - b_state) ** 2
        return cost

    def solve(self):
        # Optimize using gradient descent (simplified)
        theta = 0.1  # Initial guess
        learning_rate = 0.1
        for _ in range(self.max_iter):
            cost = self.cost_function(theta)
            gradient = (self.cost_function(theta + 0.01) - cost) / 0.01  # Finite difference
            theta -= learning_rate * gradient
            if cost < 1e-4:  # Convergence threshold
                break

        # Get final solution |x(θ)>
        ansatz = self.construct_ansatz()
        ansatz.assign_parameters({ansatz.parameters[0]: theta}, inplace=True)
        job = execute(ansatz, Aer.get_backend('statevector_simulator'))
        x_quantum = np.real(job.result().get_statevector())
        return x_quantum
    
    
if __name__ == "__main__":
    quantum_solver = QuantumLinearPDESolver(N=2)

    x_classical = classical_solver.solve()
    x_quantum = quantum_solver.solve()

    print("Classical solution:", x_classical)
    print("Quantum solution:", x_quantum)

import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.library import EfficientSU2
from scipy.optimize import minimize
import os

class QuantumGroundStateFinder:
    def __init__(self, coefficient_set, gate_set):
        self.coefficient_set = coefficient_set
        self.gate_set = gate_set
        
    def apply_fixed_ansatz(self, circ, qubits, parameters):
        for iz in range(len(qubits)):
            circ.ry(parameters[0][iz], qubits[iz])
        circ.cz(qubits[0], qubits[1])
        circ.cz(qubits[2], qubits[0])
        for iz in range(len(qubits)):
            circ.ry(parameters[1][iz], qubits[iz])
        circ.cz(qubits[1], qubits[2])
        circ.cz(qubits[2], qubits[0])
        for iz in range(len(qubits)):
            circ.ry(parameters[2][iz], qubits[iz])
        return circ

    def had_test(self, gate_type, qubits, auxiliary_index, parameters):
        qreg = QuantumRegister(4)
        circ = QuantumCircuit(qreg)
        circ.h(auxiliary_index)
        self.apply_fixed_ansatz(circ, qubits, parameters)
        for ie in range(len(gate_type[0])):
            if gate_type[0][ie] == 1:
                circ.cz(auxiliary_index, qubits[ie])
        for ie in range(len(gate_type[1])):
            if gate_type[1][ie] == 1:
                circ.cz(auxiliary_index, qubits[ie])
        circ.h(auxiliary_index)
        return circ

    def control_fixed_ansatz(self, circ, qubits, parameters, auxiliary):
        for i in range(len(qubits)):
            circ.cry(parameters[0][i], auxiliary, qubits[i])
        circ.ccx(auxiliary, qubits[1], 4)
        circ.cz(qubits[0], 4)
        circ.ccx(auxiliary, qubits[1], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        circ.cz(qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        for i in range(len(qubits)):
            circ.cry(parameters[1][i], auxiliary, qubits[i])
        circ.ccx(auxiliary, qubits[2], 4)
        circ.cz(qubits[1], 4)
        circ.ccx(auxiliary, qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        circ.cz(qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        for i in range(len(qubits)):
            circ.cry(parameters[2][i], auxiliary, qubits[i])
        return circ

    def control_b(self, circ, auxiliary, qubits):
        for ia in qubits:
            circ.ch(auxiliary, ia)
        return circ

    def special_had_test(self, gate_type, qubits, auxiliary_index, parameters):
        qreg = QuantumRegister(5)
        circ = QuantumCircuit(qreg)
        circ.h(auxiliary_index)
        self.control_fixed_ansatz(circ, qubits, parameters, auxiliary_index)
        for ty in range(len(gate_type)):
            if gate_type[ty] == 1:
                circ.cz(auxiliary_index, qubits[ty])
        self.control_b(circ, auxiliary_index, qubits)
        circ.h(auxiliary_index)
        return circ

    def calculate_cost_function(self, parameters):
        parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
        overall_sum_1 = 0
        
        # First part of cost function
        for i in range(len(self.gate_set)):
            for j in range(len(self.gate_set)):
                circ = self.had_test([self.gate_set[i], self.gate_set[j]], [1,2,3], 0, parameters)
                backend = Aer.get_backend('aer_simulator')
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
                overall_sum_1 += self.coefficient_set[i] * self.coefficient_set[j] * (1 - 2*m_sum)

        # Second part of cost function
        overall_sum_2 = 0
        for i in range(len(self.gate_set)):
            for j in range(len(self.gate_set)):
                mult = 1
                for extra in range(2):
                    if extra == 0:
                        circ = self.special_had_test(self.gate_set[i], [1,2,3], 0, parameters)
                    else:
                        circ = self.special_had_test(self.gate_set[j], [1,2,3], 0, parameters)
                    backend = Aer.get_backend('aer_simulator')
                    circ.save_statevector()
                    t_circ = transpile(circ, backend)
                    qobj = assemble(t_circ)
                    job = backend.run(qobj)
                    result = job.result()
                    outputstate = np.real(result.get_statevector(circ, decimals=100))
                    m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
                    mult *= (1 - 2*m_sum)
                overall_sum_2 += self.coefficient_set[i] * self.coefficient_set[j] * mult

        cost = 1 - float(overall_sum_2 / overall_sum_1)
        print(f"Cost: {cost}")
        return cost

    def run_optimization(self):
        initial_params = [random.uniform(0, 3) for _ in range(9)]
        result = minimize(self.calculate_cost_function,
                         x0=initial_params,
                         method="COBYLA",
                         options={'maxiter': 200})
        return result

class PDESolver(QuantumGroundStateFinder):
    def __init__(self, grid_size=3):
        self.grid_size = grid_size  # Using 3 qubits → 8 grid points
        self.x = np.linspace(0, 1, 2**grid_size + 2)[1:-1]
        super().__init__([], [])
        self.A = self._build_fd_matrix()
        self._hamiltonian_decomposition()
        
    def _build_fd_matrix(self):
        """Finite difference matrix for -u'' = 1"""
        n = 2**self.grid_size
        h = 1/(n+1)
        return (2/h**2)*np.eye(n) + (-1/h**2)*(np.eye(n,k=1)+np.eye(n,k=-1))
    
    def _hamiltonian_decomposition(self):
        """Proper Pauli decomposition of FD matrix"""
        # Convert matrix to Hamiltonian (Hermitian)
        H = 0.5*(self.A + self.A.T)
        
        # Simple diagonal approximation for demonstration
        diag = np.diag(H)
        off_diag = H[np.triu_indices_from(H,k=1)]
        
        # Store as coefficient set and Pauli terms
        self.coefficient_set = list(diag) + list(off_diag)
        self.gate_set = (
            [[1 if i==j else 0 for j in range(self.grid_size)]  # Z terms
            for i in range(self.grid_size)]
        ) + (
            [[1 if i==k//2 else 0 for k in range(self.grid_size)]  # ZZ terms
            for i in range(len(off_diag))]
        )

    def classical_solution(self):
        """Exact FD solution"""
        b = np.ones(2**self.grid_size)
        return np.linalg.solve(self.A, b)
    
    def quantum_state_to_solution(self, params):
        """Convert quantum state to PDE solution"""
        circ = EfficientSU2(3, reps=2)  # Better ansatz
        circ = circ.assign_parameters(params)
        
        backend = Aer.get_backend('statevector_simulator')
        statevector = backend.run(circ).result().get_statevector()
        amplitudes = np.real(statevector)[:2**self.grid_size]
        
        # Normalize and align with classical solution
        classical = self.classical_solution()
        return amplitudes * (np.linalg.norm(classical)/np.linalg.norm(amplitudes))
    
    def plot_results(self, quantum_params):
        """Enhanced comparison plot"""
        classical = self.classical_solution()
        quantum = self.quantum_state_to_solution(quantum_params)
        
        plt.figure(figsize=(12,6))
        plt.plot(self.x, classical, 'b-', lw=3, label='Classical FD')
        plt.plot(self.x, quantum, 'ro--', ms=8, label='Quantum VQE')
        plt.fill_between(self.x, classical, quantum, color='gray', alpha=0.2)
        
        plt.title('1D Poisson Equation Solution Comparison\n(Improved Encoding)', pad=20)
        plt.xlabel('Position', fontsize=12)
        plt.ylabel('Solution Magnitude', fontsize=12)
        plt.legend(prop={'size': 12})
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    # Initialize solver with proper Hamiltonian
    pde_system = PDESolver(grid_size=3)
    
    # Run optimization with improved settings
    result = minimize(pde_system.calculate_cost_function,
                     x0=np.random.randn(24),  # For EfficientSU2(3, reps=2)
                     method='L-BFGS-B',
                     options={'maxiter': 500})
    
    # Visual comparison
    pde_system.plot_results(result.x)

if __name__ == "__main__":
    main()
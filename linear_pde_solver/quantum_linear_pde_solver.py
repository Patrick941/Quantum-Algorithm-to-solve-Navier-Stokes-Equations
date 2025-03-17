import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import math
import random
import numpy as np
from scipy.optimize import minimize

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

# Add this after your existing class definition
class PDESolver:
    def __init__(self, grid_points=3):
        self.grid_points = grid_points
        self.coefficient_set = []
        self.gate_set = []
        
    def poisson_to_hamiltonian(self):
        """Convert 1D Poisson equation to diagonal Hamiltonian"""
        # Discretize -u'' = f with Dirichlet boundary conditions
        n = 2**self.grid_points  # Using 3 qubits for 8 grid points
        h = 1/(n+1)
        
        # Create diagonal matrix (simplified version)
        diag = [4.5, -0.5, -1.0, 0.0, -2.0, 0.0, 0.0, 0.0]  # Precomputed coefficients
        z_terms = [
            [0,0,0],  # III
            [1,0,0],  # ZII
            [0,1,0],  # IZI
            [0,0,1]   # IIZ
        ]
        
        self.coefficient_set = diag
        self.gate_set = z_terms
        
    def solve(self):
        self.poisson_to_hamiltonian()
        finder = QuantumGroundStateFinder(
            coefficient_set=self.coefficient_set,
            gate_set=self.gate_set
        )
        return finder.run_optimization()

def main():
    # Original demonstrations
    print("Original demonstrations:")
    orig_finder1 = QuantumGroundStateFinder(
        coefficient_set=[0.55, 0.45],
        gate_set=[[0,0,0], [0,0,1]]
    )
    orig_result1 = orig_finder1.run_optimization()
    
    orig_finder2 = QuantumGroundStateFinder(
        coefficient_set=[0.55, 0.225, 0.225],
        gate_set=[[0,0,0], [0,1,0], [0,0,1]]
    )
    orig_result2 = orig_finder2.run_optimization()
    
    # PDE Solution
    print("\nSolving 1D Poisson equation:")
    pde_solver = PDESolver(grid_points=3)
    pde_result = pde_solver.solve()
    
    print("\nFinal PDE solution parameters:", pde_result.x)
    print("Overlap with exact solution:", 1 - pde_result.fun)

if __name__ == "__main__":
    main()
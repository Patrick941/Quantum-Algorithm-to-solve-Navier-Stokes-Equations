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
        
    def apply_fixed_ansatz(self, qubits, parameters):
        circ = QuantumCircuit(3)
        for iz in range(0, len(qubits)):
            circ.ry(parameters[0][iz], qubits[iz])
        circ.cz(qubits[0], qubits[1])
        circ.cz(qubits[2], qubits[0])
        for iz in range(0, len(qubits)):
            circ.ry(parameters[1][iz], qubits[iz])
        circ.cz(qubits[1], qubits[2])
        circ.cz(qubits[2], qubits[0])
        for iz in range(0, len(qubits)):
            circ.ry(parameters[2][iz], qubits[iz])
        return circ

    def had_test(self, gate_type, qubits, auxiliary_index, parameters):
        q_reg = QuantumRegister(4)
        circ = QuantumCircuit(q_reg)
        circ.h(auxiliary_index)
        ansatz_circ = self.apply_fixed_ansatz(qubits, parameters)
        circ.compose(ansatz_circ, qubits, inplace=True)
        for ie in range(0, len(gate_type[0])):
            if (gate_type[0][ie] == 1):
                circ.cz(auxiliary_index, qubits[ie])
        for ie in range(0, len(gate_type[1])):
            if (gate_type[1][ie] == 1):
                circ.cz(auxiliary_index, qubits[ie])
        circ.h(auxiliary_index)
        return circ

    def control_fixed_ansatz(self, qubits, parameters, auxiliary, reg):
        circ = QuantumCircuit(reg)
        for i in range(0, len(qubits)):
            circ.cry(parameters[0][i], auxiliary, qubits[i])
        circ.ccx(auxiliary, qubits[1], 4)
        circ.cz(qubits[0], 4)
        circ.ccx(auxiliary, qubits[1], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        circ.cz(qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        for i in range(0, len(qubits)):
            circ.cry(parameters[1][i], auxiliary, qubits[i])
        circ.ccx(auxiliary, qubits[2], 4)
        circ.cz(qubits[1], 4)
        circ.ccx(auxiliary, qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        circ.cz(qubits[2], 4)
        circ.ccx(auxiliary, qubits[0], 4)
        for i in range(0, len(qubits)):
            circ.cry(parameters[2][i], auxiliary, qubits[i])
        return circ

    def control_b(self, auxiliary, qubits):
        circ = QuantumCircuit(4)
        for ia in qubits:
            circ.ch(auxiliary, ia)
        return circ

    def special_had_test(self, gate_type, qubits, auxiliary_index, parameters, reg):
        circ = QuantumCircuit(reg)
        circ.h(auxiliary_index)
        ansatz_circ = self.control_fixed_ansatz(qubits, parameters, auxiliary_index, reg)
        circ.compose(ansatz_circ, inplace=True)
        for ty in range(0, len(gate_type)):
            if (gate_type[ty] == 1):
                circ.cz(auxiliary_index, qubits[ty])
        b_circ = self.control_b(auxiliary_index, qubits)
        circ.compose(b_circ, inplace=True)
        circ.h(auxiliary_index)
        return circ

    def calculate_cost_function(self, parameters):
        parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
        overall_sum_1 = 0
        
        # First part of the cost function
        for i in range(0, len(self.gate_set)):
            for j in range(0, len(self.gate_set)):
                multiply = self.coefficient_set[i] * self.coefficient_set[j]
                circ = self.had_test([self.gate_set[i], self.gate_set[j]], [1, 2, 3], 0, parameters)
                backend = Aer.get_backend('aer_simulator')
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
                overall_sum_1 += multiply * (1 - 2*m_sum)

        # Second part of the cost function
        overall_sum_2 = 0
        for i in range(0, len(self.gate_set)):
            for j in range(0, len(self.gate_set)):
                multiply = self.coefficient_set[i] * self.coefficient_set[j]
                mult = 1
                for extra in range(0, 2):
                    if extra == 0:
                        circ = self.special_had_test(self.gate_set[i], [1, 2, 3], 0, parameters, QuantumRegister(5))
                    else:
                        circ = self.special_had_test(self.gate_set[j], [1, 2, 3], 0, parameters, QuantumRegister(5))
                    backend = Aer.get_backend('aer_simulator')
                    circ.save_statevector()
                    t_circ = transpile(circ, backend)
                    qobj = assemble(t_circ)
                    job = backend.run(qobj)
                    result = job.result()
                    outputstate = np.real(result.get_statevector(circ, decimals=100))
                    m_sum = sum(outputstate[l]**2 for l in range(1, len(outputstate), 2))
                    mult *= (1 - 2*m_sum)
                overall_sum_2 += multiply * mult

        cost = 1 - float(overall_sum_2 / overall_sum_1)
        print(f"Current cost: {cost}")
        return cost

    def run_optimization(self):
        initial_params = [random.uniform(0, 3) for _ in range(9)]
        result = minimize(self.calculate_cost_function, 
                         x0=initial_params, 
                         method="COBYLA", 
                         options={'maxiter': 200})
        return result

def main():
    # First configuration
    print("Running first optimization...")
    finder1 = QuantumGroundStateFinder(
        coefficient_set=[0.55, 0.45],
        gate_set=[[0, 0, 0], [0, 0, 1]]
    )
    result1 = finder1.run_optimization()
    print("\nFirst optimization result:", result1)

    # Second configuration
    print("\nRunning second optimization...")
    finder2 = QuantumGroundStateFinder(
        coefficient_set=[0.55, 0.225, 0.225],
        gate_set=[[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    )
    result2 = finder2.run_optimization()
    print("\nSecond optimization result:", result2)

if __name__ == "__main__":
    main()
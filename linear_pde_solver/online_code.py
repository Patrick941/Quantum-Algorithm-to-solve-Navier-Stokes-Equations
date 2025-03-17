import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import math
import random
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt  # Import for plotting

# Function to apply a fixed ansatz
def apply_fixed_ansatz(qubits, parameters):
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

# Function for the Hadamard test
def had_test(gate_type, qubits, auxiliary_index, parameters):
    circ.h(auxiliary_index)
    apply_fixed_ansatz(qubits, parameters)
    for ie in range(0, len(gate_type[0])):
        if gate_type[0][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    for ie in range(0, len(gate_type[1])):
        if gate_type[1][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    circ.h(auxiliary_index)

# Function for controlled fixed ansatz
def control_fixed_ansatz(qubits, parameters, auxiliary, reg):
    for i in range(0, len(qubits)):
        circ.cry(parameters[0][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))
    circ.ccx(auxiliary, qubits[1], 4)
    circ.cz(qubits[0], 4)
    circ.ccx(auxiliary, qubits[1], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    for i in range(0, len(qubits)):
        circ.cry(parameters[1][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))
    circ.ccx(auxiliary, qubits[2], 4)
    circ.cz(qubits[1], 4)
    circ.ccx(auxiliary, qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    circ.cz(qubits[2], 4)
    circ.ccx(auxiliary, qubits[0], 4)
    for i in range(0, len(qubits)):
        circ.cry(parameters[2][i], qiskit.circuit.Qubit(reg, auxiliary), qiskit.circuit.Qubit(reg, qubits[i]))

# Function for controlled b
def control_b(auxiliary, qubits):
    for ia in qubits:
        circ.ch(auxiliary, ia)

# Function for special Hadamard test
def special_had_test(gate_type, qubits, auxiliary_index, parameters, reg):
    circ.h(auxiliary_index)
    control_fixed_ansatz(qubits, parameters, auxiliary_index, reg)
    for ty in range(0, len(gate_type)):
        if gate_type[ty] == 1:
            circ.cz(auxiliary_index, qubits[ty])
    control_b(auxiliary_index, qubits)
    circ.h(auxiliary_index)

# Cost function calculation
def calculate_cost_function(parameters):
    global opt, cost_values
    overall_sum_1 = 0
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            qctl = QuantumRegister(5)
            qc = ClassicalRegister(5)
            circ = QuantumCircuit(qctl, qc)
            backend = Aer.get_backend('aer_simulator')
            multiply = coefficient_set[i] * coefficient_set[j]
            had_test([gate_set[i], gate_set[j]], [1, 2, 3], 0, parameters)
            circ.save_statevector()
            t_circ = transpile(circ, backend)
            qobj = assemble(t_circ)
            job = backend.run(qobj)
            result = job.result()
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            o = outputstate
            m_sum = 0
            for l in range(0, len(o)):
                if l % 2 == 1:
                    n = o[l] ** 2
                    m_sum += n
            overall_sum_1 += multiply * (1 - (2 * m_sum))
    overall_sum_2 = 0
    for i in range(0, len(gate_set)):
        for j in range(0, len(gate_set)):
            multiply = coefficient_set[i] * coefficient_set[j]
            mult = 1
            for extra in range(0, 2):
                qctl = QuantumRegister(5)
                qc = ClassicalRegister(5)
                circ = QuantumCircuit(qctl, qc)
                backend = Aer.get_backend('aer_simulator')
                if extra == 0:
                    special_had_test(gate_set[i], [1, 2, 3], 0, parameters, qctl)
                if extra == 1:
                    special_had_test(gate_set[j], [1, 2, 3], 0, parameters, qctl)
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                o = outputstate
                m_sum = 0
                for l in range(0, len(o)):
                    if l % 2 == 1:
                        n = o[l] ** 2
                        m_sum += n
                mult = mult * (1 - (2 * m_sum))
            overall_sum_2 += multiply * mult
    current_cost = 1 - float(overall_sum_2 / overall_sum_1)
    print(current_cost)
    cost_values.append(current_cost)  # Track cost values
    return current_cost

# First optimization
coefficient_set = [0.55, 0.45]
gate_set = [[0, 0, 0], [0, 0, 1]]
cost_values = []  # Track cost values for the first optimization
out = minimize(calculate_cost_function, x0=[float(random.randint(0, 3000)) / 1000 for i in range(0, 9)], method="COBYLA", options={'maxiter': 200})
print(out)
out_f = [out['x'][0:3], out['x'][3:6], out['x'][6:9]]
circ = QuantumCircuit(3, 3)
apply_fixed_ansatz([0, 1, 2], out_f)
circ.save_statevector()
backend = Aer.get_backend('aer_simulator')
t_circ = transpile(circ, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)
result = job.result()
o = result.get_statevector(circ, decimals=10)
a1 = coefficient_set[1] * np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])
a2 = coefficient_set[0] * np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
a3 = np.add(a1, a2)
b = np.array([float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8))])
prob1 = (b.dot(a3.dot(o) / (np.linalg.norm(a3.dot(o))))) ** 2
print("Probability 1:", prob1)

# Second optimization
coefficient_set = [0.55, 0.225, 0.225]
gate_set = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
cost_values = []  # Reset cost values for the second optimization
out = minimize(calculate_cost_function, x0=[float(random.randint(0, 3000)) / 1000 for i in range(0, 9)], method="COBYLA", options={'maxiter': 200})
print(out)
out_f = [out['x'][0:3], out['x'][3:6], out['x'][6:9]]
circ = QuantumCircuit(3, 3)
apply_fixed_ansatz([0, 1, 2], out_f)
circ.save_statevector()
backend = Aer.get_backend('aer_simulator')
t_circ = transpile(circ, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)
result = job.result()
o = result.get_statevector(circ, decimals=10)
a1 = coefficient_set[2] * np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])
a0 = coefficient_set[1] * np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 0, -1]])
a2 = coefficient_set[0] * np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 1]])
a3 = np.add(np.add(a2, a0), a1)
b = np.array([float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8)), float(1 / np.sqrt(8))])
prob2 = (b.dot(a3.dot(o) / (np.linalg.norm(a3.dot(o))))) ** 2
print("Probability 2:", prob2)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(cost_values, label='Cost Function Value')
plt.xlabel('Function Evaluations')
plt.ylabel('Cost Value')
plt.title('Cost Function Convergence')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
plt.bar(['Case 1', 'Case 2'], [prob1, prob2])
plt.ylabel('Success Probability')
plt.title('Final Results Comparison')
plt.show()
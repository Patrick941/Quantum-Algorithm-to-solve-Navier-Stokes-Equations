import qiskit
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit import Aer, transpile, assemble
import numpy as np
import random
from scipy.optimize import minimize

# 1. Ansatz Definition
def apply_fixed_ansatz(qubits, parameters):
    circ = QuantumCircuit(3)
    for iz in range(3):
        circ.ry(parameters[0][iz], qubits[iz])
    
    circ.cz(qubits[0], qubits[1])
    circ.cz(qubits[2], qubits[0])
    
    for iz in range(3):
        circ.ry(parameters[1][iz], qubits[iz])
    
    circ.cz(qubits[1], qubits[2])
    circ.cz(qubits[2], qubits[0])
    
    for iz in range(3):
        circ.ry(parameters[2][iz], qubits[iz])
    return circ

# 2. Hadamard Test Implementation
def had_test(gate_type, qubits, auxiliary_index, parameters):
    circ = QuantumCircuit(4)
    circ.h(auxiliary_index)
    
    ansatz_circ = apply_fixed_ansatz(qubits, parameters)
    circ.compose(ansatz_circ, qubits, inplace=True)
    
    # Apply controlled unitaries
    for ie in range(len(gate_type)):
        if gate_type[ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    
    circ.h(auxiliary_index)
    return circ

# 3. Controlled Operations
def control_fixed_ansatz(qubits, parameters, auxiliary, reg):
    circ = QuantumCircuit(reg)
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

def control_b(auxiliary, qubits):
    circ = QuantumCircuit(4)
    for ia in qubits:
        circ.ch(auxiliary, ia)
    return circ

def special_had_test(gate_type, qubits, auxiliary_index, parameters, reg):
    circ = QuantumCircuit(reg)
    circ.h(auxiliary_index)
    
    controlled_ansatz = control_fixed_ansatz(qubits, parameters, auxiliary_index, reg)
    circ.compose(controlled_ansatz, inplace=True)
    
    for ty in range(len(gate_type)):
        if gate_type[ty] == 1:
            circ.cz(auxiliary_index, qubits[ty])
    
    controlled_b = control_b(auxiliary_index, qubits)
    circ.compose(controlled_b, inplace=True)
    
    circ.h(auxiliary_index)
    return circ

# 4. Cost Function Calculation
def calculate_cost_function(parameters):
    overall_sum_1 = 0
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    
    backend = Aer.get_backend('aer_simulator')
    
    # Calculate Tr(X) where X = ⟨ψ|A†A|ψ⟩
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            circ = had_test(gate_set[i], [1,2,3], 0, parameters)
            circ.save_statevector()
            
            t_circ = transpile(circ, backend)
            qobj = assemble(t_circ)
            job = backend.run(qobj)
            
            result = job.result()
            o = np.real(result.get_statevector(circ, decimals=100))
            
            m_sum = sum(o[l]**2 for l in range(1, len(o), 2))
            overall_sum_1 += coefficient_set[i] * coefficient_set[j] * (1 - 2*m_sum)

    # Calculate Tr(Y) where Y = ⟨ψ|A†|b⟩⟨b|A|ψ⟩
    overall_sum_2 = 0
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            mult = 1
            for extra in range(2):
                q_reg = QuantumRegister(5)
                if extra == 0:
                    circ = special_had_test(gate_set[i], [1,2,3], 0, parameters, q_reg)
                else:
                    circ = special_had_test(gate_set[j], [1,2,3], 0, parameters, q_reg)
                
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                
                result = job.result()
                o = np.real(result.get_statevector(circ, decimals=100))
                
                m_sum = sum(o[l]**2 for l in range(1, len(o), 2))
                mult *= (1 - 2*m_sum)
            
            overall_sum_2 += coefficient_set[i] * coefficient_set[j] * mult

    cost = 1 - float(overall_sum_2 / overall_sum_1)
    print(f"Cost: {cost}")
    return cost

# 5. Stokes Flow Parameters
coefficient_set = [2.0, -1.0, -1.0]
gate_set = [
    [0, 0, 0],  # Identity term (2I)
    [1, 1, 0],   # X⊗X⊗I term (-1*off-diagonal)
    [0, 1, 1]    # I⊗X⊗X term (-1*off-diagonal)
]

# 6. Run Optimization
np.random.seed(0)
initial_params = [random.uniform(0, 2*np.pi) for _ in range(9)]

result = minimize(calculate_cost_function,
                 x0=initial_params,
                 method="COBYLA",
                 options={'maxiter': 100})

# 7. Post-Processing
optimal_params = [result.x[0:3], result.x[3:6], result.x[6:9]]
circ = apply_fixed_ansatz([0, 1, 2], optimal_params)
circ.save_statevector()

backend = Aer.get_backend('aer_simulator')
t_circ = transpile(circ, backend)
qobj = assemble(t_circ)
job = backend.run(qobj)

statevector = result.get_statevector(circ, decimals=10)

# 8. Verify Solution
A = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]])
b = np.array([1, 1, 1])

# Convert statevector to classical solution
solution = np.array([statevector[0], statevector[1], statevector[2]])
normalized_solution = solution / np.linalg.norm(solution)

# Calculate residual
residual = np.linalg.norm(A @ normalized_solution - b)
print(f"\nFinal Residual: {residual}")
print(f"Optimization Result:\n{result}")
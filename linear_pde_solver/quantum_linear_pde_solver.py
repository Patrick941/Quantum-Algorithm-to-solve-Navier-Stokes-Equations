import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from scipy.optimize import minimize
import random

def apply_fixed_ansatz(qubits, parameters, circ):
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

def had_test(gate_type, qubits, auxiliary_index, parameters, circ):
    circ.h(auxiliary_index)
    apply_fixed_ansatz(qubits, parameters, circ)
    
    for ie in range(len(gate_type[0])):
        if gate_type[0][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    for ie in range(len(gate_type[1])):
        if gate_type[1][ie] == 1:
            circ.cz(auxiliary_index, qubits[ie])
    
    circ.h(auxiliary_index)

def control_fixed_ansatz(qubits, parameters, auxiliary, circ):
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

def control_b(auxiliary, qubits, circ):
    for ia in qubits:
        circ.ch(auxiliary, ia)

def special_had_test(gate_type, qubits, auxiliary_index, parameters, circ):
    circ.h(auxiliary_index)
    control_fixed_ansatz(qubits, parameters, auxiliary_index, circ)
    
    for ty in range(len(gate_type)):
        if gate_type[ty] == 1:
            circ.cz(auxiliary_index, qubits[ty])
    
    control_b(auxiliary_index, qubits, circ)
    circ.h(auxiliary_index)

def calculate_cost_function(parameters, coefficient_set, gate_set):
    parameters = [parameters[0:3], parameters[3:6], parameters[6:9]]
    overall_sum_1 = 0
    
    # First term calculations
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            qr = QuantumRegister(5)
            cr = ClassicalRegister(1)
            circ = QuantumCircuit(qr, cr)
            had_test([gate_set[i], gate_set[j]], [1,2,3], 0, parameters, circ)
            
            backend = Aer.get_backend('aer_simulator')
            circ.save_statevector()
            t_circ = transpile(circ, backend)
            qobj = assemble(t_circ)
            job = backend.run(qobj)
            
            result = job.result()
            outputstate = np.real(result.get_statevector(circ, decimals=100))
            m_sum = sum(outputstate[l]**2 for l in range(len(outputstate)) if l % 2 == 1)
            overall_sum_1 += coefficient_set[i] * coefficient_set[j] * (1 - 2 * m_sum)
    
    overall_sum_2 = 0
    # Second term calculations
    for i in range(len(gate_set)):
        for j in range(len(gate_set)):
            mult = 1
            for extra in [0, 1]:
                qr = QuantumRegister(5)
                cr = ClassicalRegister(1)
                circ = QuantumCircuit(qr, cr)
                
                if extra == 0:
                    special_had_test(gate_set[i], [1,2,3], 0, parameters, circ)
                else:
                    special_had_test(gate_set[j], [1,2,3], 0, parameters, circ)
                
                backend = Aer.get_backend('aer_simulator')
                circ.save_statevector()
                t_circ = transpile(circ, backend)
                qobj = assemble(t_circ)
                job = backend.run(qobj)
                
                result = job.result()
                outputstate = np.real(result.get_statevector(circ, decimals=100))
                m_sum = sum(outputstate[l]**2 for l in range(len(outputstate)) if l % 2 == 1)
                mult *= (1 - 2 * m_sum)
            
            overall_sum_2 += coefficient_set[i] * coefficient_set[j] * mult
    
    cost = 1 - float(overall_sum_2 / overall_sum_1)
    print(f"Cost: {cost}")
    return cost

if __name__ == "__main__":
    # Example configuration
    coefficient_set = [0.55, 0.45]
    gate_set = [[0, 0, 0], [0, 0, 1]]
    
    # Initial random parameters
    initial_params = [random.uniform(0, 2*np.pi) for _ in range(9)]
    
    # Optimize parameters
    result = minimize(calculate_cost_function, initial_params,
                     args=(coefficient_set, gate_set),
                     method="COBYLA", options={'maxiter': 200})
    
    print("\nOptimization results:")
    print(result)
    
    # Get optimized parameters
    opt_params = result.x
    opt_params_reshaped = [opt_params[0:3], opt_params[3:6], opt_params[6:9]]
    
    # Create solution state
    qr = QuantumRegister(3)
    circ = QuantumCircuit(qr)
    apply_fixed_ansatz([0, 1, 2], opt_params_reshaped, circ)
    
    # Simulate final state
    backend = Aer.get_backend('statevector_simulator')
    t_circ = transpile(circ, backend)
    qobj = assemble(t_circ)
    job = backend.run(qobj)
    solution_state = job.result().get_statevector()
    
    print("\nSolution state vector:")
    print(solution_state)
    
    # Replace the verification section with this corrected version:

    # Verification (example specific)
    A = 0.55 * np.eye(8) + 0.45 * np.diag([1,1,1,1,-1,-1,-1,-1])
    b = np.ones(8) / np.sqrt(8)

    # Convert Statevector to NumPy array
    solution_state_np = np.array(solution_state)

    # Calculate the solution
    solution = A @ solution_state_np
    solution /= np.linalg.norm(solution)

    # Calculate fidelity
    fidelity = np.abs(b.dot(solution.conj()))**2
    print(f"\nFidelity with target state: {fidelity:.4f}")
    
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)

    # Define matrix A
    A = 0.55 * np.eye(8) + 0.45 * np.diag([1, 1, 1, 1, -1, -1, -1, -1])

    # Define vector b
    b = np.ones(8) / np.sqrt(8)

    # Solve the linear system classically
    x_classical = np.linalg.solve(A, b)

    # Normalize the solution (to match quantum solution format)
    x_classical /= np.linalg.norm(x_classical)

    print("Classical solution:")
    print(x_classical)

    # Verify fidelity with target state
    fidelity_classical = np.abs(b.dot(x_classical.conj()))**2
    print(f"\nFidelity with target state (classical): {fidelity_classical:.4f}")
    
    print("Matrix A:")
    print(A)
    print("\nVector b:")
    print(b)
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
import os

class HHLAlgorithm:
    def __init__(self, matrix_params, algorithm_params, state_prep_params):
        """
        Initialize HHL algorithm with parameters
        
        :param matrix_params: Dictionary with keys 'a', 'b' (matrix elements)
        :param algorithm_params: Dictionary with keys 't', 'nl', 'nb'
        :param state_prep_params: Dictionary with key 'theta'
        """
        # [Rest of the class remains unchanged]
        # Matrix parameters
        self.a = matrix_params.get('a', 1)
        self.b = matrix_params.get('b', -1/3)
        
        # Algorithm parameters
        self.t = algorithm_params.get('t', 2)
        self.nl = algorithm_params.get('nl', 2)  # Eigenvalue qubits
        self.nb = algorithm_params.get('nb', 1)  # Solution qubits
        self.na = 1  # Ancilla qubits
        
        # State preparation parameters
        self.theta = state_prep_params.get('theta', 0)
        
        # Derived parameters
        self.num_qubits = self.nb + self.nl + self.na
        self._calculate_eigenvalues()
        self._calculate_rotation_angles()
        
        # Initialize quantum circuit
        self.qr = QuantumRegister(self.num_qubits)
        self.qc = QuantumCircuit(self.qr)
        self._build_circuit()

    def _calculate_eigenvalues(self):
        """Calculate eigenvalues of the matrix A = [[a, b], [b, a]]"""
        self.lambda1 = self.a + self.b
        self.lambda2 = self.a - self.b

    def _calculate_rotation_angles(self):
        """Calculate angles for eigenvalue rotation based on current parameters"""
        # Calculate arcsin terms using matrix parameters
        self.arcsin_term = 2 * np.arcsin(abs(self.b))
        
        # Calculate rotation angles
        self.rot_angles = [
            (-np.pi + np.pi/3 - self.arcsin_term)/4,
            (-np.pi - np.pi/3 + self.arcsin_term)/4,
            (np.pi - np.pi/3 - self.arcsin_term)/4,
            (np.pi + np.pi/3 + self.arcsin_term)/4
        ]

    def _build_circuit(self):
        """Construct the quantum circuit"""
        qrb = self.qr[0:self.nb]
        qrl = self.qr[self.nb:self.nb+self.nl]
        qra = self.qr[self.nb+self.nl:self.nb+self.nl+self.na]

        # State preparation
        self.qc.ry(2*self.theta, qrb[0])

        # Quantum Phase Estimation
        self._apply_qpe(qrl, qrb)
        
        # Inverse QFT
        self._inverse_qft(qrl)
        
        # Eigenvalue rotation
        self._apply_eigenvalue_rotation(qrl, qra)
        
        self.qc.measure_all()

    def _apply_qpe(self, qrl, qrb):
        """Apply Quantum Phase Estimation"""
        for qu in qrl:
            self.qc.h(qu)
            
        self.qc.p(self.a*self.t, qrl[0])
        self.qc.p(self.a*self.t*2, qrl[1])
        self.qc.u(self.b*self.t, -np.pi/2, np.pi/2, qrb[0])
        
        # Controlled rotations
        for i, factor in enumerate([1, 2]):
            self._controlled_rotation(qrl[i], qrb[0], self.b*self.t*factor)

    def _controlled_rotation(self, control, target, params):
        """Generic controlled rotation block"""
        self.qc.p(np.pi/2, target)
        self.qc.cx(control, target)
        self.qc.ry(params, target)
        self.qc.cx(control, target)
        self.qc.ry(-params, target)
        self.qc.p(3*np.pi/2, target)

    def _inverse_qft(self, qrl):
        """Apply Inverse Quantum Fourier Transform"""
        self.qc.h(qrl[1])
        self.qc.rz(-np.pi/4, qrl[1])
        self.qc.cx(qrl[0], qrl[1])
        self.qc.rz(np.pi/4, qrl[1])
        self.qc.cx(qrl[0], qrl[1])
        self.qc.rz(-np.pi/4, qrl[0])
        self.qc.h(qrl[0])

    def _apply_eigenvalue_rotation(self, qrl, qra):
        """Apply conditional eigenvalue rotation"""
        controls = [qrl[1], qrl[0], qrl[1], qrl[0]]
        for control_qubit, angle in zip(controls, self.rot_angles):
            self.qc.cx(control_qubit, qra[0])
            self.qc.ry(angle, qra[0])

    def run(self, shots=1024):
        """Execute the circuit"""
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, simulator, shots=shots).result()
        return result.get_counts()

def main():
    # Configuration parameters
    matrix_params = {'a': 1, 'b': -1/3}
    algorithm_params = {'t': 2, 'nl': 2, 'nb': 1}
    state_prep_params = {'theta': 0}  # Prepares |b> = |0>

    # Initialize and run HHL
    hhl = HHLAlgorithm(matrix_params, algorithm_params, state_prep_params)
    
    # Print circuit info
    print(f"Circuit metrics:")
    print(f"Depth: {hhl.qc.depth()}")
    print(f"CNOTs: {hhl.qc.count_ops().get('cx', 0)}")
    print(f"Total qubits: {hhl.num_qubits}")
    
    # Run simulation
    counts = hhl.run(shots=8192)  # Increased shots for better statistics
    
    # Process HHL results with post-selection
    total_shots = sum(counts.values())
    hhl_counts_ancilla1 = {k: v for k, v in counts.items() if k.startswith('1')}  # Ancilla is leftmost qubit
    total_valid = sum(hhl_counts_ancilla1.values())
    
    # Calculate HHL probabilities
    hhl_probs = {'0': 0, '1': 0}
    for state, cnt in hhl_counts_ancilla1.items():
        solution_bit = state[-1]  # Solution qubit is rightmost bit
        hhl_probs[solution_bit] += cnt
    
    if total_valid > 0:
        hhl_prob0 = hhl_probs['0'] / total_valid
        hhl_prob1 = hhl_probs['1'] / total_valid
    else:
        hhl_prob0 = hhl_prob1 = 0.0

    # Compute classical solution
    A = np.array([[matrix_params['a'], matrix_params['b']],
                  [matrix_params['b'], matrix_params['a']]])
    b_vec = np.array([1, 0])  # Initial state |0> corresponds to [1, 0]
    x_classical = np.linalg.inv(A) @ b_vec
    x_normalized = x_classical / np.linalg.norm(x_classical)
    classical_prob0 = x_normalized[0]**2
    classical_prob1 = x_normalized[1]**2

    # Print comparison
    print("\nComparison of Results:")
    print(f"Classical Probabilities: |0> = {classical_prob0:.4f}, |1> = {classical_prob1:.4f}")
    print(f"HHL Probabilities (Post-Selected): |0> = {hhl_prob0:.4f}, |1> = {hhl_prob1:.4f}")
    print(f"Absolute Error (|0>): {abs(hhl_prob0 - classical_prob0):.4f}")
    print(f"Absolute Error (|1>): {abs(hhl_prob1 - classical_prob1):.4f}")
    print(f"Post-Selection Success Rate: {total_valid/total_shots:.2%}")

    # Plotting
    output_dir = os.path.dirname(os.path.abspath(__file__)) + '/Images'
    os.makedirs(output_dir, exist_ok=True)

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.bar(['Classical |0>', 'HHL |0>'], [classical_prob0, hhl_prob0], color=['blue', 'orange'])
    plt.bar(['Classical |1>', 'HHL |1>'], [classical_prob1, hhl_prob1], color=['blue', 'orange'])
    plt.ylabel('Probability')
    plt.title('Solution Comparison: Classical vs HHL')
    plt.savefig(f"{output_dir}/hhl_vs_classical.png")
    plt.close()

if __name__ == "__main__":
    main()
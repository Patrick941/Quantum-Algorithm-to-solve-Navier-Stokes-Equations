from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt
import os

class HHLAlgorithm:
    def __init__(self, matrix_params, algorithm_params, state_prep_params):
        # Matrix parameters
        self.a = matrix_params.get('a', 1)
        self.b = matrix_params.get('b', -1/3)
        
        # Algorithm parameters (UPDATED)
        self.t = algorithm_params.get('t', 3 * np.pi)  # Adjusted for eigenvalue scaling
        self.nl = algorithm_params.get('nl', 4)  # Increased eigenvalue precision
        self.nb = algorithm_params.get('nb', 1)
        self.na = 1
        
        # State preparation
        self.theta = state_prep_params.get('theta', 0)
        
        # Pre-compute critical values (FIXED)
        self._calculate_eigenvalues()
        self._calculate_rotation_angles()  # Now uses actual eigenvalues
        self.num_qubits = self.nb + self.nl + self.na
        
        # Initialize circuit
        self.qr = QuantumRegister(self.num_qubits)
        self.qc = QuantumCircuit(self.qr)
        self._build_circuit()

    def _calculate_eigenvalues(self):
        """Calculate exact eigenvalues for the matrix"""
        self.lambda1 = self.a + self.b  # 2/3
        self.lambda2 = self.a - self.b  # 4/3

    def _calculate_rotation_angles(self):
        """Calculate rotation angles using ACTUAL eigenvalues (FIXED)"""
        C = min(self.lambda1, self.lambda2)  # 2/3
        self.rot_angles = [
            2 * np.arcsin(C / self.lambda1),  # π (for λ1=2/3)
            2 * np.arcsin(C / self.lambda2)   # π/3 (for λ2=4/3)
        ]

    def _build_circuit(self):
        qrb = self.qr[0:self.nb]
        qrl = self.qr[self.nb:self.nb+self.nl]
        qra = self.qr[self.nb+self.nl:self.nb+self.nl+self.na]

        # State preparation
        self.qc.ry(2*self.theta, qrb[0])

        # QPE with improved precision
        self._apply_qpe(qrl, qrb)
        self._inverse_qft(qrl)
        
        # Eigenvalue rotation (FIXED IMPLEMENTATION)
        # Apply conditional rotations based on eigenvalue estimates
        for i in range(2):  # For our 2x2 matrix case
            angle = self.rot_angles[i]
            # Convert eigenvalue estimate to binary controls
            bin_rep = format(i, f'0{self.nl}b')[::-1]  # Little-endian
            controls = [qrl[j] for j, bit in enumerate(bin_rep) if bit == '1']
            if controls:
                self.qc.mcry(angle, controls, qra[0], qrb)
        
        self.qc.measure_all()

    def _apply_qpe(self, qrl, qrb):
        """Improved QPE implementation"""
        for qu in qrl:
            self.qc.h(qu)
        
        # Hamiltonian evolution with adjusted time parameter
        for i, qu in enumerate(qrl):
            self.qc.cp(2 * np.pi * self.lambda1 * self.t / (2**self.nl), qu, qrb[0])
            self.qc.cp(2 * np.pi * self.lambda2 * self.t / (2**self.nl), qu, qrb[0])

    def _inverse_qft(self, qrl):
        """Improved inverse QFT"""
        n = len(qrl)
        for j in reversed(range(n)):
            self.qc.h(qrl[j])
            for k in reversed(range(j)):
                self.qc.cp(-np.pi/(2**(j-k)), qrl[j], qrl[k])

    def run(self, shots=8192):
        simulator = Aer.get_backend('qasm_simulator')
        return execute(self.qc, simulator, shots=shots).result().get_counts()

def main():
    matrix_params = {'a': 1, 'b': -1/3}
    algorithm_params = {'t': 3 * np.pi, 'nl': 4, 'nb': 1}  # Critical params
    state_prep_params = {'theta': 0}

    hhl = HHLAlgorithm(matrix_params, algorithm_params, state_prep_params)
    
    print(f"Circuit Depth: {hhl.qc.depth()}")
    print(f"CNOT Count: {hhl.qc.count_ops().get('cx', 0)}")
    
    counts = hhl.run(shots=8192)
    
    # FIXED POST-SELECTION (ancilla is RIGHTMOST qubit)
    hhl_counts_ancilla1 = {k: v for k, v in counts.items() if k.endswith('1')}
    total_valid = sum(hhl_counts_ancilla1.values())
    
    # Calculate probabilities
    hhl_probs = {'0': 0, '1': 0}
    for state, cnt in hhl_counts_ancilla1.items():
        solution_bit = state[0]  # Leftmost qubit is solution
        hhl_probs[solution_bit] += cnt
    
    # Classical solution
    A = np.array([[1, -1/3], [-1/3, 1]])
    b = np.array([1, 0])
    x_classical = np.linalg.inv(A) @ b
    x_normalized = x_classical / np.linalg.norm(x_classical)
    classical_probs = x_normalized**2
    
    print(f"\nClassical: |0>={classical_probs[0]:.4f}, |1>={classical_probs[1]:.4f}")
    if total_valid > 0:
        hhl_prob0 = hhl_probs['0'] / total_valid
        hhl_prob1 = hhl_probs['1'] / total_valid
        print(f"HHL:       |0>={hhl_prob0:.4f}, |1>={hhl_prob1:.4f}")
        print(f"Errors:    Δ0={abs(hhl_prob0 - classical_probs[0]):.4f}, Δ1={abs(hhl_prob1 - classical_probs[1]):.4f}")
    else:
        print("No valid HHL results!")
    print(f"Post-Selection Rate: {total_valid/8192:.2%}")

if __name__ == "__main__":
    main()
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, Aer, execute
import numpy as np
import matplotlib.pyplot as plt  # Added for plotting
import os
from qiskit.utils import QuantumInstance
from linear_solvers import HHL

class CustomHHLAlgorithm:
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
    
class LibraryHHLAlgorithm:
    def __init__(self, quantum_instance=None):
        """
        Initializes the HHL solver with a quantum instance (default: statevector simulator).
        """
        if quantum_instance is None:
            quantum_instance = QuantumInstance(Aer.get_backend('statevector_simulator'))
        self.quantum_instance = quantum_instance
        self.hhl = HHL()
        self.result = None
        self.norm_b = None

    def solve(self, A, b):
        """
        Solves the linear system Ax = b using the HHL algorithm.
        
        Args:
            A (np.ndarray): Hermitian matrix of the system.
            b (np.ndarray): Right-hand side vector of the system.
        
        Returns:
            LinearSolverResult: Result object containing the quantum solution state.
        """
        if not np.allclose(A, A.conj().T):
            raise ValueError("Matrix A must be Hermitian.")
        norm_b = np.linalg.norm(b)
        if norm_b == 0:
            raise ValueError("Vector b must not be all zeros.")
        b_normalized = b / norm_b
        problem = LinearSystemProblem(A, b_normalized)
        self.result = self.hhl.solve(problem, self.quantum_instance)
        self.norm_b = norm_b
        return self.result

    def get_solution_vector(self):
        """
        Extracts and processes the solution vector from the quantum state.
        
        Returns:
            np.ndarray: Classical solution vector derived from the quantum state.
        """
        if self.result is None:
            raise RuntimeError("Run solve() first to obtain a solution.")
        circuit = self.result.state
        # Locate ancilla qubit
        ancilla_reg = next((reg for reg in circuit.qregs if reg.name == 'ancilla'), None)
        if not ancilla_reg:
            raise RuntimeError("Ancilla register not found.")
        ancilla_qubit = circuit.qubits.index(ancilla_reg[0])
        # Obtain statevector
        statevector = Statevector(circuit)
        # Extract solution amplitudes where ancilla is |1>
        solution_amps = []
        for idx, amp in enumerate(statevector):
            if (idx >> ancilla_qubit) & 1:
                solution_amps.append(amp)
        if not solution_amps:
            raise RuntimeError("Post-selection failed; ancilla not in |1> state.")
        solution_amps = np.array(solution_amps)
        solution_amps /= np.linalg.norm(solution_amps)  # Normalize
        solution = solution_amps * self.norm_b  # Scale by original norm
        return solution

def main():
    A = np.array([[1, 0], [0, 1]], dtype=float)
    b = np.array([1, 0], dtype=float)
    
    # Initialize and solve
    hhl_solver = LibraryHHLAlgorithm()
    result = hhl_solver.solve(A, b)
    solution = hhl_solver.get_solution_vector()
    print("Solution vector:", solution)

if __name__ == "__main__":
    main()
from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
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
    def __init__(self, matrix_params, algorithm_params, state_prep_params):
        # Matrix parameters
        self.a = matrix_params.get('a', 1)
        self.b = matrix_params.get('b', -1/3)
        
        # Algorithm parameters
        self.t = algorithm_params.get('t', 2)
        self.nl = 2  # Eigenvalue register size
        self.nb = 1  # Solution register size
        self.na = 1  # Ancilla register size
        
        # State preparation
        self.theta = state_prep_params.get('theta', 0)
        
        # Registers
        self.num_qubits = self.nb + self.nl + self.na
        self.qr = QuantumRegister(self.num_qubits)
        self.cr = ClassicalRegister(self.num_qubits)  # Now defined
        self.qc = QuantumCircuit(self.qr, self.cr)  # Attach both registers
        
        # Build the circuit
        self._build_circuit()

    def _build_circuit(self):
        """Example circuit with measurement"""
        self.qc.h(self.qr[0])  # Example gate
        self.qc.measure(self.qr, self.cr)  # Map all qubits to classical bits

    def run(self, shots=1024):
        """Execute the circuit"""
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, simulator, shots=shots).result()
        return result.get_counts()


def main():
    # Using my class
    print("Running custom HHL")
    matrix_params = {
        'a': 1,
        'b': -1/3
    }
    
    algorithm_params = {
        't': 2,
        'nl': 2,
        'nb': 1
    }
    
    state_prep_params = {
        'theta': 0
    }
    
    # Initialize and run Custom HHL
    hhl_custom = CustomHHLAlgorithm(matrix_params, algorithm_params, state_prep_params)
    
    # Print circuit info
    print(f"Circuit metrics (Custom HHL):")
    print(f"Depth: {hhl_custom.qc.depth()}")
    print(f"CNOTs: {hhl_custom.qc.count_ops().get('cx', 0)}")
    print(f"Total qubits: {hhl_custom.num_qubits}")
    
    # Run simulation
    counts_custom = hhl_custom.run()
    
    # Print results
    print("\nMeasurement results (Custom HHL):")
    print(counts_custom)

    # Plotting the results
    sorted_counts_custom = dict(sorted(counts_custom.items(), key=lambda item: int(item[0], 2)))
    states_custom = list(sorted_counts_custom.keys())
    counts_values_custom = list(sorted_counts_custom.values())
    output_dir = '/'.join(__file__.split('/')[:-1]) + '/Images'
    os.makedirs(output_dir, exist_ok=True)

    # Save the bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(states_custom, counts_values_custom)
    plt.xticks(rotation=45)
    plt.xlabel('Quantum State')
    plt.ylabel('Counts')
    plt.title('Custom HHL Algorithm Measurement Results')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/custom_hhl_measurement_results.png")

    # Save the quantum circuit diagram
    plt.figure(figsize=(12, 8))
    hhl_custom.qc.draw(output='mpl', style='clifford')
    plt.title('Custom HHL Quantum Circuit')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/custom_hhl_quantum_circuit.png")

    # Using library HHL
    print("\nRunning library HHL")
    hhl_library = LibraryHHLAlgorithm(matrix_params, algorithm_params, state_prep_params)
    
    # Print circuit info
    print(f"Circuit metrics (Library HHL):")
    print(f"Depth: {hhl_library.qc.depth()}")
    print(f"CNOTs: {hhl_library.qc.count_ops().get('cx', 0)}")
    print(f"Total qubits: {hhl_library.num_qubits}")
    
    # Run simulation
    counts_library = hhl_library.run()
    
    # Print results
    print("\nMeasurement results (Library HHL):")
    print(counts_library)

    # Plotting the results
    sorted_counts_library = dict(sorted(counts_library.items(), key=lambda item: int(item[0], 2)))
    states_library = list(sorted_counts_library.keys())
    counts_values_library = list(sorted_counts_library.values())

    # Save the bar plot
    plt.figure(figsize=(10, 5))
    plt.bar(states_library, counts_values_library)
    plt.xticks(rotation=45)
    plt.xlabel('Quantum State')
    plt.ylabel('Counts')
    plt.title('Library HHL Algorithm Measurement Results')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/library_hhl_measurement_results.png")

    # Save the quantum circuit diagram
    plt.figure(figsize=(12, 8))
    hhl_library.qc.draw(output='mpl', style='clifford')
    plt.title('Library HHL Quantum Circuit')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/library_hhl_quantum_circuit.png")

if __name__ == "__main__":
    main()
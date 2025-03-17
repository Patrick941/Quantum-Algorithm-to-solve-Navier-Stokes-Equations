from qiskit import QuantumRegister, QuantumCircuit, Aer, execute
import numpy as np

class HHLAlgorithm:
    def __init__(self, t=2, a=1, b=-1/3, theta=0):
        self.t = t
        self.a = a
        self.b = b
        self.theta = theta
        self.num_qubits = 4
        self.nb = 1  # Solution qubits
        self.nl = 2  # Eigenvalue qubits
        
        self.qr = QuantumRegister(self.num_qubits)
        self.qc = QuantumCircuit(self.qr)
        self._build_circuit()
        
    def _build_circuit(self):
        # Define register slices
        qrb = self.qr[0:self.nb]
        qrl = self.qr[self.nb:self.nb+self.nl]
        qra = self.qr[self.nb+self.nl:self.nb+self.nl+1]

        # State preparation
        self.qc.ry(2*self.theta, qrb[0])

        # Quantum Phase Estimation
        for qu in qrl:
            self.qc.h(qu)
            
        self.qc.p(self.a*self.t, qrl[0])
        self.qc.p(self.a*self.t*2, qrl[1])
        self.qc.u(self.b*self.t, -np.pi/2, np.pi/2, qrb[0])

        # Controlled rotations
        self._add_controlled_rotations(qrl, qrb)
        
        # Inverse QFT
        self._inverse_qft(qrl)

        # Eigenvalue rotation
        self._eigenvalue_rotation(qrl, qra)
        
        self.qc.measure_all()

    def _add_controlled_rotations(self, qrl, qrb):
        # Helper method for controlled rotation blocks
        for i, factor in enumerate([1, 2]):
            params = self.b * self.t * factor
            self.qc.p(np.pi/2, qrb[0])
            self.qc.cx(qrl[i], qrb[0])
            self.qc.ry(params, qrb[0])
            self.qc.cx(qrl[i], qrb[0])
            self.qc.ry(-params, qrb[0])
            self.qc.p(3*np.pi/2, qrb[0])

    def _inverse_qft(self, qrl):
        # Inverse Quantum Fourier Transform
        self.qc.h(qrl[1])
        self.qc.rz(-np.pi/4, qrl[1])
        self.qc.cx(qrl[0], qrl[1])
        self.qc.rz(np.pi/4, qrl[1])
        self.qc.cx(qrl[0], qrl[1])
        self.qc.rz(-np.pi/4, qrl[0])
        self.qc.h(qrl[0])

    def _eigenvalue_rotation(self, qrl, qra):
        # Eigenvalue rotation parameters
        t1 = (-np.pi + np.pi/3 - 2*np.arcsin(1/3))/4
        t2 = (-np.pi - np.pi/3 + 2*np.arcsin(1/3))/4
        t3 = (np.pi - np.pi/3 - 2*np.arcsin(1/3))/4
        t4 = (np.pi + np.pi/3 + 2*np.arcsin(1/3))/4

        # Conditional rotations
        rotations = [(qrl[1], t1), (qrl[0], t2), 
                    (qrl[1], t3), (qrl[0], t4)]
        
        for control_qubit, angle in rotations:
            self.qc.cx(control_qubit, qra[0])
            self.qc.ry(angle, qra[0])

    def run(self, shots=1024):
        simulator = Aer.get_backend('qasm_simulator')
        result = execute(self.qc, simulator, shots=shots).result()
        return result.get_counts()

def main():
    # Initialize and run HHL algorithm
    hhl = HHLAlgorithm(t=2)
    
    # Print circuit info
    print(f"Circuit depth: {hhl.qc.depth()}")
    print(f"CNOT count: {hhl.qc.count_ops().get('cx', 0)}")
    
    # Run simulation and show results
    counts = hhl.run()
    print("\nMeasurement results:")
    print(counts)

if __name__ == "__main__":
    main()
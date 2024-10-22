import qiskit
from qiskit import *

def run_quantum_circuit(backend):
    # Create a quantum circuit with 3 qubits and 3 classical bits
    qc = QuantumCircuit(3, 3)
    
    # Encode the input qubit into a 3-qubit state
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Introduce an error on the second qubit (for demonstration purposes)
    qc.x(1)
    
    # Syndrome measurement
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.ccx(1, 2, 0)
    
    print(qc.draw())
    qc.measure([0, 1, 2], [0, 1, 2])
    
    # Execute the circuit on the given backend
    qc_compiled = transpile(qc, backend=backend)
    job = backend.run(qc_compiled) 
    result = job.result()
    
    # Get the counts of the results
    counts = result.get_counts(qc)
    print(counts)
    
    return counts